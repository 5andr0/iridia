//! `clip_distances_as_varyings` and `discard_clipped`: the interfaces they leave, the code they keep, a run of the pixel
//! program in the IR interpreter on distances either side of zero, and the pair written as WGSL for a pipeline that has no
//! clipper for them.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

mod support;

use iridia_ir::{
    Program,
    interface::{InterpMode, InterpSampling, Label, ScalarType, Stage, SystemValue},
};
use iridia_passes::{ClipVaryings, clip_distances_as_varyings, discard_clipped};
use support::{from_text, run};

const F1: (ScalarType, u8) = (ScalarType::Float, 1);
const F2: (ScalarType, u8) = (ScalarType::Float, 2);
const F4: (ScalarType, u8) = (ScalarType::Float, 4);

fn bits(lanes: &[f32]) -> [u32; 4] {
    let mut out = [0u32; 4];
    for (o, l) in out.iter_mut().zip(lanes) {
        *o = l.to_bits();
    }
    out
}

/// A vertex program: the position, a clip distance varying of two lanes and one of one lane, and a `TEXCOORD0`.
fn vertex(clipped: bool) -> Program {
    let text = "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec2<f32> = %0.xy
  %2: f32 = %0.z
  output 0.xyzw = %0
  output 1.xy = %1
  output 2.x = %2
  output 3.xy = %1
  return
}
";
    let mut p = from_text(text, &[F4], &[F4, F2, F1, F2], Vec::new());
    p.stage = Stage::Vertex;
    p.interface.stage = Stage::Vertex;
    let outputs = &mut p.interface.outputs;
    outputs[0].label = Label {
        name: "SV_Position".into(),
        index: 0,
    };
    outputs[0].system = Some(SystemValue::ClipPosition);
    outputs[3].system = None;
    outputs[3].label = Label {
        name: "TEXCOORD".into(),
        index: 0,
    };
    for (v, index) in outputs[1..3].iter_mut().zip(0..) {
        v.label = Label {
            name: "SV_ClipDistance".into(),
            index,
        };
        v.system = clipped.then_some(SystemValue::ClipDistance);
    }
    p
}

/// A pixel program with an early return: below a half it returns the input; otherwise the half.
fn pixel() -> Program {
    let text = "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f32 = f32(0.5)
  %2: bool = fcmp.lt %0, %1
  if %2 {
    output 0.x = %0
    return
  } else {
    yield
  }
  output 0.x = %1
  return
}
";
    from_text(text, &[F2], &[F4], Vec::new())
}

#[test]
fn a_vertex_program_s_clip_distances_become_plain_varyings_of_the_same_labels() {
    let mut p = vertex(true);
    let before = p.entry.clone();
    let clips = clip_distances_as_varyings(&mut p).unwrap().unwrap();
    assert_eq!(
        clips,
        ClipVaryings {
            varyings: vec![
                (
                    Label {
                        name: "SV_ClipDistance".into(),
                        index: 0
                    },
                    2
                ),
                (
                    Label {
                        name: "SV_ClipDistance".into(),
                        index: 1
                    },
                    1
                ),
            ]
        }
    );
    assert_eq!(clips.count(), 3);
    assert!(p.interface.outputs.iter().skip(1).all(|v| v.system.is_none()));
    assert_eq!(p.interface.outputs[3].label.name, "TEXCOORD");
    assert_eq!(p.interface.outputs[0].system, Some(SystemValue::ClipPosition));
    // the code is untouched, and the program is still a program
    assert_eq!(p.entry, before);
    iridia_ir::validate(&p).unwrap();
    // the clipper's own requirement is gone
    assert!(
        iridia_ir::target::requirements(&p)
            .iter()
            .all(|r| r.capability != iridia_ir::target::Capability::ClipDistances)
    );
    // and what the program wrote is what it wrote
    let input = bits(&[-1.0, 2.0, 3.0, 4.0]);
    assert_eq!(run(&vertex(true), &[input], &[]), run(&p, &[input], &[]));
}

#[test]
fn a_vertex_program_with_no_clip_distance_is_left_alone_and_another_stage_is_refused() {
    let mut p = vertex(false);
    let before = p.clone();
    assert_eq!(clip_distances_as_varyings(&mut p), Ok(None));
    assert_eq!(p, before);
    let e = clip_distances_as_varyings(&mut pixel()).unwrap_err();
    assert!(
        e.contains("a Pixel program has no clip distances to carry"),
        "{e}"
    );
    // two outputs of one label could not be told apart by the pixel program
    let mut q = vertex(true);
    q.interface.outputs[2].label = q.interface.outputs[1].label.clone();
    let e = clip_distances_as_varyings(&mut q).unwrap_err();
    assert!(
        e.contains("two clip distance outputs are labelled SV_ClipDistance0"),
        "{e}"
    );
}

#[test]
fn the_pixel_program_takes_each_varying_and_discards_where_a_lane_is_below_zero_at_each_return() {
    let mut v = vertex(true);
    let clips = clip_distances_as_varyings(&mut v).unwrap().unwrap();
    let plain = pixel();
    let mut p = plain.clone();
    discard_clipped(&mut p, &clips).unwrap();
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&p.entry)));
    // its own input, then one per clip varying: perspective, at the pixel centre
    assert_eq!(p.interface.inputs.len(), 3);
    assert_eq!(p.interface.inputs[1].label.name, "SV_ClipDistance");
    assert_eq!(p.interface.inputs[1].value.width, 2);
    assert_eq!(p.interface.inputs[2].value.width, 1);
    for i in &p.interface.inputs[1..] {
        let interp = i.interp.unwrap();
        assert_eq!(
            (interp.mode, interp.sampling),
            (InterpMode::Perspective, InterpSampling::Center)
        );
    }
    // a guard before each of the two returns, each made where it is used
    let text = iridia_ir::text::print(&p.entry);
    assert_eq!(text.matches("demote").count(), 2, "{text}");
    // every distance either side of zero: 0 and positive and NaN keep, any lane below zero discards, on both paths
    let own = |x: f32| bits(&[x, 0.0]);
    let cases: [(&[f32], &[f32], bool); 7] = [
        (&[1.0, 2.0], &[3.0], false),
        (&[0.0, 0.0], &[0.0], false),
        (&[-0.5, 2.0], &[3.0], true),
        (&[1.0, -0.5], &[3.0], true),
        (&[1.0, 2.0], &[-1e-30], true),
        (&[f32::NAN, 1.0], &[1.0], false),
        (&[-f32::INFINITY, 1.0], &[1.0], true),
    ];
    for (a, b, discards) in cases {
        for x in [0.25f32, 0.75] {
            let inputs = [own(x), bits(a), bits(b)];
            let (out_plain, killed_plain) = run(&plain, &inputs[..1], &[]).unwrap();
            assert!(!killed_plain);
            let (out, killed) = run(&p, &inputs, &[]).unwrap();
            assert_eq!(killed, discards, "{a:?} {b:?} at {x}");
            // what the program writes is what it wrote
            assert_eq!(out, out_plain, "{a:?} {b:?} at {x}");
        }
    }
    support::agrees_flat(&p, &[own(0.25), bits(&[-1.0, 0.0]), bits(&[0.0])], &[]);
    support::agrees_flat(&p, &[own(0.75), bits(&[1.0, 0.0]), bits(&[0.0])], &[]);
}

#[test]
fn a_pixel_program_that_cannot_take_them_is_refused_by_name() {
    let mut v = vertex(true);
    let clips = clip_distances_as_varyings(&mut v).unwrap().unwrap();
    let e = discard_clipped(&mut vertex(false), &clips).unwrap_err();
    assert!(
        e.contains("a Vertex program discards nothing a clipper keeps"),
        "{e}"
    );
    // an input of the label of a clip varying
    let mut p = pixel();
    p.interface.inputs[0].label = clips.varyings[0].0.clone();
    let e = discard_clipped(&mut p, &clips).unwrap_err();
    assert!(
        e.contains("already has an input labelled SV_ClipDistance0"),
        "{e}"
    );
}

#[test]
fn the_pair_is_written_for_a_pipeline_with_no_clipper_for_them() {
    use iridia_naga::{BindingPlan, Pipeline, Profile};
    let mut v = vertex(true);
    // the browser's WebGPU has no clip distances: the program is refused by name
    let e = iridia_naga::wgsl(&v, &Profile::BROWSER).unwrap_err();
    assert!(e.to_string().contains("clip distance"), "{e}");
    let clips = clip_distances_as_varyings(&mut v).unwrap().unwrap();
    let mut p = pixel();
    discard_clipped(&mut p, &clips).unwrap();
    // the vertex program's varyings meet the pixel program's by label: the clip varyings take its locations 1 and 2
    let plan = |p: &Program| BindingPlan::webgpu(&p.interface).unwrap();
    let vs = iridia_naga::wgsl_for(
        &v,
        &Profile::BROWSER,
        &Pipeline {
            plan: plan(&v),
            feeds: Some(&p.interface),
        },
    )
    .unwrap();
    let fs = iridia_naga::wgsl_for(
        &p,
        &Profile::BROWSER,
        &Pipeline {
            plan: plan(&p),
            feeds: None,
        },
    )
    .unwrap();
    assert!(!vs.text.contains("clip_distances"), "{}", vs.text);
    assert!(
        vs.text.contains("@location(1)") && vs.text.contains("@location(2)"),
        "{}",
        vs.text
    );
    assert!(
        fs.text.contains("@location(1)") && fs.text.contains("@location(2)"),
        "{}",
        fs.text
    );
    assert!(fs.text.contains("discard;"), "{}", fs.text);
    assert_eq!(fs.text.matches("discard;").count(), 2, "{}", fs.text);
    // the same pair as the standard WGSL writer, which links them by label too
    let wplan = |p: &Program| iridia_wgsl::BindingPlan::webgpu(&p.interface).unwrap();
    let vs = iridia_wgsl::wgsl_for(
        &v,
        &iridia_wgsl::Profile::BROWSER,
        &iridia_wgsl::Pipeline {
            plan: wplan(&v),
            feeds: Some(&p.interface),
        },
    )
    .unwrap();
    let fs = iridia_wgsl::wgsl_for(
        &p,
        &iridia_wgsl::Profile::BROWSER,
        &iridia_wgsl::Pipeline {
            plan: wplan(&p),
            feeds: None,
        },
    )
    .unwrap();
    assert!(!vs.text.contains("clip_distances"), "{}", vs.text);
    assert!(
        vs.text.contains("@location(1)") && vs.text.contains("@location(2)"),
        "{}",
        vs.text
    );
    assert!(
        fs.text.contains("@location(1)") && fs.text.contains("@location(2)"),
        "{}",
        fs.text
    );
    assert_eq!(fs.text.matches("discard;").count(), 2, "{}", fs.text);
}
