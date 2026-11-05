//! `promote`, `simplify` and `specialise` on programs in IR text, each checked against the form it should give and
//! by running it before and after on hand-worked inputs.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

mod support;

use iridia_ir::interface::{Resource, ResourceId, ResourceKind, ResourceProvenance, ScalarType, StateParam};
use support::{from_text, run};

const F1: (ScalarType, u8) = (ScalarType::Float, 1);

fn f(x: f32) -> [u32; 4] {
    [x.to_bits(), 0, 0, 0]
}

#[test]
fn promote_carries_variables_out_of_an_if_and_through_a_loop() {
    // r = x; if x < 0 { r = -x }; loop: while r < 10 { r = r * 2 }; output r
    let text = "iridia-ir 1
fn main {
  var r: f32
  %0: f32 = input 0.x
  r = %0
  %1: f32 = f32(0.0)
  %2: bool = fcmp.lt %0, %1
  if %2 {
    %3: f32 = fneg %0
    r = %3
    yield
  } else {
    yield
  }
  %4: f32 = f32(10.0)
  %5: f32 = f32(2.0)
  loop '0 () = () {
    %6: f32 = load r
    %7: bool = fcmp.ge %6, %4
    if %7 {
      br '0
    } else {
      yield
    }
    %8: f32 = fmul.ieee %6, %5
    r = %8
    continue '0
  }
  %9: f32 = load r
  output 0.x = %9
  return
}
";
    let p = from_text(text, &[F1], &[F1], Vec::new());
    let mut q = p.clone();
    iridia_passes::promote(&mut q.entry).unwrap();
    iridia_ir::validate(&q).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&q.entry)));
    assert!(q.entry.vars.is_empty());
    let printed = iridia_ir::text::print(&q.entry);
    assert!(!printed.contains("load "), "{printed}");
    // r through the if and the loop: -3 -> 3 -> 6 -> 12; 0.5 -> 16; 1 -> 16; 20 stays
    for (x, want) in [(-3.0f32, 12.0f32), (0.5, 16.0), (1.0, 16.0), (20.0, 20.0)] {
        for prog in [&p, &q] {
            let (out, _) = run(prog, &[f(x)], &[]).unwrap();
            assert_eq!(out[0].unwrap()[0], want.to_bits(), "x {x}");
        }
    }
}

#[test]
fn simplify_folds_lane_bookkeeping_and_removes_what_is_unused() {
    let text = "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<f32> = %0.wzyx
  %2: vec2<f32> = %1.yx
  %3: f32 = %0.x
  %4: f32 = %0.y
  %5: f32 = %0.z
  %6: f32 = %0.w
  %7: vec4<f32> = compose %3, %4, %5, %6
  %8: vec4<f32> = %7.xyzw
  %9: f32 = fneg %3
  %10: vec2<f32> = compose %5, %4
  output 0.xyzw = %8
  output 1.xy = %2
  output 2.xy = %10
  return
}
";
    let p = from_text(
        text,
        &[(ScalarType::Float, 4)],
        &[(ScalarType::Float, 4); 3],
        Vec::new(),
    );
    let mut q = p.clone();
    iridia_passes::simplify(&mut q.entry);
    iridia_ir::validate(&q).unwrap();
    // the swizzle of a swizzle is one swizzle; the compose of every lane in order is the input; the compose of
    // two lanes of it is a swizzle; the negation nothing uses is gone
    let want = "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %2: vec2<f32> = %0.zw
  %10: vec2<f32> = %0.zy
  output 0.xyzw = %0
  output 1.xy = %2
  output 2.xy = %10
  return
}
";
    assert_eq!(iridia_ir::text::print(&q.entry), want);
    let input = [1.0f32, 2.0, 3.0, 4.0].map(f32::to_bits);
    assert_eq!(run(&p, &[input], &[]), run(&q, &[input], &[]));
}

#[test]
fn specialise_folds_a_known_state_and_the_branch_it_decides() {
    let text = "iridia-ir 1
fn main {
  %0: bool = state r0
  %1: f32 = input 0.x
  if %0 {
    %2: f32 = fneg %1
    yield %2
  } else {
    yield %1
  } -> %3: f32
  output 0.x = %3
  return
}
";
    let state = Resource {
        id: ResourceId(0),
        kind: ResourceKind::PipelineState(StateParam::Projected(0)),
        count: 1,
        provenance: ResourceProvenance {
            name: "projected0".into(),
            ..ResourceProvenance::default()
        },
    };
    let p = from_text(text, &[F1], &[F1], vec![state]);
    for (known, want) in [(true, -5.0f32), (false, 5.0)] {
        let mut q = p.clone();
        iridia_passes::specialise(&mut q, &[(ResourceId(0), [u64::from(known), 0, 0, 0])]).unwrap();
        iridia_passes::simplify(&mut q.entry);
        iridia_ir::validate(&q).unwrap();
        assert!(q.interface.resources.is_empty());
        let printed = iridia_ir::text::print(&q.entry);
        assert!(
            !printed.contains("if ") && !printed.contains("state"),
            "{printed}"
        );
        let (out, _) = run(&q, &[f(5.0)], &[]).unwrap();
        assert_eq!(out[0].unwrap()[0], want.to_bits());
        // and the unspecialised program agrees, given the state
        let (out, _) = run(&p, &[f(5.0)], &[(ResourceId(0), [u32::from(known), 0, 0, 0])]).unwrap();
        assert_eq!(out[0].unwrap()[0], want.to_bits());
        iridia_naga::wgsl(&q).unwrap();
    }
}
