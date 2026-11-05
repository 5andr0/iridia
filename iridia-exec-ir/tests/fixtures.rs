//! Programs in IR text, each run on inputs whose results are worked by hand from the operations' definitions.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value, Varying,
        VaryingProvenance,
    },
};

fn varying(name: &str, index: u32, value: (ScalarType, u8), system: Option<SystemValue>) -> Varying {
    Varying {
        label: Label {
            name: name.into(),
            index,
        },
        system,
        value: Value {
            scalar: value.0,
            width: value.1,
        },
        interp: None,
        used: true,
        provenance: VaryingProvenance::default(),
    }
}

/// A pixel program of the text's function, with `TEXCOORD` inputs and `SV_Target` outputs of the shapes given.
fn program(text: &str, inputs: &[(ScalarType, u8)], outputs: &[(ScalarType, u8)]) -> Program {
    let entry = iridia_ir::text::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    let n = |i: usize| u32::try_from(i).unwrap();
    let p = Program {
        stage: Stage::Pixel,
        entry,
        interface: Interface {
            stage: Stage::Pixel,
            inputs: inputs
                .iter()
                .enumerate()
                .map(|(i, v)| varying("TEXCOORD", n(i), *v, None))
                .collect(),
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, v)| varying("SV_Target", n(i), *v, Some(SystemValue::Target(n(i)))))
                .collect(),
            resources: Vec::new(),
            facts: StageFacts::default(),
        },
        conventions: iridia_ir::Conventions::D3D,
        float: FloatEnv::default(),
        provenance: Provenance {
            format: ShaderFormat::D3d9,
            target: "fixture".into(),
            lifter: "text".into(),
            lifter_version: 1,
            source_hash: None,
            instruction_count: None,
            policies: Vec::new(),
        },
        notes: Vec::new(),
    };
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    assert_eq!(
        iridia_ir::text::print(&p.entry),
        text,
        "the fixture is in the printer's own form"
    );
    p
}

fn run(p: &Program, inputs: &[[u32; 4]]) -> iridia_exec_ir::Run {
    let mut sampler = |_| Err::<[f32; 4], String>("no textures".into());
    let mut bound = iridia_exec_ir::Bound {
        inputs,
        buffers: &[],
        state: &[],
        sampler: &mut sampler,
    };
    iridia_exec_ir::run(p, &mut bound).unwrap()
}

fn f(x: [f32; 4]) -> [u32; 4] {
    x.map(f32::to_bits)
}

const F4: (ScalarType, u8) = (ScalarType::Float, 4);

/// Each lane as a float, NaN as NaN whatever its payload.
fn floats(lanes: [u32; 4]) -> [Option<f32>; 4] {
    lanes.map(|b| Some(f32::from_bits(b)).filter(|x| !x.is_nan()))
}

#[test]
fn nan_rules_and_rounding_are_the_stated_ones() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0)
  %2: vec4<f32> = fmin.minnum %0, %1
  %3: vec4<f32> = saturate.zero %0
  %4: vec4<f32> = fround.nearestaway %0
  %5: vec4<f32> = fround.nearesteven %0
  output 0.xyzw = %2
  output 1.xyzw = %3
  output 2.xyzw = %4
  output 3.xyzw = %5
  return
}
",
        &[F4],
        &[F4, F4, F4, F4],
    );
    let r = run(&p, &[f([f32::NAN, 2.5, -2.5, 0.499_999_97])]);
    let out = |n: usize| floats(r.outputs[n].unwrap());
    // minNum: a NaN operand gives the other
    assert_eq!(out(0), [Some(1.0), Some(1.0), Some(-2.5), Some(0.499_999_97)]);
    // saturate.zero: NaN gives 0
    assert_eq!(out(1), [Some(0.0), Some(1.0), Some(0.0), Some(0.499_999_97)]);
    // ties away from zero; the largest float below 0.5 rounds to 0 (a floor(x + 0.5) would give 1)
    assert_eq!(out(2), [None, Some(3.0), Some(-3.0), Some(0.0)]);
    // ties to even
    assert_eq!(out(3), [None, Some(2.0), Some(-2.0), Some(0.0)]);
}

#[test]
fn zero_wins_multiplication_and_saturating_conversion() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<f32> = vec4<f32>(0x7f800000, 0.0, 0.0, 2.0)
  %2: vec4<f32> = fmul.zerowins %0, %1
  %3: vec4<f32> = fmul.ieee %0, %1
  %4: vec4<i32> = convert.saturate %0
  %5: vec4<u32> = convert.saturate %0
  output 0.xyzw = %2
  output 1.xyzw = %3
  output 2.xyzw = %4
  output 3.xyzw = %5
  return
}
",
        &[F4],
        &[F4, F4, (ScalarType::Sint, 4), (ScalarType::Uint, 4)],
    );
    let r = run(&p, &[f([0.0, f32::INFINITY, f32::NAN, -3e9])]);
    // zero wins: 0 * inf, inf * 0 and NaN * 0 are +0
    assert_eq!(
        floats(r.outputs[0].unwrap()),
        [Some(0.0), Some(0.0), Some(0.0), Some(-6e9)]
    );
    assert_eq!(r.outputs[0].unwrap()[..3], [0, 0, 0], "+0, not -0");
    // IEEE: 0 * inf is NaN, inf * 0 is NaN, NaN stays NaN
    assert_eq!(floats(r.outputs[1].unwrap()), [None, None, None, Some(-6e9)]);
    // truncation, saturated to the range, NaN 0; inf is past every i32
    let i = r.outputs[2].unwrap().map(|b| b.cast_signed());
    assert_eq!(i, [0, i32::MAX, 0, i32::MIN]);
    assert_eq!(r.outputs[3].unwrap(), [0, u32::MAX, 0, 0]);
}

#[test]
fn masked_loads_and_stores_take_and_give_their_lanes_in_order() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec2<f32> = input 0.yw
  output 0.xz = %0
  %1: f32 = input 0.x
  output 0.w = %1
  %2: vec2<f32> = %0.yx
  output 1.yz = %2
  output 0.x = %1
  return
}
",
        &[F4],
        &[F4, F4],
    );
    let r = run(&p, &[f([1.0, 2.0, 3.0, 4.0])]);
    // x is written twice (2, then 1); y never: zero
    assert_eq!(r.outputs[0].unwrap(), f([1.0, 0.0, 4.0, 1.0]));
    assert_eq!(r.outputs[1].unwrap(), f([0.0, 4.0, 2.0, 0.0]));
}

#[test]
fn a_loop_carries_its_values_out_by_break() {
    // sum of i for i = 0, 1, 2, ... while i < n
    let p = program(
        "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f32 = f32(0.0)
  %2: f32 = f32(1.0)
  loop '0 (%3: f32, %4: f32) = (%1, %1) {
    %5: bool = fcmp.ge %3, %0
    if %5 {
      br '0 %4
    } else {
      yield
    }
    %6: f32 = fadd %4, %3
    %7: f32 = fadd %3, %2
    continue '0 %7, %6
  } -> %8: f32
  output 0.x = %8
  return
}
",
        &[(ScalarType::Float, 1)],
        &[(ScalarType::Float, 1)],
    );
    for (n, sum) in [(0.0f32, 0.0f32), (1.0, 0.0), (2.5, 3.0), (4.0, 6.0), (-1.0, 0.0)] {
        assert_eq!(
            run(&p, &[[n.to_bits(), 0, 0, 0]]).outputs[0].unwrap()[0],
            sum.to_bits(),
            "n {n}"
        );
    }
}

#[test]
fn a_kill_ends_the_invocation_with_what_it_stored() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  output 0.x = %0
  %1: f32 = f32(0.0)
  %2: bool = fcmp.lt %0, %1
  if %2 {
    kill
  } else {
    yield
  }
  %3: f32 = fneg %0
  output 0.x = %3
  return
}
",
        &[(ScalarType::Float, 1)],
        &[(ScalarType::Float, 1)],
    );
    let r = run(&p, &[[(-2.0f32).to_bits(), 0, 0, 0]]);
    assert!(r.killed);
    assert_eq!(r.outputs[0].unwrap()[0], (-2.0f32).to_bits());
    let r = run(&p, &[[2.0f32.to_bits(), 0, 0, 0]]);
    assert!(!r.killed);
    assert_eq!(r.outputs[0].unwrap()[0], (-2.0f32).to_bits());
}

#[test]
fn text_that_is_not_the_form_is_refused_with_its_line() {
    for (text, line, words) in [
        ("iridia-ir 0\nfn main {\n  return\n}\n", 1, "first line"),
        (
            "iridia-ir 1\nfn main {\n  %0: f32 = frobnicate %1\n  return\n}\n",
            3,
            "no operation frobnicate",
        ),
        (
            "iridia-ir 1\nfn main {\n  %0: f32 = f32(1.0)\n  %0: f32 = f32(2.0)\n  return\n}\n",
            4,
            "defined twice",
        ),
        (
            "iridia-ir 1\nfn main {\n  %0: vec5<f32> = undef\n  return\n}\n",
            3,
            "not a type",
        ),
        (
            "iridia-ir 1\nfn main {\n  %0: f32 = input 0.yx\n  return\n}\n",
            3,
            "out of order",
        ),
    ] {
        let e = iridia_ir::text::parse(text).unwrap_err();
        assert_eq!(e.line, line, "{e}");
        assert!(e.message.contains(words), "{e}");
    }
}
