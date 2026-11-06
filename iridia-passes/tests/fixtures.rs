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
        iridia_naga::wgsl(&q, &iridia_naga::Profile::NAGA).unwrap();
    }
}

#[test]
fn fold_computes_integer_and_bool_operations_on_literals() {
    let text = "iridia-ir 1
fn main {
  %0: u32 = u32(4294967295)
  %1: u32 = u32(2)
  %2: u32 = iadd %0, %1
  %3: i32 = i32(-3)
  %4: i32 = i32(2)
  %5: bool = scmp.lt %3, %4
  %6: bool = ucmp.lt %3, %4
  %7: u32 = u32(0)
  %8: u32 = udiv.allones %1, %7
  %9: u32 = urem.dividend %1, %7
  %10: u32 = udiv.undef %1, %7
  %11: i32 = smin %3, %4
  %12: bool = and %5, %6
  %13: bool = not %12
  %14: u32 = select %13, %2, %8
  %15: vec2<u32> = compose %14, %9
  %16: vec2<u32> = %15.yx
  %17: f32 = f32(1.5)
  %18: f32 = fneg %17
  %19: f32 = fadd %17, %17
  %20: f32 = input 0.x
  return
}
";
    let mut entry = iridia_ir::text::parse(text).unwrap();
    let folded = iridia_passes::fold(&mut entry);
    let lit = |v: usize| match entry.values[v].op {
        iridia_ir::Op::Literal(bits) => Some(bits),
        _ => None,
    };
    // wrapping in the type's width; a signed comparison of the bits as signed, an unsigned one as unsigned
    assert_eq!(lit(2), Some([1, 0, 0, 0]));
    assert_eq!(lit(5), Some([1, 0, 0, 0]));
    assert_eq!(lit(6), Some([0, 0, 0, 0]));
    // a division by zero by its rule; one whose rule is undefined is the target's
    assert_eq!(lit(8), Some([u64::from(u32::MAX), 0, 0, 0]));
    assert_eq!(lit(9), Some([2, 0, 0, 0]));
    assert_eq!(lit(10), None);
    assert_eq!(lit(11), Some([u64::from((-3i32).cast_unsigned()), 0, 0, 0]));
    assert_eq!(lit(13), Some([1, 0, 0, 0]));
    assert_eq!(lit(14), Some([1, 0, 0, 0]));
    assert_eq!(lit(16), Some([2, 1, 0, 0]));
    // a sign flip is exact everywhere; float arithmetic depends on the target's denormals and precision
    assert_eq!(lit(18), Some([u64::from((-1.5f32).to_bits()), 0, 0, 0]));
    assert_eq!(lit(19), None);
    assert_eq!(lit(20), None);
    assert_eq!(folded, 12);
    assert_eq!(iridia_passes::fold(&mut entry), 0);
}

#[test]
fn specialise_folds_a_texture_dimension_behind_a_comparison() {
    // the ps_1_x shape: a sampler of unknown dimension reads its dimension as state and picks the sample by it
    let text = "iridia-ir 1
fn main {
  %0: u32 = state r0
  %1: u32 = u32(4)
  %2: bool = ucmp.eq %0, %1
  %3: f32 = input 0.x
  if %2 {
    %4: f32 = fneg %3
    yield %4
  } else {
    yield %3
  } -> %5: f32
  output 0.x = %5
  return
}
";
    let state = Resource {
        id: ResourceId(0),
        kind: ResourceKind::PipelineState(StateParam::TextureDim(0)),
        count: 1,
        provenance: ResourceProvenance {
            name: "dim0".into(),
            ..ResourceProvenance::default()
        },
    };
    let p = from_text(text, &[F1], &[F1], vec![state]);
    for (dim, want) in [(4u32, -5.0f32), (2, 5.0)] {
        let mut q = p.clone();
        iridia_passes::specialise(&mut q, &[(ResourceId(0), [u64::from(dim), 0, 0, 0])]).unwrap();
        iridia_passes::simplify(&mut q.entry);
        iridia_ir::validate(&q).unwrap();
        let printed = iridia_ir::text::print(&q.entry);
        assert!(
            !printed.contains("if ") && !printed.contains("state"),
            "{printed}"
        );
        let (out, _) = run(&q, &[f(5.0)], &[]).unwrap();
        assert_eq!(out[0].unwrap()[0], want.to_bits());
        let (out, _) = run(&p, &[f(5.0)], &[(ResourceId(0), [dim, 0, 0, 0])]).unwrap();
        assert_eq!(out[0].unwrap()[0], want.to_bits());
        iridia_naga::wgsl(&q, &iridia_naga::Profile::BROWSER).unwrap();
    }
}
