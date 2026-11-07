//! A loop whose counter shares a `vec4` with a thread-dependent lane: WGSL's uniformity analysis takes a vector as the least
//! uniform of its lanes, so a barrier after a loop that exits on such a counter is refused, though the counter is the same in
//! every thread. The writer keeps a loop's vector variables a lane at a time, so the text is accepted. Upstream naga has the
//! same rule as Tint, so the test needs no Tint.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value, Varying,
        VaryingProvenance,
    },
};
use iridia_wgsl::Profile;

/// A compute program of the text's function, with the group-local invocation id as input 0.
fn program(text: &str) -> Program {
    let entry = iridia_ir::text::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    let p = Program {
        stage: Stage::Compute,
        entry,
        patch: None,
        workgroup: Some([64, 1, 1]),
        interface: Interface {
            stage: Stage::Compute,
            inputs: vec![Varying {
                label: Label {
                    name: "SV_GroupThreadID".into(),
                    index: 0,
                },
                system: Some(SystemValue::GroupThreadId),
                value: Value {
                    scalar: ScalarType::Uint,
                    width: 3,
                },
                interp: None,
                used: true,
                provenance: VaryingProvenance::default(),
            }],
            outputs: Vec::new(),
            resources: Vec::new(),
            facts: StageFacts::default(),
        },
        conventions: iridia_ir::Conventions::D3D,
        float: FloatEnv::default(),
        provenance: Provenance {
            format: ShaderFormat::Dxbc,
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
    p
}

/// Whether naga reads and validates `wgsl`, with its message where not.
fn accepted(wgsl: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| e.emit_to_string(wgsl))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map(|_| ())
    .map_err(|e| format!("{e:?}"))
}

/// A scan's loop: lanes x and y of its state are the thread's, lane w the stride, which doubles to 64 whatever the thread
/// (as the prefix sums of the Elden Ring and Riders Republic light organisers do).
const SCAN: &str = "iridia-ir 1
fn main {
  %0: vec3<u32> = input 0.xyz
  %1: u32 = %0.x
  %2: u32 = u32(0)
  %3: u32 = u32(1)
  %4: vec4<u32> = compose %1, %2, %2, %3
  loop '0 (%5: vec4<u32>) = (%4) {
    %6: u32 = %5.w
    %7: u32 = u32(64)
    %8: bool = ucmp.ge %6, %7
    if %8 {
      br '0 %5
    } else {
      yield
    }
    %9: u32 = %5.x
    %10: u32 = %5.y
    %11: u32 = %5.z
    barrier execution workgroup storage.workgroup
    %12: u32 = shl %6, %3
    %13: vec4<u32> = compose %9, %10, %11, %12
    continue '0 %13
  } -> %14: vec4<u32>
  return
}
";

#[test]
fn a_loop_state_in_a_vector_is_kept_a_lane_at_a_time() {
    let p = program(SCAN);
    let wgsl = iridia_wgsl::wgsl(&p, &Profile::BROWSER).unwrap_or_else(|e| panic!("{e}"));
    // naga reads and validates it (its analysis is not Tint's: see the control below)
    if let Err(e) = accepted(&wgsl) {
        panic!(
            "{e}
{wgsl}"
        );
    }
    // the state is scalars, not one `vec4`: the lanes `continue` hands back as they are (x, y, z) are the values from before
    // the loop, the stride (w) is its own variable, and the exit reads that
    assert!(
        !wgsl.contains("vec4<u32>;"),
        "a vector variable:
{wgsl}"
    );
    assert!(wgsl.contains("var p5_3: u32;"), "{wgsl}");
    for lane in ["var p5_0", "var p5_1", "var p5_2"] {
        assert!(
            !wgsl.contains(lane),
            "{lane} is a lane that never changes:
{wgsl}"
        );
    }
    assert!(wgsl.contains("let q5_3 = p5_3;"), "{wgsl}");
    assert!(
        wgsl.contains("let v6 = q5_3;"),
        "the stride is read as its own lane:
{wgsl}"
    );
}

/// A loop whose counter (lane x) halves, whose state passes through an inner loop under an `if` on the thread's index: the
/// inner loop hands lane x back as it is, so after the `if` the counter is the one from before it, not a variable assigned
/// in a branch that only some threads take (which a compiler takes as different in every thread, and so the exit and the
/// barrier after it).
const THROUGH_AN_INNER_LOOP: &str = "iridia-ir 1
fn main {
  %0: vec3<u32> = input 0.xyz
  %1: u32 = %0.x
  %2: u32 = u32(0)
  %3: u32 = u32(64)
  %7: u32 = u32(1)
  %4: vec4<u32> = compose %3, %2, %2, %2
  loop '0 (%5: vec4<u32>) = (%4) {
    %6: u32 = %5.x
    %8: bool = ucmp.lt %6, %7
    if %8 {
      br '0 %5
    } else {
      yield
    }
    %9: bool = ucmp.lt %1, %6
    if %9 {
      loop '1 (%10: vec4<u32>) = (%5) {
        %11: u32 = %10.y
        %12: u32 = u32(10)
        %13: bool = ucmp.ge %11, %12
        if %13 {
          br '1 %10
        } else {
          yield
        }
        %14: u32 = iadd %11, %7
        %15: u32 = %10.x
        %16: u32 = %10.z
        %17: u32 = %10.w
        %18: vec4<u32> = compose %15, %14, %16, %17
        continue '1 %18
      } -> %19: vec4<u32>
      yield %19
    } else {
      yield %5
    } -> %20: vec4<u32>
    barrier execution workgroup storage.workgroup
    %21: u32 = %20.x
    %23: u32 = shru %21, %7
    %25: vec4<u32> = compose %23, %2, %2, %2
    continue '0 %25
  } -> %26: vec4<u32>
  return
}
";

#[test]
fn a_lane_passed_through_an_inner_loop_under_a_divergent_if_is_the_value_from_before_it() {
    let p = program(THROUGH_AN_INNER_LOOP);
    let wgsl = iridia_wgsl::wgsl(&p, &Profile::BROWSER).unwrap_or_else(|e| panic!("{e}"));
    if let Err(e) = accepted(&wgsl) {
        panic!(
            "{e}
{wgsl}"
        );
    }
    // the inner loop's x, z, w never change, so they are the outer state's; only its counter (y) is a variable
    assert!(wgsl.contains("var p10_1: u32;"), "{wgsl}");
    for lane in ["var p10_0", "var p10_2", "var p10_3"] {
        assert!(
            !wgsl.contains(lane),
            "{lane}:
{wgsl}"
        );
    }
    // the inner loop's results x, z, w are what it started as; the `if` has the same x from both arms: the outer counter
    assert!(wgsl.contains("let q19_0 = q5_0;"), "{wgsl}");
    assert!(
        wgsl.contains("let q20_0 = q5_0;"),
        "the counter after the if is the one before it:
{wgsl}"
    );
    assert!(!wgsl.contains("var p20_0"), "{wgsl}");
    assert!(!wgsl.contains("var p19_0"), "{wgsl}");
}

/// A loop whose first state lane starts as the value an `if` arm computed (the other arm returns, so the `if` is that value)
/// and is handed back unchanged. That value is defined inside the arm, whose braces have closed by the loop, so the loop's
/// lane is a variable: a name forwarded from there would be one no code after the arm can read.
const FROM_A_CLOSED_ARM: &str = "iridia-ir 1
fn main {
  %0: vec3<u32> = input 0.xyz
  %1: u32 = %0.x
  %2: u32 = u32(8)
  %3: bool = ucmp.lt %1, %2
  if %3 {
    %4: u32 = iadd %1, %2
    yield %4
  } else {
    return
  } -> %5: u32
  %6: u32 = u32(0)
  loop '0 (%7: u32, %8: u32) = (%5, %6) {
    %9: u32 = u32(4)
    %10: bool = ucmp.ge %8, %9
    if %10 {
      br '0 %7
    } else {
      yield
    }
    %11: u32 = u32(1)
    %12: u32 = iadd %8, %11
    continue '0 %7, %12
  } -> %13: u32
  return
}
";

#[test]
fn a_lane_is_never_forwarded_from_a_value_of_a_region_that_has_closed() {
    let p = program(FROM_A_CLOSED_ARM);
    let wgsl = iridia_wgsl::wgsl(&p, &Profile::BROWSER).unwrap_or_else(|e| panic!("{e}"));
    // naga refuses a name out of scope
    if let Err(e) = accepted(&wgsl) {
        panic!(
            "{e}
{wgsl}"
        );
    }
    assert!(
        wgsl.contains("var p7: u32;"),
        "the lane is a variable:
{wgsl}"
    );
}

/// Upstream naga's uniformity analysis is more lenient than Tint's: it accepts a barrier after the exit of a loop whose state is
/// one `vec4` variable with a thread lane. Tint refuses it (`iridia-dxbc`'s `tint_uniform` test), so the structure above is what
/// keeps the text valid in a browser, and naga cannot tell the two apart.
#[test]
fn naga_accepts_the_vector_variable_that_tint_refuses() {
    let wgsl = "@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) id: vec3<u32>) {
    var s = vec4<u32>(id.x, 0u, 0u, 1u);
    loop {
        if (s.w >= 64u) {
            break;
        }
        workgroupBarrier();
        s = vec4<u32>(s.x, s.y, s.z, s.w << 1u);
    }
}
";
    accepted(wgsl).unwrap();
}
