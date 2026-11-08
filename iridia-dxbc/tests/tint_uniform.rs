#![allow(clippy::unwrap_used)]

//! The scan loop of `iridia-wgsl`'s `tests/uniform.rs` held to Tint, whose uniformity analysis is stricter than naga's: a loop
//! whose counter shares a `vec4` with a thread-dependent lane is refused when the state is one vector variable and accepted
//! when the writer keeps it a lane at a time. Needs `tools.tint`.

mod inputs;
mod wgsl_run;

use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value, Varying,
        VaryingProvenance,
    },
};

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
fn tint_takes_the_scan_loop_a_lane_at_a_time_and_refuses_it_as_a_vector() {
    let Some(tint) = wgsl_run::tint() else {
        println!("tools.tint is not configured: skipped");
        return;
    };
    let p = program(SCAN);
    let wgsl = iridia_wgsl::wgsl(&p, &iridia_wgsl::Profile::BROWSER).unwrap_or_else(|e| panic!("{e}"));
    if let Err(e) = tint.check("scan", &wgsl) {
        panic!(
            "{e}
{wgsl}"
        );
    }
    let vector = "@compute @workgroup_size(64)
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

    let e = tint.check("scan-vector", vector).unwrap_err();
    assert!(e.contains("workgroupBarrier"), "{e}");
}
