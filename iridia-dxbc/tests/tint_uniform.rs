#![allow(clippy::unwrap_used)]

//! Both writers held to Tint, whose uniformity analysis is stricter than naga's: a loop
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
    for (name, wgsl) in [
        ("scan-own", iridia_wgsl::wgsl(&p, &iridia_wgsl::Profile::BROWSER)),
        (
            "scan-naga",
            iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER)
                .map_err(|e| iridia_wgsl::Error(e.to_string())),
        ),
    ] {
        let wgsl = wgsl.unwrap_or_else(|e| panic!("{e}"));
        if let Err(e) = tint.check(name, &wgsl) {
            panic!("{e}\n{wgsl}");
        }
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
fn tint_keeps_a_counter_forwarded_through_a_divergent_inner_loop_uniform() {
    let Some(tint) = wgsl_run::tint() else {
        println!("tools.tint is not configured: skipped");
        return;
    };
    let p = program(THROUGH_AN_INNER_LOOP);
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER).unwrap_or_else(|e| panic!("{e}"));
    if let Err(e) = tint.check("forwarded-counter-naga", &wgsl) {
        panic!("{e}\n{wgsl}");
    }
}

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
fn a_branch_local_origin_remains_in_storage_after_its_scope_ends() {
    let Some(tint) = wgsl_run::tint() else {
        println!("tools.tint is not configured: skipped");
        return;
    };
    let p = program(FROM_A_CLOSED_ARM);
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER).unwrap_or_else(|e| panic!("{e}"));
    if let Err(e) = tint.check("closed-origin-naga", &wgsl) {
        panic!("{e}\n{wgsl}");
    }
}

/// D3D wave reductions operate on the active lanes at the call site, including divergent flow.
#[test]
fn subgroup_diagnostics_preserve_a_divergent_call_site() {
    let Some(tint) = wgsl_run::tint() else {
        println!("tools.tint is not configured: skipped");
        return;
    };
    let p = program(
        "iridia-ir 1
fn main {
  var sink: u32
  %0: vec3<u32> = input 0.xyz
  %1: u32 = %0.x
  %2: u32 = u32(1)
  %3: u32 = and %1, %2
  %4: bool = ucmp.eq %3, %2
  if %4 {
    %5: u32 = wave.add.reduce %1
    sink = %5
    yield
  } else {
    yield
  }
  return
}
",
    );
    let profile = iridia_wgsl::Profile::BROWSER.with(iridia_wgsl::Capabilities::of(&[
        iridia_wgsl::Capability::Subgroup,
    ]));
    let wgsl = iridia_wgsl::wgsl(&p, &profile).unwrap_or_else(|e| panic!("{e}"));
    assert!(wgsl.contains("diagnostic(off, subgroup_uniformity);"), "{wgsl}");
    tint.check("subgroup-divergent-filtered", &wgsl)
        .unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    let unfiltered = wgsl.replace("diagnostic(off, subgroup_uniformity);\n", "");
    let e = tint.check("subgroup-divergent-control", &unfiltered).unwrap_err();
    assert!(e.contains("subgroupAdd"), "{e}");
    // Unknown diagnostic rules are warnings: only remove Naga 30's unsupported enable directive.
    let native = wgsl.strip_prefix("enable subgroups;\n").unwrap();
    let module = naga::front::wgsl::parse_str(native).unwrap_or_else(|e| panic!("{e}\n{native}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}
