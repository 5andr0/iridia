//! Programs in IR text written as WGSL: what each construct and rule becomes, and that upstream naga reads every text back
//! and validates it.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        Interface, Label, Resource, ResourceId, ResourceKind, ResourceProvenance, ScalarType, ShaderFormat,
        Stage, StageFacts, StateParam, SystemValue, Value, Varying, VaryingProvenance,
    },
};
use iridia_wgsl::{Capabilities, Capability, Profile};

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
fn program(
    text: &str,
    inputs: &[(ScalarType, u8)],
    outputs: &[(ScalarType, u8)],
    resources: Vec<Resource>,
) -> Program {
    let entry = iridia_ir::text::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    let n = |i: usize| u32::try_from(i).unwrap();
    let p = Program {
        stage: Stage::Pixel,
        entry,
        patch: None,
        workgroup: None,
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
            resources,
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
    p
}

const F4: (ScalarType, u8) = (ScalarType::Float, 4);

/// What upstream naga's validator is given for a profile: the capabilities it allows.
fn naga_capabilities(profile: &Profile) -> naga::valid::Capabilities {
    use naga::valid::Capabilities as C;
    [
        (Capability::Float64, C::FLOAT64),
        (Capability::Float16, C::SHADER_FLOAT16),
        (Capability::Int16, C::SHADER_INT16),
        (Capability::Int64, C::SHADER_INT64),
        (Capability::PrimitiveIndex, C::PRIMITIVE_INDEX),
        (Capability::MultisampledShading, C::MULTISAMPLED_SHADING),
        (Capability::CubeArrayTextures, C::CUBE_ARRAY_TEXTURES),
        (Capability::ClipDistances, C::CLIP_DISTANCES),
        (Capability::Subgroup, C::SUBGROUP),
        (Capability::SubgroupVertexStage, C::SUBGROUP_VERTEX_STAGE),
    ]
    .into_iter()
    .filter(|(c, _)| profile.capabilities.contains(*c))
    .fold(C::empty(), |all, (_, c)| all | c)
}

/// The WGSL of `p`, read back and validated by upstream naga with the profile's capabilities.
fn written(p: &Program, profile: &Profile) -> String {
    let wgsl = iridia_wgsl::wgsl(p, profile).unwrap_or_else(|e| panic!("{e}"));
    let module =
        naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{}\n{wgsl}", e.emit_to_string(&wgsl)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga_capabilities(profile))
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}\n{wgsl}"));
    wgsl
}

#[test]
fn masked_loads_and_stores_are_swizzles_and_lane_assignments() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec2<f32> = input 0.yw
  output 0.xz = %0
  %1: f32 = input 0.x
  output 0.w = %1
  output 0.x = %1
  return
}
",
        &[F4],
        &[F4],
        Vec::new(),
    );
    // the input (a, b, c, d): (b, 0, d, 0), then (b, 0, d, a), then (a, 0, d, a); a lane never stored is zero
    let wgsl = written(&p, &Profile::NAGA);
    for line in [
        "let v0 = TEXCOORD0.yw;",
        "o0.x = v0.x;",
        "o0.z = v0.y;",
        "o0.w = v1;",
        "o0.x = v1;",
        "return Out(o0);",
    ] {
        assert!(wgsl.contains(line), "{line}\n{wgsl}");
    }
}

#[test]
fn a_zero_wins_multiplication_of_vectors_validates() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<f32> = input 1.xyzw
  %2: vec4<f32> = fmul.zerowins %0, %1
  output 0.xyzw = %2
  %3: f32 = input 0.x
  %4: f32 = fmul.zerowins %3, %3
  output 1.x = %4
  return
}
",
        &[F4, F4],
        &[F4, (ScalarType::Float, 1)],
        Vec::new(),
    );
    // each lane: zero where either operand is zero, the product otherwise; `|` joins the bool vectors
    let wgsl = written(&p, &Profile::NAGA);
    assert!(
        wgsl.contains(
            "select((v0 * v1), vec4<f32>(0.0f), ((v0 == vec4<f32>(0.0f)) | (v1 == vec4<f32>(0.0f))))"
        ),
        "{wgsl}"
    );
}

#[test]
fn unspecialised_pipeline_state_is_refused_by_name() {
    let state = Resource {
        id: ResourceId(0),
        kind: ResourceKind::PipelineState(StateParam::Projected(0)),
        count: 1,
        provenance: ResourceProvenance {
            name: "projected0".into(),
            ..ResourceProvenance::default()
        },
    };
    let p = program(
        "iridia-ir 1
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
",
        &[(ScalarType::Float, 1)],
        &[(ScalarType::Float, 1)],
        vec![state],
    );
    let e = iridia_wgsl::wgsl(&p, &Profile::NAGA).unwrap_err();
    assert!(e.0.contains("specialise"), "{e}");
}

#[test]
fn a_construct_has_its_results_in_variables_read_once_after_it() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: bool = fcmp.lt %0, %0
  if %1 {
    %2: f32 = fneg %0
    yield %2
  } else {
    yield %0
  } -> %3: f32
  output 0.x = %3
  return
}
",
        &[(ScalarType::Float, 1)],
        &[(ScalarType::Float, 1)],
        Vec::new(),
    );
    let wgsl = written(&p, &Profile::BROWSER);
    for line in [
        "var p3: f32;",
        "if (v1) {",
        "p3 = v2;",
        "} else {",
        "p3 = v0;",
        "let v3 = p3;",
    ] {
        assert!(wgsl.contains(line), "{line}\n{wgsl}");
    }
}

#[test]
fn a_capability_the_profile_lacks_is_refused_before_writing() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f64 = convert.undef %0
  %2: f64 = fadd %1, %1
  %3: f32 = convert.undef %2
  output 0.x = %3
  return
}
",
        &[(ScalarType::Float, 1)],
        &[(ScalarType::Float, 1)],
        Vec::new(),
    );
    // f64 itself, and 64-bit integers, which an f64's bits are moved through
    let needs = iridia_wgsl::needs(&p);
    assert_eq!(
        needs.iter().map(|r| r.capability).collect::<Vec<_>>(),
        [Capability::Float64, Capability::Int64],
        "{needs:?}"
    );
    // browser WGSL has no f64: the preflight names the value
    let e = iridia_wgsl::wgsl(&p, &Profile::BROWSER).unwrap_err();
    assert!(
        e.0.starts_with("browser WGSL cannot carry this program: value %1 is f64"),
        "{e}"
    );
    // naga's dialect has it
    let wgsl = written(&p, &Profile::NAGA);
    assert!(wgsl.contains("f64"), "{wgsl}");
}

#[test]
fn the_largest_float_is_written_as_its_bits() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: f32 = f32(0x7f7fffff)
  output 0.x = %0
  %1: vec2<f32> = vec2<f32>(0xff7fffff, 0x7f7ffffe)
  output 0.yz = %1
  return
}
",
        &[],
        &[F4],
        Vec::new(),
    );
    let wgsl = written(&p, &Profile::BROWSER);
    // its shortest decimal, 3.4028235e38, lies above it: Tint refuses it as out of range (WGSL rounds it back)
    assert!(!wgsl.contains("34028235"), "{wgsl}");
    assert!(wgsl.contains("bitcast<f32>(2139095039u)"), "{wgsl}");
    assert!(wgsl.contains("bitcast<f32>(4286578687u)"), "{wgsl}");
    // the float below it has a shortest decimal below it, written as a decimal
    assert!(wgsl.contains("3.4028233e38f"), "{wgsl}");
}

#[test]
fn browser_wgsl_enables_the_subgroups_it_uses() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f32 = wave.add.reduce %0
  output 0.x = %1
  return
}
",
        &[F4],
        &[F4],
        Vec::new(),
    );
    let subgroups = Capabilities::of(&[Capability::Subgroup]);
    // a browser requires the directive before a subgroup operation
    let browser = Profile::BROWSER.with(subgroups);
    let wgsl = iridia_wgsl::wgsl(&p, &browser).unwrap();
    assert!(wgsl.starts_with("enable subgroups;\n"), "{wgsl}");
    assert!(wgsl.contains("subgroupAdd("), "{wgsl}");
    // naga's reader refuses it, and naga's dialect leaves it out
    assert!(naga::front::wgsl::parse_str(&wgsl).is_err());
    let wgsl = written(&p, &Profile::NAGA);
    assert!(!wgsl.contains("enable subgroups"), "{wgsl}");
    // a program without one states no directive
    let plain = program(
        "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  output 0.x = %0
  return
}
",
        &[F4],
        &[F4],
        Vec::new(),
    );
    let wgsl = iridia_wgsl::wgsl(&plain, &browser).unwrap();
    assert!(!wgsl.contains("enable"), "{wgsl}");
}

/// A pixel program writing `output 0.x` from a function of literals; its text is read back and validated by naga, whose
/// constant evaluation refuses an operation on literals that comes to infinity or NaN.
fn constants_only(body: &str) -> Program {
    program(
        &format!("iridia-ir 1\nfn main {{\n{body}  return\n}}\n"),
        &[],
        &[F4],
        Vec::new(),
    )
}

const SHOW_FOURS: &str = "SHOW";

#[test]
fn operations_on_literals_that_are_not_finite_are_written_so_naga_accepts_them() {
    for (name, body) in [
        (
            "log of zero",
            "  %0: f32 = f32(0.0)\n  %1: f32 = flog2 %0\n  output 0.x = %1\n",
        ),
        (
            "1 over the inverse square root of zero",
            "  %0: f32 = f32(0.0)\n  %1: f32 = finversesqrt %0\n  %2: f32 = f32(1.0)\n  %3: f32 = fdiv %2, %1\n  output 0.x = %3\n",
        ),
        (
            "inverse square root of -1",
            "  %0: f32 = f32(-1.0)\n  %1: f32 = finversesqrt %0\n  output 0.x = %1\n",
        ),
        (
            "1 over zero",
            "  %0: f32 = f32(1.0)\n  %1: f32 = f32(0.0)\n  %2: f32 = fdiv %0, %1\n  output 0.x = %2\n",
        ),
    ] {
        let p = constants_only(body);
        for profile in [&Profile::NAGA, &Profile::BROWSER] {
            let wgsl = written(&p, profile);
            // where the texts are wanted outside the test (for another validator), `SHOW` names a folder for them
            if let Some(dir) = std::env::var_os(SHOW_FOURS) {
                let which = if profile.directives == iridia_wgsl::Directives::Naga {
                    "naga"
                } else {
                    "browser"
                };
                let file = format!(
                    "{}/{}-{which}.wgsl",
                    dir.to_string_lossy(),
                    name.replace(' ', "_")
                );
                std::fs::write(file, &wgsl).unwrap();
            }
        }
    }
}

/// The control of the test above: the same operations with the literals written in them are what naga refuses, so the
/// fixtures there can fail.
#[test]
fn naga_refuses_the_operations_on_literals_when_they_are_written_inline() {
    for inline in [
        "log2(0.0f)",
        "1.0f / 0.0f",
        "inverseSqrt(-1.0f)",
        "1.0f / inverseSqrt(0.0f)",
    ] {
        let text = format!("fn f() -> f32 {{\n    return {inline};\n}}\n");
        let refused = match naga::front::wgsl::parse_str(&text) {
            Err(_) => true,
            Ok(module) => naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .is_err(),
        };
        assert!(refused, "{inline}");
    }
}
