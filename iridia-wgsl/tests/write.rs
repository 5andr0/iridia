//! Programs in IR text written as WGSL: what each construct and rule becomes, and that upstream naga reads every text back
//! and validates it.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        Access, D3dBinding, D3dClass, Interface, Label, Resource, ResourceId, ResourceKind,
        ResourceProvenance, SampleType, ScalarType, ShaderFormat, Stage, StageFacts, StateParam,
        StorageTarget, SystemValue, TextureDim, Value, Varying, VaryingProvenance,
    },
    target::ArrayClass,
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

/// The acceleration structure at `t0`, and a pixel program that starts a ray query of it (`rayquery init`) and runs `body`
/// after it, then returns.
fn ray_query_program(body: &str, outputs: &[(ScalarType, u8)]) -> Program {
    let text = format!(
        "iridia-ir 1
fn main {{
  query q0 flags 0x20
  %0: u32 = u32(2)
  %1: u32 = u32(255)
  %2: vec3<f32> = vec3<f32>(0x0, 0x0, 0x0)
  %3: f32 = f32(0x0)
  %4: vec3<f32> = vec3<f32>(0x0, 0x0, 0x3f800000)
  %5: f32 = f32(0x447a0000)
  rayquery init q0 r0 flags %0 mask %1 origin %2 tmin %3 direction %4 tmax %5
{body}  return
}}
"
    );
    ray_query_text(&text, outputs)
}

/// A pixel program of IR text over the acceleration structure at `t0`.
fn ray_query_text(text: &str, outputs: &[(ScalarType, u8)]) -> Program {
    let scene = Resource {
        id: ResourceId(0),
        kind: ResourceKind::AccelerationStructure,
        count: 1,
        provenance: ResourceProvenance {
            name: "scene".into(),
            d3d: Some(D3dBinding {
                class: D3dClass::ShaderResource,
                register: 0,
                space: 0,
            }),
            ..ResourceProvenance::default()
        },
    };
    program(text, &[], outputs, vec![scene])
}

#[test]
fn ray_queries_are_a_shared_requirement_a_browser_refuses_by_name() {
    let p = ray_query_program("  output 0.x = %0\n", &[(ScalarType::Uint, 1)]);
    let needs: Vec<_> = iridia_wgsl::needs(&p).into_iter().map(|r| r.capability).collect();
    assert_eq!(needs, [Capability::RayQuery]);
    // a browser has none, whatever it is allowed
    for profile in [
        Profile::BROWSER,
        Profile::BROWSER.with(Capabilities::of(&[Capability::RayQuery])),
    ] {
        let e = iridia_wgsl::wgsl(&p, &profile).unwrap_err().0;
        assert!(
            e.starts_with("browser WGSL has no ray queries: ") && e.contains("ray query q0"),
            "{e}"
        );
    }
    // wgpu's extension has them
    let wgsl = written(&p, &Profile::NAGA);
    assert!(wgsl.starts_with("enable wgpu_ray_query;"), "{wgsl}");
}

#[test]
fn a_ray_query_is_the_extension_s_calls_and_naga_reads_them_back() {
    // the status, the candidate's kind, a committed hit's integer fields, and the ray's flags
    let p = ray_query_program(
        "  %6: bool = rayquery.proceed q0
  %7: u32 = rayquery.state q0 committed.status
  %8: u32 = rayquery.state q0 candidate.type
  %9: u32 = rayquery.state q0 committed.instanceid
  %10: u32 = rayquery.state q0 committed.contribution
  %11: u32 = rayquery.state q0 ray.flags
  %12: u32 = rayquery.state q0 candidate.primitive
  rayquery commit q0
  output 0.x = %7
  output 0.y = %8
  output 0.z = %9
  output 0.w = %10
  output 1.x = %11
  output 1.y = %12
",
        &[(ScalarType::Uint, 4), (ScalarType::Uint, 2)],
    );
    let wgsl = written(&p, &Profile::NAGA);
    for line in [
        "enable wgpu_ray_query;",
        "@group(1) @binding(0) var t0: acceleration_structure;",
        "var rquery: ray_query;",
        // the query's declared flags join the ray's, and the ray is kept for the read of its flags
        "rquery_flags = (v0 | 32u);",
        "rayQueryInitialize(&rquery, t0, RayDesc((v0 | 32u), v1, v3, v5, v2, v4));",
        "let v6 = rayQueryProceed(&rquery);",
        "let v7 = rayQueryGetCommittedIntersection(&rquery).kind;",
        // a candidate's kind is 1 for a triangle and 3 for a procedural primitive, where the IR says 0 and 1
        "let v8 = select(1u, 0u, (rayQueryGetCandidateIntersection(&rquery).kind == 1u));",
        "let v9 = rayQueryGetCommittedIntersection(&rquery).instance_custom_data;",
        "let v10 = rayQueryGetCommittedIntersection(&rquery).sbt_record_offset;",
        "let v11 = rquery_flags;",
        "let v12 = rayQueryGetCandidateIntersection(&rquery).primitive_index;",
        "rayQueryConfirmIntersection(&rquery);",
    ] {
        assert!(wgsl.contains(line), "{line}\n{wgsl}");
    }
}

#[test]
fn a_hit_s_floats_booleans_and_matrices_are_the_intersection_s() {
    let p = ray_query_program(
        "  %6: bool = rayquery.proceed q0
  %7: f32 = rayquery.state q0 committed.t
  %8: f32 = rayquery.state q0 candidate.barycentrics.y
  %9: f32 = rayquery.state q0 committed.objecttoworld.r1c3
  %10: f32 = rayquery.state q0 candidate.worldtoobject.r2c0
  %11: f32 = rayquery.state q0 committed.objectorigin.y
  %12: f32 = rayquery.state q0 candidate.objectdirection.z
  %13: bool = rayquery.state q0 committed.frontface
  %14: f32 = rayquery.state q0 ray.origin.z
  %15: f32 = rayquery.state q0 ray.tmin
  %16: f32 = select %13, %7, %8
  output 0.x = %7
  output 0.y = %8
  output 0.z = %9
  output 0.w = %10
  output 1.x = %11
  output 1.y = %12
  output 1.z = %14
  output 1.w = %16
  output 2.x = %15
",
        &[
            (ScalarType::Float, 4),
            (ScalarType::Float, 4),
            (ScalarType::Float, 1),
        ],
    );
    let wgsl = written(&p, &Profile::NAGA);
    for line in [
        "let v7 = rayQueryGetCommittedIntersection(&rquery).t;",
        "let v8 = rayQueryGetCandidateIntersection(&rquery).barycentrics.y;",
        // a 3 by 4 matrix's row and column: the intersection's matrix is four columns of three rows
        "let v9 = rayQueryGetCommittedIntersection(&rquery).object_to_world[3][1];",
        "let v10 = rayQueryGetCandidateIntersection(&rquery).world_to_object[0][2];",
        // the kept ray in the hit's object space
        "let v11 = (rayQueryGetCommittedIntersection(&rquery).world_to_object * vec4<f32>(rquery_origin, 1f)).y;",
        "let v12 = (rayQueryGetCandidateIntersection(&rquery).world_to_object * vec4<f32>(rquery_dir, 0f)).z;",
        "let v13 = rayQueryGetCommittedIntersection(&rquery).front_face;",
        "let v14 = rquery_origin.z;",
        "let v15 = rquery_tmin;",
    ] {
        assert!(wgsl.contains(line), "{line}\n{wgsl}");
    }
}

#[test]
fn a_query_read_for_nothing_of_its_ray_keeps_no_ray() {
    let p = ray_query_program(
        "  %6: bool = rayquery.proceed q0
  %7: u32 = rayquery.state q0 committed.status
  output 0.x = %7
",
        &[(ScalarType::Uint, 1)],
    );
    let wgsl = written(&p, &Profile::NAGA);
    assert!(!wgsl.contains("rquery_"), "{wgsl}");
}

#[test]
fn a_candidate_s_opacity_is_refused_by_name() {
    let p = ray_query_program(
        "  %6: bool = rayquery.proceed q0
  %7: bool = rayquery.state q0 candidate.nonopaque
  %8: u32 = select %7, %0, %1
  output 0.x = %8
",
        &[(ScalarType::Uint, 1)],
    );
    let e = iridia_wgsl::wgsl(&p, &Profile::NAGA).unwrap_err().0;
    assert!(
        e.contains("candidate procedural primitive is not opaque has no WGSL form"),
        "{e}"
    );
}

/// A pixel program whose region `inside` (after the common values) starts query q0, round a loop `%8` rounds long.
fn started_in(inside: &str) -> String {
    format!(
        "iridia-ir 1
fn main {{
  query q0 flags 0x0
  %0: u32 = u32(2)
  %1: u32 = u32(255)
  %2: vec3<f32> = vec3<f32>(0x0, 0x0, 0x0)
  %3: f32 = f32(0x0)
  %4: vec3<f32> = vec3<f32>(0x0, 0x0, 0x3f800000)
  %5: f32 = f32(0x447a0000)
  %6: u32 = u32(0)
  %7: u32 = u32(1)
  %8: u32 = u32(2)
{inside}  return
}}
"
    )
}

#[test]
fn a_query_started_again_is_refused_by_name() {
    let start = "rayquery init q0 r0 flags %0 mask %1 origin %2 tmin %3 direction %4 tmax %5";
    // a second start of one query
    let twice = started_in(&format!("  {start}\n  {start}\n  output 0.x = %0\n"));
    // a start in a loop that goes round again
    let in_a_loop = started_in(&format!(
        "  loop '0 (%9: u32) = (%6) {{
    %10: bool = ucmp.ge %9, %8
    if %10 {{
      br '0 %9
    }} else {{
      yield
    }}
    {start}
    %11: u32 = iadd %9, %7
    continue '0 %11
  }} -> %12: u32
  output 0.x = %12
"
    ));
    for (name, text) in [("twice", twice), ("a loop", in_a_loop)] {
        let p = ray_query_text(&text, &[(ScalarType::Uint, 1)]);
        let e = iridia_wgsl::wgsl(&p, &Profile::NAGA).unwrap_err().0;
        assert!(
            e.starts_with("ray query q0 is started again: naga cannot restart one"),
            "{name}: {e}"
        );
    }
    // a loop that never goes round again is a block that leaves early: it starts the query once
    let once = started_in(&format!(
        "  loop '0 () = () {{
    {start}
    br '0
  }}
  output 0.x = %0
"
    ));
    written(&ray_query_text(&once, &[(ScalarType::Uint, 1)]), &Profile::NAGA);
}

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
        (Capability::TextureAtomic, C::TEXTURE_ATOMIC),
        (Capability::RayQuery, C::RAY_QUERY),
    ]
    .into_iter()
    .chain(
        [
            (
                ArrayClass::TexturesAndSamplers,
                C::TEXTURE_AND_SAMPLER_BINDING_ARRAY,
                C::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING,
            ),
            (
                ArrayClass::ConstantBuffers,
                C::BUFFER_BINDING_ARRAY,
                C::BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING,
            ),
            (
                ArrayClass::Buffers,
                C::STORAGE_BUFFER_BINDING_ARRAY,
                C::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING,
            ),
            (
                ArrayClass::StorageImages,
                C::STORAGE_TEXTURE_BINDING_ARRAY,
                C::STORAGE_TEXTURE_BINDING_ARRAY_NON_UNIFORM_INDEXING,
            ),
        ]
        .into_iter()
        .flat_map(|(class, uniform, non_uniform)| {
            [
                (Capability::BindingArray(class), uniform),
                (Capability::BindingArrayNonUniform(class), non_uniform),
            ]
        }),
    )
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

/// A resource at register 0 of its class.
fn bound(id: u32, kind: ResourceKind, count: u32, name: &str, class: D3dClass) -> Resource {
    Resource {
        id: ResourceId(id),
        kind,
        count,
        provenance: ResourceProvenance {
            name: name.into(),
            d3d: Some(D3dBinding {
                class,
                register: 0,
                space: 0,
            }),
            ..ResourceProvenance::default()
        },
    }
}

const F2D: ResourceKind = ResourceKind::Texture {
    dim: TextureDim::D2,
    arrayed: false,
    multisampled: false,
    sample: SampleType::Float,
};

#[test]
fn arrays_of_resources_are_binding_arrays_of_naga_s_dialect() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec2<f32> = %0.xy
  %2: f32 = %0.z
  %3: u32 = bitcast %2
  %4: vec4<f32> = sample r0@%3, r1@%3!, %1 level 0
  output 0.xyzw = %4
  return
}
",
        &[F4],
        &[F4],
        vec![
            bound(0, F2D, 4, "tex", D3dClass::ShaderResource),
            bound(
                1,
                ResourceKind::Sampler { comparison: false },
                Resource::UNBOUNDED,
                "smp",
                D3dClass::Sampler,
            ),
        ],
    );
    // a binding array of each class, and one indexed per invocation
    let needs: Vec<_> = iridia_ir::target::requirements(&p)
        .iter()
        .map(|r| r.capability)
        .collect();
    assert_eq!(
        needs,
        [
            Capability::BindingArray(ArrayClass::TexturesAndSamplers),
            Capability::BindingArrayNonUniform(ArrayClass::TexturesAndSamplers)
        ]
    );
    // WGSL has no arrays of resources
    let e = iridia_wgsl::wgsl(&p, &Profile::BROWSER).unwrap_err();
    assert!(
        e.0.starts_with("browser WGSL cannot carry this program: resource tex is an array of textures"),
        "{e}"
    );
    // naga's dialect binds each array at its first register, a binding array of its length (unsized for the unbounded
    // one), and picks the element the program names
    let wgsl = written(&p, &Profile::NAGA);
    assert!(wgsl.starts_with("enable wgpu_binding_array;"), "{wgsl}");
    assert!(
        wgsl.contains("var t0: binding_array<texture_2d<f32>, 4>;"),
        "{wgsl}"
    );
    assert!(wgsl.contains("var s0: binding_array<sampler>;"), "{wgsl}");
    assert!(wgsl.contains("textureSampleLevel(t0[v3], s0[v3], "), "{wgsl}");
}

#[test]
fn an_array_of_buffers_is_a_binding_array_of_structures_with_one_member() {
    let words = ResourceKind::Storage {
        target: StorageTarget::Buffer(iridia_ir::interface::BufferKind::Raw),
        access: Access::ReadWrite,
    };
    let constants = ResourceKind::ConstantBuffer {
        layout: iridia_ir::interface::BufferLayout {
            size: 32,
            constants: Vec::new(),
        },
    };
    let p = program(
        "iridia-ir 1
fn main {
  %0: u32 = input 0.x
  %1: u32 = buffer r0@%0[%0 * 4] + 0 oob.zero
  %2: u32 = query.elements r0@%0
  store buffer r0@%0[%0 * 4] + 0 = %2
  %3: u32 = buffer r1@%0 + 16 oob.zero
  %4: u32 = iadd %1, %3
  output 0.x = %4
  return
}
",
        &[(ScalarType::Uint, 1)],
        &[(ScalarType::Uint, 1)],
        vec![
            bound(0, words, 3, "words", D3dClass::UnorderedAccess),
            bound(1, constants, 2, "cb", D3dClass::ConstantBuffer),
        ],
    );
    let wgsl = written(&p, &Profile::NAGA);
    assert!(wgsl.contains("data: array<u32>,"), "{wgsl}");
    assert!(wgsl.contains("binding_array<u0_element, 3>"), "{wgsl}");
    assert!(wgsl.contains("arrayLength(&u0[v0].data)"), "{wgsl}");
    assert!(wgsl.contains("binding_array<cb0_element, 2>"), "{wgsl}");
}

#[test]
fn array_elements_share_one_packed_counter_buffer() {
    let p = program(
        "iridia-ir 1
fn main {
  %0: u32 = input 0.x
  %1: u32 = counter.increment r0@%0
  %2: u32 = counter.decrement r0@%0
  return
}
",
        &[(ScalarType::Uint, 1)],
        &[],
        vec![bound(
            0,
            ResourceKind::Storage {
                target: StorageTarget::Buffer(iridia_ir::interface::BufferKind::Structured { stride: 4 }),
                access: Access::ReadWrite,
            },
            4,
            "views",
            D3dClass::UnorderedAccess,
        )],
    );
    // one counter buffer of the resource, word n counting element n
    let wgsl = written(&p, &Profile::NAGA);
    assert!(wgsl.contains("u0_counter: array<atomic<u32>>;"), "{wgsl}");
    assert!(wgsl.contains("atomicAdd(&u0_counter[v0], 1u)"), "{wgsl}");
    assert!(wgsl.contains("atomicSub(&u0_counter[v0], 1u)"), "{wgsl}");
}

/// A storage image of `u32` or `i32` texels at register `u0`.
fn image(sample: SampleType) -> Resource {
    bound(
        0,
        ResourceKind::Storage {
            target: StorageTarget::Image {
                dim: TextureDim::D2,
                arrayed: false,
                sample,
            },
            access: Access::ReadWrite,
        },
        1,
        "img",
        D3dClass::UnorderedAccess,
    )
}

/// A pixel program that changes a texel of `img` by each of `ops` (`atomic.<op>`), reading the old value of none unless
/// `read` (then the last one's is an output). An atomic is of the texels' type.
fn texel_atomics(sample: SampleType, ops: &[&str], read: bool) -> Program {
    let (t, scalar) = if sample == SampleType::Sint {
        ("i32", ScalarType::Sint)
    } else {
        ("u32", ScalarType::Uint)
    };
    let mut text = format!(
        "iridia-ir 1
fn main {{
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<{t}> = bitcast %0
  %2: vec2<{t}> = %1.xy
  %3: {t} = %1.z
"
    );
    for (k, op) in ops.iter().enumerate() {
        text.push_str(&format!(
            "  %{}: {t} = atomic.{op} image r0, %2 value %3\n",
            k + 4
        ));
    }
    if read {
        text.push_str(&format!("  output 0.x = %{}\n", ops.len() + 3));
    }
    text.push_str("  return\n}\n");
    let outputs: &[(ScalarType, u8)] = if read { &[(scalar, 1)] } else { &[] };
    program(&text, &[F4], outputs, vec![image(sample)])
}

#[test]
fn a_texel_atomic_whose_old_value_is_not_read_is_naga_s_image_atomic() {
    let p = texel_atomics(SampleType::Uint, &["or", "sub", "umax", "xor"], false);
    let needs: Vec<_> = iridia_ir::target::requirements(&p)
        .iter()
        .map(|r| r.capability)
        .collect();
    assert_eq!(needs, [Capability::TextureAtomic]);
    // WGSL has no texel atomics
    let e = iridia_wgsl::wgsl(&p, &Profile::BROWSER).unwrap_err();
    assert!(
        e.0.starts_with("browser WGSL cannot carry this program: value %4 is an atomic of storage image img"),
        "{e}"
    );
    // one texel of an `atomic` image, inside it only; a subtraction is the addition of the negation
    let wgsl = written(&p, &Profile::NAGA);
    assert!(
        wgsl.contains("var u0: texture_storage_2d<r32uint, atomic>;"),
        "{wgsl}"
    );
    for call in [
        "textureAtomicOr(u0, ",
        "textureAtomicAdd(u0, ",
        "textureAtomicMax(u0, ",
        "textureAtomicXor(u0, ",
    ] {
        assert!(wgsl.contains(call), "{call}\n{wgsl}");
    }
    assert!(wgsl.contains("(0u - "), "{wgsl}");
    // a signed image's
    let p = texel_atomics(SampleType::Sint, &["smin", "add"], false);
    let wgsl = written(&p, &Profile::NAGA);
    assert!(
        wgsl.contains("var u0: texture_storage_2d<r32sint, atomic>;"),
        "{wgsl}"
    );
    assert!(wgsl.contains("textureAtomicMin(u0, "), "{wgsl}");
}

#[test]
fn a_texel_atomic_naga_s_image_atomic_cannot_carry_is_refused_by_name() {
    let refusal = |sample, ops: &[&str], read| {
        iridia_wgsl::wgsl(&texel_atomics(sample, ops, read), &Profile::NAGA)
            .unwrap_err()
            .0
    };
    // naga's image atomic gives no old value and takes no exchange
    let e = refusal(SampleType::Uint, &["add"], true);
    assert!(
        e.contains("a storage image's texel whose old value is read"),
        "{e}"
    );
    let e = refusal(SampleType::Uint, &["exchange"], false);
    assert!(e.contains("an exchange of a texel has no naga form"), "{e}");
    // a minimum is of the texels' signedness
    let e = refusal(SampleType::Uint, &["smin"], false);
    assert!(
        e.contains("a SMin of a storage image of Uint texels has no naga form"),
        "{e}"
    );
}

#[test]
fn a_sample_position_is_read_from_direct3d_s_standard_table() {
    let texture = bound(
        0,
        ResourceKind::Texture {
            dim: TextureDim::D2,
            arrayed: false,
            multisampled: true,
            sample: SampleType::Float,
        },
        1,
        "msaa",
        D3dClass::ShaderResource,
    );
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<u32> = bitcast %0
  %2: u32 = %1.x
  %3: vec2<f32> = query.sampleposition r0 index %2
  output 0.xy = %3
  return
}
",
        &[F4],
        &[(ScalarType::Float, 2)],
        vec![texture],
    );
    // a table of every standard count's positions, its rows from the texture's sample count; core WGSL
    let wgsl = written(&p, &Profile::BROWSER);
    assert!(
        wgsl.contains("var<private> sample_positions: array<vec2<f32>, 31> = array<vec2<f32>, 31>(vec2<f32>(0.0f, 0.0f), "),
        "{wgsl}"
    );
    // the 16-sample pattern's last position, (-7, -8) sixteenths
    assert!(wgsl.contains("vec2<f32>((-4.375e-1f), (-5e-1f)));"), "{wgsl}");
    assert!(wgsl.contains("textureNumSamples(t0)"), "{wgsl}");
}

#[test]
fn a_sample_of_an_integer_texture_is_refused_by_name_and_a_gather_is_written() {
    let refusal = "a filtered sample of an integer texture has no WGSL form";
    for (sample, scalar, ty) in [
        (SampleType::Uint, ScalarType::Uint, "u32"),
        (SampleType::Sint, ScalarType::Sint, "i32"),
    ] {
        let reading = |op: &str| {
            program(
                &format!(
                    "iridia-ir 1
fn main {{
  %0: vec4<f32> = input 0.xyzw
  %1: vec2<f32> = %0.xy
  %2: f32 = %0.z
  %3: vec2<f32> = %0.zw
  %4: vec4<{ty}> = {op}
  output 0.xyzw = %4
  return
}}
"
                ),
                &[F4],
                &[(scalar, 4)],
                vec![
                    bound(
                        0,
                        ResourceKind::Texture {
                            dim: TextureDim::D2,
                            arrayed: false,
                            multisampled: false,
                            sample,
                        },
                        1,
                        "image",
                        D3dClass::ShaderResource,
                    ),
                    bound(
                        1,
                        ResourceKind::Sampler { comparison: false },
                        1,
                        "smp",
                        D3dClass::Sampler,
                    ),
                ],
            )
        };
        for op in [
            "sample r0, r1, %1",
            "sample r0, r1, %1 level 0",
            "sample r0, r1, %1 level %2",
            "sample r0, r1, %1 bias %2",
            "sample r0, r1, %1 grad %1, %3",
        ] {
            for profile in [Profile::NAGA, Profile::BROWSER] {
                let e = iridia_wgsl::wgsl(&reading(op), &profile).unwrap_err();
                assert!(e.0.contains(refusal), "{op} of {ty}: {e}");
            }
        }
        // a gather of one component has a form, which naga reads back and validates
        let p = reading("gather.x r0, r1, %1");
        for profile in [Profile::NAGA, Profile::BROWSER] {
            assert!(written(&p, &profile).contains("textureGather("), "{ty}");
        }
    }
}
