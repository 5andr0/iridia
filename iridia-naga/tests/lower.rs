//! Programs in IR text lowered to naga and written as WGSL: what each construct and rule becomes.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

use iridia_ir::{
    FloatEnv, Program, Provenance,
    interface::{
        Access, D3dBinding, D3dClass, Interface, Label, Resource, ResourceId, ResourceKind,
        ResourceProvenance, SampleType, ScalarType, ShaderFormat, Stage, StageFacts, StateParam,
        StorageTarget, SystemValue, TextureDim, Value, Varying, VaryingProvenance,
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

#[test]
fn target_checks_visit_placed_operations_and_not_unused_value_table_entries() {
    let mut p = program(
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
    let profile = iridia_naga::Profile::NAGA;
    let original = iridia_naga::wgsl(&p, &profile).unwrap();
    let v = iridia_ir::ValueId(u32::try_from(p.entry.values.len()).unwrap());
    p.entry.values.push(iridia_ir::ValueDef {
        ty: iridia_ir::Type {
            scalar: iridia_ir::Scalar::F16,
            lanes: 1,
        },
        op: iridia_ir::Op::Literal([0; 4]),
        origin: None,
        precision: iridia_ir::Precision::Full,
    });
    iridia_ir::validate(&p).unwrap();
    assert_eq!(iridia_naga::wgsl(&p, &profile).unwrap(), original);
    let iridia_ir::Body::Regions(body) = &mut p.entry.body else {
        panic!("expected regions")
    };
    body.nodes.push(iridia_ir::Node::Value(v));
    iridia_ir::validate(&p).unwrap();
    assert_eq!(
        iridia_naga::wgsl(&p, &profile).unwrap_err().0,
        "f16 literals are not lowered yet"
    );
}

#[test]
fn masked_loads_and_stores_lower_to_swizzles_and_merges() {
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
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    let body = "    let _e1 = TEXCOORD0_.yw;
    let _e7 = vec4<f32>(_e1.x, vec4<f32>().y, _e1.y, vec4<f32>().w);
    let _e13 = vec4<f32>(_e7.x, _e7.y, _e7.z, TEXCOORD0_.x);
    return Out(vec4<f32>(TEXCOORD0_.x, _e13.y, _e13.z, _e13.w));
";
    assert!(wgsl.contains(body), "{wgsl}");
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
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    assert!(
        wgsl.contains("select((TEXCOORD0_ * TEXCOORD1_), _e4, ((TEXCOORD0_ == _e4) | (TEXCOORD1_ == _e4)))"),
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
    let e = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap_err();
    assert!(e.0.contains("specialise"), "{e}");
}

#[test]
fn a_lane_used_twice_is_computed_once() {
    // %2 is a lane of a sum nothing else uses: naga gives an access no `let` of its own, so unnamed the sum would be
    // written at both uses
    let p = program(
        "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<f32> = fadd %0, %0
  %2: f32 = %1.x
  %3: vec2<f32> = compose %2, %2
  output 0.xy = %3
  return
}
",
        &[F4],
        &[(ScalarType::Float, 2)],
        Vec::new(),
    );
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    assert_eq!(wgsl.matches(" + ").count(), 1, "{wgsl}");
    assert!(wgsl.contains("let v2_ = (TEXCOORD0_ + TEXCOORD0_).x;"), "{wgsl}");
}

#[test]
fn a_capability_the_profile_lacks_is_refused_before_lowering() {
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
    let needs = iridia_naga::requirements(&p);
    assert_eq!(
        needs.iter().map(|r| r.capability).collect::<Vec<_>>(),
        [
            naga::valid::Capabilities::FLOAT64,
            naga::valid::Capabilities::SHADER_INT64
        ],
        "{needs:?}"
    );
    // browser WGSL has no f64: the preflight names the value, before any naga module is built
    let e = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER).unwrap_err();
    assert!(
        e.0.starts_with("browser WGSL cannot carry this program: value %1 is f64"),
        "{e}"
    );
    // naga's dialect has it, and naga's validator, run with the profile's capabilities alone, agrees
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    assert!(wgsl.contains("f64"), "{wgsl}");
    // a module that needs more than its profile allows is rejected by naga's validator itself
    let plan = iridia_naga::BindingPlan::d3d(&p.interface).unwrap();
    let module = iridia_naga::module(&p, &plan).unwrap();
    assert!(iridia_naga::validate(&module, &iridia_naga::Profile::BROWSER).is_err());
}

#[test]
fn deep_single_use_expressions_parse_in_both_profiles() {
    let p = program(
        include_str!("fixtures/deep_single_use.ir"),
        &[(ScalarType::Float, 1)],
        &[(ScalarType::Float, 1)],
        Vec::new(),
    );
    for profile in [iridia_naga::Profile::NAGA, iridia_naga::Profile::BROWSER] {
        let text = iridia_naga::wgsl(&p, &profile).unwrap();
        let module =
            naga::front::wgsl::parse_str(&text).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        iridia_naga::validate(&module, &profile).unwrap();
    }
}

#[test]
fn generated_remainder_nan_and_literal_payloads_are_runtime_bitcasts() {
    let p = program(
        "iridia-ir 1\nfn main {\n  %0: f32 = input 0.x\n  %1: f32 = input 0.y\n  %2: f32 = frem %0, %1\n  %3: f32 = f32(0xffc12345)\n  output 0.x = %2\n  output 0.y = %3\n  return\n}\n",
        &[F4],
        &[F4],
        Vec::new(),
    );
    for profile in [iridia_naga::Profile::NAGA, iridia_naga::Profile::BROWSER] {
        let text = iridia_naga::wgsl(&p, &profile).unwrap();
        // The helper's specified quiet NaN and the IR literal's sign/payload keep their words. A named
        // runtime word must stand between each literal and its float bitcast.
        for bits in [0x7fc0_0000u32, 0xffc1_2345] {
            assert!(text.contains(&format!("{bits}u")), "{text}");
            assert!(!text.contains(&format!("bitcast<f32>({bits}u)")), "{text}");
        }
        let module = naga::front::wgsl::parse_str(&text).unwrap();
        iridia_naga::validate(&module, &profile).unwrap();
    }
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
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER).unwrap();
    // its shortest decimal, 3.4028235e38, lies above it: Tint refuses it as out of range (WGSL rounds it back)
    assert!(!wgsl.contains("34028235"), "{wgsl}");
    // the float below it has a shortest decimal below it, written as ever
    assert!(
        wgsl.contains("340282330000000000000000000000000000000f"),
        "{wgsl}"
    );
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    // Runtime bindings may sit between the word and its bitcast. Check the bits
    // in the parsed expression graph instead of requiring an inline operand.
    let expressions = &module.entry_points[0].function.expressions;
    fn word(expressions: &naga::Arena<naga::Expression>, h: naga::Handle<naga::Expression>) -> Option<u32> {
        match expressions[h] {
            naga::Expression::Literal(naga::Literal::U32(bits)) => Some(bits),
            naga::Expression::As {
                expr,
                kind: naga::ScalarKind::Uint,
                convert: Some(4),
            } => word(expressions, expr),
            _ => None,
        }
    }
    let bitcasts: Vec<_> = expressions
        .iter()
        .filter_map(|(_, e)| match *e {
            naga::Expression::As {
                expr,
                kind: naga::ScalarKind::Float,
                convert: None,
            } => word(expressions, expr),
            _ => None,
        })
        .collect();
    assert!(bitcasts.contains(&0x7f7fffff), "{wgsl}");
    assert!(bitcasts.contains(&0xff7fffff), "{wgsl}");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}\n{wgsl}"));
}

/// A pixel program that samples the 1D texture at `r0` through the sampler at `r1`, at the coordinate x of its input,
/// `tail` being the sample's keywords (the level, the offset).
fn sampled_1d(tail: &str) -> Program {
    let bound = |id: u32, kind: ResourceKind, name: &str, class: D3dClass| Resource {
        id: ResourceId(id),
        kind,
        count: 1,
        provenance: ResourceProvenance {
            name: name.into(),
            d3d: Some(D3dBinding {
                class,
                register: 0,
                space: 0,
            }),
            ..ResourceProvenance::default()
        },
    };
    program(
        &format!(
            "iridia-ir 1
fn main {{
  %0: vec4<f32> = input 0.xyzw
  %1: f32 = %0.x
  %2: f32 = %0.y
  %3: vec4<f32> = sample r0, r1, %1 {tail}
  output 0.xyzw = %3
  return
}}
"
        ),
        &[F4],
        &[F4],
        vec![
            bound(
                0,
                ResourceKind::Texture {
                    dim: TextureDim::D1,
                    arrayed: false,
                    multisampled: false,
                    sample: SampleType::Float,
                },
                "tex",
                D3dClass::ShaderResource,
            ),
            bound(
                1,
                ResourceKind::Sampler { comparison: false },
                "smp",
                D3dClass::Sampler,
            ),
        ],
    )
}

#[test]
fn a_1d_texture_is_sampled_at_a_level_the_program_states() {
    // WGSL has textureSampleLevel for texture_1d: at level zero and at a level the program computes
    for (tail, call) in [
        ("", "textureSample(t0_, s0_, "),
        ("level 0", "textureSampleLevel(t0_, s0_, "),
        ("level %2", "textureSampleLevel(t0_, s0_, "),
    ] {
        for profile in [iridia_naga::Profile::NAGA, iridia_naga::Profile::BROWSER] {
            let wgsl =
                iridia_naga::wgsl(&sampled_1d(tail), &profile).unwrap_or_else(|e| panic!("{tail}: {e}"));
            assert!(wgsl.contains(call), "{tail}\n{wgsl}");
            let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
            iridia_naga::validate(&module, &profile).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
        }
    }
}

#[test]
fn a_1d_texture_is_not_sampled_at_a_bias_derivatives_or_an_offset_with_a_level() {
    for (tail, at) in [
        ("bias %2", "a biased level"),
        ("grad %2, %2", "explicit derivatives"),
        (
            "level 0 offset (1,0,0)",
            "an explicit level with an offset or a comparison",
        ),
    ] {
        let e = iridia_naga::wgsl(&sampled_1d(tail), &iridia_naga::Profile::NAGA)
            .unwrap_err()
            .0;
        assert_eq!(e, format!("a sample of a 1D texture at {at} has no WGSL form"));
    }
}

#[test]
fn arrays_of_resources_are_binding_arrays_of_naga_s_dialect() {
    let bound = |id: u32, kind: ResourceKind, count: u32, name: &str, class: D3dClass| Resource {
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
    };
    let textures = ResourceKind::Texture {
        dim: TextureDim::D2,
        arrayed: false,
        multisampled: false,
        sample: SampleType::Float,
    };
    let samplers = ResourceKind::Sampler { comparison: false };
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
            bound(0, textures, 4, "tex", D3dClass::ShaderResource),
            bound(1, samplers, Resource::UNBOUNDED, "smp", D3dClass::Sampler),
        ],
    );
    // a binding array of each class, and one indexed per invocation
    let needs: Vec<_> = iridia_naga::requirements(&p)
        .iter()
        .map(|r| r.capability)
        .collect();
    assert_eq!(
        needs,
        [
            naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY,
            naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING
        ]
    );
    // WGSL has no arrays of resources
    let e = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER).unwrap_err();
    assert!(
        e.0.starts_with("browser WGSL cannot carry this program: resource tex is an array of textures"),
        "{e}"
    );
    // naga's dialect binds each array at its first register, a binding array of its length (unsized for the
    // unbounded one), and picks the element the program names
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    assert!(wgsl.starts_with("enable wgpu_binding_array;"), "{wgsl}");
    assert!(
        wgsl.contains("@group(1) @binding(0) \nvar t0_: binding_array<texture_2d<f32>, 4>;"),
        "{wgsl}"
    );
    assert!(
        wgsl.contains("@group(2) @binding(0) \nvar s0_: binding_array<sampler>;"),
        "{wgsl}"
    );
    assert!(wgsl.contains("textureSampleLevel(t0_[_e3], s0_[_e3], "), "{wgsl}");
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    iridia_naga::validate(&module, &iridia_naga::Profile::NAGA).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
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
        vec![Resource {
            id: ResourceId(0),
            kind: ResourceKind::Storage {
                target: StorageTarget::Buffer(iridia_ir::interface::BufferKind::Structured { stride: 4 }),
                access: Access::ReadWrite,
            },
            count: 4,
            provenance: ResourceProvenance {
                name: "views".into(),
                d3d: Some(D3dBinding {
                    class: D3dClass::UnorderedAccess,
                    register: 0,
                    space: 0,
                }),
                ..ResourceProvenance::default()
            },
        }],
    );
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap();
    let counter = module
        .global_variables
        .iter()
        .find(|(_, g)| g.name.as_deref() == Some("u0_counter"))
        .unwrap()
        .1;
    assert!(
        matches!(module.types[counter.ty].inner, naga::TypeInner::Array { .. }),
        "{wgsl}"
    );
    assert!(wgsl.contains("u0_counter[TEXCOORD0_]"), "{wgsl}");
    iridia_naga::validate(&module, &iridia_naga::Profile::NAGA).unwrap();
    let mut plan = iridia_naga::BindingPlan::d3d(&p.interface).unwrap();
    plan.counters.clear();
    assert_eq!(
        iridia_naga::module(&p, &plan).unwrap_err().0,
        "the counter of resource 0 has no binding in the plan"
    );
}

/// A storage image of `u32` or `i32` texels at register `u0`.
fn image(sample: SampleType) -> Resource {
    Resource {
        id: ResourceId(0),
        kind: ResourceKind::Storage {
            target: StorageTarget::Image {
                dim: TextureDim::D2,
                arrayed: false,
                sample,
            },
            access: Access::ReadWrite,
        },
        count: 1,
        provenance: ResourceProvenance {
            name: "img".into(),
            d3d: Some(D3dBinding {
                class: D3dClass::UnorderedAccess,
                register: 0,
                space: 0,
            }),
            ..ResourceProvenance::default()
        },
    }
}

/// A pixel program that changes a texel of `img` by each of `ops` (`atomic.<op>`), reading the old value of none
/// unless `read` (then the last one's is an output). An atomic is of the texels' type.
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
    let needs: Vec<_> = iridia_naga::requirements(&p)
        .iter()
        .map(|r| r.capability)
        .collect();
    assert_eq!(needs, [naga::valid::Capabilities::TEXTURE_ATOMIC]);
    // WGSL has no texel atomics
    let e = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER).unwrap_err();
    assert!(
        e.0.starts_with("browser WGSL cannot carry this program: value %4 is an atomic of storage image img"),
        "{e}"
    );
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    // one texel of an `atomic` image, inside it only; a subtraction is the addition of the negation
    assert!(
        wgsl.contains("var u0_: texture_storage_2d<r32uint,atomic>;"),
        "{wgsl}"
    );
    for call in [
        "textureAtomicOr(u0_, ",
        "textureAtomicAdd(u0_, ",
        "textureAtomicMax(u0_, ",
        "textureAtomicXor(u0_, ",
    ] {
        assert!(wgsl.contains(call), "{call}\n{wgsl}");
    }
    assert!(wgsl.contains("(0u - "), "{wgsl}");
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    iridia_naga::validate(&module, &iridia_naga::Profile::NAGA).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    // a signed image's
    let p = texel_atomics(SampleType::Sint, &["smin", "add"], false);
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    assert!(
        wgsl.contains("var u0_: texture_storage_2d<r32sint,atomic>;"),
        "{wgsl}"
    );
    assert!(wgsl.contains("textureAtomicMin(u0_, "), "{wgsl}");
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    iridia_naga::validate(&module, &iridia_naga::Profile::NAGA).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
}

#[test]
fn a_texel_atomic_naga_s_image_atomic_cannot_carry_is_refused_by_name() {
    let refusal = |sample, ops: &[&str], read| {
        iridia_naga::wgsl(&texel_atomics(sample, ops, read), &iridia_naga::Profile::NAGA)
            .unwrap_err()
            .0
    };
    // naga's image atomic gives no old value and takes no exchange
    assert!(
        refusal(SampleType::Uint, &["add"], true).contains("a texel atomic whose old value is read"),
        "{}",
        refusal(SampleType::Uint, &["add"], true)
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
    let texture = Resource {
        id: ResourceId(0),
        kind: ResourceKind::Texture {
            dim: TextureDim::D2,
            arrayed: false,
            multisampled: true,
            sample: SampleType::Float,
        },
        count: 1,
        provenance: ResourceProvenance {
            name: "msaa".into(),
            d3d: Some(D3dBinding {
                class: D3dClass::ShaderResource,
                register: 0,
                space: 0,
            }),
            ..ResourceProvenance::default()
        },
    };
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
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER).unwrap();
    assert!(
        wgsl.contains(
            "var<private> sample_positions: array<vec2<f32>, 31> = array<vec2<f32>, 31>(vec2<f32>(0f, 0f), "
        ),
        "{wgsl}"
    );
    // the 16-sample pattern's last position, (-7, -8) sixteenths
    assert!(wgsl.contains("vec2<f32>(-0.4375f, -0.5f))"), "{wgsl}");
    assert!(wgsl.contains("textureNumSamples(t0_)"), "{wgsl}");
    let module = naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
    iridia_naga::validate(&module, &iridia_naga::Profile::BROWSER).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
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
    let subgroups = naga::valid::Capabilities::SUBGROUP;
    // a browser requires the directive before a subgroup operation
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER.with(subgroups)).unwrap();
    assert!(wgsl.starts_with("enable subgroups;\n"), "{wgsl}");
    assert!(wgsl.contains("subgroupAdd("), "{wgsl}");
    assert!(wgsl.contains("diagnostic(off, derivative_uniformity);"), "{wgsl}");
    assert!(!wgsl.contains("diagnostic(off, subgroup_uniformity);"), "{wgsl}");
    let language_profile = iridia_naga::Profile::BROWSER
        .with_language_features(iridia_naga::WgslLanguageFeatures {
            subgroup_uniformity: true,
        })
        .with(subgroups);
    let filtered = iridia_naga::wgsl(&p, &language_profile).unwrap();
    assert!(
        filtered.contains("diagnostic(off, subgroup_uniformity);"),
        "{filtered}"
    );
    // Naga 30 accepts the unknown diagnostic rule; only the already-unsupported enable directive is adapted.
    let body = filtered.strip_prefix("enable subgroups;\n").unwrap();
    let module = naga::front::wgsl::parse_str(body).unwrap();
    iridia_naga::validate(&module, &language_profile).unwrap();
    // naga's reader refuses it, and naga's dialect leaves it out
    assert!(naga::front::wgsl::parse_str(&wgsl).is_err());
    let wgsl = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap();
    assert!(!wgsl.contains("enable subgroups"), "{wgsl}");
    assert!(!wgsl.contains("diagnostic("), "{wgsl}");
    naga::front::wgsl::parse_str(&wgsl).unwrap_or_else(|e| panic!("{e}\n{wgsl}"));
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
    let wgsl = iridia_naga::wgsl(&plain, &iridia_naga::Profile::BROWSER.with(subgroups)).unwrap();
    assert!(!wgsl.contains("enable"), "{wgsl}");
    assert!(wgsl.contains("diagnostic(off, derivative_uniformity);"), "{wgsl}");
    let plain = iridia_naga::wgsl(&plain, &language_profile).unwrap();
    assert!(!plain.contains("subgroup_uniformity"), "{plain}");
}

/// A texture of `sample` and a sampler, for the programs that read an integer texture.
fn image_and_sampler(sample: SampleType) -> Vec<Resource> {
    let bound = |id: u32, kind: ResourceKind, name: &str, class: D3dClass| Resource {
        id: ResourceId(id),
        kind,
        count: 1,
        provenance: ResourceProvenance {
            name: name.into(),
            d3d: Some(D3dBinding {
                class,
                register: 0,
                space: 0,
            }),
            ..ResourceProvenance::default()
        },
    };
    vec![
        bound(
            0,
            ResourceKind::Texture {
                dim: TextureDim::D2,
                arrayed: false,
                multisampled: false,
                sample,
            },
            "image",
            D3dClass::ShaderResource,
        ),
        bound(
            1,
            ResourceKind::Sampler { comparison: false },
            "smp",
            D3dClass::Sampler,
        ),
    ]
}

#[test]
fn a_sample_of_an_integer_texture_is_refused_by_name_and_a_load_or_gather_is_written() {
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
                image_and_sampler(sample),
            )
        };
        // every kind of filtered sample: the implicit level, the level zero, a level, a bias, gradients
        for op in [
            "sample r0, r1, %1",
            "sample r0, r1, %1 level 0",
            "sample r0, r1, %1 level %2",
            "sample r0, r1, %1 bias %2",
            "sample r0, r1, %1 grad %1, %3",
        ] {
            for profile in [iridia_naga::Profile::NAGA, iridia_naga::Profile::BROWSER] {
                let e = iridia_naga::wgsl(&reading(op), &profile).unwrap_err();
                assert!(e.0.contains(refusal), "{op} of {ty}: {e}");
            }
        }
        // a gather of one component has a form, and the text is valid
        let p = reading("gather.x r0, r1, %1");
        for profile in [iridia_naga::Profile::NAGA, iridia_naga::Profile::BROWSER] {
            let wgsl = iridia_naga::wgsl(&p, &profile).unwrap_or_else(|e| panic!("{ty}: {e}"));
            assert!(wgsl.contains("textureGather("), "{wgsl}");
        }
    }
}

/// A pixel program that traces a ray query of the acceleration structure at `t0` and writes `reads`, the text's
/// lines that read the query's state into `%7` and `%8`.
fn ray_query_program(reads: &str) -> Program {
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
    let text = format!(
        "iridia-ir 1
fn main {{
  query q0 flags 0x0
  %0: u32 = u32(0)
  %1: u32 = u32(255)
  %2: vec3<f32> = vec3<f32>(0x0, 0x0, 0x0)
  %3: f32 = f32(0x0)
  %4: vec3<f32> = vec3<f32>(0x0, 0x0, 0x3f800000)
  %5: f32 = f32(0x447a0000)
  rayquery init q0 r0 flags %0 mask %1 origin %2 tmin %3 direction %4 tmax %5
  %6: bool = rayquery.proceed q0
{reads}  output 0.x = %7
  output 0.y = %8
  return
}}
"
    );
    program(&text, &[], &[(ScalarType::Float, 2)], vec![scene])
}

#[test]
fn a_ray_query_lowers_to_a_module_naga_validates_and_its_wgsl_writer_refuses_by_name() {
    // the ray in object space is the hit's world-to-object transform of the kept ray, which naga's intersection has
    let p = ray_query_program(
        "  %7: f32 = rayquery.state q0 candidate.objectorigin.x
  %8: f32 = rayquery.state q0 committed.objectdirection.z
",
    );
    let plan = iridia_naga::BindingPlan::d3d(&p.interface).unwrap();
    let module = iridia_naga::module(&p, &plan).unwrap();
    iridia_naga::validate(&module, &iridia_naga::Profile::NAGA).unwrap();
    // naga's validator holds the module to the ray query capability, which browser WGSL lacks: the preflight says so
    assert!(iridia_naga::validate(&module, &iridia_naga::Profile::BROWSER).is_err());
    let e = iridia_naga::wgsl(&p, &iridia_naga::Profile::BROWSER).unwrap_err();
    assert!(e.0.contains("ray query q0"), "{e}");
    let e = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap_err();
    assert!(
        e.0.starts_with("naga's WGSL writer has no ray queries: iridia-wgsl writes them"),
        "{e}"
    );

    // whether a candidate procedural primitive is opaque naga's intersection does not say
    let p = ray_query_program(
        "  %7: f32 = rayquery.state q0 committed.t
  %9: bool = rayquery.state q0 candidate.nonopaque
  %8: f32 = select %9, %7, %3
",
    );
    let plan = iridia_naga::BindingPlan::d3d(&p.interface).unwrap();
    let e = iridia_naga::module(&p, &plan).unwrap_err();
    assert!(
        e.0.contains("candidate procedural primitive is opaque is not lowered"),
        "{e}"
    );
}
