//! Vertex inputs are located before interpolation; interpolation still identifies fragment inputs.
use iridia::flow::{Node, Values};
use iridia::{Interface, Label, Program, ScalarType, Stage, SystemValue, Value, Varying, VaryingProvenance};

fn varying(name: &str, scalar: ScalarType, width: u8, system: Option<SystemValue>) -> Varying {
    Varying {
        label: Label {
            name: name.into(),
            index: 0,
        },
        system,
        value: Value { scalar, width },
        interp: None,
        used: true,
        provenance: VaryingProvenance::default(),
    }
}

fn vertex() -> Program {
    Program {
        module: naga::front::wgsl::parse_str(
            "struct Out { @location(0) @interpolate(perspective) uv: vec2<f32>,
                @builtin(position) position: vec4<f32> }
            @vertex fn main(@location(0) uv: vec2<f32>, @builtin(vertex_index) id: u32) -> Out {
                return Out(uv, vec4<f32>(0.0, 0.0, 0.0, 1.0));
            }",
        )
        .expect("owned vertex fixture"),
        interface: Interface {
            stage: Stage::Vertex,
            inputs: vec![
                varying("uv", ScalarType::Float, 2, None),
                varying("vertex", ScalarType::Uint, 1, Some(SystemValue::VertexIndex)),
            ],
            outputs: vec![
                varying("uv", ScalarType::Float, 2, None),
                varying("position", ScalarType::Float, 4, Some(SystemValue::ClipPosition)),
            ],
            resources: vec![],
            facts: Default::default(),
        },
        stage: Stage::Vertex,
        entry: "main".into(),
        bindings: vec![],
        conventions: iridia::Conventions::D3D,
        modes: Default::default(),
        notes: vec![],
        provenance: iridia::Provenance {
            format: iridia::ShaderFormat::Glsl,
            target: "owned fixture".into(),
            lifter: "fixture".into(),
            lifter_version: 1,
            source_hash: None,
            instruction_count: None,
        },
    }
}

#[test]
fn vertex_inputs_match_with_or_without_wgsl_interpolation_defaults() {
    let mut program = vertex();
    assert!(matches!(
        program.module.entry_points[0].function.arguments[0].binding,
        Some(naga::Binding::Location {
            interpolation: Some(naga::Interpolation::Perspective),
            sampling: Some(naga::Sampling::Center),
            ..
        })
    ));
    let expected = iridia::entry_io(&program.interface)
        .expect("vertex interface")
        .arguments[0]
        .binding
        .clone();
    assert!(matches!(
        expected,
        naga::Binding::Location {
            interpolation: None,
            sampling: None,
            ..
        }
    ));
    let parsed = Values::of_program(&program).expect("WGSL vertex inputs match by location");
    for lane in 0..2 {
        assert_eq!(
            parsed.node(parsed.outputs()[&(0, lane)]),
            &Node::Input { input: 0, lane }
        );
    }
    program.module.entry_points[0].function.arguments[0].binding = Some(expected);
    let normalized = Values::of_program(&program).expect("stage-aware vertex input");
    for (&key, &id) in parsed.outputs() {
        assert_eq!(parsed.node(id), normalized.node(normalized.outputs()[&key]));
    }
}

#[test]
fn vertex_input_location_and_builtin_mismatches_are_still_refused() {
    let mut program = vertex();
    let Some(naga::Binding::Location { location, .. }) =
        &mut program.module.entry_points[0].function.arguments[0].binding
    else {
        panic!("fixture location");
    };
    *location = 1;
    let why = Values::of_program(&program).expect_err("a different location is not the same input");
    assert!(
        why.contains("entry argument 0: no interface input has its binding"),
        "{why}"
    );

    let mut program = vertex();
    program.module.entry_points[0].function.arguments[1].binding =
        Some(naga::Binding::BuiltIn(naga::BuiltIn::InstanceIndex));
    let why = Values::of_program(&program).expect_err("a different builtin is not the same input");
    assert!(
        why.contains("entry argument 1: no interface input has its binding"),
        "{why}"
    );
}

#[test]
fn fragment_input_interpolation_and_sampling_mismatches_are_still_refused() {
    let mut program = vertex();
    program.stage = Stage::Pixel;
    program.interface.stage = Stage::Pixel;
    program.interface.inputs.truncate(1);
    program.interface.outputs = vec![varying(
        "target",
        ScalarType::Float,
        4,
        Some(SystemValue::Target(0)),
    )];
    program.module = naga::front::wgsl::parse_str(
        "@fragment fn main(@location(0) @interpolate(perspective) uv: vec2<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(uv, 0.0, 1.0);
        }",
    )
    .expect("owned fragment fixture");
    Values::of_program(&program).expect("matching fragment interpolation");
    let mut linear = program.clone();
    let Some(naga::Binding::Location { interpolation, .. }) =
        &mut linear.module.entry_points[0].function.arguments[0].binding
    else {
        panic!("fixture location");
    };
    *interpolation = Some(naga::Interpolation::Linear);
    assert!(Values::of_program(&linear).is_err());
    let Some(naga::Binding::Location { sampling, .. }) =
        &mut program.module.entry_points[0].function.arguments[0].binding
    else {
        panic!("fixture location");
    };
    *sampling = Some(naga::Sampling::Centroid);
    assert!(Values::of_program(&program).is_err());
}
