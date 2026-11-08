//! The public target contract, without a writer's types or its binding adapter.
#![allow(clippy::unwrap_used, reason = "interfaces built by the tests")]

use iridia_ir::interface::{
    Interface, InterpMode, InterpSampling, Interpolation, Label, ScalarType, Stage, StageFacts, SystemValue,
    Value, Varying, VaryingProvenance,
};
use iridia_ir::target::{Binding, BuiltIn, IoType, entry_io};
use proptest::prelude::*;

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

fn interface(stage: Stage, inputs: Vec<Varying>, outputs: Vec<Varying>) -> Interface {
    Interface {
        stage,
        inputs,
        outputs,
        resources: Vec::new(),
        facts: StageFacts::None,
    }
}

#[test]
fn pipeline_builtins_have_their_api_types_and_targets_keep_their_numbers() {
    let i = interface(
        Stage::Pixel,
        vec![
            varying("SV_Position", ScalarType::Float, 2, Some(SystemValue::FragCoord)),
            varying(
                "SV_IsFrontFace",
                ScalarType::Uint,
                1,
                Some(SystemValue::FrontFacing),
            ),
            varying(
                "SV_SampleIndex",
                ScalarType::Sint,
                1,
                Some(SystemValue::SampleIndex),
            ),
            varying("SV_GroupIndex", ScalarType::Sint, 1, Some(SystemValue::LaneIndex)),
            varying("SV_GroupSize", ScalarType::Sint, 1, Some(SystemValue::LaneCount)),
        ],
        vec![
            varying("SV_Target", ScalarType::Float, 1, Some(SystemValue::Target(3))),
            varying(
                "SV_Depth",
                ScalarType::Float,
                1,
                Some(SystemValue::DepthGreaterEqual),
            ),
        ],
    );
    let io = entry_io(&i).unwrap();
    let expected = [
        (BuiltIn::Position, ScalarType::Float, 4),
        (BuiltIn::FrontFacing, ScalarType::Bool, 1),
        (BuiltIn::SampleIndex, ScalarType::Uint, 1),
        (BuiltIn::SubgroupInvocationId, ScalarType::Uint, 1),
        (BuiltIn::SubgroupSize, ScalarType::Uint, 1),
    ];
    for (at, (item, (builtin, scalar, width))) in io.arguments.iter().zip(expected).enumerate() {
        assert_eq!(item.binding, Binding::BuiltIn(builtin));
        assert_eq!(item.ty, IoType::Value(Value { scalar, width }));
        assert_eq!(item.varyings, vec![at]);
    }
    assert_eq!(io.arguments.len(), expected.len());
    assert_eq!(
        io.results[0].binding,
        Binding::Location {
            location: 3,
            interpolation: None,
            sampling: InterpSampling::Center,
        }
    );
    assert_eq!(
        io.results[0].ty,
        IoType::Value(Value {
            scalar: ScalarType::Float,
            width: 4
        })
    );
    assert_eq!(io.results[1].binding, Binding::BuiltIn(BuiltIn::FragDepth));
}

#[test]
fn pixel_interpolation_keeps_sampling_for_floats_and_is_flat_for_integers() {
    let mut f = varying("UV", ScalarType::Float, 2, None);
    let mut n = varying("INDEX", ScalarType::Uint, 1, None);
    for v in [&mut f, &mut n] {
        v.interp = Some(Interpolation {
            mode: InterpMode::Linear,
            sampling: InterpSampling::Centroid,
        });
    }
    let io = entry_io(&interface(Stage::Pixel, vec![f, n], Vec::new())).unwrap();
    assert_eq!(
        io.arguments[0].binding,
        Binding::Location {
            location: 0,
            interpolation: Some(InterpMode::Linear),
            sampling: InterpSampling::Centroid,
        }
    );
    assert_eq!(
        io.arguments[1].binding,
        Binding::Location {
            location: 1,
            interpolation: Some(InterpMode::Flat),
            sampling: InterpSampling::Center,
        }
    );
}

#[test]
fn clip_distances_are_one_array_while_cull_distances_keep_a_location() {
    let mut second_clip = varying(
        "SV_ClipDistance",
        ScalarType::Float,
        1,
        Some(SystemValue::ClipDistance),
    );
    second_clip.label.index = 1;
    let io = entry_io(&interface(
        Stage::Vertex,
        Vec::new(),
        vec![
            varying(
                "SV_Position",
                ScalarType::Float,
                4,
                Some(SystemValue::ClipPosition),
            ),
            varying(
                "SV_ClipDistance",
                ScalarType::Float,
                2,
                Some(SystemValue::ClipDistance),
            ),
            varying("UV", ScalarType::Float, 2, None),
            second_clip,
            varying(
                "SV_CullDistance",
                ScalarType::Float,
                2,
                Some(SystemValue::CullDistance),
            ),
        ],
    ))
    .unwrap();
    assert_eq!(io.results.len(), 4);
    assert_eq!(io.results[1].name, "clip_distances");
    assert_eq!(io.results[1].ty, IoType::ClipDistances(3));
    assert_eq!(io.results[1].varyings, vec![1, 3]);
    assert_eq!(io.results[1].binding, Binding::BuiltIn(BuiltIn::ClipDistances));
    assert!(matches!(
        io.results[2].binding,
        Binding::Location { location: 0, .. }
    ));
    assert!(matches!(
        io.results[3].binding,
        Binding::Location { location: 1, .. }
    ));
    assert_eq!(io.results[3].varyings, vec![4]);
}

#[test]
fn an_unused_unbindable_signature_element_is_omitted_but_a_used_one_is_refused() {
    for system in [SystemValue::ClipDistance, SystemValue::CullDistance] {
        let mut v = varying("DISTANCE", ScalarType::Float, 1, Some(system));
        v.used = false;
        let mut i = interface(Stage::Pixel, vec![v], Vec::new());
        assert!(entry_io(&i).unwrap().arguments.is_empty());
        i.inputs[0].used = true;
        let reason = entry_io(&i).unwrap_err();
        assert!(
            reason.contains("DISTANCE0") && reason.contains("Pixel input"),
            "{reason}"
        );
    }
}

#[test]
fn mesh_and_amplification_entry_points_take_dispatch_ids_and_return_no_outputs() {
    for stage in [Stage::Mesh, Stage::Amplification] {
        let i = interface(
            stage,
            vec![varying(
                "SV_DispatchThreadID",
                ScalarType::Sint,
                1,
                Some(SystemValue::DispatchThreadId),
            )],
            vec![varying("UNWRITTEN", ScalarType::Float, 4, None)],
        );
        let before = i.clone();
        let io = entry_io(&i).unwrap();
        assert_eq!(i, before);
        assert!(io.results.is_empty());
        assert_eq!(
            io.arguments[0].binding,
            Binding::BuiltIn(BuiltIn::GlobalInvocationId)
        );
        assert_eq!(
            io.arguments[0].ty,
            IoType::Value(Value {
                scalar: ScalarType::Uint,
                width: 3
            })
        );
    }
}

proptest! {
    #[test]
    fn supplied_builtins_never_consume_user_locations(builtins in prop::collection::vec(any::<bool>(), 0..48)) {
        let inputs = builtins.iter().enumerate().map(|(at, builtin)| {
            varying(&format!("IN{at}_"), ScalarType::Float, 2,
                builtin.then_some(SystemValue::FragCoord))
        }).collect();
        let io = entry_io(&interface(Stage::Pixel, inputs, Vec::new())).unwrap();
        let locations: Vec<_> = io.arguments.iter().filter_map(|item| match item.binding {
            Binding::Location { location, .. } => Some(location),
            _ => None,
        }).collect();
        prop_assert_eq!(locations, (0..builtins.iter().filter(|b| !**b).count() as u32).collect::<Vec<_>>());
        for (at, item) in io.arguments.iter().enumerate() {
            prop_assert_eq!(&item.varyings, &vec![at]);
        }
    }
}
