#![allow(clippy::unwrap_used)]
//! fxc 10.0.26100.0 fixtures from dx 292b616; their reproducible HLSL is alongside the binaries.
use iridia::{D3dClass, Resource, ResourceKind, Stage};

const PIXEL: &[u8] = include_bytes!("fixtures/reflection_sm51.ps51.dxbc");
const COMPUTE: &[u8] = include_bytes!("fixtures/bindings_sm51.cs.dxbc");

#[test]
fn reflection_retains_register_spaces_arrays_and_unused_constant_layout() {
    let interface = iridia_dxbc::resource_reflection(PIXEL).unwrap().unwrap();
    assert_eq!(interface.stage, Stage::Pixel);
    interface.verify().unwrap();
    let binding = |name: &str| {
        let resource = interface
            .resources
            .iter()
            .find(|r| r.provenance.name == name)
            .unwrap();
        (resource.provenance.d3d.unwrap(), resource.count)
    };
    for (name, class, register, space, count) in [
        ("Params", D3dClass::ConstantBuffer, 1, 2, 1),
        ("albedo", D3dClass::ShaderResource, 3, 1, 1),
        ("detail", D3dClass::ShaderResource, 4, 1, 4),
        ("linear_wrap", D3dClass::Sampler, 2, 3, 1),
    ] {
        let (got, n) = binding(name);
        assert_eq!(
            (got.class, got.register, got.space, n),
            (class, register, space, count),
            "{name}"
        );
    }
    let params = interface
        .resources
        .iter()
        .find(|r| r.provenance.name == "Params")
        .unwrap();
    let ResourceKind::ConstantBuffer { layout } = &params.kind else {
        panic!("Params is a constant buffer")
    };
    let fields: Vec<_> = layout
        .constants
        .iter()
        .map(|c| (c.name.as_str(), c.offset))
        .collect();
    assert_eq!(fields, [("tint", 0), ("unused_scale", 16), ("layer", 32)]);
}

#[test]
fn reflection_retains_constant_and_uav_arrays_and_an_unbounded_srv_range() {
    let interface = iridia_dxbc::resource_reflection(COMPUTE).unwrap().unwrap();
    assert_eq!(interface.stage, Stage::Compute);
    interface.verify().unwrap();
    for (name, register, space, count) in [
        ("params", 0, 4, 2),
        ("Frame", 5, 0, 1),
        ("textures", 0, 1, Resource::UNBOUNDED),
        ("shadow", 2, 2, 1),
        ("raw", 3, 2, 1),
        ("lights", 4, 2, 1),
        ("indices", 5, 2, 1),
        ("output", 0, 3, 1),
        ("counters", 1, 3, 1),
        ("lights_out", 2, 3, 3),
    ] {
        let resource = interface
            .resources
            .iter()
            .find(|r| r.provenance.name == name)
            .unwrap();
        let binding = resource.provenance.d3d.as_ref().unwrap();
        assert_eq!(
            (binding.register, binding.space, resource.count),
            (register, space, count),
            "{name}"
        );
    }
}

#[test]
fn lifting_refuses_sm51_before_a_declaration_id_can_be_mistaken_for_a_register() {
    for bytes in [PIXEL, COMPUTE] {
        let decoded = dxbc_reader::decode(&dxbc_reader::code(bytes).unwrap()).unwrap();
        assert_eq!((decoded.major, decoded.minor), (5, 1));
        let refusal = iridia_dxbc::lift(bytes).unwrap_err();
        assert_eq!(
            refusal.to_string(),
            "DXBC at word 0: SM5.1 descriptor ranges and register spaces are not lifted"
        );
    }
}
