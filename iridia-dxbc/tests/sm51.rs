#![allow(clippy::unwrap_used)]
//! fxc 10.0.26100.0 fixtures from dx 292b616; their reproducible HLSL is alongside the binaries.
use iridia_ir::interface::{D3dClass, Resource, ResourceKind, Stage};

const PIXEL: &[u8] = include_bytes!("fixtures/reflection_sm51.ps51.dxbc");
const COMPUTE: &[u8] = include_bytes!("fixtures/bindings_sm51.cs.dxbc");

#[test]
fn reflection_retains_register_spaces_arrays_and_unused_constant_layout() {
    let interface = iridia_dxbc::reflection(PIXEL).unwrap();
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
    let interface = iridia_dxbc::reflection(COMPUTE).unwrap();
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
fn lifting_resolves_sm51_declaration_ids_to_registers_and_spaces() {
    for bytes in [PIXEL, COMPUTE] {
        let decoded = dxbc_reader::decode(&dxbc_reader::code(bytes).unwrap()).unwrap();
        assert_eq!((decoded.major, decoded.minor), (5, 1));
        let program = iridia_dxbc::ir::lift(bytes).unwrap();
        iridia_ir::validate(&program).unwrap();
        let reflected = iridia_dxbc::resource_reflection(bytes).unwrap().unwrap();
        for resource in &program.interface.resources {
            let reference = reflected
                .iter()
                .find(|r| r.provenance.name == resource.provenance.name)
                .unwrap();
            assert_eq!(resource.provenance.d3d, reference.provenance.d3d);
            assert_eq!(resource.count, reference.count);
        }
    }
}

#[test]
fn stripped_sm51_keeps_declared_ranges_and_spaces() {
    for bytes in [PIXEL, COMPUTE] {
        let parts: Vec<_> = dxcontainer::chunks(bytes)
            .unwrap()
            .into_iter()
            .filter(|p| &p.kind != b"RDEF")
            .collect();
        let stripped = dxcontainer::serialize(&parts).unwrap();
        let program = iridia_dxbc::ir::lift(&stripped).unwrap();
        iridia_ir::validate(&program).unwrap();
        let original = iridia_dxbc::ir::lift(bytes).unwrap();
        let bindings = |p: &iridia_ir::Program| {
            p.interface
                .resources
                .iter()
                .map(|r| (r.provenance.d3d, r.count))
                .collect::<Vec<_>>()
        };
        assert_eq!(bindings(&program), bindings(&original));
    }
}

#[test]
fn reflection_cannot_substitute_another_declaration_id_or_space() {
    let rdef = dxbc_reader::rdef(PIXEL).unwrap().unwrap();
    let at = rdef.bindings[0].word * 4;
    for field in [8, 9] {
        // SM5.1's appended space and declaration-ID words.
        let mut parts = dxcontainer::chunks(PIXEL).unwrap();
        let part = parts.iter_mut().find(|p| &p.kind == b"RDEF").unwrap();
        let mut data = part.data.to_vec();
        data[at + field * 4..at + field * 4 + 4].copy_from_slice(&999u32.to_le_bytes());
        part.data = &data;
        let bytes = dxcontainer::serialize(&parts).unwrap();
        let error = iridia_dxbc::ir::lift(&bytes).unwrap_err();
        assert!(error.detail.contains("reflected declaration"), "{error}");
    }
}
