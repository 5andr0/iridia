#![allow(clippy::unwrap_used)]
//! A comparing gather of a depth texture's green channel (`gather_cmp_green.hlsl`, after vkd3d's gather-cmp.shader_test):
//! WGSL gathers a depth texture's one lane, so both lifters refuse the fxc and the dxc program by name, and the IR lift
//! takes it and its lowering to WGSL refuses it in the same words.

fn fixture(dir: &str, name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(dir)
            .join(name),
    )
    .unwrap()
}

const REFUSAL: &str = "a gather of lane 1 of a depth texture";

fn lowered(p: &iridia_ir::Program) -> String {
    iridia_naga::wgsl(p, &iridia_naga::Profile::NAGA).unwrap_err().0
}

#[test]
fn a_gather_of_a_depth_textures_green_channel_is_refused_by_name() {
    let dxbc = fixture("tests/fixtures", "gather_cmp_green.dxbc");
    let e = iridia_dxbc::lift(&dxbc).unwrap_err().to_string();
    assert!(e.contains(REFUSAL), "{e}");
    let e = lowered(&iridia_dxbc::ir::lift(&dxbc).unwrap());
    assert!(e.contains(REFUSAL), "{e}");

    let dxil = fixture("../iridia-dxil/tests/fixtures", "gather_cmp_green.dxil");
    let e = iridia_dxil::lift(&dxil).unwrap_err().to_string();
    assert!(e.contains(REFUSAL), "{e}");
    let mut p = iridia_dxil::ir::lift(&dxil).unwrap();
    iridia_passes::structurize(&mut p.entry).unwrap();
    let e = lowered(&p);
    assert!(e.contains(REFUSAL), "{e}");
}
