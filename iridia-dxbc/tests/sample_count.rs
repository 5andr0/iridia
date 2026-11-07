#![allow(clippy::unwrap_used)]
//! A program that reads the render target's sample count (`sample_count.hlsl`): the count is the pipeline's, its
//! rasterizer state, not the program's, so every lifter refuses the fxc and the dxc program by name, onto naga and onto
//! the IR alike. It stays refused: it is the refusal a caller's tests can rely on.

fn fixture(dir: &str, name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(dir)
            .join(name),
    )
    .unwrap()
}

const REFUSAL: &str = "the sample count of the rasterizer is the pipeline's, not the program's";

#[test]
fn the_render_targets_sample_count_is_refused_by_every_lifter() {
    let dxbc = fixture("tests/fixtures", "sample_count.dxbc");
    let e = iridia_dxbc::lift(&dxbc).unwrap_err().to_string();
    assert!(e.contains(REFUSAL), "{e}");
    let e = iridia_dxbc::ir::lift(&dxbc).unwrap_err().to_string();
    assert!(e.contains(REFUSAL), "{e}");

    let dxil = fixture("../iridia-dxil/tests/fixtures", "sample_count.dxil");
    let e = iridia_dxil::lift(&dxil).unwrap_err().to_string();
    assert!(e.contains(REFUSAL), "{e}");
    let e = iridia_dxil::ir::lift(&dxil).unwrap_err().to_string();
    assert!(e.contains(REFUSAL), "{e}");
}
