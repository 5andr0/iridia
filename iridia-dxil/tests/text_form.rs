//! A program WGSL cannot express is refused by name by both writers, and the IR's text form shows what it holds.

#![allow(clippy::unwrap_used)]

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}

fn lifted(name: &str) -> iridia_ir::Program {
    let mut program = iridia_dxil::ir::lift(&fixture(name)).unwrap();
    iridia_passes::prepare(&mut program).unwrap();
    program
}

/// `image_atomic_results.dxil` reads the old value of texture atomics, which WGSL's texture atomics do not return.
#[test]
fn a_program_wgsl_cannot_write_is_refused_by_name_and_kept_as_ir_text() {
    let program = lifted("image_atomic_results.dxil");
    // a browser has no texture atomics at all; wgpu's naga has them, without the old value
    let browser = iridia_wgsl::wgsl(&program, &iridia_wgsl::Profile::BROWSER)
        .unwrap_err()
        .to_string();
    assert!(browser.contains("TextureAtomic"), "{browser}");
    let wgsl = iridia_wgsl::wgsl(&program, &iridia_wgsl::Profile::NAGA)
        .unwrap_err()
        .to_string();
    assert!(wgsl.contains("whose old value is read"), "{wgsl}");
    let naga = iridia_naga::wgsl(&program, &iridia_naga::Profile::NAGA)
        .unwrap_err()
        .to_string();
    assert!(naga.contains("whose old value is read"), "{naga}");
    let text = iridia_ir::text::print(&program.entry);
    assert!(text.contains("atomic"), "the text holds the atomics:\n{text}");
    assert_eq!(
        text,
        iridia_ir::text::print(&program.entry),
        "the same program, the same text"
    );
}

#[test]
fn a_program_wgsl_can_write_is_written_as_wgsl() {
    let program = lifted("pixel_ops.dxil");
    for text in [
        iridia_wgsl::wgsl(&program, &iridia_wgsl::Profile::BROWSER).unwrap(),
        iridia_naga::wgsl(&program, &iridia_naga::Profile::BROWSER).unwrap(),
    ] {
        assert!(text.contains("fn "), "WGSL functions\n{text}");
    }
}
