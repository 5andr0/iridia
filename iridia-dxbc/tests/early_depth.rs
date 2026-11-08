#![allow(clippy::unwrap_used)]
//! The compiled early-depth attribute is an IR fact; the same shader without it is the control.
use iridia_ir::interface::{Stage, StageFacts};

#[test]
fn early_depth_is_a_pixel_stage_fact() {
    let early = iridia_dxbc::ir::lift(include_bytes!("fixtures/early_depth.dxbc")).unwrap();
    assert_eq!(early.interface.facts, StageFacts::EarlyDepthStencil);
    iridia_ir::validate(&early).unwrap();
    let late = iridia_dxbc::ir::lift(include_bytes!("fixtures/late_depth.dxbc")).unwrap();
    assert_eq!(late.interface.facts, StageFacts::None);
    iridia_ir::validate(&late).unwrap();
    let mut wrong_stage = early.interface;
    wrong_stage.stage = Stage::Vertex;
    assert_eq!(
        wrong_stage.verify().unwrap_err(),
        "Vertex program with EarlyDepthStencil"
    );
}

#[test]
fn browser_wgsl_notes_early_depth_without_refusing_the_program() {
    let mut program = iridia_dxbc::ir::lift(include_bytes!("fixtures/early_depth.dxbc")).unwrap();
    iridia_passes::prepare(&mut program).unwrap();
    let texts = [
        iridia_naga::wgsl(&program, &iridia_naga::Profile::BROWSER).unwrap(),
        iridia_wgsl::wgsl(&program, &iridia_wgsl::Profile::BROWSER).unwrap(),
    ];
    for text in texts {
        assert!(
            text.contains("// early_depth_stencil: WGSL has no early depth/stencil guarantee."),
            "{text}"
        );
    }
}
