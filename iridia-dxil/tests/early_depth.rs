#![allow(clippy::unwrap_used)]
//! The compiled early-depth attribute is an IR fact; the same shader without it is the control.
use iridia_ir::interface::{Stage, StageFacts};

#[test]
fn early_depth_is_a_pixel_stage_fact() {
    let early = iridia_dxil::ir::lift(include_bytes!("fixtures/early_depth.dxil")).unwrap();
    assert_eq!(early.interface.facts, StageFacts::EarlyDepthStencil);
    iridia_ir::validate(&early).unwrap();
    let late = iridia_dxil::ir::lift(include_bytes!("fixtures/late_depth.dxil")).unwrap();
    assert_eq!(late.interface.facts, StageFacts::None);
    iridia_ir::validate(&late).unwrap();
    let mut wrong_stage = early.interface;
    wrong_stage.stage = Stage::Vertex;
    assert_eq!(
        wrong_stage.verify().unwrap_err(),
        "Vertex program with EarlyDepthStencil"
    );
}
