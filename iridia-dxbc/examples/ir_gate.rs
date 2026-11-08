//! Inspect one container's lifts, seeded runs and request differences.
//! `IR_GATE_ONE=<container> cargo run --release -p iridia-dxbc --example ir_gate`

// The diagnostic uses the same case generation and comparisons as the corpus gate.
#[allow(dead_code)]
#[path = "../tests/ir_gate.rs"]
mod gate;

// The shared corpus helpers resolve the input registry from their crate root.
pub(crate) use gate::inputs;

fn main() {
    gate::diagnostic::run();
}
