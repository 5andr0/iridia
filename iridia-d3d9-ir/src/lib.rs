//! Reads raw Direct3D 9 token streams of every shader model (vertex 1.1 to 3.0, pixel 1.1 to 3.0) and lifts them
//! into iridia's shader IR ([`iridia_ir::Program`]).
//!
//! The records are read by `d3d9-reader`; [`read`] checks every register against its version's banks and makes the
//! inputs and outputs a version implies by register explicit. [`signature`] states the interface the declarations,
//! registers and constant table give, and [`lift`] builds the program: one variable per register, each instruction's
//! operations in order, the outputs stored at the end.
#![deny(missing_docs)]
#![deny(clippy::disallowed_methods, clippy::disallowed_types)]
#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation
    )
)]

mod lift;
mod model;
mod signature;

pub use d3d9_reader::{Comment, Constant, ConstantTable, Type, TypeMember};
pub use lift::{LIFTER, LIFTER_VERSION, Options, Ps1x, lift, lift_with};
pub use model::{
    Comparison, Destination, Index, Input, Instruction, Opcode, Output, Register, Shader, Source, Version,
    read, usage,
};
pub use signature::{
    Bank, PARTIAL_PRECISION, PIXEL_CONSTANTS, Sampler, Signature, VERTEX_CONSTANTS, lanes_in, outputs,
    read_input, signature, varying,
};

/// A refusal states the word offset in the raw token stream and the unsupported or malformed feature.
pub type Error = iridia_ir::Refusal;
/// The result of every function in this crate: `Ok` or a refusal that names the word offset.
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn error(at: usize, detail: impl Into<String>) -> Error {
    Error::new("D3D9", iridia_ir::Unit::Word, at, detail)
}

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
