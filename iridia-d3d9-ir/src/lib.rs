//! Reads raw Direct3D 9 `ps_3_0` and `vs_3_0` token streams and lifts them into iridia's shader IR
//! ([`iridia_ir::Program`]).
//!
//! The records are read by `d3d9-reader`; [`read`] takes the subset this crate lifts (straight-line float
//! arithmetic, texture sampling and discard) and refuses the rest by name and word. [`signature`] states the
//! interface the declarations and constant table give, and [`lift`] builds the program: one variable per register,
//! each instruction's operations in order, the outputs stored at the end.
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
mod signature;
mod subset;

pub use d3d9_reader::{Comment, Constant, ConstantTable, Type, TypeMember};
pub use lift::{LIFTER, LIFTER_VERSION, lift};
pub use signature::{
    Bank, PARTIAL_PRECISION, Sampler, Signature, VERTEX_CONSTANTS, lanes_in, outputs, read_input, signature, varying,
};
pub use subset::{Destination, Input, Instruction, Opcode, Output, Register, Shader, Source, read};

/// A refusal states the word offset in the raw token stream and the unsupported or malformed feature.
pub type Error = iridia_ir::Refusal;
/// The result of every function in this crate: `Ok` or a refusal that names the word offset.
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn error(at: usize, detail: impl Into<String>) -> Error {
    Error::new("D3D9", iridia_ir::Unit::Word, at, detail)
}
