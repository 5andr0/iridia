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
mod sampler;
mod signature;

pub use d3d9_reader::{Comment, Constant, ConstantTable, Type, TypeMember};
pub use lift::{
    LIFTER, LIFTER_VERSION, Options, Projection, Ps1x, lift, lift_with, project, project_vertex_output,
    project_vertex_output_with, project_with, reencode, reencode_with,
};
pub use model::{
    Comparison, Destination, Index, Input, Instruction, Opcode, Output, Register, Shader, Source, Version,
    read, usage,
};
pub use sampler::sampler_states;
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

/// Frame the first raw SM3 vertex or pixel program in a container's remaining bytes.
/// Returns its stage and exact bytes through END; subsequent container metadata is not consumed.
/// `d3d9-reader` reads every record up to END, so END is never searched for inside operands or comments; an
/// operation the lifter does not support is retained as bytecode for [`lift`] to refuse.
pub fn frame(bytes: &[u8]) -> Result<(iridia_ir::interface::Stage, &[u8])> {
    use iridia_ir::interface::Stage;
    let (program, span) = d3d9_reader::frame(bytes).map_err(|e| error(e.word, e.detail))?;
    let stage = match program.version.token() {
        0xfffe_0300 => Stage::Vertex,
        0xffff_0300 => Stage::Pixel,
        _ => {
            return Err(error(
                0,
                format!("version {}: only vs_3_0 and ps_3_0 are framed", program.version),
            ));
        }
    };
    Ok((stage, span))
}
