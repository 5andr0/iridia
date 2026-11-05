//! iridia's shader IR: one vocabulary of typed SSA values and operations, in a function body that is either a tree
//! of structured regions or a control flow graph with block parameters, beside the interface vocabulary a program
//! exposes ([`interface`]).
//!
//! The design is `docs/design/ir.md` in the repository. In brief:
//!
//! - A [`Function`] owns its values ([`ValueDef`], by [`ValueId`]): each has a [`Type`], an [`Op`], a source
//!   [`Origin`] and a [`Precision`] permission. Operations that differ between sources carry their rule in an enum
//!   ([`MinMaxNan`], [`SaturateNan`], [`MulZero`], [`OnZero`], [`FloatToInt`], [`Rounding`]); none has a default.
//! - Its [`Body`] is [`Body::Regions`] (constructs [`Node::If`], [`Node::Block`] and [`Node::Loop`] whose regions end in
//!   an [`End`]: a yield, a labelled break or continue carrying values, a return, a kill) or [`Body::Cfg`] (blocks with
//!   parameters and [`Terminator`]s, block 0 the entry).
//! - A [`Program`] is the entry function with its [`interface::Interface`], [`Conventions`], [`FloatEnv`],
//!   [`Provenance`] and notes. [`validate()`] checks it; [`Builder`] makes one; [`text`] prints it.
#![no_std]
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

extern crate alloc;

mod builder;
mod func;
pub mod interface;
mod op;
mod program;
mod refusal;
pub mod text;
mod ty;
mod validate;

pub use builder::Builder;
pub use func::{Block, Body, End, Function, Node, Origin, Precision, Region, Target, Terminator, ValueDef, Var};
pub use op::{
    Binary, Cmp, FloatToInt, LabelId, Level, MinMaxNan, MulZero, Op, OnZero, OutOfBounds, Rounding, SaturateNan,
    Unary, ValueId, VarId,
};
pub use program::{ClipDepth, Conventions, Corner, Denormals, FloatEnv, Program, Provenance};
pub use refusal::{Refusal, Unit, source_hash};
pub use ty::{Scalar, Type};
pub use validate::{Problem, derived_type, dominators, validate, value_type};
