//! Analyses of iridia's shader IR ([`iridia_ir`]): questions about what a lifted program computes, asked of its code
//! with no game in it.
//!
//! - [`flow`]: the program's value graph lane by lane ([`flow::Values`]: each output lane, discard condition and
//!   write as a graph of inputs, texels, constants and the operations between them) and what may reach each output
//!   lane and the discard, and how ([`flow::Flow`]); [`flow::VertexDemand`] is the vertex input lanes a pixel program
//!   needs.
//! - [`output_uses`]: evidence from output alpha, discard comparisons and blend state.
//! - [`matrix`]: how constant-buffer lines form rows or columns of the applied matrix.
//! - [`reads`]: the words of its constant buffers a program's results are made of ([`reads::WordsRead`]).
//!
//! An analysis reads a [`iridia_ir::Program`] as a lifter gives it: a body that is a control flow graph is structured
//! first ([`iridia_passes::structurize`], on a copy), and variables are followed through their loads and stores, so a
//! program need not be promoted. What an analysis cannot follow is an error that says so, never an answer that leaves
//! it out.
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

pub mod flow;
pub mod matrix;
pub mod output_uses;
pub mod reads;

pub use reads::{GraphError, ProgramGraph, WordsRead};

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
