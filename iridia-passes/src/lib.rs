//! Passes over iridia's shader IR ([`iridia_ir`]): each takes a function and gives one that means the same, in a form
//! closer to what an output needs.
//!
//! - [`break_flags`]: exits innermost-only `break` and `continue` cannot take, routed through a jump variable.
//! - [`flatten`]: a `Regions` body made a `Cfg`, the form a graph-shaped source lifts to.
//! - [`fold`]: values computed from literals, where every target computes them alike, made literals.
//! - [`prepare`]: the sequence every output and executor takes a lifted program through: `structurize`, `promote`,
//!   `simplify` and `break_flags`, on each of its functions.
//! - [`promote`]: variables to values, carried out of constructs as results and loop parameters.
//! - [`simplify`]: lane bookkeeping (swizzles of swizzles, composes of lanes) folded, unused values removed.
//! - [`specialise`]: known pipeline state folded in, with what it computes and the branches it decides.
//! - [`structurize`]: a `Cfg` body made `Regions`.
//! - [`structure`]: Ramsey's structuring of a reducible control flow graph, by node and edge.
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

mod break_flags;
mod flatten;
mod fold;
mod prepare;
mod promote;
mod simplify;
mod specialise;
pub mod structure;
mod structurize;

pub use break_flags::break_flags;
pub use flatten::flatten;
pub use fold::fold;
pub use prepare::prepare;
pub use promote::promote;
pub use simplify::simplify;
pub use specialise::specialise;
pub use structurize::structurize;

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
