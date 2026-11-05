//! Passes over iridia's shader IR ([`iridia_ir`]): each takes a function and gives one that means the same, in a form
//! closer to what an output needs.
//!
//! - [`promote`]: variables to values, carried out of constructs as results and loop parameters.
//! - [`simplify`]: lane bookkeeping (swizzles of swizzles, composes of lanes) folded, unused values removed.
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

mod promote;
mod simplify;

pub use promote::promote;
pub use simplify::simplify;
