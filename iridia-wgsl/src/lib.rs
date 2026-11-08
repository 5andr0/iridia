//! Writes iridia's shader IR ([`iridia_ir::Program`]) as WGSL, with no other dependency than the IR.
//!
//! A program reaches here after its passes: promoted, simplified, structured. [`wgsl`] writes it for a target [`Profile`],
//! each resource where a [`BindingPlan`] binds it and the interface carried by the entry point `main` (an argument per
//! input, a member of the result structure `Out` per output); [`wgsl_for`] writes it for one pipeline (its bindings, and a
//! vertex program's outputs linked to the pixel program they feed). [`preflight`] refuses, before anything is written, a
//! program that needs a capability the profile does not allow.
//!
//! The text is one `let` per value of the program, in its order, with the structured control flow of its regions; what
//! takes more than an expression (`f32tof16`, a remainder that is exact, the high half of a product) is a helper function
//! at the top of the module.
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

mod emit;
mod flow;
mod helpers;
mod image;
mod io;
mod memory;
mod module;
mod names;
mod ops;
mod pipeline;
mod profile;
mod rayquery;
mod text;
mod wave;

use alloc::string::String;
use core::fmt;

pub use iridia_ir::target::{Bind, BindingPlan, Bindings, Capability, Requirement};
pub use pipeline::{Pipeline, Written, wgsl_for};
pub use profile::{Capabilities, Directives, Profile, needs, preflight};

use emit::Writer;
use module::Linked;

/// Why a program could not be written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for Error {}

/// The WGSL of a program for `profile`, with the profile's binding plan ([`BindingPlan::new`] of [`Profile::bindings`]):
/// preflighted and written.
pub fn wgsl(program: &iridia_ir::Program, profile: &Profile) -> Result<String, Error> {
    preflight(program, profile)?;
    let plan = BindingPlan::new(&program.interface, profile.bindings).map_err(Error)?;
    Writer::module(program, &plan, profile, |results| {
        Ok(results
            .iter()
            .map(|item| Linked {
                binding: item.binding,
                value: None,
            })
            .collect())
    })
}
