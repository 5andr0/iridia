//! Lowers iridia's shader IR ([`iridia_ir::Program`]) to a stock naga module, validated, for naga's writers.
//!
//! A program reaches here after its passes: promoted, simplified, structured. [`module`] builds the module, with each
//! resource where a [`BindingPlan`] binds it and the interface carried by the entry point `main` (an argument per
//! input, a member of the result structure `Out` per output); [`validate`] runs naga's validator on it with a target
//! [`Profile`]'s capabilities; [`wgsl`] writes it as WGSL, and [`wgsl_for`] writes it for one pipeline (its bindings,
//! and a vertex program's outputs linked to the pixel program they feed). [`preflight`] refuses, before lowering, a
//! program that needs a capability the profile does not allow, so naga rejecting a module the preflight accepted is a
//! lowering bug.
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

mod io;
mod lower;
#[cfg(feature = "wgsl")]
mod pipeline;
mod profile;

use alloc::{format, string::String};
use core::fmt;

pub use io::{EntryIo, Item, entry_io};
pub use iridia_ir::target::{Bind, BindingPlan, Bindings};
pub use lower::module;
#[cfg(feature = "wgsl")]
pub use pipeline::{Pipeline, Written, wgsl_for};
pub use profile::{Directives, Profile, Requirement, preflight, requirements};

/// Why a program could not be lowered, validated or written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for Error {}

/// Run naga's validator on a module, with the capabilities `profile` allows and no others.
pub fn validate(module: &naga::Module, profile: &Profile) -> Result<naga::valid::ModuleInfo, Error> {
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), profile.capabilities)
        .validate(module)
        .map_err(|e| {
            // the validator's error is a chain (entry point, function, expression): every link names part of the fault
            let mut message = format!("naga validation: {}", e.as_inner());
            let mut cause = core::error::Error::source(e.as_inner());
            while let Some(c) = cause {
                message.push_str(&format!(": {c}"));
                cause = c.source();
            }
            Error(message)
        })
}

/// The WGSL of a program for `profile`, with the profile's binding plan ([`BindingPlan::new`] of [`Profile::bindings`]): preflighted,
/// lowered, validated and written. A module naga rejects after the preflight accepted the program is reported as a
/// lowering bug.
#[cfg(feature = "wgsl")]
pub fn wgsl(program: &iridia_ir::Program, profile: &Profile) -> Result<String, Error> {
    preflight(program, profile)?;
    let plan = BindingPlan::new(&program.interface, profile.bindings).map_err(Error)?;
    let module = module(program, &plan)?;
    let info = validate(&module, profile).map_err(|e| {
        Error(format!(
            "lowering bug: the preflight accepted the program and {}",
            e.0
        ))
    })?;
    write(program, &module, &info, profile)
}

/// The WGSL of `program`'s validated `module`, with the directives of `profile`: naga's writer's, and for the
/// standard's `enable subgroups;` where the program needs naga's subgroup capability.
#[cfg(feature = "wgsl")]
fn write(
    program: &iridia_ir::Program,
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    profile: &Profile,
) -> Result<String, Error> {
    let text = naga::back::wgsl::write_string(module, info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| Error(format!("naga's WGSL writer: {e}")))?;
    let subgroups = profile.directives == Directives::Standard
        && requirements(program)
            .iter()
            .any(|r| r.capability == naga::valid::Capabilities::SUBGROUP);
    // a directive comes before every declaration; naga's own are its first lines, and their order is free
    Ok(if subgroups {
        format!("enable subgroups;\n{text}")
    } else {
        text
    })
}

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
