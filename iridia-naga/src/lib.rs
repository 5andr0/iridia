//! Lowers iridia's shader IR ([`iridia_ir::Program`]) to a stock naga module, validated, for naga's writers.
//!
//! A program reaches here after its passes: promoted, simplified, structured. [`module`] builds the module, with each
//! resource where a [`BindingPlan`] binds it and the interface carried by the entry point `main` (an argument per
//! input, a member of the result structure `Out` per output); [`validate`] runs naga's validator on it; [`wgsl`]
//! writes it as WGSL.
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

use alloc::{format, string::String, vec::Vec};
use core::fmt;

use iridia_ir::{
    Program,
    interface::{D3dClass, Interface, ResourceId},
};

pub use io::{EntryIo, Item, entry_io};
pub use lower::module;

/// Why a program could not be lowered, validated or written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for Error {}

/// Where one resource is bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bind {
    /// The resource.
    pub resource: ResourceId,
    /// Its bind group.
    pub group: u32,
    /// Its binding in the group.
    pub binding: u32,
    /// The global's name.
    pub name: String,
}

/// Where an output binds each resource of a program.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BindingPlan {
    /// One per resource.
    pub bindings: Vec<Bind>,
}

impl BindingPlan {
    /// Direct3D's register classes as bind groups, each register its binding: constant buffers in group 0 (`cbN`),
    /// shader resources in group 1 (`tN`), samplers in group 2 (`sN`), unordered-access views in group 3 (`uN`). A
    /// resource with no D3D register is refused.
    pub fn d3d(interface: &Interface) -> Result<BindingPlan, Error> {
        let mut bindings = Vec::new();
        for r in &interface.resources {
            let d = r
                .provenance
                .d3d
                .as_ref()
                .ok_or_else(|| Error(format!("resource {} ({}) has no D3D register", r.id.0, r.provenance.name)))?;
            let (group, prefix) = match d.class {
                D3dClass::ConstantBuffer => (0, "cb"),
                D3dClass::ShaderResource => (1, "t"),
                D3dClass::Sampler => (2, "s"),
                D3dClass::UnorderedAccess => (3, "u"),
            };
            if d.space != 0 {
                return Err(Error(format!("resource {} is in register space {}", r.provenance.name, d.space)));
            }
            bindings.push(Bind { resource: r.id, group, binding: d.register, name: format!("{prefix}{}", d.register) });
        }
        Ok(BindingPlan { bindings })
    }
}

/// Run naga's validator on a module, with every capability the module may use.
pub fn validate(module: &naga::Module) -> Result<naga::valid::ModuleInfo, Error> {
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(module)
        .map_err(|e| Error(format!("naga validation: {}", e.as_inner())))
}

/// The WGSL of a program, with the D3D binding plan.
#[cfg(feature = "wgsl")]
pub fn wgsl(program: &Program) -> Result<String, Error> {
    let plan = BindingPlan::d3d(&program.interface)?;
    let module = module(program, &plan)?;
    let info = validate(&module)?;
    naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| Error(format!("naga's WGSL writer: {e}")))
}
