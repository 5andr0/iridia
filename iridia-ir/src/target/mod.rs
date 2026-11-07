//! What a module written from a program carries of it, decided once for every writer and executor: where each resource
//! binds ([`BindingPlan`]), how the entry point carries the interface ([`entry_io`]: an argument per input varying, a
//! result member per output, each at a location or a [`BuiltIn`]), and which capabilities of a target the program needs
//! ([`requirements`]). It names no writer's types: a writer maps these to its own (naga's bindings and capabilities,
//! WGSL's attributes), so two writers of one program bind and carry it alike.

mod bind;
mod capability;
mod entry;

pub use bind::{Bind, BindingPlan, Bindings};
pub use capability::{ArrayClass, Capability, Requirement, requirements};
pub use entry::{Binding, BuiltIn, EntryIo, IoItem, IoType, carried, entry_io, varying_name};
