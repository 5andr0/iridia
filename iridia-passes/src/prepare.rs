//! The passes a lifted program takes before an output or the executor: one sequence, so each caller runs the same.

use alloc::string::String;

use iridia_ir::Program;

use crate::{break_flags, promote, simplify, structurize};

/// Take a lifted program through the passes every output and executor expects, on each of its functions (the entry,
/// and a hull program's patch function): [`structurize`] (a `Cfg` body made `Regions`; a `Regions` body is left as it
/// is), [`promote`], [`simplify`], then [`break_flags`]. What comes out has a `Regions` body whose exits leave only the
/// innermost loop or switch: the form `iridia-exec`'s `::from_ir` constructors and `iridia-naga` take, and it means
/// what the lifted program meant.
///
/// Stops at the first pass that refuses, with the pass's name and its reason (an irreducible graph, for one); the
/// program is then left part-way and should not be used.
pub fn prepare(program: &mut Program) -> Result<(), String> {
    for f in core::iter::once(&mut program.entry).chain(program.patch.as_mut()) {
        structurize(f).map_err(|e| alloc::format!("structurize: {e}"))?;
        promote(f).map_err(|e| alloc::format!("promote: {e}"))?;
        simplify(f);
        break_flags(f).map_err(|e| alloc::format!("break_flags: {e}"))?;
    }
    Ok(())
}
