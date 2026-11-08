//! Small readers of the container and module that both the lift and the IR lift use.

use dxil_reader::module::{Function, Module, ValueId};
use dxil_reader::reflection;
use iridia::Varying;

use crate::{Result, error, read};

/// The low 32 bits of a constant, as an LLVM integer constant of 32 bits or fewer is stored (sign-extended to 64).
pub(crate) fn low32(value: u64) -> u32 {
    u32::try_from(value & 0xffff_ffff).unwrap_or(u32::MAX)
}

/// A hull program's patch-constant function, which with the entry (its control-point phase) makes the program; `None`
/// for a program of another kind.
pub(crate) fn patch_constant_function(module: &Module, program_version: u32) -> Result<Option<&Function>> {
    if program_version >> 16 != 3 {
        return Ok(None);
    }
    let state = reflection::hull_state(module).map_err(read)?;
    module
        .functions
        .iter()
        .find(|f| f.value == state.patch_constant)
        .map(Some)
        .ok_or_else(|| error(0, "the patch-constant function has no body"))
}

/// The target the program was compiled for, as `dxc -T` names it (`ps_6_0`, `as_6_5`); a shader kind no target names
/// (a ray tracing kind, which only a library's functions have) is refused by name.
pub(crate) fn target_of(part: &dxil_reader::Part<'_>) -> Result<String> {
    part.target().ok_or_else(|| {
        error(
            0,
            format!(
                "a program of shader kind {}, which no target names",
                part.program_version >> 16
            ),
        )
    })
}

/// The patch-constant varying whose register and lanes hold this cell, and the lane's position in its value.
pub(crate) fn patch_cell(list: &[Varying], register: u32, lane: u8) -> Option<(usize, u8)> {
    list.iter().enumerate().find_map(|(i, v)| {
        let d3d = v.provenance.d3d?;
        (d3d.register == register && lanes_below(d3d.mask, lane).is_some())
            .then(|| (i, lanes_below(d3d.mask, lane).unwrap_or(0)))
    })
}

/// How many lanes of `mask` come before `lane`, where the mask names `lane`.
pub(crate) fn lanes_below(mask: u8, lane: u8) -> Option<u8> {
    let lane = u32::from(lane);
    (mask.checked_shr(lane)? & 1 != 0).then(|| {
        let below = (u32::from(mask) & 1u32.checked_shl(lane).unwrap_or(0).wrapping_sub(1)).count_ones();
        u8::try_from(below).unwrap_or(0)
    })
}

/// `WaveActiveOp` codes.
pub(crate) const WAVE_SUM: i64 = 0;
pub(crate) const WAVE_PRODUCT: i64 = 1;
pub(crate) const WAVE_MIN: i64 = 2;
pub(crate) const WAVE_MAX: i64 = 3;

/// Argument `n` of a call at bit `at`, an error naming the call when it has fewer.
pub(crate) fn argument(args: &[ValueId], n: usize, at: usize) -> Result<ValueId> {
    args.get(n).copied().ok_or_else(|| {
        error(
            at,
            format!(
                "an operation reads argument {n}, where the call has {}",
                args.len()
            ),
        )
    })
}
