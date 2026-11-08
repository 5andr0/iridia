//! The IR shape each `COMPUTE_EXPECTED` row for a source-specific cause states, checked on every run: a source hash does not
//! change when a lift moves a barrier or reorders a conversion chain, so a row stands only while the lifted program still has
//! the shape its excuse needs. A shape is a necessary condition for its cause, not a proof of it: the proof stays in the row's
//! evidence.
mod groups;
mod quantized;
mod shared;

use iridia_ir::interface::SystemValue;
use iridia_ir::{Function, Program};

use super::{SHARED_NO_SYNC, SHARED_SCAN_RACE, WARP_QUANTIZED_FLOAT_BITS};

/// Whether the program reads which thread group it is in (`SV_GroupID` or `SV_DispatchThreadID`): only then do two groups
/// do different work.
pub(super) fn reads_group(p: &Program) -> bool {
    p.interface.inputs.iter().any(|v| {
        matches!(
            v.system,
            Some(SystemValue::GroupId | SystemValue::DispatchThreadId)
        )
    })
}

/// The race shapes `function` has, for the reason of a `WARP_UNSTABLE` row: a sort, scan or transpose that WARP gives
/// differently between repeats usually races in the source (group-shared memory with no barrier or read and written in one
/// phase, or two thread groups doing the same work on the same words), not in WARP. The shape is a necessary condition for
/// the race, not a proof of it.
pub(super) fn race_note(function: &Function, reads_group: bool, groups: u32) -> String {
    let shared = shared::shape(function);
    let mut notes = Vec::new();
    if shared.unsynchronised() {
        notes.push("SHARED_NO_SYNC shape (group-shared memory written and read, no barrier)");
    }
    if shared.races_within_a_phase() {
        notes.push("SHARED_SCAN_RACE shape (group-shared word read then written in one barrier phase)");
    }
    if groups > 1 && !reads_group {
        notes.push("cross-group shape (no group number read, dispatched over more than one group)");
    }
    if notes.is_empty() {
        "no race shape found".to_owned()
    } else {
        notes.join(", ")
    }
}

/// Whether `p` has the shape of a cross-group race: several groups dispatched (`groups`), none told apart (no group number
/// read), and one buffer both loaded and stored, so a group's stores land among the others' loads.
pub(super) fn cross_group_race(p: &Program, groups: [u32; 3]) -> Result<(), &'static str> {
    if groups.iter().product::<u32>() > 1 && !reads_group(p) && groups::reads_and_writes_a_buffer(&p.entry) {
        Ok(())
    } else {
        Err("a program WARP_UNSTABLE lists for a cross-group race no longer has that shape: remove the row")
    }
}

/// The shapes of the `COMPUTE_EXPECTED` rows that `function` has, for a report on a program that differs from WARP and is not
/// listed: it names the row to consider (the shape is necessary for its cause, not a proof of it).
pub(super) fn census(function: &Function) -> String {
    let shared = shared::shape(function);
    let mut found = Vec::new();
    if quantized::check(function).is_ok() {
        found.push("quantized-float shape");
    }
    if shared.unsynchronised() {
        found.push("group-shared, no barrier");
    }
    if shared.races_within_a_phase() {
        found.push("group-shared, read then written in a phase");
    }
    if found.is_empty() {
        "no row shape".to_owned()
    } else {
        found.join(", ")
    }
}

/// Whether `function` still has the shape the excuse `why` needs: `Ok` for an excuse that states none.
pub(super) fn check(why: &str, function: &Function) -> Result<(), &'static str> {
    match why {
        WARP_QUANTIZED_FLOAT_BITS => quantized::check(function),
        SHARED_NO_SYNC if shared::shape(function).unsynchronised() => Ok(()),
        SHARED_SCAN_RACE if shared::shape(function).races_within_a_phase() => Ok(()),
        SHARED_NO_SYNC | SHARED_SCAN_RACE => Err(
            "a program COMPUTE_EXPECTED excuses for a group-shared race no longer has that shape: remove the row",
        ),
        _ => Ok(()),
    }
}
