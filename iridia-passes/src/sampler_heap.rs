//! Sampler heaps as one sampler: for a pipeline that binds one sampler where the program indexes an array of them at run
//! time (WebGPU has no arrays of resources; a sampler heap is Shader Model 6's, `SamplerDescriptorHeap[i]`).

use alloc::format;

use iridia_ir::{
    Op, Program,
    interface::{ResourceId, ResourceKind},
};

/// Each array of samplers made one sampler: its `count` is 1 and every sample or gather that names an element of it names
/// the sampler itself. Returns how many arrays there were. Which sampler an index picked is lost, so a program whose
/// samplers differ by index is not run as written: the program's notes say so, once.
///
/// The index values an operation no longer reads stay in the function (a [`crate::simplify`] drops them).
pub fn flatten_sampler_heaps(program: &mut Program) -> usize {
    let heaps: alloc::vec::Vec<ResourceId> = program
        .interface
        .resources
        .iter()
        .filter(|r| r.is_array() && matches!(r.kind, ResourceKind::Sampler { .. }))
        .map(|r| r.id)
        .collect();
    if heaps.is_empty() {
        return 0;
    }
    for r in &mut program.interface.resources {
        if heaps.contains(&r.id) {
            r.count = 1;
        }
    }
    for f in core::iter::once(&mut program.entry).chain(program.patch.as_mut()) {
        for v in &mut f.values {
            if let Op::Sample { sampler, .. } | Op::Gather { sampler, .. } = &mut v.op
                && heaps.contains(&sampler.id)
            {
                sampler.index = None;
                sampler.nonuniform = false;
            }
        }
    }
    let n = heaps.len();
    program.notes.push(format!(
        "{n} sampler heap{} written as one sampler: every index reads the same sampler",
        if n == 1 { "" } else { "s" }
    ));
    program.notes.sort();
    program.notes.dedup();
    n
}
