#![allow(clippy::unwrap_used)]
//! `iridia_passes::flatten_sampler_heaps` on dxc's programs that sample with a sampler from the sampler heap
//! (`SamplerDescriptorHeap[i]`, an unbounded array of samplers indexed at run time): the heap becomes one sampler, every
//! sample names it, the WGSL has no binding array and enables none, and a program with no heap is left as it is.
use std::path::Path;

use iridia_ir::{Op, Program, interface::ResourceKind, target::BindingPlan};
use iridia_naga::{Pipeline, Profile};

fn lifted(name: &str) -> Program {
    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap();
    let mut p = iridia_dxil::ir::lift(&bytes).unwrap();
    iridia_passes::prepare(&mut p).unwrap();
    p
}

fn heaps(p: &Program) -> usize {
    p.interface
        .resources
        .iter()
        .filter(|r| r.is_array() && matches!(r.kind, ResourceKind::Sampler { .. }))
        .count()
}

fn sampler_elements(p: &Program) -> usize {
    p.entry
        .values
        .iter()
        .filter(|v| match &v.op {
            Op::Sample { sampler, .. } | Op::Gather { sampler, .. } => sampler.index.is_some(),
            _ => false,
        })
        .count()
}

fn written(p: &Program, profile: &Profile) -> String {
    let plan = BindingPlan::d3d(&p.interface).unwrap();
    iridia_naga::wgsl_for(p, profile, &Pipeline { plan, feeds: None })
        .unwrap()
        .text
}

#[test]
fn a_sampler_heap_is_one_sampler_and_every_sample_names_it() {
    // heap.dxil also reads its textures and buffers from the resource heap, which stays an array of descriptors (the
    // browser profile refuses it): only the sampler's text is checked there
    for (name, writable) in [("heap.dxil", false), ("sampler_heap.dxil", true)] {
        let mut p = lifted(name);
        assert_eq!(heaps(&p), 1, "{name}");
        assert!(sampler_elements(&p) > 0, "{name}");
        assert!(
            written(&p, &Profile::NAGA).contains("binding_array<sampler>"),
            "{name}"
        );
        let notes = p.notes.len();
        assert_eq!(iridia_passes::flatten_sampler_heaps(&mut p), 1, "{name}");
        iridia_ir::validate(&p).unwrap();
        assert_eq!(heaps(&p), 0, "{name}");
        assert_eq!(sampler_elements(&p), 0, "{name}");
        assert_eq!(p.notes.len(), notes + 1, "{name}: {:?}", p.notes);
        assert!(
            p.notes
                .iter()
                .any(|n| n == "1 sampler heap written as one sampler: every index reads the same sampler"),
            "{:?}",
            p.notes
        );
        // the sampler binds after the registers there are, and no variable of it is an array: the text enables no
        // extension for one, which a front end that does not know it would refuse whole
        assert!(
            !written(&p, &Profile::NAGA).contains("binding_array<sampler>"),
            "{name}"
        );
        if writable {
            for profile in [Profile::NAGA, Profile::BROWSER] {
                let text = written(&p, &profile);
                assert!(!text.contains("binding_array"), "{name}: {text}");
            }
        }
        // a second run finds nothing
        let mut again = p.clone();
        assert_eq!(iridia_passes::flatten_sampler_heaps(&mut again), 0);
        assert_eq!(again, p);
    }
}

#[test]
fn a_program_with_no_sampler_heap_is_left_as_it_is() {
    let mut p = lifted("vertex.dxil");
    let before = p.clone();
    assert_eq!(iridia_passes::flatten_sampler_heaps(&mut p), 0);
    assert_eq!(p, before);
}
