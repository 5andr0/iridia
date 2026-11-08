//! The binding plans: [`BindingPlan::webgpu`] keeps every resource of a generated interface inside WebGPU's default
//! limits (four bind groups, bindings below 1000), no two at one binding, a space-0 register at its number, with the
//! names and counters of [`BindingPlan::d3d`].

#![allow(clippy::unwrap_used, reason = "generated interfaces")]

use std::collections::BTreeSet;

use iridia_ir::interface::{
    Access, BufferKind, D3dBinding, D3dClass, Interface, Resource, ResourceId, ResourceKind,
    ResourceProvenance, Stage, StageFacts, StorageTarget,
};
use iridia_ir::target::{BindingPlan, Bindings};
use proptest::prelude::*;

/// A resource of register class `class` (0 constant buffer, 1 shader resource, 2 sampler, 3 unordered access) at
/// `register` of `space`; an unordered-access view structured or raw.
fn resource(id: u32, class: u8, register: u32, space: u32, structured: bool) -> Resource {
    let class = match class {
        0 => D3dClass::ConstantBuffer,
        1 => D3dClass::ShaderResource,
        2 => D3dClass::Sampler,
        _ => D3dClass::UnorderedAccess,
    };
    let kind = if class == D3dClass::UnorderedAccess {
        ResourceKind::Storage {
            target: StorageTarget::Buffer(if structured {
                BufferKind::Structured { stride: 4 }
            } else {
                BufferKind::Raw
            }),
            access: Access::ReadWrite,
        }
    } else {
        // the plan reads the register class, not the kind
        ResourceKind::Buffer(BufferKind::Raw)
    };
    Resource {
        id: ResourceId(id),
        kind,
        count: 1,
        provenance: ResourceProvenance {
            name: format!("r{id}"),
            d3d: Some(D3dBinding {
                class,
                register,
                space,
            }),
            counter_of: None,
            layout: None,
        },
    }
}

fn interface(resources: Vec<Resource>) -> Interface {
    Interface {
        stage: Stage::Compute,
        inputs: Vec::new(),
        outputs: Vec::new(),
        resources,
        facts: StageFacts::None,
    }
}

/// Resources at different registers of each class and space.
fn resources() -> impl Strategy<Value = Vec<Resource>> {
    proptest::collection::btree_map((0u8..4, 0u32..48, 0u32..4), any::<bool>(), 0..40).prop_map(|m| {
        (0u32..)
            .zip(m)
            .map(|(id, ((class, register, space), structured))| {
                resource(id, class, register, space, structured)
            })
            .collect()
    })
}

proptest! {
    #[test]
    fn the_webgpu_plan_keeps_inside_webgpu_limits(resources in resources()) {
        let i = interface(resources);
        let web = BindingPlan::webgpu(&i).unwrap();
        let d3d = BindingPlan::d3d(&i).unwrap();
        let mut taken = BTreeSet::new();
        for b in web.bindings.iter().chain(&web.counters) {
            prop_assert!(b.group < 4, "{} in group {}", b.name, b.group);
            prop_assert!(b.binding < 1000, "{} at {}", b.name, b.binding);
            prop_assert!(taken.insert((b.group, b.binding)), "two resources at {}:{}", b.group, b.binding);
        }
        // the same resources and names as the D3D plan, a space-0 register at its number
        prop_assert_eq!(web.bindings.len(), d3d.bindings.len());
        for (w, d) in web.bindings.iter().zip(&d3d.bindings) {
            prop_assert_eq!(w.resource, d.resource);
            prop_assert_eq!(&w.name, &d.name);
            prop_assert_eq!(w.group, d.group);
            let r = i.resources.iter().find(|r| r.id == w.resource).unwrap();
            let reg = r.provenance.d3d.unwrap();
            if reg.space == 0 {
                prop_assert_eq!(w.binding, reg.register);
            }
        }
        let names = |p: &BindingPlan| p.counters.iter().map(|b| (b.resource, b.name.clone())).collect::<BTreeSet<_>>();
        prop_assert_eq!(names(&web), names(&d3d));
        prop_assert!(web.counters.iter().all(|b| b.group == 3));
    }
}

#[test]
fn a_binding_at_the_limit_is_refused_by_name() {
    // space 0 takes u999; the view of space 1 would bind at 1000
    let i = interface(vec![resource(0, 3, 999, 0, false), resource(1, 3, 0, 1, false)]);
    let e = BindingPlan::webgpu(&i).unwrap_err();
    assert!(e.contains("u0_space1") && e.contains("1000"), "{e}");
}

#[test]
fn the_webgpu_plan_binds_other_spaces_and_counters_after_space_0() {
    // a structured view of space 2 and its counter: inside group 3, below 1000
    let i = interface(vec![resource(0, 3, 5, 0, true), resource(1, 3, 1, 2, true)]);
    let plan = BindingPlan::new(&i, Bindings::WebGpu).unwrap();
    let at: Vec<(u32, u32, &str)> = plan
        .bindings
        .iter()
        .chain(&plan.counters)
        .map(|b| (b.group, b.binding, b.name.as_str()))
        .collect();
    assert_eq!(
        at,
        vec![
            (3, 5, "u5"),
            (3, 6, "u1_space2"),
            (3, 7, "u5_counter"),
            (3, 8, "u1_space2_counter"),
        ]
    );
}

/// A resource of `kind` with no register of the source's own, as a descriptor heap's is.
fn registerless(id: u32, kind: ResourceKind) -> Resource {
    Resource {
        id: ResourceId(id),
        kind,
        count: 1,
        provenance: ResourceProvenance {
            name: format!("heap{id}"),
            d3d: None,
            counter_of: None,
            layout: None,
        },
    }
}

#[test]
fn a_resource_with_no_register_binds_after_the_highest_register_of_its_class_by_resource_id() {
    let sampler = || ResourceKind::Sampler { comparison: false };
    let i = interface(vec![
        resource(0, 2, 0, 0, false),
        // a sampler of another space does not move the space-0 end
        resource(1, 2, 7, 3, false),
        resource(2, 2, 3, 0, false),
        resource(3, 1, 1, 0, false),
        registerless(4, sampler()),
        registerless(5, ResourceKind::Buffer(BufferKind::Raw)),
        registerless(6, sampler()),
    ]);
    for bindings in [Bindings::D3d, Bindings::WebGpu] {
        let plan = BindingPlan::new(&i, bindings).unwrap();
        let at: Vec<(u32, u32, u32)> = plan
            .bindings
            .iter()
            .filter(|b| b.resource.0 >= 4)
            .map(|b| (b.resource.0, b.group, b.binding))
            .collect();
        // samplers end at s3: the two take s4 and s5; the buffer takes t2 after t1
        assert_eq!(at, vec![(4, 2, 4), (5, 1, 2), (6, 2, 5)], "{bindings:?}");
        let names: Vec<&str> = plan
            .bindings
            .iter()
            .filter(|b| b.resource.0 >= 4)
            .map(|b| b.name.as_str())
            .collect();
        assert_eq!(names, ["s4", "t2", "s5"], "{bindings:?}");
    }
    // all alone, the first of a class is register 0
    let alone = BindingPlan::d3d(&interface(vec![registerless(0, sampler())])).unwrap();
    assert_eq!((alone.bindings[0].group, alone.bindings[0].binding), (2, 0));
}

#[test]
fn a_hidden_counter_with_no_register_is_still_refused() {
    let mut counter = registerless(
        1,
        ResourceKind::Storage {
            target: StorageTarget::Buffer(BufferKind::Raw),
            access: Access::ReadWrite,
        },
    );
    counter.provenance.counter_of = Some(ResourceId(0));
    let e = BindingPlan::d3d(&interface(vec![resource(0, 3, 0, 0, true), counter])).unwrap_err();
    assert!(e.contains("no D3D register"), "{e}");
}
