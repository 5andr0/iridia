//! Where a module binds each resource of a program: Direct3D's register classes as bind groups, as they are or kept
//! inside WebGPU's default limits.

use alloc::{collections::BTreeMap, format, string::String, vec::Vec};

use crate::interface::{
    BufferKind, D3dBinding, D3dClass, Interface, ResourceId, ResourceKind, StorageTarget,
};

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

/// How a target binds a program's resources.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bindings {
    /// Direct3D's registers as they are ([`BindingPlan::d3d`]).
    D3d,
    /// Inside WebGPU's default limits ([`BindingPlan::webgpu`]).
    WebGpu,
}

/// Where a module binds each resource of a program.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BindingPlan {
    /// One per resource.
    pub bindings: Vec<Bind>,
    /// The hidden counter of a structured unordered-access view (D3D's append and consume buffers), a buffer of its
    /// own in WebGPU: one per view that has one, `resource` the view. A module binds only those the program changes.
    pub counters: Vec<Bind>,
}

/// One bound resource as Direct3D places it: the bind group of its register class, its register and space, its name
/// (`cbN`, `tN`, `sN`, `uN`, `uN_spaceS`), and whether it has a hidden counter.
struct Placed {
    resource: ResourceId,
    group: u32,
    register: u32,
    space: u32,
    name: String,
    counter: bool,
}

/// The register class a resource of this kind is read through; pipeline state has none.
fn class_of(kind: &ResourceKind) -> Option<D3dClass> {
    Some(match kind {
        ResourceKind::ConstantBuffer { .. } => D3dClass::ConstantBuffer,
        ResourceKind::Texture { .. } | ResourceKind::Buffer(_) | ResourceKind::AccelerationStructure => {
            D3dClass::ShaderResource
        }
        ResourceKind::Sampler { .. } => D3dClass::Sampler,
        ResourceKind::Storage { .. } => D3dClass::UnorderedAccess,
        ResourceKind::PipelineState(_) => return None,
    })
}

/// The interface's bound resources as Direct3D places them; pipeline state is not bound. A resource with no register of
/// the source's own (a descriptor heap's, a flattened sampler heap's sampler) is placed in space 0 after the highest
/// register of its class, in the order of resource ids; a hidden counter, bound with its view, is refused.
fn placed(interface: &Interface) -> Result<Vec<Placed>, String> {
    let mut out = Vec::new();
    // pipeline state is set with the draw, not bound: specialisation folds it (a state block would be planned here)
    let bound = || {
        interface
            .resources
            .iter()
            .filter(|r| !matches!(r.kind, ResourceKind::PipelineState(_)))
    };
    // the next free register of each class in space 0, for the resources that state none
    let mut next = [0u32; 4];
    let slot = |class: D3dClass| match class {
        D3dClass::ConstantBuffer => 0,
        D3dClass::ShaderResource => 1,
        D3dClass::Sampler => 2,
        D3dClass::UnorderedAccess => 3,
    };
    for d in bound()
        .filter_map(|r| r.provenance.d3d.as_ref())
        .filter(|d| d.space == 0)
    {
        if let Some(n) = next.get_mut(slot(d.class)) {
            *n = (*n).max(d.register.saturating_add(1));
        }
    }
    let mut assigned = BTreeMap::new();
    for r in bound().filter(|r| r.provenance.d3d.is_none()) {
        if r.provenance.counter_of.is_some() {
            return Err(format!(
                "resource {} ({}) has no D3D register",
                r.id.0, r.provenance.name
            ));
        }
        let class = class_of(&r.kind).ok_or_else(|| {
            format!(
                "resource {} ({}) has no register class",
                r.id.0, r.provenance.name
            )
        })?;
        let n = next.get_mut(slot(class)).ok_or_else(|| {
            format!(
                "resource {} ({}) has no register class",
                r.id.0, r.provenance.name
            )
        })?;
        assigned.insert(
            r.id,
            D3dBinding {
                class,
                register: *n,
                space: 0,
            },
        );
        *n = n.saturating_add(1);
    }
    for r in bound() {
        let d = r
            .provenance
            .d3d
            .as_ref()
            .or_else(|| assigned.get(&r.id))
            .ok_or_else(|| format!("resource {} ({}) has no D3D register", r.id.0, r.provenance.name))?;
        let (group, prefix) = match d.class {
            D3dClass::ConstantBuffer => (0, "cb"),
            D3dClass::ShaderResource => (1, "t"),
            D3dClass::Sampler => (2, "s"),
            D3dClass::UnorderedAccess => (3, "u"),
        };
        let name = if d.space == 0 {
            format!("{prefix}{}", d.register)
        } else {
            format!("{prefix}{}_space{}", d.register, d.space)
        };
        out.push(Placed {
            resource: r.id,
            group,
            register: d.register,
            space: d.space,
            name,
            counter: matches!(
                r.kind,
                ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Structured { .. }),
                    ..
                }
            ),
        });
    }
    Ok(out)
}

/// WebGPU's default `maxBindingsPerBindGroup`: every binding is below it.
const WEBGPU_BINDINGS: u32 = 1000;

impl BindingPlan {
    /// Direct3D's register classes as bind groups, each register its binding: constant buffers in group 0 (`cbN`),
    /// shader resources in group 1 (`tN`), samplers in group 2 (`sN`), unordered-access views in group 3 (`uN`). A
    /// register of a space other than 0 binds at `(space << 16) | register` (`uN_spaceS`). A structured view's hidden
    /// counter binds in group 4 at the view's binding (`uN_counter`). Pipeline state is not bound. A resource with no
    /// D3D register of the source's own (a descriptor heap's, the sampler a flattened sampler heap leaves) is placed in
    /// space 0 after the highest register of its class that space has, in the order of resource ids, and named as a
    /// register of that number is: the number is the plan's, not the source's. A hidden counter, bound with its view,
    /// is refused. The bindings past 999 and group 4 are past WebGPU's default limits:
    /// [`BindingPlan::webgpu`] keeps inside them.
    pub fn d3d(interface: &Interface) -> Result<BindingPlan, String> {
        let mut bindings = Vec::new();
        let mut counters = Vec::new();
        for p in placed(interface)? {
            let binding = if p.space == 0 {
                p.register
            } else {
                if p.space > 0xffff || p.register > 0xffff {
                    return Err(format!(
                        "resource {}: register {} of space {} is past what a binding of (space << 16) | register \
                         holds",
                        p.name, p.register, p.space
                    ));
                }
                (p.space << 16) | p.register
            };
            if p.counter {
                counters.push(Bind {
                    resource: p.resource,
                    group: 4,
                    binding,
                    name: format!("{}_counter", p.name),
                });
            }
            bindings.push(Bind {
                resource: p.resource,
                group: p.group,
                binding,
                name: p.name,
            });
        }
        Ok(BindingPlan { bindings, counters })
    }

    /// The register classes as bind groups as [`BindingPlan::d3d`] makes them, kept inside WebGPU's default limits
    /// (four bind groups, bindings below 1000): a register of space 0 binds at its number; the registers of other
    /// spaces bind after the largest binding of their group, in order of space and register; and the hidden counters
    /// of structured views bind in group 3 after its views, in the views' order. A binding the plan would place at 1000
    /// or past is refused, naming the resource.
    pub fn webgpu(interface: &Interface) -> Result<BindingPlan, String> {
        let placed = placed(interface)?;
        let mut next = [0u32; 4];
        for p in placed.iter().filter(|p| p.space == 0) {
            if let Some(n) = usize::try_from(p.group).ok().and_then(|g| next.get_mut(g)) {
                *n = (*n).max(p.register.saturating_add(1));
            }
        }
        let mut order: Vec<usize> = (0..placed.len()).collect();
        order.sort_by_key(|&i| placed.get(i).map(|p| (p.space, p.register)));
        let mut binding_of = alloc::vec![0u32; placed.len()];
        for &i in &order {
            let (Some(p), Some(b)) = (placed.get(i), binding_of.get_mut(i)) else {
                continue;
            };
            *b = if p.space == 0 {
                p.register
            } else {
                let n = usize::try_from(p.group)
                    .ok()
                    .and_then(|g| next.get_mut(g))
                    .ok_or_else(|| format!("resource {}: bind group {}", p.name, p.group))?;
                let at = *n;
                *n = n.saturating_add(1);
                at
            };
        }
        let mut bindings = Vec::new();
        let mut counters = Vec::new();
        let within = |name: &str, group: u32, binding: u32| {
            if binding < WEBGPU_BINDINGS {
                Ok(())
            } else {
                Err(format!(
                    "{name} would bind at {binding} of group {group}, past WebGPU's {WEBGPU_BINDINGS} bindings a group"
                ))
            }
        };
        for (p, &binding) in placed.iter().zip(&binding_of) {
            within(&p.name, p.group, binding)?;
            bindings.push(Bind {
                resource: p.resource,
                group: p.group,
                binding,
                name: p.name.clone(),
            });
        }
        for &i in &order {
            let Some(p) = placed.get(i).filter(|p| p.counter) else {
                continue;
            };
            let Some(n) = next.get_mut(3) else {
                continue;
            };
            let binding = *n;
            *n = binding.saturating_add(1);
            let name = format!("{}_counter", p.name);
            within(&name, 3, binding)?;
            counters.push(Bind {
                resource: p.resource,
                group: 3,
                binding,
                name,
            });
        }
        Ok(BindingPlan { bindings, counters })
    }

    /// The plan `bindings` names.
    pub fn new(interface: &Interface, bindings: Bindings) -> Result<BindingPlan, String> {
        match bindings {
            Bindings::D3d => BindingPlan::d3d(interface),
            Bindings::WebGpu => BindingPlan::webgpu(interface),
        }
    }
}
