//! How a program's interface is carried by its entry point: the one place that says which module argument or result
//! member a varying is, with its name, type and binding. The validator checks a module against it and a lifter builds
//! the module from it.

use naga::Binding;

use iridia_ir::target::{self, BuiltIn};

use crate::{Interface, Stage, StageFacts};

pub use iridia_ir::target::{FnItem, FunctionSig, IoType, ItemList, stage_functions, varying_name};

/// One argument or result member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoItem {
    /// `<NAME><index>` after the label, or `clip_distances`.
    pub name: String,
    /// The type of the member.
    pub ty: IoType,
    /// The naga binding it has: a location or a built-in.
    pub binding: Binding,
    /// Which varyings of the interface's list (inputs for an argument, outputs for a result) it carries: one, or every
    /// clip distance.
    pub varyings: Vec<usize>,
}

/// The entry point's arguments (one per input varying) and result members (one per output varying, the clip distances
/// being one together), in interface order. A varying nothing reads or writes (`used` false) that no binding exists
/// for is left out: the interface lists every element of the container's signature, the module carries what it can.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryIo {
    /// One per input varying that is used or has a binding.
    pub arguments: Vec<IoItem>,
    /// One per output varying (the clip distances together), in interface order.
    pub results: Vec<IoItem>,
}

/// naga's binding for a target's.
fn naga_binding(b: target::Binding) -> Binding {
    match b {
        target::Binding::BuiltIn(b) => Binding::BuiltIn(match b {
            BuiltIn::Position => naga::BuiltIn::Position { invariant: false },
            BuiltIn::FrontFacing => naga::BuiltIn::FrontFacing,
            BuiltIn::VertexIndex => naga::BuiltIn::VertexIndex,
            BuiltIn::InstanceIndex => naga::BuiltIn::InstanceIndex,
            BuiltIn::PrimitiveIndex => naga::BuiltIn::PrimitiveIndex,
            BuiltIn::SampleIndex => naga::BuiltIn::SampleIndex,
            BuiltIn::SampleMask => naga::BuiltIn::SampleMask,
            BuiltIn::FragDepth => naga::BuiltIn::FragDepth,
            BuiltIn::WorkgroupId => naga::BuiltIn::WorkGroupId,
            BuiltIn::LocalInvocationId => naga::BuiltIn::LocalInvocationId,
            BuiltIn::GlobalInvocationId => naga::BuiltIn::GlobalInvocationId,
            BuiltIn::LocalInvocationIndex => naga::BuiltIn::LocalInvocationIndex,
            BuiltIn::SubgroupInvocationId => naga::BuiltIn::SubgroupInvocationId,
            BuiltIn::SubgroupSize => naga::BuiltIn::SubgroupSize,
            BuiltIn::ClipDistances => naga::BuiltIn::ClipDistances,
        }),
        target::Binding::Location {
            location,
            interpolation,
            sampling,
        } => Binding::Location {
            location,
            interpolation: interpolation.map(|i| match i {
                crate::InterpMode::Perspective => naga::Interpolation::Perspective,
                crate::InterpMode::Linear => naga::Interpolation::Linear,
                crate::InterpMode::Flat => naga::Interpolation::Flat,
            }),
            sampling: match sampling {
                crate::InterpSampling::Center => None,
                crate::InterpSampling::Centroid => Some(naga::Sampling::Centroid),
                crate::InterpSampling::Sample => Some(naga::Sampling::Sample),
            },
            blend_src: None,
            per_primitive: false,
        },
    }
}

/// The members of a mesh program's vertex type, one for each output varying, bound as a vertex program's outputs are.
pub fn mesh_vertex_io(interface: &Interface) -> Result<Vec<IoItem>, String> {
    let mut vertex = interface.clone();
    vertex.stage = Stage::Vertex;
    vertex.inputs.clear();
    vertex.facts = StageFacts::None;
    Ok(entry_io(&vertex)?.results)
}

/// The entry point's arguments and result members for `interface` ([`iridia_ir::target::entry_io`], with naga's
/// bindings). `Err` says why the module cannot carry it (a system value no stage binding exists for).
pub fn entry_io(interface: &Interface) -> Result<EntryIo, String> {
    let io = target::entry_io(interface)?;
    let item = |i: target::IoItem| IoItem {
        name: i.name,
        ty: i.ty,
        binding: naga_binding(i.binding),
        varyings: i.varyings,
    };
    Ok(EntryIo {
        arguments: io.arguments.into_iter().map(item).collect(),
        results: io.results.into_iter().map(item).collect(),
    })
}
