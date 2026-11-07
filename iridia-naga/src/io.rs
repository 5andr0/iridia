//! How an entry point carries the interface ([`iridia_ir::target::entry_io`]): an argument per input varying, a member
//! of the result structure `Out` per output varying, each named `<NAME><index>` after its label, with a location (in
//! interface order) or a built-in, as naga's bindings.

use alloc::{format, string::String, vec::Vec};

use iridia_ir::interface::{Interface, InterpMode, InterpSampling, ScalarType, Stage, Value};
use iridia_ir::target::{self, BuiltIn, IoType};
use naga::Binding;

/// One argument or result member.
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    /// `<NAME><index>`.
    pub name: String,
    /// The value carried, which may be wider than the varying (a pixel program's colour target has four lanes).
    pub value: Value,
    /// Its location or built-in.
    pub binding: Binding,
    /// The varying it carries, by index into the inputs or outputs. A vertex program's clip distances are one item,
    /// `clip_distances`, an array of every clip distance output's lanes in order: this is the first of them.
    pub varying: usize,
}

/// The entry point's arguments and result members.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct EntryIo {
    /// One per input varying that is used or has a binding.
    pub arguments: Vec<Item>,
    /// One per output varying.
    pub results: Vec<Item>,
}

/// naga's binding for a target's.
pub(crate) fn binding(b: target::Binding) -> Binding {
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
                InterpMode::Perspective => naga::Interpolation::Perspective,
                InterpMode::Linear => naga::Interpolation::Linear,
                InterpMode::Flat => naga::Interpolation::Flat,
            }),
            sampling: match sampling {
                InterpSampling::Center => None,
                InterpSampling::Centroid => Some(naga::Sampling::Centroid),
                InterpSampling::Sample => Some(naga::Sampling::Sample),
            },
            blend_src: None,
            per_primitive: false,
        },
    }
}

/// The entry point's arguments and result members for a vertex, pixel or compute interface (a compute program's
/// inputs are its invocation ids, built-ins). `Err` says why the module cannot carry it.
pub fn entry_io(interface: &Interface) -> Result<EntryIo, String> {
    let stage = interface.stage;
    if !matches!(stage, Stage::Vertex | Stage::Pixel | Stage::Compute) {
        return Err(format!(
            "{} program's entry point is not lowered yet",
            stage.with_article()
        ));
    }
    let io = target::entry_io(interface)?;
    let item = |i: target::IoItem| -> Result<Item, String> {
        let value = match i.ty {
            IoType::Value(v) => v,
            IoType::ClipDistances(n) => Value {
                scalar: ScalarType::Float,
                width: u8::try_from(n).map_err(|_| format!("{}: {n} clip distances", i.name))?,
            },
            ty => return Err(format!("{}: {ty:?} in an entry point", i.name)),
        };
        Ok(Item {
            varying: *i
                .varyings
                .first()
                .ok_or_else(|| format!("{}: no varying", i.name))?,
            name: i.name,
            value,
            binding: binding(i.binding),
        })
    };
    Ok(EntryIo {
        arguments: io.arguments.into_iter().map(item).collect::<Result<_, _>>()?,
        results: io.results.into_iter().map(item).collect::<Result<_, _>>()?,
    })
}
