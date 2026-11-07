//! How an entry point carries the interface: an argument per input varying, a member of the result structure `Out`
//! per output varying, each named `<NAME><index>` after its label, with a location (in interface order) or a
//! built-in.

use alloc::{format, string::String, vec::Vec};

use iridia_ir::interface::{
    Interface, InterpMode, InterpSampling, ScalarType, Stage, SystemValue, Value, Varying,
};
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

/// The value the module carries for a varying: the interface's own, except where the pipeline fixes it. A built-in has
/// its API type, and a pixel program's colour target always has four lanes (D3D lets a target element be narrower
/// than its render target, WebGPU does not).
fn carried(stage: Stage, output: bool, v: &Varying) -> Value {
    let (scalar, width) = match v.system {
        Some(SystemValue::ClipPosition | SystemValue::FragCoord) => (ScalarType::Float, 4),
        Some(SystemValue::FrontFacing) => (ScalarType::Bool, 1),
        Some(
            SystemValue::VertexIndex
            | SystemValue::InstanceIndex
            | SystemValue::PrimitiveIndex
            | SystemValue::SampleIndex
            | SystemValue::Coverage
            | SystemValue::GroupIndex,
        ) => (ScalarType::Uint, 1),
        Some(SystemValue::GroupId | SystemValue::GroupThreadId | SystemValue::DispatchThreadId) => {
            (ScalarType::Uint, 3)
        }
        Some(SystemValue::Depth | SystemValue::DepthGreaterEqual | SystemValue::DepthLessEqual) => {
            (ScalarType::Float, 1)
        }
        Some(SystemValue::Target(_)) if stage == Stage::Pixel && output => (v.value.scalar, 4),
        _ => (v.value.scalar, v.value.width),
    };
    Value { scalar, width }
}

/// The built-in a system value is as an input or output of `stage`; `None` for one the module cannot carry.
fn builtin(stage: Stage, output: bool, system: SystemValue) -> Option<naga::BuiltIn> {
    use SystemValue::*;
    Some(match (stage, output, system) {
        (Stage::Vertex, true, ClipPosition) | (Stage::Pixel, false, FragCoord) => {
            naga::BuiltIn::Position { invariant: false }
        }
        (Stage::Pixel, false, FrontFacing) => naga::BuiltIn::FrontFacing,
        (Stage::Vertex, false, VertexIndex) => naga::BuiltIn::VertexIndex,
        (Stage::Vertex, false, InstanceIndex) => naga::BuiltIn::InstanceIndex,
        (Stage::Pixel, false, PrimitiveIndex) => naga::BuiltIn::PrimitiveIndex,
        (Stage::Pixel, false, SampleIndex) => naga::BuiltIn::SampleIndex,
        (Stage::Pixel, _, Coverage) => naga::BuiltIn::SampleMask,
        // naga has no conservative-depth variants: the depth hints are not carried
        (Stage::Pixel, true, Depth | DepthGreaterEqual | DepthLessEqual) => naga::BuiltIn::FragDepth,
        (Stage::Compute, false, GroupId) => naga::BuiltIn::WorkGroupId,
        (Stage::Compute, false, GroupThreadId) => naga::BuiltIn::LocalInvocationId,
        (Stage::Compute, false, DispatchThreadId) => naga::BuiltIn::GlobalInvocationId,
        (Stage::Compute, false, GroupIndex) => naga::BuiltIn::LocalInvocationIndex,
        (Stage::Compute | Stage::Pixel, false, LaneIndex) => naga::BuiltIn::SubgroupInvocationId,
        (Stage::Compute | Stage::Pixel, false, LaneCount) => naga::BuiltIn::SubgroupSize,
        _ => return None,
    })
}

/// The location binding of a plain varying. A pixel input takes the interface's interpolation; a value that is not a
/// float is always flat, and a flat value has no sampling. A vertex output states the perspective default (or flat,
/// for a value that is not a float) its consumer is matched against.
fn location(stage: Stage, output: bool, v: &Varying, at: u32) -> Binding {
    let float = v.value.scalar == ScalarType::Float;
    let (interpolation, sampling) = match (stage, output) {
        (Stage::Pixel, false) => {
            let declared = v.interp.map(|i| {
                (
                    match i.mode {
                        InterpMode::Perspective => naga::Interpolation::Perspective,
                        InterpMode::Linear => naga::Interpolation::Linear,
                        InterpMode::Flat => naga::Interpolation::Flat,
                    },
                    match i.sampling {
                        InterpSampling::Center => None,
                        InterpSampling::Centroid => Some(naga::Sampling::Centroid),
                        InterpSampling::Sample => Some(naga::Sampling::Sample),
                    },
                )
            });
            match declared {
                Some((mode, sampling)) if float && mode != naga::Interpolation::Flat => {
                    (Some(mode), sampling)
                }
                Some(_) => (Some(naga::Interpolation::Flat), None),
                None if float => (Some(naga::Interpolation::Perspective), None),
                None => (Some(naga::Interpolation::Flat), None),
            }
        }
        (Stage::Vertex, true) => (
            Some(if float {
                naga::Interpolation::Perspective
            } else {
                naga::Interpolation::Flat
            }),
            None,
        ),
        _ => (None, None),
    };
    Binding::Location {
        location: at,
        interpolation,
        sampling,
        blend_src: None,
        per_primitive: false,
    }
}

/// The entry point's arguments and result members for a vertex, pixel or compute interface (a compute program's
/// inputs are its invocation ids, built-ins). `Err` says why the module cannot carry it.
pub fn entry_io(interface: &Interface) -> Result<EntryIo, String> {
    let stage = interface.stage;
    if !matches!(stage, Stage::Vertex | Stage::Pixel | Stage::Compute) {
        return Err(format!("a {stage:?} program's entry point is not lowered yet"));
    }
    let mut io = EntryIo::default();
    for output in [false, true] {
        let list = if output {
            &interface.outputs
        } else {
            &interface.inputs
        };
        let mut next = 0u32;
        let mut items: Vec<Item> = Vec::new();
        let mut clip: Option<usize> = None;
        for (i, v) in list.iter().enumerate() {
            let name = format!("{}{}", v.label.name, v.label.index);
            let direction = if output { "output" } else { "input" };
            let mut plain = |v: &Varying| {
                let at = next;
                next = at.saturating_add(1);
                location(stage, output, v, at)
            };
            let binding = match v.system {
                Some(SystemValue::Target(n)) if stage == Stage::Pixel && output => Binding::Location {
                    location: n,
                    interpolation: None,
                    sampling: None,
                    blend_src: None,
                    per_primitive: false,
                },
                // a vertex program's clip distances are one array of floats, every clip distance output's lanes in
                // order, at the place of the first
                Some(SystemValue::ClipDistance) if stage == Stage::Vertex && output => {
                    let width = v.value.width;
                    match clip.and_then(|at| items.get_mut(at)) {
                        Some(item) => item.value.width = item.value.width.saturating_add(width),
                        None => {
                            clip = Some(items.len());
                            items.push(Item {
                                name: "clip_distances".into(),
                                value: Value {
                                    scalar: ScalarType::Float,
                                    width,
                                },
                                binding: Binding::BuiltIn(naga::BuiltIn::ClipDistances),
                                varying: i,
                            });
                        }
                    }
                    continue;
                }
                // WebGPU has no cull distance: a vertex program's value travels as a varying, and culling the primitive
                // whose vertices are all negative is the pipeline's
                Some(SystemValue::CullDistance) if stage == Stage::Vertex && output => plain(v),
                // a clip or cull distance a pixel program reads has no built-in; one it does not read is not carried
                Some(SystemValue::ClipDistance | SystemValue::CullDistance) if !v.used => continue,
                Some(SystemValue::ClipDistance | SystemValue::CullDistance) => {
                    return Err(format!(
                        "{name}: a clip or cull distance as a {stage:?} {direction} has no built-in"
                    ));
                }
                Some(SystemValue::RenderTargetArrayIndex | SystemValue::ViewportArrayIndex) => plain(v),
                Some(system) => match builtin(stage, output, system) {
                    Some(b) => Binding::BuiltIn(b),
                    None if !v.used => continue,
                    None => {
                        return Err(format!(
                            "{name}: {system:?} as a {stage:?} {direction} has no built-in binding"
                        ));
                    }
                },
                None => plain(v),
            };
            items.push(Item {
                name,
                value: carried(stage, output, v),
                binding,
                varying: i,
            });
        }
        if output {
            io.results = items;
        } else {
            io.arguments = items;
        }
    }
    Ok(io)
}
