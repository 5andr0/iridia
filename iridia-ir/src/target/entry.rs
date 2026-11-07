//! How an entry point carries a program's interface: an argument per input varying, a member of the result structure
//! per output varying, each named `<NAME><index>` after its label, at a location (numbered in interface order) or a
//! built-in.
//!
//! A system value the pipeline supplies or consumes is its stage's built-in; every other varying has a location, 0,
//! 1, ... in interface order (built-ins use none), and a pixel input carries the interpolation the interface states. A
//! pixel program's `Target(n)` output is location `n`. A vertex program's clip distances are one member, an array of
//! every clip distance output's lanes. WebGPU has no layer, viewport index or cull distance: those travel as plain
//! varyings.

use alloc::{format, string::String, vec, vec::Vec};

use crate::interface::{
    Interface, InterpMode, InterpSampling, Label, ScalarType, Stage, StageFacts, SystemValue, Value, Varying,
};

/// A value the pipeline supplies to an entry point or takes from it, by WGSL's name for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltIn {
    /// `position`: a vertex program's clip-space position, a pixel program's window position.
    Position,
    /// `front_facing`.
    FrontFacing,
    /// `vertex_index`.
    VertexIndex,
    /// `instance_index`.
    InstanceIndex,
    /// `primitive_index`.
    PrimitiveIndex,
    /// `sample_index`.
    SampleIndex,
    /// `sample_mask`: the coverage a pixel program reads or writes.
    SampleMask,
    /// `frag_depth`.
    FragDepth,
    /// `workgroup_id`.
    WorkgroupId,
    /// `local_invocation_id`.
    LocalInvocationId,
    /// `global_invocation_id`.
    GlobalInvocationId,
    /// `local_invocation_index`.
    LocalInvocationIndex,
    /// `subgroup_invocation_id`.
    SubgroupInvocationId,
    /// `subgroup_size`.
    SubgroupSize,
    /// `clip_distances`: an array of floats.
    ClipDistances,
}

/// Where an argument or result member is bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// A value the pipeline supplies or takes.
    BuiltIn(BuiltIn),
    /// A user location.
    Location {
        /// The location's number.
        location: u32,
        /// How the value is interpolated: stated for a pixel program's inputs and a vertex program's outputs (flat for
        /// a value that is not a float), `None` elsewhere (a pixel program's targets).
        interpolation: Option<InterpMode>,
        /// Where in the pixel it is interpolated; [`InterpSampling::Center`], the default, for a flat value and where
        /// no interpolation is stated.
        sampling: InterpSampling,
    },
}

/// The type of an argument or result member.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoType {
    /// A scalar (width 1) or vector.
    Value(Value),
    /// An array of this many floats: the clip distances of a vertex stage, all of its `SV_ClipDistance` outputs
    /// together.
    ClipDistances(u32),
    /// An array of this value: a hull or domain program's control points (one per point), a hull program's outputs.
    Array(Value, u32),
    /// A pointer, in function memory, to a structure the module defines: a ray tracing program's payload or hit
    /// attributes, whose layout is the shader's own.
    Memory,
}

/// One argument or result member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoItem {
    /// `<NAME><index>` after the label, or `clip_distances`.
    pub name: String,
    /// The type of the member.
    pub ty: IoType,
    /// Its location or built-in.
    pub binding: Binding,
    /// Which varyings of the interface's list (inputs for an argument, outputs for a result) it carries: one, or every
    /// clip distance.
    pub varyings: Vec<usize>,
}

/// The entry point's arguments (one per input varying) and result members (one per output varying, the clip distances
/// being one together), in interface order. A varying nothing reads or writes (`used` false) that no binding exists
/// for is left out: the interface lists every element of the container's signature, the module carries what it can.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct EntryIo {
    /// One per input varying that is used or has a binding.
    pub arguments: Vec<IoItem>,
    /// One per output varying (the clip distances together), in interface order.
    pub results: Vec<IoItem>,
}

/// The name a varying has as an argument or result member: `<NAME><index>`.
pub fn varying_name(label: &Label) -> String {
    format!("{}{}", label.name, label.index)
}

/// The value the module carries for a varying: the interface's own, except where the pipeline fixes it. A built-in has
/// its API type (a front-facing flag is a bool, the position four floats, an index a u32) and a pixel stage's colour
/// output always has four lanes (D3D lets a target element be narrower than its render target, WebGPU does not).
pub fn carried(stage: Stage, output: bool, v: &Varying) -> Value {
    let (scalar, width) = match v.system {
        Some(SystemValue::ClipPosition | SystemValue::FragCoord) => (ScalarType::Float, 4),
        Some(SystemValue::FrontFacing) => (ScalarType::Bool, 1),
        Some(
            SystemValue::VertexIndex
            | SystemValue::InstanceIndex
            | SystemValue::PrimitiveIndex
            | SystemValue::SampleIndex
            | SystemValue::Coverage,
        ) => (ScalarType::Uint, 1),
        Some(SystemValue::GroupId | SystemValue::GroupThreadId | SystemValue::DispatchThreadId) => {
            (ScalarType::Uint, 3)
        }
        Some(SystemValue::GroupIndex | SystemValue::LaneIndex | SystemValue::LaneCount) => {
            (ScalarType::Uint, 1)
        }
        Some(SystemValue::DispatchRaysIndex | SystemValue::DispatchRaysDimensions) => (ScalarType::Uint, 3),
        Some(
            SystemValue::WorldRayOrigin
            | SystemValue::WorldRayDirection
            | SystemValue::ObjectRayOrigin
            | SystemValue::ObjectRayDirection,
        ) => (ScalarType::Float, 3),
        Some(SystemValue::RayTMin | SystemValue::RayTCurrent) => (ScalarType::Float, 1),
        Some(SystemValue::ObjectToWorld | SystemValue::WorldToObject) => (ScalarType::Float, 4),
        Some(
            SystemValue::RayFlags
            | SystemValue::InstanceId
            | SystemValue::GeometryIndex
            | SystemValue::HitKind,
        ) => (ScalarType::Uint, 1),
        Some(SystemValue::Depth | SystemValue::DepthGreaterEqual | SystemValue::DepthLessEqual) => {
            (ScalarType::Float, 1)
        }
        Some(SystemValue::Target(_)) if stage == Stage::Pixel && output => (v.value.scalar, 4),
        _ => (v.value.scalar, v.value.width),
    };
    Value { scalar, width }
}

/// The built-in a system value is as an input or output of `stage`; `None` for one the module cannot carry.
fn builtin(stage: Stage, output: bool, system: SystemValue) -> Option<BuiltIn> {
    use SystemValue::*;
    let workgroup = matches!(stage, Stage::Compute | Stage::Mesh | Stage::Amplification);
    Some(match (stage, output, system) {
        (Stage::Vertex, true, ClipPosition) | (Stage::Pixel, false, FragCoord) => BuiltIn::Position,
        (Stage::Pixel, false, FrontFacing) => BuiltIn::FrontFacing,
        (Stage::Vertex, false, VertexIndex) => BuiltIn::VertexIndex,
        (Stage::Vertex, false, InstanceIndex) => BuiltIn::InstanceIndex,
        (Stage::Pixel, false, PrimitiveIndex) => BuiltIn::PrimitiveIndex,
        (Stage::Pixel, false, SampleIndex) => BuiltIn::SampleIndex,
        (Stage::Pixel, _, Coverage) => BuiltIn::SampleMask,
        (_, false, GroupId) if workgroup => BuiltIn::WorkgroupId,
        (_, false, GroupThreadId) if workgroup => BuiltIn::LocalInvocationId,
        (_, false, DispatchThreadId) if workgroup => BuiltIn::GlobalInvocationId,
        (_, false, GroupIndex) if workgroup => BuiltIn::LocalInvocationIndex,
        (Stage::Compute | Stage::Pixel, false, LaneIndex) => BuiltIn::SubgroupInvocationId,
        (Stage::Compute | Stage::Pixel, false, LaneCount) => BuiltIn::SubgroupSize,
        // WGSL has no conservative-depth variants: the depth hints are not carried
        (Stage::Pixel, true, Depth | DepthGreaterEqual | DepthLessEqual) => BuiltIn::FragDepth,
        _ => return None,
    })
}

/// The location binding of a plain varying. A pixel input takes the interface's interpolation; a value that is not a
/// float is always flat, and a flat value has no sampling (naga and WebGPU both insist). A vertex output states the
/// perspective default (or flat, for a value that is not a float) that its consumer's interface is matched against.
fn location(stage: Stage, output: bool, v: &Varying, at: u32) -> Binding {
    let float = v.value.scalar == ScalarType::Float;
    let (interpolation, sampling) = match (stage, output) {
        (Stage::Pixel, false) => match v.interp {
            Some(i) if float && i.mode != InterpMode::Flat => (Some(i.mode), i.sampling),
            Some(_) => (Some(InterpMode::Flat), InterpSampling::Center),
            None if float => (Some(InterpMode::Perspective), InterpSampling::Center),
            None => (Some(InterpMode::Flat), InterpSampling::Center),
        },
        (Stage::Vertex, true) => (
            Some(if float {
                InterpMode::Perspective
            } else {
                InterpMode::Flat
            }),
            InterpSampling::Center,
        ),
        _ => (None, InterpSampling::Center),
    };
    Binding::Location {
        location: at,
        interpolation,
        sampling,
    }
}

/// The entry point's arguments and result members for `interface`. A mesh or amplification program takes a compute
/// program's ids and returns nothing (what a mesh program writes is its output arrays, whose vertex members are bound
/// as a vertex program's outputs are). `Err` says why the module cannot carry it: a varying nothing reads or writes is
/// left out where no binding exists for it, and one that is used is refused, naming it.
pub fn entry_io(interface: &Interface) -> Result<EntryIo, String> {
    let stage = interface.stage;
    if matches!(stage, Stage::Mesh | Stage::Amplification) {
        let mut ids = interface.clone();
        ids.stage = Stage::Compute;
        ids.outputs.clear();
        ids.facts = StageFacts::None;
        return Ok(EntryIo {
            arguments: entry_io(&ids)?.arguments,
            results: Vec::new(),
        });
    }
    let mut io = EntryIo::default();
    for output in [false, true] {
        let (list, items) = if output {
            (&interface.outputs, &mut io.results)
        } else {
            (&interface.inputs, &mut io.arguments)
        };
        let direction = if output { "output" } else { "input" };
        let mut next_location = 0u32;
        let mut plain = |v: &Varying| {
            let at = next_location;
            next_location = at.saturating_add(1);
            location(stage, output, v, at)
        };
        let mut clip: Option<usize> = None;
        for (i, v) in list.iter().enumerate() {
            let name = varying_name(&v.label);
            let binding = match v.system {
                Some(SystemValue::Target(n)) if stage == Stage::Pixel && output => Binding::Location {
                    location: n,
                    interpolation: None,
                    sampling: InterpSampling::Center,
                },
                // a vertex program's clip distances are one array of floats, every clip distance output's lanes in
                // order, at the place of the first
                Some(SystemValue::ClipDistance) if stage == Stage::Vertex && output => {
                    let width = u32::from(v.value.width);
                    match clip {
                        Some(at) => {
                            let item = items
                                .get_mut(at)
                                .ok_or_else(|| format!("{name}: the clip distances item is missing"))?;
                            item.varyings.push(i);
                            let IoType::ClipDistances(n) = &mut item.ty else {
                                return Err(format!("{name}: the clip distances item is not clip distances"));
                            };
                            *n = n.saturating_add(width);
                        }
                        None => {
                            clip = Some(items.len());
                            items.push(IoItem {
                                name: "clip_distances".into(),
                                ty: IoType::ClipDistances(width),
                                binding: Binding::BuiltIn(BuiltIn::ClipDistances),
                                varyings: vec![i],
                            });
                        }
                    }
                    continue;
                }
                // WebGPU has no cull distance: a vertex program's value travels as a varying, and culling the primitive
                // whose vertices are all negative is the pipeline's
                Some(SystemValue::CullDistance) if stage == Stage::Vertex && output => plain(v),
                // the interface lists what the container declares; an element nothing reads or writes, and the module
                // cannot bind, is not carried
                Some(SystemValue::ClipDistance | SystemValue::CullDistance) if !v.used => continue,
                Some(SystemValue::ClipDistance | SystemValue::CullDistance) => {
                    return Err(format!(
                        "{name}: a clip or cull distance as {} {direction} has no built-in",
                        stage.with_article()
                    ));
                }
                // WebGPU has no layer or viewport index to read or write: the value travels as a flat varying, and
                // routing a primitive to the layer it names is the pipeline's
                Some(SystemValue::RenderTargetArrayIndex | SystemValue::ViewportArrayIndex) => plain(v),
                Some(system) => match builtin(stage, output, system) {
                    Some(b) => Binding::BuiltIn(b),
                    None if !v.used => continue,
                    None => {
                        return Err(format!(
                            "{name}: {system:?} as {} {direction} has no built-in binding",
                            stage.with_article()
                        ));
                    }
                },
                None => plain(v),
            };
            items.push(IoItem {
                name,
                ty: IoType::Value(carried(stage, output, v)),
                binding,
                varyings: vec![i],
            });
        }
    }
    Ok(io)
}
