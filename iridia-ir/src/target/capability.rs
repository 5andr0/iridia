//! What a program needs of a target beyond WebGPU's core: [`requirements`] names each [`Capability`] once, with the
//! first feature of the program that needs it, so a writer refuses a program its target cannot carry by name before
//! writing it.

use alloc::{format, string::String, vec::Vec};

use crate::interface::{
    InterpSampling, ResourceKind, ScalarType, Stage, StorageTarget, SystemValue, TextureDim,
};
use crate::{Body, Function, Memory, Node, Op, Program, Res, Scalar};

/// A capability of a target that WebGPU's core does not promise everywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    /// 64-bit floats.
    Float64,
    /// 16-bit floats (WebGPU's `shader-f16`).
    Float16,
    /// 16-bit integers.
    Int16,
    /// 64-bit integers.
    Int64,
    /// A pixel program's primitive index (WebGPU's `primitive-index`).
    PrimitiveIndex,
    /// Per-sample shading: the sample index, per-sample interpolation.
    MultisampledShading,
    /// Cube array textures.
    CubeArrayTextures,
    /// A vertex program's clip distances (WebGPU's `clip-distances`).
    ClipDistances,
    /// Subgroup operations and built-ins (WebGPU's `subgroups`).
    Subgroup,
    /// Subgroup operations in a vertex program.
    SubgroupVertexStage,
    /// An array of resources of the class (a resource whose `count` is not 1): naga's binding arrays; WGSL has none.
    BindingArray(ArrayClass),
    /// An element of an array of resources of the class that an invocation picks as its own (Direct3D's
    /// `NonUniformResourceIndex`).
    BindingArrayNonUniform(ArrayClass),
    /// An atomic operation on a storage image's texel (naga's image atomics; WGSL has none).
    TextureAtomic,
}

/// What an array of resources is of, as targets tell their binding arrays apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArrayClass {
    /// Textures, or samplers.
    TexturesAndSamplers,
    /// Constant buffers.
    ConstantBuffers,
    /// Buffers read or written (storage buffers).
    Buffers,
    /// Storage images.
    StorageImages,
}

/// The class of an array of this kind of resource, and what the array is of in words; `None` for pipeline state and
/// acceleration structures, which are not in arrays.
fn array_class(kind: &ResourceKind) -> Option<(ArrayClass, &'static str)> {
    Some(match kind {
        ResourceKind::ConstantBuffer { .. } => (ArrayClass::ConstantBuffers, "constant buffers"),
        ResourceKind::Buffer(_)
        | ResourceKind::Storage {
            target: StorageTarget::Buffer(_),
            ..
        } => (ArrayClass::Buffers, "buffers"),
        ResourceKind::Texture { .. } => (ArrayClass::TexturesAndSamplers, "textures"),
        ResourceKind::Sampler { .. } => (ArrayClass::TexturesAndSamplers, "samplers"),
        ResourceKind::Storage {
            target: StorageTarget::Image { .. },
            ..
        } => (ArrayClass::StorageImages, "storage images"),
        _ => return None,
    })
}

/// Every resource an operation or a store of `f` names, with the element it picks.
fn named(f: &Function) -> Vec<Res> {
    let mut out = Vec::new();
    for v in &f.values {
        match v.op {
            Op::LoadBuffer { resource: r, .. }
            | Op::LoadTexel { texture: r, .. }
            | Op::Query(r, _)
            | Op::Counter(r, _)
            | Op::Atomic {
                memory: Memory::Buffer(r, _) | Memory::Image(r, _),
                ..
            } => out.push(r),
            Op::Sample { texture, sampler, .. } | Op::Gather { texture, sampler, .. } => {
                out.push(texture);
                out.push(sampler);
            }
            _ => {}
        }
    }
    fn stores(nodes: &[Node], out: &mut Vec<Res>) {
        for n in nodes {
            match n {
                Node::StoreBuffer { resource: r, .. } | Node::StoreTexel { image: r, .. } => out.push(*r),
                _ => {}
            }
            for r in n.regions() {
                stores(&r.nodes, out);
            }
        }
    }
    match &f.body {
        Body::Regions(r) => stores(&r.nodes, &mut out),
        Body::Cfg(blocks) => {
            for b in blocks {
                stores(&b.nodes, &mut out);
            }
        }
    }
    out
}

/// One capability a program needs, and the first feature of the program that needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requirement {
    /// The capability.
    pub capability: Capability,
    /// What needs it: a value, a variable, a varying or a resource.
    pub feature: String,
}

fn capability_of(s: Scalar) -> Option<(Capability, &'static str)> {
    match s {
        Scalar::F64 => Some((Capability::Float64, "f64")),
        Scalar::F16 => Some((Capability::Float16, "f16")),
        Scalar::I16 | Scalar::U16 => Some((Capability::Int16, "16-bit integers")),
        Scalar::I64 | Scalar::U64 => Some((Capability::Int64, "64-bit integers")),
        _ => None,
    }
}

/// Every capability `program` needs, each once, with the first feature that needs it. An f64's bits are moved and
/// tested as 64-bit integers (a bitcast to or from 32-bit lanes, a NaN test), so an f64 needs [`Capability::Int64`]
/// too.
pub fn requirements(program: &Program) -> Vec<Requirement> {
    let mut found: Vec<Requirement> = Vec::new();
    let mut need = |capability: Capability, feature: String| {
        if !found.iter().any(|r| r.capability == capability) {
            found.push(Requirement { capability, feature });
        }
    };
    let f = &program.entry;
    for (n, v) in f.values.iter().enumerate() {
        if let Some((c, what)) = capability_of(v.ty.scalar) {
            need(c, format!("value %{n} is {what}"));
        }
        if matches!(v.op, Op::Wave(_)) {
            need(Capability::Subgroup, format!("value %{n} is a wave operation"));
            if program.interface.stage == Stage::Vertex {
                need(
                    Capability::SubgroupVertexStage,
                    format!("value %{n} is a wave operation of a vertex program"),
                );
            }
        }
        if v.ty.scalar == Scalar::F64 {
            need(
                Capability::Int64,
                format!("value %{n} is f64, whose bits the lowering moves as 64-bit integers"),
            );
        }
    }
    for v in &f.vars {
        if let Some((c, what)) = capability_of(v.ty.scalar) {
            need(c, format!("variable {} is {what}", v.name));
        }
    }
    let stage = program.interface.stage;
    let inputs = program.interface.inputs.iter().map(|v| ("input", v));
    let outputs = program.interface.outputs.iter().map(|v| ("output", v));
    for (list, v) in inputs.chain(outputs) {
        let name = format!("{list} {}{}", v.label.name, v.label.index);
        if v.value.scalar == ScalarType::Double {
            need(Capability::Float64, format!("{name} is f64"));
        }
        let pixel_input = stage == Stage::Pixel && list == "input";
        if stage == Stage::Vertex && list == "output" && v.system == Some(SystemValue::ClipDistance) {
            need(Capability::ClipDistances, format!("{name} is a clip distance"));
        }
        match v.system {
            Some(SystemValue::PrimitiveIndex) if pixel_input => {
                need(
                    Capability::PrimitiveIndex,
                    format!("{name} is the primitive index"),
                );
            }
            Some(SystemValue::LaneIndex | SystemValue::LaneCount) if list == "input" => {
                need(
                    Capability::Subgroup,
                    format!("{name} is the lane number or the wave's lane count"),
                );
            }
            Some(SystemValue::SampleIndex) if pixel_input => {
                need(
                    Capability::MultisampledShading,
                    format!("{name} is the sample index"),
                );
            }
            _ => {}
        }
        if pixel_input && v.interp.is_some_and(|i| i.sampling == InterpSampling::Sample) {
            need(
                Capability::MultisampledShading,
                format!("{name} is interpolated per sample"),
            );
        }
    }
    for r in &program.interface.resources {
        if let ResourceKind::Texture {
            dim: TextureDim::Cube,
            arrayed: true,
            ..
        } = r.kind
        {
            need(
                Capability::CubeArrayTextures,
                format!("resource {} is a cube array texture", r.provenance.name),
            );
        }
        if r.is_array()
            && let Some((class, what)) = array_class(&r.kind)
        {
            need(
                Capability::BindingArray(class),
                format!("resource {} is an array of {what}", r.provenance.name),
            );
        }
    }
    for (n, v) in f.values.iter().enumerate() {
        let Op::Atomic {
            memory: Memory::Image(r, _),
            ..
        } = v.op
        else {
            continue;
        };
        if let Some(resource) = program.interface.resources.iter().find(|x| x.id == r.id)
            && let ResourceKind::Storage {
                target: StorageTarget::Image { .. },
                ..
            } = resource.kind
        {
            need(
                Capability::TextureAtomic,
                format!(
                    "value %{n} is an atomic of storage image {}",
                    resource.provenance.name
                ),
            );
        }
    }
    for r in named(f).into_iter().filter(|r| r.nonuniform && r.index.is_some()) {
        let Some(resource) = program
            .interface
            .resources
            .iter()
            .find(|x| x.id == r.id && x.is_array())
        else {
            continue;
        };
        if let Some((class, what)) = array_class(&resource.kind) {
            need(
                Capability::BindingArrayNonUniform(class),
                format!(
                    "resource {}, an array of {what}, is indexed per invocation",
                    resource.provenance.name
                ),
            );
        }
    }
    found
}
