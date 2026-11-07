//! Target profiles and the preflight: what a program needs of naga's validator, checked against what a target allows
//! before the program is lowered.
//!
//! [`requirements`] names every capability a program's lowering needs and the feature of the program that needs it;
//! [`preflight`] refuses a program that needs one its [`Profile`] does not allow, naming the feature. A module is then
//! validated with the profile's capabilities and nothing more, so a module the preflight accepted and naga rejects is a
//! lowering bug, and says so.

use alloc::{format, string::String, vec::Vec};

use iridia_ir::Program;
use iridia_ir::target::{self, ArrayClass, Bindings, Capability};
use naga::valid::Capabilities;

use crate::Error;

/// Which `enable` directives the WGSL states (see [`crate::wgsl`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Directives {
    /// naga's: the ones naga's WGSL writer states. naga 30 has no `enable subgroups;`: its writer leaves it out and its
    /// reader refuses it, and uses the subgroup operations without it.
    Naga,
    /// The WGSL standard's: naga's, and `enable subgroups;` where the program uses a subgroup operation or built-in,
    /// which a browser requires.
    Standard,
}

/// What a target lets a module use: the naga capabilities its validator runs with, how it binds resources and which
/// directives the WGSL states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    /// The target's name, as refusals spell it.
    pub name: &'static str,
    /// The capabilities a module may use.
    pub capabilities: Capabilities,
    /// How [`crate::wgsl`] binds the program's resources.
    pub bindings: Bindings,
    /// Which `enable` directives [`crate::wgsl`] states.
    pub directives: Directives,
}

impl Profile {
    /// naga's own dialect (its WGSL writer, its other writers): every capability this crate's lowering can need.
    pub const NAGA: Profile = Profile {
        name: "naga",
        capabilities: Capabilities::FLOAT64
            .union(Capabilities::SHADER_FLOAT16)
            .union(Capabilities::SHADER_INT16)
            .union(Capabilities::SHADER_INT64)
            .union(Capabilities::PRIMITIVE_INDEX)
            .union(Capabilities::MULTISAMPLED_SHADING)
            .union(Capabilities::CUBE_ARRAY_TEXTURES)
            .union(Capabilities::CLIP_DISTANCES)
            .union(Capabilities::SUBGROUP)
            .union(Capabilities::SUBGROUP_VERTEX_STAGE)
            .union(Capabilities::TEXTURE_ATOMIC)
            .union(BINDING_ARRAYS),
        bindings: Bindings::D3d,
        directives: Directives::Naga,
    };

    /// WebGPU's core WGSL: per-sample shading (`sample_index`, `@interpolate(..., sample)`) and cube array textures
    /// are core; `f16` (`shader-f16`), `primitive_index` (`primitive-index`) and the subgroup operations
    /// (`subgroups`, `Capabilities::SUBGROUP`) are optional device features a consumer adds with [`Profile::with`]
    /// when its device has them; WGSL has no 64-bit or 16-bit integers, no `f64`, no arrays of resources (naga's
    /// binding arrays, `enable wgpu_binding_array;` in its dialect) and no texel atomics (naga's `textureAtomic*`).
    /// Resources bind inside WebGPU's default limits, and the WGSL states the standard's directives.
    pub const BROWSER: Profile = Profile {
        name: "browser WGSL",
        capabilities: Capabilities::MULTISAMPLED_SHADING.union(Capabilities::CUBE_ARRAY_TEXTURES),
        bindings: Bindings::WebGpu,
        directives: Directives::Standard,
    };

    /// This profile with `capabilities` allowed too.
    #[must_use]
    pub const fn with(self, capabilities: Capabilities) -> Profile {
        Profile {
            name: self.name,
            capabilities: self.capabilities.union(capabilities),
            bindings: self.bindings,
            directives: self.directives,
        }
    }
}

/// naga's binding arrays, of each class of resource, indexed alike or per invocation: an array of resources is one.
const BINDING_ARRAYS: Capabilities = Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
    .union(Capabilities::BUFFER_BINDING_ARRAY)
    .union(Capabilities::STORAGE_TEXTURE_BINDING_ARRAY)
    .union(Capabilities::STORAGE_BUFFER_BINDING_ARRAY)
    .union(Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING)
    .union(Capabilities::BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING)
    .union(Capabilities::STORAGE_TEXTURE_BINDING_ARRAY_NON_UNIFORM_INDEXING)
    .union(Capabilities::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING);

/// One capability a program's lowering needs, and the first feature of the program that needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requirement {
    /// The capability.
    pub capability: Capabilities,
    /// What needs it: a value, a variable, a varying or a resource.
    pub feature: String,
}

/// naga's capability for a target's.
fn capability(c: Capability) -> Capabilities {
    match c {
        Capability::Float64 => Capabilities::FLOAT64,
        Capability::Float16 => Capabilities::SHADER_FLOAT16,
        Capability::Int16 => Capabilities::SHADER_INT16,
        Capability::Int64 => Capabilities::SHADER_INT64,
        Capability::PrimitiveIndex => Capabilities::PRIMITIVE_INDEX,
        Capability::MultisampledShading => Capabilities::MULTISAMPLED_SHADING,
        Capability::CubeArrayTextures => Capabilities::CUBE_ARRAY_TEXTURES,
        Capability::ClipDistances => Capabilities::CLIP_DISTANCES,
        Capability::Subgroup => Capabilities::SUBGROUP,
        Capability::SubgroupVertexStage => Capabilities::SUBGROUP_VERTEX_STAGE,
        Capability::BindingArray(class) => match class {
            ArrayClass::TexturesAndSamplers => Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY,
            ArrayClass::ConstantBuffers => Capabilities::BUFFER_BINDING_ARRAY,
            ArrayClass::Buffers => Capabilities::STORAGE_BUFFER_BINDING_ARRAY,
            ArrayClass::StorageImages => Capabilities::STORAGE_TEXTURE_BINDING_ARRAY,
        },
        Capability::TextureAtomic => Capabilities::TEXTURE_ATOMIC,
        Capability::BindingArrayNonUniform(class) => match class {
            ArrayClass::TexturesAndSamplers => {
                Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING
            }
            ArrayClass::ConstantBuffers => Capabilities::BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING,
            ArrayClass::Buffers => Capabilities::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING,
            ArrayClass::StorageImages => Capabilities::STORAGE_TEXTURE_BINDING_ARRAY_NON_UNIFORM_INDEXING,
        },
    }
}

/// Every capability `program`'s lowering needs, each once, with the first feature that needs it
/// ([`iridia_ir::target::requirements`], as naga's capabilities).
pub fn requirements(program: &Program) -> Vec<Requirement> {
    target::requirements(program)
        .into_iter()
        .map(|r| Requirement {
            capability: capability(r.capability),
            feature: r.feature,
        })
        .collect()
}

/// Refuse `program` if its lowering needs a capability `profile` does not allow, naming the first such feature.
pub fn preflight(program: &Program, profile: &Profile) -> Result<(), Error> {
    match requirements(program)
        .into_iter()
        .find(|r| !profile.capabilities.contains(r.capability))
    {
        Some(r) => Err(Error(format!(
            "{} cannot carry this program: {} (naga capability {:?})",
            profile.name, r.feature, r.capability
        ))),
        None => Ok(()),
    }
}
