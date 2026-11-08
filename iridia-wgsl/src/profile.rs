//! Target profiles and the preflight: what a program needs of a WGSL target, checked before any text is written.
//!
//! [`iridia_ir::target::requirements`] names every capability a program needs and the feature of the program that needs
//! it; [`preflight`] refuses a program that needs one its [`Profile`] does not allow, naming the feature.

use alloc::{format, vec::Vec};

use iridia_ir::Program;
use iridia_ir::target::{ArrayClass, Bindings, Capability, Requirement, requirements};

use crate::Error;

/// A set of [`Capability`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities(u32);

const fn bit(c: Capability) -> u32 {
    match c {
        Capability::Float64 => 1,
        Capability::Float16 => 1 << 1,
        Capability::Int16 => 1 << 2,
        Capability::Int64 => 1 << 3,
        Capability::PrimitiveIndex => 1 << 4,
        Capability::MultisampledShading => 1 << 5,
        Capability::CubeArrayTextures => 1 << 6,
        Capability::ClipDistances => 1 << 7,
        Capability::Subgroup => 1 << 8,
        Capability::SubgroupVertexStage => 1 << 9,
        Capability::BindingArray(ArrayClass::TexturesAndSamplers) => 1 << 10,
        Capability::BindingArray(ArrayClass::ConstantBuffers) => 1 << 11,
        Capability::BindingArray(ArrayClass::Buffers) => 1 << 12,
        Capability::BindingArray(ArrayClass::StorageImages) => 1 << 13,
        Capability::BindingArrayNonUniform(ArrayClass::TexturesAndSamplers) => 1 << 14,
        Capability::BindingArrayNonUniform(ArrayClass::ConstantBuffers) => 1 << 15,
        Capability::BindingArrayNonUniform(ArrayClass::Buffers) => 1 << 16,
        Capability::BindingArrayNonUniform(ArrayClass::StorageImages) => 1 << 17,
        Capability::TextureAtomic => 1 << 18,
        Capability::RayQuery => 1 << 19,
    }
}

impl Capabilities {
    /// None.
    pub const NONE: Capabilities = Capabilities(0);

    /// The set of these capabilities.
    pub const fn of(list: &[Capability]) -> Capabilities {
        let mut bits = 0;
        let mut rest = list;
        while let [first, tail @ ..] = rest {
            bits |= bit(*first);
            rest = tail;
        }
        Capabilities(bits)
    }

    /// Whether `c` is in the set.
    pub const fn contains(self, c: Capability) -> bool {
        self.0 & bit(c) != 0
    }

    /// This set and `other`'s capabilities.
    #[must_use]
    pub const fn with(self, other: Capabilities) -> Capabilities {
        Capabilities(self.0 | other.0)
    }
}

/// Which `enable` directives the WGSL states beyond the ones a type or a binding makes necessary (`f16`,
/// `clip_distances`, `primitive_index`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Directives {
    /// naga's: naga 30 has no `enable subgroups;`: its writer leaves it out and its reader refuses it, and it takes the
    /// subgroup operations without it.
    Naga,
    /// The WGSL standard's: `enable subgroups;` where the program uses a subgroup operation or built-in, which a browser
    /// requires.
    Standard,
}

/// What a target lets a text use: the capabilities, how resources bind and which directives the text states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    /// The target's name, as refusals spell it.
    pub name: &'static str,
    /// The capabilities a text may use.
    pub capabilities: Capabilities,
    /// How [`crate::wgsl`] binds the program's resources.
    pub bindings: Bindings,
    /// Which `enable` directives [`crate::wgsl`] states.
    pub directives: Directives,
}

impl Profile {
    /// naga's own dialect (the one its WGSL writer writes and its reader reads): every capability the writer can need,
    /// 64-bit floats and integers and 16-bit integers included.
    pub const NAGA: Profile = Profile {
        name: "naga",
        capabilities: Capabilities::of(&[
            Capability::Float64,
            Capability::Float16,
            Capability::Int16,
            Capability::Int64,
            Capability::PrimitiveIndex,
            Capability::MultisampledShading,
            Capability::CubeArrayTextures,
            Capability::ClipDistances,
            Capability::Subgroup,
            Capability::SubgroupVertexStage,
            Capability::BindingArray(ArrayClass::TexturesAndSamplers),
            Capability::BindingArray(ArrayClass::ConstantBuffers),
            Capability::BindingArray(ArrayClass::Buffers),
            Capability::BindingArray(ArrayClass::StorageImages),
            Capability::BindingArrayNonUniform(ArrayClass::TexturesAndSamplers),
            Capability::BindingArrayNonUniform(ArrayClass::ConstantBuffers),
            Capability::BindingArrayNonUniform(ArrayClass::Buffers),
            Capability::BindingArrayNonUniform(ArrayClass::StorageImages),
            Capability::TextureAtomic,
            Capability::RayQuery,
        ]),
        bindings: Bindings::D3d,
        directives: Directives::Naga,
    };

    /// WebGPU's core WGSL: per-sample shading and cube array textures are core; `f16` (`shader-f16`), `primitive_index`
    /// (`primitive-index`), `clip_distances` and the subgroup operations (`subgroups`) are optional device features a
    /// consumer adds with [`Profile::with`] when its device has them; WGSL has no 64-bit or 16-bit integers and no
    /// `f64`, no arrays of resources (naga's binding arrays, `enable wgpu_binding_array;` in its dialect) and no texel
    /// atomics (naga's `textureAtomic*`). Resources bind inside WebGPU's default limits, and the text states the standard's
    /// directives.
    pub const BROWSER: Profile = Profile {
        name: "browser WGSL",
        capabilities: Capabilities::of(&[Capability::MultisampledShading, Capability::CubeArrayTextures]),
        bindings: Bindings::WebGpu,
        directives: Directives::Standard,
    };

    /// This profile with `capabilities` allowed too.
    #[must_use]
    pub const fn with(self, capabilities: Capabilities) -> Profile {
        Profile {
            name: self.name,
            capabilities: self.capabilities.with(capabilities),
            bindings: self.bindings,
            directives: self.directives,
        }
    }
}

/// Every capability `program` needs, each once, with the first feature that needs it.
pub fn needs(program: &Program) -> Vec<Requirement> {
    requirements(program)
}

/// Refuse `program` if it needs a capability `profile` does not allow, naming the first such feature.
pub fn preflight(program: &Program, profile: &Profile) -> Result<(), Error> {
    let needs = requirements(program);
    // wgpu's `wgpu_ray_query` extension is native only: a browser's WGSL, whatever it is allowed, has no ray queries
    if profile.directives == Directives::Standard
        && let Some(r) = needs.iter().find(|r| r.capability == Capability::RayQuery)
    {
        return Err(Error(format!(
            "{} has no ray queries: {} (capability {:?})",
            profile.name, r.feature, r.capability
        )));
    }
    match needs
        .into_iter()
        .find(|r| !profile.capabilities.contains(r.capability))
    {
        Some(r) => Err(Error(format!(
            "{} cannot carry this program: {} (capability {:?})",
            profile.name, r.feature, r.capability
        ))),
        None => Ok(()),
    }
}
