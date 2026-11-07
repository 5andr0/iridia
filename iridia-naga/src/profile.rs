//! Target profiles and the preflight: what a program needs of naga's validator, checked against what a target allows
//! before the program is lowered.
//!
//! [`requirements`] names every capability a program's lowering needs and the feature of the program that needs it;
//! [`preflight`] refuses a program that needs one its [`Profile`] does not allow, naming the feature. A module is then
//! validated with the profile's capabilities and nothing more, so a module the preflight accepted and naga rejects is a
//! lowering bug, and says so.

use alloc::{format, string::String, vec::Vec};

use iridia_ir::interface::{InterpSampling, ResourceKind, ScalarType, Stage, SystemValue, TextureDim};
use iridia_ir::{Program, Scalar};
use naga::valid::Capabilities;

use crate::Error;

/// What a target lets a module use: the naga capabilities its validator runs with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    /// The target's name, as refusals spell it.
    pub name: &'static str,
    /// The capabilities a module may use.
    pub capabilities: Capabilities,
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
            .union(Capabilities::CLIP_DISTANCES),
    };

    /// WebGPU's core WGSL: per-sample shading (`sample_index`, `@interpolate(..., sample)`) and cube array textures
    /// are core; `f16` (`shader-f16`) and `primitive_index` (`primitive-index`) are optional device features a
    /// consumer adds with [`Profile::with`] when its device has them; WGSL has no 64-bit or 16-bit integers and no
    /// `f64`.
    pub const BROWSER: Profile = Profile {
        name: "browser WGSL",
        capabilities: Capabilities::MULTISAMPLED_SHADING.union(Capabilities::CUBE_ARRAY_TEXTURES),
    };

    /// This profile with `capabilities` allowed too.
    #[must_use]
    pub const fn with(self, capabilities: Capabilities) -> Profile {
        Profile {
            name: self.name,
            capabilities: self.capabilities.union(capabilities),
        }
    }
}

/// One capability a program's lowering needs, and the first feature of the program that needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requirement {
    /// The capability.
    pub capability: Capabilities,
    /// What needs it: a value, a variable, a varying or a resource.
    pub feature: String,
}

fn capability_of(s: Scalar) -> Option<(Capabilities, &'static str)> {
    match s {
        Scalar::F64 => Some((Capabilities::FLOAT64, "f64")),
        Scalar::F16 => Some((Capabilities::SHADER_FLOAT16, "f16")),
        Scalar::I16 | Scalar::U16 => Some((Capabilities::SHADER_INT16, "16-bit integers")),
        Scalar::I64 | Scalar::U64 => Some((Capabilities::SHADER_INT64, "64-bit integers")),
        _ => None,
    }
}

/// Every capability `program`'s lowering needs, each once, with the first feature that needs it.
pub fn requirements(program: &Program) -> Vec<Requirement> {
    let mut found: Vec<Requirement> = Vec::new();
    let mut need = |capability: Capabilities, feature: String| {
        if !found.iter().any(|r| r.capability == capability) {
            found.push(Requirement { capability, feature });
        }
    };
    let f = &program.entry;
    for (n, v) in f.values.iter().enumerate() {
        if let Some((c, what)) = capability_of(v.ty.scalar) {
            need(c, format!("value %{n} is {what}"));
        }
        // an f64's bits are moved and tested as 64-bit integers (a bitcast to or from 32-bit lanes, a NaN test)
        if v.ty.scalar == Scalar::F64 {
            need(
                Capabilities::SHADER_INT64,
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
            need(Capabilities::FLOAT64, format!("{name} is f64"));
        }
        let pixel_input = stage == Stage::Pixel && list == "input";
        if stage == Stage::Vertex && list == "output" && v.system == Some(SystemValue::ClipDistance) {
            need(Capabilities::CLIP_DISTANCES, format!("{name} is a clip distance"));
        }
        match v.system {
            Some(SystemValue::PrimitiveIndex) if pixel_input => {
                need(
                    Capabilities::PRIMITIVE_INDEX,
                    format!("{name} is the primitive index"),
                );
            }
            Some(SystemValue::SampleIndex) if pixel_input => {
                need(
                    Capabilities::MULTISAMPLED_SHADING,
                    format!("{name} is the sample index"),
                );
            }
            _ => {}
        }
        if pixel_input && v.interp.is_some_and(|i| i.sampling == InterpSampling::Sample) {
            need(
                Capabilities::MULTISAMPLED_SHADING,
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
                Capabilities::CUBE_ARRAY_TEXTURES,
                format!("resource {} is a cube array texture", r.provenance.name),
            );
        }
    }
    found
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
