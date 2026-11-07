//! How an entry point carries the interface: the attributes of an argument or a result member
//! ([`iridia_ir::target::entry_io`] says where each goes), and a result read as another value than the program wrote it.

use alloc::{format, string::String, vec::Vec};

use iridia_ir::interface::{InterpMode, InterpSampling, ScalarType, Value};
use iridia_ir::target::{Binding, BuiltIn};

use crate::emit::R;
use crate::text::{self, err};

/// WGSL's name of a built-in.
fn builtin(b: BuiltIn) -> &'static str {
    match b {
        BuiltIn::Position => "position",
        BuiltIn::FrontFacing => "front_facing",
        BuiltIn::VertexIndex => "vertex_index",
        BuiltIn::InstanceIndex => "instance_index",
        BuiltIn::PrimitiveIndex => "primitive_index",
        BuiltIn::SampleIndex => "sample_index",
        BuiltIn::SampleMask => "sample_mask",
        BuiltIn::FragDepth => "frag_depth",
        BuiltIn::WorkgroupId => "workgroup_id",
        BuiltIn::LocalInvocationId => "local_invocation_id",
        BuiltIn::GlobalInvocationId => "global_invocation_id",
        BuiltIn::LocalInvocationIndex => "local_invocation_index",
        BuiltIn::SubgroupInvocationId => "subgroup_invocation_id",
        BuiltIn::SubgroupSize => "subgroup_size",
        BuiltIn::ClipDistances => "clip_distances",
    }
}

/// The attributes of an argument or a result member: `@builtin(position) `, `@location(1) @interpolate(flat) `. WGSL's
/// default interpolation (perspective, at the centre) is not written.
pub(crate) fn attributes(b: &Binding) -> String {
    match b {
        Binding::BuiltIn(b) => format!("@builtin({}) ", builtin(*b)),
        Binding::Location {
            location,
            interpolation,
            sampling,
        } => {
            let at = format!("@location({location}) ");
            let mode = match interpolation {
                Some(InterpMode::Perspective) | None => None,
                Some(InterpMode::Linear) => Some("linear"),
                Some(InterpMode::Flat) => Some("flat"),
            };
            let sampling = match sampling {
                InterpSampling::Center => None,
                InterpSampling::Centroid => Some("centroid"),
                InterpSampling::Sample => Some("sample"),
            };
            match (mode, sampling) {
                (_, Some(s)) => format!("{at}@interpolate({}, {s}) ", mode.unwrap_or("perspective")),
                (Some(m), None) => format!("{at}@interpolate({m}) "),
                (None, None) => at,
            }
        }
    }
}

/// WGSL's 32-bit scalar of a linked value; a boolean or 64-bit one is refused (its bits are not one 32-bit lane).
fn linked_scalar(s: ScalarType) -> R<iridia_ir::Scalar> {
    match s {
        ScalarType::Float => Ok(iridia_ir::Scalar::F32),
        ScalarType::Uint => Ok(iridia_ir::Scalar::U32),
        ScalarType::Sint => Ok(iridia_ir::Scalar::I32),
        ScalarType::Bool | ScalarType::Double => Err(err(format!(
            "a {s:?} output read as another value: its bits are not a 32-bit lane"
        ))),
    }
}

/// `x` (a name of a value of `from`) as the value `to`: its first lanes, or padded with zeros, then the same bits as `to`'s
/// scalar.
pub(crate) fn relink(x: &str, from: Value, to: Value) -> R<String> {
    let mut e: String = x.into();
    let (fw, tw) = (from.width, to.width);
    let from_scalar = linked_scalar(from.scalar)?;
    if tw < fw {
        e = format!("{e}.{}", text::swizzle(0..tw)?);
    } else if tw > fw {
        let mut parts: Vec<String> = if fw == 1 {
            alloc::vec![e]
        } else {
            (0..fw)
                .map(|k| Ok(format!("{x}.{}", text::lane(k)?)))
                .collect::<R<_>>()?
        };
        let zero = text::literal(from_scalar, 0)?;
        parts.resize(usize::from(tw), zero);
        e = format!(
            "{}({})",
            text::ty(iridia_ir::Type::vector(from_scalar, tw))?,
            parts.join(", ")
        );
    }
    if from.scalar != to.scalar {
        let to_scalar = linked_scalar(to.scalar)?;
        e = format!(
            "bitcast<{}>({e})",
            text::ty(iridia_ir::Type::vector(to_scalar, tw))?
        );
    }
    Ok(e)
}
