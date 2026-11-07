//! The text of types and literals.

use alloc::{
    format,
    string::{String, ToString},
};

use iridia_ir::{Scalar, Type};

use crate::Error;

pub(crate) fn err(message: impl Into<String>) -> Error {
    Error(message.into())
}

/// WGSL's name of a scalar type. `i8` and `u8` have none.
pub(crate) fn scalar(s: Scalar) -> Result<&'static str, Error> {
    match s {
        Scalar::I8 | Scalar::U8 => Err(err(format!("{} has no WGSL type", s.name()))),
        other => Ok(other.name()),
    }
}

/// WGSL's name of a value's type: `f32`, `vec3<u32>`.
pub(crate) fn ty(t: Type) -> Result<String, Error> {
    let s = scalar(t.scalar)?;
    match t.lanes {
        1 => Ok(s.to_string()),
        2..=4 => Ok(format!("vec{}<{s}>", t.lanes)),
        n => Err(err(format!("a vector of {n} lanes"))),
    }
}

/// The name of lane `k` as a component: `x`, `y`, `z`, `w`.
pub(crate) fn lane(k: u8) -> Result<&'static str, Error> {
    match k {
        0 => Ok("x"),
        1 => Ok("y"),
        2 => Ok("z"),
        3 => Ok("w"),
        n => Err(err(format!("lane {n}"))),
    }
}

/// The components `lanes` name, in order: `xz`, `w`.
pub(crate) fn swizzle(lanes: impl IntoIterator<Item = u8>) -> Result<String, Error> {
    lanes.into_iter().map(lane).collect()
}

/// A literal of `s` with these bits (the low bits of the word; a bool is 0 or 1). A float that is not an ordinary one (an
/// infinity, a NaN, a denormal, a negative zero, and the largest finite `f32`, whose shortest decimal Chrome's compiler
/// refuses as out of range) is written as the bits of an integer, bitcast: exactly the value, whatever a compiler does
/// with decimals.
pub(crate) fn literal(s: Scalar, bits: u64) -> Result<String, Error> {
    let low = u32::try_from(bits & 0xffff_ffff).unwrap_or(0);
    Ok(match s {
        Scalar::Bool => (bits != 0).to_string(),
        Scalar::F32 => {
            let f = f32::from_bits(low);
            if f == 0.0 && low == 0 {
                "0.0f".to_string()
            } else if !f.is_finite() {
                // a constant infinity or NaN is an error in WGSL: a call hides the bits from the compiler's folding
                format!("iridia_f32_from_bits({low}u)")
            } else if !f.is_normal() || f.abs() == f32::MAX {
                format!("bitcast<f32>({low}u)")
            } else if f < 0.0 {
                format!("({f:e}f)")
            } else {
                format!("{f:e}f")
            }
        }
        Scalar::F64 => {
            let f = f64::from_bits(bits);
            if f == 0.0 && bits == 0 {
                "0.0lf".to_string()
            } else if !f.is_normal() {
                format!("bitcast<f64>({bits}lu)")
            } else if f < 0.0 {
                format!("({f:e}lf)")
            } else {
                format!("{f:e}lf")
            }
        }
        Scalar::U32 => format!("{low}u"),
        Scalar::I32 => {
            let v = low.cast_signed();
            if v == i32::MIN {
                // `-2147483648i` is not valid WGSL; the abstract integer negated, converted, is
                format!("i32({v})")
            } else if v < 0 {
                format!("({v}i)")
            } else {
                format!("{v}i")
            }
        }
        Scalar::U64 => format!("{bits}lu"),
        Scalar::I64 => {
            let v = bits.cast_signed();
            if v == i64::MIN {
                format!("i64({} - 1)", v.saturating_add(1))
            } else if v < 0 {
                format!("({v}li)")
            } else {
                format!("{v}li")
            }
        }
        Scalar::U16 => format!("u16({})", bits & 0xffff),
        Scalar::I16 => format!("i16({})", u16::try_from(bits & 0xffff).unwrap_or(0).cast_signed()),
        Scalar::F16 => return Err(err("f16 literals are not written yet")),
        Scalar::I8 | Scalar::U8 => return Err(err(format!("{} has no WGSL type", s.name()))),
    })
}

/// A value of type `t` with these bits in each lane (one literal, as a vector's single-argument constructor for several
/// lanes).
pub(crate) fn splat(t: Type, s: Scalar, bits: u64) -> Result<String, Error> {
    let l = literal(s, bits)?;
    Ok(if t.lanes == 1 {
        l
    } else {
        format!("{}({l})", ty(Type::vector(s, t.lanes))?)
    })
}

/// A literal value of type `t`, each lane's own bits.
pub(crate) fn value(t: Type, bits: [u64; 4]) -> Result<String, Error> {
    let lanes: alloc::vec::Vec<String> = bits
        .iter()
        .take(usize::from(t.lanes))
        .map(|b| literal(t.scalar, *b))
        .collect::<Result<_, _>>()?;
    Ok(match lanes.as_slice() {
        [one] => one.clone(),
        [first, rest @ ..] if rest.iter().all(|l| l == first) => format!("{}({first})", ty(t)?),
        _ => format!("{}({})", ty(t)?, lanes.join(", ")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_are_written_exactly() {
        assert_eq!(literal(Scalar::F32, u64::from(1.0f32.to_bits())).unwrap(), "1e0f");
        assert_eq!(
            literal(Scalar::F32, u64::from((-0.5f32).to_bits())).unwrap(),
            "(-5e-1f)"
        );
        assert_eq!(literal(Scalar::F32, 0).unwrap(), "0.0f");
        // negative zero, a denormal and the largest finite value are bits
        for bits in [0x8000_0000u32, 1, 0x7f7f_ffff, 0xff7f_ffff] {
            assert_eq!(
                literal(Scalar::F32, u64::from(bits)).unwrap(),
                format!("bitcast<f32>({bits}u)")
            );
        }
        // an infinity or a NaN is the bits through a call, which no compiler folds into a constant
        for bits in [0x7f80_0000u32, 0xff80_0000, 0x7fc0_0000] {
            assert_eq!(
                literal(Scalar::F32, u64::from(bits)).unwrap(),
                format!("iridia_f32_from_bits({bits}u)")
            );
        }
    }

    #[test]
    fn integers_are_written_with_their_suffix() {
        assert_eq!(literal(Scalar::U32, 7).unwrap(), "7u");
        assert_eq!(literal(Scalar::I32, 0xffff_ffff).unwrap(), "(-1i)");
        assert_eq!(literal(Scalar::I32, 0x8000_0000).unwrap(), "i32(-2147483648)");
        assert_eq!(literal(Scalar::Bool, 1).unwrap(), "true");
    }

    #[test]
    fn a_vector_of_equal_lanes_is_one_argument() {
        assert_eq!(
            value(Type::vector(Scalar::U32, 3), [5, 5, 5, 0]).unwrap(),
            "vec3<u32>(5u)"
        );
        assert_eq!(
            value(Type::vector(Scalar::U32, 2), [1, 2, 0, 0]).unwrap(),
            "vec2<u32>(1u, 2u)"
        );
    }
}
