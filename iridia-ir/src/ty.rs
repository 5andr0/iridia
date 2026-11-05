//! The types of values: scalars and short vectors.

use core::fmt;

/// A scalar type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum Scalar {
    /// A boolean.
    Bool,
    /// A signed integer of 8, 16, 32 or 64 bits.
    I8,
    /// See [`Scalar::I8`].
    I16,
    /// See [`Scalar::I8`].
    I32,
    /// See [`Scalar::I8`].
    I64,
    /// An unsigned integer of 8, 16, 32 or 64 bits.
    U8,
    /// See [`Scalar::U8`].
    U16,
    /// See [`Scalar::U8`].
    U32,
    /// See [`Scalar::U8`].
    U64,
    /// An IEEE 754 binary16 float.
    F16,
    /// An IEEE 754 binary32 float.
    F32,
    /// An IEEE 754 binary64 float.
    F64,
}

impl Scalar {
    /// The width in bits; a bool counts as 1.
    pub const fn bits(self) -> u32 {
        match self {
            Scalar::Bool => 1,
            Scalar::I8 | Scalar::U8 => 8,
            Scalar::I16 | Scalar::U16 | Scalar::F16 => 16,
            Scalar::I32 | Scalar::U32 | Scalar::F32 => 32,
            Scalar::I64 | Scalar::U64 | Scalar::F64 => 64,
        }
    }
    /// A float type.
    pub const fn is_float(self) -> bool {
        matches!(self, Scalar::F16 | Scalar::F32 | Scalar::F64)
    }
    /// A signed integer type.
    pub const fn is_signed(self) -> bool {
        matches!(self, Scalar::I8 | Scalar::I16 | Scalar::I32 | Scalar::I64)
    }
    /// An unsigned integer type.
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Scalar::U8 | Scalar::U16 | Scalar::U32 | Scalar::U64)
    }
    /// An integer type, signed or not.
    pub const fn is_integer(self) -> bool {
        self.is_signed() || self.is_unsigned()
    }
    /// The name in the text form (`f32`).
    pub const fn name(self) -> &'static str {
        match self {
            Scalar::Bool => "bool",
            Scalar::I8 => "i8",
            Scalar::I16 => "i16",
            Scalar::I32 => "i32",
            Scalar::I64 => "i64",
            Scalar::U8 => "u8",
            Scalar::U16 => "u16",
            Scalar::U32 => "u32",
            Scalar::U64 => "u64",
            Scalar::F16 => "f16",
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
        }
    }
    /// The scalar of a text-form name.
    pub fn parse(name: &str) -> Option<Scalar> {
        Some(match name {
            "bool" => Scalar::Bool,
            "i8" => Scalar::I8,
            "i16" => Scalar::I16,
            "i32" => Scalar::I32,
            "i64" => Scalar::I64,
            "u8" => Scalar::U8,
            "u16" => Scalar::U16,
            "u32" => Scalar::U32,
            "u64" => Scalar::U64,
            "f16" => Scalar::F16,
            "f32" => Scalar::F32,
            "f64" => Scalar::F64,
            _ => return None,
        })
    }
}

/// The type of a value: a scalar, or a vector of 2 to 4 lanes of one scalar type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct Type {
    /// The type of every lane.
    pub scalar: Scalar,
    /// The number of lanes, 1 (a scalar) to 4.
    pub lanes: u8,
}

impl Type {
    /// A scalar of `scalar`.
    pub const fn scalar(scalar: Scalar) -> Type {
        Type { scalar, lanes: 1 }
    }
    /// A vector of `lanes` lanes, or a scalar for 1.
    pub const fn vector(scalar: Scalar, lanes: u8) -> Type {
        Type { scalar, lanes }
    }
    /// `f32`.
    pub const F32: Type = Type::scalar(Scalar::F32);
    /// `vec4<f32>`.
    pub const F32X4: Type = Type::vector(Scalar::F32, 4);
    /// `u32`.
    pub const U32: Type = Type::scalar(Scalar::U32);
    /// `vec4<u32>`.
    pub const U32X4: Type = Type::vector(Scalar::U32, 4);
    /// `i32`.
    pub const I32: Type = Type::scalar(Scalar::I32);
    /// `bool`.
    pub const BOOL: Type = Type::scalar(Scalar::Bool);
    /// The same shape with another scalar.
    pub const fn with(self, scalar: Scalar) -> Type {
        Type { scalar, lanes: self.lanes }
    }
    /// Whether the lane count is one a value can have.
    pub const fn is_valid(self) -> bool {
        self.lanes >= 1 && self.lanes <= 4
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.lanes == 1 {
            f.write_str(self.scalar.name())
        } else {
            write!(f, "vec{}<{}>", self.lanes, self.scalar.name())
        }
    }
}
