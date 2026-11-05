//! Operations. Each states what it gives for every input, including NaN, infinities, signed zero and out-of-range
//! values; where sources' rules differ the operation carries a mandatory semantic enum naming the behaviour (never a
//! source). Target legalization expands what a target lacks, in one place (`iridia-passes`).

use alloc::vec::Vec;

use crate::interface::ResourceId;

/// A value of a function: an index into its value arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct ValueId(pub u32);

/// A mutable variable of a function (a source register, a local array), until `promote` turns it into values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct VarId(pub u32);

/// A label of a `Regions` construct (`Block`, `Loop`) that exits name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct LabelId(pub u32);

/// What `FMin` and `FMax` give when an operand is NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum MinMaxNan {
    /// IEEE 754-2008 minNum/maxNum: one NaN operand gives the other operand; two give NaN (D3D10 and later).
    MinNum,
    /// Any NaN operand gives NaN.
    Propagate,
    /// Either operand or NaN: the source leaves it undefined (GLSL, WGSL).
    Any,
}

/// What `Saturate` gives for NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum SaturateNan {
    /// NaN gives 0 (D3D).
    Zero,
    /// Any value in 0..1, or NaN: undefined by the source.
    Any,
}

/// How `FMul` treats a zero operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum MulZero {
    /// IEEE 754: `0 * inf` is NaN, signs multiply.
    Ieee,
    /// Zero times anything, infinity and NaN included, is +0 (D3D9's legacy multiply, a compatibility policy).
    ZeroWins,
}

/// What `UDiv` and `URem` give for a zero divisor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum OnZero {
    /// All ones (D3D's `udiv`, for quotient and remainder).
    AllOnes,
    /// The dividend (WGSL's rule).
    Dividend,
    /// Undefined (SPIR-V, GLSL).
    Undef,
}

/// How a float becomes an integer, out of range and for NaN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum FloatToInt {
    /// Truncate toward zero, saturate to the integer type's range, NaN gives 0 (D3D's `ftoi`/`ftou`).
    Saturate,
    /// Truncate toward zero; out of range and NaN undefined.
    Undef,
}

/// A float rounding to an integral value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Rounding {
    /// Toward negative infinity.
    Floor,
    /// Toward positive infinity.
    Ceil,
    /// Toward zero.
    Trunc,
    /// To the nearest, ties to even.
    NearestEven,
    /// To the nearest, ties away from zero (D3D9 `mova`).
    NearestAway,
}

/// A comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Cmp {
    /// Equal (ordered for floats: false with a NaN).
    Eq,
    /// Not equal (unordered for floats: true with a NaN).
    Ne,
    /// Less than (ordered).
    Lt,
    /// At most (ordered).
    Le,
    /// Greater than (ordered).
    Gt,
    /// At least (ordered).
    Ge,
}

/// A one-operand operation; the result has the operand's type unless noted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Unary {
    /// Float negation (flips the sign bit, NaN included).
    FNeg,
    /// Float absolute value (clears the sign bit).
    FAbs,
    /// `x - floor(x)`.
    FFract,
    /// Rounding to an integral float.
    FRound(Rounding),
    /// `sqrt(x)`, correctly rounded is not required (WGSL's precision).
    FSqrt,
    /// `1 / sqrt(x)`, at the precision WGSL's `inverseSqrt` states.
    FInverseSqrt,
    /// `2^x`.
    FExp2,
    /// `log2(x)`; 0 gives -inf, negative values NaN.
    FLog2,
    /// `sin(x)`, at WGSL's precision.
    FSin,
    /// `cos(x)`, at WGSL's precision.
    FCos,
    /// Clamp to 0..1, with the stated NaN rule.
    Saturate(SaturateNan),
    /// Boolean not.
    Not,
    /// Integer negation (wrapping).
    INeg,
    /// Whether any lane of a bool vector is true; the result is a `bool`.
    Any,
    /// Whether every lane of a bool vector is true; the result is a `bool`.
    All,
    /// Whether a float is NaN, by its bits; the result is bool of the operand's lanes.
    IsNan,
}

/// A two-operand operation on operands of one type; the result has their type unless noted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Binary {
    /// Float addition.
    FAdd,
    /// Float subtraction.
    FSub,
    /// Float multiplication with the stated zero rule.
    FMul(MulZero),
    /// Float division (IEEE: `x / 0` is a signed infinity or NaN).
    FDiv,
    /// Float minimum with the stated NaN rule.
    FMin(MinMaxNan),
    /// Float maximum with the stated NaN rule.
    FMax(MinMaxNan),
    /// `x^y` for `x >= 0`; undefined for negative `x`, and for `x = 0` with `y <= 0`.
    FPow,
    /// Float comparison; the result is bool of the operands' lanes.
    FCmp(Cmp),
    /// Signed integer comparison; the result is bool.
    SCmp(Cmp),
    /// Unsigned integer comparison (also equality of any integers and bools); the result is bool.
    UCmp(Cmp),
    /// Integer addition, wrapping.
    IAdd,
    /// Integer subtraction, wrapping.
    ISub,
    /// Integer multiplication, wrapping.
    IMul,
    /// Unsigned division with the stated zero rule.
    UDiv(OnZero),
    /// Unsigned remainder with the stated zero rule.
    URem(OnZero),
    /// Signed minimum.
    SMin,
    /// Signed maximum.
    SMax,
    /// Bitwise and (logical and of bools).
    And,
    /// Bitwise or (logical or of bools).
    Or,
    /// Bitwise exclusive or.
    Xor,
}

/// The level of detail a sample is taken at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Level {
    /// From the coordinate's derivatives (pixel programs only).
    Auto,
    /// From the derivatives, plus a bias (`f32`).
    Bias(ValueId),
    /// An explicit level (`f32`).
    Exact(ValueId),
    /// Level zero.
    Zero,
}

/// What a buffer read outside the resource's size gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum OutOfBounds {
    /// Zero (D3D9 relative constants outside the bank, D3D10+ buffer reads).
    Zero,
    /// Undefined.
    Undef,
}

/// An operation that defines a value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Op {
    /// A constant: each lane's exact bits, in the low bits of its word (a bool is 0 or 1).
    Literal([u64; 4]),
    /// An undefined value.
    Undef,
    /// A value given by its construct: a `Loop`'s parameter, a construct's result, a `Cfg` block's parameter.
    Param,
    /// A one-operand operation.
    Unary(Unary, ValueId),
    /// A two-operand operation.
    Binary(Binary, ValueId, ValueId),
    /// `cond ? accept : reject`, lane by lane when `cond` is a vector, else whole.
    Select {
        /// A bool, or a bool vector of the result's lanes.
        cond: ValueId,
        /// The value where `cond` holds.
        accept: ValueId,
        /// The value where it does not.
        reject: ValueId,
    },
    /// A vector of the lanes of the parts, in order (scalars and vectors of the result's scalar type).
    Compose(Vec<ValueId>),
    /// The result's lanes taken from the operand's lanes `lanes[0..result lanes]`.
    Swizzle(ValueId, [u8; 4]),
    /// Every lane of the result is the scalar operand.
    Splat(ValueId),
    /// The float dot product of two vectors of one type, a scalar: `a0*b0 + a1*b1 + ...`, left to right, each step
    /// rounded (no fused multiply-add).
    Dot(ValueId, ValueId),
    /// A numeric conversion between scalar types of the same lane count. Float to integer follows `FloatToInt`;
    /// integer to float rounds to nearest even.
    Convert(ValueId, FloatToInt),
    /// The same bits as another type of the same total size.
    Bitcast(ValueId),
    /// Lanes of an interface input (`interface.inputs[slot]`): the lanes of mask `.1` (bit n is lane n, at least one,
    /// within the varying's width), in ascending order, as a value of the varying's scalar type.
    LoadInput(u32, u8),
    /// The value a variable holds.
    LoadVar(VarId),
    /// The value of a pipeline state resource (a [`crate::interface::ResourceKind::PipelineState`]), as its
    /// [`crate::interface::StateParam::value`].
    LoadState(ResourceId),
    /// Words of a buffer resource: the byte at `offset + stride * index` and the result's size after it.
    LoadBuffer {
        /// The buffer (a constant buffer).
        resource: ResourceId,
        /// The dynamic element index (`u32` or `i32`), or none for a fixed address.
        index: Option<ValueId>,
        /// The bytes between elements.
        stride: u32,
        /// The byte offset of element 0.
        offset: u32,
        /// What a read past the resource's size (or at a negative index) gives.
        oob: OutOfBounds,
    },
    /// A filtered sample of a texture through a sampler: four lanes of the texture's sample type.
    Sample {
        /// The texture.
        texture: ResourceId,
        /// The sampler.
        sampler: ResourceId,
        /// The coordinate: two lanes for 2D, three for cube and volume.
        coord: ValueId,
        /// The level of detail.
        level: Level,
    },
}

impl Op {
    /// The values the operation reads, in operand order.
    pub fn operands(&self) -> Vec<ValueId> {
        let mut v = Vec::new();
        match self {
            Op::Literal(_)
            | Op::Undef
            | Op::Param
            | Op::LoadInput(..)
            | Op::LoadVar(_)
            | Op::LoadState(_) => {}
            Op::Unary(_, a) | Op::Swizzle(a, _) | Op::Splat(a) | Op::Convert(a, _) | Op::Bitcast(a) => {
                v.push(*a)
            }
            Op::Binary(_, a, b) | Op::Dot(a, b) => {
                v.push(*a);
                v.push(*b);
            }
            Op::Select { cond, accept, reject } => v.extend([*cond, *accept, *reject]),
            Op::Compose(parts) => v.extend(parts.iter().copied()),
            Op::LoadBuffer { index, .. } => v.extend(index.iter().copied()),
            Op::Sample { coord, level, .. } => {
                v.push(*coord);
                if let Level::Bias(x) | Level::Exact(x) = level {
                    v.push(*x);
                }
            }
        }
        v
    }

    /// Whether the operation reads state other than its operands (inputs, variables, resources); such a value is
    /// placed where its node is and is not moved across stores.
    pub fn reads_state(&self) -> bool {
        matches!(
            self,
            Op::LoadInput(..) | Op::LoadVar(_) | Op::LoadState(_) | Op::LoadBuffer { .. } | Op::Sample { .. }
        )
    }

    /// Replace each operand by `f` of it.
    pub fn map_operands(&mut self, mut f: impl FnMut(ValueId) -> ValueId) {
        match self {
            Op::Literal(_)
            | Op::Undef
            | Op::Param
            | Op::LoadInput(..)
            | Op::LoadVar(_)
            | Op::LoadState(_) => {}
            Op::Unary(_, a) | Op::Swizzle(a, _) | Op::Splat(a) | Op::Convert(a, _) | Op::Bitcast(a) => {
                *a = f(*a)
            }
            Op::Binary(_, a, b) | Op::Dot(a, b) => {
                *a = f(*a);
                *b = f(*b);
            }
            Op::Select { cond, accept, reject } => {
                *cond = f(*cond);
                *accept = f(*accept);
                *reject = f(*reject);
            }
            Op::Compose(parts) => parts.iter_mut().for_each(|p| *p = f(*p)),
            Op::LoadBuffer { index, .. } => {
                if let Some(i) = index {
                    *i = f(*i);
                }
            }
            Op::Sample { coord, level, .. } => {
                *coord = f(*coord);
                if let Level::Bias(x) | Level::Exact(x) = level {
                    *x = f(*x);
                }
            }
        }
    }
}
