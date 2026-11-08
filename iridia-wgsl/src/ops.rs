//! Operations: each IR operation as the expression of its value. Operations WGSL has whole are one call or operator; the
//! others expand here, each in one function (`is_nan`, `saturate`, `min_max`, `float_to_int`, `round_away`, ...), or are
//! a call of a helper function (`helpers`).
//!
//! Every operand is a name (`v3`, `x7`) or a literal, so an expression may use one twice; a subexpression the expansion
//! reads more than once is a temporary of its own.

use alloc::{format, string::String, vec::Vec};

use iridia_ir::interface::ScalarType;
use iridia_ir::{
    Axis, Binary, Cmp, Fineness, FloatToInt, MinMaxNan, MulZero, OnZero, Op, Rounding, SaturateNan, Scalar,
    Ternary, Type, Unary, ValueId,
};

use crate::emit::{R, Writer};
use crate::helpers::Helper;
use crate::text::{self, err};

/// The lanes a mask names, in ascending order.
fn masked(mask: u8) -> impl Iterator<Item = u8> {
    (0..4u8).filter(move |n| mask >> n & 1 != 0)
}

impl Writer<'_> {
    /// Define value `v` from its operation.
    pub(crate) fn value(&mut self, v: ValueId) -> R<()> {
        let def = self
            .f
            .value(v)
            .ok_or_else(|| err(format!("%{} is not a value", v.0)))?;
        let t = def.ty;
        let op = def.op.clone();
        let expr = match op {
            Op::Literal(bits) => {
                if t.lanes > 1 {
                    let scalars = bits
                        .iter()
                        .take(usize::from(t.lanes))
                        .map(|b| text::literal(t.scalar, *b))
                        .collect::<Result<Vec<_>, _>>()?;
                    self.lanes.insert(v.0, scalars);
                }
                text::value(t, bits)?
            }
            Op::Undef => format!("{}()", self.ty(t)?),
            Op::Param => return Err(err(format!("parameter %{} has no value here", v.0))),
            Op::Unary(u, a) => self.unary(u, a)?,
            Op::Binary(b, l, r) => self.binary(b, l, r)?,
            Op::Ternary(op, a, b, c) => self.ternary(op, a, b, c)?,
            Op::BitInsert {
                base,
                insert,
                offset,
                count,
            } => {
                let at = self.vty(base)?;
                let (b, i, o, c) = (
                    self.get(base)?,
                    self.get(insert)?,
                    self.get(offset)?,
                    self.get(count)?,
                );
                self.per_lane(at, |w, k| {
                    let bs = w.lane(&b, at, k)?;
                    let is = w.lane(&i, at, k)?;
                    let os = w.low_five(&w.lane(&o, at.with(Scalar::U32), k)?);
                    let cs = w.low_five(&w.lane(&c, at.with(Scalar::U32), k)?);
                    Ok(format!("insertBits({bs}, {is}, {os}, {cs})"))
                })?
            }
            Op::Select { cond, accept, reject } => {
                format!(
                    "select({}, {}, {})",
                    self.get(reject)?,
                    self.get(accept)?,
                    self.get(cond)?
                )
            }
            Op::Compose(parts) => {
                let mut scalars = Vec::new();
                for p in &parts {
                    scalars.extend(self.lane_names(*p)?);
                }
                let text = format!("{}({})", self.ty(t)?, scalars.join(", "));
                if scalars.len() == usize::from(t.lanes) {
                    self.lanes.insert(v.0, scalars);
                }
                text
            }
            Op::Swizzle(a, lanes) if self.lanes.contains_key(&a.0) => {
                let known = self.lane_names(a)?;
                let picked = lanes
                    .iter()
                    .copied()
                    .take(usize::from(t.lanes))
                    .map(|k| {
                        known
                            .get(usize::from(k))
                            .cloned()
                            .ok_or_else(|| err("a lane past the vector's"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                match picked.as_slice() {
                    [one] => one.clone(),
                    _ => {
                        let text = format!("{}({})", self.ty(t)?, picked.join(", "));
                        self.lanes.insert(v.0, picked);
                        text
                    }
                }
            }
            Op::Swizzle(a, lanes) => {
                let x = self.get(a)?;
                let at = self.vty(a)?;
                if at.lanes == 1 {
                    // every lane of a scalar is the scalar
                    if t.lanes == 1 {
                        x
                    } else {
                        format!("{}({x})", self.ty(t)?)
                    }
                } else {
                    let picked: Vec<u8> = lanes.iter().copied().take(usize::from(t.lanes)).collect();
                    format!("{x}.{}", text::swizzle(picked)?)
                }
            }
            Op::Splat(a) => {
                let x = self.get(a)?;
                if self.vty(a)?.lanes == 1 {
                    self.lanes
                        .insert(v.0, alloc::vec![x.clone(); usize::from(t.lanes)]);
                }
                format!("{}({x})", self.ty(t)?)
            }
            Op::Dot(a, b) => format!("dot({}, {})", self.get(a)?, self.get(b)?),
            Op::Convert(a, rule) => self.convert(a, t, rule)?,
            Op::Bitcast(a) => {
                let x = self.get(a)?;
                let from = self.vty(a)?;
                if from.lanes == t.lanes {
                    self.bitcast(&x, from, t.scalar)?
                } else {
                    self.bitcast_lanes(&x, from, t)?
                }
            }
            Op::LoadInput(slot, mask) => self.load_input(slot, mask, t)?,
            Op::LoadState(r) => {
                return Err(err(format!(
                    "pipeline state r{} is read: specialise it first (a state block is not written yet)",
                    r.0
                )));
            }
            Op::LoadVar(var) => self.load_var(var, t)?,
            Op::LoadElement(var, index, oob) => self.load_element_of_var(var, index, oob, t)?,
            Op::LoadInputAt(..) => return Err(err("a read of one vertex's input is not written yet")),
            Op::LoadOutputAt(..) => {
                return Err(err("a read of one control point's output is not written yet"));
            }
            Op::LoadPatch(..) => return Err(err("a patch constant read is not written yet")),
            Op::Builtin(b) => return Err(err(format!("builtin {b:?} is not written yet"))),
            Op::Gather {
                texture,
                sampler,
                coord,
                component,
                compare,
                offset,
            } => self.gather(texture, sampler, coord, component, compare, offset)?,
            Op::LoadTexel { texture, coord, .. } if self.typed_buffer(texture.id)? => {
                self.load_typed_element(texture, coord, t)?
            }
            Op::LoadTexel {
                texture,
                coord,
                level,
                sample,
                offset,
            } => self.load_texel(texture, coord, level, sample, offset, t)?,
            Op::Query(r, iridia_ir::Query::Elements) => self.elements(r)?,
            Op::Query(r, q) => self.image_query(r, q, t)?,
            Op::Atomic {
                memory,
                op,
                value,
                compare,
            } => self.atomic(memory, op, value, compare, t)?,
            Op::Wave(w) => self.wave(w, t)?,
            Op::ReportHit { .. } => {
                return Err(err("an intersection program's reported hit is not written yet"));
            }
            Op::RayQueryProceed(q) => self.ray_query_proceed(q)?,
            Op::RayQueryState(q, field) => self.ray_query_state(q, field)?,
            Op::Counter(r, op) => self.counter(r, op)?,
            Op::LoadBuffer {
                resource,
                address,
                oob,
            } => {
                if self.words_buffer(resource.id)? {
                    self.load_words(resource, address, t)?
                } else {
                    self.load_constants(resource, address, oob, t)?
                }
            }
            Op::Sample {
                texture,
                sampler,
                coord,
                level,
                compare,
                offset,
                clamp,
            } => self.sample(texture, sampler, coord, level, compare, offset, clamp)?,
        };
        self.define(v, &expr)
    }

    // small expansions

    /// `x & 31u`: a count modulo 32.
    pub(crate) fn low_five(&self, x: &str) -> String {
        format!("({x} & 31u)")
    }

    /// Whether a float is NaN, by its bits (WGSL has no `isNan`, and a compiler may assume no NaNs in comparisons).
    pub(crate) fn is_nan(&self, x: &str, t: Type) -> R<String> {
        match t.scalar {
            Scalar::F32 => {
                let bits = self.bitcast(x, t, Scalar::U32)?;
                let abs = self.u32s(t, 0x7fff_ffff)?;
                let infinity = self.u32s(t, 0x7f80_0000)?;
                Ok(format!("(({bits} & {abs}) > {infinity})"))
            }
            Scalar::F64 => {
                let bits = self.bitcast(x, t, Scalar::U64)?;
                let abs = self.konst(t, Scalar::U64, 0x7fff_ffff_ffff_ffff)?;
                let infinity = self.konst(t, Scalar::U64, 0x7ff0_0000_0000_0000)?;
                Ok(format!("(({bits} & {abs}) > {infinity})"))
            }
            _ => Err(err(format!("a NaN test of {t} is not written yet"))),
        }
    }

    /// Whether a float is an infinity, by its bits.
    fn is_inf(&self, x: &str, t: Type) -> R<String> {
        if t.scalar != Scalar::F32 {
            return Err(err(format!("an infinity test of {t} is not written yet")));
        }
        let bits = self.bitcast(x, t, Scalar::U32)?;
        let abs = self.u32s(t, 0x7fff_ffff)?;
        let infinity = self.u32s(t, 0x7f80_0000)?;
        Ok(format!("(({bits} & {abs}) == {infinity})"))
    }

    fn saturate(&self, x: &str, t: Type, nan: SaturateNan) -> R<String> {
        let clamped = format!("saturate({x})");
        match nan {
            SaturateNan::Any => Ok(clamped),
            SaturateNan::Zero => {
                let is_nan = self.is_nan(x, t)?;
                Ok(format!("select({clamped}, {}, {is_nan})", self.f32s(t, 0.0)?))
            }
        }
    }

    fn clamp_range(&self, x: &str, t: Type, lo: u32, hi: u32, nan: SaturateNan) -> R<String> {
        let low = self.f32s(t, f32::from_bits(lo))?;
        let high = self.f32s(t, f32::from_bits(hi))?;
        let clamped = format!("clamp({x}, {low}, {high})");
        match nan {
            SaturateNan::Any => Ok(clamped),
            SaturateNan::Zero => {
                let is_nan = self.is_nan(x, t)?;
                Ok(format!("select({clamped}, {}, {is_nan})", self.f32s(t, 0.0)?))
            }
        }
    }

    /// To the nearest integer, a half away from zero: the truncation, moved one away from zero where the part cut off is
    /// at least a half (exact: `x - trunc(x)` is representable).
    fn round_away(&mut self, x: &str, t: Type) -> R<String> {
        let truncated = self.tmp(&format!("trunc({x})"));
        let half = self.f32s(t, 0.5)?;
        Ok(format!(
            "select({truncated}, ({truncated} + sign({x})), (abs(({x} - {truncated})) >= {half}))"
        ))
    }

    /// `round(x * 2^f) / 2^f`, and `x` itself from `2^(23 - f)` on, where every float is on the grid already (and the scaled
    /// value could overflow).
    fn quantize(&mut self, x: &str, t: Type, frac_bits: u8, rounding: Rounding) -> R<String> {
        let bits = u32::from(frac_bits);
        let scale = self.f32s(t, f32::from_bits(127u32.saturating_add(bits) << 23))?;
        let grid = self.f32s(t, f32::from_bits(150u32.saturating_sub(bits) << 23))?;
        let scaled = self.tmp(&format!("({x} * {scale})"));
        let rounded = self.round(&scaled, t, rounding)?;
        Ok(format!(
            "select(({rounded} / {scale}), {x}, (abs({x}) >= {grid}))"
        ))
    }

    fn round(&mut self, x: &str, t: Type, rounding: Rounding) -> R<String> {
        Ok(match rounding {
            Rounding::Floor => format!("floor({x})"),
            Rounding::Ceil => format!("ceil({x})"),
            Rounding::Trunc => format!("trunc({x})"),
            Rounding::NearestEven => format!("round({x})"),
            Rounding::NearestAway => self.round_away(x, t)?,
        })
    }

    fn min_max(&self, fun: &str, a: &str, b: &str, t: Type, nan: MinMaxNan) -> R<String> {
        let plain = format!("{fun}({a}, {b})");
        match nan {
            MinMaxNan::Any => Ok(plain),
            MinMaxNan::MinNum => {
                let (na, nb) = (self.is_nan(a, t)?, self.is_nan(b, t)?);
                Ok(format!("select(select({plain}, {a}, {nb}), {b}, {na})"))
            }
            MinMaxNan::Propagate => {
                let (na, nb) = (self.is_nan(a, t)?, self.is_nan(b, t)?);
                Ok(format!("select({plain}, ({a} + {b}), ({na} | {nb}))"))
            }
        }
    }

    /// -1, 0 or 1 by the sign; a zero gives itself and NaN 0 (WGSL's `sign` leaves both open).
    fn sign(&self, x: &str, t: Type) -> R<String> {
        let zero = self.f32s(t, 0.0)?;
        let nan = self.is_nan(x, t)?;
        Ok(format!(
            "select(select(sign({x}), {x}, ({x} == {zero})), {zero}, {nan})"
        ))
    }

    /// The same value with the signedness `signed` states: `bitcast<u32>(x)` of an `i32` read as unsigned.
    fn as_sign(&self, x: &str, t: Type, signed: bool) -> R<String> {
        if t.scalar.is_signed() == signed {
            return Ok(x.into());
        }
        let to = match (t.scalar.bits(), signed) {
            (32, true) => Scalar::I32,
            (32, false) => Scalar::U32,
            (64, true) => Scalar::I64,
            (64, false) => Scalar::U64,
            (16, true) => Scalar::I16,
            (16, false) => Scalar::U16,
            _ => return Err(err(format!("integer arithmetic of {t}"))),
        };
        self.bitcast(x, t, to)
    }

    /// `x` of the signedness `signed` states, cast back to `t`'s.
    fn back_to(&self, x: &str, t: Type, signed: bool) -> R<String> {
        if t.scalar.is_signed() == signed {
            return Ok(x.into());
        }
        Ok(format!("bitcast<{}>({x})", self.ty(t)?))
    }

    /// D3D's `firstbit_hi` (of `u32` bits) or `firstbit_shi` (of `i32` bits) of an integer of type `t`: 31 less WGSL's
    /// `firstLeadingBit`, all ones where that finds none.
    fn first_bit_high(&mut self, x: &str, t: Type, signed: bool) -> R<String> {
        let own = t.scalar;
        let xs = self.as_sign(x, t, signed)?;
        let index = format!("firstLeadingBit({xs})");
        let index = if signed {
            self.tmp(&format!("bitcast<{}>({index})", self.ty(t.with(Scalar::U32))?))
        } else {
            self.tmp(&index)
        };
        let none = self.u32s(t, u32::MAX)?;
        let top = self.u32s(t, 31)?;
        let r = format!("select(({top} - {index}), {none}, ({index} == {none}))");
        if own == Scalar::U32 {
            Ok(r)
        } else {
            Ok(format!("bitcast<{}>({r})", self.ty(t)?))
        }
    }

    fn unary(&mut self, u: Unary, a: ValueId) -> R<String> {
        let at = self.vty(a)?;
        let x = self.get(a)?;
        Ok(match u {
            Unary::FNeg => format!("(-{x})"),
            Unary::INeg => {
                if at.scalar.is_signed() {
                    format!("(-{x})")
                } else {
                    format!("({} - {x})", self.konst(at, at.scalar, 0)?)
                }
            }
            Unary::FAbs => format!("abs({x})"),
            Unary::FFract => format!("fract({x})"),
            Unary::FRound(r) => self.round(&x, at, r)?,
            Unary::FSqrt => format!("sqrt({x})"),
            Unary::FInverseSqrt => format!("inverseSqrt({x})"),
            Unary::FExp2 => format!("exp2({x})"),
            Unary::FLog2 => format!("log2({x})"),
            Unary::FSin => format!("sin({x})"),
            Unary::FCos => format!("cos({x})"),
            Unary::FTan => format!("tan({x})"),
            Unary::FAsin => format!("asin({x})"),
            Unary::FAcos => format!("acos({x})"),
            Unary::FAtan => format!("atan({x})"),
            Unary::FSinh => format!("sinh({x})"),
            Unary::FCosh => format!("cosh({x})"),
            Unary::FTanh => format!("tanh({x})"),
            Unary::FSign => self.sign(&x, at)?,
            Unary::Saturate(nan) => self.saturate(&x, at, nan)?,
            Unary::ClampRange { lo, hi, nan } => self.clamp_range(&x, at, lo, hi, nan)?,
            Unary::Quantize { frac_bits, rounding } => self.quantize(&x, at, frac_bits, rounding)?,
            Unary::Derivative(axis, fineness) => {
                let a = match axis {
                    Axis::X => "dpdx",
                    Axis::Y => "dpdy",
                };
                let f = match fineness {
                    Fineness::Coarse => "Coarse",
                    Fineness::Fine => "Fine",
                    Fineness::Any => "",
                };
                format!("{a}{f}({x})")
            }
            Unary::Not if at.scalar == Scalar::Bool => format!("(!{x})"),
            Unary::Not => format!("(~{x})"),
            Unary::Any => format!("any({x})"),
            Unary::All => format!("all({x})"),
            Unary::IsNan => self.is_nan(&x, at)?,
            Unary::IsInf => self.is_inf(&x, at)?,
            Unary::CountOnes => format!("countOneBits({x})"),
            Unary::BitReverse => format!("reverseBits({x})"),
            Unary::FirstBitHigh => self.first_bit_high(&x, at, false)?,
            Unary::FirstBitHighSigned => self.first_bit_high(&x, at, true)?,
            // WGSL's `firstTrailingBit` gives all ones where no bit is set, as D3D's `firstbit_lo` does
            Unary::FirstBitLow => format!("firstTrailingBit({x})"),
            Unary::F32ToF16 => self.lanewise(Helper::F32ToF16, &[x.as_str()], at, Scalar::U32)?,
            Unary::F16ToF32 => self.lanewise(Helper::F16ToF32, &[x.as_str()], at, Scalar::F32)?,
        })
    }

    /// An integer division or remainder (`op`) with the zero rule; WGSL's own is `ByOne`, and the most negative value by -1
    /// is the IR's (itself, and 0), as WGSL's.
    fn divide(&self, op: &str, a: &str, b: &str, t: Type, zero: OnZero) -> R<String> {
        match zero {
            OnZero::ByOne | OnZero::Undef => Ok(format!("({a} {op} {b})")),
            OnZero::AllOnes => {
                let z = self.konst(t, t.scalar, 0)?;
                let one = self.konst(t, t.scalar, 1)?;
                let ones = self.konst(t, t.scalar, u64::MAX)?;
                Ok(format!(
                    "select(({a} {op} select({b}, {one}, ({b} == {z}))), {ones}, ({b} == {z}))"
                ))
            }
        }
    }

    fn binary(&mut self, b: Binary, l: ValueId, r: ValueId) -> R<String> {
        let at = self.vty(l)?;
        let (x, y) = (self.get(l)?, self.get(r)?);
        let cmp = |c: Cmp| match c {
            Cmp::Eq => "==",
            Cmp::Ne => "!=",
            Cmp::Lt => "<",
            Cmp::Le => "<=",
            Cmp::Gt => ">",
            Cmp::Ge => ">=",
        };
        Ok(match b {
            Binary::FAdd | Binary::IAdd => format!("({x} + {y})"),
            Binary::FSub | Binary::ISub => format!("({x} - {y})"),
            Binary::FMul(MulZero::Ieee) | Binary::IMul => format!("({x} * {y})"),
            Binary::FMul(MulZero::ZeroWins) => {
                let zero = self.f32s(at, 0.0)?;
                format!("select(({x} * {y}), {zero}, (({x} == {zero}) | ({y} == {zero})))")
            }
            Binary::FDiv => format!("({x} / {y})"),
            Binary::FRem => self.lanewise(Helper::Fmod, &[x.as_str(), y.as_str()], at, Scalar::F32)?,
            Binary::UMulHigh | Binary::SMulHigh => {
                let signed = b == Binary::SMulHigh;
                let helper = if signed {
                    Helper::SMulHigh
                } else {
                    Helper::UMulHigh
                };
                let (xs, ys) = (self.as_sign(&x, at, signed)?, self.as_sign(&y, at, signed)?);
                let (xs, ys) = (self.tmp(&xs), self.tmp(&ys));
                let scalar = if signed { Scalar::I32 } else { Scalar::U32 };
                let product = self.lanewise(helper, &[xs.as_str(), ys.as_str()], at, scalar)?;
                self.back_to(&product, at, signed)?
            }
            Binary::UAddCarry => {
                let (one, zero) = (self.konst(at, at.scalar, 1)?, self.konst(at, at.scalar, 0)?);
                format!("select({zero}, {one}, (({x} + {y}) < {x}))")
            }
            Binary::USubBorrow => {
                let (one, zero) = (self.konst(at, at.scalar, 1)?, self.konst(at, at.scalar, 0)?);
                format!("select({zero}, {one}, ({x} < {y}))")
            }
            Binary::SDiv(z) | Binary::SRem(z) | Binary::UDiv(z) | Binary::URem(z) => {
                let signed = matches!(b, Binary::SDiv(_) | Binary::SRem(_));
                let op = if matches!(b, Binary::SDiv(_) | Binary::UDiv(_)) {
                    "/"
                } else {
                    "%"
                };
                let (xs, ys) = (self.as_sign(&x, at, signed)?, self.as_sign(&y, at, signed)?);
                let (xs, ys) = (self.tmp(&xs), self.tmp(&ys));
                let ts = at.with(signed_of(at.scalar, signed)?);
                let q = self.divide(op, &xs, &ys, ts, z)?;
                self.back_to(&q, at, signed)?
            }
            Binary::UMin | Binary::UMax | Binary::SMin | Binary::SMax => {
                let signed = matches!(b, Binary::SMin | Binary::SMax);
                let f = if matches!(b, Binary::UMin | Binary::SMin) {
                    "min"
                } else {
                    "max"
                };
                let (xs, ys) = (self.as_sign(&x, at, signed)?, self.as_sign(&y, at, signed)?);
                let m = format!("{f}({xs}, {ys})");
                self.back_to(&m, at, signed)?
            }
            Binary::Shl => format!("({x} << {y})"),
            Binary::ShrU | Binary::ShrS => {
                let signed = b == Binary::ShrS;
                let xs = self.as_sign(&x, at, signed)?;
                let shifted = format!("({xs} >> {y})");
                self.back_to(&shifted, at, signed)?
            }
            Binary::FMin(nan) => self.min_max("min", &x, &y, at, nan)?,
            Binary::FMax(nan) => self.min_max("max", &x, &y, at, nan)?,
            Binary::FPow => format!("pow({x}, {y})"),
            Binary::FCmp(c) => format!("({x} {} {y})", cmp(c)),
            Binary::SCmp(c) => {
                let (xs, ys) = (self.as_sign(&x, at, true)?, self.as_sign(&y, at, true)?);
                format!("({xs} {} {ys})", cmp(c))
            }
            Binary::UCmp(c) => {
                // equality of any integers and bools is the same compared signed or not
                if at.scalar == Scalar::Bool || matches!(c, Cmp::Eq | Cmp::Ne) {
                    format!("({x} {} {y})", cmp(c))
                } else {
                    let (xs, ys) = (self.as_sign(&x, at, false)?, self.as_sign(&y, at, false)?);
                    format!("({xs} {} {ys})", cmp(c))
                }
            }
            Binary::And => format!("({x} & {y})"),
            Binary::Or => format!("({x} | {y})"),
            Binary::Xor if at.scalar == Scalar::Bool => format!("({x} != {y})"),
            Binary::Xor => format!("({x} ^ {y})"),
        })
    }

    fn ternary(&mut self, op: Ternary, a: ValueId, b: ValueId, c: ValueId) -> R<String> {
        let at = self.vty(a)?;
        let (x, y, z) = (self.get(a)?, self.get(b)?, self.get(c)?);
        Ok(match op {
            Ternary::Fma => format!("fma({x}, {y}, {z})"),
            // WGSL's `extractBits` cuts the field at the top bit as D3D does; it does not take the counts modulo 32, so
            // they are masked first; its counts are one scalar for every lane, so a vector's lanes are taken one at a time
            Ternary::BitExtract { signed } => {
                let xs = self.as_sign(&x, at, signed)?;
                let xs = self.tmp(&xs);
                let own = at.with(signed_of(at.scalar, signed)?);
                let e = self.per_lane(own, |w, k| {
                    let v = w.lane(&xs, own, k)?;
                    let o = w.low_five(&w.lane(&y, at.with(Scalar::U32), k)?);
                    let n = w.low_five(&w.lane(&z, at.with(Scalar::U32), k)?);
                    Ok(format!("extractBits({v}, {o}, {n})"))
                })?;
                self.back_to(&e, at, signed)?
            }
        })
    }

    /// `a` converted to `t`: a float to an integer by `rule`, anything else by WGSL's conversion.
    fn convert(&mut self, a: ValueId, t: Type, rule: FloatToInt) -> R<String> {
        let from = self.vty(a)?;
        let x = self.get(a)?;
        if from.scalar.is_float() && t.scalar.is_integer() {
            self.float_to_int(&x, from, t, rule)
        } else {
            Ok(format!("{}({x})", self.ty(t)?))
        }
    }

    /// D3D's `ftoi` and `ftou`: truncate toward zero, saturate to the integer type's range, NaN gives 0.
    fn float_to_int(&mut self, x: &str, from: Type, to: Type, rule: FloatToInt) -> R<String> {
        let convert = |w: &Self, v: &str| -> R<String> { Ok(format!("{}({v})", w.ty(to)?)) };
        if rule == FloatToInt::Undef {
            return convert(self, x);
        }
        // the largest f32s below 2^31 and 2^32 are 2147483520 and 4294967040 (an f64 holds the integers themselves); from
        // those limits on the result is the type's maximum
        let wide = match from.scalar {
            Scalar::F32 => false,
            Scalar::F64 => true,
            _ => {
                return Err(err(format!(
                    "a saturating conversion from {from} is not written yet"
                )));
            }
        };
        let (low, high, limit, max) = match (to.scalar, wide) {
            (Scalar::I32, false) => (
                -2_147_483_648.0,
                2_147_483_520.0,
                2_147_483_648.0,
                u64::from(i32::MAX.cast_unsigned()),
            ),
            (Scalar::I32, true) => (
                -2_147_483_648.0,
                2_147_483_647.0,
                2_147_483_648.0,
                u64::from(i32::MAX.cast_unsigned()),
            ),
            (Scalar::U32, false) => (0.0, 4_294_967_040.0, 4_294_967_296.0, u64::from(u32::MAX)),
            (Scalar::U32, true) => (0.0, 4_294_967_295.0, 4_294_967_296.0, u64::from(u32::MAX)),
            (other, _) => {
                return Err(err(format!(
                    "a saturating conversion to {} is not written yet",
                    other.name()
                )));
            }
        };
        let float = |w: &Self, v: f64| -> R<String> {
            if wide {
                w.konst(from, Scalar::F64, v.to_bits())
            } else {
                #[allow(clippy::cast_possible_truncation, reason = "each bound is an f32")]
                w.f32s(from, v as f32)
            }
        };
        let clamped = format!("clamp({x}, {}, {})", float(self, low)?, float(self, high)?);
        let converted = convert(self, &clamped)?;
        let too_big = format!("({x} >= {})", float(self, limit)?);
        let max = self.konst(to, to.scalar, max)?;
        let saturated = format!("select({converted}, {max}, {too_big})");
        let nan = self.is_nan(x, from)?;
        let zero = self.konst(to, to.scalar, 0)?;
        Ok(format!("select({saturated}, {zero}, {nan})"))
    }

    /// The same bits in another number of lanes: 32-bit lanes two to each 64-bit lane (the first the low half), or each
    /// 64-bit lane as two 32-bit ones, through 64-bit integers.
    fn bitcast_lanes(&mut self, x: &str, from: Type, to: Type) -> R<String> {
        let (fb, tb) = (from.scalar.bits(), to.scalar.bits());
        if u32::from(from.lanes).saturating_mul(fb) != u32::from(to.lanes).saturating_mul(tb) {
            return Err(err(format!("a bitcast of {from} to {to}")));
        }
        let mut parts = Vec::new();
        match (fb, tb) {
            (32, 64) => {
                let words = self.bitcast(x, from, Scalar::U32)?;
                let u = from.with(Scalar::U32);
                for k in 0..to.lanes {
                    let lo = self.lane(&words, u, k.saturating_mul(2))?;
                    let hi = self.lane(&words, u, k.saturating_mul(2).saturating_add(1))?;
                    let both = format!("((u64({hi}) << 32u) | u64({lo}))");
                    parts.push(format!("bitcast<{}>({both})", text::scalar(to.scalar)?));
                }
            }
            (64, 32) => {
                let words = self.bitcast(x, from, Scalar::U64)?;
                let u = from.with(Scalar::U64);
                for k in 0..from.lanes {
                    let w = self.lane(&words, u, k)?;
                    for half in [format!("u32({w} & 0xfffffffflu)"), format!("u32({w} >> 32u)")] {
                        parts.push(format!("bitcast<{}>({half})", text::scalar(to.scalar)?));
                    }
                }
            }
            _ => return Err(err(format!("a bitcast of {from} to {to} is not written yet"))),
        }
        Ok(match parts.as_slice() {
            [one] => one.clone(),
            _ => format!("{}({})", self.ty(to)?, parts.join(", ")),
        })
    }

    /// Lanes `mask` of an input argument, as the value `t`.
    fn load_input(&mut self, slot: u32, mask: u8, t: Type) -> R<String> {
        let item = self
            .io
            .arguments
            .iter()
            .find(|i| i.varyings.first() == Some(&(slot as usize)))
            .ok_or_else(|| err(format!("input {slot} is not an argument of the entry point")))?;
        let name = item.name.clone();
        let carried = match item.ty {
            iridia_ir::target::IoType::Value(v) => v,
            ref other => return Err(err(format!("input {slot} is {other:?}"))),
        };
        let mut all = name;
        // a flag carried as a bool (WGSL's front facing) read as an integer: all ones where it holds, as Direct3D gives it
        if carried.scalar == ScalarType::Bool && t.scalar != Scalar::Bool {
            if !t.scalar.is_integer() || t.lanes != 1 {
                return Err(err(format!("a flag input read as {t}")));
            }
            let ones = text::literal(t.scalar, u64::MAX)?;
            let zero = text::literal(t.scalar, 0)?;
            all = format!("select({zero}, {ones}, {all})");
        }
        let width = self
            .program
            .interface
            .inputs
            .get(slot as usize)
            .map_or(0, |v| v.value.width);
        let lanes: Vec<u8> = masked(mask).collect();
        Ok(if lanes.len() == usize::from(width) {
            all
        } else {
            format!("{all}.{}", text::swizzle(lanes)?)
        })
    }
}

/// The integer scalar of the width of `s` with the signedness `signed` states.
fn signed_of(s: Scalar, signed: bool) -> R<Scalar> {
    Ok(match (s.bits(), signed) {
        (32, true) => Scalar::I32,
        (32, false) => Scalar::U32,
        (64, true) => Scalar::I64,
        (64, false) => Scalar::U64,
        (16, true) => Scalar::I16,
        (16, false) => Scalar::U16,
        _ => return Err(err(format!("integer arithmetic of {}", s.name()))),
    })
}
