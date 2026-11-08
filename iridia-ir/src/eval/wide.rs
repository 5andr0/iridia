//! The pure operations whose result or an operand is a 64-bit value (`f64`, `i64`, `u64`): a lane of such a value is
//! two words, the low first (lane 0 in words 0 and 1, lane 1 in words 2 and 3, as Direct3D keeps a double in a
//! register's `.xy` and `.zw`), so a 64-bit value has at most two lanes. A bitcast keeps the words. Each operation
//! is its 32-bit rule at 64 bits; a double is never flushed (Direct3D keeps double denormals).

use super::{Lanes, Operands, compare, flushed};
use crate::{Binary, MinMaxNan, MulZero, OnZero, Op, Rounding, SaturateNan, Scalar, Ternary, Type, Unary};
use alloc::{format, string::String};

/// Whether a value of `t` is a 64-bit one.
pub(crate) fn wide(t: Type) -> bool {
    t.scalar.bits() == 64
}

/// The lanes of a value of type `t` as 64-bit numbers: each 64-bit lane from its two words, each 32-bit lane from its
/// word (zero-extended; a signed operation reads it by `signed32`).
fn get(x: &Lanes, t: Type) -> Result<[u64; 4], String> {
    let mut out = [0u64; 4];
    if wide(t) {
        if t.lanes > 2 {
            return Err(format!("{t}: a 64-bit value of more than two lanes is not run"));
        }
        for (k, o) in out.iter_mut().enumerate().take(usize::from(t.lanes)) {
            let at = k.saturating_mul(2);
            let (lo, hi) = (
                x.get(at).copied().unwrap_or(0),
                x.get(at.saturating_add(1)).copied().unwrap_or(0),
            );
            *o = u64::from(lo) | u64::from(hi) << 32;
        }
    } else {
        for (o, w) in out.iter_mut().zip(x).take(usize::from(t.lanes)) {
            *o = u64::from(*w);
        }
    }
    Ok(out)
}

/// `lanes` as a value of type `t`: a 64-bit lane in two words, a 32-bit lane in its low word.
fn put(lanes: &[u64; 4], t: Type) -> Result<Lanes, String> {
    let mut out = [0u32; 4];
    if wide(t) {
        if t.lanes > 2 {
            return Err(format!("{t}: a 64-bit value of more than two lanes is not run"));
        }
        for (k, v) in lanes.iter().enumerate().take(usize::from(t.lanes)) {
            let at = k.saturating_mul(2);
            *out.get_mut(at).ok_or("a lane past four words")? = (*v & 0xffff_ffff) as u32;
            *out.get_mut(at.saturating_add(1))
                .ok_or("a lane past four words")? = (*v >> 32) as u32;
        }
    } else {
        for (o, v) in out.iter_mut().zip(lanes).take(usize::from(t.lanes)) {
            *o = (*v & 0xffff_ffff) as u32;
        }
    }
    Ok(out)
}

/// Each of `n` lanes through `op`.
fn each(n: u8, a: &[u64; 4], op: impl Fn(u64) -> u64) -> [u64; 4] {
    let mut r = [0; 4];
    for (o, x) in r.iter_mut().zip(a).take(usize::from(n)) {
        *o = op(*x);
    }
    r
}

fn each2(n: u8, a: &[u64; 4], b: &[u64; 4], op: impl Fn(u64, u64) -> u64) -> [u64; 4] {
    let mut r = [0; 4];
    for ((o, x), y) in r.iter_mut().zip(a).zip(b).take(usize::from(n)) {
        *o = op(*x, *y);
    }
    r
}

fn double(n: u8, a: &[u64; 4], op: impl Fn(f64) -> f64) -> [u64; 4] {
    each(n, a, |x| op(f64::from_bits(x)).to_bits())
}

fn double2(n: u8, a: &[u64; 4], b: &[u64; 4], op: impl Fn(f64, f64) -> f64) -> [u64; 4] {
    each2(n, a, b, |x, y| op(f64::from_bits(x), f64::from_bits(y)).to_bits())
}

fn is_nan(x: f64) -> bool {
    x.to_bits() & 0x7fff_ffff_ffff_ffff > 0x7ff0_0000_0000_0000
}

fn min_max(a: f64, b: f64, nan: MinMaxNan, max: bool) -> f64 {
    match (nan, is_nan(a), is_nan(b)) {
        (MinMaxNan::Propagate, true, _) | (MinMaxNan::Propagate, _, true) => f64::NAN,
        _ if max => a.max(b),
        _ => a.min(b),
    }
}

/// A lane of a signed 32-bit value read as `i64`.
fn signed32(w: u64) -> i64 {
    i64::from((w & 0xffff_ffff) as u32 as i32)
}

/// The lanes of pure operation `op` of type `t` where it or an operand is 64-bit; `None` for one that is not pure.
/// `flush` flushes an `f32` operand and an `f32` result of a conversion, as the 32-bit operations are flushed.
pub(crate) fn eval(op: &Op, t: Type, ops: &dyn Operands, flush: bool) -> Result<Option<Lanes>, String> {
    let n = t.lanes;
    let read = |v| -> Result<([u64; 4], Type), String> {
        let ty = ops.ty(v)?;
        Ok((get(&ops.arg(v, flush)?, ty)?, ty))
    };
    let out = match op {
        Op::Unary(u, a) => {
            let (x, at) = read(*a)?;
            let m = at.lanes;
            let float = at.scalar == Scalar::F64;
            let bits = at.scalar.bits();
            match u {
                Unary::FNeg if float => each(m, &x, |w| w ^ 1 << 63),
                Unary::FAbs if float => each(m, &x, |w| w & !(1 << 63)),
                Unary::FFract if float => double(m, &x, |f| f - libm::floor(f)),
                Unary::FRound(r) if float => each(m, &x, |bits| {
                    if bits & 0x7fff_ffff_ffff_ffff > 0x7ff0_0000_0000_0000 {
                        bits | 0x0008_0000_0000_0000
                    } else {
                        let round = match r {
                            Rounding::Floor => libm::floor,
                            Rounding::Ceil => libm::ceil,
                            Rounding::Trunc => libm::trunc,
                            Rounding::NearestEven => libm::roundeven,
                            Rounding::NearestAway => libm::round,
                        };
                        round(f64::from_bits(bits)).to_bits()
                    }
                }),
                Unary::FSqrt if float => double(m, &x, libm::sqrt),
                Unary::FSign if float => double(m, &x, |f| {
                    if is_nan(f) {
                        0.0
                    } else if f == 0.0 {
                        f
                    } else {
                        f.signum()
                    }
                }),
                Unary::Saturate(SaturateNan::Zero | SaturateNan::Any) if float => {
                    double(m, &x, |f| if is_nan(f) { 0.0 } else { f.clamp(0.0, 1.0) })
                }
                Unary::IsNan if float => each(m, &x, |w| u64::from(is_nan(f64::from_bits(w)))),
                Unary::IsInf if float => each(m, &x, |w| u64::from(w & !(1 << 63) == 0x7ff0_0000_0000_0000)),
                Unary::INeg if !float => each(m, &x, u64::wrapping_neg),
                Unary::Not if !float => each(m, &x, |w| !w),
                Unary::CountOnes if !float => each(m, &x, |w| u64::from(w.count_ones())),
                Unary::BitReverse if !float => each(m, &x, u64::reverse_bits),
                Unary::FirstBitLow if !float => each(m, &x, |w| {
                    if w == 0 {
                        u64::from(u32::MAX)
                    } else {
                        u64::from(w.trailing_zeros())
                    }
                }),
                Unary::FirstBitHigh if !float => each(m, &x, |w| {
                    if w == 0 {
                        u64::from(u32::MAX)
                    } else {
                        u64::from(w.leading_zeros())
                    }
                }),
                Unary::FirstBitHighSigned if !float => each(m, &x, |w| {
                    let w = if w >> 63 == 1 { !w } else { w };
                    if w == 0 {
                        u64::from(u32::MAX)
                    } else {
                        u64::from(w.leading_zeros())
                    }
                }),
                u => {
                    return Err(format!(
                        "{u:?} of a {bits}-bit {} is not run yet",
                        at.scalar.name()
                    ));
                }
            }
        }
        Op::Binary(b, l, r) => {
            let ((x, at), (y, _)) = (read(*l)?, read(*r)?);
            let m = at.lanes;
            let float = at.scalar == Scalar::F64;
            let signed = at.scalar.is_signed();
            let ordered = |p: u64, q: u64| {
                if float {
                    f64::from_bits(p).partial_cmp(&f64::from_bits(q))
                } else if signed {
                    Some(p.cast_signed().cmp(&q.cast_signed()))
                } else {
                    Some(p.cmp(&q))
                }
            };
            match b {
                Binary::FAdd if float => double2(m, &x, &y, |p, q| p + q),
                Binary::FSub if float => double2(m, &x, &y, |p, q| p - q),
                Binary::FMul(MulZero::Ieee) if float => double2(m, &x, &y, |p, q| p * q),
                Binary::FMul(MulZero::ZeroWins) if float => {
                    double2(m, &x, &y, |p, q| if p == 0.0 || q == 0.0 { 0.0 } else { p * q })
                }
                Binary::FDiv if float => double2(m, &x, &y, |p, q| p / q),
                Binary::FRem if float => double2(m, &x, &y, |p, q| p % q),
                Binary::FMin(nan) if float => double2(m, &x, &y, |p, q| min_max(p, q, *nan, false)),
                Binary::FMax(nan) if float => double2(m, &x, &y, |p, q| min_max(p, q, *nan, true)),
                Binary::FCmp(c) | Binary::SCmp(c) | Binary::UCmp(c) => {
                    let ne = *c == crate::Cmp::Ne;
                    let c = compare(*c);
                    each2(m, &x, &y, |p, q| u64::from(ordered(p, q).map_or(ne, &c)))
                }
                Binary::IAdd if !float => each2(m, &x, &y, u64::wrapping_add),
                Binary::ISub if !float => each2(m, &x, &y, u64::wrapping_sub),
                Binary::IMul if !float => each2(m, &x, &y, u64::wrapping_mul),
                Binary::UMulHigh if !float => each2(m, &x, &y, |p, q| {
                    (u128::from(p).wrapping_mul(u128::from(q)) >> 64) as u64
                }),
                Binary::SMulHigh if !float => each2(m, &x, &y, |p, q| {
                    (i128::from(p.cast_signed()).wrapping_mul(i128::from(q.cast_signed())) >> 64) as u64
                }),
                Binary::UAddCarry if !float => each2(m, &x, &y, |p, q| u64::from(p.checked_add(q).is_none())),
                Binary::USubBorrow if !float => each2(m, &x, &y, |p, q| u64::from(p < q)),
                Binary::SDiv(z) if !float => each2(m, &x, &y, |p, q| match (q, z) {
                    (0, OnZero::AllOnes) => u64::MAX,
                    (0, _) => p,
                    // only the most negative value by -1 overflows: it wraps to itself
                    _ => p
                        .cast_signed()
                        .checked_div(q.cast_signed())
                        .map_or(p, i64::cast_unsigned),
                }),
                Binary::SRem(z) if !float => each2(m, &x, &y, |p, q| match (q, z) {
                    (0, OnZero::AllOnes) => u64::MAX,
                    (0, _) => 0,
                    _ => p
                        .cast_signed()
                        .checked_rem(q.cast_signed())
                        .map_or(0, i64::cast_unsigned),
                }),
                Binary::UDiv(z) if !float => each2(m, &x, &y, |p, q| match (q, z) {
                    (0, OnZero::AllOnes) => u64::MAX,
                    (0, _) => p,
                    _ => p.checked_div(q).unwrap_or(0),
                }),
                Binary::URem(z) if !float => each2(m, &x, &y, |p, q| match (q, z) {
                    (0, OnZero::AllOnes) => u64::MAX,
                    (0, _) => 0,
                    _ => p.checked_rem(q).unwrap_or(0),
                }),
                Binary::UMin if !float => each2(m, &x, &y, u64::min),
                Binary::UMax if !float => each2(m, &x, &y, u64::max),
                Binary::SMin if !float => each2(m, &x, &y, |p, q| {
                    p.cast_signed().min(q.cast_signed()).cast_unsigned()
                }),
                Binary::SMax if !float => each2(m, &x, &y, |p, q| {
                    p.cast_signed().max(q.cast_signed()).cast_unsigned()
                }),
                // the count modulo the operand's width, as for 32 bits
                Binary::Shl if !float => each2(m, &x, &y, |p, q| p.wrapping_shl((q & 63) as u32)),
                Binary::ShrU if !float => each2(m, &x, &y, |p, q| p.wrapping_shr((q & 63) as u32)),
                Binary::ShrS if !float => each2(m, &x, &y, |p, q| {
                    p.cast_signed().wrapping_shr((q & 63) as u32).cast_unsigned()
                }),
                Binary::And if !float => each2(m, &x, &y, |p, q| p & q),
                Binary::Or if !float => each2(m, &x, &y, |p, q| p | q),
                Binary::Xor if !float => each2(m, &x, &y, |p, q| p ^ q),
                b => return Err(format!("{b:?} of a 64-bit {} is not run yet", at.scalar.name())),
            }
        }
        Op::Ternary(Ternary::Fma, a, b, c) => {
            let ((x, at), (y, _), (z, _)) = (read(*a)?, read(*b)?, read(*c)?);
            if at.scalar != Scalar::F64 {
                return Err(format!(
                    "a fused multiply-add of a 64-bit {} is not run",
                    at.scalar.name()
                ));
            }
            let mut out = [0u64; 4];
            for (k, o) in out.iter_mut().enumerate().take(usize::from(at.lanes)) {
                let lane = |l: &[u64; 4]| f64::from_bits(l.get(k).copied().unwrap_or(0));
                *o = libm::fma(lane(&x), lane(&y), lane(&z)).to_bits();
            }
            out
        }
        Op::Select { cond, accept, reject } => {
            let c = ops.get(*cond)?;
            let per_lane = ops.ty(*cond)?.lanes > 1;
            let ((a, _), (r, _)) = (read(*accept)?, read(*reject)?);
            let mut out = [0u64; 4];
            for (i, o) in out.iter_mut().enumerate().take(usize::from(n)) {
                let pick = if per_lane { c.get(i) } else { c.first() }.copied().unwrap_or(0) != 0;
                *o = if pick { a.get(i) } else { r.get(i) }.copied().unwrap_or(0);
            }
            out
        }
        Op::Compose(parts) => {
            let mut out = [0u64; 4];
            let mut at = 0usize;
            for p in parts {
                let (x, pt) = read(*p)?;
                for v in x.iter().take(usize::from(pt.lanes)) {
                    *out.get_mut(at).ok_or("a compose of more than four lanes")? = *v;
                    at = at.saturating_add(1);
                }
            }
            out
        }
        Op::Swizzle(a, lanes) => {
            let (x, _) = read(*a)?;
            let mut out = [0u64; 4];
            for (o, &l) in out.iter_mut().zip(lanes).take(usize::from(n)) {
                *o = *x.get(usize::from(l)).ok_or("a swizzle of a lane past four")?;
            }
            out
        }
        Op::Splat(a) => {
            let (x, _) = read(*a)?;
            each(n, &[x[0]; 4], |v| v)
        }
        Op::Convert(a, _) => {
            let (x, from) = read(*a)?;
            convert(n, &x, from.scalar, t.scalar)?
        }
        // the words are the value's either way
        Op::Bitcast(a) => return Ok(Some(ops.get(*a)?)),
        Op::Dot(..) => return Err("a dot product of 64-bit values is not run yet".into()),
        _ => return Ok(None),
    };
    let mut lanes = put(&out, t)?;
    if flush && t.scalar == Scalar::F32 && matches!(op, Op::Convert(..)) {
        lanes = lanes.map(flushed);
    }
    Ok(Some(lanes))
}

/// A numeric conversion of `n` lanes where one side is 64-bit: float to integer truncated toward zero, clamped to the
/// type's range and NaN 0 (Direct3D's `dtoi` and `dtou`; under `FloatToInt::Undef` that is one of the results it
/// allows), integer to float to nearest even, `f64` to `f32` to nearest even and `f32` to `f64` exact, an integer to a
/// wider one extended by its own sign and to a narrower one cut to its low bits.
fn convert(n: u8, x: &[u64; 4], from: Scalar, to: Scalar) -> Result<[u64; 4], String> {
    if converted(0, from, to).is_none() {
        return Err(format!(
            "a conversion of {} to {} is not run yet",
            from.name(),
            to.name()
        ));
    }
    Ok(each(n, x, |w| converted(w, from, to).unwrap_or(0)))
}

/// One lane of [`convert`]; `None` for a pair of types it does not take. Rust's `as` from a float truncates toward
/// zero, saturates and gives 0 for NaN; to a float it rounds to nearest even.
#[allow(clippy::cast_possible_truncation)] // the saturating and rounding conversions are the operation
fn converted(w: u64, from: Scalar, to: Scalar) -> Option<u64> {
    use Scalar::{F32, F64, I32, I64, U32, U64};
    let low = (w & 0xffff_ffff) as u32;
    let float = if from == F64 {
        f64::from_bits(w)
    } else {
        f64::from(f32::from_bits(low))
    };
    Some(match (from, to) {
        (F64 | F32, F64) => float.to_bits(),
        (F64, F32) => u64::from((float as f32).to_bits()),
        (F64 | F32, I32) => u64::from((float as i32).cast_unsigned()),
        (F64 | F32, U32) => u64::from(float as u32),
        (F64 | F32, I64) => (float as i64).cast_unsigned(),
        (F64 | F32, U64) => float as u64,
        (I32, F64) => (signed32(w) as f64).to_bits(),
        (U32, F64) => f64::from(low).to_bits(),
        (I64, F64) => (w.cast_signed() as f64).to_bits(),
        (U64, F64) => (w as f64).to_bits(),
        (I64, F32) => u64::from((w.cast_signed() as f32).to_bits()),
        (U64, F32) => u64::from((w as f32).to_bits()),
        (I32, I64 | U64) => signed32(w).cast_unsigned(),
        (U32, I64 | U64) | (I64 | U64, I32 | U32) => u64::from(low),
        (I64 | U64, I64 | U64) => w,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cmp, FloatToInt, ValueId};
    use alloc::{vec, vec::Vec};

    /// Operands given by their lanes and types, value `k` the `k`-th.
    struct Given(Vec<(Lanes, Type)>);
    impl Operands for Given {
        fn get(&self, v: ValueId) -> Result<Lanes, String> {
            self.0
                .get(v.0 as usize)
                .map(|g| g.0)
                .ok_or_else(|| "no such value".into())
        }
        fn ty(&self, v: ValueId) -> Result<Type, String> {
            self.0
                .get(v.0 as usize)
                .map(|g| g.1)
                .ok_or_else(|| "no such value".into())
        }
    }

    fn words(x: u64) -> [u32; 2] {
        [(x & 0xffff_ffff) as u32, (x >> 32) as u32]
    }
    fn double(f: f64) -> (Lanes, Type) {
        let [lo, hi] = words(f.to_bits());
        ([lo, hi, 0, 0], Type::vector(Scalar::F64, 1))
    }
    fn long(x: u64, s: Scalar) -> (Lanes, Type) {
        let [lo, hi] = words(x);
        ([lo, hi, 0, 0], Type::vector(s, 1))
    }
    fn run(op: Op, t: Type, given: Vec<(Lanes, Type)>) -> Lanes {
        eval(&op, t, &Given(given), false).unwrap().unwrap()
    }
    fn wide_of(l: Lanes) -> u64 {
        u64::from(l[0]) | u64::from(l[1]) << 32
    }

    #[test]
    fn a_double_to_an_integer_truncates_saturates_and_gives_zero_for_nan() {
        let to = |f: f64, s: Scalar| {
            run(
                Op::Convert(ValueId(0), FloatToInt::Saturate),
                Type::vector(s, 1),
                vec![double(f)],
            )[0]
        };
        assert_eq!(to(-2.75, Scalar::I32), (-2i32).cast_unsigned());
        assert_eq!(to(3.0e10, Scalar::I32), i32::MAX.cast_unsigned());
        assert_eq!(to(-3.0e10, Scalar::I32), i32::MIN.cast_unsigned());
        assert_eq!(to(f64::NAN, Scalar::I32), 0);
        assert_eq!(to(-1.0, Scalar::U32), 0);
        assert_eq!(to(5.0e9, Scalar::U32), u32::MAX);
        // to nearest even: a tie of two f32s
        let tie = f64::from(1.0f32) + f64::from(f32::EPSILON) / 2.0;
        assert_eq!(to(tie, Scalar::F32), 1.0f32.to_bits());
    }

    #[test]
    fn a_double_keeps_its_two_words_through_a_bitcast_and_back() {
        let x = double(-1234.5678e-300);
        let as_words = run(Op::Bitcast(ValueId(0)), Type::vector(Scalar::U32, 2), vec![x]);
        assert_eq!(as_words, x.0);
        let back = run(
            Op::Bitcast(ValueId(0)),
            Type::vector(Scalar::F64, 1),
            vec![(as_words, Type::vector(Scalar::U32, 2))],
        );
        assert_eq!(back, x.0);
    }

    #[test]
    fn a_wide_integer_division_by_zero_and_overflow_follow_the_rule() {
        let i64t = Type::vector(Scalar::I64, 1);
        let div = |p: i64, q: i64, z| {
            wide_of(run(
                Op::Binary(Binary::SDiv(z), ValueId(0), ValueId(1)),
                i64t,
                vec![
                    long(p.cast_unsigned(), Scalar::I64),
                    long(q.cast_unsigned(), Scalar::I64),
                ],
            ))
        };
        assert_eq!(div(7, 0, OnZero::AllOnes), u64::MAX);
        assert_eq!(div(7, 0, OnZero::ByOne), 7);
        assert_eq!(div(i64::MIN, -1, OnZero::AllOnes), i64::MIN.cast_unsigned());
        assert_eq!(div(-7, 2, OnZero::AllOnes), (-3i64).cast_unsigned());
        let shl = wide_of(run(
            Op::Binary(Binary::Shl, ValueId(0), ValueId(1)),
            Type::vector(Scalar::U64, 1),
            vec![long(1, Scalar::U64), long(65, Scalar::U64)],
        ));
        assert_eq!(shl, 2, "the count modulo 64");
    }

    #[test]
    fn a_comparison_of_doubles_gives_bools_and_nan_compares_unordered() {
        let cmp = |c, p: f64, q: f64| {
            run(
                Op::Binary(Binary::FCmp(c), ValueId(0), ValueId(1)),
                Type::vector(Scalar::Bool, 1),
                vec![double(p), double(q)],
            )[0]
        };
        assert_eq!(cmp(Cmp::Lt, 1.0, 2.0), 1);
        assert_eq!(cmp(Cmp::Eq, f64::NAN, f64::NAN), 0);
        assert_eq!(cmp(Cmp::Ne, f64::NAN, 1.0), 1);
    }
}
