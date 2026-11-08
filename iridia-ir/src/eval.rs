//! Pure operation evaluation by the IR reference semantics, shared by execution and material specialisation.
//! Arithmetic uses one portable `libm` backend on every target; transcendental functions compute in `f64` and round
//! once to `f32`. NaN payloads and signs are unspecified. Values are packed as four words (two words per 64-bit lane).

use crate::{
    Binary, Cmp, FloatToInt, MinMaxNan, MulZero, OnZero, Op, Rounding, SaturateNan, Scalar, Ternary, Type,
    Unary, ValueId,
};
use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
mod wide;

/// A value's lanes as 32-bit words (a bool is 0 or 1); a 64-bit lane occupies two words, low first.
pub type Lanes = [u32; 4];

/// An `f32` word with a subnormal made a zero of its sign.
pub fn flush_denormal(w: u32) -> u32 {
    if w & 0x7f80_0000 == 0 { w & 0x8000_0000 } else { w }
}
use flush_denormal as flushed;

fn narrow(v: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)] // the rounding is the operation
    let r = v as f32;
    r
}

/// `count` bits of `a` from bit `offset` (both modulo 32), cut at bit 31, extended by sign or zero.
fn bit_extract(a: u32, offset: u32, count: u32, signed: bool) -> u32 {
    let (offset, count) = (offset % 32, count % 32);
    if count == 0 {
        return 0;
    }
    // the field's width, 1 to 31 bits: moved to the top, then back down with the sign or zeros
    let width = count.min(32u32.wrapping_sub(offset));
    let top = (a >> offset).wrapping_shl(32u32.wrapping_sub(width));
    let down = 32u32.wrapping_sub(width);
    if signed {
        top.cast_signed().wrapping_shr(down).cast_unsigned()
    } else {
        top.wrapping_shr(down)
    }
}

/// `count` low bits of `insert` placed into `base` from bit `offset` (both modulo 32), cut at bit 31.
fn bit_insert(base: u32, insert: u32, offset: u32, count: u32) -> u32 {
    let (offset, count) = (offset % 32, count % 32);
    let field = 1u32.wrapping_shl(count).wrapping_sub(1).wrapping_shl(offset);
    base & !field | insert.wrapping_shl(offset) & field
}

/// The binary16 nearest an `f32`, ties to even, in the low 16 bits: a NaN is the quiet NaN 0x7e00, a value past the
/// range an infinity, and a value below the smallest normal a denormal or a zero. IEEE 754 leaves the sign and payload
/// of an operation's NaN open (an operation on two NaNs gives either), so the half keeps neither: what the conversion
/// gives depends on the value alone, and a pass that reorders an operation's operands leaves it.
fn f32_to_f16(w: u32) -> u32 {
    let sign = w >> 16 & 0x8000;
    let exp = w >> 23 & 0xff;
    let man = w & 0x7f_ffff;
    if exp == 0xff {
        return if man == 0 { sign | 0x7c00 } else { 0x7e00 };
    }
    // binary16's biased exponent is f32's less 112
    if exp >= 143 {
        return sign | 0x7c00;
    }
    if exp < 102 {
        return sign;
    }
    // a normal half keeps the exponent and drops 13 bits; below one, the implicit bit joins a significand shifted
    // further, a denormal
    let (sig, shift, high) = if exp > 112 {
        (man, 13u32, exp.wrapping_sub(112) << 10)
    } else {
        (man | 0x80_0000, 126u32.wrapping_sub(exp), 0)
    };
    let half = sig >> shift;
    let rest = sig & 1u32.wrapping_shl(shift).wrapping_sub(1);
    let mid = 1u32.wrapping_shl(shift.wrapping_sub(1));
    let up = u32::from(rest > mid || (rest == mid && half & 1 == 1));
    // a carry out of the significand moves into the exponent, to an infinity at the top
    sign | (high | half).wrapping_add(up)
}

/// The `f32` of the binary16 in the low 16 bits.
fn f16_to_f32(w: u32) -> u32 {
    let sign = (w & 0x8000) << 16;
    let exp = w >> 10 & 0x1f;
    let man = w & 0x3ff;
    match exp {
        0 if man == 0 => sign,
        0 => {
            // a denormal, normalised: its leading bit becomes the implicit one
            let lead = man.leading_zeros().wrapping_sub(21);
            let e = 113u32.wrapping_sub(lead);
            sign | e << 23 | (man.wrapping_shl(lead) & 0x3ff) << 13
        }
        0x1f => sign | 0x7f80_0000 | man << 13,
        _ => sign | exp.wrapping_add(112) << 23 | man << 13,
    }
}

fn fb(x: u32) -> f32 {
    f32::from_bits(x)
}

/// Each lane of `n` through `op`.
fn each(n: u8, a: &Lanes, op: impl Fn(u32) -> u32) -> Lanes {
    let mut r = [0; 4];
    for (o, x) in r.iter_mut().zip(a).take(usize::from(n)) {
        *o = op(*x);
    }
    r
}
fn each2(n: u8, a: &Lanes, b: &Lanes, op: impl Fn(u32, u32) -> u32) -> Lanes {
    let mut r = [0; 4];
    for ((o, x), y) in r.iter_mut().zip(a).zip(b).take(usize::from(n)) {
        *o = op(*x, *y);
    }
    r
}
fn float(n: u8, a: &Lanes, op: impl Fn(f32) -> f32) -> Lanes {
    each(n, a, |x| op(fb(x)).to_bits())
}
fn float2(n: u8, a: &Lanes, b: &Lanes, op: impl Fn(f32, f32) -> f32) -> Lanes {
    each2(n, a, b, |x, y| op(fb(x), fb(y)).to_bits())
}
fn compare(c: Cmp) -> impl Fn(core::cmp::Ordering) -> bool {
    use core::cmp::Ordering::*;
    move |o| match c {
        Cmp::Eq => o == Equal,
        Cmp::Ne => o != Equal,
        Cmp::Lt => o == Less,
        Cmp::Le => o != Greater,
        Cmp::Gt => o == Greater,
        Cmp::Ge => o != Less,
    }
}
fn is_nan(x: f32) -> bool {
    x.to_bits() & 0x7fff_ffff > 0x7f80_0000
}
fn min_max(a: f32, b: f32, nan: MinMaxNan, max: bool) -> f32 {
    match (nan, is_nan(a), is_nan(b)) {
        (MinMaxNan::Propagate, true, _) | (MinMaxNan::Propagate, _, true) => f32::NAN,
        _ if max => a.max(b),
        _ => a.min(b),
    }
}

/// Whether values of `t` are run: a bool, a 32-bit scalar or vector, or a 64-bit one of at most two lanes (two words
/// a lane, the low first). Values of 8 or 16 bits are not.
pub fn runs(t: Type) -> bool {
    match t.scalar.bits() {
        32 => true,
        64 => t.lanes <= 2,
        _ => t.scalar == Scalar::Bool,
    }
}

/// The words of a literal of type `t`: each 64-bit lane in two words, the low first; each other lane in its word.
pub fn literal_words(bits: &[u64; 4], t: Type) -> Lanes {
    let mut out = [0u32; 4];
    if t.scalar.bits() == 64 {
        for ([lo, hi], b) in out.as_chunks_mut::<2>().0.iter_mut().zip(bits) {
            *lo = (b & 0xffff_ffff) as u32;
            *hi = (b >> 32) as u32;
        }
    } else {
        for (o, b) in out.iter_mut().zip(bits) {
            *o = (b & 0xffff_ffff) as u32;
        }
    }
    out
}

/// What an operation reads: its operands' lanes and types.
pub trait Operands {
    /// The lanes of value `v`.
    fn get(&self, v: ValueId) -> Result<Lanes, String>;
    /// The type of value `v`.
    fn ty(&self, v: ValueId) -> Result<Type, String>;
    /// An operand of an arithmetic operation: its lanes, flushed where `flush` holds and it is an `f32`.
    fn arg(&self, v: ValueId, flush: bool) -> Result<Lanes, String> {
        let x = self.get(v)?;
        Ok(if flush && self.ty(v)?.scalar == Scalar::F32 {
            x.map(flushed)
        } else {
            x
        })
    }
}

/// The lanes of a pure operation of type `t` (one that reads only its operands: the arithmetic, `select`, the lane
/// moves, `dot`, the conversions and bitcasts), by the IR's definition; `None` for an operation that is not pure (a
/// literal, a read, a derivative). Where `flush` holds, the `f32` operands and result of each float operation (the sign
/// operations among them, as a Direct3D `mov` with a source modifier) are flushed to zero (sign kept); a move, select
/// or bitcast keeps its words. A 64-bit value (at most two lanes) is two words a
/// lane, the low first; values of 8 or 16 bits are refused. It is [`Prepared::run`] of the operation prepared.
pub fn eval(op: &Op, t: Type, ops: &dyn Operands, flush: bool) -> Result<Option<Lanes>, String> {
    Prepared::new(op, t, |v| ops.ty(v)).run(|v| ops.get(v), flush)
}

/// A pure operation of a fixed type, what [`eval`] computes with every lookup of a type and every choice that depends
/// only on the types made once: an executor that runs an operation many times prepares it once.
#[derive(Clone, Debug)]
pub struct Prepared {
    kernel: Kernel,
    /// The operands, in the order the kernel reads them: the first `arity` of them.
    src: [ValueId; 4],
    arity: usize,
    /// Which operands are the `f32` operands of a float operation (flushed where the run flushes).
    flush_in: [bool; 4],
    /// Whether a flushing run flushes the result: a float operation's (not a sign operation's or a move's) of `f32`.
    flush_out: bool,
}

/// How a binary operation orders its operands' lanes.
#[derive(Clone, Copy, Debug)]
enum Order {
    Float,
    Signed,
    Unsigned,
}

/// A 32-bit conversion.
#[derive(Clone, Copy, Debug)]
enum Conversion {
    /// `f32` to `i32`, truncated toward zero and saturated, NaN 0.
    FloatToSigned,
    /// `f32` to `u32`, the same.
    FloatToUnsigned,
    /// `i32` to `f32`, to nearest even.
    SignedToFloat,
    /// `u32` to `f32`, to nearest even.
    UnsignedToFloat,
    /// Between `i32` and `u32`: the words.
    Words,
}

/// What a prepared operation does.
#[derive(Clone, Debug)]
enum Kernel {
    /// A 32-bit operation: a function of its operands' lanes.
    Math(Math),
    /// A 64-bit operation, run by `wide::eval` with its operands' types.
    Wide {
        op: Box<Op>,
        t: Type,
        types: Box<[(ValueId, Result<Type, String>)]>,
    },
    /// An operation that fails where it runs, with this error.
    Fail(String),
    /// An operation that is not pure.
    NotPure,
}

/// A 32-bit operation on its operands' lanes (`m` the lanes of its first operand, `n` those of its result), the types
/// it depends on read once.
#[derive(Clone, Debug)]
enum Math {
    Unary {
        u: Unary,
        m: u8,
        /// A `not` of bools: a logical one.
        logical: bool,
    },
    Binary {
        b: Binary,
        m: u8,
        order: Order,
    },
    Ternary {
        op: Ternary,
        m: u8,
    },
    BitInsert {
        n: u8,
    },
    Select {
        n: u8,
        /// Whether the condition has a lane per result lane (else one for all).
        per_lane: bool,
    },
    /// Each part's lanes, at most four lanes in all.
    Compose(Box<[u8]>),
    Swizzle {
        n: u8,
        lanes: [u8; 4],
    },
    Splat {
        n: u8,
    },
    Dot {
        m: u8,
    },
    Convert {
        n: u8,
        conversion: Conversion,
    },
    Bitcast,
}

/// The operands and their types of a 64-bit operation, as `wide::eval` reads them.
struct Typed<'a, G> {
    get: &'a G,
    types: &'a [(ValueId, Result<Type, String>)],
}

impl<G: Fn(ValueId) -> Result<Lanes, String>> Operands for Typed<'_, G> {
    fn get(&self, v: ValueId) -> Result<Lanes, String> {
        (self.get)(v)
    }
    fn ty(&self, v: ValueId) -> Result<Type, String> {
        self.types
            .iter()
            .find(|(w, _)| *w == v)
            .map_or_else(|| Err(format!("%{} has no type", v.0)), |(_, t)| t.clone())
    }
}

/// `n` lanes, each from `accept` or `reject` by the condition's lane (or the first for all, where it is not `per_lane`).
#[inline(always)]
fn select(n: u8, per_lane: bool, c: &Lanes, a: &Lanes, r: &Lanes) -> Lanes {
    let mut out = [0; 4];
    for (i, o) in out.iter_mut().enumerate().take(usize::from(n)) {
        let pick = if per_lane {
            c.get(i).copied()
        } else {
            c.first().copied()
        }
        .unwrap_or(0)
            != 0;
        *o = if pick { a.get(i) } else { r.get(i) }.copied().unwrap_or(0);
    }
    out
}

/// The words of each part (its first `lanes` words), one after the other; `part` gives the words of part `i`.
#[inline(always)]
fn compose(parts: &[u8], part: impl Fn(usize) -> Lanes) -> Lanes {
    let mut out = [0; 4];
    let mut at = 0usize;
    for (i, lanes) in parts.iter().enumerate() {
        for w in part(i).iter().take(usize::from(*lanes)) {
            if let Some(o) = out.get_mut(at) {
                *o = *w;
            }
            at = at.saturating_add(1);
        }
    }
    out
}

/// The `n` lanes of `x` the pattern names.
#[inline(always)]
fn swizzle(n: u8, lanes: &[u8; 4], x: &Lanes) -> Lanes {
    let mut out = [0; 4];
    for (o, &l) in out.iter_mut().zip(lanes).take(usize::from(n)) {
        *o = x.get(usize::from(l)).copied().unwrap_or(0);
    }
    out
}

/// `n` lanes of `x`'s first.
#[inline(always)]
fn splat(n: u8, x: &Lanes) -> Lanes {
    let mut out = [0; 4];
    for o in out.iter_mut().take(usize::from(n)) {
        *o = x[0];
    }
    out
}

impl Math {
    /// The result's lanes from the operands' lanes `x` (in the order the operation reads them, `f32` operands already
    /// flushed where the run flushes); `flush` is the run's.
    #[inline(always)]
    fn compute(&self, x: &[Lanes], flush: bool) -> Lanes {
        let lane = |i: usize| x.get(i).copied().unwrap_or([0; 4]);
        match self {
            Math::Unary { u, m, logical } => unary(*u, lane(0), *m, *logical),
            Math::Binary { b, m, order } => binary(*b, lane(0), lane(1), *m, *order),
            Math::Ternary { op, m } => {
                let (x, y, z) = (lane(0), lane(1), lane(2));
                let mut out = [0; 4];
                for (k, o) in out.iter_mut().enumerate().take(usize::from(*m)) {
                    let at = |l: &Lanes| l.get(k).copied().unwrap_or(0);
                    *o = match op {
                        Ternary::Fma => libm::fmaf(fb(at(&x)), fb(at(&y)), fb(at(&z))).to_bits(),
                        Ternary::BitExtract { signed } => bit_extract(at(&x), at(&y), at(&z), *signed),
                    };
                }
                out
            }
            Math::BitInsert { n } => {
                let (b, i, o, c) = (lane(0), lane(1), lane(2), lane(3));
                let mut out = [0; 4];
                for (k, r) in out.iter_mut().enumerate().take(usize::from(*n)) {
                    let at = |l: &Lanes| l.get(k).copied().unwrap_or(0);
                    *r = bit_insert(at(&b), at(&i), at(&o), at(&c));
                }
                out
            }
            Math::Select { n, per_lane } => select(*n, *per_lane, &lane(0), &lane(1), &lane(2)),
            Math::Compose(parts) => compose(parts, lane),
            Math::Swizzle { n, lanes } => swizzle(*n, lanes, &lane(0)),
            Math::Splat { n } => splat(*n, &lane(0)),
            Math::Dot { m } => {
                let (x, y) = (lane(0), lane(1));
                // each step is rounded, and flushed where the run flushes
                let step = |f: f32| if flush { fb(flushed(f.to_bits())) } else { f };
                let mut acc = 0.0f32;
                for (i, (p, q)) in x.iter().zip(&y).take(usize::from(*m)).enumerate() {
                    let prod = step(fb(*p) * fb(*q));
                    acc = if i == 0 { prod } else { step(acc + prod) };
                }
                [acc.to_bits(), 0, 0, 0]
            }
            Math::Convert { n, conversion } => {
                let x = lane(0);
                match conversion {
                    // Rust's `as` truncates toward zero, saturates and gives 0 for NaN: D3D's ftoi and ftou
                    Conversion::FloatToSigned => each(*n, &x, |w| {
                        #[allow(clippy::cast_possible_truncation)] // the saturation is the operation
                        let r = fb(w) as i32;
                        r.cast_unsigned()
                    }),
                    Conversion::FloatToUnsigned => each(*n, &x, |w| {
                        #[allow(clippy::cast_possible_truncation)]
                        let r = fb(w) as u32;
                        r
                    }),
                    Conversion::SignedToFloat => each(*n, &x, |w| {
                        #[allow(clippy::cast_precision_loss)] // to nearest even, the conversion's rule
                        let r = w.cast_signed() as f32;
                        r.to_bits()
                    }),
                    Conversion::UnsignedToFloat => each(*n, &x, |w| {
                        #[allow(clippy::cast_precision_loss)]
                        let r = w as f32;
                        r.to_bits()
                    }),
                    Conversion::Words => x,
                }
            }
            Math::Bitcast => lane(0),
        }
    }
}

impl Prepared {
    /// Prepare `op`, of type `t`, its operands' types given by `ty`. Preparing never fails: an operation that `eval`
    /// would refuse (a type it does not run, a type `ty` does not give) fails where it runs, with `eval`'s error.
    pub fn new(op: &Op, t: Type, ty: impl Fn(ValueId) -> Result<Type, String>) -> Prepared {
        let none = ValueId(0);
        let mut p = Prepared {
            kernel: Kernel::NotPure,
            src: [none; 4],
            arity: 0,
            flush_in: [false; 4],
            flush_out: false,
        };
        match kernel(op, t, &ty) {
            Ok(Kernel::Math(math)) => {
                let src = operands(op, &ty);
                p.arity = src.len().min(4);
                for ((s, f), (v, f32)) in p.src.iter_mut().zip(p.flush_in.iter_mut()).zip(src) {
                    *s = v;
                    *f = f32 && flushes_operands(op);
                }
                p.kernel = Kernel::Math(math);
            }
            Ok(other) => p.kernel = other,
            Err(e) => p.kernel = Kernel::Fail(e),
        }
        // the float operations whose result a flushing run flushes: not the sign operations and moves
        let arithmetic = match op {
            Op::Unary(u, _) => !matches!(u, Unary::FNeg | Unary::FAbs),
            Op::Binary(..) | Op::Ternary(..) | Op::Dot(..) | Op::Convert(..) => true,
            _ => false,
        };
        p.flush_out = arithmetic && t.scalar == Scalar::F32;
        p
    }

    /// Whether the operation is pure: [`Prepared::run`] gives its lanes (or fails), not `None`.
    pub fn is_pure(&self) -> bool {
        !matches!(self.kernel, Kernel::NotPure)
    }

    /// The operation's lanes, its operands' lanes given by `get`; `None` for an operation that is not pure. Where
    /// `flush` holds, `f32` denormals are flushed as [`eval`] says.
    #[inline]
    pub fn run(
        &self,
        get: impl Fn(ValueId) -> Result<Lanes, String>,
        flush: bool,
    ) -> Result<Option<Lanes>, String> {
        match &self.kernel {
            Kernel::Math(math) => {
                let mut x = [[0; 4]; 4];
                for ((x, v), f) in x.iter_mut().zip(self.src).zip(self.flush_in).take(self.arity) {
                    let lanes = get(v)?;
                    *x = if flush && f { lanes.map(flushed) } else { lanes };
                }
                let out = math.compute(x.get(..self.arity).unwrap_or(&[]), flush);
                Ok(Some(if flush && self.flush_out {
                    out.map(flushed)
                } else {
                    out
                }))
            }
            Kernel::Wide { op, t, types } => wide::eval(op, *t, &Typed { get: &get, types }, flush),
            Kernel::Fail(e) => Err(e.clone()),
            Kernel::NotPure => Ok(None),
        }
    }

    /// The operation for each lane of `active`, in a register file of `stride` lanes to a value: value `v`'s lane `k`
    /// is `regs[v * stride + k]`, and each lane's result goes to `regs[dst + k]` (`dst` is the result's first lane).
    /// Each lane is what [`Prepared::run`] gives it, the operation's own choice made once for all the lanes. Whether
    /// the operation is pure: `false`, with nothing written, where it is not.
    pub fn run_lanes(
        &self,
        regs: &mut [Lanes],
        stride: usize,
        dst: usize,
        active: impl Fn(usize) -> bool,
        flush: bool,
    ) -> Result<bool, String> {
        let first = |i: usize| {
            self.src
                .get(i)
                .map_or(0, |v| (v.0 as usize).saturating_mul(stride))
        };
        let to = Lanewise {
            regs,
            stride,
            dst,
            active,
        };
        let flush_in = |i: usize| flush && self.flush_in.get(i).copied().unwrap_or(false);
        let flush_out = flush && self.flush_out;
        match &self.kernel {
            Kernel::Math(Math::Unary { u, m, logical }) => unary_arms(
                *u,
                *m,
                *logical,
                Lanes1 {
                    src: first(0),
                    m: *m,
                    flush: flush_in(0),
                    flush_out,
                    to,
                },
            )?,
            Kernel::Math(Math::Binary { b, m, order }) => binary_arms(
                *b,
                *order,
                Lanes2 {
                    src: [first(0), first(1)],
                    m: *m,
                    flush: [flush_in(0), flush_in(1)],
                    flush_out,
                    to,
                },
            )?,
            // the moves and the select: no flushing, each its own loop
            Kernel::Math(Math::Bitcast) => to.drive([first(0)], [false], false, |[x]| x)?,
            Kernel::Math(Math::Swizzle { n, lanes }) => {
                to.drive([first(0)], [false], false, |[x]| swizzle(*n, lanes, &x))?;
            }
            Kernel::Math(Math::Splat { n }) => to.drive([first(0)], [false], false, |[x]| splat(*n, &x))?,
            Kernel::Math(Math::Select { n, per_lane }) => {
                to.drive([first(0), first(1), first(2)], [false; 3], false, |[c, a, r]| {
                    select(*n, *per_lane, &c, &a, &r)
                })?;
            }
            Kernel::Math(Math::Compose(parts)) => match parts.len() {
                1 => to.drive([first(0)], [false], false, |[a]| compose(parts, |_| a))?,
                2 => to.drive([first(0), first(1)], [false; 2], false, |x| {
                    compose(parts, |i| x.get(i).copied().unwrap_or([0; 4]))
                })?,
                3 => to.drive([first(0), first(1), first(2)], [false; 3], false, |x| {
                    compose(parts, |i| x.get(i).copied().unwrap_or([0; 4]))
                })?,
                _ => to.drive([first(0), first(1), first(2), first(3)], [false; 4], false, |x| {
                    compose(parts, |i| x.get(i).copied().unwrap_or([0; 4]))
                })?,
            },
            Kernel::Math(math) => {
                macro_rules! lanes {
                    ($n:literal) => {
                        to.drive(
                            core::array::from_fn::<_, $n, _>(first),
                            core::array::from_fn::<_, $n, _>(flush_in),
                            flush_out,
                            |x| math.compute(&x, flush),
                        )?
                    };
                }
                match self.arity {
                    1 => lanes!(1),
                    2 => lanes!(2),
                    3 => lanes!(3),
                    _ => lanes!(4),
                }
            }
            Kernel::NotPure => return Ok(false),
            // a 64-bit operation, or one that fails, a lane at a time
            Kernel::Wide { .. } | Kernel::Fail(_) => {
                let Lanewise { regs, active, .. } = to;
                for k in 0..stride {
                    if !active(k) {
                        continue;
                    }
                    let out = self
                        .run(
                            |v| {
                                let index = (v.0 as usize).saturating_mul(stride).saturating_add(k);
                                regs.get(index)
                                    .copied()
                                    .ok_or_else(|| no_register(index, regs.len()))
                            },
                            flush,
                        )?
                        .ok_or("internal error: an operation that is not pure")?;
                    let index = dst.saturating_add(k);
                    let len = regs.len();
                    *regs.get_mut(index).ok_or_else(|| no_register(index, len))? = out;
                }
            }
        }
        Ok(true)
    }
}

/// The operands of a 32-bit operation in the order its [`Math`] reads them, each with whether it is an `f32`.
fn operands(op: &Op, ty: &dyn Fn(ValueId) -> Result<Type, String>) -> Vec<(ValueId, bool)> {
    let list: Vec<ValueId> = match op {
        Op::Unary(_, a) | Op::Swizzle(a, _) | Op::Splat(a) | Op::Convert(a, _) | Op::Bitcast(a) => vec![*a],
        Op::Binary(_, a, b) | Op::Dot(a, b) => vec![*a, *b],
        Op::Ternary(_, a, b, c) => vec![*a, *b, *c],
        Op::BitInsert {
            base,
            insert,
            offset,
            count,
        } => vec![*base, *insert, *offset, *count],
        Op::Select { cond, accept, reject } => vec![*cond, *accept, *reject],
        Op::Compose(parts) => parts.clone(),
        _ => Vec::new(),
    };
    list.into_iter()
        .map(|v| (v, ty(v).is_ok_and(|t| t.scalar == Scalar::F32)))
        .collect()
}

/// Whether a flushing run flushes the `f32` operands of `op`: the operations that `arg` its operands (the sign
/// operations among them), not the moves.
fn flushes_operands(op: &Op) -> bool {
    matches!(
        op,
        Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) | Op::Dot(..) | Op::Convert(..)
    )
}

/// What `op` of type `t` does, its operands' types given by `ty`; an error for an operation `eval` refuses whatever
/// its operands' lanes.
fn kernel(op: &Op, t: Type, ty: &dyn Fn(ValueId) -> Result<Type, String>) -> Result<Kernel, String> {
    // the operand whose type tells a 64-bit operation from a 32-bit one (the operands of each are of one width, but
    // for a conversion's and a bitcast's result)
    let first = match op {
        Op::Unary(_, a)
        | Op::Binary(_, a, _)
        | Op::Ternary(_, a, _, _)
        | Op::Select { accept: a, .. }
        | Op::Swizzle(a, _)
        | Op::Splat(a)
        | Op::Dot(a, _)
        | Op::Convert(a, _)
        | Op::Bitcast(a) => Some(*a),
        Op::Compose(parts) => parts.first().copied(),
        _ => None,
    };
    if wide::wide(t) || first.map(ty).transpose()?.is_some_and(wide::wide) {
        return Ok(Kernel::Wide {
            op: Box::new(op.clone()),
            t,
            types: op.operands().into_iter().map(|v| (v, ty(v))).collect(),
        });
    }
    if !runs(t) {
        return Err(format!("{t} values are not run yet"));
    }
    let n = t.lanes;
    Ok(Kernel::Math(match op {
        // a derivative reads the quad, which is the run's
        Op::Unary(Unary::Derivative(..), _) => return Ok(Kernel::NotPure),
        Op::Unary(u, a) => {
            let at = ty(*a)?;
            Math::Unary {
                u: *u,
                m: at.lanes,
                logical: at.scalar == Scalar::Bool,
            }
        }
        Op::Binary(b, l, _) => {
            let at = ty(*l)?;
            let order = if at.scalar.is_float() {
                Order::Float
            } else if at.scalar.is_signed() {
                Order::Signed
            } else {
                Order::Unsigned
            };
            Math::Binary {
                b: *b,
                m: at.lanes,
                order,
            }
        }
        Op::Ternary(op, a, ..) => {
            let at = ty(*a)?;
            Math::Ternary { op: *op, m: at.lanes }
        }
        Op::BitInsert { .. } => Math::BitInsert { n },
        Op::Select { cond, .. } => Math::Select {
            n,
            per_lane: ty(*cond)?.lanes > 1,
        },
        Op::Compose(parts) => {
            let mut lanes = 0usize;
            let mut typed = Vec::with_capacity(parts.len());
            for p in parts {
                // a part of more than four lanes gives four
                let m = ty(*p)?.lanes.min(4);
                lanes = lanes.saturating_add(usize::from(m));
                typed.push(m);
            }
            if lanes > 4 {
                return Err("a compose of more than four lanes".into());
            }
            Math::Compose(typed.into())
        }
        Op::Swizzle(_, lanes) => {
            if lanes.iter().take(usize::from(t.lanes)).any(|&l| l >= 4) {
                return Err("a swizzle of lane past four".into());
            }
            Math::Swizzle { n, lanes: *lanes }
        }
        Op::Splat(_) => Math::Splat { n },
        Op::Dot(a, _) => {
            let at = ty(*a)?;
            Math::Dot { m: at.lanes }
        }
        Op::Convert(a, rule) => {
            let from = ty(*a)?;
            let conversion = match (from.scalar, t.scalar) {
                (Scalar::F32, Scalar::I32) if *rule == FloatToInt::Saturate => Conversion::FloatToSigned,
                (Scalar::F32, Scalar::U32) if *rule == FloatToInt::Saturate => Conversion::FloatToUnsigned,
                (Scalar::I32, Scalar::F32) => Conversion::SignedToFloat,
                (Scalar::U32, Scalar::F32) => Conversion::UnsignedToFloat,
                (Scalar::I32, Scalar::U32) | (Scalar::U32, Scalar::I32) => Conversion::Words,
                (s, d) => {
                    return Err(format!(
                        "a conversion of {} to {} ({rule:?}) is not run yet",
                        s.name(),
                        d.name()
                    ));
                }
            };
            Math::Convert { n, conversion }
        }
        Op::Bitcast(_) => Math::Bitcast,
        _ => return Ok(Kernel::NotPure),
    }))
}

#[inline(always)]
fn unary_arms<D: Drive1>(u: Unary, m: u8, logical: bool, d: D) -> D::Out {
    match u {
        Unary::FNeg => d.each(|w| w ^ 0x8000_0000),
        Unary::FAbs => d.each(|w| w & 0x7fff_ffff),
        Unary::FFract => d.float(|f| f - libm::floorf(f)),
        Unary::FRound(Rounding::Floor) => d.float(libm::floorf),
        Unary::FRound(Rounding::Ceil) => d.float(libm::ceilf),
        Unary::FRound(Rounding::Trunc) => d.float(libm::truncf),
        Unary::FRound(Rounding::NearestEven) => d.float(libm::roundevenf),
        // Rust's `round` is to the nearest, halves away from zero
        Unary::FRound(Rounding::NearestAway) => d.float(libm::roundf),
        Unary::FSqrt => d.float(|f| narrow(libm::sqrt(f64::from(f)))),
        Unary::FInverseSqrt => d.float(|f| narrow(1.0 / libm::sqrt(f64::from(f)))),
        Unary::FExp2 => d.float(|f| narrow(libm::exp2(f64::from(f)))),
        Unary::FLog2 => d.float(|f| narrow(libm::log2(f64::from(f)))),
        Unary::FSin => d.float(|f| narrow(libm::sin(f64::from(f)))),
        Unary::FCos => d.float(|f| narrow(libm::cos(f64::from(f)))),
        Unary::FTan => d.float(|f| narrow(libm::tan(f64::from(f)))),
        Unary::FAsin => d.float(|f| narrow(libm::asin(f64::from(f)))),
        Unary::FAcos => d.float(|f| narrow(libm::acos(f64::from(f)))),
        Unary::FAtan => d.float(|f| narrow(libm::atan(f64::from(f)))),
        Unary::FSinh => d.float(|f| narrow(libm::sinh(f64::from(f)))),
        Unary::FCosh => d.float(|f| narrow(libm::cosh(f64::from(f)))),
        Unary::FTanh => d.float(|f| narrow(libm::tanh(f64::from(f)))),
        // a zero gives itself, NaN 0
        Unary::FSign => d.float(|f| {
            if is_nan(f) {
                0.0
            } else if f == 0.0 {
                f
            } else {
                f.signum()
            }
        }),
        Unary::Saturate(SaturateNan::Zero | SaturateNan::Any) => {
            d.float(|f| if is_nan(f) || f <= 0.0 { 0.0 } else { f.min(1.0) })
        }
        // NaN gives 0 under either rule (`Any` allows it)
        Unary::ClampRange { lo, hi, .. } => {
            let (lo, hi) = (fb(lo), fb(hi));
            d.float(|f| if is_nan(f) { 0.0 } else { f.clamp(lo, hi) })
        }
        Unary::Quantize { frac_bits, rounding } => {
            let scale = f32::from_bits(127u32.saturating_add(u32::from(frac_bits)) << 23);
            // from 2^(23 - frac_bits) on every f32 is on the grid already
            let grid = f32::from_bits(150u32.saturating_sub(u32::from(frac_bits)) << 23);
            let round = match rounding {
                Rounding::Floor => libm::floorf,
                Rounding::Ceil => libm::ceilf,
                Rounding::Trunc => libm::truncf,
                Rounding::NearestEven => libm::roundevenf,
                Rounding::NearestAway => libm::roundf,
            };
            d.float(|f| {
                if f.abs() >= grid {
                    f
                } else {
                    round(f * scale) / scale
                }
            })
        }
        // not prepared: a derivative is not pure
        Unary::Derivative(..) => d.whole(|_| [0; 4]),
        Unary::Not if logical => d.each(|w| u32::from(w == 0)),
        Unary::Not => d.each(|w| !w),
        Unary::INeg => d.each(u32::wrapping_neg),
        Unary::Any => d.whole(|x| [u32::from(x.iter().take(usize::from(m)).any(|&w| w != 0)), 0, 0, 0]),
        Unary::All => d.whole(|x| [u32::from(x.iter().take(usize::from(m)).all(|&w| w != 0)), 0, 0, 0]),
        Unary::IsNan => d.each(|w| u32::from(is_nan(fb(w)))),
        Unary::IsInf => d.each(|w| u32::from(w & 0x7fff_ffff == 0x7f80_0000)),
        Unary::CountOnes => d.each(u32::count_ones),
        Unary::BitReverse => d.each(u32::reverse_bits),
        Unary::FirstBitHigh => d.each(|w| if w == 0 { u32::MAX } else { w.leading_zeros() }),
        Unary::FirstBitHighSigned => d.each(|w| {
            let w = if w.cast_signed() < 0 { !w } else { w };
            if w == 0 { u32::MAX } else { w.leading_zeros() }
        }),
        Unary::FirstBitLow => d.each(|w| if w == 0 { u32::MAX } else { w.trailing_zeros() }),
        Unary::F32ToF16 => d.each(f32_to_f16),
        Unary::F16ToF32 => d.each(f16_to_f32),
    }
}

#[inline(always)]
fn binary_arms<D: Drive2>(b: Binary, order: Order, d: D) -> D::Out {
    let order = |p: u32, q: u32| match order {
        Order::Float => fb(p).partial_cmp(&fb(q)),
        Order::Signed => Some(p.cast_signed().cmp(&q.cast_signed())),
        Order::Unsigned => Some(p.cmp(&q)),
    };
    match b {
        Binary::FAdd => d.float2(|p, q| p + q),
        Binary::FSub => d.float2(|p, q| p - q),
        Binary::FMul(MulZero::Ieee) => d.float2(|p, q| p * q),
        Binary::FMul(MulZero::ZeroWins) => d.float2(|p, q| if p == 0.0 || q == 0.0 { 0.0 } else { p * q }),
        Binary::FDiv => d.float2(|p, q| p / q),
        // C's fmod, exact
        Binary::FRem => d.float2(|p, q| p % q),
        Binary::FMin(nan) => d.float2(|p, q| min_max(p, q, nan, false)),
        Binary::FMax(nan) => d.float2(|p, q| min_max(p, q, nan, true)),
        Binary::FPow => d.float2(|p, q| narrow(libm::pow(f64::from(p), f64::from(q)))),
        Binary::FCmp(c) | Binary::SCmp(c) | Binary::UCmp(c) => {
            let ne = c == Cmp::Ne;
            let c = compare(c);
            d.each2(|p, q| u32::from(order(p, q).map_or(ne, &c)))
        }
        Binary::IAdd => d.each2(u32::wrapping_add),
        Binary::ISub => d.each2(u32::wrapping_sub),
        Binary::IMul => d.each2(u32::wrapping_mul),
        Binary::UMulHigh => d.each2(|p, q| {
            let wide = u64::from(p).wrapping_mul(u64::from(q));
            u32::try_from(wide >> 32).unwrap_or(0)
        }),
        Binary::SMulHigh => d.each2(|p, q| {
            let wide = i64::from(p.cast_signed()).wrapping_mul(i64::from(q.cast_signed()));
            u32::try_from(wide.cast_unsigned() >> 32).unwrap_or(0)
        }),
        Binary::UAddCarry => d.each2(|p, q| u32::from(p.checked_add(q).is_none())),
        Binary::USubBorrow => d.each2(|p, q| u32::from(p < q)),
        Binary::SDiv(z) => d.each2(|p, q| match (q, z) {
            (0, OnZero::AllOnes) => u32::MAX,
            (0, _) => p,
            // only the most negative value by -1 overflows: it wraps to itself
            _ => p
                .cast_signed()
                .checked_div(q.cast_signed())
                .map_or(p, i32::cast_unsigned),
        }),
        Binary::SRem(z) => d.each2(|p, q| match (q, z) {
            (0, OnZero::AllOnes) => u32::MAX,
            (0, _) => 0,
            _ => p
                .cast_signed()
                .checked_rem(q.cast_signed())
                .map_or(0, i32::cast_unsigned),
        }),
        Binary::UMin => d.each2(u32::min),
        Binary::UMax => d.each2(u32::max),
        Binary::Shl => d.each2(|p, q| p.wrapping_shl(q)),
        Binary::ShrU => d.each2(|p, q| p.wrapping_shr(q)),
        Binary::ShrS => d.each2(|p, q| p.cast_signed().wrapping_shr(q).cast_unsigned()),
        Binary::UDiv(z) => d.each2(|p, q| match (q, z) {
            (0, OnZero::AllOnes) => u32::MAX,
            (0, _) => p,
            _ => p.checked_div(q).unwrap_or(0),
        }),
        Binary::URem(z) => d.each2(|p, q| match (q, z) {
            (0, OnZero::AllOnes) => u32::MAX,
            (0, _) => 0,
            _ => p.checked_rem(q).unwrap_or(0),
        }),
        Binary::SMin => d.each2(|p, q| p.cast_signed().min(q.cast_signed()).cast_unsigned()),
        Binary::SMax => d.each2(|p, q| p.cast_signed().max(q.cast_signed()).cast_unsigned()),
        Binary::And => d.each2(|p, q| p & q),
        Binary::Or => d.each2(|p, q| p | q),
        Binary::Xor => d.each2(|p, q| p ^ q),
    }
}

/// How the lanes of a unary operation's operand become the lanes of its result: the operation's table
/// ([`unary_arms`]) names what it does to each lane (or to all of them), the driver where it does it.
trait Drive1 {
    type Out;
    /// Each lane of the operand as a word.
    fn each(self, op: impl Fn(u32) -> u32) -> Self::Out;
    /// Each lane of the operand as an `f32`.
    fn float(self, op: impl Fn(f32) -> f32) -> Self::Out;
    /// All the operand's lanes at once.
    fn whole(self, op: impl Fn(&Lanes) -> Lanes) -> Self::Out;
}

/// A binary operation's, as [`Drive1`].
trait Drive2 {
    type Out;
    fn each2(self, op: impl Fn(u32, u32) -> u32) -> Self::Out;
    fn float2(self, op: impl Fn(f32, f32) -> f32) -> Self::Out;
}

/// One operation on the lanes it is given: [`unary`], [`binary`].
struct Once<const N: usize>(u8, [Lanes; N]);

impl Drive1 for Once<1> {
    type Out = Lanes;
    fn each(self, op: impl Fn(u32) -> u32) -> Lanes {
        each(self.0, &self.1[0], op)
    }
    fn float(self, op: impl Fn(f32) -> f32) -> Lanes {
        float(self.0, &self.1[0], op)
    }
    fn whole(self, op: impl Fn(&Lanes) -> Lanes) -> Lanes {
        op(&self.1[0])
    }
}

impl Drive2 for Once<2> {
    type Out = Lanes;
    fn each2(self, op: impl Fn(u32, u32) -> u32) -> Lanes {
        each2(self.0, &self.1[0], &self.1[1], op)
    }
    fn float2(self, op: impl Fn(f32, f32) -> f32) -> Lanes {
        float2(self.0, &self.1[0], &self.1[1], op)
    }
}

fn unary(u: Unary, x: Lanes, m: u8, logical: bool) -> Lanes {
    unary_arms(u, m, logical, Once(m, [x]))
}

fn binary(b: Binary, x: Lanes, y: Lanes, m: u8, order: Order) -> Lanes {
    binary_arms(b, order, Once(m, [x, y]))
}

fn no_register(index: usize, len: usize) -> String {
    format!("register {index} is out of range of {len} registers")
}

/// A register file of `stride` lanes to a value, the lanes of one value together: the lanes an operation of a lane
/// loop ([`Prepared::run_lanes`]) runs, where it reads its operands and writes its result.
struct Lanewise<'r, A> {
    regs: &'r mut [Lanes],
    stride: usize,
    /// The register of the result's first lane.
    dst: usize,
    active: A,
}

impl<A: Fn(usize) -> bool> Lanewise<'_, A> {
    /// Each active lane's result from its operands (the registers `src` names, from their first lanes), the operands
    /// flushed where `flush` says and the result where `flush_out` does.
    #[inline(always)]
    fn drive<const N: usize>(
        self,
        src: [usize; N],
        flush: [bool; N],
        flush_out: bool,
        op: impl Fn([Lanes; N]) -> Lanes,
    ) -> Result<(), String> {
        for k in 0..self.stride {
            if !(self.active)(k) {
                continue;
            }
            let mut x = [[0; 4]; N];
            for ((x, at), flush) in x.iter_mut().zip(src).zip(flush) {
                let index = at.saturating_add(k);
                let lanes = self
                    .regs
                    .get(index)
                    .copied()
                    .ok_or_else(|| no_register(index, self.regs.len()))?;
                *x = if flush { lanes.map(flushed) } else { lanes };
            }
            let out = op(x);
            let index = self.dst.saturating_add(k);
            let len = self.regs.len();
            *self.regs.get_mut(index).ok_or_else(|| no_register(index, len))? =
                if flush_out { out.map(flushed) } else { out };
        }
        Ok(())
    }
}

/// [`Drive1`] over the lanes of a register file.
struct Lanes1<'r, A> {
    to: Lanewise<'r, A>,
    src: usize,
    m: u8,
    flush: bool,
    flush_out: bool,
}

impl<A: Fn(usize) -> bool> Drive1 for Lanes1<'_, A> {
    type Out = Result<(), String>;
    #[inline(always)]
    fn each(self, op: impl Fn(u32) -> u32) -> Self::Out {
        let m = self.m;
        self.to
            .drive([self.src], [self.flush], self.flush_out, |[x]| each(m, &x, &op))
    }
    #[inline(always)]
    fn float(self, op: impl Fn(f32) -> f32) -> Self::Out {
        let m = self.m;
        self.to
            .drive([self.src], [self.flush], self.flush_out, |[x]| float(m, &x, &op))
    }
    #[inline(always)]
    fn whole(self, op: impl Fn(&Lanes) -> Lanes) -> Self::Out {
        self.to
            .drive([self.src], [self.flush], self.flush_out, |[x]| op(&x))
    }
}

/// [`Drive2`] over the lanes of a register file.
struct Lanes2<'r, A> {
    to: Lanewise<'r, A>,
    src: [usize; 2],
    m: u8,
    flush: [bool; 2],
    flush_out: bool,
}

impl<A: Fn(usize) -> bool> Drive2 for Lanes2<'_, A> {
    type Out = Result<(), String>;
    #[inline(always)]
    fn each2(self, op: impl Fn(u32, u32) -> u32) -> Self::Out {
        let m = self.m;
        self.to.drive(self.src, self.flush, self.flush_out, |[x, y]| {
            each2(m, &x, &y, &op)
        })
    }
    #[inline(always)]
    fn float2(self, op: impl Fn(f32, f32) -> f32) -> Self::Out {
        let m = self.m;
        self.to.drive(self.src, self.flush, self.flush_out, |[x, y]| {
            float2(m, &x, &y, &op)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every NaN, of either sign and any payload, converts to the one quiet half NaN; an infinity keeps its sign.
    #[test]
    fn every_nan_is_one_half_nan() {
        for sign in [0, 0x8000_0000u32] {
            for man in 1..0x80_0000u32 {
                let w = sign | 0x7f80_0000 | man;
                assert_eq!(f32_to_f16(w), 0x7e00, "{w:#x}");
            }
            assert_eq!(f32_to_f16(sign | 0x7f80_0000), sign >> 16 | 0x7c00);
        }
    }
}
