//! Runs a program of iridia's shader IR ([`iridia_ir::Program`]) on the CPU, one invocation or a pixel quad, by the
//! IR's own definition of each operation: the reference the lowerings are checked against.
//!
//! The arithmetic is the reference `iridia-exec` uses: IEEE 754 `f32` for the basic operations, the transcendental
//! functions (`exp2`, `log2`, `sin`, `cos`, `sqrt`, `inverseSqrt`, `pow`) in `f64` rounded once, a dot product summed
//! left to right with each step rounded. Denormals are kept, or flushed to zero (sign kept) on the inputs and result
//! of each float operation but the sign operations and the moves, by the program's float environment or, where it
//! leaves them open, by the run's [`Bound::denormals`]. This subset runs 32-bit scalars and vectors, both
//! control phases (every construct of a `Regions` body, the blocks of a `Cfg` one), constant buffer reads, pipeline
//! state, samples (answered by the caller), kill and demotion; it refuses the rest by name.
//!
//! [`run`] runs one invocation, which has no screen derivatives: it refuses a derivative and leaves an implicit-LOD
//! sample's level to the sampler. [`run_quad`] runs a pixel program's 2 x 2 quad, whose derivatives are differences
//! between its lanes.
#![deny(missing_docs)]
#![deny(clippy::disallowed_methods, clippy::disallowed_types)]
#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation
    )
)]

use iridia_ir::{
    Axis, Binary, Block, Body, Cmp, Denormals, End, Fineness, FloatToInt, Function, LabelId, Level,
    MinMaxNan, MulZero, Node, OnZero, Op, OutOfBounds, Program, Region, Rounding, SaturateNan, Scalar,
    Terminator, Type, Unary, ValueId,
    interface::{ResourceId, ResourceKind},
};

/// A value's lanes as 32-bit words (a bool is 0 or 1).
pub type Lanes = [u32; 4];

/// How a sample picks its level of detail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lod {
    /// From the derivatives, which a single invocation does not have: the sampler decides.
    Auto,
    /// From the derivatives, which a single invocation does not have, plus a bias.
    Bias(f32),
    /// An explicit level.
    Exact(f32),
    /// From derivatives of the coordinate along x and y (the coordinate's lanes of each, zero past them): the
    /// program's own, or for an implicit level in a quad the quad's.
    Grad([f32; 3], [f32; 3]),
    /// An implicit level in a quad: from the quad's derivatives of the coordinate, as `Grad`, plus a bias.
    GradBias([f32; 3], [f32; 3], f32),
}

/// A sample request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// The texture.
    pub texture: ResourceId,
    /// The sampler.
    pub sampler: ResourceId,
    /// The coordinate's lanes (two for 2D, three for cube and volume).
    pub coord: [f32; 3],
    /// How many lanes `coord` has.
    pub lanes: u8,
    /// The level of detail.
    pub lod: Lod,
}

/// What a run gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    /// Each output slot's lanes in varying order, each as the program stored it last (zero for a lane it never
    /// stored); `None` for a slot it did not store at all.
    pub outputs: Vec<Option<Lanes>>,
    /// Whether the invocation was killed or demoted (its outputs then are those stored before).
    pub killed: bool,
}

/// The resources and inputs of a run.
#[allow(missing_debug_implementations)] // the sampler is a closure
pub struct Bound<'a> {
    /// Each interface input slot's lanes.
    pub inputs: &'a [Lanes],
    /// Each constant buffer's rows.
    pub buffers: &'a [(ResourceId, &'a [Lanes])],
    /// Each pipeline state resource's value (a bool is 0 or 1).
    pub state: &'a [(ResourceId, Lanes)],
    /// Answers each sample with the texel's four lanes.
    pub sampler: &'a mut dyn FnMut(Sample) -> Result<[f32; 4], String>,
    /// The `f32` denormal rule of a program whose float environment leaves it open ([`Denormals::Any`]): kept for
    /// `Any` and `Preserve`, flushed for `Flush`. A program that states a rule runs by it.
    pub denormals: Denormals,
}

/// The most loop iterations a run takes before it is refused.
pub const STEP_LIMIT: u64 = 1 << 24;

/// The most derivatives and implicit-LOD samples a quad's lanes meet before the quad run is refused.
pub const MEETING_LIMIT: usize = 1 << 12;

/// Why a lane's run stopped before its end.
enum Stop {
    /// It failed.
    Fail(String),
    /// A lane of a quad reached an operation whose neighbours' operands it does not know yet.
    Meet,
}

impl From<String> for Stop {
    fn from(e: String) -> Self {
        Stop::Fail(e)
    }
}

impl From<&str> for Stop {
    fn from(e: &str) -> Self {
        Stop::Fail(e.into())
    }
}

type Step<T> = Result<T, Stop>;

/// Where a lane met an operation that reads its quad: the operation, how many loop iterations or block executions the
/// lane had run, and how many times it had run the operation before. Lanes that meet at different places diverge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Place {
    value: ValueId,
    steps: u64,
    before: u32,
}

/// What a lane of a quad knows of the meetings before: each lane's operand at each meeting, the lanes in quad order.
struct Quad<'q> {
    /// This lane: 0 and 1 the top row, left to right, 2 and 3 the bottom.
    lane: usize,
    known: &'q [[Lanes; 4]],
    /// How many meetings this run of the lane has passed.
    met: usize,
    /// How many times this run has met each value's operation.
    times: Vec<u32>,
    /// The meeting the run stopped at, and its operand there.
    waiting: Option<(Place, Lanes)>,
}

/// How a region ended.
enum Exit {
    Yield(Vec<ValueId>),
    Br(LabelId, Vec<ValueId>),
    Continue(LabelId, Vec<ValueId>),
    Return,
    Kill,
}

struct Machine<'a, 'b> {
    program: &'a Program,
    f: &'a Function,
    values: Vec<Option<Lanes>>,
    vars: Vec<Option<Lanes>>,
    bound: &'a mut Bound<'b>,
    outputs: Vec<Option<Lanes>>,
    steps: u64,
    /// Whether `f32` denormals are flushed.
    flush: bool,
    /// The lane's quad, where the run is one.
    quad: Option<Quad<'a>>,
    /// Whether the invocation was demoted: a helper from there, whose output stores have no effect.
    demoted: bool,
}

/// The two lanes of a quad whose difference is lane `lane`'s derivative: (from, to). A coarse derivative (and
/// `Any`'s, as `iridia-exec` takes it) is the top row's or the left column's for every lane, a fine one the lane's own
/// row's or column's.
fn pair(axis: Axis, fineness: Fineness, lane: usize) -> (usize, usize) {
    let fine = fineness == Fineness::Fine;
    match axis {
        Axis::X if fine && lane >= 2 => (2, 3),
        Axis::X => (0, 1),
        Axis::Y if fine && lane % 2 == 1 => (1, 3),
        Axis::Y => (0, 2),
    }
}

/// An `f32` lane with a denormal made a zero of its sign.
fn flushed(w: u32) -> u32 {
    if w & 0x7f80_0000 == 0 { w & 0x8000_0000 } else { w }
}

/// The lanes a mask names, in ascending order.
fn masked(mask: u8) -> impl Iterator<Item = usize> {
    (0..4).filter(move |n| mask >> n & 1 != 0)
}

fn narrow(v: f64) -> f32 {
    #[allow(clippy::cast_possible_truncation)] // the rounding is the operation
    let r = v as f32;
    r
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

impl Machine<'_, '_> {
    fn get(&self, v: ValueId) -> Result<Lanes, String> {
        self.values
            .get(v.0 as usize)
            .copied()
            .flatten()
            .ok_or_else(|| format!("%{} has no value here", v.0))
    }
    fn set(&mut self, v: ValueId, x: Lanes) -> Result<(), String> {
        *self
            .values
            .get_mut(v.0 as usize)
            .ok_or_else(|| format!("%{} is out of range", v.0))? = Some(x);
        Ok(())
    }
    fn ty(&self, v: ValueId) -> Result<Type, String> {
        self.f.ty(v).ok_or_else(|| format!("%{} has no type", v.0))
    }
    fn buffer(&self, id: ResourceId) -> Result<(&[Lanes], u32), String> {
        let rows = match self
            .program
            .interface
            .resources
            .iter()
            .find(|r| r.id == id)
            .map(|r| &r.kind)
        {
            Some(ResourceKind::ConstantBuffer { layout }) => layout.size.div_ceil(16),
            _ => return Err(format!("resource {} is not a constant buffer", id.0)),
        };
        let data = self
            .bound
            .buffers
            .iter()
            .find(|(r, _)| *r == id)
            .map(|(_, d)| *d)
            .ok_or_else(|| format!("constant buffer {} is not bound", id.0))?;
        Ok((data, rows))
    }
    /// Meet the quad at value `v`'s operation with this lane's `operand`: every lane's operand in quad order where the
    /// lanes have all met here before, a stop where they have not, `None` where the run is not a quad's.
    fn meet(&mut self, v: ValueId, operand: Lanes) -> Step<Option<[Lanes; 4]>> {
        let steps = self.steps;
        let Some(q) = self.quad.as_mut() else {
            return Ok(None);
        };
        let times = q.times.get_mut(v.0 as usize).ok_or("a value out of range")?;
        let before = *times;
        *times = times.saturating_add(1);
        if let Some(known) = q.known.get(q.met) {
            q.met = q.met.saturating_add(1);
            return Ok(Some(*known));
        }
        q.waiting = Some((
            Place {
                value: v,
                steps,
                before,
            },
            operand,
        ));
        Err(Stop::Meet)
    }
    fn op(&mut self, v: ValueId) -> Step<Lanes> {
        let def = self
            .f
            .value(v)
            .ok_or_else(|| format!("%{} is not a value", v.0))?;
        let t = def.ty;
        if t.scalar.bits() > 32 || (t.scalar.bits() < 32 && t.scalar != Scalar::Bool) {
            return Err(format!("{t} values are not run yet").into());
        }
        // the float operations whose inputs and result a flushing run flushes: not the sign operations and moves
        let arithmetic = match &def.op {
            Op::Unary(u, _) => !matches!(u, Unary::FNeg | Unary::FAbs),
            Op::Binary(..) | Op::Dot(..) | Op::Convert(..) => true,
            _ => false,
        };
        let out = self.compute(v, &def.op, t)?;
        Ok(if self.flush && arithmetic && t.scalar == Scalar::F32 {
            out.map(flushed)
        } else {
            out
        })
    }
    /// An operand of an arithmetic operation: its lanes, flushed where the run flushes and it is a float.
    fn arg(&self, v: ValueId) -> Result<Lanes, String> {
        let x = self.get(v)?;
        Ok(if self.flush && self.ty(v)?.scalar == Scalar::F32 {
            x.map(flushed)
        } else {
            x
        })
    }
    fn compute(&mut self, v: ValueId, op: &Op, t: Type) -> Step<Lanes> {
        let n = t.lanes;
        Ok(match op {
            Op::Literal(bits) => bits.map(|b| u32::try_from(b & 0xffff_ffff).unwrap_or(0)),
            Op::Undef => [0; 4],
            Op::Param => return Err(format!("parameter %{} has no value here", v.0).into()),
            Op::Unary(u, a) => {
                let x = self.arg(*a)?;
                let at = self.ty(*a)?;
                let m = at.lanes;
                match u {
                    Unary::FNeg => each(m, &x, |w| w ^ 0x8000_0000),
                    Unary::FAbs => each(m, &x, |w| w & 0x7fff_ffff),
                    Unary::FFract => float(m, &x, |f| f - f.floor()),
                    Unary::FRound(Rounding::Floor) => float(m, &x, f32::floor),
                    Unary::FRound(Rounding::Ceil) => float(m, &x, f32::ceil),
                    Unary::FRound(Rounding::Trunc) => float(m, &x, f32::trunc),
                    Unary::FRound(Rounding::NearestEven) => float(m, &x, f32::round_ties_even),
                    // Rust's `round` is to the nearest, halves away from zero
                    Unary::FRound(Rounding::NearestAway) => float(m, &x, f32::round),
                    Unary::FSqrt => float(m, &x, |f| narrow(f64::from(f).sqrt())),
                    Unary::FInverseSqrt => float(m, &x, |f| narrow(1.0 / f64::from(f).sqrt())),
                    Unary::FExp2 => float(m, &x, |f| narrow(f64::from(f).exp2())),
                    Unary::FLog2 => float(m, &x, |f| narrow(f64::from(f).log2())),
                    Unary::FSin => float(m, &x, |f| narrow(f64::from(f).sin())),
                    Unary::FCos => float(m, &x, |f| narrow(f64::from(f).cos())),
                    Unary::Saturate(SaturateNan::Zero | SaturateNan::Any) => {
                        float(m, &x, |f| if is_nan(f) { 0.0 } else { f.clamp(0.0, 1.0) })
                    }
                    // NaN gives 0 under either rule (`Any` allows it)
                    Unary::ClampRange { lo, hi, .. } => {
                        let (lo, hi) = (fb(*lo), fb(*hi));
                        float(m, &x, |f| if is_nan(f) { 0.0 } else { f.clamp(lo, hi) })
                    }
                    Unary::Quantize { frac_bits, rounding } => {
                        let scale = f32::from_bits(127u32.saturating_add(u32::from(*frac_bits)) << 23);
                        // from 2^(23 - frac_bits) on every f32 is on the grid already
                        let grid = f32::from_bits(150u32.saturating_sub(u32::from(*frac_bits)) << 23);
                        let round = match rounding {
                            Rounding::Floor => f32::floor,
                            Rounding::Ceil => f32::ceil,
                            Rounding::Trunc => f32::trunc,
                            Rounding::NearestEven => f32::round_ties_even,
                            Rounding::NearestAway => f32::round,
                        };
                        float(m, &x, |f| {
                            if f.abs() >= grid {
                                f
                            } else {
                                round(f * scale) / scale
                            }
                        })
                    }
                    Unary::Derivative(axis, fineness) => {
                        let quad = self.meet(v, x)?.ok_or(
                            "a derivative needs the invocation's quad, which a single run does not have",
                        )?;
                        let (from, to) = pair(*axis, *fineness, self.quad.as_ref().map_or(0, |q| q.lane));
                        let lane = |l: usize| quad.get(l).copied().unwrap_or_default();
                        float2(m, &lane(to), &lane(from), |p, q| p - q)
                    }
                    Unary::Not if at.scalar == Scalar::Bool => each(m, &x, |w| u32::from(w == 0)),
                    Unary::Not => each(m, &x, |w| !w),
                    Unary::INeg => each(m, &x, u32::wrapping_neg),
                    Unary::Any => [u32::from(x.iter().take(usize::from(m)).any(|&w| w != 0)), 0, 0, 0],
                    Unary::All => [u32::from(x.iter().take(usize::from(m)).all(|&w| w != 0)), 0, 0, 0],
                    Unary::IsNan => each(m, &x, |w| u32::from(is_nan(fb(w)))),
                }
            }
            Op::Binary(b, l, r) => {
                let (x, y) = (self.arg(*l)?, self.arg(*r)?);
                let at = self.ty(*l)?;
                let m = at.lanes;
                let signed = at.scalar.is_signed();
                let order = |p: u32, q: u32| {
                    if at.scalar.is_float() {
                        fb(p).partial_cmp(&fb(q))
                    } else if signed {
                        Some(p.cast_signed().cmp(&q.cast_signed()))
                    } else {
                        Some(p.cmp(&q))
                    }
                };
                match b {
                    Binary::FAdd => float2(m, &x, &y, |p, q| p + q),
                    Binary::FSub => float2(m, &x, &y, |p, q| p - q),
                    Binary::FMul(MulZero::Ieee) => float2(m, &x, &y, |p, q| p * q),
                    Binary::FMul(MulZero::ZeroWins) => {
                        float2(m, &x, &y, |p, q| if p == 0.0 || q == 0.0 { 0.0 } else { p * q })
                    }
                    Binary::FDiv => float2(m, &x, &y, |p, q| p / q),
                    Binary::FMin(nan) => float2(m, &x, &y, |p, q| min_max(p, q, *nan, false)),
                    Binary::FMax(nan) => float2(m, &x, &y, |p, q| min_max(p, q, *nan, true)),
                    Binary::FPow => float2(m, &x, &y, |p, q| narrow(f64::from(p).powf(f64::from(q)))),
                    Binary::FCmp(c) | Binary::SCmp(c) | Binary::UCmp(c) => {
                        let c = compare(*c);
                        let ne = matches!(
                            b,
                            Binary::FCmp(Cmp::Ne) | Binary::SCmp(Cmp::Ne) | Binary::UCmp(Cmp::Ne)
                        );
                        each2(m, &x, &y, |p, q| u32::from(order(p, q).map_or(ne, &c)))
                    }
                    Binary::IAdd => each2(m, &x, &y, u32::wrapping_add),
                    Binary::ISub => each2(m, &x, &y, u32::wrapping_sub),
                    Binary::IMul => each2(m, &x, &y, u32::wrapping_mul),
                    Binary::UDiv(z) => each2(m, &x, &y, |p, q| match (q, z) {
                        (0, OnZero::AllOnes) => u32::MAX,
                        (0, _) => p,
                        _ => p.checked_div(q).unwrap_or(0),
                    }),
                    Binary::URem(z) => each2(m, &x, &y, |p, q| match (q, z) {
                        (0, OnZero::AllOnes) => u32::MAX,
                        (0, _) => p,
                        _ => p.checked_rem(q).unwrap_or(0),
                    }),
                    Binary::SMin => each2(m, &x, &y, |p, q| {
                        p.cast_signed().min(q.cast_signed()).cast_unsigned()
                    }),
                    Binary::SMax => each2(m, &x, &y, |p, q| {
                        p.cast_signed().max(q.cast_signed()).cast_unsigned()
                    }),
                    Binary::And => each2(m, &x, &y, |p, q| p & q),
                    Binary::Or => each2(m, &x, &y, |p, q| p | q),
                    Binary::Xor => each2(m, &x, &y, |p, q| p ^ q),
                }
            }
            Op::Select { cond, accept, reject } => {
                let (c, a, r) = (self.get(*cond)?, self.get(*accept)?, self.get(*reject)?);
                let wide = self.ty(*cond)?.lanes > 1;
                let mut out = [0; 4];
                for (i, o) in out.iter_mut().enumerate().take(usize::from(n)) {
                    let pick = if wide {
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
            Op::Compose(parts) => {
                let mut out = [0; 4];
                let mut at = 0usize;
                for p in parts {
                    let x = self.get(*p)?;
                    for w in x.iter().take(usize::from(self.ty(*p)?.lanes)) {
                        *out.get_mut(at).ok_or("a compose of more than four lanes")? = *w;
                        at = at.saturating_add(1);
                    }
                }
                out
            }
            Op::Swizzle(a, lanes) => {
                let x = self.get(*a)?;
                let mut out = [0; 4];
                for (o, &l) in out.iter_mut().zip(lanes).take(usize::from(n)) {
                    *o = *x.get(usize::from(l)).ok_or("a swizzle of lane past four")?;
                }
                out
            }
            Op::Splat(a) => {
                let x = self.get(*a)?;
                let mut out = [0; 4];
                for o in out.iter_mut().take(usize::from(n)) {
                    *o = x[0];
                }
                out
            }
            Op::Dot(a, b) => {
                let (x, y) = (self.arg(*a)?, self.arg(*b)?);
                let m = self.ty(*a)?.lanes;
                // each step is rounded, and flushed where the run flushes
                let step = |f: f32| if self.flush { fb(flushed(f.to_bits())) } else { f };
                let mut acc = 0.0f32;
                for (i, (p, q)) in x.iter().zip(&y).take(usize::from(m)).enumerate() {
                    let prod = step(fb(*p) * fb(*q));
                    acc = if i == 0 { prod } else { step(acc + prod) };
                }
                [acc.to_bits(), 0, 0, 0]
            }
            Op::Convert(a, rule) => {
                let x = self.arg(*a)?;
                let from = self.ty(*a)?;
                match (from.scalar, t.scalar) {
                    // Rust's `as` truncates toward zero, saturates and gives 0 for NaN: D3D's ftoi and ftou
                    (Scalar::F32, Scalar::I32) if *rule == FloatToInt::Saturate => {
                        each(n, &x, |w| {
                            #[allow(clippy::cast_possible_truncation)] // the saturation is the operation
                            let r = fb(w) as i32;
                            r.cast_unsigned()
                        })
                    }
                    (Scalar::F32, Scalar::U32) if *rule == FloatToInt::Saturate => each(n, &x, |w| {
                        #[allow(clippy::cast_possible_truncation)]
                        let r = fb(w) as u32;
                        r
                    }),
                    (Scalar::I32, Scalar::F32) => each(n, &x, |w| {
                        #[allow(clippy::cast_precision_loss)] // to nearest even, the conversion's rule
                        let r = w.cast_signed() as f32;
                        r.to_bits()
                    }),
                    (Scalar::U32, Scalar::F32) => each(n, &x, |w| {
                        #[allow(clippy::cast_precision_loss)]
                        let r = w as f32;
                        r.to_bits()
                    }),
                    (Scalar::I32, Scalar::U32) | (Scalar::U32, Scalar::I32) => x,
                    (s, d) => {
                        return Err(format!(
                            "a conversion of {} to {} ({rule:?}) is not run yet",
                            s.name(),
                            d.name()
                        )
                        .into());
                    }
                }
            }
            Op::Bitcast(a) => self.get(*a)?,
            Op::LoadInput(slot, mask) => {
                let all = self
                    .bound
                    .inputs
                    .get(*slot as usize)
                    .ok_or_else(|| format!("input {slot} is not given"))?;
                let mut x = [0; 4];
                for (to, from) in x.iter_mut().zip(masked(*mask)) {
                    *to = all.get(from).copied().unwrap_or(0);
                }
                x
            }
            Op::LoadVar(var) => self.vars.get(var.0 as usize).copied().flatten().unwrap_or([0; 4]),
            Op::LoadState(r) => self
                .bound
                .state
                .iter()
                .find(|(id, _)| id == r)
                .map(|(_, x)| *x)
                .ok_or_else(|| format!("state {} is not given", r.0))?,
            Op::LoadBuffer {
                resource,
                index,
                stride,
                offset,
                oob,
            } => {
                let (data, rows) = self.buffer(*resource)?;
                let base = match index {
                    None => Some(i64::from(*offset)),
                    Some(i) => {
                        let it = self.ty(*i)?;
                        let w = self.get(*i)?[0];
                        let i = if it.scalar.is_signed() {
                            i64::from(w.cast_signed())
                        } else {
                            i64::from(w)
                        };
                        i.checked_mul(i64::from(*stride))
                            .and_then(|b| b.checked_add(i64::from(*offset)))
                    }
                };
                let bytes = u32::from(n).saturating_mul(4);
                let size = i64::from(rows).saturating_mul(16);
                match base {
                    Some(b) if b >= 0 && b.saturating_add(i64::from(bytes)) <= size && b % 4 == 0 => {
                        let mut out = [0; 4];
                        for (k, o) in out.iter_mut().enumerate().take(usize::from(n)) {
                            let word = (b / 4).saturating_add(i64::try_from(k).unwrap_or(0));
                            let (row, lane) = (word / 4, word % 4);
                            let row = usize::try_from(row).map_err(|_| "a negative row")?;
                            let lane = usize::try_from(lane).map_err(|_| "a negative lane")?;
                            *o = data.get(row).and_then(|r| r.get(lane)).copied().unwrap_or(0);
                        }
                        out
                    }
                    _ if *oob == OutOfBounds::Zero => [0; 4],
                    _ => return Err("an undefined read outside a buffer".into()),
                }
            }
            Op::Sample {
                texture,
                sampler,
                coord,
                level,
            } => {
                let c = self.get(*coord)?;
                let lanes = self.ty(*coord)?.lanes;
                // a quad's implicit level of detail comes from the coordinate's differences: the right lane's and the
                // lower lane's less the top left's, as `iridia-exec` takes them
                let footprint = match level {
                    Level::Auto | Level::Bias(_) => self.meet(v, c)?.map(|quad| {
                        let d = |l: usize| {
                            let mut r = [0.0; 3];
                            for (k, o) in r.iter_mut().enumerate().take(usize::from(lanes)) {
                                let lane =
                                    |l: usize| quad.get(l).and_then(|q| q.get(k)).copied().unwrap_or(0);
                                *o = fb(lane(l)) - fb(lane(0));
                            }
                            r
                        };
                        (d(1), d(2))
                    }),
                    _ => None,
                };
                let lod = match (level, footprint) {
                    (Level::Auto, Some((dx, dy))) => Lod::Grad(dx, dy),
                    (Level::Bias(b), Some((dx, dy))) => Lod::GradBias(dx, dy, fb(self.get(*b)?[0])),
                    (Level::Auto, None) => Lod::Auto,
                    (Level::Zero, _) => Lod::Exact(0.0),
                    (Level::Bias(b), None) => Lod::Bias(fb(self.get(*b)?[0])),
                    (Level::Exact(l), _) => Lod::Exact(fb(self.get(*l)?[0])),
                    (Level::Grad(x, y), _) => {
                        let (x, y) = (self.get(*x)?, self.get(*y)?);
                        Lod::Grad([fb(x[0]), fb(x[1]), fb(x[2])], [fb(y[0]), fb(y[1]), fb(y[2])])
                    }
                };
                let request = Sample {
                    texture: *texture,
                    sampler: *sampler,
                    coord: [fb(c[0]), fb(c[1]), fb(c[2])],
                    lanes,
                    lod,
                };
                (self.bound.sampler)(request)?.map(f32::to_bits)
            }
        })
    }
    /// A node that is not a construct: a value, a variable store or an output store.
    fn plain(&mut self, node: &Node) -> Step<()> {
        match node {
            Node::Value(v) => {
                let x = self.op(*v)?;
                self.set(*v, x)?;
            }
            Node::StoreVar(var, x) => {
                let x = self.get(*x)?;
                *self
                    .vars
                    .get_mut(var.0 as usize)
                    .ok_or("a store to no variable")? = Some(x);
            }
            Node::Demote => self.demoted = true,
            Node::StoreOutput(..) if self.demoted => {}
            Node::StoreOutput(slot, mask, x) => {
                let x = self.get(*x)?;
                let out = self
                    .outputs
                    .get_mut(*slot as usize)
                    .ok_or("a store to no output")?;
                let lanes = out.get_or_insert([0; 4]);
                for (from, to) in x.iter().zip(masked(*mask)) {
                    if let Some(l) = lanes.get_mut(to) {
                        *l = *from;
                    }
                }
            }
            Node::If { .. } | Node::Block { .. } | Node::Loop { .. } => {
                return Err("a construct in a graph block".into());
            }
        }
        Ok(())
    }
    /// A graph body from its entry block until it returns or kills: whether it killed.
    fn graph(&mut self, blocks: &[Block]) -> Step<bool> {
        let mut at = 0usize;
        loop {
            self.steps = self.steps.saturating_add(1);
            if self.steps > STEP_LIMIT {
                return Err(format!("more than {STEP_LIMIT} block executions").into());
            }
            let block = blocks
                .get(at)
                .ok_or_else(|| format!("block {at} does not exist"))?;
            for n in &block.nodes {
                self.plain(n)?;
            }
            let target = match &block.term {
                Terminator::Jump(t) => t,
                Terminator::Branch { cond, accept, reject } => {
                    if self.get(*cond)?[0] != 0 {
                        accept
                    } else {
                        reject
                    }
                }
                Terminator::Return => return Ok(false),
                Terminator::Kill => return Ok(true),
                Terminator::Unreachable => {
                    return Err(format!("reached block {at}'s unreachable end").into());
                }
            };
            let next = blocks
                .get(target.block as usize)
                .ok_or_else(|| format!("block {} does not exist", target.block))?;
            // the parameters take their arguments together: every argument is read before any is set
            let args: Vec<Lanes> = target
                .args
                .iter()
                .map(|a| self.get(*a))
                .collect::<Result<_, _>>()?;
            for (p, a) in next.params.iter().zip(args) {
                self.set(*p, a)?;
            }
            at = target.block as usize;
        }
    }
    fn region(&mut self, r: &Region) -> Step<Exit> {
        for node in &r.nodes {
            match node {
                Node::Value(_) | Node::StoreVar(..) | Node::StoreOutput(..) | Node::Demote => {
                    self.plain(node)?
                }
                Node::If {
                    cond,
                    accept,
                    reject,
                    results,
                } => {
                    let arm = if self.get(*cond)?[0] != 0 { accept } else { reject };
                    match self.region(arm)? {
                        Exit::Yield(vs) => self.give(results, &vs)?,
                        other => return Ok(other),
                    }
                }
                Node::Block { label, body, results } => match self.region(body)? {
                    Exit::Yield(vs) => self.give(results, &vs)?,
                    Exit::Br(l, vs) if l == *label => self.give(results, &vs)?,
                    other => return Ok(other),
                },
                Node::Loop {
                    label,
                    params,
                    inits,
                    body,
                    results,
                } => {
                    let mut args: Vec<Lanes> =
                        inits.iter().map(|i| self.get(*i)).collect::<Result<_, _>>()?;
                    loop {
                        self.steps = self.steps.saturating_add(1);
                        if self.steps > STEP_LIMIT {
                            return Err(format!("more than {STEP_LIMIT} loop iterations").into());
                        }
                        for (p, a) in params.iter().zip(&args) {
                            self.set(*p, *a)?;
                        }
                        match self.region(body)? {
                            Exit::Continue(l, vs) if l == *label => {
                                args = vs.iter().map(|v| self.get(*v)).collect::<Result<_, _>>()?;
                            }
                            Exit::Yield(vs) => {
                                self.give(results, &vs)?;
                                break;
                            }
                            Exit::Br(l, vs) if l == *label => {
                                self.give(results, &vs)?;
                                break;
                            }
                            other => return Ok(other),
                        }
                    }
                }
            }
        }
        Ok(match &r.end {
            Some(End::Yield(vs)) => Exit::Yield(vs.clone()),
            Some(End::Br(l, vs)) => Exit::Br(*l, vs.clone()),
            Some(End::Continue(l, vs)) => Exit::Continue(*l, vs.clone()),
            Some(End::Return) => Exit::Return,
            Some(End::Kill) => Exit::Kill,
            Some(End::Unreachable) => return Err("reached an unreachable end".into()),
            None => return Err("a region with no end".into()),
        })
    }
    fn give(&mut self, results: &[ValueId], vs: &[ValueId]) -> Result<(), String> {
        for (r, v) in results.iter().zip(vs) {
            let x = self.get(*v)?;
            self.set(*r, x)?;
        }
        Ok(())
    }
}

/// How a lane's run ended: its outputs, or the meeting it waits at.
enum Ended {
    Ran(Run),
    Waits(Place, Lanes),
}

/// Run a program's entry function for one lane, the quad's meetings known so far where it is a quad's.
fn lane(
    program: &Program,
    bound: &mut Bound<'_>,
    quad: Option<(usize, &[[Lanes; 4]])>,
) -> Result<Ended, String> {
    let f = &program.entry;
    let flush = match program.float.f32 {
        Denormals::Flush => true,
        Denormals::Preserve => false,
        Denormals::Any => bound.denormals == Denormals::Flush,
    };
    let mut m = Machine {
        program,
        f,
        values: vec![None; f.values.len()],
        vars: vec![None; f.vars.len()],
        bound,
        outputs: vec![None; program.interface.outputs.len()],
        steps: 0,
        flush,
        quad: quad.map(|(lane, known)| Quad {
            lane,
            known,
            met: 0,
            times: vec![0; f.values.len()],
            waiting: None,
        }),
        demoted: false,
    };
    let ended = match &f.body {
        Body::Regions(body) => match m.region(body) {
            Ok(Exit::Kill) => Ok(true),
            Ok(Exit::Return | Exit::Yield(_)) => Ok(false),
            Ok(Exit::Br(..) | Exit::Continue(..)) => return Err("an exit past the function's body".into()),
            Err(e) => Err(e),
        },
        Body::Cfg(blocks) => m.graph(blocks),
    };
    match ended {
        Ok(killed) => Ok(Ended::Ran(Run {
            outputs: m.outputs,
            killed: killed || m.demoted,
        })),
        Err(Stop::Fail(e)) => Err(e),
        Err(Stop::Meet) => match m.quad.and_then(|q| q.waiting) {
            Some((place, operand)) => Ok(Ended::Waits(place, operand)),
            None => Err("a run stopped at a meeting it did not record".into()),
        },
    }
}

/// Run a program's entry function once, in either control phase.
pub fn run(program: &Program, bound: &mut Bound<'_>) -> Result<Run, String> {
    match lane(program, bound, None)? {
        Ended::Ran(run) => Ok(run),
        Ended::Waits(..) => Err("a single run met its quad".into()),
    }
}

/// Whether a program needs the other lanes of its 2 x 2 quad ([`run_quad`]): it takes a screen derivative, or samples
/// at an implicit level of detail, whose footprint is the quad's. A program that does not runs the same alone as in a
/// quad.
pub fn needs_quad(program: &Program) -> bool {
    program.entry.values.iter().any(|d| {
        matches!(
            d.op,
            Op::Unary(Unary::Derivative(..), _)
                | Op::Sample {
                    level: Level::Auto | Level::Bias(_),
                    ..
                }
        )
    })
}

/// Run a pixel program's entry function for a 2 x 2 quad of invocations, each lane with its own [`Bound`]: lanes 0 and
/// 1 are the top row, left to right, and 2 and 3 the bottom.
///
/// A derivative is a difference between two lanes' operands: along x the right lane's less the left's, along y the
/// lower's less the upper's, of the top row or left column for a coarse derivative (and for [`Fineness::Any`]), of the
/// lane's own for a fine one. An implicit-LOD sample (`Level::Auto`, `Level::Bias`) asks the sampler with the quad's
/// coarse derivatives of its coordinate ([`Lod::Grad`], [`Lod::GradBias`]).
///
/// The lanes run as they would in lockstep, with no threads: each runs until it meets a derivative or implicit-LOD
/// sample whose operands are not all known, and once all four have met it there, they run again knowing them. Each
/// sampler may thus be asked a request again, and must answer it the same way. Lanes must meet each such operation
/// together: at the same operation, after as many loop iterations or block executions, as often before; a lane that
/// ends (returns or kills) while the others go on to one, or that meets one elsewhere, makes the run refused as
/// divergent. So is a run of more than [`MEETING_LIMIT`] meetings.
pub fn run_quad(program: &Program, bounds: [&mut Bound<'_>; 4]) -> Result<[Run; 4], String> {
    let mut bounds = bounds;
    let mut known: Vec<[Lanes; 4]> = Vec::new();
    loop {
        let mut ran = Vec::with_capacity(4);
        let mut waits = Vec::with_capacity(4);
        for (l, bound) in bounds.iter_mut().enumerate() {
            match lane(program, bound, Some((l, &known))).map_err(|e| format!("lane {l}: {e}"))? {
                Ended::Ran(run) => ran.push(run),
                Ended::Waits(place, operand) => waits.push((l, place, operand)),
            }
        }
        if waits.is_empty() {
            return ran
                .try_into()
                .map_err(|_| "a quad of other than four lanes".to_string());
        }
        let Some(&(_, first, _)) = waits.first() else {
            return Err("a quad with no lane waiting".into());
        };
        if waits.len() < 4 || waits.iter().any(|(_, place, _)| *place != first) {
            let at: Vec<String> = (0..4)
                .map(|l| match waits.iter().find(|(w, ..)| *w == l) {
                    Some((_, p, _)) => format!(
                        "lane {l} at %{} (after {} steps, met {} times before)",
                        p.value.0, p.steps, p.before
                    ),
                    None => format!("lane {l} ended"),
                })
                .collect();
            return Err(format!(
                "a quad's lanes diverge at a derivative or an implicit-LOD sample: {}",
                at.join(", ")
            ));
        }
        if known.len() >= MEETING_LIMIT {
            return Err(format!("more than {MEETING_LIMIT} quad meetings"));
        }
        let mut operands = [[0; 4]; 4];
        for (l, _, operand) in &waits {
            if let Some(o) = operands.get_mut(*l) {
                *o = *operand;
            }
        }
        known.push(operands);
    }
}

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
