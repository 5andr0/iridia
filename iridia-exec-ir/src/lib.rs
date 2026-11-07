//! Runs a program of iridia's shader IR ([`iridia_ir::Program`]) on the CPU, one invocation or a pixel quad, by the
//! IR's own definition of each operation: the reference the lowerings are checked against.
//!
//! The arithmetic is the reference `iridia-exec` uses: IEEE 754 `f32` for the basic operations, the transcendental
//! functions (`exp2`, `log2`, `sin`, `cos`, `sqrt`, `inverseSqrt`, `pow`) in `f64` rounded once, a dot product summed
//! left to right with each step rounded. Denormals are kept, or flushed to zero (sign kept) on the inputs and result
//! of each float operation but the sign operations and the moves, by the program's float environment or, where it
//! leaves them open, by the run's [`Bound::denormals`]; a double never is. The sign and payload of the NaN an
//! operation on NaNs gives are open (IEEE 754 leaves them so, and which operand's NaN a multiplication keeps depends on
//! how the compiler orders its operands): a caller compares every NaN as one. This subset runs 32-bit scalars and
//! vectors and 64-bit ones of up to two lanes ([`runs`]), both control phases (every construct of a `Regions` body,
//! the blocks of a `Cfg` one), constant buffer reads, pipeline state, samples (answered by the caller), kill and
//! demotion; it refuses the rest by name.
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

mod wide;

use iridia_ir::{
    Axis, Binary, Block, Body, Cmp, Denormals, End, Fineness, FloatToInt, Function, LabelId, Level,
    MinMaxNan, MulZero, Node, OnZero, Op, OutOfBounds, Program, Region, Rounding, SaturateNan, Scalar,
    Terminator, Ternary, Type, Unary, ValueId,
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

/// The most loop iterations and block executions ("steps") a run takes before it is refused: the budget of [`run`] and
/// [`run_quad`]. [`run_counted`] and [`run_quad_counted`] take the budget their caller gives.
pub const STEP_LIMIT: u64 = 1 << 24;

/// The most derivatives and implicit-LOD samples a quad's lanes meet before the quad run is refused.
pub const MEETING_LIMIT: usize = 1 << 12;

/// Why a lane's run stopped before its end.
enum Stop {
    /// It failed.
    Fail(String),
    /// It was to take a step past its budget: of what kind.
    Budget(Spent),
    /// A lane of a quad reached an operation whose neighbours' operands it does not know yet.
    Meet,
}

/// What a run was to take a step past its budget of.
#[derive(Clone, Copy, Debug)]
enum Spent {
    /// A loop iteration.
    Iteration,
    /// A block execution of a graph body.
    Block,
}

impl Spent {
    /// What the refusal of a run past [`STEP_LIMIT`] names.
    fn noun(self) -> &'static str {
        match self {
            Spent::Iteration => "loop iterations",
            Spent::Block => "block executions",
        }
    }
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
    /// Each variable's elements (one for a variable that is not an array), zero where it starts undefined.
    vars: Vec<Vec<Lanes>>,
    bound: &'a mut Bound<'b>,
    outputs: Vec<Option<Lanes>>,
    steps: u64,
    /// The most steps the run takes.
    budget: u64,
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
/// An operation's name, for a refusal.
fn op_name(op: &Op) -> &'static str {
    match op {
        Op::LoadInputAt(..) => "a read of one vertex's input",
        Op::LoadOutputAt(..) => "a read of one control point's output",
        Op::LoadPatch(..) => "a patch constant read",
        Op::Builtin(_) => "a stage builtin",
        Op::Gather { .. } => "a gather",
        Op::LoadTexel { .. } => "a texel load",
        Op::Query(..) => "a resource query",
        Op::Atomic { .. } => "an atomic",
        Op::Counter(..) => "a buffer counter",
        Op::ReportHit { .. } => "a reported hit",
        Op::Wave(_) => "a wave operation",
        _ => "the operation",
    }
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
                        Ternary::Fma => fb(at(&x)).mul_add(fb(at(&y)), fb(at(&z))).to_bits(),
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
                            std::array::from_fn::<_, $n, _>(first),
                            std::array::from_fn::<_, $n, _>(flush_in),
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
        Unary::FFract => d.float(|f| f - f.floor()),
        Unary::FRound(Rounding::Floor) => d.float(f32::floor),
        Unary::FRound(Rounding::Ceil) => d.float(f32::ceil),
        Unary::FRound(Rounding::Trunc) => d.float(f32::trunc),
        Unary::FRound(Rounding::NearestEven) => d.float(f32::round_ties_even),
        // Rust's `round` is to the nearest, halves away from zero
        Unary::FRound(Rounding::NearestAway) => d.float(f32::round),
        Unary::FSqrt => d.float(|f| narrow(f64::from(f).sqrt())),
        Unary::FInverseSqrt => d.float(|f| narrow(1.0 / f64::from(f).sqrt())),
        Unary::FExp2 => d.float(|f| narrow(f64::from(f).exp2())),
        Unary::FLog2 => d.float(|f| narrow(f64::from(f).log2())),
        Unary::FSin => d.float(|f| narrow(f64::from(f).sin())),
        Unary::FCos => d.float(|f| narrow(f64::from(f).cos())),
        Unary::FTan => d.float(|f| narrow(f64::from(f).tan())),
        Unary::FAsin => d.float(|f| narrow(f64::from(f).asin())),
        Unary::FAcos => d.float(|f| narrow(f64::from(f).acos())),
        Unary::FAtan => d.float(|f| narrow(f64::from(f).atan())),
        Unary::FSinh => d.float(|f| narrow(f64::from(f).sinh())),
        Unary::FCosh => d.float(|f| narrow(f64::from(f).cosh())),
        Unary::FTanh => d.float(|f| narrow(f64::from(f).tanh())),
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
                Rounding::Floor => f32::floor,
                Rounding::Ceil => f32::ceil,
                Rounding::Trunc => f32::trunc,
                Rounding::NearestEven => f32::round_ties_even,
                Rounding::NearestAway => f32::round,
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
        Binary::FPow => d.float2(|p, q| narrow(f64::from(p).powf(f64::from(q)))),
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

impl Operands for Machine<'_, '_> {
    fn get(&self, v: ValueId) -> Result<Lanes, String> {
        Machine::get(self, v)
    }
    fn ty(&self, v: ValueId) -> Result<Type, String> {
        Machine::ty(self, v)
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
        if !runs(t) {
            return Err(format!("{t} values are not run yet").into());
        }
        self.compute(v, &def.op, t)
    }
    /// An operand of an arithmetic operation: its lanes, flushed where the run flushes and it is a float.
    fn arg(&self, v: ValueId) -> Result<Lanes, String> {
        Operands::arg(self, v, self.flush)
    }
    fn compute(&mut self, v: ValueId, op: &Op, t: Type) -> Step<Lanes> {
        if let Some(x) = eval(op, t, &*self, self.flush)? {
            return Ok(x);
        }
        let n = t.lanes;
        Ok(match op {
            Op::Literal(bits) => literal_words(bits, t),
            Op::Undef => [0; 4],
            Op::Param => return Err(format!("parameter %{} has no value here", v.0).into()),
            Op::Unary(Unary::Derivative(axis, fineness), a) => {
                let x = self.arg(*a)?;
                let m = self.ty(*a)?.lanes;
                let quad = self
                    .meet(v, x)?
                    .ok_or("a derivative needs the invocation's quad, which a single run does not have")?;
                let (from, to) = pair(*axis, *fineness, self.quad.as_ref().map_or(0, |q| q.lane));
                let lane = |l: usize| quad.get(l).copied().unwrap_or_default();
                let d = float2(m, &lane(to), &lane(from), |p, q| p - q);
                if self.flush { d.map(flushed) } else { d }
            }
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
            Op::LoadVar(var) => self
                .vars
                .get(var.0 as usize)
                .and_then(|e| e.first())
                .copied()
                .ok_or("a load of no variable")?,
            Op::LoadElement(var, index, oob) => {
                let i = self.get(*index)?[0] as usize;
                match self
                    .vars
                    .get(var.0 as usize)
                    .ok_or("a load of no variable")?
                    .get(i)
                {
                    Some(x) => *x,
                    None if *oob == OutOfBounds::Zero => [0; 4],
                    None => return Err("an undefined read past an array's end".into()),
                }
            }
            Op::LoadInputAt(..)
            | Op::LoadOutputAt(..)
            | Op::LoadPatch(..)
            | Op::Builtin(_)
            | Op::Gather { .. }
            | Op::LoadTexel { .. }
            | Op::Query(..)
            | Op::Atomic { .. }
            | Op::Counter(..)
            | Op::ReportHit { .. }
            | Op::Wave(_) => {
                return Err(format!("{} is not run here: iridia-exec runs it", op_name(op)).into());
            }
            // the pure operations, which `eval` took above
            Op::Unary(..)
            | Op::Binary(..)
            | Op::Ternary(..)
            | Op::BitInsert { .. }
            | Op::Select { .. }
            | Op::Compose(_)
            | Op::Swizzle(..)
            | Op::Splat(_)
            | Op::Dot(..)
            | Op::Convert(..)
            | Op::Bitcast(_) => return Err("a pure operation eval did not take".into()),
            Op::LoadState(r) => self
                .bound
                .state
                .iter()
                .find(|(id, _)| id == r)
                .map(|(_, x)| *x)
                .ok_or_else(|| format!("state {} is not given", r.0))?,
            Op::LoadBuffer {
                resource,
                address,
                oob,
            } => {
                if resource.index.is_some() {
                    return Err("an element of an array of constant buffers is not run here".into());
                }
                let (data, rows) = self.buffer(resource.id)?;
                // the address in wrapping 32-bit arithmetic, its low two bits ignored
                let base = match address.index {
                    None => address.offset,
                    Some(i) => self.get(i)?[0]
                        .wrapping_mul(address.stride)
                        .wrapping_add(address.offset),
                } / 4;
                let words = rows.saturating_mul(4);
                let mut out = [0; 4];
                for (k, o) in out.iter_mut().enumerate().take(usize::from(n)) {
                    let word = base
                        .checked_add(u32::try_from(k).unwrap_or(u32::MAX))
                        .filter(|w| *w < words);
                    *o = match word {
                        Some(w) => data
                            .get(w as usize / 4)
                            .and_then(|r| r.get(w as usize % 4))
                            .copied()
                            .unwrap_or(0),
                        None if *oob == OutOfBounds::Zero => 0,
                        None => return Err("an undefined read outside a buffer".into()),
                    };
                }
                out
            }
            Op::Sample {
                texture,
                sampler,
                coord,
                level,
                compare,
                offset,
                clamp,
            } => {
                if texture.index.is_some() || sampler.index.is_some() {
                    return Err("a sample of an element of an array of resources is not run here".into());
                }
                if compare.is_some() || clamp.is_some() || *offset != [0; 3] {
                    return Err("a sample with a comparison, offset or level clamp is not run here".into());
                }
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
                    texture: texture.id,
                    sampler: sampler.id,
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
                    .and_then(|e| e.first_mut())
                    .ok_or("a store to no variable")? = x;
            }
            Node::StoreElement(var, index, x, oob) => {
                let (i, x) = (self.get(*index)?[0] as usize, self.get(*x)?);
                match self
                    .vars
                    .get_mut(var.0 as usize)
                    .ok_or("a store to no variable")?
                    .get_mut(i)
                {
                    Some(e) => *e = x,
                    None if *oob == OutOfBounds::Zero => {}
                    None => return Err("an undefined write past an array's end".into()),
                }
            }
            Node::StoreBuffer { .. }
            | Node::StoreTexel { .. }
            | Node::StorePatch(..)
            | Node::Barrier(_)
            | Node::Emit(_)
            | Node::Cut(_)
            | Node::SetMeshCounts { .. }
            | Node::StoreVertex { .. }
            | Node::StoreIndices { .. }
            | Node::DispatchMesh { .. } => {
                return Err("a buffer, image or patch store, a barrier, a vertex emit, a meshlet store or a mesh launch is not run here: iridia-exec runs it".into());
            }
            Node::TraceRay { .. } | Node::CallShader { .. } => {
                return Err("a traced ray or a shader call is not run here: iridia-exec's ray tracing pipeline runs it".into());
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
            Node::If { .. } | Node::Block { .. } | Node::Loop { .. } | Node::Switch { .. } => {
                return Err("a construct in a graph block".into());
            }
        }
        Ok(())
    }
    /// Take a step, or stop where the budget is spent.
    fn step(&mut self, what: Spent) -> Step<()> {
        if self.steps >= self.budget {
            return Err(Stop::Budget(what));
        }
        self.steps = self.steps.saturating_add(1);
        Ok(())
    }
    /// A graph body from its entry block until it returns or kills: whether it killed.
    fn graph(&mut self, blocks: &[Block]) -> Step<bool> {
        let mut at = 0usize;
        loop {
            self.step(Spent::Block)?;
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
                Terminator::Switch {
                    selector,
                    cases,
                    default,
                } => {
                    let v = self.get(*selector)?[0];
                    cases.iter().find(|(c, _)| *c == v).map_or(default, |(_, t)| t)
                }
                Terminator::Return => return Ok(false),
                Terminator::Kill => return Ok(true),
                Terminator::IgnoreHit | Terminator::AcceptHitAndEndSearch => {
                    return Err("an any-hit program's decision is not run here: iridia-exec runs it".into());
                }
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
                Node::Switch {
                    label,
                    selector,
                    cases,
                    default,
                    results,
                } => {
                    let v = self.get(*selector)?[0];
                    let arm = cases
                        .iter()
                        .find(|c| c.values.contains(&v))
                        .map_or(default, |c| &c.body);
                    match self.region(arm)? {
                        Exit::Yield(vs) => self.give(results, &vs)?,
                        Exit::Br(l, vs) if l == *label => self.give(results, &vs)?,
                        other => return Ok(other),
                    }
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
                        self.step(Spent::Iteration)?;
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
                _ => self.plain(node)?,
            }
        }
        Ok(match &r.end {
            Some(End::Yield(vs)) => Exit::Yield(vs.clone()),
            Some(End::Br(l, vs)) => Exit::Br(*l, vs.clone()),
            Some(End::Continue(l, vs)) => Exit::Continue(*l, vs.clone()),
            Some(End::Return) => Exit::Return,
            Some(End::Kill) => Exit::Kill,
            Some(End::IgnoreHit | End::AcceptHitAndEndSearch) => {
                return Err("an any-hit program's decision is not run here: iridia-exec runs it".into());
            }
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
    /// It ended, in this many steps.
    Ran(Run, u64),
    Waits(Place, Lanes),
    /// It was to take a step past its budget.
    Spent(Spent),
}

/// Run a program's entry function for one lane, the quad's meetings known so far where it is a quad's.
fn lane(
    program: &Program,
    bound: &mut Bound<'_>,
    quad: Option<(usize, &[[Lanes; 4]])>,
    budget: u64,
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
        vars: f
            .vars
            .iter()
            .map(|v| {
                let len = v.len.unwrap_or(1) as usize;
                let mut e: Vec<Lanes> = v
                    .init
                    .iter()
                    .map(|bits| bits.map(|b| u32::try_from(b & 0xffff_ffff).unwrap_or(0)))
                    .collect();
                e.resize(len, [0; 4]);
                e
            })
            .collect(),
        bound,
        outputs: vec![None; program.interface.outputs.len()],
        steps: 0,
        budget,
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
        Ok(killed) => Ok(Ended::Ran(
            Run {
                outputs: m.outputs,
                killed: killed || m.demoted,
            },
            m.steps,
        )),
        Err(Stop::Fail(e)) => Err(e),
        Err(Stop::Budget(what)) => Ok(Ended::Spent(what)),
        Err(Stop::Meet) => match m.quad.and_then(|q| q.waiting) {
            Some((place, operand)) => Ok(Ended::Waits(place, operand)),
            None => Err("a run stopped at a meeting it did not record".into()),
        },
    }
}

/// How a run bounded by a step budget ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The program ended (returned or was killed).
    Complete(Run),
    /// The program was to take a step past the budget: it did not end.
    Exhausted,
}

/// A bounded run ([`run_counted`]): how it finished, and the steps it took.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counted {
    /// How the run finished.
    pub finish: Finish,
    /// The loop iterations and block executions the run took: how many the program needs where it is complete, the
    /// budget where it is exhausted.
    pub steps: u64,
}

/// Run a program's entry function once, in either control phase, taking at most `budget` loop iterations and block
/// executions: a program that needs more is [`Finish::Exhausted`], which is a result, not an error. An error is a
/// program the run refuses or fails.
pub fn run_counted(program: &Program, bound: &mut Bound<'_>, budget: u64) -> Result<Counted, String> {
    match lane(program, bound, None, budget)? {
        Ended::Ran(run, steps) => Ok(Counted {
            finish: Finish::Complete(run),
            steps,
        }),
        Ended::Spent(_) => Ok(Counted {
            finish: Finish::Exhausted,
            steps: budget,
        }),
        Ended::Waits(..) => Err("a single run met its quad".into()),
    }
}

/// Run a program's entry function once, in either control phase, refusing a run of more than [`STEP_LIMIT`] steps.
pub fn run(program: &Program, bound: &mut Bound<'_>) -> Result<Run, String> {
    match lane(program, bound, None, STEP_LIMIT)? {
        Ended::Ran(run, _) => Ok(run),
        Ended::Spent(what) => Err(spent(what)),
        Ended::Waits(..) => Err("a single run met its quad".into()),
    }
}

/// The refusal of a run past [`STEP_LIMIT`].
fn spent(what: Spent) -> String {
    format!("more than {STEP_LIMIT} {}", what.noun())
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
    match quad(program, bounds, STEP_LIMIT)? {
        QuadEnd::Complete { runs, .. } => Ok(runs),
        QuadEnd::Spent { lane, what, .. } => Err(format!("lane {lane}: {}", spent(what))),
    }
}

/// How a quad run bounded by a step budget ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuadFinish {
    /// Every lane ended (returned or was killed).
    Complete([Run; 4]),
    /// This lane was to take a step past the budget (the lowest lane of those that were).
    Exhausted {
        /// The lane.
        lane: usize,
    },
}

/// A bounded quad run ([`run_quad_counted`]): how it finished, and each lane's steps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CountedQuad {
    /// How the run finished.
    pub finish: QuadFinish,
    /// Each lane's loop iterations and block executions in the pass the run ended in (the lanes run again after each
    /// meeting): how many its run needs where the quad is complete; where it is exhausted, the budget for the lane that
    /// ran out, what the lanes before it took in that pass, and 0 for the lanes after it, which did not run.
    pub steps: [u64; 4],
}

/// [`run_quad`] with each lane taking at most `budget` loop iterations and block executions in a pass: a quad in which
/// a lane needs more is [`QuadFinish::Exhausted`], which is a result, not an error.
pub fn run_quad_counted(
    program: &Program,
    bounds: [&mut Bound<'_>; 4],
    budget: u64,
) -> Result<CountedQuad, String> {
    Ok(match quad(program, bounds, budget)? {
        QuadEnd::Complete { runs, steps } => CountedQuad {
            finish: QuadFinish::Complete(runs),
            steps,
        },
        QuadEnd::Spent { lane, steps, .. } => CountedQuad {
            finish: QuadFinish::Exhausted { lane },
            steps,
        },
    })
}

/// How a quad run under a budget ended.
enum QuadEnd {
    Complete {
        runs: [Run; 4],
        steps: [u64; 4],
    },
    /// A lane was to take a step past the budget: which, of what, and the steps as [`CountedQuad::steps`] has them.
    Spent {
        lane: usize,
        what: Spent,
        steps: [u64; 4],
    },
}

/// A quad run under `budget`.
fn quad(program: &Program, bounds: [&mut Bound<'_>; 4], budget: u64) -> Result<QuadEnd, String> {
    let mut bounds = bounds;
    let mut known: Vec<[Lanes; 4]> = Vec::new();
    loop {
        let mut ran = Vec::with_capacity(4);
        let mut steps = [0u64; 4];
        let mut waits = Vec::with_capacity(4);
        for (l, bound) in bounds.iter_mut().enumerate() {
            match lane(program, bound, Some((l, &known)), budget).map_err(|e| format!("lane {l}: {e}"))? {
                Ended::Ran(run, taken) => {
                    ran.push(run);
                    if let Some(s) = steps.get_mut(l) {
                        *s = taken;
                    }
                }
                Ended::Waits(place, operand) => {
                    if let Some(s) = steps.get_mut(l) {
                        *s = place.steps;
                    }
                    waits.push((l, place, operand));
                }
                Ended::Spent(what) => {
                    if let Some(s) = steps.get_mut(l) {
                        *s = budget;
                    }
                    return Ok(QuadEnd::Spent { lane: l, what, steps });
                }
            }
        }
        if waits.is_empty() {
            let runs = ran
                .try_into()
                .map_err(|_| "a quad of other than four lanes".to_string())?;
            return Ok(QuadEnd::Complete { runs, steps });
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

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
