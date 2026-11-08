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

pub mod value;

use iridia_ir::{
    Axis, Block, Body, Denormals, End, Fineness, Function, LabelId, Level, Node, Op, OutOfBounds, Program,
    Region, Terminator, Type, Unary, ValueId,
    interface::{ResourceId, ResourceKind},
};

// Existing callers keep the executor's API; the numeric definition lives with the IR.
use iridia_ir::eval::flush_denormal as flushed;
pub use iridia_ir::eval::{Lanes, Operands, Prepared, eval, literal_words, runs};

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

/// The lanes a mask names, in ascending order.
fn masked(mask: u8) -> impl Iterator<Item = usize> {
    (0..4).filter(move |n| mask >> n & 1 != 0)
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
        Op::RayQueryProceed(_) | Op::RayQueryState(..) => "a ray query",
        Op::Wave(_) => "a wave operation",
        _ => "the operation",
    }
}

fn fb(w: u32) -> f32 {
    f32::from_bits(w)
}
fn float2(n: u8, a: &Lanes, b: &Lanes, op: impl Fn(f32, f32) -> f32) -> Lanes {
    let mut out = [0; 4];
    for ((o, x), y) in out.iter_mut().zip(a).zip(b).take(usize::from(n)) {
        *o = op(fb(*x), fb(*y)).to_bits();
    }
    out
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
            | Op::RayQueryProceed(_)
            | Op::RayQueryState(..)
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
            Node::RayQueryInit { .. }
            | Node::RayQueryAbort(_)
            | Node::RayQueryCommitTriangle(_)
            | Node::RayQueryCommitProcedural { .. } => {
                return Err("a ray query is not run here: iridia-exec runs it".into());
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

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
