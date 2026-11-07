//! Runs of a program in `iridia-exec-ir` with one source changed: the evidence for an analysis that leaves a source
//! out of what reaches a lane. A run in which the lane changes shows the analysis wrong; runs in which it never does
//! are evidence for it (not proof: a dependence may show only at values the runs did not take).
//!
//! Each run's world is drawn from a seed: each input lane a value of its varying's type (floats from a small grid, so
//! arithmetic stays finite; integers small, as indices and counts are), each constant word a small float or a small
//! integer, each pipeline state a plausible value, and each texel lane a value of the sampled texture, coordinate and
//! level hashed with the seed. A program that needs its quad runs in one, its lanes' float inputs a step apart.

use std::collections::{BTreeMap, BTreeSet};

use iridia_exec_ir::{Bound, Lanes, Lod, Run, Sample};
use iridia_exec_ir::{Counted, Finish, QuadFinish};
use iridia_ir::interface::{
    BufferLayout, ConstantType, Majorness, ResourceId, ResourceKind, ScalarType, StateParam,
};
use iridia_ir::{Body, Denormals, Function, Op as IrOp, Program, Scalar, ValueId};

/// What may be changed between two runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    /// An input slot's lane.
    Input(usize, u8),
    /// A constant buffer's 32-bit word.
    Word(ResourceId, u32),
    /// A lane of each texel of a texture.
    Texel(ResourceId, u8),
}

/// What a run is watched for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Target {
    /// An output slot's lane, where neither run is killed.
    Output(usize, u8),
    /// Whether the invocation is killed or demoted.
    Discard,
}

/// What the runs showed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The target was the same in this many pairs of runs.
    Kept(usize),
    /// A pair of runs in which the target changed, described.
    Changed(String),
    /// No pair ran: why the executor refused.
    NotRun(String),
    /// The execution needed more than this many loop iterations or CFG blocks.
    BudgetExhausted(u64),
}

/// The next power of two above the pilot's largest completed run (8193 loop iterations / CFG blocks).
pub const RUN_BUDGET: u64 = 1 << 14;

/// Nested compute integration loops reached their final store below this ceiling in the high-budget probe.
const COMPUTE_BUDGET: u64 = 1 << 20;

pub fn execution_budget(program: &Program) -> (u64, &'static str) {
    fn depth(r: &iridia_ir::Region) -> usize {
        r.nodes
            .iter()
            .map(|n| {
                usize::from(matches!(n, iridia_ir::Node::Loop { .. }))
                    + n.regions().into_iter().map(depth).max().unwrap_or(0)
            })
            .max()
            .unwrap_or(0)
    }
    if program.stage == iridia_ir::interface::Stage::Compute
        && matches!(&program.entry.body, Body::Regions(r) if depth(r) >= 2)
    {
        (COMPUTE_BUDGET, "nested compute loops")
    } else {
        (RUN_BUDGET, "default")
    }
}

/// Counts retained with each program's oracle row, including every perturbation run.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepCounts {
    pub completed: u64,
    pub maximum: u64,
    pub exhausted: u64,
    pub budget: u64,
    pub input_ambiguous: bool,
    pub enlarged: u64,
}

std::thread_local! {
    static COUNTS: std::cell::Cell<StepCounts> = const { std::cell::Cell::new(StepCounts { completed: 0, maximum: 0, exhausted: 0, budget: 0, input_ambiguous: false, enlarged: 0 }) };
    static EXHAUSTIONS: std::cell::RefCell<BTreeSet<String>> = const { std::cell::RefCell::new(BTreeSet::new()) };
}

pub fn take_counts() -> StepCounts {
    COUNTS.with(|c| c.replace(StepCounts::default()))
}

pub fn take_exhaustions() -> BTreeSet<String> {
    EXHAUSTIONS.with(|e| e.take())
}

fn counted(result: Counted) -> Counted {
    COUNTS.with(|c| {
        let mut counts = c.get();
        match &result.finish {
            Finish::Complete(_) => {
                counts.completed += 1;
                counts.maximum = counts.maximum.max(result.steps);
            }
            Finish::Exhausted => counts.exhausted += 1,
        }
        c.set(counts);
    });
    result
}

/// A splitmix64 stream.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u32 {
        u32::try_from(self.next() % n).unwrap_or(0)
    }

    /// A number below `n`, at most 256.
    fn small(&mut self, n: u64) -> u8 {
        u8::try_from(self.next() % n.min(256)).unwrap_or(0)
    }

    /// A float of the grid -4, -3.75, .., 4.
    fn float(&mut self) -> u32 {
        (f32::from(self.small(33)) / 4.0 - 4.0).to_bits()
    }

    /// A value of a scalar type.
    fn of(&mut self, scalar: ScalarType) -> u32 {
        match scalar {
            ScalarType::Float => self.float(),
            ScalarType::Bool => self.below(2),
            _ => self.below(8),
        }
    }

    /// A constant word: a float of the grid or a small integer.
    fn word(&mut self) -> u32 {
        if self.below(2) == 0 {
            self.float()
        } else {
            self.below(8)
        }
    }

    fn declared_word(&mut self, kind: Option<(ScalarType, u32)>) -> u32 {
        match kind {
            Some((ScalarType::Double, 0)) => self.next() as u32,
            Some((ScalarType::Double, _)) => (f64::from(f32::from_bits(self.float())).to_bits() >> 32) as u32,
            Some((scalar, _)) => self.of(scalar),
            None => self.word(),
        }
    }
}

/// The declared scalar containing a word, and the word's byte offset within that scalar. Padding stays untyped.
fn scalar_at(ty: &ConstantType, offset: u32) -> Option<(ScalarType, u32)> {
    match ty {
        ConstantType::Scalar(s) if offset < s.bytes() => Some((*s, offset)),
        ConstantType::Vector(v) if offset < u32::from(v.width) * v.scalar.bytes() => {
            Some((v.scalar, offset % v.scalar.bytes()))
        }
        ConstantType::Matrix {
            scalar,
            rows,
            columns,
            major,
            stride,
        } if *stride != 0 => {
            let (lines, lanes) = match major {
                Majorness::Row => (*rows, *columns),
                Majorness::Column => (*columns, *rows),
            };
            let within = offset % stride;
            (offset / stride < u32::from(lines) && within < u32::from(lanes) * scalar.bytes())
                .then_some((*scalar, within % scalar.bytes()))
        }
        ConstantType::Array {
            element,
            count,
            stride,
        } if *stride != 0 && offset / stride < *count => scalar_at(element, offset % stride),
        ConstantType::Struct { members } => members
            .iter()
            .find_map(|m| offset.checked_sub(m.offset).and_then(|at| scalar_at(&m.ty, at))),
        _ => None,
    }
}

fn buffer_scalar(layout: &BufferLayout, word: u32) -> Option<(ScalarType, u32)> {
    let offset = word.checked_mul(4)?;
    layout
        .constants
        .iter()
        .find_map(|c| offset.checked_sub(c.offset).and_then(|at| scalar_at(&c.ty, at)))
}

type Word = (ResourceId, u32);

/// Follow only direct lane moves back to a fixed buffer word. Arithmetic and memory are deliberately not inferred.
fn buffer_word(f: &Function, value: ValueId, lane: usize, remaining: usize) -> Option<Word> {
    let d = f.value(value)?;
    let next = remaining.checked_sub(1)?;
    match &d.op {
        IrOp::LoadBuffer {
            resource, address, ..
        } if address.index.is_none() && d.ty.scalar.bits() == 32 => Some((
            resource.id,
            address
                .offset
                .checked_div(4)?
                .checked_add(u32::try_from(lane).ok()?)?,
        )),
        IrOp::Swizzle(a, pick) => buffer_word(f, *a, usize::from(*pick.get(lane)?), next),
        IrOp::Splat(a) => buffer_word(f, *a, 0, next),
        IrOp::Bitcast(a) if f.ty(*a)?.lanes == d.ty.lanes && f.ty(*a)?.scalar.bits() == 32 => {
            buffer_word(f, *a, lane, next)
        }
        IrOp::Compose(parts) => {
            let mut lane = lane;
            for &part in parts {
                let n = usize::from(f.ty(part)?.lanes);
                if lane < n {
                    return buffer_word(f, part, lane, next);
                }
                lane -= n;
            }
            None
        }
        _ => None,
    }
}

/// Numeric interpretations differ from operations on the same raw integer bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum WordView {
    Float,
    Signed,
    Unsigned,
    Bits,
}

fn inferred_kind(views: &BTreeSet<WordView>) -> Result<Option<ScalarType>, ()> {
    use WordView::*;
    if (views.contains(&Float) && views.len() > 1) || (views.contains(&Signed) && views.contains(&Unsigned)) {
        return Err(());
    }
    Ok(if views.contains(&Float) {
        Some(ScalarType::Float)
    } else if views.contains(&Signed) {
        Some(ScalarType::Sint)
    } else if !views.is_empty() {
        Some(ScalarType::Uint)
    } else {
        None
    })
}

fn operand_views(f: &Function, d: &iridia_ir::ValueDef) -> Vec<(ValueId, WordView)> {
    use iridia_ir::{Binary as B, Cmp, Ternary, Unary};
    let operands = match &d.op {
        IrOp::Unary(_, a) | IrOp::Convert(a, _) => vec![*a],
        IrOp::Binary(_, a, b) | IrOp::Dot(a, b) => vec![*a, *b],
        IrOp::Ternary(_, a, b, c) => vec![*a, *b, *c],
        _ => Vec::new(),
    };
    operands
        .into_iter()
        .enumerate()
        .filter_map(|(index, value)| {
            let scalar = f.ty(value)?.scalar;
            if scalar.bits() != 32 {
                return None;
            }
            let bits = match &d.op {
                IrOp::Unary(
                    Unary::Not
                    | Unary::INeg
                    | Unary::CountOnes
                    | Unary::BitReverse
                    | Unary::FirstBitHigh
                    | Unary::FirstBitLow
                    | Unary::F16ToF32,
                    _,
                ) => true,
                IrOp::Binary(
                    B::IAdd
                    | B::ISub
                    | B::IMul
                    | B::And
                    | B::Or
                    | B::Xor
                    | B::Shl
                    | B::SCmp(Cmp::Eq | Cmp::Ne)
                    | B::UCmp(Cmp::Eq | Cmp::Ne),
                    _,
                    _,
                ) => true,
                IrOp::Binary(B::ShrU | B::ShrS, _, _) => index == 1,
                IrOp::Ternary(Ternary::BitExtract { .. }, _, _, _) => index > 0,
                _ => false,
            };
            let view = if bits {
                WordView::Bits
            } else {
                match scalar {
                    Scalar::F32 => WordView::Float,
                    Scalar::I32 => WordView::Signed,
                    Scalar::U32 => WordView::Unsigned,
                    _ => return None,
                }
            };
            Some((value, view))
        })
        .collect()
}

/// Every explicit scalar view of directly loaded words. Conflicting views stay conflicting, never resolved by order.
fn read_values(f: &Function) -> BTreeSet<ValueId> {
    fn nodes(f: &Function, ns: &[iridia_ir::Node], roots: &mut Vec<ValueId>) {
        for n in ns {
            roots.extend(n.values());
            if let iridia_ir::Node::Value(v) = n
                && f.value(*v).is_some_and(|d| d.op.has_effect())
            {
                roots.push(*v);
            }
            for inner in n.regions() {
                region(f, inner, roots);
            }
        }
    }
    fn region(f: &Function, r: &iridia_ir::Region, roots: &mut Vec<ValueId>) {
        nodes(f, &r.nodes, roots);
        if let Some(iridia_ir::End::Yield(v) | iridia_ir::End::Br(_, v) | iridia_ir::End::Continue(_, v)) =
            &r.end
        {
            roots.extend(v);
        }
    }
    let mut roots = Vec::new();
    match &f.body {
        Body::Regions(r) => region(f, r, &mut roots),
        Body::Cfg(blocks) => {
            for b in blocks {
                nodes(f, &b.nodes, &mut roots);
                for t in b.term.targets() {
                    roots.extend(&t.args);
                }
                match b.term {
                    iridia_ir::Terminator::Branch { cond, .. } => roots.push(cond),
                    iridia_ir::Terminator::Switch { selector, .. } => roots.push(selector),
                    _ => {}
                }
            }
        }
    }
    let mut read = BTreeSet::new();
    while let Some(v) = roots.pop() {
        if read.insert(v)
            && let Some(d) = f.value(v)
        {
            roots.extend(d.op.operands());
        }
    }
    read
}

fn word_views(program: &Program) -> BTreeMap<Word, BTreeSet<WordView>> {
    let mut views: BTreeMap<Word, BTreeSet<WordView>> = BTreeMap::new();
    for f in std::iter::once(&program.entry).chain(program.patch.as_ref()) {
        let read = read_values(f);
        for id in read {
            let Some(d) = f.value(id) else {
                continue;
            };
            let operands = operand_views(f, d);
            for (value, scalar) in operands {
                for lane in 0..usize::from(f.ty(value).map_or(0, |t| t.lanes)) {
                    if let Some(word) = buffer_word(f, value, lane, 64) {
                        views.entry(word).or_default().insert(scalar);
                    }
                }
            }
        }
    }
    views
}

/// The operations responsible for a sample's conflicting, unreflected word views.
pub fn view_report(program: &Program) -> String {
    let views = word_views(program);
    let mut out = String::new();
    for (word, kinds) in views
        .iter()
        .filter(|(_, kinds)| inferred_kind(kinds).is_err())
        .filter(|((r, w), _)| {
            program
                .interface
                .resources
                .iter()
                .find(|res| res.id == *r)
                .is_some_and(|res| match &res.kind {
                    ResourceKind::ConstantBuffer { layout } => buffer_scalar(layout, *w).is_none(),
                    _ => false,
                })
        })
        .take(3)
    {
        use std::fmt::Write as _;
        writeln!(out, "{word:?} views {kinds:?}").unwrap();
        let mut examples = BTreeMap::new();
        for f in std::iter::once(&program.entry).chain(program.patch.as_ref()) {
            let read = read_values(f);
            for (id, d) in f.values.iter().enumerate() {
                if !read.contains(&ValueId(id as u32)) {
                    continue;
                }
                for (value, scalar) in operand_views(f, d) {
                    if (0..usize::from(f.ty(value).map_or(0, |t| t.lanes)))
                        .any(|lane| buffer_word(f, value, lane, 64) == Some(*word))
                    {
                        examples
                            .entry(scalar)
                            .or_insert_with(|| format!("%{id}: {:?} = {:?}", d.ty, d.op));
                    }
                }
            }
        }
        writeln!(out, "{examples:?}").unwrap();
    }
    out
}

/// The inputs, buffers and textures of one run.
#[derive(Clone)]
struct World {
    inputs: Vec<Lanes>,
    floats: Vec<[bool; 4]>,
    buffers: Vec<(ResourceId, Vec<Lanes>)>,
    state: Vec<(ResourceId, Lanes)>,
    seed: u64,
    /// A texture's lane whose every texel is changed, and the salt it is changed by.
    texel: Option<(ResourceId, u8, u64)>,
    word_types: BTreeMap<Word, (ScalarType, u32)>,
}

impl World {
    fn new(program: &Program, seed: u64) -> World {
        let mut rng = Rng(seed);
        let views = word_views(program);
        let mut word_types = BTreeMap::new();
        let interface = &program.interface;
        let inputs = interface
            .inputs
            .iter()
            .map(|v| core::array::from_fn(|_| rng.of(v.value.scalar)))
            .collect();
        let floats = interface
            .inputs
            .iter()
            .map(|v| [v.value.scalar == ScalarType::Float; 4])
            .collect();
        let mut buffers = Vec::new();
        let mut state = Vec::new();
        for r in &interface.resources {
            match &r.kind {
                ResourceKind::ConstantBuffer { layout } => {
                    let rows = layout.size.div_ceil(16).min(1 << 16);
                    for word in 0..rows * 4 {
                        let declared = buffer_scalar(layout, word);
                        let inferred = views.get(&(r.id, word));
                        let kind = declared.or_else(|| {
                            let views = inferred?;
                            match inferred_kind(views) {
                                Err(()) => {
                                    COUNTS.with(|c| {
                                        let mut n = c.get();
                                        n.input_ambiguous = true;
                                        c.set(n);
                                    });
                                    None
                                }
                                Ok(kind) => kind.map(|scalar| (scalar, 0)),
                            }
                        });
                        if let Some(k) = kind {
                            word_types.insert((r.id, word), k);
                        }
                    }
                    buffers.push((
                        r.id,
                        (0..rows)
                            .map(|row| {
                                core::array::from_fn(|lane| {
                                    rng.declared_word(word_types.get(&(r.id, row * 4 + lane as u32)).copied())
                                })
                            })
                            .collect(),
                    ));
                }
                ResourceKind::PipelineState(param) => {
                    let value = match param {
                        StateParam::Projected(_) => [0; 4],
                        StateParam::TextureDim(_) => [2, 0, 0, 0],
                        _ => core::array::from_fn(|_| rng.float()),
                    };
                    state.push((r.id, value));
                }
                _ => {}
            }
        }
        World {
            inputs,
            floats,
            buffers,
            state,
            seed,
            texel: None,
            word_types,
        }
    }

    /// The world with `source` changed to another value of its kind, or `None` where it is not there to change (a
    /// word past its buffer's end).
    fn changed(&self, program: &Program, source: Source, rng: &mut Rng) -> Option<World> {
        let mut w = self.clone();
        match source {
            Source::Input(slot, lane) => {
                let scalar = program.interface.inputs.get(slot)?.value.scalar;
                let x = w.inputs.get_mut(slot)?.get_mut(usize::from(lane))?;
                let was = *x;
                while *x == was {
                    *x = rng.of(scalar);
                }
            }
            Source::Word(resource, word) => {
                let scalar = self.word_types.get(&(resource, word)).copied();
                let (row, lane) = (usize::try_from(word / 4).ok()?, usize::try_from(word % 4).ok()?);
                let x = match w.buffers.iter_mut().find(|(r, _)| *r == resource) {
                    Some((_, rows)) => rows.get_mut(row)?.get_mut(lane)?,
                    // a state parameter's lanes are its words
                    None => w
                        .state
                        .iter_mut()
                        .find(|(r, _)| *r == resource)?
                        .1
                        .get_mut(usize::try_from(word).ok()?)?,
                };
                let was = *x;
                while *x == was {
                    *x = rng.declared_word(scalar);
                }
            }
            Source::Texel(resource, lane) => w.texel = Some((resource, lane, rng.next() | 1)),
        }
        Some(w)
    }

    /// The texel lanes a sample answers: hashed from the texture, coordinate and level with the seed, each a float in
    /// 0..1 by sixteenths, the changed lane of the changed texture another.
    fn sample(&self, s: &Sample) -> [f32; 4] {
        let mut h = Rng(self.seed ^ u64::from(s.texture.0).wrapping_mul(0x1000_0000_01b3));
        let lod = match s.lod {
            Lod::Exact(l) | Lod::Bias(l) => l.to_bits(),
            _ => 0,
        };
        for c in s
            .coord
            .iter()
            .take(usize::from(s.lanes))
            .map(|c| c.to_bits())
            .chain([lod])
        {
            h.0 ^= u64::from(c);
            h.next();
        }
        let mut out: [f32; 4] = core::array::from_fn(|_| f32::from(h.small(17)) / 16.0);
        if let Some((texture, lane, salt)) = self.texel
            && texture == s.texture
        {
            let mut g = Rng(h.0 ^ salt);
            let x = &mut out[usize::from(lane & 3)];
            *x += 1.0 + f32::from(g.small(16)) / 16.0;
        }
        out
    }

    /// The world's inputs for a quad's lane: float lanes a quarter along x for the right lanes, an eighth along y
    /// for the lower.
    fn quad_inputs(&self, lane: usize) -> Vec<Lanes> {
        let step = f32::from(u8::try_from(lane & 1).unwrap_or(0)) * 0.25
            + f32::from(u8::try_from(lane >> 1).unwrap_or(0)) * 0.125;
        self.inputs
            .iter()
            .zip(&self.floats)
            .map(|(x, f)| {
                core::array::from_fn(|k| {
                    if f[k] {
                        (f32::from_bits(x[k]) + step).to_bits()
                    } else {
                        x[k]
                    }
                })
            })
            .collect()
    }

    /// The program run in this world (lane 0 of a quad where it needs one).
    fn run(&self, program: &Program) -> Result<Counted, String> {
        let result = self.run_with_budget(program, RUN_BUDGET)?;
        let maximum = execution_budget(program).0;
        let result = if matches!(result.finish, Finish::Exhausted) && maximum > RUN_BUDGET {
            COUNTS.with(|c| {
                let mut n = c.get();
                n.enlarged += 1;
                c.set(n);
            });
            self.run_with_budget(program, maximum).map(counted)
        } else {
            Ok(counted(result))
        };
        if matches!(
            &result,
            Ok(Counted {
                finish: Finish::Exhausted,
                ..
            })
        ) {
            let reason = match program
                .provenance
                .source_hash
                .as_deref()
                .and_then(|h| h.strip_prefix("sha256:"))
            {
                Some("f61b7e97b8b31c2f98e733d19b03fbd305b103d7988a833513714c295cc262cd") => format!(
                    "a two-word countdown from 0xffffffff00000000 (18446744069414584320) needs about 2^64 iterations, beyond the budget; input.x={}; input.x=0 early exit compared separately",
                    self.inputs.first().map_or(0, |x| x[0])
                ),
                Some("41ee8c4dae89015d52e77b640b2bc8a8e529bfab941e1b1a81a87fe5177f78bc") => {
                    let bound = program
                        .interface
                        .resources
                        .iter()
                        .find(|r| {
                            r.provenance.d3d.is_some_and(|d| {
                                d.class == iridia_ir::interface::D3dClass::ConstantBuffer
                                    && d.register == 4
                                    && d.space == 0
                            })
                        })
                        .and_then(|r| self.word(r.id, 46));
                    format!(
                        "input type ambiguous: cb4[11].z has signed and unsigned numeric views; mixed raw fill gives initial loop bound {bound:?}, counter advances by four; class probe also exhausted 1048576 steps"
                    )
                }
                _ => "execution exhaustion cause not judged".into(),
            };
            EXHAUSTIONS.with(|e| {
                e.borrow_mut().insert(reason);
            });
        }
        result
    }

    fn run_with_budget(&self, program: &Program, budget: u64) -> Result<Counted, String> {
        COUNTS.with(|c| {
            let mut n = c.get();
            n.budget = n.budget.max(budget);
            c.set(n);
        });
        let buffers: Vec<(ResourceId, &[Lanes])> =
            self.buffers.iter().map(|(r, d)| (*r, d.as_slice())).collect();
        let state = self.state.as_slice();
        if !iridia_exec_ir::needs_quad(program) {
            let mut sampler = |s: Sample| Ok(self.sample(&s));
            return iridia_exec_ir::run_counted(
                program,
                &mut bound(&self.inputs, &buffers, state, &mut sampler),
                budget,
            );
        }
        let inputs: [Vec<Lanes>; 4] = core::array::from_fn(|l| self.quad_inputs(l));
        let mut samplers: [_; 4] = core::array::from_fn(|_| |s: Sample| Ok(self.sample(&s)));
        let [s0, s1, s2, s3] = &mut samplers;
        let [i0, i1, i2, i3] = &inputs;
        let result = iridia_exec_ir::run_quad_counted(
            program,
            [
                &mut bound(i0, &buffers, state, s0),
                &mut bound(i1, &buffers, state, s1),
                &mut bound(i2, &buffers, state, s2),
                &mut bound(i3, &buffers, state, s3),
            ],
            budget,
        )?;
        let steps = result.steps.into_iter().max().unwrap_or(0);
        let finish = match result.finish {
            QuadFinish::Complete([r, ..]) => Finish::Complete(r),
            QuadFinish::Exhausted { .. } => Finish::Exhausted,
        };
        Ok(Counted { finish, steps })
    }
}

/// An explicitly selected world run once with a diagnostic budget, including its raw loop-bound inputs.
pub fn budget_probe(program: &Program, seed: u64, source: Option<Source>, budget: u64) {
    let world = World::new(program, seed);
    let world = match source {
        Some(s) => world
            .changed(program, s, &mut Rng(seed ^ 0x5eed))
            .expect("the probe source exists"),
        None => world,
    };
    eprintln!("probe seed={seed}, source={source:?}, budget={budget}");
    eprintln!("inputs: {:x?}", world.inputs);
    for (resource, rows) in &world.buffers {
        eprintln!(
            "buffer {resource:?}: {:x?}",
            rows.iter().take(64).collect::<Vec<_>>()
        );
    }
    let started = std::time::Instant::now();
    eprintln!(
        "probe result: {:?}; elapsed {:?}",
        world.run_with_budget(program, budget),
        started.elapsed()
    );
}

/// A run's resources and inputs.
fn bound<'a>(
    inputs: &'a [Lanes],
    buffers: &'a [(ResourceId, &'a [Lanes])],
    state: &'a [(ResourceId, Lanes)],
    sampler: &'a mut dyn FnMut(Sample) -> Result<[f32; 4], String>,
) -> Bound<'a> {
    Bound {
        inputs,
        buffers,
        state,
        sampler,
        denormals: Denormals::Any,
    }
}

fn output_word(scalar: Option<ScalarType>, word: u32) -> u32 {
    if scalar == Some(ScalarType::Float) && f32::from_bits(word).is_nan() {
        0x7fc0_0000
    } else {
        word
    }
}

#[test]
fn nan_normalisation_preserves_integer_outputs() {
    assert_eq!(output_word(Some(ScalarType::Float), 0xffff_ffff), 0x7fc0_0000);
    for scalar in [ScalarType::Sint, ScalarType::Uint, ScalarType::Bool] {
        assert_eq!(output_word(Some(scalar), 0xffff_ffff), 0xffff_ffff);
        assert_eq!(output_word(Some(scalar), 0xffff_fffe), 0xffff_fffe);
    }
}

/// An output lane of a run: `None` where the slot is not stored, a float NaN as one NaN whatever its payload.
fn lane(program: &Program, run: &Run, slot: usize, lane: u8) -> Option<u32> {
    let x = (*run.outputs.get(slot)?)?[usize::from(lane & 3)];
    Some(output_word(
        program.interface.outputs.get(slot).map(|v| v.value.scalar),
        x,
    ))
}

/// `runs` pairs of runs of `program`, each pair in a world of its own seed and that world with `source` changed,
/// watched for `target` changing.
pub fn check(program: &Program, target: Target, source: Source, runs: u64) -> Outcome {
    let mut kept = 0;
    let mut refused = None;
    for seed in 0..runs {
        let world = World::new(program, seed);
        let mut rng = Rng(seed ^ 0x5eed);
        let Some(other) = world.changed(program, source, &mut rng) else {
            continue;
        };
        let (a, b) = match (world.run(program), other.run(program)) {
            (
                Ok(Counted {
                    finish: Finish::Complete(a),
                    ..
                }),
                Ok(Counted {
                    finish: Finish::Complete(b),
                    ..
                }),
            ) => (a, b),
            (
                Ok(Counted {
                    finish: Finish::Exhausted,
                    ..
                }),
                _,
            )
            | (
                _,
                Ok(Counted {
                    finish: Finish::Exhausted,
                    ..
                }),
            ) => {
                eprintln!("execution budget exhausted: seed={seed} target={target:?} source={source:?}");
                return Outcome::BudgetExhausted(execution_budget(program).0);
            }
            (Err(e), _) | (_, Err(e)) => {
                refused.get_or_insert(e);
                continue;
            }
        };
        let changed = match target {
            Target::Discard => (a.killed != b.killed).then(|| format!("killed {} -> {}", a.killed, b.killed)),
            Target::Output(slot, l) if !a.killed && !b.killed => {
                let (x, y) = (lane(program, &a, slot, l), lane(program, &b, slot, l));
                (x != y).then(|| format!("{x:x?} -> {y:x?}"))
            }
            Target::Output(..) => continue,
        };
        if let Some(what) = changed {
            return Outcome::Changed(format!("{target:?} {what} with {source:?} changed (seed {seed})"));
        }
        kept += 1;
    }
    match (kept, refused) {
        (0, Some(e)) => Outcome::NotRun(e),
        (0, None) => Outcome::NotRun("no surviving executions".into()),
        (n, _) => Outcome::Kept(n),
    }
}

impl World {
    /// A constant word or a pipeline state lane of the world.
    fn word(&self, resource: ResourceId, word: u32) -> Option<u32> {
        let (row, lane) = (usize::try_from(word / 4).ok()?, usize::try_from(word % 4).ok()?);
        match self.buffers.iter().find(|(r, _)| *r == resource) {
            Some((_, rows)) => rows.get(row)?.get(lane).copied(),
            None => self
                .state
                .iter()
                .find(|(r, _)| *r == resource)?
                .1
                .get(usize::try_from(word).ok()?)
                .copied(),
        }
    }
}

/// One execution of a selected test input, recorded independently of analysis agreement.
#[derive(Debug, Default)]
pub struct Execution {
    pub complete: Option<Run>,
    pub not_run: Option<String>,
}

pub fn execution(program: &Program) -> Execution {
    execute(program, &World::new(program, 0))
}

pub fn execution_with_input(program: &Program, input: usize, lane: usize, word: u32) -> Execution {
    let mut world = World::new(program, 0);
    world.inputs[input][lane] = word;
    execute(program, &world)
}

fn execute(program: &Program, world: &World) -> Execution {
    match world.run(program) {
        Ok(Counted {
            finish: Finish::Complete(r),
            ..
        }) if r.killed => Execution {
            not_run: Some("the run is killed".into()),
            ..Default::default()
        },
        Ok(Counted {
            finish: Finish::Complete(r),
            ..
        }) => Execution {
            complete: Some(r),
            ..Default::default()
        },
        Ok(Counted {
            finish: Finish::Exhausted,
            ..
        }) => Execution {
            not_run: Some(format!("budget exhausted at {}", execution_budget(program).0)),
            ..Default::default()
        },
        Err(e) => Execution {
            not_run: Some(e),
            ..Default::default()
        },
    }
}
