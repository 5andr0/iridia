//! Amplified rounding: an output lane that is `fract(x * c)` for a constant `c` of magnitude 1e3 or more (a hash such as
//! `fract(sin(dot(p, k)) * 43758.5453)`, a depth packed as `fract(depth * 65025)`) turns the few ulp WGSL lets `x` be off
//! by into a difference anywhere in the lane's range. Such a lane is told apart from an error by the program's data flow,
//! never by its name, and only where the difference it shows implies a difference in `x` within the bound WGSL puts on
//! the operation that rounds `x` ([`Amplified::explains`]). A lane that only depends on such a value further down
//! ([`downstream`]) has no such bound to check: it is named, not excused.

use std::collections::{BTreeMap, VecDeque};

use iridia_ir::interface::Varying;
use iridia_ir::{Binary, Body, End, Function, LabelId, Node, Op, Program, Region, Unary, ValueId};

/// The least magnitude of a constant that amplifies.
const AMPLIFYING: f32 = 1e3;

/// The operation that rounds `x`, which WGSL bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Source {
    /// `sin` or `cos`: absolute error at most 2^-11 (WGSL states it within -pi..pi and nothing outside, so the bound is
    /// the most WGSL ever allows).
    SinCos,
    /// `x / y`: 2.5 ulp.
    Division,
    /// `+`, `-` or `*`: correctly rounded, either float beside the exact result (1 ulp).
    Arithmetic,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::SinCos => "sin or cos",
            Source::Division => "division",
            Source::Arithmetic => "add, subtract or multiply",
        }
    }
}

/// An output lane that is `fract(x * constant)`, `x` rounded by `source` and at most `magnitude` in magnitude
/// (`f32::MAX` where the program does not bound it).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Amplified {
    pub constant: f32,
    pub source: Source,
    pub magnitude: f32,
}

/// The largest distance between neighbouring floats of magnitude at most `m` (WGSL's ulp: the least distance between
/// two floats around a value).
fn ulp(m: f32) -> f64 {
    let bits = m.abs().to_bits();
    let mut exponent = i32::try_from(bits >> 23).unwrap_or(255) - 127;
    if bits & 0x007f_ffff == 0 {
        // a power of two is the top of the binade below it
        exponent -= 1;
    }
    2f64.powi(exponent.max(-126) - 23)
}

impl Amplified {
    /// The most `x` may be off by under WGSL.
    pub fn bound(&self) -> f64 {
        match self.source {
            Source::SinCos => 2f64.powi(-11),
            Source::Division => 2.5 * ulp(self.magnitude),
            Source::Arithmetic => ulp(self.magnitude),
        }
    }

    /// Whether the lane's words `a` and `b`, each `fract(x * c)` of its own `x`, imply `x`s within [`Amplified::bound`]
    /// of each other: two fractions `d` apart (`1 - d` across a whole number) need `x`s at least `min(d, 1 - d) / |c|`
    /// apart.
    pub fn explains(&self, a: u32, b: u32) -> bool {
        let (a, b) = (f64::from(f32::from_bits(a)), f64::from(f32::from_bits(b)));
        let fraction = |x: f64| (0.0..=1.0).contains(&x);
        if !fraction(a) || !fraction(b) {
            return false;
        }
        let d = (a - b).abs();
        d.min(1.0 - d) / f64::from(self.constant.abs()) <= self.bound()
    }
}

/// The value and lane that lane `k` of `v` moves from, through swizzles, splats and composes.
fn moved(f: &Function, mut v: ValueId, mut k: usize) -> (ValueId, usize) {
    loop {
        let Some(def) = f.value(v) else { return (v, k) };
        match &def.op {
            Op::Swizzle(a, lanes) => {
                v = *a;
                k = usize::from(lanes[k.min(3)]);
            }
            Op::Splat(a) => {
                v = *a;
                k = 0;
            }
            Op::Compose(parts) => {
                let mut at = 0;
                let Some((part, lane)) = parts.iter().find_map(|&p| {
                    let n = usize::from(f.ty(p).map_or(1, |t| t.lanes));
                    let hit = (k < at + n).then_some((p, k - at));
                    at += n;
                    hit
                }) else {
                    return (v, k);
                };
                v = part;
                k = lane;
            }
            _ => return (v, k),
        }
    }
}

/// Lane `k` of `v` as an `f32` constant.
fn constant(f: &Function, v: ValueId, k: usize) -> Option<f32> {
    let (v, k) = moved(f, v, k);
    let def = f.value(v)?;
    match def.op {
        Op::Literal(lanes) if def.ty.scalar == iridia_ir::Scalar::F32 => {
            Some(f32::from_bits(u32::try_from(lanes[k.min(3)] & 0xffff_ffff).ok()?))
        }
        _ => None,
    }
}

/// The operation that rounds lane `k` of `x`, through the operations that change no value's precision (a clamp, a
/// negation, an absolute value), and the magnitude those bound it to.
fn source(f: &Function, x: ValueId, k: usize) -> Option<(Source, f32)> {
    let mut magnitude = f32::MAX;
    let (mut x, mut k) = moved(f, x, k);
    loop {
        match &f.value(x)?.op {
            Op::Unary(Unary::Saturate(_), a) => {
                magnitude = magnitude.min(1.0);
                (x, k) = moved(f, *a, k);
            }
            Op::Unary(Unary::ClampRange { lo, hi, .. }, a) => {
                let m = f32::from_bits(*lo).abs().max(f32::from_bits(*hi).abs());
                magnitude = magnitude.min(m);
                (x, k) = moved(f, *a, k);
            }
            Op::Unary(Unary::FNeg | Unary::FAbs, a) => (x, k) = moved(f, *a, k),
            Op::Unary(Unary::FSin | Unary::FCos, _) => return Some((Source::SinCos, magnitude.min(1.0))),
            Op::Binary(Binary::FDiv, ..) => return Some((Source::Division, magnitude)),
            Op::Binary(Binary::FAdd | Binary::FSub | Binary::FMul(_), ..) => {
                return Some((Source::Arithmetic, magnitude));
            }
            _ => return None,
        }
    }
}

/// Lane `k` of `v` as `fract(x * c)` with `|c|` at least [`AMPLIFYING`].
fn amplification(f: &Function, v: ValueId, k: usize) -> Option<Amplified> {
    let (v, k) = moved(f, v, k);
    let Op::Unary(Unary::FFract, m) = f.value(v)?.op else {
        return None;
    };
    let (m, k) = moved(f, m, k);
    let Op::Binary(Binary::FMul(_), a, b) = f.value(m)?.op else {
        return None;
    };
    [(a, b), (b, a)].into_iter().find_map(|(x, c)| {
        let constant = constant(f, c, k).filter(|c| c.is_finite() && c.abs() >= AMPLIFYING)?;
        let (source, magnitude) = source(f, x, k)?;
        Some(Amplified {
            constant,
            source,
            magnitude,
        })
    })
}

/// Each node of a body, constructs' regions included.
fn nodes(body: &Body) -> Vec<&Node> {
    fn region<'a>(r: &'a Region, out: &mut Vec<&'a Node>) {
        for n in &r.nodes {
            out.push(n);
            for inner in n.regions() {
                region(inner, out);
            }
        }
    }
    let mut out = Vec::new();
    match body {
        Body::Regions(r) => region(r, &mut out),
        Body::Cfg(blocks) => out.extend(blocks.iter().flat_map(|b| &b.nodes)),
    }
    out
}

/// The D3D register lanes a varying's lanes are.
fn register_lanes(v: &Varying) -> Option<(u32, Vec<usize>)> {
    let sig = v.provenance.d3d?;
    Some((sig.register, (0..4).filter(|b| sig.mask >> b & 1 != 0).collect()))
}

/// Each output store's lanes by D3D register and lane, with the value and lane of the value that lane takes.
fn stores(p: &Program) -> Vec<((u32, usize), ValueId, usize)> {
    let mut out = Vec::new();
    for n in nodes(&p.entry.body) {
        let Node::StoreOutput(slot, mask, v) = n else {
            continue;
        };
        let Some((register, lanes)) = p.interface.outputs.get(*slot as usize).and_then(register_lanes) else {
            continue;
        };
        for (k, lane) in (0..4usize).filter(|l| mask >> l & 1 != 0).enumerate() {
            if let Some(&r) = lanes.get(lane) {
                out.push(((register, r), *v, k));
            }
        }
    }
    out
}

/// The output lanes, by D3D register and lane, that every store of them gives `fract(x * c)` of one amplification.
pub fn outputs(p: &Program) -> BTreeMap<(u32, usize), Amplified> {
    let mut seen: BTreeMap<(u32, usize), Option<Amplified>> = BTreeMap::new();
    for (at, v, k) in stores(p) {
        let a = amplification(&p.entry, v, k);
        seen.entry(at)
            .and_modify(|was| {
                if *was != a {
                    *was = None;
                }
            })
            .or_insert(a);
    }
    seen.into_iter().filter_map(|(at, a)| Some((at, a?))).collect()
}

/// The values each construct parameter or result (`Op::Param`) takes: a loop's initial values and continues', a
/// construct's yields and exits', a `Cfg` block's branch arguments.
fn param_sources(f: &Function) -> BTreeMap<ValueId, Vec<ValueId>> {
    struct Walk<'a> {
        out: BTreeMap<ValueId, Vec<ValueId>>,
        /// Per enclosing label: the construct's results, and a loop's parameters.
        labels: Vec<(LabelId, &'a [ValueId], &'a [ValueId])>,
    }
    impl<'a> Walk<'a> {
        fn give(&mut self, to: &[ValueId], values: &[ValueId]) {
            for (p, v) in to.iter().zip(values) {
                self.out.entry(*p).or_default().push(*v);
            }
        }
        fn label(&self, l: LabelId) -> Option<(&'a [ValueId], &'a [ValueId])> {
            self.labels
                .iter()
                .rev()
                .find(|(at, ..)| *at == l)
                .map(|&(_, results, params)| (results, params))
        }
        fn region(&mut self, r: &'a Region, results: &'a [ValueId]) {
            for n in &r.nodes {
                match n {
                    Node::If {
                        accept,
                        reject,
                        results,
                        ..
                    } => {
                        self.region(accept, results);
                        self.region(reject, results);
                    }
                    Node::Block { label, body, results } => {
                        self.labels.push((*label, results, &[]));
                        self.region(body, results);
                        self.labels.pop();
                    }
                    Node::Loop {
                        label,
                        params,
                        inits,
                        body,
                        results,
                    } => {
                        self.give(params, inits);
                        self.labels.push((*label, results, params));
                        self.region(body, results);
                        self.labels.pop();
                    }
                    Node::Switch {
                        label,
                        cases,
                        default,
                        results,
                        ..
                    } => {
                        self.labels.push((*label, results, &[]));
                        for c in cases {
                            self.region(&c.body, results);
                        }
                        self.region(default, results);
                        self.labels.pop();
                    }
                    _ => {}
                }
            }
            match &r.end {
                Some(End::Yield(values)) => self.give(results, values),
                Some(End::Br(l, values)) => {
                    if let Some((to, _)) = self.label(*l) {
                        self.give(to, values);
                    }
                }
                Some(End::Continue(l, values)) => {
                    if let Some((_, to)) = self.label(*l) {
                        self.give(to, values);
                    }
                }
                _ => {}
            }
        }
    }
    let mut walk = Walk {
        out: BTreeMap::new(),
        labels: Vec::new(),
    };
    match &f.body {
        Body::Regions(r) => walk.region(r, &[]),
        Body::Cfg(blocks) => {
            for b in blocks {
                for t in b.term.targets() {
                    if let Some(to) = blocks.get(t.block as usize) {
                        walk.give(&to.params, &t.args);
                    }
                }
            }
        }
    }
    walk.out
}

/// The output lanes, by D3D register and lane, whose stored values depend on an amplification through any operation
/// (lane by lane or not) and through constructs' parameters and results, and the source of one found: each source in
/// turn spreads to every value computed from its amplifications that no earlier source reached.
pub fn downstream(p: &Program) -> BTreeMap<(u32, usize), Source> {
    let f = &p.entry;
    let sources = param_sources(f);
    let mut users: BTreeMap<ValueId, Vec<ValueId>> = BTreeMap::new();
    let mut seeds: BTreeMap<Source, Vec<ValueId>> = BTreeMap::new();
    for (i, def) in f.values.iter().enumerate() {
        let v = ValueId(u32::try_from(i).unwrap());
        let from = sources.get(&v).into_iter().flatten().copied();
        for o in def.op.operands().into_iter().chain(from) {
            users.entry(o).or_default().push(v);
        }
        if let Some(a) = (0..usize::from(def.ty.lanes)).find_map(|k| amplification(f, v, k)) {
            seeds.entry(a.source).or_default().push(v);
        }
    }
    let mut reached: BTreeMap<ValueId, Source> = BTreeMap::new();
    for (source, seeds) in seeds {
        let mut queue: VecDeque<ValueId> = seeds.into_iter().collect();
        while let Some(v) = queue.pop_front() {
            if reached.contains_key(&v) {
                continue;
            }
            reached.insert(v, source);
            queue.extend(users.get(&v).into_iter().flatten().copied());
        }
    }
    stores(p)
        .into_iter()
        .filter_map(|(at, v, _)| Some((at, *reached.get(&v)?)))
        .collect()
}
