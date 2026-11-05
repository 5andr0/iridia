//! The validator: types of every operation, scoping of values in regions and dominance in a graph, exits against
//! their constructs, interface and resource references against the interface, and stage rules.

use alloc::{format, string::String, vec, vec::Vec};
use core::fmt;

use crate::{
    Binary, Block, Body, End, Function, LabelId, Level, Node, Op, Program, Region, Scalar, Terminator, Type,
    Unary, ValueId,
    interface::{Interface, ResourceKind, ScalarType, Stage, TextureDim, Value},
};

/// One broken rule, with the value it is about where there is one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// The value the rule is about.
    pub value: Option<ValueId>,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value {
            Some(v) => write!(f, "%{}: {}", v.0, self.message),
            None => f.write_str(&self.message),
        }
    }
}

/// The IR type of an interface value.
pub fn value_type(v: Value) -> Type {
    let scalar = match v.scalar {
        ScalarType::Bool => Scalar::Bool,
        ScalarType::Sint => Scalar::I32,
        ScalarType::Uint => Scalar::U32,
        ScalarType::Float => Scalar::F32,
        ScalarType::Double => Scalar::F64,
    };
    Type::vector(scalar, v.width)
}

fn same(a: Type, b: Type, what: &str) -> Result<(), String> {
    if a == b { Ok(()) } else { Err(format!("{what}: {a} and {b} differ")) }
}

/// The type an operation gives from its operands' types (`ty` looks them up), or `None` where the operation's type is
/// stated rather than derived (literals, parameters, conversions, bitcasts, loads, samples). An error names the rule
/// the operands break.
pub fn derived_type(op: &Op, ty: impl Fn(ValueId) -> Option<Type>) -> Result<Option<Type>, String> {
    let get = |v: ValueId| ty(v).ok_or_else(|| format!("operand %{} is not a value of the function", v.0));
    let float = |t: Type, what: &str| {
        if t.scalar.is_float() { Ok(()) } else { Err(format!("{what} of {t}, which is not a float")) }
    };
    Ok(Some(match op {
        Op::Literal(_) | Op::Undef | Op::Param | Op::Convert(..) | Op::Bitcast(_) | Op::LoadInput(_)
        | Op::LoadVar(_) | Op::LoadBuffer { .. } | Op::Sample { .. } => return Ok(None),
        Op::Unary(u, a) => {
            let t = get(*a)?;
            match u {
                Unary::Not => {
                    if t.scalar != Scalar::Bool && !t.scalar.is_integer() {
                        return Err(format!("not of {t}"));
                    }
                    t
                }
                Unary::INeg => {
                    if !t.scalar.is_integer() {
                        return Err(format!("integer negation of {t}"));
                    }
                    t
                }
                Unary::Any | Unary::All => {
                    if t.scalar != Scalar::Bool {
                        return Err(format!("any/all of {t}"));
                    }
                    Type::BOOL
                }
                Unary::IsNan => {
                    float(t, "isnan")?;
                    t.with(Scalar::Bool)
                }
                _ => {
                    float(t, "a float operation")?;
                    t
                }
            }
        }
        Op::Binary(b, x, y) => {
            let (a, c) = (get(*x)?, get(*y)?);
            same(a, c, "binary operands")?;
            match b {
                Binary::FAdd | Binary::FSub | Binary::FMul(_) | Binary::FDiv | Binary::FMin(_) | Binary::FMax(_)
                | Binary::FPow => {
                    float(a, "a float operation")?;
                    a
                }
                Binary::FCmp(_) => {
                    float(a, "a float comparison")?;
                    a.with(Scalar::Bool)
                }
                Binary::SCmp(_) | Binary::UCmp(_) => {
                    if a.scalar.is_float() {
                        return Err(format!("an integer comparison of {a}"));
                    }
                    a.with(Scalar::Bool)
                }
                Binary::IAdd | Binary::ISub | Binary::IMul | Binary::UDiv(_) | Binary::URem(_) | Binary::SMin
                | Binary::SMax | Binary::Xor => {
                    if !a.scalar.is_integer() {
                        return Err(format!("an integer operation of {a}"));
                    }
                    a
                }
                Binary::And | Binary::Or => {
                    if a.scalar.is_float() {
                        return Err(format!("a bitwise operation of {a}"));
                    }
                    a
                }
            }
        }
        Op::Select { cond, accept, reject } => {
            let (c, a, r) = (get(*cond)?, get(*accept)?, get(*reject)?);
            same(a, r, "select arms")?;
            if c.scalar != Scalar::Bool || (c.lanes != 1 && c.lanes != a.lanes) {
                return Err(format!("select condition {c} for {a}"));
            }
            a
        }
        Op::Compose(parts) => {
            let mut scalar = None;
            let mut lanes: u32 = 0;
            for p in parts {
                let t = get(*p)?;
                if *scalar.get_or_insert(t.scalar) != t.scalar {
                    return Err("compose parts of different scalar types".into());
                }
                lanes = lanes.saturating_add(u32::from(t.lanes));
            }
            let lanes = u8::try_from(lanes).ok().filter(|n| (2..=4).contains(n));
            match (scalar, lanes) {
                (Some(s), Some(n)) => Type::vector(s, n),
                _ => return Err("compose of other than 2 to 4 lanes".into()),
            }
        }
        Op::Swizzle(a, lanes) => {
            let t = get(*a)?;
            if lanes.iter().any(|&l| l >= t.lanes) {
                return Err(format!("swizzle {lanes:?} of {t}"));
            }
            return Ok(None);
        }
        Op::Splat(a) => {
            let t = get(*a)?;
            if t.lanes != 1 {
                return Err(format!("splat of {t}"));
            }
            return Ok(None);
        }
        Op::Dot(x, y) => {
            let (a, b) = (get(*x)?, get(*y)?);
            same(a, b, "dot operands")?;
            float(a, "a dot product")?;
            if a.lanes < 2 {
                return Err("a dot product of scalars".into());
            }
            Type::scalar(a.scalar)
        }
    }))
}

/// Check an operation whose type is stated: the operands' types and the stated type agree with the rule.
fn stated(op: &Op, t: Type, f: &Function, interface: &Interface) -> Result<(), String> {
    let get = |v: ValueId| f.ty(v).ok_or_else(|| format!("operand %{} is not a value of the function", v.0));
    match op {
        Op::Literal(_) | Op::Undef | Op::Param => Ok(()),
        Op::Convert(a, _) => {
            let a = get(*a)?;
            if a.lanes != t.lanes || a.scalar == Scalar::Bool || t.scalar == Scalar::Bool {
                return Err(format!("conversion of {a} to {t}"));
            }
            Ok(())
        }
        Op::Bitcast(a) => {
            let a = get(*a)?;
            let bits = |t: Type| t.scalar.bits().saturating_mul(u32::from(t.lanes));
            if bits(a) != bits(t) || a.scalar == Scalar::Bool || t.scalar == Scalar::Bool {
                return Err(format!("bitcast of {a} to {t}"));
            }
            Ok(())
        }
        Op::Swizzle(a, _) => {
            let a = get(*a)?;
            if a.scalar == t.scalar { Ok(()) } else { Err(format!("swizzle of {a} to {t}")) }
        }
        Op::Splat(a) => {
            if get(*a)? == Type::scalar(t.scalar) && t.lanes >= 2 {
                Ok(())
            } else {
                Err(format!("splat of {} to {t}", get(*a)?))
            }
        }
        Op::LoadInput(slot) => {
            let v = interface
                .inputs
                .get(*slot as usize)
                .ok_or_else(|| format!("input slot {slot}, where the interface has {}", interface.inputs.len()))?;
            same(value_type(v.value), t, "input")
        }
        Op::LoadVar(var) => {
            let v = f.vars.get(var.0 as usize).ok_or_else(|| format!("variable {} does not exist", var.0))?;
            same(v.ty, t, "variable load")
        }
        Op::LoadBuffer { resource, index, .. } => {
            let r = interface
                .resources
                .iter()
                .find(|r| r.id == *resource)
                .ok_or_else(|| format!("resource {} is not in the interface", resource.0))?;
            if !matches!(r.kind, ResourceKind::ConstantBuffer { .. }) {
                return Err(format!("buffer load from {:?}", r.kind));
            }
            if let Some(i) = index {
                let i = get(*i)?;
                if i != Type::U32 && i != Type::I32 {
                    return Err(format!("buffer index of {i}"));
                }
            }
            if t.scalar.bits() != 32 || t.scalar == Scalar::Bool {
                return Err(format!("buffer load of {t}"));
            }
            Ok(())
        }
        Op::Sample { texture, sampler, coord, level } => {
            let kind = |id| {
                interface
                    .resources
                    .iter()
                    .find(|r| r.id == id)
                    .map(|r| &r.kind)
                    .ok_or_else(|| format!("resource {} is not in the interface", id.0))
            };
            let ResourceKind::Texture { dim, arrayed, .. } = kind(*texture)? else {
                return Err("sample of a resource that is not a texture".into());
            };
            if !matches!(kind(*sampler)?, ResourceKind::Sampler { .. }) {
                return Err("sample through a resource that is not a sampler".into());
            }
            let lanes: u8 = match (dim, arrayed) {
                (TextureDim::D1, false) => 1,
                (TextureDim::D1, true) | (TextureDim::D2, false) => 2,
                (TextureDim::D2, true) | (TextureDim::D3 | TextureDim::Cube, false) => 3,
                (TextureDim::D3 | TextureDim::Cube, true) => 4,
            };
            same(get(*coord)?, Type::vector(Scalar::F32, lanes), "sample coordinate")?;
            if let Level::Bias(x) | Level::Exact(x) = level {
                same(get(*x)?, Type::F32, "sample level")?;
            }
            same(t, Type::F32X4, "sample result")
        }
        _ => Ok(()),
    }
}

struct Checker<'a> {
    f: &'a Function,
    interface: &'a Interface,
    stage: Stage,
    visible: Vec<bool>,
    placed: Vec<bool>,
    labels: Vec<(LabelId, Vec<ValueId>, Option<Vec<ValueId>>)>,
    problems: Vec<Problem>,
}

impl Checker<'_> {
    fn problem(&mut self, value: Option<ValueId>, message: impl Into<String>) {
        self.problems.push(Problem { value, message: message.into() });
    }
    fn ty(&self, v: ValueId) -> Option<Type> {
        self.f.ty(v)
    }
    fn uses(&mut self, at: Option<ValueId>, v: ValueId) {
        if !self.visible.get(v.0 as usize).copied().unwrap_or(false) {
            self.problem(at, format!("uses %{}, which is not defined before it in an enclosing region", v.0));
        }
    }
    fn define(&mut self, v: ValueId, param: bool) {
        let Some(def) = self.f.value(v) else {
            self.problem(Some(v), "is not a value of the function");
            return;
        };
        if (def.op == Op::Param) != param {
            self.problem(Some(v), if param { "a construct value that is not a parameter" } else { "a parameter placed as a node" });
        }
        match self.placed.get_mut(v.0 as usize) {
            Some(p) if *p => self.problem(Some(v), "is defined twice"),
            Some(p) => *p = true,
            None => {}
        }
        if let Some(s) = self.visible.get_mut(v.0 as usize) {
            *s = true;
        }
    }
    fn hide(&mut self, from: &[bool]) {
        // what a region defined is not visible after it: restore the visibility before it, keeping the slots
        for (v, &before) in self.visible.iter_mut().zip(from) {
            *v = before;
        }
    }
    fn op(&mut self, v: ValueId) {
        let Some(def) = self.f.value(v) else { return };
        for o in def.op.operands() {
            self.uses(Some(v), o);
        }
        if !def.ty.is_valid() {
            self.problem(Some(v), format!("type {} has no lanes count of 1 to 4", def.ty));
        }
        match derived_type(&def.op, |o| self.ty(o)) {
            Ok(Some(t)) if t != def.ty => self.problem(Some(v), format!("states {} where its operation gives {t}", def.ty)),
            Ok(_) => {}
            Err(e) => self.problem(Some(v), e),
        }
        if let Err(e) = stated(&def.op, def.ty, self.f, self.interface) {
            self.problem(Some(v), e);
        }
        if matches!(def.op, Op::Sample { level: Level::Auto | Level::Bias(_), .. }) && self.stage != Stage::Pixel {
            self.problem(Some(v), "an implicit-level sample outside a pixel program");
        }
    }
    fn node(&mut self, n: &Node) {
        match n {
            Node::Value(v) => {
                self.op(*v);
                self.define(*v, false);
            }
            Node::StoreVar(var, value) => {
                self.uses(None, *value);
                match (self.f.vars.get(var.0 as usize), self.ty(*value)) {
                    (Some(d), Some(t)) if d.ty == t => {}
                    (Some(d), Some(t)) => {
                        let m = format!("stores {t} into variable {} of {}", d.name, d.ty);
                        self.problem(Some(*value), m);
                    }
                    _ => self.problem(Some(*value), format!("store to variable {}, which does not exist", var.0)),
                }
            }
            Node::StoreOutput(slot, value) => {
                self.uses(None, *value);
                match self.interface.outputs.get(*slot as usize) {
                    Some(o) if Some(value_type(o.value)) == self.ty(*value) => {}
                    Some(o) => {
                        let m = format!("output {slot} of {} takes {:?}", value_type(o.value), self.ty(*value));
                        self.problem(Some(*value), m);
                    }
                    None => self.problem(Some(*value), format!("output slot {slot} is not in the interface")),
                }
            }
            Node::If { cond, accept, reject, results } => {
                self.uses(None, *cond);
                if self.ty(*cond) != Some(Type::BOOL) {
                    self.problem(Some(*cond), "an if condition that is not a bool");
                }
                let before = self.visible.clone();
                self.region(accept, Some(results));
                self.hide(&before);
                self.region(reject, Some(results));
                self.hide(&before);
                for r in results {
                    self.define(*r, true);
                }
            }
            Node::Block { label, body, results } => {
                let before = self.visible.clone();
                self.labels.push((*label, results.clone(), None));
                self.region(body, Some(results));
                self.labels.pop();
                self.hide(&before);
                for r in results {
                    self.define(*r, true);
                }
            }
            Node::Loop { label, params, inits, body, results } => {
                if params.len() != inits.len() {
                    self.problem(None, "a loop with as many initial values as parameters");
                }
                for (p, i) in params.iter().zip(inits) {
                    self.uses(Some(*p), *i);
                    if self.ty(*p) != self.ty(*i) {
                        self.problem(Some(*p), "a loop parameter of another type than its initial value");
                    }
                }
                let before = self.visible.clone();
                for p in params {
                    self.define(*p, true);
                }
                self.labels.push((*label, results.clone(), Some(params.clone())));
                self.region(body, Some(results));
                self.labels.pop();
                self.hide(&before);
                for r in results {
                    self.define(*r, true);
                }
            }
        }
    }
    fn args(&mut self, what: &str, given: &[ValueId], wanted: &[ValueId]) {
        for g in given {
            self.uses(None, *g);
        }
        let types = |c: &Self, vs: &[ValueId]| vs.iter().map(|v| c.ty(*v)).collect::<Vec<_>>();
        if types(self, given) != types(self, wanted) {
            self.problem(None, format!("{what} gives {} values that do not match the {} it fills", given.len(), wanted.len()));
        }
    }
    fn region(&mut self, r: &Region, results: Option<&Vec<ValueId>>) {
        for n in &r.nodes {
            self.node(n);
        }
        match &r.end {
            None => self.problem(None, "a region with no end"),
            Some(End::Yield(vs)) => {
                let wanted = results.cloned().unwrap_or_default();
                self.args("a yield", vs, &wanted);
            }
            Some(End::Br(l, vs)) => match self.labels.iter().rev().find(|(x, ..)| x == l).cloned() {
                Some((_, results, _)) => self.args("a break", vs, &results),
                None => self.problem(None, format!("a break to label {}, which no enclosing construct has", l.0)),
            },
            Some(End::Continue(l, vs)) => match self.labels.iter().rev().find(|(x, ..)| x == l).cloned() {
                Some((_, _, Some(params))) => self.args("a continue", vs, &params),
                Some(_) => self.problem(None, format!("a continue to label {}, which is not a loop", l.0)),
                None => self.problem(None, format!("a continue to label {}, which no enclosing construct has", l.0)),
            },
            Some(End::Kill) if self.stage != Stage::Pixel => self.problem(None, "a kill outside a pixel program"),
            Some(End::Return | End::Kill | End::Unreachable) => {}
        }
    }
    fn cfg(&mut self, blocks: &[Block]) {
        let n = blocks.len();
        if n == 0 {
            self.problem(None, "a graph with no blocks");
            return;
        }
        let succ = |b: &Block| -> Vec<u32> {
            match &b.term {
                Terminator::Jump(t) => vec![t.block],
                Terminator::Branch { accept, reject, .. } => vec![accept.block, reject.block],
                _ => Vec::new(),
            }
        };
        let idom = dominators(blocks.iter().map(succ).collect());
        // a value is visible in a block when its own block dominates it (earlier in the same block)
        let mut home = vec![None; self.f.values.len()];
        for (i, b) in blocks.iter().enumerate() {
            for v in b.params.iter().chain(b.nodes.iter().filter_map(|n| match n {
                Node::Value(v) => Some(v),
                _ => None,
            })) {
                if let Some(h) = home.get_mut(v.0 as usize) {
                    *h = Some(i);
                }
            }
        }
        let dominates = |a: usize, mut b: usize| loop {
            if a == b {
                return true;
            }
            match idom.get(b).copied().flatten() {
                Some(p) if p != b => b = p,
                _ => return false,
            }
        };
        for (i, b) in blocks.iter().enumerate() {
            let mut local = Vec::new();
            let seen = |c: &mut Self, local: &Vec<ValueId>, at: Option<ValueId>, v: ValueId| {
                let ok = local.contains(&v)
                    || home.get(v.0 as usize).copied().flatten().is_some_and(|h| h != i && dominates(h, i));
                if !ok {
                    c.problem(at, format!("uses %{}, which does not dominate it", v.0));
                }
            };
            for p in &b.params {
                self.define(*p, true);
                local.push(*p);
            }
            for node in &b.nodes {
                match node {
                    Node::Value(v) => {
                        if let Some(def) = self.f.value(*v) {
                            for o in def.op.operands() {
                                seen(self, &local, Some(*v), o);
                            }
                        }
                        // operand visibility was checked above; mark them so `op` checks only the rules
                        let saved = self.visible.clone();
                        self.visible.iter_mut().for_each(|s| *s = true);
                        self.op(*v);
                        self.visible = saved;
                        self.define(*v, false);
                        local.push(*v);
                    }
                    Node::StoreVar(_, x) | Node::StoreOutput(_, x) => {
                        seen(self, &local, None, *x);
                        let saved = self.visible.clone();
                        self.visible.iter_mut().for_each(|s| *s = true);
                        self.node(node);
                        self.visible = saved;
                    }
                    _ => self.problem(None, "a construct inside a graph block"),
                }
            }
            let targets: Vec<&crate::Target> = match &b.term {
                Terminator::Jump(t) => vec![t],
                Terminator::Branch { cond, accept, reject } => {
                    seen(self, &local, None, *cond);
                    if self.ty(*cond) != Some(Type::BOOL) {
                        self.problem(Some(*cond), "a branch condition that is not a bool");
                    }
                    vec![accept, reject]
                }
                Terminator::Kill if self.stage != Stage::Pixel => {
                    self.problem(None, "a kill outside a pixel program");
                    Vec::new()
                }
                _ => Vec::new(),
            };
            for t in targets {
                match blocks.get(t.block as usize) {
                    None => self.problem(None, format!("a branch to block {}, of {n}", t.block)),
                    Some(to) => {
                        for a in &t.args {
                            seen(self, &local, None, *a);
                        }
                        let types = |vs: &[ValueId]| vs.iter().map(|v| self.ty(*v)).collect::<Vec<_>>();
                        if types(&t.args) != types(&to.params) {
                            self.problem(None, format!("a branch to block {} whose arguments do not match its parameters", t.block));
                        }
                    }
                }
            }
        }
    }
}

/// Immediate dominators of a graph given by successor lists, block 0 the entry; `None` for unreachable blocks (and
/// the entry is its own).
pub fn dominators(succ: Vec<Vec<u32>>) -> Vec<Option<usize>> {
    let n = succ.len();
    // reverse post-order
    let mut order = Vec::new();
    let mut seen = vec![false; n];
    let mut stack = vec![(0usize, 0usize)];
    if let Some(s) = seen.get_mut(0) {
        *s = true;
    }
    while let Some((b, i)) = stack.pop() {
        let next = succ.get(b).and_then(|s| s.get(i)).map(|&x| x as usize);
        match next {
            Some(x) => {
                stack.push((b, i.saturating_add(1)));
                if x < n && !seen.get(x).copied().unwrap_or(true) {
                    if let Some(s) = seen.get_mut(x) {
                        *s = true;
                    }
                    stack.push((x, 0));
                }
            }
            None => order.push(b),
        }
    }
    order.reverse();
    let mut rpo = vec![usize::MAX; n];
    for (i, &b) in order.iter().enumerate() {
        if let Some(r) = rpo.get_mut(b) {
            *r = i;
        }
    }
    let mut preds = vec![Vec::new(); n];
    for (b, s) in succ.iter().enumerate() {
        for &x in s {
            if let Some(p) = preds.get_mut(x as usize) {
                p.push(b);
            }
        }
    }
    let mut idom: Vec<Option<usize>> = vec![None; n];
    if let Some(e) = idom.get_mut(0) {
        *e = Some(0);
    }
    let intersect = |idom: &Vec<Option<usize>>, mut a: usize, mut b: usize| {
        let r = |x: usize| rpo.get(x).copied().unwrap_or(usize::MAX);
        while a != b {
            while r(a) > r(b) {
                a = idom.get(a).copied().flatten().unwrap_or(0);
            }
            while r(b) > r(a) {
                b = idom.get(b).copied().flatten().unwrap_or(0);
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().skip(1) {
            let mut new = None;
            for &p in preds.get(b).map(Vec::as_slice).unwrap_or(&[]) {
                if idom.get(p).copied().flatten().is_some() {
                    new = Some(match new {
                        None => p,
                        Some(q) => intersect(&idom, p, q),
                    });
                }
            }
            if new.is_some() && idom.get(b).copied().flatten() != new {
                if let Some(slot) = idom.get_mut(b) {
                    *slot = new;
                }
                changed = true;
            }
        }
    }
    idom
}

/// Check a program: every problem found.
pub fn validate(p: &Program) -> Result<(), Vec<Problem>> {
    let f = &p.entry;
    let mut c = Checker {
        f,
        interface: &p.interface,
        stage: p.stage,
        visible: vec![false; f.values.len()],
        placed: vec![false; f.values.len()],
        labels: Vec::new(),
        problems: Vec::new(),
    };
    if p.stage != p.interface.stage {
        c.problem(None, "the program's stage is not its interface's");
    }
    match &f.body {
        Body::Regions(r) => {
            c.region(r, None);
            if !matches!(r.end, Some(End::Return | End::Kill | End::Unreachable | End::Yield(_))) {
                c.problem(None, "the body ends in an exit to a construct it is not in");
            }
        }
        Body::Cfg(blocks) => c.cfg(blocks),
    }
    if c.problems.is_empty() { Ok(()) } else { Err(c.problems) }
}
