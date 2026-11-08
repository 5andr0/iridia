//! Control flow: the regions of a function as WGSL statements. A construct's results and a loop's parameters are
//! variables (`p3`) that each exit assigns and the code after the construct reads once, as a value (`let v3 = p3;`).
//!
//! WGSL's `break` and `continue` reach only the innermost loop or switch: run `iridia_passes::break_flags` first.

use alloc::{format, string::String, vec::Vec};

use iridia_ir::{End, Function, LabelId, Memory, Node, Op, Region, Scalar, Type, ValueId};

use crate::emit::{R, Writer};
use crate::text::{self, err};

/// A construct being written: the label exits name, and where they put their values.
#[derive(Clone, Debug)]
pub(crate) struct Construct {
    label: LabelId,
    /// The variables a yield or a break stores the construct's results in.
    results: Vec<String>,
    /// The variables a continue stores the loop's parameters in; `None` for a construct that is not a loop.
    params: Option<Vec<String>>,
    kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Block,
    Loop,
    Switch,
}

/// What a region's yield does.
#[derive(Clone, Copy)]
pub(crate) enum Yield<'a> {
    /// The function's body: a yield of nothing falls off its end.
    Top,
    /// An `If` arm or a `Switch` case: store the results and fall through.
    Fall(&'a [String]),
    /// A loop body: store the results and leave the loop.
    Break(&'a [String]),
    /// An `If` arm: record the scalar of each lane of what it yields, for the `If` to give its results.
    Gather,
}

/// The variable of lane `k` of the result `r` of `lanes` lanes.
fn lane_var(r: ValueId, lanes: u8, k: usize) -> String {
    if lanes == 1 {
        format!("p{}", r.0)
    } else {
        format!("p{}_{k}", r.0)
    }
}

/// Whether the scalar `s` (a name, a name's lane, or a literal) means the same after the arms `texts` as before them: a
/// literal does, and a name does unless an arm declares it.
fn outer_scalar(s: &str, texts: &[&str]) -> bool {
    let base = s.split('.').next().unwrap_or(s);
    let name = !base.is_empty() && base.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if name && !base.starts_with(|c: char| c.is_ascii_digit()) {
        return !texts
            .iter()
            .any(|t| t.contains(&format!("let {base} ")) || t.contains(&format!("var {base}")));
    }
    // a literal: `0u`, `(-1i)`, `bitcast<f32>(5u)`, `true`
    s.starts_with(|c: char| c.is_ascii_digit())
        || s.starts_with("(-")
        || s == "true"
        || s == "false"
        || s.starts_with("bitcast<f32>(")
        || s.starts_with("iridia_f32_from_bits(")
        || s.starts_with("i32(-")
}

impl Writer<'_> {
    fn construct(&self, l: LabelId) -> R<(usize, &Construct)> {
        self.constructs
            .iter()
            .enumerate()
            .rev()
            .find(|(_, c)| c.label == l)
            .ok_or_else(|| {
                err(format!(
                    "an exit to label {}, which no enclosing construct has",
                    l.0
                ))
            })
    }

    /// A break to construct `at` is a WGSL `break` only when it is the innermost loop or switch; a continue, when it is the
    /// innermost loop (a switch between does not stop it).
    fn innermost(&self, at: usize, continues: bool) -> R<()> {
        let last = self
            .constructs
            .iter()
            .rposition(|c| c.kind == Kind::Loop || (!continues && c.kind == Kind::Switch));
        if last == Some(at) {
            Ok(())
        } else {
            Err(err(
                "an exit past an enclosing construct: run iridia_passes::break_flags first",
            ))
        }
    }

    /// A variable for each value, of the value's type, and its name.
    fn storage_for(&mut self, vs: &[ValueId]) -> R<Vec<String>> {
        self.storage_for_with(vs, &[])
    }

    /// The scalar each lane of `vs` is where it is the same as a value bound before the construct (`None` where a variable
    /// holds it): the lanes a loop leaves unchanged, and the lanes every exit of a loop gives the same value.
    fn forwards_of(&self, vs: &[ValueId]) -> R<Vec<Vec<Option<String>>>> {
        vs.iter()
            .map(|v| {
                let t = self.vty(*v)?;
                (0..t.lanes)
                    .map(|k| {
                        let (x, l) = self.sites.origin(*v, k);
                        if (x, l) == (*v, k) || !self.visible(x) {
                            return Ok(None);
                        }
                        Ok(self.lane_names(x)?.get(usize::from(l)).cloned())
                    })
                    .collect()
            })
            .collect()
    }

    /// [`Writer::storage_for`], with `forwards` (per value, per lane) the lanes that need no variable.
    fn storage_for_with(&mut self, vs: &[ValueId], forwards: &[Vec<Option<String>>]) -> R<Vec<String>> {
        vs.iter()
            .enumerate()
            .map(|(i, v)| {
                let t = self.vty(*v)?;
                let name = format!("p{}", v.0);
                let fwd = forwards.get(i).filter(|f| f.iter().any(Option::is_some)).cloned();
                if let Some(fwd) = fwd {
                    let mut lanes = Vec::new();
                    for (k, f) in fwd.iter().enumerate() {
                        let var = lane_var(*v, t.lanes, k);
                        if f.is_none() {
                            self.line(&format!("var {var}: {};", text::scalar(t.scalar)?));
                        }
                        lanes.push(var);
                    }
                    self.forward.insert(name.clone(), fwd);
                    self.split.insert(name.clone(), lanes);
                } else if t.lanes > 1 {
                    // a variable of a vector is a variable a lane: a compiler's uniformity analysis takes a whole vector
                    // as the least uniform of its lanes, and a loop's counter often shares a register with a thread's
                    // index
                    let lanes: Vec<String> = (0..t.lanes).map(|k| format!("{name}_{k}")).collect();
                    for l in &lanes {
                        self.line(&format!("var {l}: {};", text::scalar(t.scalar)?));
                    }
                    self.split.insert(name.clone(), lanes);
                } else {
                    self.line(&format!("var {name}: {};", text::ty(t)?));
                }
                Ok(name)
            })
            .collect()
    }

    /// Assign `v` to the storage `s`: a lane at a time where the variable is split.
    fn assign(&mut self, s: &str, v: ValueId) -> R<()> {
        if let Some(lanes) = self.split.get(s).cloned() {
            let values = self.lane_names(v)?;
            let fwd = self.forward.get(s).cloned().unwrap_or_default();
            for (k, (l, x)) in lanes.iter().zip(&values).enumerate() {
                if fwd.get(k).is_some_and(Option::is_some) {
                    continue;
                }
                self.line(&format!("{l} = {x};"));
            }
        } else {
            let value = self.get(v)?;
            self.line(&format!("{s} = {value};"));
        }
        Ok(())
    }

    /// The construct's values read the variables they were stored in: `let v3 = p3;`.
    fn load_storage(&mut self, vs: &[ValueId], storage: &[String]) -> R<()> {
        for (v, s) in vs.iter().zip(storage) {
            if let Some(lanes) = self.split.get(s).cloned() {
                // each lane read once into a value of its own, so a lane read later is that scalar
                let t = self.vty(*v)?;
                let fwd = self.forward.get(s).cloned().unwrap_or_default();
                let mut scalars = Vec::new();
                for (k, l) in lanes.iter().enumerate() {
                    let from = fwd.get(k).cloned().flatten().unwrap_or_else(|| l.clone());
                    if t.lanes == 1 {
                        scalars.push(from);
                    } else {
                        let q = format!("q{}_{k}", v.0);
                        self.line(&format!("let {q} = {from};"));
                        scalars.push(q);
                    }
                }
                if t.lanes == 1 {
                    self.define(*v, scalars.first().map_or("", String::as_str))?;
                } else {
                    self.define(*v, &format!("{}({})", text::ty(t)?, scalars.join(", ")))?;
                    self.lanes.insert(v.0, scalars);
                }
            } else {
                self.define(*v, s)?;
            }
        }
        Ok(())
    }

    /// Assign the variables the values: `p3 = v9;`. Every value is read before any variable is written, as a loop's
    /// parameters are all read from the values the iteration started with.
    fn store_all(&mut self, storage: &[String], vs: &[ValueId]) -> R<()> {
        if storage.len() != vs.len() {
            return Err(err(
                "an exit with another number of values than its construct takes",
            ));
        }
        for (s, v) in storage.iter().zip(vs) {
            self.assign(s, *v)?;
        }
        Ok(())
    }

    /// What each result of an `If` is after it, lane by lane: the scalar both arms yield where it is one that was bound
    /// before the `If` (a lane the arms agree on needs no variable, and a variable assigned under a condition that differs
    /// between invocations is, to a compiler's uniformity analysis, as different as the condition, whatever is stored), or
    /// `None` where a variable holds it.
    fn arms_give(
        &self,
        results: &[ValueId],
        from_a: Option<&[Vec<String>]>,
        from_b: Option<&[Vec<String>]>,
        a: &str,
        b: &str,
    ) -> R<Vec<Vec<Option<String>>>> {
        let before = |s: &str| outer_scalar(s, &[a, b]);
        results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let n = usize::from(self.vty(*r)?.lanes);
                Ok((0..n)
                    .map(|k| {
                        let lane = |from: Option<&[Vec<String>]>| {
                            from.and_then(|f| f.get(i)).and_then(|l| l.get(k)).cloned()
                        };
                        // the value both arms come to, where that is one bound before the `If`
                        let (x, l) = self.sites.origin(*r, u8::try_from(k).unwrap_or(0));
                        let bound = self
                            .visible(x)
                            .then(|| {
                                self.lane_names(x)
                                    .ok()
                                    .and_then(|ls| ls.get(usize::from(l)).cloned())
                            })
                            .flatten()
                            .filter(|name| (x, l) != (*r, u8::try_from(k).unwrap_or(0)) && before(name));
                        if bound.is_some() {
                            return bound;
                        }
                        match (lane(from_a), lane(from_b)) {
                            (Some(x), Some(y)) if x == y && before(&x) => Some(x),
                            (Some(x), None) | (None, Some(x)) if before(&x) => Some(x),
                            _ => None,
                        }
                    })
                    .collect())
            })
            .collect()
    }

    /// Declare the variables of the lanes that no scalar is given for, and give for each arm the assignments that end it.
    fn assign_arms(
        &mut self,
        results: &[ValueId],
        given: &[Vec<Option<String>>],
        from_a: Option<&[Vec<String>]>,
        from_b: Option<&[Vec<String>]>,
    ) -> R<(String, String)> {
        let mut for_a = Vec::new();
        let mut for_b = Vec::new();
        for (i, r) in results.iter().enumerate() {
            let t = self.vty(*r)?;
            for (k, g) in given.get(i).into_iter().flatten().enumerate() {
                if g.is_some() {
                    continue;
                }
                let var = lane_var(*r, t.lanes, k);
                self.line(&format!("var {var}: {};", text::scalar(t.scalar)?));
                let from =
                    |f: Option<&[Vec<String>]>| f.and_then(|f| f.get(i)).and_then(|l| l.get(k)).cloned();
                if let Some(x) = from(from_a) {
                    for_a.push(format!("{var} = {x};"));
                }
                if let Some(x) = from(from_b) {
                    for_b.push(format!("{var} = {x};"));
                }
            }
        }
        let text = |lines: Vec<String>, w: &mut Self| {
            w.capture(|w| {
                for l in &lines {
                    w.line(l);
                }
                Ok(())
            })
        };
        let a = text(for_a, self)?;
        let b = text(for_b, self)?;
        Ok((a, b))
    }

    /// The results of an `If`: each lane read once into a value of its own, from the scalar both arms gave or from its
    /// variable.
    fn define_results(&mut self, results: &[ValueId], given: &[Vec<Option<String>>]) -> R<()> {
        for (i, r) in results.iter().enumerate() {
            let t = self.vty(*r)?;
            let mut scalars = Vec::new();
            for (k, g) in given.get(i).into_iter().flatten().enumerate() {
                let from = g.clone().unwrap_or_else(|| lane_var(*r, t.lanes, k));
                if t.lanes == 1 {
                    scalars.push(from);
                } else {
                    let q = format!("q{}_{k}", r.0);
                    self.line(&format!("let {q} = {from};"));
                    scalars.push(q);
                }
            }
            if t.lanes == 1 {
                let only = scalars.first().cloned().unwrap_or_default();
                self.define(*r, &only)?;
            } else {
                self.define(*r, &format!("{}({})", text::ty(t)?, scalars.join(", ")))?;
                self.lanes.insert(r.0, scalars);
            }
        }
        Ok(())
    }

    pub(crate) fn region(&mut self, r: &Region, yields: Yield<'_>) -> R<()> {
        for n in &r.nodes {
            self.node(n)?;
        }
        self.finish(&r.end, yields)
    }

    /// How a region ends.
    fn finish(&mut self, end: &Option<End>, yields: Yield<'_>) -> R<()> {
        match end {
            None => return Err(err("a region with no end")),
            Some(End::Yield(vs)) => match yields {
                Yield::Top if vs.is_empty() => {}
                Yield::Top => return Err(err("a yield of values at the end of the function")),
                Yield::Fall(storage) => self.store_all(storage, vs)?,
                Yield::Gather => {
                    let lanes = vs.iter().map(|v| self.lane_names(*v)).collect::<R<Vec<_>>>()?;
                    self.gathered = Some(lanes);
                }
                Yield::Break(storage) => {
                    self.store_all(storage, vs)?;
                    self.line("break;");
                }
            },
            Some(End::Br(l, vs)) => {
                let (at, c) = self.construct(*l)?;
                let (storage, kind) = (c.results.clone(), c.kind);
                if kind == Kind::Block {
                    return Err(err(
                        "a break out of a block: run iridia_passes::break_flags first",
                    ));
                }
                self.store_all(&storage, vs)?;
                self.innermost(at, false)?;
                self.line("break;");
            }
            Some(End::Continue(l, vs)) => {
                let (at, c) = self.construct(*l)?;
                let storage = c
                    .params
                    .clone()
                    .ok_or_else(|| err("a continue to a construct that is not a loop"))?;
                self.store_all(&storage, vs)?;
                self.innermost(at, true)?;
                self.line("continue;");
            }
            Some(End::Return) => self.ret()?,
            // WGSL's `discard` demotes the invocation and goes on: the return ends it
            Some(End::Kill) => {
                self.line("discard;");
                self.ret()?;
            }
            Some(End::IgnoreHit | End::AcceptHitAndEndSearch) => {
                return Err(err("an any-hit program's decision is not written yet"));
            }
            // a function that returns a value ends in a return: the function's unreachable end returns its outputs, which
            // no invocation reaches
            Some(End::Unreachable) if matches!(yields, Yield::Top) => self.ret()?,
            Some(End::Unreachable) => {}
        }
        Ok(())
    }

    pub(crate) fn node(&mut self, n: &Node) -> R<()> {
        match n {
            // a storage image's texel atomic whose old value nothing reads: naga's image atomic, which gives none
            Node::Value(v) => match self.f.value(*v).map(|d| &d.op) {
                Some(&Op::Atomic {
                    memory: Memory::Image(r, coord),
                    op,
                    value,
                    compare: None,
                }) if !self.typed_buffer(r.id)? && !self.read.get(v.0 as usize).copied().unwrap_or(true) => {
                    self.image_atomic(r, coord, op, value)?;
                }
                _ => self.value(*v)?,
            },
            // WGSL's `discard` is a demotion: the invocation goes on as a helper
            Node::Demote => self.line("discard;"),
            Node::StoreVar(var, x) => self.store_var(*var, *x)?,
            Node::StoreOutput(slot, mask, x) => self.store_output(*slot, *mask, *x)?,
            Node::StoreElement(var, index, x, oob) => self.store_element_of_var(*var, *index, *x, *oob)?,
            Node::StoreBuffer {
                resource,
                address,
                value,
            } => {
                if !self.words_buffer(resource.id)? {
                    return Err(err(format!("a buffer write of resource {}", resource.id.0)));
                }
                self.store_words(*resource, *address, *value)?;
            }
            Node::StoreTexel { image, coord, value } if self.typed_buffer(image.id)? => {
                self.store_typed_element(*image, *coord, *value)?;
            }
            Node::StoreTexel { image, coord, value } => self.store_texel(*image, *coord, *value)?,
            Node::StorePatch(..) => return Err(err("a patch constant store is not written yet")),
            Node::Barrier(b) => self.barrier(*b),
            Node::Emit(_) | Node::Cut(_) => {
                return Err(err("a geometry program's emit and cut are not written yet"));
            }
            Node::SetMeshCounts { .. } | Node::StoreVertex { .. } | Node::StoreIndices { .. } => {
                return Err(err("a mesh program's meshlet is not written yet"));
            }
            Node::DispatchMesh { .. } => {
                return Err(err("an amplification program's mesh launch is not written yet"));
            }
            Node::TraceRay { .. } | Node::CallShader { .. } => {
                return Err(err("a traced ray or a shader call is not written yet"));
            }
            Node::RayQueryInit { .. }
            | Node::RayQueryAbort(_)
            | Node::RayQueryCommitTriangle(_)
            | Node::RayQueryCommitProcedural { .. } => self.ray_query_node(n)?,
            Node::If {
                cond,
                accept,
                reject,
                results,
            } => {
                let condition = self.get(*cond)?;
                self.gathered = None;
                let mut a = self.capture(|w| w.region(accept, Yield::Gather))?;
                let from_a = self.gathered.take();
                let mut b = self.capture(|w| w.region(reject, Yield::Gather))?;
                let from_b = self.gathered.take();
                let given = self.arms_give(results, from_a.as_deref(), from_b.as_deref(), &a, &b)?;
                // the lanes the arms do not agree on are variables, assigned where each arm ends
                let (assign_a, assign_b) =
                    self.assign_arms(results, &given, from_a.as_deref(), from_b.as_deref())?;
                a.push_str(&assign_a);
                b.push_str(&assign_b);
                self.line(&format!("if ({condition}) {{"));
                self.out.push_str(&a);
                if b.is_empty() {
                    self.line("}");
                } else {
                    self.line("} else {");
                    self.out.push_str(&b);
                    self.line("}");
                }
                self.define_results(results, &given)?;
            }
            Node::Switch {
                label,
                selector,
                cases,
                default,
                results,
            } => {
                let sel = self.get(*selector)?;
                let signed = self.vty(*selector)? == Type::I32;
                let storage = self.storage_for(results)?;
                self.constructs.push(Construct {
                    label: *label,
                    results: storage.clone(),
                    params: None,
                    kind: Kind::Switch,
                });
                let mut bodies = Vec::new();
                // the cases are one level in from the `switch`, their bodies one more
                self.depth = self.depth.saturating_add(1);
                for c in cases {
                    let body = self.capture(|w| w.region(&c.body, Yield::Fall(&storage)))?;
                    let values: Vec<String> = c
                        .values
                        .iter()
                        .map(|v| text::literal(if signed { Scalar::I32 } else { Scalar::U32 }, u64::from(*v)))
                        .collect::<Result<_, _>>()?;
                    bodies.push((values.join(", "), body));
                }
                let default_body = self.capture(|w| w.region(default, Yield::Fall(&storage)));
                self.depth = self.depth.saturating_sub(1);
                let default_body = default_body?;
                self.constructs.pop();
                self.line(&format!("switch ({sel}) {{"));
                self.depth = self.depth.saturating_add(1);
                for (values, body) in bodies {
                    self.line(&format!("case {values}: {{"));
                    self.out.push_str(&body);
                    self.line("}");
                }
                self.line("default: {");
                self.out.push_str(&default_body);
                self.line("}");
                self.depth = self.depth.saturating_sub(1);
                self.line("}");
                self.load_storage(results, &storage)?;
            }
            Node::Block { label, body, results } => {
                // a block is left by falling off its end (an exit to it is refused): its nodes are the enclosing region's,
                // and its results the values it yields
                self.constructs.push(Construct {
                    label: *label,
                    results: Vec::new(),
                    params: None,
                    kind: Kind::Block,
                });
                for n in &body.nodes {
                    self.node(n)?;
                }
                self.constructs.pop();
                match &body.end {
                    Some(End::Yield(vs)) if vs.len() == results.len() => {
                        for (r, v) in results.iter().zip(vs) {
                            let value = self.get(*v)?;
                            self.define(*r, &value)?;
                        }
                    }
                    Some(End::Yield(_)) => {
                        return Err(err("a block yields another number of values than it has results"));
                    }
                    // a block that ends in a return, a kill or an exit of an enclosing construct has no results to give
                    other => self.finish(other, Yield::Fall(&[]))?,
                }
            }
            Node::Loop {
                label,
                params,
                inits,
                body,
                results,
            } => {
                // the lanes of the state a `continue` always hands back unchanged, and of the results every exit gives alike,
                // are values from before the loop
                let param_forwards = self.forwards_of(params)?;
                let result_forwards = self.forwards_of(results)?;
                let param_storage = self.storage_for_with(params, &param_forwards)?;
                for (s, v) in param_storage.iter().zip(inits) {
                    self.assign(s, *v)?;
                }
                let storage = self.storage_for_with(results, &result_forwards)?;
                self.constructs.push(Construct {
                    label: *label,
                    results: storage.clone(),
                    params: Some(param_storage.clone()),
                    kind: Kind::Loop,
                });
                let inner = self.capture(|w| {
                    w.load_storage(params, &param_storage)?;
                    w.region(body, Yield::Break(&storage))
                })?;
                self.constructs.pop();
                self.line("loop {");
                self.out.push_str(&inner);
                self.line("}");
                self.load_storage(results, &storage)?;
            }
        }
        Ok(())
    }

    /// The text a return is.
    pub(crate) fn ret(&mut self) -> R<()> {
        if self.members.is_empty() {
            self.line("return;");
            return Ok(());
        }
        let parts: Vec<String> = (0..self.members.len())
            .map(|m| self.member_value(m))
            .collect::<Result<_, _>>()?;
        let value = format!("Out({})", parts.join(", "));
        self.line(&format!("return {value};"));
        Ok(())
    }
}

/// Whether each value of `f`, whose body is `body`, is read where the body places it: by an operation a node places, by a
/// node, or by a region's end (a value no node places is not written, so what it reads is not read).
pub(crate) fn read_values(f: &Function, body: &Region) -> Vec<bool> {
    fn region(f: &Function, r: &Region, read: &mut [bool]) {
        let mut mark = |v: ValueId| {
            if let Some(slot) = read.get_mut(v.0 as usize) {
                *slot = true;
            }
        };
        for n in &r.nodes {
            if let Node::Value(v) = n
                && let Some(d) = f.value(*v)
            {
                d.op.operands().into_iter().for_each(&mut mark);
            }
            n.values().into_iter().for_each(&mut mark);
        }
        if let Some(End::Yield(vs) | End::Br(_, vs) | End::Continue(_, vs)) = &r.end {
            vs.iter().copied().for_each(&mut mark);
        }
        for inner in r.nodes.iter().flat_map(Node::regions) {
            region(f, inner, read);
        }
    }
    let mut read = alloc::vec![false; f.values.len()];
    region(f, body, &mut read);
    read
}
