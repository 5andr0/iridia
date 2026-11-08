//! Route state travels only across boundaries an exit actually crosses. Normal and escaping outcomes share
//! a typed tuple; the tag proves that inactive tuple columns are never observed.
#![allow(
    clippy::indexing_slicing,
    reason = "frame and route indices are allocated by this module"
)]

use super::Frame;
use alloc::{
    collections::{BTreeMap, BTreeSet},
    format,
    string::String,
    vec,
    vec::Vec,
};
use iridia_ir::{Binary, Cmp, End, Function, Node, Op, Precision, Region, Type, ValueDef, ValueId};

fn value(f: &mut Function, ty: Type, op: Op) -> Result<ValueId, String> {
    let id = ValueId(u32::try_from(f.values.len()).map_err(|_| "more than 2^32 values")?);
    f.values.push(ValueDef {
        ty,
        op,
        precision: Precision::Full,
        origin: None,
    });
    Ok(id)
}

struct FramePlan {
    kind: Option<Frame>,
    normal: Vec<ValueId>,
    params: Vec<ValueId>,
    outgoing: BTreeSet<usize>,
    columns: BTreeMap<usize, Vec<usize>>,
    tag: Option<ValueId>,
    full: Vec<ValueId>,
}

impl FramePlan {
    fn new(kind: Option<Frame>, normal: Vec<ValueId>, params: Vec<ValueId>) -> Self {
        Self {
            kind,
            full: normal.clone(),
            normal,
            params,
            outgoing: BTreeSet::new(),
            columns: BTreeMap::new(),
            tag: None,
        }
    }
}

struct Route {
    target: usize,
    resume: bool,
    types: Vec<Type>,
}

struct Plan {
    frames: Vec<FramePlan>,
    routes: Vec<Route>,
    keys: BTreeMap<(usize, bool), usize>,
    exits: usize,
}

impl Plan {
    fn target(&self, stack: &[usize], label: iridia_ir::LabelId) -> Result<usize, String> {
        stack
            .iter()
            .rposition(|&id| self.frames[id].kind.and_then(Frame::label) == Some(label))
            .ok_or_else(|| format!("an exit to label {}, which no enclosing construct has", label.0))
    }

    fn direct(&self, stack: &[usize], at: usize, resume: bool) -> bool {
        let inner = stack.get(at.saturating_add(1)..).unwrap_or_default();
        match (self.frames[stack[at]].kind, resume) {
            (Some(Frame::Loop(_)), true) => !inner
                .iter()
                .any(|&id| matches!(self.frames[id].kind, Some(Frame::Loop(_)))),
            (Some(f), false) if f.breakable() => !inner
                .iter()
                .any(|&id| self.frames[id].kind.is_some_and(Frame::breakable)),
            _ => false,
        }
    }

    fn tail(&self, stack: &[usize], at: usize, resume: bool) -> bool {
        !resume
            && at.saturating_add(1) == stack.len()
            && matches!(self.frames[stack[at]].kind, Some(Frame::Block(_)))
    }

    fn scan(&mut self, f: &Function, r: &Region, stack: &mut Vec<usize>) -> Result<(), String> {
        for node in &r.nodes {
            if let Some(kind) = Frame::of(node) {
                let (normal, params) = match node {
                    Node::Loop { results, params, .. } => (results.clone(), params.clone()),
                    Node::If { results, .. } | Node::Block { results, .. } | Node::Switch { results, .. } => {
                        (results.clone(), Vec::new())
                    }
                    _ => return Err("a construct without results".into()),
                };
                let id = self.frames.len();
                self.frames.push(FramePlan::new(Some(kind), normal, params));
                stack.push(id);
                for child in node.regions() {
                    self.scan(f, child, stack)?;
                }
                stack.pop();
            }
        }
        if let Some(End::Br(label, _) | End::Continue(label, _)) = &r.end {
            let resume = matches!(r.end, Some(End::Continue(..)));
            let at = self.target(stack, *label)?;
            if self.direct(stack, at, resume) || self.tail(stack, at, resume) {
                return Ok(());
            }
            self.exits = self
                .exits
                .checked_add(1)
                .ok_or("more routed exits than the address space")?;
            let target = stack[at];
            let route = if let Some(&id) = self.keys.get(&(target, resume)) {
                id
            } else {
                let id = self.routes.len();
                let args = if resume {
                    &self.frames[target].params
                } else {
                    &self.frames[target].normal
                };
                let types = args
                    .iter()
                    .map(|&v| f.ty(v).ok_or_else(|| format!("%{} is not a value", v.0)))
                    .collect::<Result<_, _>>()?;
                self.routes.push(Route {
                    target,
                    resume,
                    types,
                });
                self.keys.insert((target, resume), id);
                id
            };
            for &id in stack.get(at.saturating_add(1)..).unwrap_or_default() {
                self.frames[id].outgoing.insert(route);
            }
        }
        Ok(())
    }

    fn allocate(&mut self, f: &mut Function) -> Result<(), String> {
        for frame in &mut self.frames {
            if frame.outgoing.is_empty() {
                continue;
            }
            let tag = value(f, Type::U32, Op::Param)?;
            frame.tag = Some(tag);
            frame.full.push(tag);
            // The tag makes normal completion and every route mutually exclusive. Reuse typed fields
            // across these alternatives, keeping the original normal result IDs in their existing positions.
            // Full-precision payloads never borrow a field with a weaker precision permission.
            let mut pools: Vec<(Type, Vec<usize>)> = Vec::new();
            for (index, &v) in frame.normal.iter().enumerate() {
                let def = f.value(v).ok_or("a normal result without a definition")?;
                if def.precision != Precision::Full {
                    continue;
                }
                if let Some((_, slots)) = pools.iter_mut().find(|(ty, _)| *ty == def.ty) {
                    slots.push(index);
                } else {
                    pools.push((def.ty, vec![index]));
                }
            }
            for &route in &frame.outgoing {
                let mut used = vec![0usize; pools.len()];
                let mut fields = Vec::new();
                for &ty in &self.routes[route].types {
                    // Type has a bounded scalar/lane domain; no scan grows with the number of fields.
                    let pool = if let Some(index) = pools.iter().position(|(t, _)| *t == ty) {
                        index
                    } else {
                        let index = pools.len();
                        pools.push((ty, Vec::new()));
                        used.push(0);
                        index
                    };
                    let slot = if let Some(&slot) = pools[pool].1.get(used[pool]) {
                        slot
                    } else {
                        let slot = frame.full.len();
                        frame.full.push(value(f, ty, Op::Param)?);
                        pools[pool].1.push(slot);
                        slot
                    };
                    used[pool] = used[pool].saturating_add(1);
                    fields.push(slot);
                }
                frame.columns.insert(route, fields);
            }
        }
        Ok(())
    }
}

struct Router<'a> {
    f: &'a mut Function,
    plan: Plan,
    next: usize,
    prefix: Vec<Node>,
    constants: BTreeMap<u32, ValueId>,
    undefined: Vec<(Type, ValueId)>,
}

impl Router<'_> {
    fn constant(&mut self, n: u32) -> Result<ValueId, String> {
        if let Some(&v) = self.constants.get(&n) {
            return Ok(v);
        }
        let v = value(self.f, Type::U32, Op::Literal([u64::from(n), 0, 0, 0]))?;
        self.prefix.push(Node::Value(v));
        self.constants.insert(n, v);
        Ok(v)
    }

    fn undef(&mut self, ty: Type) -> Result<ValueId, String> {
        if let Some(&(_, v)) = self.undefined.iter().find(|(t, _)| *t == ty) {
            return Ok(v);
        }
        let v = value(self.f, ty, Op::Undef)?;
        self.prefix.push(Node::Value(v));
        self.undefined.push((ty, v));
        Ok(v)
    }

    fn types(&self, vs: &[ValueId]) -> Result<Vec<Type>, String> {
        vs.iter()
            .map(|&v| self.f.ty(v).ok_or_else(|| format!("%{} is not a value", v.0)))
            .collect()
    }

    fn empty_normal(&mut self, frame: usize) -> Result<Vec<ValueId>, String> {
        let types = self.types(&self.plan.frames[frame].normal)?;
        types.into_iter().map(|ty| self.undef(ty)).collect()
    }

    fn packed(
        &mut self,
        frame: usize,
        mut normal: Vec<ValueId>,
        escape: Option<(usize, Vec<ValueId>)>,
    ) -> Result<Vec<ValueId>, String> {
        if normal.len() != self.plan.frames[frame].normal.len() {
            return Err("a route's normal result arity differs".into());
        }
        if self.plan.frames[frame].outgoing.is_empty() {
            return if escape.is_none() {
                Ok(normal)
            } else {
                Err("a route crossing an unmarked boundary".into())
            };
        }
        let tag = escape.as_ref().map_or(Ok(0), |(id, _)| {
            u32::try_from(id.saturating_add(1)).map_err(|_| "more than 2^32 routes")
        })?;
        normal.push(self.constant(tag)?);
        let extra = self.plan.frames[frame]
            .full
            .get(normal.len()..)
            .unwrap_or_default()
            .to_vec();
        for v in extra {
            let ty = self.f.ty(v).ok_or("a union field without a type")?;
            normal.push(self.undef(ty)?);
        }
        if let Some((route, args)) = escape {
            let fields = self.plan.frames[frame]
                .columns
                .get(&route)
                .ok_or("a route crossing an unmarked boundary")?;
            if fields.len() != args.len() {
                return Err("a route's payload arity differs".into());
            }
            for (&field, arg) in fields.iter().zip(args) {
                normal[field] = arg;
            }
        }
        Ok(normal)
    }

    fn leave(&self, frame: usize, args: Vec<ValueId>) -> Result<End, String> {
        match self.plan.frames[frame].kind {
            Some(Frame::Loop(label)) => Ok(End::Br(label, args)),
            Some(_) => Ok(End::Yield(args)),
            None => Err("a route past the function's body".into()),
        }
    }

    fn test(&mut self, tag: ValueId, cmp: Cmp, n: u32, nodes: &mut Vec<Node>) -> Result<ValueId, String> {
        let n = self.constant(n)?;
        let cond = value(self.f, Type::BOOL, Op::Binary(Binary::UCmp(cmp), tag, n))?;
        nodes.push(Node::Value(cond));
        Ok(cond)
    }

    /// Both arms return this frame's output tuple, or leave control before observing it.
    fn guard(
        &mut self,
        frame: usize,
        cond: ValueId,
        accept: Region,
        reject: Region,
        mut nodes: Vec<Node>,
    ) -> Result<Region, String> {
        let types = self.types(&self.plan.frames[frame].full)?;
        let results = types
            .into_iter()
            .map(|ty| value(self.f, ty, Op::Param))
            .collect::<Result<Vec<_>, _>>()?;
        nodes.push(Node::If {
            cond,
            accept,
            reject,
            results: results.clone(),
        });
        Ok(Region {
            nodes,
            end: Some(End::Yield(results)),
        })
    }

    fn take(&mut self, frame: usize, child: usize, route: usize) -> Result<Region, String> {
        let args = self.plan.frames[child]
            .columns
            .get(&route)
            .ok_or("a route without its payload")?
            .iter()
            .map(|&field| self.plan.frames[child].full[field])
            .collect();
        let end = if self.plan.routes[route].target == frame {
            match (self.plan.frames[frame].kind, self.plan.routes[route].resume) {
                (Some(Frame::Loop(label)), true) => End::Continue(label, args),
                (Some(Frame::Loop(label)), false) => End::Br(label, self.packed(frame, args, None)?),
                (Some(Frame::Block(_) | Frame::Switch(_)), false) => {
                    End::Yield(self.packed(frame, args, None)?)
                }
                _ => return Err("a route arriving at a construct its exit cannot take".into()),
            }
        } else {
            let normal = self.empty_normal(frame)?;
            let args = self.packed(frame, normal, Some((route, args)))?;
            self.leave(frame, args)?
        };
        Ok(Region {
            nodes: Vec::new(),
            end: Some(end),
        })
    }

    fn dispatch(&mut self, frame: usize, child: usize, tag: ValueId) -> Result<Region, String> {
        let routes = self.plan.frames[child]
            .outgoing
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let mut routes = routes.into_iter().rev();
        let last = routes.next().ok_or("an empty route dispatch")?;
        let mut arm = self.take(frame, child, last)?;
        for route in routes {
            let mut nodes = Vec::new();
            let id = u32::try_from(route.saturating_add(1)).map_err(|_| "more than 2^32 routes")?;
            let cond = self.test(tag, Cmp::Eq, id, &mut nodes)?;
            let take = self.take(frame, child, route)?;
            arm = self.guard(frame, cond, take, arm, nodes)?;
        }
        Ok(arm)
    }

    fn region(&mut self, r: Region, stack: &mut Vec<usize>) -> Result<Region, String> {
        self.sequence(r.nodes.into_iter(), r.end, stack)
    }

    fn sequence(
        &mut self,
        mut nodes: alloc::vec::IntoIter<Node>,
        end: Option<End>,
        stack: &mut Vec<usize>,
    ) -> Result<Region, String> {
        let frame = *stack.last().ok_or("a region without its frame")?;
        let mut out = Vec::new();
        while let Some(mut node) = nodes.next() {
            let Some(kind) = Frame::of(&node) else {
                out.push(node);
                continue;
            };
            let child = self.next;
            self.next = self
                .next
                .checked_add(1)
                .ok_or("more constructs than the address space")?;
            if self.plan.frames[child].kind != Some(kind) {
                return Err("a route plan's construct order differs".into());
            }
            stack.push(child);
            for inner in node.regions_mut() {
                *inner = self.region(core::mem::take(inner), stack)?;
            }
            stack.pop();
            match &mut node {
                Node::If { results, .. }
                | Node::Block { results, .. }
                | Node::Loop { results, .. }
                | Node::Switch { results, .. } => results.clone_from(&self.plan.frames[child].full),
                _ => return Err("a route plan without a construct".into()),
            }
            out.push(node);
            if let Some(tag) = self.plan.frames[child].tag {
                let rest = self.sequence(nodes, end, stack)?;
                let arm = self.dispatch(frame, child, tag)?;
                let cond = self.test(tag, Cmp::Ne, 0, &mut out)?;
                return self.guard(frame, cond, arm, rest, out);
            }
        }
        let end = match end {
            Some(End::Yield(args)) => Some(End::Yield(self.packed(frame, args, None)?)),
            Some(e @ (End::Br(..) | End::Continue(..))) => {
                let (label, args, resume) = match e {
                    End::Br(l, a) => (l, a, false),
                    End::Continue(l, a) => (l, a, true),
                    _ => return Err("an unknown exit".into()),
                };
                let at = self.plan.target(stack, label)?;
                let target = stack[at];
                if self.plan.direct(stack, at, resume) {
                    Some(if resume {
                        End::Continue(label, args)
                    } else {
                        End::Br(label, self.packed(target, args, None)?)
                    })
                } else if self.plan.tail(stack, at, resume) {
                    Some(End::Yield(self.packed(frame, args, None)?))
                } else {
                    let route = *self.plan.keys.get(&(target, resume)).ok_or("an unplanned exit")?;
                    let normal = self.empty_normal(frame)?;
                    let args = self.packed(frame, normal, Some((route, args)))?;
                    Some(self.leave(frame, args)?)
                }
            }
            other => other,
        };
        Ok(Region { nodes: out, end })
    }
}

pub(super) fn route(f: &mut Function, body: Region) -> Result<(Region, usize), String> {
    let mut plan = Plan {
        frames: vec![FramePlan::new(None, Vec::new(), Vec::new())],
        routes: Vec::new(),
        keys: BTreeMap::new(),
        exits: 0,
    };
    plan.scan(f, &body, &mut vec![0])?;
    plan.allocate(f)?;
    let count = plan.exits;
    let mut router = Router {
        f,
        plan,
        next: 1,
        prefix: Vec::new(),
        constants: BTreeMap::new(),
        undefined: Vec::new(),
    };
    let mut body = router.region(body, &mut vec![0])?;
    router.prefix.append(&mut body.nodes);
    body.nodes = router.prefix;
    Ok((body, count))
}
