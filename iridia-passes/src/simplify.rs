//! Lane bookkeeping folded away, and values nothing uses removed. Exact: no arithmetic is touched.
//!
//! - A swizzle of a swizzle is one swizzle; a swizzle that gives its operand back is the operand.
//! - A swizzle of a compose of scalars, or of a splat, takes the parts it names.
//! - A compose of single lanes of one value, in order and all of them, is that value; in another order, a swizzle.

use alloc::{vec, vec::Vec};

use iridia_ir::{Body, End, Function, Node, Op, Region, Type, ValueId};

mod live;

fn ty(f: &Function, v: ValueId) -> Option<Type> {
    f.ty(v)
}

/// What a value's lanes are, where it is lane bookkeeping: the value and lane each comes from.
fn lanes_of(f: &Function, v: ValueId) -> Option<Vec<(ValueId, u8)>> {
    let def = f.value(v)?;
    let n = usize::from(def.ty.lanes);
    match &def.op {
        Op::Swizzle(x, lanes) => Some(lanes.iter().take(n).map(|&l| (*x, l)).collect()),
        _ => None,
    }
}

/// The simpler form of value `v`, if there is one: another value it equals, or a new operation.
enum Fold {
    Same(ValueId),
    Op(Op),
}

fn fold(f: &Function, v: ValueId) -> Option<Fold> {
    let def = f.value(v)?;
    let n = usize::from(def.ty.lanes);
    match &def.op {
        Op::Swizzle(x, lanes) => {
            let src = f.value(*x)?;
            let picked: Vec<u8> = lanes.iter().take(n).copied().collect();
            match &src.op {
                Op::Swizzle(y, inner) => {
                    let mut all = [0u8; 4];
                    for (to, &l) in all.iter_mut().zip(&picked) {
                        *to = *inner.get(usize::from(l))?;
                    }
                    Some(Fold::Op(Op::Swizzle(*y, all)))
                }
                Op::Compose(parts) if parts.iter().all(|p| ty(f, *p).is_some_and(|t| t.lanes == 1)) => {
                    let chosen: Vec<ValueId> = picked
                        .iter()
                        .map(|&l| parts.get(usize::from(l)).copied())
                        .collect::<Option<_>>()?;
                    match chosen.as_slice() {
                        [one] => Some(Fold::Same(*one)),
                        _ => Some(Fold::Op(Op::Compose(chosen))),
                    }
                }
                Op::Splat(s) if n == 1 => Some(Fold::Same(*s)),
                Op::Splat(s) => Some(Fold::Op(Op::Splat(*s))),
                _ if src.ty == def.ty && picked.iter().enumerate().all(|(i, &l)| usize::from(l) == i) => {
                    Some(Fold::Same(*x))
                }
                _ => None,
            }
        }
        Op::Compose(parts) => {
            let mut from: Vec<(ValueId, u8)> = Vec::new();
            for p in parts {
                let l = lanes_of(f, *p)?;
                if l.len() != 1 {
                    return None;
                }
                from.extend(l);
            }
            let (x, _) = *from.first()?;
            if from.iter().any(|(y, _)| *y != x) {
                return None;
            }
            let lanes: Vec<u8> = from.iter().map(|&(_, l)| l).collect();
            let src = ty(f, x)?;
            if src == def.ty && lanes.iter().enumerate().all(|(i, &l)| usize::from(l) == i) {
                return Some(Fold::Same(x));
            }
            let mut all = [0u8; 4];
            for (to, l) in all.iter_mut().zip(&lanes) {
                *to = *l;
            }
            Some(Fold::Op(Op::Swizzle(x, all)))
        }
        _ => None,
    }
}

struct Simplify<'a> {
    f: &'a mut Function,
    subst: Vec<Option<ValueId>>,
}

impl Simplify<'_> {
    fn sub(&self, v: ValueId) -> ValueId {
        self.subst.get(v.0 as usize).copied().flatten().unwrap_or(v)
    }
    fn subs(&self, vs: &mut [ValueId]) {
        for v in vs {
            *v = self.sub(*v);
        }
    }
    fn region(&mut self, r: &mut Region) {
        let nodes = core::mem::take(&mut r.nodes);
        for mut n in nodes {
            match &mut n {
                Node::Value(v) => {
                    let v = *v;
                    let mut op = match self.f.value(v) {
                        Some(d) => d.op.clone(),
                        None => continue,
                    };
                    op.map_operands(|o| self.sub(o));
                    if let Some(d) = self.f.values.get_mut(v.0 as usize) {
                        d.op = op;
                    }
                    // fold until it no longer changes (each fold makes the operation smaller)
                    let mut placed = true;
                    while let Some(simpler) = fold(self.f, v) {
                        match simpler {
                            Fold::Same(x) => {
                                if let Some(s) = self.subst.get_mut(v.0 as usize) {
                                    *s = Some(x);
                                }
                                placed = false;
                                break;
                            }
                            Fold::Op(op) => {
                                if let Some(d) = self.f.values.get_mut(v.0 as usize) {
                                    d.op = op;
                                }
                            }
                        }
                    }
                    if !placed {
                        continue;
                    }
                }
                other => {
                    for x in other.values_mut() {
                        *x = self.sub(*x);
                    }
                    for inner in other.regions_mut() {
                        self.region(inner);
                    }
                }
            }
            r.nodes.push(n);
        }
        if let Some(End::Yield(vs) | End::Br(_, vs) | End::Continue(_, vs)) = &mut r.end {
            self.subs(vs);
        }
    }
}

/// Fold lane bookkeeping and remove unused values and construct slots, in a `Regions` function (a graph body is
/// left as it is). A worklist follows operations and live transfer destinations to a fixed point: dead loop-carried
/// cycles do not keep themselves alive. Effects and control flow stay in place, including unused atomics.
pub fn simplify(f: &mut Function) {
    let mut body = match &mut f.body {
        Body::Regions(body) => core::mem::take(body),
        Body::Cfg(_) => return,
    };
    let mut s = Simplify {
        subst: vec![None; f.values.len()],
        f,
    };
    s.region(&mut body);
    prune_unused(f, &mut body);
    f.body = Body::Regions(body);
}

pub(crate) fn prune_unused(f: &Function, body: &mut Region) {
    let live = live::values(f, body);
    live::prune(body, &live);
}
