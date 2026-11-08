//! Reachability through SSA operations and structured transfers. An exit argument is needed only when its
//! destination slot is needed; treating every argument as a root retains otherwise dead phi cycles.

use alloc::{collections::BTreeMap, vec, vec::Vec};
use iridia_ir::{End, Function, LabelId, Node, Region, ValueId};

#[derive(Clone, Copy)]
struct Targets<'a> {
    results: &'a [ValueId],
    params: &'a [ValueId],
}

struct Graph {
    operands: Vec<Vec<ValueId>>,
    roots: Vec<ValueId>,
}

impl Graph {
    fn transfer(&mut self, dest: &[ValueId], src: &[ValueId]) {
        for (d, s) in dest.iter().zip(src) {
            if let Some(operands) = self.operands.get_mut(d.0 as usize) {
                operands.push(*s);
            }
        }
    }

    fn region<'a>(
        &mut self,
        f: &Function,
        r: &'a Region,
        yields: &'a [ValueId],
        labels: &mut BTreeMap<LabelId, Targets<'a>>,
    ) {
        for node in &r.nodes {
            match node {
                Node::Value(v) => {
                    if let Some(d) = f.value(*v) {
                        if let Some(operands) = self.operands.get_mut(v.0 as usize) {
                            operands.extend(d.op.operands());
                        }
                        if d.op.has_effect() {
                            self.roots.push(*v);
                        }
                    }
                }
                Node::If {
                    cond,
                    accept,
                    reject,
                    results,
                } => {
                    self.roots.push(*cond);
                    self.region(f, accept, results, labels);
                    self.region(f, reject, results, labels);
                }
                Node::Block { label, body, results } => {
                    let old = labels.insert(*label, Targets { results, params: &[] });
                    self.region(f, body, results, labels);
                    restore(labels, *label, old);
                }
                Node::Loop {
                    label,
                    body,
                    results,
                    params,
                    inits,
                } => {
                    self.transfer(params, inits);
                    let old = labels.insert(*label, Targets { results, params });
                    self.region(f, body, results, labels);
                    restore(labels, *label, old);
                }
                Node::Switch {
                    label,
                    selector,
                    cases,
                    default,
                    results,
                } => {
                    self.roots.push(*selector);
                    let old = labels.insert(*label, Targets { results, params: &[] });
                    for case in cases {
                        self.region(f, &case.body, results, labels);
                    }
                    self.region(f, default, results, labels);
                    restore(labels, *label, old);
                }
                // Stores, atomics' addressing nodes and other effects keep their operands. Control remains in
                // place even when all of a construct's slots disappear; no branch or loop is speculated away.
                other => self.roots.extend(other.values()),
            }
        }
        match &r.end {
            Some(End::Yield(vs)) => self.transfer(yields, vs),
            Some(End::Br(label, vs)) => {
                if let Some(target) = labels.get(label) {
                    self.transfer(target.results, vs);
                }
            }
            Some(End::Continue(label, vs)) => {
                if let Some(target) = labels.get(label) {
                    self.transfer(target.params, vs);
                }
            }
            _ => {}
        }
    }
}

fn restore<T>(labels: &mut BTreeMap<LabelId, T>, label: LabelId, old: Option<T>) {
    if let Some(old) = old {
        labels.insert(label, old);
    } else {
        labels.remove(&label);
    }
}

/// One graph walk reaches the fixed point, including cycles of loop parameters and nested construct results.
pub(super) fn values(f: &Function, body: &Region) -> Vec<bool> {
    let mut graph = Graph {
        operands: vec![Vec::new(); f.values.len()],
        roots: Vec::new(),
    };
    graph.region(f, body, &[], &mut BTreeMap::new());
    let mut live = vec![false; f.values.len()];
    while let Some(v) = graph.roots.pop() {
        if let Some(mark) = live.get_mut(v.0 as usize) {
            if core::mem::replace(mark, true) {
                continue;
            }
            if let Some(operands) = graph.operands.get(v.0 as usize) {
                graph.roots.extend(operands);
            }
        }
    }
    live
}

fn mask(vs: &[ValueId], live: &[bool]) -> Vec<bool> {
    vs.iter()
        .map(|v| live.get(v.0 as usize).copied().unwrap_or(true))
        .collect()
}

fn select<T>(vs: &mut Vec<T>, keep: &[bool]) {
    let mut flags = keep.iter();
    vs.retain(|_| flags.next().copied().unwrap_or(true));
}

#[derive(Default)]
struct Keep {
    results: Vec<bool>,
    params: Vec<bool>,
}

/// Remove the dead definitions and exactly their incoming argument positions, preserving lexical label shadowing.
fn region(r: &mut Region, yields: &[bool], live: &[bool], labels: &mut BTreeMap<LabelId, Keep>) {
    r.nodes
        .retain(|n| !matches!(n, Node::Value(v) if !live.get(v.0 as usize).copied().unwrap_or(true)));
    for node in &mut r.nodes {
        match node {
            Node::If {
                accept,
                reject,
                results,
                ..
            } => {
                let keep = mask(results, live);
                region(accept, &keep, live, labels);
                region(reject, &keep, live, labels);
                select(results, &keep);
            }
            Node::Block { label, body, results } => {
                let keep = mask(results, live);
                let old = labels.insert(
                    *label,
                    Keep {
                        results: keep.clone(),
                        ..Keep::default()
                    },
                );
                region(body, &keep, live, labels);
                restore(labels, *label, old);
                select(results, &keep);
            }
            Node::Loop {
                label,
                body,
                results,
                params,
                inits,
            } => {
                let keep = mask(results, live);
                let carried = mask(params, live);
                let old = labels.insert(
                    *label,
                    Keep {
                        results: keep.clone(),
                        params: carried.clone(),
                    },
                );
                region(body, &keep, live, labels);
                restore(labels, *label, old);
                select(results, &keep);
                select(params, &carried);
                select(inits, &carried);
            }
            Node::Switch {
                label,
                cases,
                default,
                results,
                ..
            } => {
                let keep = mask(results, live);
                let old = labels.insert(
                    *label,
                    Keep {
                        results: keep.clone(),
                        ..Keep::default()
                    },
                );
                for case in cases {
                    region(&mut case.body, &keep, live, labels);
                }
                region(default, &keep, live, labels);
                restore(labels, *label, old);
                select(results, &keep);
            }
            _ => {}
        }
    }
    match &mut r.end {
        Some(End::Yield(vs)) => select(vs, yields),
        Some(End::Br(label, vs)) => {
            if let Some(keep) = labels.get(label) {
                select(vs, &keep.results);
            }
        }
        Some(End::Continue(label, vs)) => {
            if let Some(keep) = labels.get(label) {
                select(vs, &keep.params);
            }
        }
        _ => {}
    }
}

pub(super) fn prune(body: &mut Region, live: &[bool]) {
    region(body, &[], live, &mut BTreeMap::new());
}
