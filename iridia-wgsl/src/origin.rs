//! Where each lane of a value comes from: the value it is, through the vectors composed of it and the constructs that pass it
//! on. A lane of a loop's state that every `continue` hands back unchanged is the value it started as, and a lane that every
//! exit of a loop gives the same value is that value: neither needs a variable. A variable assigned under a condition that
//! differs between invocations is, to a compiler's uniformity analysis, as different as the condition, whatever is stored
//! (a loop counter that passes through an inner loop under a divergent `if` stops being the same in every thread), so
//! the lanes that are the same everywhere are kept out of variables.

use alloc::{collections::BTreeMap, vec::Vec};

use iridia_ir::{End, Function, LabelId, Node, Op, Region, ValueId};

/// A lane of a value that is not a composition or a pass-through of another: the value and the lane.
pub(crate) type Origin = (ValueId, u8);

/// Where a value is defined by a construct.
#[derive(Clone, Copy)]
enum Site<'a> {
    /// Result `index` of an `If`.
    If {
        accept: &'a Region,
        reject: &'a Region,
        index: usize,
    },
    /// Parameter `index` of a loop.
    LoopParam {
        label: LabelId,
        body: &'a Region,
        inits: &'a [ValueId],
        index: usize,
    },
    /// Result `index` of a loop.
    LoopResult {
        label: LabelId,
        body: &'a Region,
        index: usize,
    },
}

/// The constructs of a function that define values, by value.
pub(crate) struct Sites<'a> {
    f: &'a Function,
    at: BTreeMap<u32, Site<'a>>,
}

/// How deep a chain of passes-on is followed before a lane is taken as its own origin.
const DEPTH: u32 = 512;

impl<'a> Sites<'a> {
    pub(crate) fn of(f: &'a Function, body: &'a Region) -> Sites<'a> {
        let mut s = Sites {
            f,
            at: BTreeMap::new(),
        };
        s.walk(body);
        s
    }

    fn walk(&mut self, r: &'a Region) {
        for n in &r.nodes {
            match n {
                Node::If {
                    accept,
                    reject,
                    results,
                    ..
                } => {
                    for (index, v) in results.iter().enumerate() {
                        self.at.insert(
                            v.0,
                            Site::If {
                                accept,
                                reject,
                                index,
                            },
                        );
                    }
                    self.walk(accept);
                    self.walk(reject);
                }
                Node::Switch { cases, default, .. } => {
                    for c in cases {
                        self.walk(&c.body);
                    }
                    self.walk(default);
                }
                Node::Block { body, .. } => self.walk(body),
                Node::Loop {
                    label,
                    params,
                    inits,
                    body,
                    results,
                } => {
                    for (index, v) in params.iter().enumerate() {
                        self.at.insert(
                            v.0,
                            Site::LoopParam {
                                label: *label,
                                body,
                                inits,
                                index,
                            },
                        );
                    }
                    for (index, v) in results.iter().enumerate() {
                        self.at.insert(
                            v.0,
                            Site::LoopResult {
                                label: *label,
                                body,
                                index,
                            },
                        );
                    }
                    self.walk(body);
                }
                _ => {}
            }
        }
    }

    /// The origin of lane `k` of `v`.
    pub(crate) fn origin(&self, v: ValueId, k: u8) -> Origin {
        self.resolve(v, k, &mut Vec::new(), 0)
    }

    fn resolve(&self, v: ValueId, k: u8, visiting: &mut Vec<Origin>, depth: u32) -> Origin {
        let own = (v, k);
        if depth > DEPTH {
            return own;
        }
        let next = depth.saturating_add(1);
        if let Some(site) = self.at.get(&v.0) {
            return match *site {
                Site::If {
                    accept,
                    reject,
                    index,
                } => {
                    let from = |r: &Region| match &r.end {
                        Some(End::Yield(vs)) => vs.get(index).copied(),
                        _ => None,
                    };
                    match (from(accept), from(reject)) {
                        (Some(a), Some(b)) => {
                            let (oa, ob) = (
                                self.resolve(a, k, visiting, next),
                                self.resolve(b, k, visiting, next),
                            );
                            if oa == ob { oa } else { own }
                        }
                        (Some(x), None) | (None, Some(x)) => self.resolve(x, k, visiting, next),
                        (None, None) => own,
                    }
                }
                Site::LoopResult { label, body, index } => {
                    let mut agreed: Option<Origin> = None;
                    let mut exits = Vec::new();
                    // a loop body that ends in a yield leaves the loop with those values
                    if let Some(End::Yield(vs)) = &body.end {
                        exits.push(vs.as_slice());
                    }
                    exits_of(body, label, true, &mut exits);
                    for vs in exits {
                        let Some(x) = vs.get(index) else { return own };
                        let o = self.resolve(*x, k, visiting, next);
                        match agreed {
                            None => agreed = Some(o),
                            Some(p) if p == o => {}
                            Some(_) => return own,
                        }
                    }
                    agreed.unwrap_or(own)
                }
                Site::LoopParam {
                    label,
                    body,
                    inits,
                    index,
                } => {
                    if visiting.contains(&own) {
                        return own;
                    }
                    let Some(init) = inits.get(index) else { return own };
                    let start = self.resolve(*init, k, visiting, next);
                    visiting.push(own);
                    let mut back = Vec::new();
                    exits_of(body, label, false, &mut back);
                    // every `continue` gives the lane back as it is, or gives what it started as
                    let same = back.iter().all(|vs| {
                        vs.get(index).is_some_and(|x| {
                            let o = self.resolve(*x, k, visiting, next);
                            o == own || o == start
                        })
                    });
                    visiting.pop();
                    if same { start } else { own }
                }
            };
        }
        let Some(def) = self.f.value(v) else { return own };
        match &def.op {
            Op::Compose(parts) => {
                let mut at = 0u8;
                for p in parts {
                    let n = self.f.ty(*p).map_or(1, |t| t.lanes);
                    if k < at.saturating_add(n) {
                        return self.resolve(*p, k.saturating_sub(at), visiting, next);
                    }
                    at = at.saturating_add(n);
                }
                own
            }
            Op::Swizzle(a, lanes) => {
                let from = if self.f.ty(*a).is_some_and(|t| t.lanes > 1) {
                    lanes.get(usize::from(k)).copied().unwrap_or(0)
                } else {
                    0
                };
                self.resolve(*a, from, visiting, next)
            }
            Op::Splat(a) => self.resolve(*a, 0, visiting, next),
            _ => own,
        }
    }
}

/// The values every exit of `label` in `r` gives: a `break` (`br`) of a loop's results, or a `continue` of its parameters.
fn exits_of<'a>(r: &'a Region, label: LabelId, results: bool, out: &mut Vec<&'a [ValueId]>) {
    match &r.end {
        Some(End::Br(l, vs)) if results && *l == label => out.push(vs),
        Some(End::Continue(l, vs)) if !results && *l == label => out.push(vs),
        _ => {}
    }
    for n in &r.nodes {
        match n {
            Node::If { accept, reject, .. } => {
                exits_of(accept, label, results, out);
                exits_of(reject, label, results, out);
            }
            Node::Switch { cases, default, .. } => {
                for c in cases {
                    exits_of(&c.body, label, results, out);
                }
                exits_of(default, label, results, out);
            }
            Node::Block { body, .. } | Node::Loop { body, .. } => exits_of(body, label, results, out),
            _ => {}
        }
    }
}
