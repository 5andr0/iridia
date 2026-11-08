//! Where each lane of a value comes from: the value it is, through the vectors composed of it and the constructs that pass it
//! on. A lane of a loop's state that every `continue` hands back unchanged is the value it started as, and a lane that every
//! exit of a loop gives the same value is that value: neither needs a variable. A variable assigned under a condition that
//! differs between invocations is, to a compiler's uniformity analysis, as different as the condition, whatever is stored
//! (a loop counter that passes through an inner loop under a divergent `if` stops being the same in every thread), so
//! the lanes that are the same everywhere are kept out of variables.

use alloc::{collections::BTreeMap, vec::Vec};
use core::cell::RefCell;

use crate::{End, Function, LabelId, Node, Op, Region, ValueId};

/// A lane of a value that is not a composition or a pass-through of another: the value and the lane.
pub type LaneOrigin = (ValueId, u8);

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
pub struct LaneOrigins<'a> {
    f: &'a Function,
    at: BTreeMap<u32, Site<'a>>,
    /// The answers of `resolve` found so far, by lane.
    memo: RefCell<BTreeMap<(u32, u8), Vec<Entry>>>,
}

impl core::fmt::Debug for LaneOrigins<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LaneOrigins")
            .field("function", &self.f.name)
            .finish_non_exhaustive()
    }
}

/// How deep a chain of passes-on is followed before a lane is taken as its own origin.
const DEPTH: u32 = 512;

impl<'a> LaneOrigins<'a> {
    /// Index the constructs of a structured function for lane-origin queries.
    pub fn of(f: &'a Function, body: &'a Region) -> LaneOrigins<'a> {
        let mut s = LaneOrigins {
            f,
            at: BTreeMap::new(),
            memo: RefCell::new(BTreeMap::new()),
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
    pub fn origin(&self, v: ValueId, k: u8) -> LaneOrigin {
        self.resolve(v, k, &mut Vec::new(), 0).origin
    }

    /// `compute`, or the answer it gave before where that is the answer it would give now (see `Entry::holds`): what is
    /// resolved through an `If` or a loop is resolved once, not once for every way to reach it.
    fn resolve(&self, v: ValueId, k: u8, visiting: &mut Vec<LaneOrigin>, depth: u32) -> Res {
        if depth > DEPTH {
            return Res {
                origin: (v, k),
                ext: Vec::new(),
                deepest: depth,
                capped: true,
            };
        }
        let key = (v.0, k);
        if let Some(entry) = self
            .memo
            .borrow()
            .get(&key)
            .and_then(|entries| entries.iter().find(|e| e.holds(visiting, depth)))
        {
            return Res {
                origin: entry.origin,
                ext: entry.ext.clone(),
                deepest: depth.saturating_add(entry.height),
                capped: entry.capped_at.is_some(),
            };
        }
        let res = self.compute(v, k, visiting, depth);
        self.memo.borrow_mut().entry(key).or_default().push(Entry {
            origin: res.origin,
            ext: res.ext.clone(),
            height: res.deepest.saturating_sub(depth),
            capped_at: res.capped.then_some(depth),
        });
        res
    }

    fn compute(&self, v: ValueId, k: u8, visiting: &mut Vec<LaneOrigin>, depth: u32) -> Res {
        let own = (v, k);
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
                            let (ra, rb) = (
                                self.resolve(a, k, visiting, next),
                                self.resolve(b, k, visiting, next),
                            );
                            let mut out =
                                Res::at(if ra.origin == rb.origin { ra.origin } else { own }, depth);
                            out.absorb(&ra, None);
                            out.absorb(&rb, None);
                            out
                        }
                        (Some(x), None) | (None, Some(x)) => self.resolve(x, k, visiting, next),
                        (None, None) => Res::at(own, depth),
                    }
                }
                Site::LoopResult { label, body, index } => {
                    let mut out = Res::at(own, depth);
                    let mut agreed: Option<LaneOrigin> = None;
                    let mut exits = Vec::new();
                    // a loop body that ends in a yield leaves the loop with those values
                    if let Some(End::Yield(vs)) = &body.end {
                        exits.push(vs.as_slice());
                    }
                    exits_of(body, label, true, &mut exits);
                    for vs in exits {
                        let Some(x) = vs.get(index) else { return out };
                        let r = self.resolve(*x, k, visiting, next);
                        out.absorb(&r, None);
                        match agreed {
                            None => agreed = Some(r.origin),
                            Some(p) if p == r.origin => {}
                            Some(_) => return out,
                        }
                    }
                    out.origin = agreed.unwrap_or(own);
                    out
                }
                Site::LoopParam {
                    label,
                    body,
                    inits,
                    index,
                } => {
                    if visiting.contains(&own) {
                        // the answer depends on this parameter being on the way here
                        let mut out = Res::at(own, depth);
                        out.ext.push((own, true));
                        return out;
                    }
                    let mut out = Res::at(own, depth);
                    out.ext.push((own, false));
                    let Some(init) = inits.get(index) else { return out };
                    let r = self.resolve(*init, k, visiting, next);
                    out.absorb(&r, None);
                    let start = r.origin;
                    visiting.push(own);
                    let mut back = Vec::new();
                    exits_of(body, label, false, &mut back);
                    // every `continue` gives the lane back as it is, or gives what it started as
                    let mut same = true;
                    for vs in back {
                        let Some(x) = vs.get(index) else {
                            same = false;
                            break;
                        };
                        let r = self.resolve(*x, k, visiting, next);
                        // the stack this parameter pushed is gone when the answer is used
                        out.absorb(&r, Some(own));
                        if r.origin != own && r.origin != start {
                            same = false;
                            break;
                        }
                    }
                    visiting.pop();
                    out.origin = if same { start } else { own };
                    out
                }
            };
        }
        let Some(def) = self.f.value(v) else {
            return Res::at(own, depth);
        };
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
                Res::at(own, depth)
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
            _ => Res::at(own, depth),
        }
    }
}

/// What `LaneOrigins::compute` found, and what the answer depends on beyond the lane's own definition.
struct Res {
    origin: LaneOrigin,
    /// The loop parameters whose place on the way here was asked, and whether they were on it: the answer is the same
    /// wherever those are on the way as they were.
    ext: Vec<(LaneOrigin, bool)>,
    /// The deepest chain of passes-on followed to find it.
    deepest: u32,
    /// The chain was cut short at `DEPTH`: the answer depends on how deep it was asked at.
    capped: bool,
}

impl Res {
    fn at(origin: LaneOrigin, depth: u32) -> Res {
        Res {
            origin,
            ext: Vec::new(),
            deepest: depth,
            capped: false,
        }
    }

    /// What `other`, asked on the way, depends on too (but not on `except` being on the way, which it was only inside).
    fn absorb(&mut self, other: &Res, except: Option<LaneOrigin>) {
        for e in &other.ext {
            if except == Some(e.0) && e.1 {
                continue;
            }
            if !self.ext.contains(e) {
                self.ext.push(*e);
            }
        }
        self.deepest = self.deepest.max(other.deepest);
        self.capped |= other.capped;
    }
}

/// An answer of `LaneOrigins::resolve` kept for a lane: it is the answer wherever the same loop parameters are on the way as were
/// when it was found, and the chain it followed still fits under `DEPTH`; one cut short at `DEPTH` is the answer only at the
/// depth it was found at (where it was cut depends on that).
struct Entry {
    origin: LaneOrigin,
    ext: Vec<(LaneOrigin, bool)>,
    height: u32,
    capped_at: Option<u32>,
}

impl Entry {
    fn holds(&self, visiting: &[LaneOrigin], depth: u32) -> bool {
        let depth_ok = match self.capped_at {
            Some(at) => depth == at,
            None => depth.saturating_add(self.height) <= DEPTH,
        };
        depth_ok && self.ext.iter().all(|(o, was)| visiting.contains(o) == *was)
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::String};

    /// The recursion `LaneOrigins::resolve` replaced, as it was: no memory, so it does the work again for every way to reach a
    /// lane.
    fn reference(
        s: &LaneOrigins<'_>,
        v: ValueId,
        k: u8,
        visiting: &mut Vec<LaneOrigin>,
        depth: u32,
    ) -> LaneOrigin {
        let own = (v, k);
        if depth > DEPTH {
            return own;
        }
        let next = depth.saturating_add(1);
        if let Some(site) = s.at.get(&v.0) {
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
                                reference(s, a, k, visiting, next),
                                reference(s, b, k, visiting, next),
                            );
                            if oa == ob { oa } else { own }
                        }
                        (Some(x), None) | (None, Some(x)) => reference(s, x, k, visiting, next),
                        (None, None) => own,
                    }
                }
                Site::LoopResult { label, body, index } => {
                    let mut agreed: Option<LaneOrigin> = None;
                    let mut exits = Vec::new();
                    // a loop body that ends in a yield leaves the loop with those values
                    if let Some(End::Yield(vs)) = &body.end {
                        exits.push(vs.as_slice());
                    }
                    exits_of(body, label, true, &mut exits);
                    for vs in exits {
                        let Some(x) = vs.get(index) else { return own };
                        let o = reference(s, *x, k, visiting, next);
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
                    let start = reference(s, *init, k, visiting, next);
                    visiting.push(own);
                    let mut back = Vec::new();
                    exits_of(body, label, false, &mut back);
                    // every `continue` gives the lane back as it is, or gives what it started as
                    let same = back.iter().all(|vs| {
                        vs.get(index).is_some_and(|x| {
                            let o = reference(s, *x, k, visiting, next);
                            o == own || o == start
                        })
                    });
                    visiting.pop();
                    if same { start } else { own }
                }
            };
        }
        let Some(def) = s.f.value(v) else { return own };
        match &def.op {
            Op::Compose(parts) => {
                let mut at = 0u8;
                for p in parts {
                    let n = s.f.ty(*p).map_or(1, |t| t.lanes);
                    if k < at.saturating_add(n) {
                        return reference(s, *p, k.saturating_sub(at), visiting, next);
                    }
                    at = at.saturating_add(n);
                }
                own
            }
            Op::Swizzle(a, lanes) => {
                let from = if s.f.ty(*a).is_some_and(|t| t.lanes > 1) {
                    lanes.get(usize::from(k)).copied().unwrap_or(0)
                } else {
                    0
                };
                reference(s, *a, from, visiting, next)
            }
            Op::Splat(a) => reference(s, *a, 0, visiting, next),
            _ => own,
        }
    }

    const SCAN: &str = "iridia-ir 1
fn main {
  %0: vec3<u32> = input 0.xyz
  %1: u32 = %0.x
  %2: u32 = u32(0)
  %3: u32 = u32(1)
  %4: vec4<u32> = compose %1, %2, %2, %3
  loop '0 (%5: vec4<u32>) = (%4) {
    %6: u32 = %5.w
    %7: u32 = u32(64)
    %8: bool = ucmp.ge %6, %7
    if %8 {
      br '0 %5
    } else {
      yield
    }
    %9: u32 = %5.x
    %10: u32 = %5.y
    %11: u32 = %5.z
    barrier execution workgroup storage.workgroup
    %12: u32 = shl %6, %3
    %13: vec4<u32> = compose %9, %10, %11, %12
    continue '0 %13
  } -> %14: vec4<u32>
  return
}
";

    const THROUGH_AN_INNER_LOOP: &str = "iridia-ir 1
fn main {
  %0: vec3<u32> = input 0.xyz
  %1: u32 = %0.x
  %2: u32 = u32(0)
  %3: u32 = u32(64)
  %7: u32 = u32(1)
  %4: vec4<u32> = compose %3, %2, %2, %2
  loop '0 (%5: vec4<u32>) = (%4) {
    %6: u32 = %5.x
    %8: bool = ucmp.lt %6, %7
    if %8 {
      br '0 %5
    } else {
      yield
    }
    %9: bool = ucmp.lt %1, %6
    if %9 {
      loop '1 (%10: vec4<u32>) = (%5) {
        %11: u32 = %10.y
        %12: u32 = u32(10)
        %13: bool = ucmp.ge %11, %12
        if %13 {
          br '1 %10
        } else {
          yield
        }
        %14: u32 = iadd %11, %7
        %15: u32 = %10.x
        %16: u32 = %10.z
        %17: u32 = %10.w
        %18: vec4<u32> = compose %15, %14, %16, %17
        continue '1 %18
      } -> %19: vec4<u32>
      yield %19
    } else {
      yield %5
    } -> %20: vec4<u32>
    barrier execution workgroup storage.workgroup
    %21: u32 = %20.x
    %23: u32 = shru %21, %7
    %25: vec4<u32> = compose %23, %2, %2, %2
    continue '0 %25
  } -> %26: vec4<u32>
  return
}
";

    fn regions(f: &Function) -> &Region {
        match &f.body {
            crate::Body::Regions(r) => r,
            crate::Body::Cfg(_) => panic!("a control flow graph"),
        }
    }

    /// Every lane of every value of `text` resolves as the recursion resolved it, whichever order they are asked in (the
    /// answers kept from one lane are used for the next, so the order is part of the test).
    fn agrees(text: &str) {
        let f = crate::text::parse(text).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
        let ids: Vec<ValueId> = (0..f.values.len() as u32).map(ValueId).collect();
        for order in 0..3 {
            let sites = LaneOrigins::of(&f, regions(&f));
            let mut lanes: Vec<(ValueId, u8)> = ids
                .iter()
                .flat_map(|v| (0..f.ty(*v).map_or(1, |t| t.lanes)).map(move |k| (*v, k)))
                .collect();
            match order {
                1 => lanes.reverse(),
                2 => lanes.sort_by_key(|(v, k)| (k.wrapping_mul(7), v.0.wrapping_mul(2_654_435_761) >> 7)),
                _ => {}
            }
            for (v, k) in lanes {
                let want = reference(&LaneOrigins::of(&f, regions(&f)), v, k, &mut Vec::new(), 0);
                assert_eq!(
                    sites.origin(v, k),
                    want,
                    "order {order}: lane {k} of %{}\n{text}",
                    v.0
                );
            }
        }
    }

    #[test]
    fn a_loop_state_in_a_vector_resolves_as_it_did() {
        agrees(SCAN);
    }

    #[test]
    fn a_loop_in_a_branch_in_a_loop_resolves_as_it_did() {
        agrees(THROUGH_AN_INNER_LOOP);
    }

    /// `depth` ifs, each yielding the one before from both sides.
    fn nested_ifs(depth: u32) -> String {
        let mut text =
            String::from("iridia-ir 1\nfn main {\n  %0: u32 = u32(1)\n  %1: bool = ucmp.ge %0, %0\n");
        let mut last = 0;
        for i in 0..depth {
            let n = 2 + i;
            text.push_str(&format!(
                "  if %1 {{\n    yield %{last}\n  }} else {{\n    yield %{last}\n  }} -> %{n}: u32\n"
            ));
            last = n;
        }
        text.push_str("  return\n}\n");
        text
    }

    #[test]
    fn nested_branches_resolve_as_they_did() {
        agrees(&nested_ifs(10));
    }

    #[test]
    fn forty_nested_branches_resolve_at_once() {
        // 2^40 ways to reach the first value: the recursion would not finish
        let text = nested_ifs(40);
        let f = crate::text::parse(&text).unwrap();
        let sites = LaneOrigins::of(&f, regions(&f));
        let last = ValueId(41);
        assert_eq!(sites.origin(last, 0), (ValueId(0), 0));
    }

    #[test]
    fn branches_nested_deeper_than_the_depth_limit_resolve_at_once() {
        // the chain is cut at `DEPTH` where it was asked from, and each way to reach the same value is cut in the same place
        let text = nested_ifs(DEPTH + 88);
        let f = crate::text::parse(&text).unwrap();
        let sites = LaneOrigins::of(&f, regions(&f));
        let last = ValueId(DEPTH + 89);
        assert_eq!(sites.origin(last, 0), (ValueId(last.0 - DEPTH - 1), 0));
    }
}
