//! Equal lane identities across structured results. No loop-carried value is assumed invariant.
use alloc::{collections::BTreeMap, vec, vec::Vec};
use iridia_ir::{Body, End, Function, LabelId, Node, Op, Region, ValueId};

type Lane = (ValueId, u8);
pub(super) type Aliases = BTreeMap<Lane, Option<Lane>>;

fn incoming(f: &Function) -> BTreeMap<ValueId, Vec<ValueId>> {
    fn add(results: &[ValueId], values: &[ValueId], out: &mut BTreeMap<ValueId, Vec<ValueId>>) {
        for (&result, &value) in results.iter().zip(values) {
            out.entry(result).or_default().push(value);
        }
    }
    fn walk(
        r: &Region,
        results: &[ValueId],
        exits: &mut Vec<(LabelId, Vec<ValueId>)>,
        out: &mut BTreeMap<ValueId, Vec<ValueId>>,
    ) {
        for n in &r.nodes {
            match n {
                Node::If {
                    accept,
                    reject,
                    results,
                    ..
                } => {
                    walk(accept, results, exits, out);
                    walk(reject, results, exits, out);
                }
                Node::Block { label, body, results } => {
                    exits.push((*label, results.clone()));
                    walk(body, results, exits, out);
                    exits.pop();
                }
                Node::Switch {
                    label,
                    cases,
                    default,
                    results,
                    ..
                } => {
                    exits.push((*label, results.clone()));
                    for case in cases {
                        walk(&case.body, results, exits, out);
                    }
                    walk(default, results, exits, out);
                    exits.pop();
                }
                // Breaks to enclosing blocks still contribute, but neither loop parameters nor results are aliases.
                Node::Loop { label, body, .. } => {
                    exits.push((*label, Vec::new()));
                    walk(body, &[], exits, out);
                    exits.pop();
                }
                _ => {}
            }
        }
        match &r.end {
            Some(End::Yield(values)) => add(results, values, out),
            Some(End::Br(label, values)) => {
                if let Some((_, results)) = exits.iter().rev().find(|(at, _)| at == label) {
                    add(results, values, out);
                }
            }
            _ => {}
        }
    }
    let mut out = BTreeMap::new();
    if let Body::Regions(r) = &f.body {
        walk(r, &[], &mut Vec::new(), &mut out);
    }
    out
}

/// The same source lane must reach every exit. Other lanes of the same result may remain dynamic.
/// The explicit stack and memoisation visit each lane once even with deeply nested shared results.
pub(super) fn aliases(f: &Function) -> Aliases {
    let incoming = incoming(f);
    let mut cache = Aliases::new();
    // Constant folding may turn only one arm's compose into a literal. Preserve lane identity across that change.
    let mut literals = BTreeMap::new();
    for (index, d) in f.values.iter().enumerate() {
        let Ok(index) = u32::try_from(index) else { break };
        if let Op::Literal(bits) = &d.op {
            for (lane, bits) in bits.iter().enumerate().take(usize::from(d.ty.lanes)) {
                let Ok(lane) = u8::try_from(lane) else { continue };
                let key = (ValueId(index), lane);
                let canonical = *literals.entry((d.ty.scalar.bits(), *bits)).or_insert(key);
                cache.insert(key, Some(canonical));
            }
        }
    }
    for (index, d) in f.values.iter().enumerate() {
        let Ok(index) = u32::try_from(index) else { break };
        for lane in 0..d.ty.lanes {
            let mut stack = vec![((ValueId(index), lane), false)];
            while let Some((key @ (id, lane), expanded)) = stack.pop() {
                if !expanded && cache.contains_key(&key) {
                    continue;
                }
                let Some(d) = f.value(id) else {
                    cache.insert(key, None);
                    continue;
                };
                let sources = match &d.op {
                    Op::Param => incoming
                        .get(&id)
                        .map(|vs| vs.iter().map(|&v| (v, lane)).collect::<Vec<_>>()),
                    Op::Swizzle(v, picks) => picks.get(usize::from(lane)).map(|&l| vec![(*v, l)]),
                    Op::Splat(v) => Some(vec![(*v, 0)]),
                    Op::Compose(parts) => {
                        let mut lane = lane;
                        parts.iter().find_map(|&v| {
                            let width = f.ty(v)?.lanes;
                            if lane < width {
                                Some(vec![(v, lane)])
                            } else {
                                lane = lane.saturating_sub(width);
                                None
                            }
                        })
                    }
                    Op::Bitcast(v) if f.ty(*v).is_some_and(|t| t.scalar.bits() == d.ty.scalar.bits()) => {
                        Some(vec![(*v, lane)])
                    }
                    _ => {
                        cache.insert(key, Some(key));
                        continue;
                    }
                };
                let Some(sources) = sources.filter(|s| !s.is_empty()) else {
                    cache.insert(key, None);
                    continue;
                };
                if !expanded {
                    cache.insert(key, None); // A cycle is unknown, never an invariant guessed from its initial value.
                    stack.push((key, true));
                    stack.extend(
                        sources
                            .into_iter()
                            .filter(|v| !cache.contains_key(v))
                            .map(|v| (v, false)),
                    );
                } else {
                    let root = sources.first().and_then(|v| cache.get(v)).copied().flatten();
                    let equal =
                        root.filter(|root| sources.iter().all(|v| cache.get(v) == Some(&Some(*root))));
                    cache.insert(key, equal);
                }
            }
        }
    }
    cache
}

pub(super) fn root(aliases: &Aliases, id: ValueId, lane: u8) -> Option<Lane> {
    aliases.get(&(id, lane)).copied().flatten()
}
