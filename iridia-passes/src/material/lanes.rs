//! Lane identities through moves and equal literals; structured results remain unknown.
use alloc::{collections::BTreeMap, vec, vec::Vec};
use iridia_ir::{Function, Op, ValueId};

type Lane = (ValueId, u8);
pub(super) type Aliases = BTreeMap<Lane, Option<Lane>>;

/// The same source lane must reach every exit. Other lanes of the same result may remain dynamic.
/// The explicit stack and memoisation visit each lane once even with deeply nested shared results.
pub(super) fn aliases(f: &Function) -> Aliases {
    let incoming = BTreeMap::<ValueId, Vec<ValueId>>::new();
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
