//! Constant lanes evaluated by the IR's numeric definition, including lanes of partly dynamic vectors.
use super::lanes::{self, Aliases};
use alloc::{collections::BTreeMap, string::String, vec, vec::Vec};
use iridia_ir::{
    Denormals, Function, Op, Precision, Type, Unary, ValueId,
    eval::{self, Operands},
};

type Lane = (ValueId, u8);
type Known = BTreeMap<Lane, Option<u64>>;

fn lanewise(op: &Op) -> bool {
    match op {
        Op::Unary(Unary::Any | Unary::All, _) => false,
        Op::Unary(..)
        | Op::Binary(..)
        | Op::Ternary(..)
        | Op::BitInsert { .. }
        | Op::Select { .. }
        | Op::Convert(..) => true,
        _ => false,
    }
}
fn at(f: &Function, v: ValueId, lane: u8) -> Lane {
    (
        v,
        if f.ty(v).is_some_and(|t| t.lanes == 1) {
            0
        } else {
            lane
        },
    )
}
fn bits(words: eval::Lanes, ty: Type, lane: u8) -> Option<u64> {
    if ty.scalar.bits() == 64 {
        let index = usize::from(lane).checked_mul(2)?;
        Some(u64::from(*words.get(index)?) | u64::from(*words.get(index.checked_add(1)?)?) << 32)
    } else {
        words.get(usize::from(lane)).copied().map(u64::from)
    }
}
struct Constants<'a> {
    f: &'a Function,
    known: &'a Known,
    lane: Option<u8>,
}
impl Operands for Constants<'_> {
    fn ty(&self, v: ValueId) -> Result<Type, String> {
        let mut ty = self.f.ty(v).ok_or_else(|| String::from("absent operand type"))?;
        if self.lane.is_some() {
            ty.lanes = 1;
        }
        Ok(ty)
    }
    fn get(&self, v: ValueId) -> Result<eval::Lanes, String> {
        let ty = self.ty(v)?;
        let mut words = [0; 4];
        for (lane, out) in words.iter_mut().enumerate().take(usize::from(ty.lanes)) {
            let lane = self.lane.unwrap_or(u8::try_from(lane).unwrap_or(u8::MAX));
            *out = self
                .known
                .get(&at(self.f, v, lane))
                .copied()
                .flatten()
                .ok_or_else(|| String::from("dynamic operand lane"))?;
        }
        Ok(eval::literal_words(&words, ty))
    }
}

/// Pure componentwise operations can be evaluated for one fixed lane even while their neighbours stay uniforms.
/// Cross-lane operations retain all their operand lanes; the reference evaluator defines every operation's bits.
pub(super) fn values(f: &Function, aliases: &Aliases, denormals: Denormals) -> Known {
    let mut known = Known::new();
    for (index, d) in f.values.iter().enumerate() {
        let Ok(index) = u32::try_from(index) else { break };
        for lane in 0..d.ty.lanes {
            let mut stack = vec![((ValueId(index), lane), false)];
            while let Some((key @ (v, lane), expanded)) = stack.pop() {
                if !expanded && known.contains_key(&key) {
                    continue;
                }
                let Some(d) = f.value(v) else {
                    known.insert(key, None);
                    continue;
                };
                if d.precision != Precision::Full {
                    known.insert(key, None);
                    continue;
                }
                if let Op::Literal(words) = &d.op {
                    known.insert(key, words.get(usize::from(lane)).copied());
                    continue;
                }
                let alias = lanes::root(aliases, v, lane).filter(|&r| r != key);
                let scalar = lanewise(&d.op);
                let operands: Vec<_> = if let Some(root) = alias {
                    vec![root]
                } else {
                    d.op.operands()
                        .into_iter()
                        .flat_map(|v| {
                            if scalar {
                                vec![at(f, v, lane)]
                            } else {
                                (0..f.ty(v).map_or(0, |t| t.lanes))
                                    .map(|lane| (v, lane))
                                    .collect()
                            }
                        })
                        .collect()
                };
                if !expanded {
                    known.insert(key, None); // Loops cannot be proved constant from a provisional value.
                    stack.push((key, true));
                    stack.extend(
                        operands
                            .into_iter()
                            .filter(|o| !known.contains_key(o))
                            .map(|o| (o, false)),
                    );
                } else {
                    let result = if let Some(root) = alias {
                        known.get(&root).copied().flatten()
                    } else {
                        let mut ty = d.ty;
                        if scalar {
                            ty.lanes = 1;
                        }
                        eval::eval(
                            &d.op,
                            ty,
                            &Constants {
                                f,
                                known: &known,
                                lane: scalar.then_some(lane),
                            },
                            denormals == Denormals::Flush,
                        )
                        .ok()
                        .flatten()
                        .and_then(|words| bits(words, ty, if scalar { 0 } else { lane }))
                    };
                    known.insert(key, result);
                }
            }
        }
    }
    known
}
