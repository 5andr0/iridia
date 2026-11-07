//! A node of a program's value graph ([`iridia_analysis::flow::Values`]) evaluated from the words of the leaves it
//! reads, each operation by the IR's definition ([`crate::eval`], the interpreter's own): what a run of the program
//! computes for that lane where it is given the same leaves, bit for bit. The caller answers the leaves (an input lane,
//! a supplied value, a buffer's word, a texel, a query's answer, a counter) or leaves them unknown; a node that needs an
//! unknown leaf is unknown, naming it. A select whose condition is unknown is known where both its values are and are
//! the same bits.
//!
//! Refused with an error, never guessed: an operation the graph states no type for, a derivative, a wave operation, an
//! atomic's result, an undefined value, a loop's value, a merge whose arms differ (the graph does not state which
//! condition picks which arm), an element picked by a run-time index ([`Op::Index`]: the graph does not state which
//! candidate an index picks), a lane of a bitcast between widths ([`Op::Join`], [`Op::Part`]), a value that is no
//! bool or 32-bit scalar, and an operation the interpreter refuses.
use std::collections::BTreeMap;

use iridia_analysis::flow::{At, Node, NodeId, Op, Values};
use iridia_ir::{Scalar, Type, Unary, ValueId};

use crate::{Lanes, Operands};

/// What a node evaluates to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evaluated {
    /// The lane's word (a bool is 0 or 1).
    Word(u32),
    /// The leaf whose word is not known (the caller's answer was `None`, or a value the leaf is read at is unknown).
    Unknown(NodeId),
}

/// Evaluates nodes of one value graph, keeping what it has evaluated: the leaves are asked once each.
pub struct Evaluator<'v, L> {
    values: &'v Values,
    flush_denormals: bool,
    leaf: L,
    known: BTreeMap<NodeId, Evaluated>,
}

impl<L> core::fmt::Debug for Evaluator<'_, L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Evaluator")
            .field("flush_denormals", &self.flush_denormals)
            .field("known", &self.known.len())
            .finish_non_exhaustive()
    }
}

/// One step of the walk: the node's result, or the operands it needs first.
enum Step {
    Done(Evaluated),
    Need(Vec<NodeId>),
}

/// An operation's operands for [`crate::eval`]: value `k` is operand `k`, of its type.
struct Args {
    lanes: Vec<Lanes>,
    types: Vec<Type>,
}

impl Operands for Args {
    fn get(&self, v: ValueId) -> Result<Lanes, String> {
        self.lanes
            .get(usize::try_from(v.0).map_err(|e| format!("{e}"))?)
            .copied()
            .ok_or_else(|| format!("operand {} of none", v.0))
    }

    fn ty(&self, v: ValueId) -> Result<Type, String> {
        self.types
            .get(usize::try_from(v.0).map_err(|e| format!("{e}"))?)
            .copied()
            .ok_or_else(|| format!("operand {} of none", v.0))
    }
}

/// Operand `k`'s id.
fn arg(k: usize) -> ValueId {
    ValueId(u32::try_from(k).unwrap_or(u32::MAX))
}

impl<'v, L: FnMut(NodeId, &Node, &[u32]) -> Option<u32>> Evaluator<'v, L> {
    /// An evaluator of `values`, whose program flushes subnormal floats to zero where `flush_denormals` (the lifted
    /// program's float environment), asking `leaf(id, node, at)` for each leaf's word: `at` the words it is read at, a
    /// run-time indexed buffer word's index values (outermost first), a texel's coordinate then its controls or a
    /// query's operands, empty for the other leaves.
    pub fn new(values: &'v Values, flush_denormals: bool, leaf: L) -> Self {
        Evaluator {
            values,
            flush_denormals,
            leaf,
            known: BTreeMap::new(),
        }
    }

    /// The word node `id` evaluates to, or the leaf it needs that is unknown.
    pub fn evaluate(&mut self, id: NodeId) -> Result<Evaluated, String> {
        let mut stack = vec![id];
        while let Some(&top) = stack.last() {
            if self.known.contains_key(&top) {
                stack.pop();
                continue;
            }
            match self.step(top)? {
                Step::Done(v) => {
                    self.known.insert(top, v);
                    stack.pop();
                }
                // a node may be pushed again while an earlier push of it waits; the later one is evaluated first and
                // the earlier then found known. The graph's only cycles are loops' values, refused before their
                // operands are asked for, so the walk ends
                Step::Need(more) => stack.extend(more),
            }
        }
        self.known
            .get(&id)
            .copied()
            .ok_or_else(|| "the walk ended without the node's value".into())
    }

    /// The words of `ids`, or the first unknown leaf among them; `None` while some are not evaluated yet.
    fn words(&self, ids: &[NodeId]) -> Option<Result<Vec<u32>, NodeId>> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            match self.known.get(id)? {
                Evaluated::Word(w) => out.push(*w),
                Evaluated::Unknown(leaf) => return Some(Err(*leaf)),
            }
        }
        Some(Ok(out))
    }

    /// The operands of `ids` not yet evaluated.
    fn pending(&self, ids: &[NodeId]) -> Vec<NodeId> {
        ids.iter()
            .filter(|id| !self.known.contains_key(id))
            .copied()
            .collect()
    }

    /// A leaf read at the words of `at`: unknown where one of them is (the leaf itself is what is not known).
    fn leaf(&mut self, id: NodeId, node: &Node, at: &[NodeId]) -> Step {
        match self.words(at) {
            None => Step::Need(self.pending(at)),
            Some(Err(_)) => Step::Done(Evaluated::Unknown(id)),
            Some(Ok(words)) => Step::Done(match (self.leaf)(id, node, &words) {
                Some(w) => Evaluated::Word(w),
                None => Evaluated::Unknown(id),
            }),
        }
    }

    fn step(&mut self, id: NodeId) -> Result<Step, String> {
        let node = self
            .values
            .get(id)
            .ok_or_else(|| format!("node {id:?} is not of this graph"))?
            .clone();
        Ok(match &node {
            Node::Input { .. } | Node::Supplied { .. } | Node::Counter { .. } => self.leaf(id, &node, &[]),
            Node::Global { at, .. } => match at {
                At::Component(_) => self.leaf(id, &node, &[]),
                At::Indexed { index, .. } => self.leaf(id, &node, index),
            },
            Node::Texel {
                coordinate, controls, ..
            } => {
                let at: Vec<NodeId> = coordinate.iter().chain(controls).copied().collect();
                self.leaf(id, &node, &at)
            }
            Node::Query { operands, .. } => self.leaf(id, &node, operands),
            Node::Literal(l) => Step::Done(Evaluated::Word(literal(l.0, self.values.kind(id))?)),
            Node::Undef => return Err("an undefined value is not evaluated".into()),
            Node::Loop { .. } => return Err("a loop's value is not evaluated".into()),
            Node::Merge { arms, .. } => match self.words(arms) {
                None => Step::Need(self.pending(arms)),
                Some(Err(leaf)) => Step::Done(Evaluated::Unknown(leaf)),
                Some(Ok(words))
                    if let Some(&first) = words.first()
                        && words.iter().all(|w| *w == first) =>
                {
                    Step::Done(Evaluated::Word(first))
                }
                Some(Ok(_)) => {
                    return Err(
                        "a merge whose arms differ: the graph does not state which condition picks which"
                            .into(),
                    );
                }
            },
            Node::Op {
                op: Op::Select,
                operands,
            } => {
                let &[condition, accept, reject] = operands.as_slice() else {
                    return Err(format!("a select of {} operands", operands.len()));
                };
                match self.known.get(&condition) {
                    None => Step::Need(vec![condition]),
                    Some(Evaluated::Word(c)) => {
                        let chosen = if *c != 0 { accept } else { reject };
                        match self.known.get(&chosen) {
                            None => Step::Need(vec![chosen]),
                            Some(v) => Step::Done(*v),
                        }
                    }
                    // unknown which: known where both values are and are the same bits
                    Some(Evaluated::Unknown(leaf)) => {
                        let leaf = *leaf;
                        match self.words(&[accept, reject]) {
                            None => Step::Need(self.pending(&[accept, reject])),
                            Some(Ok(w))
                                if let &[a, b] = w.as_slice()
                                    && a == b =>
                            {
                                Step::Done(Evaluated::Word(a))
                            }
                            Some(_) => Step::Done(Evaluated::Unknown(leaf)),
                        }
                    }
                }
            }
            Node::Op { op, operands } => match self.words(operands) {
                None => Step::Need(self.pending(operands)),
                Some(Err(leaf)) => Step::Done(Evaluated::Unknown(leaf)),
                Some(Ok(w)) => Step::Done(Evaluated::Word(self.operation(id, *op, &w)?)),
            },
        })
    }

    /// Operation `op` of node `id` on the operand words `w`, by [`crate::eval`]: each operand a value of the type the
    /// operation reads in (a vector of the lanes `Any`, `All` and a dot product take), the result of the node's type.
    fn operation(&self, id: NodeId, op: Op, w: &[u32]) -> Result<u32, String> {
        let stated = |t: Option<Scalar>| {
            t.ok_or_else(|| format!("an operation (node {id:?}) the graph states no type for"))
        };
        let (reads, result) = (stated(self.values.reads(id))?, stated(self.values.kind(id))?);
        for s in [reads, result] {
            if s != Scalar::Bool && s.bits() != 32 {
                return Err(format!(
                    "a {} operation is not evaluated: only bools and 32-bit scalars are",
                    s.name()
                ));
            }
        }
        let scalars = |n: usize| Args {
            lanes: w.iter().map(|x| [*x, 0, 0, 0]).collect(),
            types: vec![Type::scalar(reads); n],
        };
        let count = |n: usize| {
            if w.len() == n {
                Ok(())
            } else {
                Err(format!("{op:?} of {} operands", w.len()))
            }
        };
        // a vector of every operand word
        let vector = || -> Result<Args, String> {
            let lanes = u8::try_from(w.len())
                .ok()
                .filter(|n| (1..=4).contains(n))
                .ok_or_else(|| format!("{op:?} of {} lanes", w.len()))?;
            let mut x = [0; 4];
            x.iter_mut().zip(w).for_each(|(o, v)| *o = *v);
            Ok(Args {
                lanes: vec![x],
                types: vec![Type::vector(reads, lanes)],
            })
        };
        let (ir, args) = match op {
            Op::Unary(Unary::Derivative(..)) => return Err("a derivative is not evaluated".into()),
            Op::Unary(u @ (Unary::Any | Unary::All)) => (iridia_ir::Op::Unary(u, arg(0)), vector()?),
            Op::Unary(u) => {
                count(1)?;
                (iridia_ir::Op::Unary(u, arg(0)), scalars(1))
            }
            Op::Binary(b) => {
                count(2)?;
                (iridia_ir::Op::Binary(b, arg(0), arg(1)), scalars(2))
            }
            Op::Ternary(t) => {
                count(3)?;
                (iridia_ir::Op::Ternary(t, arg(0), arg(1), arg(2)), scalars(3))
            }
            Op::BitInsert => {
                count(4)?;
                let mut args = scalars(4);
                // the offset and count are `u32`s whatever the base's type
                args.types.iter_mut().skip(2).for_each(|t| *t = Type::U32);
                let op = iridia_ir::Op::BitInsert {
                    base: arg(0),
                    insert: arg(1),
                    offset: arg(2),
                    count: arg(3),
                };
                (op, args)
            }
            Op::Select => {
                count(3)?;
                let mut args = scalars(3);
                if let Some(t) = args.types.first_mut() {
                    *t = Type::BOOL;
                }
                let op = iridia_ir::Op::Select {
                    cond: arg(0),
                    accept: arg(1),
                    reject: arg(2),
                };
                (op, args)
            }
            Op::Dot => {
                let n = w.len() / 2;
                let dot = || format!("a dot product of {} operands", w.len());
                if !w.len().is_multiple_of(2) || !(1..=4).contains(&n) {
                    return Err(dot());
                }
                let (a, b) = w.split_at_checked(n).ok_or_else(dot)?;
                let lanes = u8::try_from(n).map_err(|_| dot())?;
                let reg = |v: &[u32]| {
                    let mut x = [0; 4];
                    x.iter_mut().zip(v).for_each(|(o, v)| *o = *v);
                    x
                };
                let args = Args {
                    lanes: vec![reg(a), reg(b)],
                    types: vec![Type::vector(reads, lanes); 2],
                };
                (iridia_ir::Op::Dot(arg(0), arg(1)), args)
            }
            Op::Convert(rule) => {
                count(1)?;
                (iridia_ir::Op::Convert(arg(0), rule), scalars(1))
            }
            Op::Index => {
                return Err(
                    "an element picked by a run-time index: the graph does not state which candidate an index picks"
                        .into(),
                );
            }
            Op::Join | Op::Part(_) => {
                return Err("a lane of a bitcast between widths is not evaluated".into());
            }
            Op::Wave(_) => return Err("a wave operation is not evaluated".into()),
            Op::Atomic(_) => return Err("an atomic's result is not evaluated".into()),
        };
        let lanes = crate::eval(&ir, Type::scalar(result), &args, self.flush_denormals)?
            .ok_or_else(|| format!("{op:?} is not a pure operation"))?;
        Ok(lanes[0])
    }
}

/// A literal's word: its value as the scalar type the graph states it of.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // each is checked to hold the value exactly
fn literal(v: f64, kind: Option<Scalar>) -> Result<u32, String> {
    let exact = |w: u32, back: f64| {
        if back.to_bits() == v.to_bits() || (v.is_nan() && back.is_nan()) {
            Ok(w)
        } else {
            Err(format!("literal {v} is no 32-bit {kind:?}"))
        }
    };
    match kind {
        Some(Scalar::F32) => {
            let f = v as f32;
            exact(f.to_bits(), f64::from(f))
        }
        Some(Scalar::U32) => exact(v as u32, f64::from(v as u32)),
        Some(Scalar::I32) => exact((v as i32) as u32, f64::from(v as i32)),
        Some(Scalar::Bool) => Ok(u32::from(v != 0.)),
        other => Err(format!("a literal of type {other:?}")),
    }
}
