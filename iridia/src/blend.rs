//! Layers a program lays over one another, read from its value graph ([`Values`]) with no game in it: a lane that is
//! a chain of interpolations, each towards a layer's value by a weight ([`Over`]), and any scalar value as the float
//! operations it is made of ([`Scalar`]), down to the nodes the caller says what they are of (an input lane, a texel,
//! a constant word).
//!
//! A [`Scalar`] means what the program computes, operation by operation: each of its terms is one float operation of
//! the program rounded to `f32` on its own, in the program's order, nothing fused or reassociated (the lifters write
//! Direct3D's `mad` as a product and then a sum). Direct3D's results that the lifters spell out ([`crate::D3dModes`])
//! are read back as the one operation they spell: a saturate guarded against NaN is [`Term::Saturate`] (NaN gives 0),
//! a `min` or `max` guarded on each operand is [`Term::Min`] or [`Term::Max`] (a NaN operand gives the other). Exact
//! rewrites only: `a + -b` is `a - b`, `-a` is `-1 * a`, a select by a literal condition is the arm it selects. An
//! unguarded naga `saturate`, `min` or `max`, which leaves NaN to the implementation, and every other operation is a
//! [`Term::Leaf`].
use std::collections::{BTreeMap, BTreeSet};

use crate::flow::{
    BinaryOperator as B, MathFunction as M, Node, NodeId, Op, ScalarKind, UnaryOperator as U, Values,
};

/// One operation of a [`Scalar`]; operands index earlier terms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Term {
    /// A node no term spells, its lane's bits read as an `f32`: what it is (an input lane, a texel, a constant word,
    /// an operation outside the terms) is the caller's to say.
    Leaf(NodeId),
    /// A literal's value: a float literal's, or an integer literal's bits where a float operation reads them.
    Constant(f32),
    /// `a + b`.
    Add([usize; 2]),
    /// `a - b`.
    Subtract([usize; 2]),
    /// `a * b`.
    Multiply([usize; 2]),
    /// The lesser; a NaN operand gives the other.
    Min([usize; 2]),
    /// The greater; a NaN operand gives the other.
    Max([usize; 2]),
    /// Clamped to 0 to 1; NaN gives 0.
    Saturate(usize),
}

/// A scalar value of a graph as the float operations it is made of ([`Term`]), each once, in an order where operands
/// come first; the last term is the value.
#[derive(Clone, Debug, PartialEq)]
pub struct Scalar {
    /// The terms; the last is the value.
    pub terms: Vec<Term>,
}

impl Scalar {
    /// Node `node` of `values` as its terms, down to the nodes in `leaves` (which are [`Term::Leaf`]s whatever they
    /// are made of) and those no term spells.
    pub fn of(values: &Values, node: NodeId, leaves: &BTreeSet<NodeId>) -> Scalar {
        Scalar {
            terms: Reader::read(values, node, leaves).terms,
        }
    }

    /// The nodes its leaves are, each once, in the order of the terms.
    pub fn leaves(&self) -> Vec<NodeId> {
        let mut seen = BTreeSet::new();
        self.terms
            .iter()
            .filter_map(|t| match t {
                Term::Leaf(n) if seen.insert(*n) => Some(*n),
                _ => None,
            })
            .collect()
    }

    /// Its value given each leaf's ([`Term::Leaf`]), under the terms' rules; NaN where a leaf has no value or an
    /// operand index is not an earlier term's.
    pub fn evaluate(&self, leaf: impl Fn(NodeId) -> Option<f32>) -> f32 {
        let mut done: Vec<f32> = Vec::with_capacity(self.terms.len());
        for t in &self.terms {
            let at = |i: &usize| done.get(*i).copied().unwrap_or(f32::NAN);
            let v = match t {
                Term::Leaf(n) => leaf(*n).unwrap_or(f32::NAN),
                Term::Constant(c) => *c,
                Term::Add([a, b]) => at(a) + at(b),
                Term::Subtract([a, b]) => at(a) - at(b),
                Term::Multiply([a, b]) => at(a) * at(b),
                Term::Min([a, b]) => at(a).min(at(b)),
                Term::Max([a, b]) => at(a).max(at(b)),
                Term::Saturate(a) => {
                    let x = at(a);
                    if x.is_nan() { 0.0 } else { x.clamp(0.0, 1.0) }
                }
            };
            done.push(v);
        }
        done.last().copied().unwrap_or(f32::NAN)
    }
}

struct Reader<'a> {
    values: &'a Values,
    leaves: &'a BTreeSet<NodeId>,
    terms: Vec<Term>,
    /// (node, read as a float operation's operand) -> its term
    made: BTreeMap<(NodeId, bool), usize>,
    /// The nodes each node's term was made of, in the order it read them.
    read: BTreeMap<NodeId, Vec<NodeId>>,
    /// The node whose term is being made.
    reading: Option<NodeId>,
}

impl<'a> Reader<'a> {
    /// `node`'s terms, down to `leaves`.
    fn read(values: &'a Values, node: NodeId, leaves: &'a BTreeSet<NodeId>) -> Reader<'a> {
        let mut reader = Reader {
            values,
            leaves,
            terms: Vec::new(),
            made: BTreeMap::new(),
            read: BTreeMap::new(),
            reading: None,
        };
        reader.term(node, true);
        reader
    }

    /// The term of `node`; `float` where a float operation reads it (an integer literal's bits are then its value).
    fn term(&mut self, node: NodeId, float: bool) -> usize {
        if let Some(parent) = self.reading {
            self.read.entry(parent).or_default().push(node);
        }
        if let Some(&t) = self.made.get(&(node, float)) {
            return t;
        }
        let parent = self.reading.replace(node);
        let spelled = self.spell(node, float);
        self.reading = parent;
        let term = match spelled {
            Spelled::Term(t) => t,
            Spelled::Same(other) => {
                let parent = self.reading.replace(node);
                let t = self.term(other, float);
                self.reading = parent;
                self.made.insert((node, float), t);
                return t;
            }
        };
        let index = self.terms.len();
        self.terms.push(term);
        self.made.insert((node, float), index);
        index
    }

    fn spell(&mut self, node: NodeId, float: bool) -> Spelled {
        let v = self.values;
        if self.leaves.contains(&node) {
            return Spelled::Term(Term::Leaf(node));
        }
        if let Node::Literal(l) = v.node(node) {
            return Spelled::Term(match (v.kind(node), float) {
                (Some(ScalarKind::Float | ScalarKind::AbstractFloat), _) => {
                    single(l.0).map_or(Term::Leaf(node), Term::Constant)
                }
                (Some(ScalarKind::Uint | ScalarKind::Sint | ScalarKind::AbstractInt), true) => {
                    bits(l.0).map_or(Term::Leaf(node), |b| Term::Constant(f32::from_bits(b)))
                }
                _ => Term::Leaf(node),
            });
        }
        let Node::Op { op, operands } = v.node(node) else {
            return Spelled::Term(Term::Leaf(node));
        };
        if v.kind(node) != Some(ScalarKind::Float) {
            return Spelled::Term(Term::Leaf(node));
        }
        match (op, operands.as_slice()) {
            (Op::Binary(B::Add), &[a, b]) => {
                // a + -b is a - b, -a + b is b - a: the same rounding, the same zero
                if let Some(nb) = negated(v, b) {
                    return Spelled::Term(Term::Subtract([self.term(a, true), self.term(nb, true)]));
                }
                if let Some(na) = negated(v, a) {
                    return Spelled::Term(Term::Subtract([self.term(b, true), self.term(na, true)]));
                }
                Spelled::Term(Term::Add([self.term(a, true), self.term(b, true)]))
            }
            (Op::Binary(B::Subtract), &[a, b]) => {
                Spelled::Term(Term::Subtract([self.term(a, true), self.term(b, true)]))
            }
            (Op::Binary(B::Multiply), &[a, b]) => {
                Spelled::Term(Term::Multiply([self.term(a, true), self.term(b, true)]))
            }
            (Op::Unary(U::Negate), &[a]) => {
                let minus_one = self.constant(-1.0);
                Spelled::Term(Term::Multiply([minus_one, self.term(a, true)]))
            }
            (Op::Select, &[condition, accept, reject]) => {
                if let Node::Literal(c) = v.node(condition) {
                    return Spelled::Same(if c.0 != 0.0 { accept } else { reject });
                }
                if let Some(x) = guarded_saturate(v, condition, accept, reject) {
                    return Spelled::Term(Term::Saturate(self.term(x, true)));
                }
                match guarded_min_max(v, condition, accept, reject) {
                    Some((M::Min, a, b)) => {
                        Spelled::Term(Term::Min([self.term(a, true), self.term(b, true)]))
                    }
                    Some((_, a, b)) => Spelled::Term(Term::Max([self.term(a, true), self.term(b, true)])),
                    None => Spelled::Term(Term::Leaf(node)),
                }
            }
            _ => Spelled::Term(Term::Leaf(node)),
        }
    }

    fn constant(&mut self, c: f32) -> usize {
        self.terms.push(Term::Constant(c));
        self.terms.len().saturating_sub(1)
    }
}

enum Spelled {
    Term(Term),
    /// The same value as another node (a select by a literal condition).
    Same(NodeId),
}

/// A float literal's value as an `f32`, `None` where it has more precision than one holds.
#[allow(clippy::cast_possible_truncation)] // the rounding is checked: the f32 must be the literal
fn single(v: f64) -> Option<f32> {
    let c = v as f32;
    (f64::from(c).to_bits() == v.to_bits() || v.is_nan()).then_some(c)
}

/// The 32 bits an integer literal's value is, `None` past them.
#[allow(clippy::cast_possible_truncation)] // whole and in range: checked before each cast
fn bits(v: f64) -> Option<u32> {
    if v.fract() != 0.0 {
        return None;
    }
    if (0.0..=f64::from(u32::MAX)).contains(&v) {
        Some(v as u32)
    } else if (f64::from(i32::MIN)..0.0).contains(&v) {
        Some((v as i32).cast_unsigned())
    } else {
        None
    }
}

/// `x` where `node` is a float `-x`.
fn negated(v: &Values, node: NodeId) -> Option<NodeId> {
    match v.node(node) {
        Node::Op {
            op: Op::Unary(U::Negate),
            operands,
        } if v.kind(node) == Some(ScalarKind::Float) => match operands[..] {
            [x] => Some(x),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `node` is the literal `value` (its bits as an integer literal's).
fn literal_is(v: &Values, node: NodeId, value: f64) -> bool {
    matches!(v.node(node), Node::Literal(l) if l.0 == value)
}

/// `x` where `condition` tests `x` for NaN on its bits, as the lifters write it: `(bits(x) & 0x7fffffff) >
/// 0x7f800000`.
fn nan_test(v: &Values, condition: NodeId) -> Option<NodeId> {
    let Node::Op {
        op: Op::Binary(B::Greater),
        operands,
    } = v.node(condition)
    else {
        return None;
    };
    let [magnitude, infinity] = operands[..] else {
        return None;
    };
    if !literal_is(v, infinity, f64::from(0x7f80_0000u32)) {
        return None;
    }
    let Node::Op {
        op: Op::Binary(B::And),
        operands,
    } = v.node(magnitude)
    else {
        return None;
    };
    if v.kind(magnitude) != Some(ScalarKind::Uint) {
        return None;
    }
    match operands[..] {
        [x, mask] if literal_is(v, mask, f64::from(0x7fff_ffffu32)) => Some(x),
        [mask, x] if literal_is(v, mask, f64::from(0x7fff_ffffu32)) => Some(x),
        _ => None,
    }
}

/// `x` where `select(reject, accept, condition)` is Direct3D's saturate of `x`: 0 where `x` is NaN, else naga's
/// saturate of it.
fn guarded_saturate(v: &Values, condition: NodeId, accept: NodeId, reject: NodeId) -> Option<NodeId> {
    let x = nan_test(v, condition)?;
    if !literal_is(v, accept, 0.0) {
        return None;
    }
    match v.node(reject) {
        Node::Op {
            op: Op::Math(M::Saturate),
            operands,
        } if v.kind(reject) == Some(ScalarKind::Float) && operands[..] == [x] => Some(x),
        _ => None,
    }
}

/// (min or max, a, b) where `select(reject, accept, condition)` is Direct3D's `min` or `max` of `a` and `b`: `b`
/// where `a` is NaN, else `a` where `b` is NaN, else naga's of them.
fn guarded_min_max(
    v: &Values,
    condition: NodeId,
    accept: NodeId,
    reject: NodeId,
) -> Option<(M, NodeId, NodeId)> {
    let a = nan_test(v, condition)?;
    let b = accept;
    let Node::Op {
        op: Op::Select,
        operands,
    } = v.node(reject)
    else {
        return None;
    };
    let [inner_condition, inner_accept, plain] = operands[..] else {
        return None;
    };
    if nan_test(v, inner_condition)? != b || inner_accept != a {
        return None;
    }
    match v.node(plain) {
        Node::Op {
            op: Op::Math(f @ (M::Min | M::Max)),
            operands,
        } if v.kind(plain) == Some(ScalarKind::Float) && operands[..] == [a, b] => Some((*f, a, b)),
        _ => None,
    }
}

/// A layer of an [`Over`]: laid over what is below it by its weight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layer {
    /// The weight, 0 leaving what is below, 1 the layer.
    pub weight: NodeId,
    /// The layer's value.
    pub value: NodeId,
    /// The interpolation's own node: what is below with the layer laid over it.
    pub node: NodeId,
}

/// A lane made of layers laid over one another: `base`, then each of `layers` (bottom first) over what is below it,
/// `below + weight * (layer - below)`. The program spells each as `weight * (layer - below) + below`, its operands in
/// either order and the difference also as `layer + -below`, or as naga's `mix(below, layer, weight)`. The chain is
/// the structure only: how the program rounds each step is its nodes' ([`Scalar::of`] reads them).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Over {
    /// What the bottom layer is laid over.
    pub base: NodeId,
    /// The layers, bottom first.
    pub layers: Vec<Layer>,
}

impl Over {
    /// The chain whose top is `node`: `None` where `node` is no interpolation.
    pub fn at(values: &Values, node: NodeId) -> Option<Over> {
        let mut layers = Vec::new();
        let mut at = node;
        while let Some((below, layer)) = interpolation(values, at) {
            layers.push(layer);
            at = below;
        }
        if layers.is_empty() {
            return None;
        }
        layers.reverse();
        Some(Over { base: at, layers })
    }

    /// The first chain under `lane` (an output lane's node): the lane itself, or one the lane is made of through the
    /// operations a [`Scalar`] spells (a saturate, a product, a sum), operands in order, first found first.
    pub fn under(values: &Values, lane: NodeId) -> Option<Over> {
        let leaves = BTreeSet::new();
        let reader = Reader::read(values, lane, &leaves);
        let mut seen = BTreeSet::new();
        let mut pending = vec![lane];
        while let Some(n) = pending.pop() {
            if !seen.insert(n) {
                continue;
            }
            if let Some(over) = Over::at(values, n) {
                return Some(over);
            }
            // the nodes its term was made of, last pushed first popped
            if let Some(next) = reader.read.get(&n) {
                pending.extend(next.iter().rev());
            }
        }
        None
    }
}

/// (below, the layer) where `node` is one interpolation of an [`Over`] chain; `None` where it is none, or where its
/// operands could be read as more than one.
fn interpolation(v: &Values, node: NodeId) -> Option<(NodeId, Layer)> {
    if v.kind(node) != Some(ScalarKind::Float) {
        return None;
    }
    let Node::Op { op, operands } = v.node(node) else {
        return None;
    };
    match (op, operands.as_slice()) {
        (Op::Math(M::Mix), &[below, value, weight]) => Some((below, Layer { weight, value, node })),
        (Op::Binary(B::Add), &[p, q]) => {
            let mut found = None;
            for (product, below) in [(p, q), (q, p)] {
                let Node::Op {
                    op: Op::Binary(B::Multiply),
                    operands,
                } = v.node(product)
                else {
                    continue;
                };
                if v.kind(product) != Some(ScalarKind::Float) {
                    continue;
                }
                let [x, y] = operands[..] else {
                    continue;
                };
                for (difference, weight) in [(x, y), (y, x)] {
                    if let Some(value) = difference_from(v, difference, below) {
                        if found.is_some() {
                            return None;
                        }
                        found = Some((below, Layer { weight, value, node }));
                    }
                }
            }
            found
        }
        _ => None,
    }
}

/// `layer` where `node` is the float `layer - below` or `layer + -below`.
fn difference_from(v: &Values, node: NodeId, below: NodeId) -> Option<NodeId> {
    if v.kind(node) != Some(ScalarKind::Float) {
        return None;
    }
    match v.node(node) {
        Node::Op {
            op: Op::Binary(B::Subtract),
            operands,
        } => match operands[..] {
            [layer, b] if b == below => Some(layer),
            _ => None,
        },
        Node::Op {
            op: Op::Binary(B::Add),
            operands,
        } => match operands[..] {
            [layer, nb] if negated(v, nb) == Some(below) => Some(layer),
            [nb, layer] if negated(v, nb) == Some(below) => Some(layer),
            _ => None,
        },
        _ => None,
    }
}
