//! Why a compute program may run out of the executor's step budget: whether a loop's exit is decided by a word of a constant
//! buffer, or by data read from another buffer or an image. The harness seeds both, and a word used as a count there is
//! whatever the seed made it; a program whose loops end on fixed values does not depend on the seed.
use std::collections::BTreeSet;

use iridia_ir::interface::ResourceId;
use iridia_ir::{Body, End, Function, LabelId, Node, Op, Region};

/// The loops of a function.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Facts {
    /// How many loops it has.
    pub loops: usize,
    /// How many of them leave on a condition computed from a word of a constant buffer.
    pub constant_exits: usize,
    /// How many leave on a condition computed from data read from a buffer or an image (and not from a constant buffer).
    pub data_exits: usize,
}

/// Where a value comes from: a word of a constant buffer, or a read of any other buffer or image.
const CONSTANT: u8 = 1;
const DATA: u8 = 2;

struct Analysis<'f> {
    function: &'f Function,
    constants: &'f BTreeSet<ResourceId>,
    /// Per value: what it is computed from ([`CONSTANT`], [`DATA`]).
    from: Vec<u8>,
}

impl Analysis<'_> {
    /// The values in definition order: a load of a constant buffer, and what is computed from one. A loop's parameters and
    /// a construct's results are set by [`Analysis::carry`].
    fn define(&mut self) -> bool {
        let mut changed = false;
        for (i, def) in self.function.values.iter().enumerate() {
            let operands = |op: &Op| op.operands().iter().fold(0, |acc, v| acc | self.flags(*v));
            let now = match &def.op {
                Op::LoadBuffer { resource, .. } if self.constants.contains(&resource.id) => CONSTANT,
                Op::LoadBuffer { .. } | Op::LoadTexel { .. } | Op::Sample { .. } | Op::Gather { .. } => {
                    DATA | operands(&def.op)
                }
                Op::Param => 0,
                op => operands(op),
            };
            if self.from[i] | now != self.from[i] {
                self.from[i] |= now;
                changed = true;
            }
        }
        changed
    }

    fn flags(&self, v: iridia_ir::ValueId) -> u8 {
        self.from.get(v.0 as usize).copied().unwrap_or(0)
    }

    /// The value `to` also comes from what `from` comes from.
    fn pass(&mut self, from: iridia_ir::ValueId, to: iridia_ir::ValueId) -> bool {
        let add = self.flags(from);
        match self.from.get_mut(to.0 as usize) {
            Some(slot) if *slot | add != *slot => {
                *slot |= add;
                true
            }
            _ => false,
        }
    }

    /// Loop parameters take their entry values and the values each `continue` hands them; a construct's results take the
    /// values its arms yield.
    fn carry(&mut self, region: &Region) -> bool {
        let mut changed = false;
        for node in &region.nodes {
            match node {
                Node::Loop {
                    label,
                    params,
                    inits,
                    body,
                    ..
                } => {
                    for (p, i) in params.iter().zip(inits) {
                        changed |= self.pass(*i, *p);
                    }
                    changed |= self.continues(*label, params, body);
                }
                Node::If {
                    accept,
                    reject,
                    results,
                    ..
                } => {
                    for arm in [accept, reject] {
                        if let Some(End::Yield(values)) = &arm.end {
                            for (r, v) in results.iter().zip(values) {
                                changed |= self.pass(*v, *r);
                            }
                        }
                    }
                }
                _ => {}
            }
            for inner in node.regions() {
                changed |= self.carry(inner);
            }
        }
        changed
    }

    /// The parameters of the loop `label` that a `continue` in `region` hands a value.
    fn continues(&mut self, label: LabelId, params: &[iridia_ir::ValueId], region: &Region) -> bool {
        let mut changed = false;
        if let Some(End::Continue(l, values)) = &region.end
            && *l == label
        {
            for (p, v) in params.iter().zip(values) {
                changed |= self.pass(*v, *p);
            }
        }
        for node in &region.nodes {
            for inner in node.regions() {
                changed |= self.continues(label, params, inner);
            }
        }
        changed
    }

    /// What the conditions of the `if`s in `region` that have an arm leaving the loop `label` come from.
    fn exit_flags(&self, label: LabelId, region: &Region) -> u8 {
        region.nodes.iter().fold(0, |acc, node| {
            let here = match node {
                Node::If {
                    cond, accept, reject, ..
                } if [accept, reject]
                    .iter()
                    .any(|arm| matches!(&arm.end, Some(End::Br(l, _)) if *l == label)) =>
                {
                    self.flags(*cond)
                }
                _ => 0,
            };
            node.regions()
                .iter()
                .fold(acc | here, |acc, r| acc | self.exit_flags(label, r))
        })
    }

    fn count(&self, region: &Region, facts: &mut Facts) {
        for node in &region.nodes {
            if let Node::Loop { label, body, .. } = node {
                facts.loops += 1;
                let exit = self.exit_flags(*label, body);
                if exit & CONSTANT != 0 {
                    facts.constant_exits += 1;
                } else if exit & DATA != 0 {
                    facts.data_exits += 1;
                }
            }
            for inner in node.regions() {
                self.count(inner, facts);
            }
        }
    }
}

/// The loops of `function` and how many leave on a word of one of the constant buffers `constants`, or on data read from
/// another buffer or an image. A function in blocks
/// rather than regions is not analysed (no loops are counted).
pub(super) fn facts(function: &Function, constants: &BTreeSet<ResourceId>) -> Facts {
    let Body::Regions(region) = &function.body else {
        return Facts::default();
    };
    let mut analysis = Analysis {
        function,
        constants,
        from: vec![0; function.values.len()],
    };
    // a value's origin can travel round a loop, so go on until nothing changes
    while analysis.define() | analysis.carry(region) {}
    let mut facts = Facts::default();
    analysis.count(region, &mut facts);
    facts
}

#[cfg(test)]
mod tests {
    use super::*;
    use iridia_ir::interface::ResourceId;
    use iridia_ir::{Address, Binary, Cmp, OutOfBounds, Precision, Res, Scalar, Type, ValueDef, ValueId};

    fn def(op: Op, scalar: Scalar) -> ValueDef {
        ValueDef {
            ty: Type::scalar(scalar),
            op,
            origin: None,
            precision: Precision::Full,
        }
    }

    /// `i = start; loop { if i >= limit { exit } i += 1 }` where `limit` is a word of buffer 0 or a literal.
    fn counting_loop(limit_from_buffer: bool) -> Function {
        let label = LabelId(0);
        let values = vec![
            def(Op::Literal([0; 4]), Scalar::U32), // 0 start
            def(Op::Param, Scalar::U32),           // 1 i
            if limit_from_buffer {
                def(
                    Op::LoadBuffer {
                        resource: Res::of(ResourceId(0)),
                        address: Address::fixed(0),
                        oob: OutOfBounds::Zero,
                    },
                    Scalar::U32,
                )
            } else {
                def(Op::Literal([8, 0, 0, 0]), Scalar::U32)
            }, // 2 limit
            def(
                Op::Binary(Binary::UCmp(Cmp::Ge), ValueId(1), ValueId(2)),
                Scalar::Bool,
            ), // 3 i >= limit
            def(Op::Literal([1, 0, 0, 0]), Scalar::U32), // 4 one
            def(Op::Binary(Binary::IAdd, ValueId(1), ValueId(4)), Scalar::U32), // 5 i + 1
        ];
        let body = Region {
            nodes: vec![
                Node::Value(ValueId(2)),
                Node::Value(ValueId(3)),
                Node::If {
                    cond: ValueId(3),
                    accept: Region {
                        nodes: Vec::new(),
                        end: Some(End::Br(label, Vec::new())),
                    },
                    reject: Region::default(),
                    results: Vec::new(),
                },
                Node::Value(ValueId(4)),
                Node::Value(ValueId(5)),
            ],
            end: Some(End::Continue(label, vec![ValueId(5)])),
        };
        Function {
            name: "main".into(),
            values,
            vars: Vec::new(),
            queries: Vec::new(),
            body: Body::Regions(Region {
                nodes: vec![
                    Node::Value(ValueId(0)),
                    Node::Loop {
                        label,
                        params: vec![ValueId(1)],
                        inits: vec![ValueId(0)],
                        body,
                        results: Vec::new(),
                    },
                ],
                end: Some(End::Return),
            }),
        }
    }

    #[test]
    fn a_loop_that_leaves_on_a_constant_buffer_word_is_found() {
        let constants = BTreeSet::from([ResourceId(0)]);
        assert_eq!(
            facts(&counting_loop(true), &constants),
            Facts {
                loops: 1,
                constant_exits: 1,
                data_exits: 0
            }
        );
        assert_eq!(
            facts(&counting_loop(false), &constants),
            Facts {
                loops: 1,
                constant_exits: 0,
                data_exits: 0
            }
        );
    }
}
