//! The shape of a compute program's group-shared memory use, which the `COMPUTE_EXPECTED` rows for programs that race on it
//! state: a row excusing a program for reading shared memory across threads with no barrier, or for reading and writing it in
//! one barrier phase, is kept only while the lifted IR still has that shape. A source hash does not do this: a lift that
//! moved a barrier would leave the hash as it was.
//!
//! A phase ends only at a barrier the invocations wait at (`execution`): a workgroup-only fence orders the invocation's own
//! memory accesses but makes no other invocation wait, so it is counted apart (`fences`) and ends no phase.
//!
//! The shape is a necessary condition for each race, not a proof of it (an invocation's own `g[tid] += x` also reads and then
//! writes in one phase, and an unsynchronised read of its own word is no race): the proof stays in each row's evidence.
use std::collections::BTreeSet;

use iridia_ir::{Body, Function, Node, Op, Region, Space, VarId};

/// What a function does with its workgroup variables.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Shape {
    /// Barriers the workgroup's invocations all wait at (`execution`): each ends a phase.
    pub barriers: usize,
    /// Memory fences with no execution barrier (`workgroup` only): they order an invocation's accesses, end no phase.
    pub fences: usize,
    /// Reads of a workgroup variable.
    pub loads: usize,
    /// Writes of a workgroup variable.
    pub stores: usize,
    /// Some workgroup variable is read and then written with no barrier between: an invocation's write can land
    /// before another's read of the same variable in one phase.
    pub read_then_write_in_a_phase: bool,
}

impl Shape {
    /// Group-shared memory is written and read, and no execution barrier orders them (a fence alone does not).
    pub fn unsynchronised(&self) -> bool {
        self.barriers == 0 && self.loads > 0 && self.stores > 0
    }

    /// An execution barrier exists, yet some variable is read and then written within one phase.
    pub fn races_within_a_phase(&self) -> bool {
        self.barriers > 0 && self.read_then_write_in_a_phase
    }
}

struct Walk<'f> {
    function: &'f Function,
    shape: Shape,
    /// The workgroup variables read since the last execution barrier.
    read: BTreeSet<VarId>,
}

impl Walk<'_> {
    fn shared(&self, var: VarId) -> bool {
        self.function
            .vars
            .get(var.0 as usize)
            .is_some_and(|v| v.space == Space::Workgroup)
    }

    fn region(&mut self, region: &Region) {
        for node in &region.nodes {
            match node {
                Node::Barrier(b) if b.execution => {
                    self.shape.barriers += 1;
                    self.read.clear();
                }
                Node::Barrier(b) if b.workgroup => self.shape.fences += 1,
                Node::Value(v) => {
                    if let Some(def) = self.function.values.get(v.0 as usize)
                        && let Op::LoadVar(var) | Op::LoadElement(var, ..) = def.op
                        && self.shared(var)
                    {
                        self.shape.loads += 1;
                        self.read.insert(var);
                    }
                }
                Node::StoreVar(var, _) | Node::StoreElement(var, ..) if self.shared(*var) => {
                    self.shape.stores += 1;
                    if self.read.contains(var) {
                        self.shape.read_then_write_in_a_phase = true;
                    }
                }
                _ => {}
            }
            for inner in node.regions() {
                self.region(inner);
            }
        }
    }
}

/// The shape of `function`'s use of workgroup variables, its nodes taken in order (a branch's arms one after the other).
pub(super) fn shape(function: &Function) -> Shape {
    let mut walk = Walk {
        function,
        shape: Shape::default(),
        read: BTreeSet::new(),
    };
    match &function.body {
        Body::Regions(region) => walk.region(region),
        // a function in blocks has no nested regions to walk in order: say nothing of its shape
        Body::Cfg(_) => {}
    }
    walk.shape
}

#[cfg(test)]
mod tests {
    use super::*;
    use iridia_ir::{Barrier, OutOfBounds, Precision, Type, ValueDef, ValueId, Var};

    /// `loads` shared reads and `stores` shared writes of variable 0, in the order of `steps` ('r' read, 'w' write, 'b' execution barrier, 'f' workgroup fence only).
    fn function(steps: &str) -> Function {
        let mut shared = Var::new("g0", Type::scalar(iridia_ir::Scalar::U32));
        shared.space = Space::Workgroup;
        shared.len = Some(8);
        let mut values = vec![ValueDef {
            ty: Type::scalar(iridia_ir::Scalar::U32),
            op: Op::Literal([0; 4]),
            origin: None,
            precision: Precision::Full,
        }];
        let mut nodes = Vec::new();
        for step in steps.chars() {
            match step {
                'r' => {
                    values.push(ValueDef {
                        ty: Type::scalar(iridia_ir::Scalar::U32),
                        op: Op::LoadElement(VarId(0), ValueId(0), OutOfBounds::Zero),
                        origin: None,
                        precision: Precision::Full,
                    });
                    nodes.push(Node::Value(ValueId(values.len() as u32 - 1)));
                }
                'w' => nodes.push(Node::StoreElement(
                    VarId(0),
                    ValueId(0),
                    ValueId(0),
                    OutOfBounds::Zero,
                )),
                'b' => nodes.push(Node::Barrier(Barrier {
                    execution: true,
                    workgroup: true,
                    storage: None,
                })),
                'f' => nodes.push(Node::Barrier(Barrier {
                    execution: false,
                    workgroup: true,
                    storage: None,
                })),
                other => panic!("no step {other}"),
            }
        }
        Function {
            name: "main".into(),
            values,
            vars: vec![shared],
            queries: Vec::new(),
            body: Body::Regions(Region {
                nodes,
                end: Some(iridia_ir::End::Return),
            }),
        }
    }

    #[test]
    fn shared_memory_with_no_barrier_is_unsynchronised() {
        let s = shape(&function("wrr"));
        assert!(s.unsynchronised());
        assert!(!s.races_within_a_phase());
        // moving a barrier between the write and the reads is a different shape
        assert!(!shape(&function("wbrr")).unsynchronised());
    }

    #[test]
    fn a_scan_step_reads_then_writes_in_one_phase() {
        // read, write, barrier, again: the in-place scan's steps
        assert!(shape(&function("rwbrwb")).races_within_a_phase());
        // a barrier between each read and write ends the race
        assert!(!shape(&function("rbwbrbwb")).races_within_a_phase());
        // writes then reads in a phase, as the first step of a staged copy, is not a read then write
        assert!(!shape(&function("wbrb")).races_within_a_phase());
    }

    #[test]
    fn a_fence_alone_ends_no_phase() {
        // a workgroup fence orders memory but no invocation waits at it: the writes and reads still race
        let s = shape(&function("wfrr"));
        assert!(s.unsynchronised());
        assert_eq!((s.barriers, s.fences), (0, 1));
        // with fences only the program is unsynchronised, and with an execution barrier somewhere a fence still splits nothing
        assert!(shape(&function("rfwfrfwf")).unsynchronised());
        assert!(shape(&function("rfwbrfwb")).races_within_a_phase());
        // an execution barrier between them does
        assert!(!shape(&function("rbwbrbwb")).races_within_a_phase());
    }
}
