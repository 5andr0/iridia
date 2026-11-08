//! The shape of a compute program every thread group runs the same way over buffer words it both reads and writes, which a
//! `WARP_UNSTABLE` row for a cross-group race states: with several groups dispatched and no group number read, each group does
//! the same work on the same words, so one group's stores land among another's loads (the executor runs the groups one after
//! the other and WARP does not). The shape is a necessary condition for the race, not a proof of it: the proof stays in the
//! row's evidence (the words differ between repeats at the dispatched groups and not at one).
use std::collections::BTreeSet;

use iridia_ir::interface::ResourceId;
use iridia_ir::{Body, Function, Node, Op, Region};

/// Whether `function` loads and stores one buffer.
pub(super) fn reads_and_writes_a_buffer(function: &Function) -> bool {
    fn region(
        function: &Function,
        r: &Region,
        loaded: &mut BTreeSet<ResourceId>,
        stored: &mut BTreeSet<ResourceId>,
    ) {
        for node in &r.nodes {
            match node {
                Node::Value(v) => {
                    if let Some(def) = function.values.get(v.0 as usize)
                        && let Op::LoadBuffer { resource, .. } = &def.op
                    {
                        loaded.insert(resource.id);
                    }
                }
                Node::StoreBuffer { resource, .. } => {
                    stored.insert(resource.id);
                }
                _ => {}
            }
            for inner in node.regions() {
                region(function, inner, loaded, stored);
            }
        }
    }
    let (mut loaded, mut stored) = (BTreeSet::new(), BTreeSet::new());
    match &function.body {
        Body::Regions(r) => region(function, r, &mut loaded, &mut stored),
        // a function in blocks has no nested regions to walk: say nothing of its shape
        Body::Cfg(_) => {}
    }
    loaded.intersection(&stored).next().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use iridia_ir::{Address, End, Precision, Res, Type, ValueDef, ValueId};

    fn function(load: u32, store: u32) -> Function {
        let value = ValueDef {
            ty: Type::U32,
            op: Op::LoadBuffer {
                resource: Res::of(ResourceId(load)),
                address: Address::fixed(0),
                oob: iridia_ir::OutOfBounds::Zero,
            },
            origin: None,
            precision: Precision::Full,
        };
        Function {
            name: "main".into(),
            values: vec![value],
            vars: Vec::new(),
            queries: Vec::new(),
            body: Body::Regions(Region {
                nodes: vec![
                    Node::Value(ValueId(0)),
                    Node::StoreBuffer {
                        resource: Res::of(ResourceId(store)),
                        address: Address::fixed(0),
                        value: ValueId(0),
                    },
                ],
                end: Some(End::Return),
            }),
        }
    }

    #[test]
    fn a_buffer_read_and_written_is_the_shape_and_two_buffers_are_not() {
        assert!(reads_and_writes_a_buffer(&function(3, 3)));
        assert!(!reads_and_writes_a_buffer(&function(3, 4)));
    }
}
