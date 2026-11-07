//! Remove loop and merge phis that carry one value, including mutually recursive phis. Conditions do not matter
//! when every arm carries that same value. Components are visited after their operands, so aliases are final.
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use super::{At, Node, NodeId, Values, node_id};

fn arms(node: &Node) -> &[NodeId] {
    match node {
        Node::Loop { arms, .. } | Node::Merge { arms, .. } => arms,
        _ => &[],
    }
}

fn remap(node: &mut Node, aliases: &[NodeId]) {
    let map = |ids: &mut Vec<NodeId>| {
        for id in ids {
            *id = aliases.get(id.0 as usize).copied().unwrap_or(*id);
        }
    };
    match node {
        Node::Global {
            at: At::Indexed { index, .. },
            ..
        } => map(index),
        Node::Texel {
            coordinate, controls, ..
        } => {
            map(coordinate);
            map(controls);
        }
        Node::Query { operands, .. } | Node::Op { operands, .. } => map(operands),
        Node::Merge { condition, arms } => {
            map(condition);
            map(arms);
        }
        Node::Loop { arms, exit } => {
            map(arms);
            map(exit);
        }
        _ => {}
    }
}

// NodeIds come from this graph; indexed stacks implement both DFS passes without recursion on the native stack.
#[allow(clippy::indexing_slicing)]
fn phis(values: &mut Values) -> bool {
    let n = values.nodes.len();
    let is_phi: Vec<bool> = values
        .nodes
        .iter()
        .map(|n| matches!(n, Node::Loop { .. } | Node::Merge { .. }))
        .collect();
    let mut reverse = vec![Vec::new(); n];
    for (i, node) in values.nodes.iter().enumerate() {
        for arm in arms(node) {
            let j = arm.0 as usize;
            if is_phi[j] {
                reverse[j].push(i);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut order = Vec::new();
    for i in 0..n {
        if !is_phi[i] || seen[i] {
            continue;
        }
        seen[i] = true;
        let mut stack = vec![(i, 0usize)];
        while let Some((i, next)) = stack.last_mut() {
            if let Some(arm) = arms(&values.nodes[*i]).get(*next) {
                *next = next.saturating_add(1);
                let j = arm.0 as usize;
                if is_phi[j] && !seen[j] {
                    seen[j] = true;
                    stack.push((j, 0));
                }
            } else {
                order.push(*i);
                stack.pop();
            }
        }
    }
    let mut component = vec![usize::MAX; n];
    let mut groups = Vec::new();
    for &i in order.iter().rev() {
        if component[i] != usize::MAX {
            continue;
        }
        let group = groups.len();
        let mut members = Vec::new();
        let mut stack = vec![i];
        component[i] = group;
        while let Some(i) = stack.pop() {
            members.push(i);
            for &j in &reverse[i] {
                if component[j] == usize::MAX {
                    component[j] = group;
                    stack.push(j);
                }
            }
        }
        groups.push(members);
    }
    let mut aliases: Vec<NodeId> = (0..n).map(node_id).collect();
    for (group, members) in groups.iter().enumerate().rev() {
        let mut outside = None;
        let mut different = false;
        for &i in members {
            for arm in arms(&values.nodes[i]) {
                let j = arm.0 as usize;
                if component[j] == group {
                    continue;
                }
                let value = aliases[j];
                if outside.is_some_and(|o| o != value) {
                    different = true;
                    break;
                }
                outside = Some(value);
            }
        }
        if !different && let Some(value) = outside {
            for &i in members {
                aliases[i] = value;
            }
        }
    }
    apply(values, &aliases)
}

#[allow(clippy::indexing_slicing)]
fn apply(values: &mut Values, aliases: &[NodeId]) -> bool {
    let mut changed = false;
    for node in &mut values.nodes {
        let before = node.clone();
        remap(node, aliases);
        changed |= before != *node;
    }
    for (i, alias) in aliases.iter().enumerate() {
        let j = alias.0 as usize;
        if i != j {
            changed |= values.nodes[i] != values.nodes[j] || values.kinds[i] != values.kinds[j];
            values.nodes[i] = values.nodes[j].clone();
            values.kinds[i] = values.kinds[j];
        }
    }
    for id in values
        .outputs
        .values_mut()
        .chain(values.patch_constants.values_mut())
        .chain(values.discards.iter_mut().flatten())
        .chain(values.writes.iter_mut().flatten())
    {
        changed |= *id != aliases[id.0 as usize];
        *id = aliases[id.0 as usize];
    }
    changed
}

/// Re-intern computations after trivial header phis are substituted. An invariant load inside a loop and the same
/// load before it must remain one value; otherwise a loop choosing between them spuriously depends on its exit.
#[allow(clippy::indexing_slicing)]
fn computations(values: &mut Values) -> bool {
    let n = values.nodes.len();
    let mut seen = vec![false; n];
    let mut aliases: Vec<NodeId> = (0..n).map(node_id).collect();
    let mut interned = BTreeMap::new();
    for i in 0..n {
        let mut stack = vec![(i, false)];
        while let Some((i, expanded)) = stack.pop() {
            let is_loop = matches!(values.nodes[i], Node::Loop { .. });
            if expanded {
                let mut node = values.nodes[i].clone();
                remap(&mut node, &aliases);
                if let Node::Merge { arms, .. } = &node
                    && let Some(&first) = arms.first()
                    && first != node_id(i)
                    && arms.iter().all(|a| *a == first)
                {
                    aliases[i] = first;
                    continue;
                }
                aliases[i] = *interned.entry((node, values.kinds[i])).or_insert(node_id(i));
            } else if !seen[i] {
                seen[i] = true;
                if !is_loop {
                    stack.push((i, true));
                    stack.extend(
                        values
                            .operands(node_id(i))
                            .into_iter()
                            .map(|o| (o.0 as usize, false)),
                    );
                }
            }
        }
    }
    apply(values, &aliases)
}

pub(super) fn simplify(values: &mut Values) {
    // Substitution can make two computations equal, and interning them can make another phi trivial. Neither pass
    // walks shader code again; every change removes a distinction between graph values.
    while phis(values) | computations(values) {}
}
