#![allow(clippy::unwrap_used)]
//! What the compute programs of a corpus do that a comparison against WARP would have to account for: which resources they
//! write, whether they share memory within a group, take atomics (and whether any reads the old value, which depends on the
//! order the invocations run in) or waves. A census, not a test of anything: it prints one line per class.
//!
//! Needs a corpus folder; `shader.census=1` runs it (it lifts every compute program of the four corpora).
mod common;
mod inputs;

use std::collections::BTreeMap;

use iridia_ir::interface::{ResourceKind, StorageTarget};
use iridia_ir::{End, Function, Memory, Node, Op, Region, Space, ValueId};

fn has_dxil(bytes: &[u8]) -> bool {
    let word = |at: usize| -> Option<usize> {
        let b = bytes.get(at..at.checked_add(4)?)?;
        usize::try_from(u32::from_le_bytes(b.try_into().ok()?)).ok()
    };
    let Some(count) = word(28) else {
        return false;
    };
    (0..count).any(|k| {
        word(32 + 4 * k)
            .and_then(|at| bytes.get(at..at.checked_add(4)?))
            .is_some_and(|k| k == b"DXIL")
    })
}

fn lifted(bytes: &[u8]) -> Result<iridia_ir::Program, String> {
    let mut p = if has_dxil(bytes) {
        iridia_dxil::ir::lift(bytes)
    } else {
        iridia_dxbc::ir::lift(bytes)
    }
    .map_err(|e| format!("the IR lift: {e}"))?;
    iridia_passes::prepare(&mut p)?;
    Ok(p)
}

/// Every value some operation, node or region end reads.
fn read_values(f: &Function, body: &Region) -> Vec<bool> {
    fn region(f: &Function, r: &Region, read: &mut [bool]) {
        let mut mark = |v: ValueId| {
            if let Some(slot) = read.get_mut(v.0 as usize) {
                *slot = true;
            }
        };
        for n in &r.nodes {
            if let Node::Value(v) = n
                && let Some(d) = f.value(*v)
            {
                d.op.operands().into_iter().for_each(&mut mark);
            }
            n.values().into_iter().for_each(&mut mark);
        }
        if let Some(End::Yield(vs) | End::Br(_, vs) | End::Continue(_, vs)) = &r.end {
            vs.iter().copied().for_each(&mut mark);
        }
        for inner in r.nodes.iter().flat_map(Node::regions) {
            region(f, inner, read);
        }
    }
    let mut read = vec![false; f.values.len()];
    region(f, body, &mut read);
    read
}

fn has_barrier(r: &Region) -> bool {
    r.nodes
        .iter()
        .any(|n| matches!(n, Node::Barrier(_)) || n.regions().into_iter().any(has_barrier))
}

/// The facts of one compute program that decide how a comparison could run it.
#[derive(Default)]
struct Facts {
    waves: bool,
    arrays: bool,
    images: bool,
    storage_images: bool,
    groupshared: bool,
    barrier: bool,
    atomics_unread: bool,
    atomics_read: bool,
    counters: bool,
    buffer_writes: bool,
}

fn facts(p: &iridia_ir::Program) -> Facts {
    let mut x = Facts::default();
    let f = &p.entry;
    let iridia_ir::Body::Regions(body) = &f.body else {
        return x;
    };
    let read = read_values(f, body);
    for r in &p.interface.resources {
        x.arrays |= r.count != 1 && !matches!(r.kind, ResourceKind::PipelineState(_));
        x.storage_images |= matches!(
            r.kind,
            ResourceKind::Storage {
                target: StorageTarget::Image { .. },
                ..
            }
        );
    }
    x.groupshared = f.vars.iter().any(|v| v.space == Space::Workgroup);
    x.barrier = has_barrier(body);
    for (n, v) in f.values.iter().enumerate() {
        match v.op {
            Op::Wave(_) => x.waves = true,
            Op::Counter(..) => x.counters = true,
            Op::Atomic { memory, .. } => {
                if read.get(n).copied().unwrap_or(true) {
                    x.atomics_read = true;
                } else {
                    x.atomics_unread = true;
                }
                if matches!(memory, Memory::Image(r, _) if p.interface.resources.iter().any(|x| x.id == r.id && matches!(x.kind, ResourceKind::Storage { target: StorageTarget::Image { .. }, .. })))
                {
                    x.images = true;
                }
            }
            _ => {}
        }
    }
    fn stores(r: &Region, x: &mut Facts) {
        for n in &r.nodes {
            match n {
                Node::StoreBuffer { .. } | Node::StoreTexel { .. } => x.buffer_writes = true,
                _ => {}
            }
            for inner in n.regions() {
                stores(inner, x);
            }
        }
    }
    stores(body, &mut x);
    x
}

/// The one class a program is counted in, the first of these that it is.
fn class(x: &Facts) -> &'static str {
    if x.waves {
        "waves"
    } else if x.arrays {
        "arrays of resources"
    } else if x.storage_images {
        "storage images"
    } else if x.atomics_read || x.counters {
        "atomics or counters whose old value is read"
    } else if x.atomics_unread {
        "atomics whose old value nothing reads"
    } else if x.groupshared || x.barrier {
        "group shared memory, no atomics"
    } else if x.buffer_writes {
        "buffer writes only"
    } else {
        "writes nothing"
    }
}

fn census(game: inputs::Game) {
    if inputs::debug("shader.census").as_deref() != Some("1") {
        println!("shader.census is not 1: skipped");
        return;
    }
    let Some(corpus) = common::corpus(game) else {
        return;
    };
    let mut by_class: BTreeMap<&'static str, (usize, String)> = BTreeMap::new();
    let (mut compute, mut unlifted) = (0, 0);
    for file in &corpus.names {
        let bytes = corpus.read(file);
        let p = match lifted(&bytes) {
            Ok(p) => p,
            Err(_) => {
                unlifted += 1;
                continue;
            }
        };
        if p.interface.stage != iridia_ir::interface::Stage::Compute {
            continue;
        }
        compute += 1;
        let c = class(&facts(&p));
        let e = by_class.entry(c).or_insert((0, corpus.at(file)));
        e.0 += 1;
    }
    println!(
        "{}: {compute} compute programs lifted ({unlifted} programs not lifted)",
        corpus.name
    );
    for (c, (n, at)) in by_class {
        println!("  {n:5} {c}  (e.g. {at})");
    }
}

#[test]
fn riders_republic_compute_census() {
    census(inputs::Game::Riders);
}
#[test]
fn elden_compute_census() {
    census(inputs::Game::Elden);
}
