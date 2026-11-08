//! Dead transfers can form cycles even when ordinary expression dead-code elimination reaches a fixed point.
#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]
mod support;

use iridia_ir::interface::ScalarType;
use iridia_ir::{Body, End, LabelId, Node, Op, Precision, Region, Type, ValueDef, ValueId};
use support::{from_text, run};

const F1: (ScalarType, u8) = (ScalarType::Float, 1);

fn slots(r: &Region) -> usize {
    r.nodes
        .iter()
        .map(|n| {
            let own = match n {
                Node::If { results, .. } | Node::Block { results, .. } | Node::Switch { results, .. } => {
                    results.len()
                }
                Node::Loop { params, results, .. } => params.len() + results.len(),
                _ => 0,
            };
            own + n.regions().into_iter().map(slots).sum::<usize>()
        })
        .sum()
}

fn count(p: &iridia_ir::Program) -> usize {
    let Body::Regions(r) = &p.entry.body else {
        panic!("expected regions")
    };
    slots(r)
}

fn checked(p: &iridia_ir::Program, q: &iridia_ir::Program) {
    iridia_ir::validate(q).unwrap();
    let text = iridia_ir::text::print(&q.entry);
    assert_eq!(
        iridia_ir::text::print(&iridia_ir::text::parse(&text).unwrap()),
        text
    );
    for x in [-4.0f32, 0.0, 1.0, 12.0] {
        assert_eq!(
            run(p, &[[x.to_bits(), 0, 0, 0]], &[]),
            run(q, &[[x.to_bits(), 0, 0, 0]], &[]),
            "x={x}"
        );
    }
    let mut twice = q.clone();
    iridia_passes::simplify(&mut twice.entry);
    assert_eq!(twice.entry, q.entry, "one worklist reaches the fixed point");
}

#[test]
fn a_dead_loop_carried_cycle_does_not_keep_its_slots_or_arguments_alive() {
    let p = from_text(
        "iridia-ir 1
fn main {
  var counter: f32
  var dead: f32
  %0: f32 = input 0.x
  %1: f32 = f32(0.0)
  %2: f32 = f32(1.0)
  %3: f32 = f32(3.0)
  counter = %1
  dead = %0
  loop '0 () = () {
    %4: f32 = load counter
    %5: bool = fcmp.ge %4, %3
    if %5 {
      br '0
    } else {
      yield
    }
    %6: f32 = fadd %4, %2
    counter = %6
    %7: f32 = load dead
    %8: f32 = fadd %7, %2
    dead = %8
    continue '0
  }
  %9: f32 = load counter
  output 0.x = %9
  return
}
",
        &[F1],
        &[F1],
        Vec::new(),
    );
    let mut promoted = p.clone();
    iridia_passes::promote(&mut promoted.entry).unwrap();
    assert_eq!(count(&promoted), 4);
    let mut q = promoted.clone();
    iridia_passes::simplify(&mut q.entry);
    assert_eq!(count(&q), 2, "only the counter's parameter and result remain");
    checked(&p, &q);
}

#[test]
fn nested_exits_prune_the_nearest_shadowed_labels_argument_positions() {
    let mut p = from_text(
        "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f32 = fneg %0
  %2: f32 = f32(0.0)
  %3: bool = fcmp.lt %0, %2
  output 0.x = %0
  return
}
",
        &[F1],
        &[F1],
        Vec::new(),
    );
    let param = |p: &mut iridia_ir::Program| {
        let v = ValueId(u32::try_from(p.entry.values.len()).unwrap());
        p.entry.values.push(ValueDef {
            ty: Type::F32,
            op: Op::Param,
            origin: None,
            precision: Precision::Full,
        });
        v
    };
    let outer_live = param(&mut p);
    let outer_dead = param(&mut p);
    let inner_live = param(&mut p);
    let inner_dead = param(&mut p);
    let Body::Regions(base) = &p.entry.body else {
        panic!("expected regions")
    };
    let mut output = base.nodes.last().unwrap().clone();
    for v in output.values_mut() {
        *v = outer_live;
    }
    let label = LabelId(0);
    let inner = Node::Block {
        label,
        results: vec![inner_live, inner_dead],
        body: Region {
            nodes: vec![Node::If {
                cond: ValueId(3),
                results: Vec::new(),
                accept: Region {
                    nodes: Vec::new(),
                    end: Some(End::Br(label, vec![ValueId(0), ValueId(1)])),
                },
                reject: Region {
                    nodes: Vec::new(),
                    end: Some(End::Yield(Vec::new())),
                },
            }],
            end: Some(End::Yield(vec![ValueId(1), ValueId(0)])),
        },
    };
    p.entry.body = Body::Regions(Region {
        nodes: vec![
            Node::Value(ValueId(0)),
            Node::Value(ValueId(1)),
            Node::Value(ValueId(2)),
            Node::Value(ValueId(3)),
            Node::Block {
                label,
                results: vec![outer_live, outer_dead],
                body: Region {
                    nodes: vec![inner],
                    end: Some(End::Br(label, vec![inner_live, inner_dead])),
                },
            },
            output,
        ],
        end: Some(End::Return),
    });
    iridia_ir::validate(&p).unwrap();
    let mut q = p.clone();
    iridia_passes::simplify(&mut q.entry);
    assert_eq!(count(&q), 2);
    checked(&p, &q);
}

#[test]
fn a_graph_body_is_left_unchanged() {
    let mut p = from_text(
        "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  output 0.x = %0
  return
}
",
        &[F1],
        &[F1],
        Vec::new(),
    );
    iridia_passes::flatten(&mut p.entry).unwrap();
    let before = p.entry.clone();
    iridia_passes::simplify(&mut p.entry);
    assert_eq!(p.entry, before);
}

#[test]
fn an_atomic_effect_stays_when_nothing_reads_its_old_value() {
    use iridia_ir::interface::{
        Access, Resource, ResourceId, ResourceKind, ResourceProvenance, SampleType, StorageTarget, TextureDim,
    };
    let mut p = from_text(
        "iridia-ir 1
fn main {
  %0: vec2<u32> = vec2<u32>(0, 0)
  %1: u32 = u32(1)
  %2: u32 = atomic.add image r0, %0 value %1
  return
}
",
        &[],
        &[],
        vec![Resource {
            id: ResourceId(0),
            count: 1,
            kind: ResourceKind::Storage {
                target: StorageTarget::Image {
                    dim: TextureDim::D2,
                    arrayed: false,
                    sample: SampleType::Uint,
                },
                access: Access::ReadWrite,
            },
            provenance: ResourceProvenance {
                name: "image".into(),
                ..ResourceProvenance::default()
            },
        }],
    );
    iridia_passes::simplify(&mut p.entry);
    iridia_ir::validate(&p).unwrap();
    let Body::Regions(r) = &p.entry.body else {
        panic!("expected regions")
    };
    assert!(r.nodes.iter().any(|n| matches!(n, Node::Value(v)
        if p.entry.value(*v).is_some_and(|d|matches!(d.op,Op::Atomic{..})))));
}
