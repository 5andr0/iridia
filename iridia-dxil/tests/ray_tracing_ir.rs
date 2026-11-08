#![allow(clippy::unwrap_used)]
//! The ray tracing programs of `ray_tracing.hlsl` lifted onto the IR ([`iridia_dxil::ir`]): the payload and attributes
//! their functions take as variables of the ray payload and hit attributes spaces, as many words as the bytes the
//! library states, the pipeline's operations in the text form, and what the validator holds them to. What they compute
//! is `pipeline.rs`'s and `exec_warp_pipeline.rs`'s to check.
use iridia_ir::{
    Program, Space,
    interface::{Leaf, LeafKind, RayTracingFacts, Stage, StageFacts},
};

fn program(name: &str) -> Program {
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap();
    let mut p = iridia_dxil::ir::lift(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
    iridia_passes::structurize(&mut p.entry).unwrap();
    iridia_passes::break_flags(&mut p.entry).unwrap();
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    p
}

fn problems(p: &Program) -> Vec<String> {
    iridia_ir::validate(p)
        .err()
        .unwrap_or_default()
        .iter()
        .map(ToString::to_string)
        .collect()
}

fn refused(p: &Program, what: &str) {
    let found = problems(p);
    assert!(found.iter().any(|e| e.contains(what)), "{what}: {found:?}");
}

/// `p` as a program of another ray tracing stage, its facts kept.
fn as_stage(p: &Program, stage: Stage) -> Program {
    let mut q = p.clone();
    q.stage = stage;
    q.interface.stage = stage;
    q
}

/// The scalars each program's facts state for the structures its function takes: the payload (a float, a uint and a
/// float3 as Direct3D lays them out, a callable's parameter of one float) and the triangle's barycentrics, each at the
/// byte its field starts at, and none for a program that takes none.
#[test]
fn the_ray_tracing_programs_state_the_scalars_of_their_payloads_and_attributes() {
    let at = |kinds: &[(u32, LeafKind)]| -> Vec<Leaf> {
        kinds
            .iter()
            .map(|&(offset, kind)| Leaf { offset, kind })
            .collect()
    };
    let payload = at(&[
        (0, LeafKind::Float32),
        (4, LeafKind::Int32),
        (8, LeafKind::Float32),
        (12, LeafKind::Float32),
        (16, LeafKind::Float32),
    ]);
    let barycentrics = at(&[(0, LeafKind::Float32), (4, LeafKind::Float32)]);
    let expected = [
        ("raygen", vec![], vec![]),
        ("miss", payload.clone(), vec![]),
        ("closest_hit", payload.clone(), barycentrics.clone()),
        ("any_hit", payload, barycentrics),
        ("intersection", vec![], vec![]),
        ("callable", at(&[(0, LeafKind::Float32)]), vec![]),
    ];
    for variant in ["", ".od"] {
        for (export, payload, attributes) in &expected {
            let name = format!("ray_tracing.{export}{variant}.dxil");
            let p = program(&name);
            let StageFacts::RayTracing(facts) = &p.interface.facts else {
                panic!("{name}: {:?}", p.interface.facts);
            };
            assert_eq!(&facts.payload_leaves, payload, "{name}");
            assert_eq!(&facts.attribute_leaves, attributes, "{name}");
        }
    }
}

/// Leaves that overlap, run past the bytes the facts state or are out of order are the interface's to refuse.
#[test]
fn leaves_that_do_not_fit_their_bytes_are_refused() {
    let closest = program("ray_tracing.closest_hit.dxil");
    for (what, edit) in [
        (
            "payload: a Float64 at byte 16 of 20 bytes",
            (|f: &mut RayTracingFacts| {
                f.payload_leaves.push(Leaf {
                    offset: 16,
                    kind: LeafKind::Float64,
                });
            }) as fn(&mut RayTracingFacts),
        ),
        (
            "hit attributes: a Float32 at byte 0 of 8 bytes, after the leaves that end at 8",
            |f: &mut RayTracingFacts| {
                f.attribute_leaves.insert(
                    0,
                    Leaf {
                        offset: 4,
                        kind: LeafKind::Float32,
                    },
                );
            },
        ),
    ] {
        let mut q = closest.clone();
        let StageFacts::RayTracing(facts) = &mut q.interface.facts else {
            unreachable!()
        };
        edit(facts);
        let found = q.interface.verify().unwrap_err();
        assert!(found.contains(what), "{what}: {found}");
    }
}

/// Each program's payload (a payload of a distance, a count and a colour: 20 bytes as Direct3D lays it out; a
/// callable's parameter of one float) and attributes (the triangle's barycentrics, 8 bytes) are the words of the
/// variables of their spaces, and the text form reads back as it was printed.
#[test]
fn the_ray_tracing_programs_take_their_payloads_and_attributes_as_variables() {
    let expected = [
        ("raygen", Stage::RayGeneration, 0, 0),
        ("miss", Stage::Miss, 20, 0),
        ("closest_hit", Stage::ClosestHit, 20, 8),
        ("any_hit", Stage::AnyHit, 20, 8),
        ("intersection", Stage::Intersection, 0, 0),
        ("callable", Stage::Callable, 4, 0),
    ];
    for variant in ["", ".od"] {
        for (export, stage, payload, attributes) in expected {
            let name = format!("ray_tracing.{export}{variant}.dxil");
            let p = program(&name);
            assert_eq!(p.stage, stage, "{name}");
            let StageFacts::RayTracing(RayTracingFacts {
                payload: p_bytes,
                attributes: a_bytes,
                ..
            }) = &p.interface.facts
            else {
                panic!("{name}: {:?}", p.interface.facts);
            };
            assert_eq!((*p_bytes, *a_bytes), (payload, attributes), "{name}");
            let words = |space: Space| -> Option<u32> {
                p.entry
                    .vars
                    .iter()
                    .find(|v| v.space == space)
                    .map(|v| v.len.unwrap_or(1))
            };
            assert_eq!(
                words(Space::RayPayload),
                (payload != 0).then(|| payload.div_ceil(4)),
                "{name}"
            );
            assert_eq!(
                words(Space::HitAttributes),
                (attributes != 0).then(|| attributes.div_ceil(4)),
                "{name}"
            );
            let text = iridia_ir::text::print(&p.entry);
            let again = iridia_ir::text::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}\n{text}"));
            assert_eq!(iridia_ir::text::print(&again), text, "{name}");
            let wanted: &[&str] = match export {
                "raygen" => &["trace ray r", "call shader %"],
                "any_hit" => &["ignorehit", "accepthit"],
                "intersection" => &["reporthit %"],
                _ => &[],
            };
            for w in wanted {
                assert!(text.contains(w), "{name}: no {w}\n{text}");
            }
        }
    }
}

/// The pipeline's operations belong to their stages: a ray traced and a shader called from an intersection program, a
/// hit reported from an any-hit program and a hit decided in a closest-hit program are refused, and a payload or
/// attributes outside the stages that have them.
#[test]
fn the_pipelines_operations_belong_to_their_stages() {
    let raygen = program("ray_tracing.raygen.dxil");
    let q = as_stage(&raygen, Stage::Intersection);
    refused(
        &q,
        "a traced ray outside a ray generation, closest-hit or miss program",
    );
    refused(
        &q,
        "a shader call outside a ray generation, closest-hit, miss or callable program",
    );
    // a miss program may trace a ray and call a shader
    assert_eq!(problems(&as_stage(&raygen, Stage::Miss)), Vec::<String>::new());

    let intersection = program("ray_tracing.intersection.dxil");
    refused(
        &as_stage(&intersection, Stage::AnyHit),
        "a reported hit outside an intersection program",
    );

    let any_hit = program("ray_tracing.any_hit.dxil");
    refused(
        &as_stage(&any_hit, Stage::ClosestHit),
        "a hit decided outside an any-hit program",
    );
    let q = as_stage(&any_hit, Stage::RayGeneration);
    refused(&q, "ray payload payload in a RayGeneration program");
    refused(&q, "hit attributes attributes in a RayGeneration program");
}

/// The attributes are read alone, and the payload and attributes are as many words as the facts' bytes fill.
#[test]
fn a_hits_attributes_are_read_alone_and_the_payloads_are_the_facts_size() {
    let closest = program("ray_tracing.closest_hit.dxil");
    let text = iridia_ir::text::print(&closest.entry);
    // a store of the first attribute's word back into it
    let load = text.lines().find(|l| l.contains("= load attributes[")).unwrap();
    let (value, rest) = load.trim().split_once(':').unwrap();
    let index = rest
        .split("attributes[")
        .nth(1)
        .unwrap()
        .split(']')
        .next()
        .unwrap();
    let indent: String = load.chars().take_while(|c| c.is_whitespace()).collect();
    let stored = text.replacen(
        load,
        &format!("{load}\n{indent}attributes[{index}] = {value} oob.zero"),
        1,
    );
    let mut q = closest.clone();
    q.entry = iridia_ir::text::parse(&stored).unwrap_or_else(|e| panic!("{e}\n{stored}"));
    refused(&q, "a store to hit attributes attributes, which is read alone");

    let mut q = closest.clone();
    let StageFacts::RayTracing(facts) = &mut q.interface.facts else {
        unreachable!()
    };
    facts.payload = 24;
    refused(
        &q,
        "ray payload payload is not the 6 u32 words of the 24 bytes the facts state",
    );
    let mut q = closest;
    let StageFacts::RayTracing(facts) = &mut q.interface.facts else {
        unreachable!()
    };
    facts.attributes = 4;
    refused(
        &q,
        "hit attributes attributes is not the 1 u32 words of the 4 bytes the facts state",
    );
}
