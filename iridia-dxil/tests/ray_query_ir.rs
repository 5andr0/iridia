#![allow(clippy::unwrap_used)]
//! The inline ray tracing programs (`ray_query_*.hlsl`, DXR 1.1's `RayQuery`) lifted onto the IR
//! ([`iridia_dxil::ir`]): each `RayQuery` a query of the function, declared with its template's flags, its operations
//! in the text form, and what the validator holds them to. What they compute is `exec_warp_rq.rs`'s (against WARP) and
//! `semantics.rs`'s (against the HLSL) to check, but for `ray_query_state.hlsl`, whose reads of the ray are here.
use iridia_ir::{HitField, Op, Program, QueryField, QueryId};

fn program(name: &str) -> Program {
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap();
    let mut p = iridia_dxil::ir::lift(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
    iridia_passes::prepare(&mut p).unwrap_or_else(|e| panic!("{name}: {e}"));
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

/// Every ray query fixture lifts, its queries declared with their templates' flags, and its text form reads back as
/// it was printed.
#[test]
fn every_ray_query_program_lifts_with_its_queries_declared() {
    // (fixture, the flags each query is declared with, what its text has)
    let expected: [(&str, &[u32], &[&str]); 8] = [
        (
            "ray_query_grid",
            &[0],
            &["rayquery init q0", "rayquery commit q0\n", "committed.status"],
        ),
        (
            "ray_query_hits",
            &[0],
            &[
                "rayquery commit q0\n",
                "committed.barycentrics.y",
                "committed.status",
            ],
        ),
        ("ray_query_hits.od", &[0], &["rayquery commit q0\n"]),
        (
            "ray_query_matrices",
            &[0],
            &["candidate.objecttoworld.r2c3", "candidate.worldtoobject.r0c0"],
        ),
        (
            "ray_query_procedural",
            &[0],
            &["rayquery commit q0 t %", "candidate.type", "committed.status"],
        ),
        (
            "ray_query_state",
            &[0x10],
            &["ray.flags", "ray.origin.z", "ray.direction.x", "ray.tmin"],
        ),
        ("ray_query_state.od", &[0x10], &["ray.flags"]),
        (
            "ray_query_object",
            &[0x8, 0x100],
            &[
                "candidate.objectorigin.x",
                "committed.objectdirection.z",
                "candidate.nonopaque",
                "rayquery.proceed q1",
                "committed.status",
            ],
        ),
    ];
    for (name, flags, wanted) in expected {
        let p = program(&format!("{name}.dxil"));
        let declared: Vec<u32> = p.entry.queries.iter().map(|q| q.flags).collect();
        assert_eq!(declared, flags, "{name}");
        let text = iridia_ir::text::print(&p.entry);
        let again = iridia_ir::text::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}\n{text}"));
        assert_eq!(iridia_ir::text::print(&again), text, "{name}");
        for w in wanted.iter().chain(&["rayquery.proceed q0"]) {
            assert!(text.contains(w), "{name}: no {w}\n{text}");
        }
    }
}

/// The damaged ray query containers `malformed.rs` holds the naga-path lift to (one byte changed each) are refused by
/// the IR lift too, by the same name: never lifted, never a panic.
#[test]
fn damaged_ray_query_programs_are_refused_by_name() {
    for (name, at, was, now, expected) in [
        (
            "ray_query_grid",
            4304,
            0xe8,
            0xbe,
            "dx.op 188 called with 2 arguments, where it takes 4",
        ),
        (
            "ray_query_hits",
            4274,
            0xbb,
            0xbc,
            "an instruction that yields a value but defines none",
        ),
        (
            "ray_query_hits.od",
            4330,
            0xbb,
            0xbc,
            "an instruction that yields a value but defines none",
        ),
    ] {
        let mut bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(format!("{name}.dxil")),
        )
        .unwrap();
        assert_eq!(bytes[at], was, "{name} is not the fixture the case was made from");
        bytes[at] = now;
        match iridia_dxil::ir::lift(&bytes) {
            Ok(_) => panic!("{name}: the damaged program was lifted"),
            Err(e) => assert!(e.detail.contains(expected), "{name}: {}", e.detail),
        }
    }
}

/// The validator holds a ray query's operations to a query the function declares, a field to the lanes, rows and
/// columns it has, the search to an acceleration structure, and a committed distance to an `f32`.
#[test]
fn the_validator_holds_ray_queries_to_their_declarations() {
    let p = program("ray_query_procedural.dxil");
    assert_eq!(problems(&p), Vec::<String>::new());

    let mut q = p.clone();
    q.entry.queries.clear();
    refused(&q, "ray query q0, where the function declares 0");

    let mut q = p.clone();
    let state = q
        .entry
        .values
        .iter_mut()
        .find(|d| matches!(d.op, Op::RayQueryState(..)))
        .unwrap();
    state.op = Op::RayQueryState(QueryId(0), QueryField::Committed(HitField::Barycentrics(2)));
    state.ty = iridia_ir::Type::F32;
    refused(&q, "reads past its lanes, rows or columns");

    let mut q = p.clone();
    let state = q
        .entry
        .values
        .iter_mut()
        .find(|d| matches!(d.op, Op::RayQueryState(_, QueryField::CommittedStatus)))
        .unwrap();
    state.ty = iridia_ir::Type::F32;
    refused(&q, "states f32 where its operation gives u32");

    // the search started in the constant buffer, by the text form
    let constants = p
        .interface
        .resources
        .iter()
        .find(|r| matches!(r.kind, iridia_ir::interface::ResourceKind::ConstantBuffer { .. }))
        .unwrap()
        .id;
    let text = iridia_ir::text::print(&p.entry);
    let at = text.find("rayquery init q0 ").unwrap() + "rayquery init q0 ".len();
    let end = at + text[at..].find(' ').unwrap();
    let moved = format!("{}r{}{}", &text[..at], constants.0, &text[end..]);
    let mut q = p.clone();
    q.entry = iridia_ir::text::parse(&moved).unwrap_or_else(|e| panic!("{e}\n{moved}"));
    refused(
        &q,
        &format!(
            "a ray query started in resource {}, not an acceleration structure",
            constants.0
        ),
    );
}
