#![allow(clippy::unwrap_used)]
//! Matrix orientation on the IR, using fxc and dxc fixtures for fixed matrices, indexed palettes, joint blends
//! and cloth blends chosen by a flag. Hold their constants' answers to the recorded ones (`expected/matrix-*.tsv`), and
//! exercise palette addresses written in bytes as well as rows.
use iridia_analysis::flow::Values;
use iridia_analysis::matrix::{Unshown, Words, lines};
use iridia_analysis::{GraphError, ProgramGraph};
use iridia_ir::{
    Program,
    interface::{Majorness, ResourceKind},
};
mod inputs;

/// The arrangement `lines` reads for the constant `name`, by its place in the program's own layout, or, for a palette,
/// by the indexed reads of its buffer.
fn lines_of(program: &Program, name: &str) -> Result<Majorness, Unshown> {
    read(program, name, false)
}

fn palette_lines_of(program: &Program, name: &str) -> Result<Majorness, Unshown> {
    read(program, name, true)
}

fn read(program: &Program, name: &str, indexed: bool) -> Result<Majorness, Unshown> {
    let values = Values::of_program(program).expect("the value graph");
    for r in &program.interface.resources {
        let ResourceKind::ConstantBuffer { layout } = &r.kind else {
            continue;
        };
        if let Some(c) = layout.constants.iter().find(|c| c.name == name) {
            let size = c.ty.size().expect("a sized constant");
            let words = if indexed {
                Words::Indexed
            } else {
                Words::At(c.offset / 4..(c.offset + size).div_ceil(4))
            };
            return lines(&values, r.id, &words);
        }
    }
    panic!("no constant {name}")
}

fn dxbc(bytes: &[u8]) -> Program {
    iridia_dxbc::ir::lift(bytes).expect("fxc's program lifts")
}

fn dxil(bytes: &[u8]) -> Program {
    iridia_dxil::ir::lift(bytes).expect("dxc's program lifts")
}

const SINGLE_UV_VS: &[u8] = include_bytes!("../../iridia/tests/fixtures/single_uv_vs.dxbc");
const SKIN_VS: &[u8] = include_bytes!("../../iridia/tests/fixtures/layered_skin_vs.dxbc");

/// `mul(M, v)` of a `column_major` matrix scales each line by a lane of the vector: its lines are columns.
#[test]
fn a_matrix_applied_to_a_column_vector_is_read_by_columns() {
    let single = dxbc(SINGLE_UV_VS);
    assert_eq!(lines_of(&single, "VC_MatrixViewProj"), Ok(Majorness::Column));
    let skin = dxbc(SKIN_VS);
    for name in ["VC_MatrixViewProj", "FC_MatrixView"] {
        assert_eq!(lines_of(&skin, name), Ok(Majorness::Column), "{name}");
    }
}

/// A palette chosen by an index is read by the line of its element the index adds: `mul(v, M[i])` of a `column_major`
/// `float4x3` dots the vector with each of the element's lines, and a blend of the joints' matrices line by line
/// (`M[a] * wa + M[b] * wb`) that the program then dots the position with is read by rows too, the blend not being a
/// sum across lines.
#[test]
fn a_palette_is_read_by_the_lines_of_its_elements() {
    let skin = dxbc(SKIN_VS);
    for name in ["VC_aClothCancelObjMatrix", "VC_aObjMatrix"] {
        assert_eq!(palette_lines_of(&skin, name), Ok(Majorness::Row), "{name}");
    }
}

/// A cloth program that blends its joints' matrices word by word and then chooses each blended word by a flag against
/// another buffer's matrix (Elden Ring's, as dxc and fxc compile its shape) is read through the choice: the position is
/// dotted with each blended line, a row.
#[test]
fn a_palette_blended_then_chosen_by_a_flag_is_read_by_rows() {
    for (fixture, program) in [
        (
            "dxc",
            dxil(include_bytes!(
                "../../iridia/tests/fixtures/layered_cloth_blend_vs.dxil"
            )),
        ),
        (
            "fxc",
            dxbc(include_bytes!(
                "../../iridia/tests/fixtures/layered_cloth_blend_vs.dxbc"
            )),
        ),
    ] {
        assert_eq!(
            palette_lines_of(&program, "VC_aObjMatrix"),
            Ok(Majorness::Row),
            "{fixture}"
        );
    }
}

/// A program that reads a palette's translations alone (Elden Ring's LOD billboards add them to an offset) applies none
/// of its matrices: it shows no arrangement.
#[test]
fn a_palette_read_only_for_its_translations_shows_no_arrangement() {
    let billboard = dxbc(include_bytes!(
        "../../iridia/tests/fixtures/layered_billboard_vs.dxbc"
    ));
    assert_eq!(
        palette_lines_of(&billboard, "VC_aObjMatrix"),
        Err(Unshown::NotApplied)
    );
}

/// `mul(v, M)` of a `column_major` matrix dots the vector with each line: they are the rows of the matrix applied to
/// column vectors (`M` transposed).
#[test]
fn a_matrix_applied_to_a_row_vector_is_read_by_rows() {
    for (fixture, program) in [
        (
            "vertex.dxil",
            dxil(include_bytes!("../../iridia/tests/fixtures/vertex.dxil")),
        ),
        (
            "vertex.66.dxil",
            dxil(include_bytes!("../../iridia/tests/fixtures/vertex.66.dxil")),
        ),
        (
            "vertex.od.dxil",
            dxil(include_bytes!("../../iridia/tests/fixtures/vertex.od.dxil")),
        ),
    ] {
        assert_eq!(
            lines_of(&program, "world_view_projection"),
            Ok(Majorness::Row),
            "{fixture}"
        );
    }
}

/// A constant the program reads but never multiplies by another value shows no arrangement, and is refused with that.
#[test]
fn a_constant_never_applied_shows_no_arrangement() {
    let program = dxil(include_bytes!("../../iridia/tests/fixtures/vertex.dxil"));
    assert_eq!(lines_of(&program, "tint"), Err(Unshown::NotApplied));
}

/// `lines_at` reads a constant's lines from the program, finding its buffer by the D3D binding: the clip matrix by
/// columns, the joints' palette (its buffer's one array) through its indexed reads by rows, and a matrix the program
/// reads but never applies unshown, with the reason; a graph that could not be built leaves it unexamined, with why.
#[test]
fn a_constant_s_lines_are_read_at_its_place() {
    use iridia_analysis::matrix::lines_at;
    use iridia_ir::interface::MatrixLines;
    let skin = dxbc(SKIN_VS);
    let graph = ProgramGraph::of_program(&skin).expect("the value graph");
    let at = |graph: Result<&ProgramGraph, &GraphError>, name: &str| {
        let (resource, c) = skin
            .interface
            .resources
            .iter()
            .find_map(|r| match &r.kind {
                ResourceKind::ConstantBuffer { layout } => layout
                    .constants
                    .iter()
                    .find(|c| c.name == name)
                    .map(|c| (r.id, c)),
                _ => None,
            })
            .expect("the constant");
        let size = c.ty.size().expect("a sized constant");
        lines_at(graph, &skin.interface, resource, c.offset, size)
    };
    assert_eq!(at(Ok(&graph), "VC_MatrixViewProj"), MatrixLines::Columns);
    assert_eq!(at(Ok(&graph), "VC_aObjMatrix"), MatrixLines::Rows);
    // read, but only added and scaled lane by lane: not applied as a matrix
    assert_eq!(
        at(Ok(&graph), "SC_CameraPos"),
        MatrixLines::Unshown(Unshown::NotApplied)
    );
    assert_eq!(
        at(
            Err(&GraphError::Walk {
                reason: "a call".into()
            }),
            "VC_MatrixViewProj"
        ),
        MatrixLines::Unshown(Unshown::Unexamined {
            why: "the value graph: a call".into()
        })
    );
}

/// The `<file>\t<constant>\t<offset>\t<answer>` row of every sized constant of a program's constant buffers, in
/// declaration order.
fn rows(label: &str, program: &Program, graph: &ProgramGraph, only_matrices: bool) -> String {
    let mut out = String::new();
    for r in &program.interface.resources {
        let ResourceKind::ConstantBuffer { layout } = &r.kind else {
            continue;
        };
        for c in &layout.constants {
            let Some(size) = c.ty.size() else {
                continue;
            };
            if only_matrices && !matches!(c.ty, iridia_ir::interface::ConstantType::Matrix { .. }) {
                continue;
            }
            let answer =
                iridia_analysis::matrix::lines_at(Ok(graph), &program.interface, r.id, c.offset, size);
            out.push_str(&format!("{label}\t{}\t{}\t{answer:?}\n", c.name, c.offset));
        }
    }
    out
}

/// What the matrix analysis says of each constant of the fixtures, against `expected/matrix-fixtures.tsv`: the answers
/// the analysis this one replaced gave (equal to it on all 33 constants before it was removed).
#[test]
fn matrix_constants_are_the_recorded_answers() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../iridia/tests/fixtures");
    let mut got = String::new();
    for file in [
        "single_uv_vs.dxbc",
        "layered_skin_vs.dxbc",
        "layered_cloth_blend_vs.dxbc",
        "layered_cloth_blend_vs.dxil",
        "layered_billboard_vs.dxbc",
        "vertex.dxil",
        "vertex.66.dxil",
        "vertex.od.dxil",
    ] {
        let bytes = std::fs::read(root.join(file)).unwrap();
        let program = if file.ends_with(".dxil") {
            iridia_dxil::ir::lift(&bytes).unwrap()
        } else {
            iridia_dxbc::ir::lift(&bytes).unwrap()
        };
        let graph = ProgramGraph::of_program(&program).unwrap();
        got.push_str(&rows(file, &program, &graph, false));
    }
    let expected = include_str!("expected/matrix-fixtures.tsv");
    assert_eq!(got, expected.replace("\r\n", "\n"));
    assert!(
        got.lines().count() > 20,
        "fixture coverage: {}",
        got.lines().count()
    );
}

/// The public slim corpus includes integer matrix products, both expanded and lowered from multiply-adds.
#[test]
fn slim_integer_matrix_constants_are_the_recorded_answers() {
    use inputs::{Game, OracleSet, Query, expect_found, oracle_tree};
    let Some(tree) = oracle_tree(OracleSet::Corpus, Game::General, "slim") else {
        return;
    };
    let cases = [
        "08fa724d23209f4c1bdeb8d87fdb1c5782bbd259582bc6749c0eae62ce5a3321",
        "5d3d51ad2e7e09aad79a38992c434676a413972b22c318cb21888e4ab8d0347a",
        "6b60197f7eae96d68d978e9fd3a63e99914195916511d77c576e011937080386",
        "ba65413f9629317bb089e377f4f6b36c10e95c2aee81893b7d4de68a024776d5",
        "dc280fa1549f473901a2befa898efd680c3aa658efdd57087996e9e9c441c84e",
        "dfda106a3f116bf01864716b224b7e96c291066cfbd6438b8ed635b85930d812",
    ];
    let mut got = String::new();
    for hash in cases {
        let bytes = tree.read(&format!("programs/dxbc/{hash}.dxbc")).unwrap();
        let program = iridia_dxbc::ir::lift(&bytes).unwrap();
        let graph = ProgramGraph::of_program(&program).unwrap();
        got.push_str(&rows(hash, &program, &graph, true));
    }
    expect_found(
        Query::Oracle(OracleSet::Corpus, Game::General, "slim"),
        "integer matrix regression programs",
        cases.len(),
    );
    let expected = include_str!("expected/matrix-slim.tsv");
    assert_eq!(got, expected.replace("\r\n", "\n"));
    assert!(
        got.lines().count() >= 20,
        "matrix coverage: {}",
        got.lines().count()
    );
}
