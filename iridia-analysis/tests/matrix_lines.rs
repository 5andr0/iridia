#![allow(clippy::unwrap_used)]
//! Matrix orientation on the IR, using fxc and dxc fixtures for fixed matrices, indexed palettes, joint blends
//! and cloth blends chosen by a flag. Compare their constant layouts with the legacy analysis too, and exercise
//! palette addresses written in bytes as well as rows.
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

#[test]
fn matrix_constants_agree_with_the_legacy_analysis() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../iridia/tests/fixtures");
    let mut compared = 0;
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
        let (old, new) = if file.ends_with(".dxil") {
            (
                iridia_dxil::lift(&bytes).unwrap(),
                iridia_dxil::ir::lift(&bytes).unwrap(),
            )
        } else {
            (
                iridia_dxbc::lift(&bytes).unwrap(),
                iridia_dxbc::ir::lift(&bytes).unwrap(),
            )
        };
        let old_graph = old.graph().unwrap();
        let new_graph = ProgramGraph::of_program(&new).unwrap();
        for r in &old.interface.resources {
            let ResourceKind::ConstantBuffer { layout } = &r.kind else {
                continue;
            };
            for c in &layout.constants {
                let Some(size) = c.ty.size() else {
                    continue;
                };
                let expected = iridia::matrix::lines_at(Ok(&old_graph), &old.interface, r.id, c.offset, size);
                let new =
                    iridia_analysis::matrix::lines_at(Ok(&new_graph), &old.interface, r.id, c.offset, size);
                assert_eq!(expected, new, "{file}: {}", c.name);
                compared += 1;
            }
        }
    }
    assert!(compared > 20, "fixture coverage: {compared}");
    eprintln!("matrix oracle: {compared} constants agree");
}

/// The public slim corpus includes integer matrix products, both expanded and lowered from multiply-adds.
#[test]
fn slim_integer_matrix_constants_agree_with_the_legacy_analysis() {
    use inputs::{Game, OracleSet, Query, expect_found, oracle_tree};
    use iridia_ir::interface::ConstantType;
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
    let mut compared = 0;
    let mut failures = Vec::new();
    for hash in cases {
        let name = format!("programs/dxbc/{hash}.dxbc");
        let bytes = tree.read(&name).unwrap();
        let old = iridia_dxbc::lift(&bytes).unwrap();
        let new = iridia_dxbc::ir::lift(&bytes).unwrap();
        let old_graph = old.graph().unwrap();
        let new_graph = ProgramGraph::of_program(&new).unwrap();
        for r in &old.interface.resources {
            let ResourceKind::ConstantBuffer { layout } = &r.kind else {
                continue;
            };
            for c in &layout.constants {
                if !matches!(c.ty, ConstantType::Matrix { .. }) {
                    continue;
                }
                let size = c.ty.size().unwrap();
                let a = iridia::matrix::lines_at(Ok(&old_graph), &old.interface, r.id, c.offset, size);
                let b =
                    iridia_analysis::matrix::lines_at(Ok(&new_graph), &old.interface, r.id, c.offset, size);
                if a != b {
                    failures.push(format!("{name}: {}: {a:?} -> {b:?}", c.name));
                }
                compared += 1;
            }
        }
    }
    expect_found(
        Query::Oracle(OracleSet::Corpus, Game::General, "slim"),
        "integer matrix regression programs",
        cases.len(),
    );
    assert!(compared >= 20, "matrix coverage: {compared}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    eprintln!(
        "slim integer matrix oracle: {} programs, {compared} constants agree",
        cases.len()
    );
}

#[test]
fn byte_addressed_palette_lines_match_row_addressed_lines() {
    use iridia_ir::{Address, Binary, Body, Node, Op, Region, ValueDef, ValueId};
    let source = dxbc(SKIN_VS);
    let mut byte_addressed = source.clone();
    let f = &mut byte_addressed.entry;
    let mut before = std::collections::BTreeMap::new();
    for index in 0..f.values.len() {
        let Some(d) = f.values.get(index).cloned() else {
            continue;
        };
        let Op::LoadBuffer {
            resource,
            address:
                Address {
                    index: Some(i),
                    stride: 16,
                    offset,
                },
            oob,
        } = d.op
        else {
            continue;
        };
        let ty = f.ty(i).unwrap();
        let k = ValueId(u32::try_from(f.values.len()).unwrap());
        let scaled = ValueId(k.0 + 1);
        f.values.push(ValueDef {
            ty,
            op: Op::Literal([16, 0, 0, 0]),
            origin: d.origin,
            precision: d.precision,
        });
        f.values.push(ValueDef {
            ty,
            op: Op::Binary(Binary::IMul, i, k),
            origin: d.origin,
            precision: d.precision,
        });
        f.values[index].op = Op::LoadBuffer {
            resource,
            address: Address {
                index: Some(scaled),
                stride: 1,
                offset,
            },
            oob,
        };
        before.insert(
            ValueId(u32::try_from(index).unwrap()),
            vec![Node::Value(k), Node::Value(scaled)],
        );
    }
    assert!(!before.is_empty());
    fn place(r: &mut Region, before: &std::collections::BTreeMap<ValueId, Vec<Node>>) {
        let mut out = Vec::new();
        for mut n in std::mem::take(&mut r.nodes) {
            if let Node::Value(v) = n
                && let Some(add) = before.get(&v)
            {
                out.extend(add.clone());
            }
            for r in n.regions_mut() {
                place(r, before);
            }
            out.push(n);
        }
        r.nodes = out;
    }
    let Body::Regions(r) = &mut f.body else {
        panic!("region fixture")
    };
    place(r, &before);
    iridia_ir::validate(&byte_addressed).unwrap();
    for name in ["VC_aClothCancelObjMatrix", "VC_aObjMatrix"] {
        assert_eq!(
            palette_lines_of(&byte_addressed, name),
            palette_lines_of(&source, name),
            "{name}"
        );
    }
}

#[test]
fn column_palettes_keep_the_line_offset_inside_a_byte_index() {
    fn emit(text: &mut String, next: &mut u32, ty: &str, op: String) -> u32 {
        let id = *next;
        *next += 1;
        text.push_str(&format!("  %{id}: {ty} = {op}\n"));
        id
    }
    let mut p = dxbc(SINGLE_UV_VS);
    p.interface.inputs.first_mut().unwrap().value.width = 4;
    let (resource, offset) = p
        .interface
        .resources
        .iter()
        .find_map(|r| {
            let ResourceKind::ConstantBuffer { layout } = &r.kind else {
                return None;
            };
            layout
                .constants
                .iter()
                .find(|c| c.name == "VC_MatrixViewProj")
                .map(|c| (r.id, c.offset))
        })
        .unwrap();
    let mut text = "iridia-ir 1\nfn main {\n".to_owned();
    let mut next = 0;
    let input = emit(&mut text, &mut next, "vec4<f32>", "input 0.xyzw".into());
    let index_float = emit(&mut text, &mut next, "f32", format!("%{input}.x"));
    let index = emit(&mut text, &mut next, "u32", format!("bitcast %{index_float}"));
    let four = emit(&mut text, &mut next, "u32", "u32(4)".into());
    let sixteen = emit(&mut text, &mut next, "u32", "u32(16)".into());
    let base = emit(&mut text, &mut next, "u32", format!("imul %{index}, %{four}"));
    let mut sum = None;
    for (line, lane) in ["x", "y", "z", "w"].into_iter().enumerate() {
        let k = emit(&mut text, &mut next, "u32", format!("u32({line})"));
        let row = emit(&mut text, &mut next, "u32", format!("iadd %{base}, %{k}"));
        let byte = emit(&mut text, &mut next, "u32", format!("imul %{row}, %{sixteen}"));
        let matrix = emit(
            &mut text,
            &mut next,
            "vec4<f32>",
            format!("buffer r{}[%{byte} * 1] + {offset} oob.zero", resource.0),
        );
        let scalar = emit(&mut text, &mut next, "f32", format!("%{input}.{lane}"));
        let vector = emit(&mut text, &mut next, "vec4<f32>", format!("splat %{scalar}"));
        let product = emit(
            &mut text,
            &mut next,
            "vec4<f32>",
            format!("fmul.ieee %{matrix}, %{vector}"),
        );
        sum = Some(match sum {
            None => product,
            Some(s) => emit(
                &mut text,
                &mut next,
                "vec4<f32>",
                format!("fadd %{s}, %{product}"),
            ),
        });
    }
    text.push_str(&format!("  output 0.xyzw = %{}\n  return\n}}\n", sum.unwrap()));
    p.entry = iridia_ir::text::parse(&text).unwrap();
    iridia_ir::validate(&p).unwrap();
    let values = Values::of_program(&p).unwrap();
    assert_eq!(lines(&values, resource, &Words::Indexed), Ok(Majorness::Column));
}
