#![allow(clippy::unwrap_used)]
//! Lane slicing of one vertex output: what a slice reads and refuses. What the slices compute is no longer run (the
//! executor runs IR programs only, and `project_vertex_output` goes with this crate); the whole program's answer is
//! held on the IR (iridia-d3d9-ir's tests/vertex_output.rs).
use iridia::flow::{At, Node, Values};
use iridia_d3d9::{lift, project_vertex_output};

fn op(code: u32, args: &[u32]) -> Vec<u32> {
    [vec![code | (args.len() as u32) << 24], args.to_vec()].concat()
}
fn bytes(body: &[u32]) -> Vec<u8> {
    [vec![0xfffe_0300], body.to_vec(), vec![65535]]
        .concat()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}
fn declarations() -> Vec<u32> {
    [
        op(31, &[0x8000_0000, 0x900f_0000]), // v0 POSITION
        op(31, &[0x8000_0005, 0x900f_0001]), // v1 TEXCOORD0
        op(31, &[0x8000_0003, 0x900f_0002]), // v2 NORMAL
        op(31, &[0x8000_0000, 0xe00f_0007]), // o7 POSITION
        op(31, &[0x8007_0005, 0xe00f_0003]), // o3 TEXCOORD7
        op(31, &[0x8000_000a, 0xe00f_0005]), // o5 COLOR0
    ]
    .concat()
}
fn fixture() -> Vec<u8> {
    bytes(
        &[
            declarations(),
            op(81, &[0xa00f_0001, 1f32.to_bits(), 0, 0, 0]),
            op(1, &[0xe00f_0007, 0x90e4_0000]),              // position
            op(5, &[0xe00f_0005, 0x90e4_0002, 0xa0e4_0004]), // unrelated frame value
            op(4, &[0x8007_0000, 0x90c4_0001, 0xa050_0001, 0xa005_0001]), // r0.xyz = (u,v,1)
            op(8, &[0xe001_0003, 0xa0e4_000a, 0x80e4_0000]),
            op(8, &[0xe002_0003, 0xa0e4_000b, 0x80e4_0000]),
            op(1, &[0xe00c_0003, 0xa0e4_000c]), // unrelated z/w
        ]
        .concat(),
    )
}

#[test]
fn output_slice_keeps_original_hash_and_reads_only_what_the_output_needs() {
    let raw = fixture();
    let full = lift(&raw).unwrap();
    let sliced = project_vertex_output(&raw, 3, 3).unwrap();
    assert_eq!(sliced.stage, iridia::Stage::Pixel);
    assert_eq!(sliced.provenance.source_hash, full.provenance.source_hash);
    assert_ne!(sliced.provenance.lifter, full.provenance.lifter);
    assert!(sliced.notes.iter().any(|n| n.contains("placeholders")));
    iridia::emit::wgsl(&sliced).unwrap();
    let graph = Values::of_program(&sliced).unwrap();
    let mut input_lanes = Vec::new();
    let mut words = Vec::new();
    for n in graph.reached_from(graph.roots()) {
        match graph.node(n) {
            Node::Input { input, lane } => input_lanes.push((*input, *lane)),
            Node::Global {
                at: At::Component(word),
                ..
            } => words.push(*word),
            _ => {}
        }
    }
    input_lanes.sort();
    words.sort();
    assert_eq!(input_lanes, [(1, 0), (1, 1)]);
    assert_eq!(words, [40, 41, 42, 44, 45, 46]);
}

#[test]
fn every_mask_of_an_output_with_partial_overwrites_is_sliced() {
    let raw = bytes(
        &[
            declarations(),
            op(1, &[0xe00f_0007, 0x90e4_0000]),
            op(1, &[0xe00f_0005, 0x90e4_0002]),
            op(1, &[0x800f_0000, 0x90e4_0001]),
            op(5, &[0x8001_0000, 0x8055_0000, 0xa000_000a]), // r0.x = old r0.y * c10.x
            op(2, &[0x8002_0000, 0x8000_0000, 0xa055_000a]), // r0.y = new r0.x + c10.y
            op(1, &[0xe00f_0003, 0x80e4_0000]),
        ]
        .concat(),
    );
    lift(&raw).unwrap();
    for mask in 1..16 {
        let p = project_vertex_output(&raw, 3, mask).unwrap();
        iridia::emit::wgsl(&p).unwrap_or_else(|e| panic!("mask {mask}: {e}"));
    }
}

#[test]
fn projection_refuses_bad_selection_wrong_stage_and_invalid_dead_code() {
    let raw = fixture();
    for (register, mask) in [(3, 0), (3, 16), (9, 3)] {
        assert!(
            project_vertex_output(&raw, register, mask)
                .unwrap_err()
                .detail
                .contains("mask")
        );
    }
    let ps = [0xffff_0300u32, 65535]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    assert!(
        project_vertex_output(&ps, 0, 3)
            .unwrap_err()
            .detail
            .contains("requires vs_3_0")
    );
    let mut invalid = raw.clone();
    let end = invalid.len() - 4;
    invalid.splice(
        end..end,
        op(40, &[0xa0e4_0000]).into_iter().flat_map(u32::to_le_bytes),
    );
    assert!(
        project_vertex_output(&invalid, 3, 3)
            .unwrap_err()
            .detail
            .contains("opcode 40")
    );
    let missing = bytes(&[declarations(), op(1, &[0xe00f_0003, 0x90e4_0001])].concat());
    assert!(
        project_vertex_output(&missing, 3, 3)
            .unwrap_err()
            .detail
            .contains("uninitialized output")
    );
}

#[test]
fn projected_vertex_output_has_a_versioned_ir_baseline() {
    let outcome = project_vertex_output(&fixture(), 3, 3)
        .unwrap()
        .text()
        .map_err(|e| e.to_string());
    let table = iridia::gate::table(
        "iridia-d3d9 vertex output",
        iridia_d3d9::LIFTER_VERSION,
        &[("uv".into(), outcome)],
    );
    iridia::gate::check(include_str!("vertex-output-versions.tsv"), &table)
        .unwrap_or_else(|e| panic!("{e}\n{table}"));
}

#[test]
fn normalize_fourth_lane_and_scalar_swizzles_are_sliced() {
    let wrap = |middle: Vec<u32>| {
        bytes(
            &[
                declarations(),
                op(1, &[0xe00f_0007, 0x90e4_0000]),
                op(1, &[0xe00f_0005, 0x90e4_0002]),
                middle,
                op(1, &[0xe00f_0003, 0x80e4_0000]),
            ]
            .concat(),
        )
    };
    let raw = wrap(op(36, &[0x800f_0000, 0x90e4_0001]));
    for mask in [8, 2] {
        let p = project_vertex_output(&raw, 3, mask).unwrap();
        iridia::emit::wgsl(&p).unwrap_or_else(|e| panic!("mask {mask}: {e}"));
    }
    let raw = wrap(
        [
            op(1, &[0x800f_0000, 0x90e4_0001]),
            op(32, &[0x8002_0000, 0x9000_0001, 0xa000_000a]),
        ]
        .concat(),
    );
    let p = project_vertex_output(&raw, 3, 2).unwrap();
    iridia::emit::wgsl(&p).unwrap();
}
