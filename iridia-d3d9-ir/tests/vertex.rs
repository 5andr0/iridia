#![allow(clippy::unwrap_used)]
//! Hand-assembled `vs_3_0` tokens: bank edges, packed output semantics and independent arithmetic answers, on both
//! executors.
mod executors;

use executors::{by_register, clip, execute, lifted, no_sample, slots};
use iridia_d3d9_ir::read;
use iridia_ir::interface::{Stage, SystemValue};

fn instruction(op: u32, operands: &[u32]) -> Vec<u32> {
    [vec![op | (operands.len() as u32) << 24], operands.to_vec()].concat()
}
fn program(body: &[u32]) -> Vec<u8> {
    [vec![0xfffe_0300], body.to_vec(), vec![0x0000_ffff]]
        .concat()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect()
}
fn dcl(usage: u32, index: u32, dst: u32) -> Vec<u32> {
    instruction(31, &[0x8000_0000 | index << 16 | usage, dst])
}
fn position() -> Vec<u32> {
    [dcl(0, 0, 0x900f_000f), dcl(0, 0, 0xe00f_0007)].concat()
}
fn fixture() -> Vec<u8> {
    program(
        &[
            position(),
            dcl(10, 4, 0xe001_0003), // COLOR4 -> o3.x
            dcl(5, 3, 0xe006_0003),  // TEXCOORD3 -> o3.yz
            dcl(11, 0, 0xe008_0003), // FOG -> o3.w
            instruction(
                81,
                &[
                    0xa00f_00fe,
                    2f32.to_bits(),
                    3f32.to_bits(),
                    4f32.to_bits(),
                    5f32.to_bits(),
                ],
            ),
            instruction(1, &[0xe00f_0007, 0x90e4_000f]), // position = v15
            instruction(4, &[0x800f_0000, 0x90e4_000f, 0xa0e4_00fe, 0xa0e4_00ff]),
            instruction(1, &[0xe00f_0003, 0x80e4_0000]), // o3 = v15 * def c254 + external c255
        ]
        .concat(),
    )
}

#[test]
fn vertex_banks_clip_position_and_packed_semantics_keep_their_values() {
    let bytes = fixture();
    let source = read(&bytes).unwrap();
    assert_eq!(source.stage, Stage::Vertex);
    assert_eq!(source.outputs.len(), 4);
    let p = lifted(&bytes);
    assert_eq!(p.stage, Stage::Vertex);
    assert_eq!(p.provenance.target, "vs_3_0");
    assert_eq!(p.provenance.source_hash, Some(iridia_ir::source_hash(&bytes)));
    assert!(p.interface.inputs.iter().all(|v| v.interp.is_none()));
    let outs = &p.interface.outputs;
    assert_eq!(outs[0].system, Some(SystemValue::ClipPosition));
    assert_eq!(
        outs.iter()
            .skip(1)
            .map(|v| (&*v.label.name, v.label.index, v.provenance.d3d.unwrap().mask))
            .collect::<Vec<_>>(),
        [("COLOR", 4, 1), ("TEXCOORD", 3, 6), ("FOG", 0, 8)]
    );
    let mut input = [[0u32; 4]; 16];
    input[15] = [1f32, 2., 3., 1.].map(f32::to_bits);
    let mut constants = [[0u32; 4]; 256];
    constants[255] = [10f32, 20., 30., 40.].map(f32::to_bits);
    let run = execute(&p, &slots(&p, &input), &constants, &mut no_sample);
    assert_eq!(clip(&p, &run), [1f32, 2., 3., 1.].map(f32::to_bits));
    assert_eq!(by_register(&p, &run)[3].map(f32::from_bits), [12., 26., 42., 45.]);
}
