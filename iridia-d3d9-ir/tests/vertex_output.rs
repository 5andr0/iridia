#![allow(clippy::unwrap_used)]
//! A whole `vs_3_0` program's outputs worked by hand, on both executors: a partial overwrite of a temporary that reads
//! its own earlier lanes keeps each lane where the original puts it. (The slices of one output that `iridia-d3d9`
//! projects stay with that crate.)
mod executors;

use executors::{by_register, execute, lifted, no_sample, slots};

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

#[test]
fn partial_overwrites_and_aliasing_preserve_original_lane_positions() {
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
    let inputs = [
        [0f32, 0., 0., 1.].map(f32::to_bits),
        [2f32, 3., 4., 5.].map(f32::to_bits),
        [0; 4],
    ];
    let mut constants = [[0u32; 4]; 11];
    constants[10] = [7f32, 11., 0., 0.].map(f32::to_bits);
    let p = lifted(&raw);
    let run = execute(&p, &slots(&p, &inputs), &constants, &mut no_sample);
    assert_eq!(by_register(&p, &run)[3].map(f32::from_bits), [21., 32., 4., 5.]);
}
