#![allow(clippy::unwrap_used, clippy::indexing_slicing)]
//! Programs lifted onto the IR ([`iridia_dxil::ir`]) and run on the executor against the memory Direct3D 12 WARP leaves,
//! recorded beside each fixture (`<name>.warp.json`, written by `<name>.warp.cpp` from the same starting words): every
//! word must agree bit for bit.
use iridia::D3dClass;
use iridia_exec::{Backend, Element, Executable, ImageSample, Resources};
use std::collections::BTreeMap;
use std::path::Path;

struct NoBackend;
impl Backend for NoBackend {
    fn sample(&mut self, _: ImageSample<Element>) -> Result<[f32; 4], String> {
        Err("no sampling in this fixture".into())
    }
}

/// Word `k` of buffer `b` as the WARP harness starts it: two halves in [1, 2) each, different in every buffer and word.
fn seed(b: u32, k: u32) -> u32 {
    let half = |j: u32| 0x3c00 + (j * 37) % 0x0400;
    let j = (b * 1000 + k) * 2;
    half(j) | (half(j + 1) << 16)
}

/// `buffer_lanes16.hlsl`: 16-bit lanes of raw and structured buffers read and written at offsets that share and
/// straddle words (two threads' halves of one word among them), each buffer's words after one group of 8 as WARP's.
#[test]
fn sixteen_bit_buffer_lanes_leave_warps_memory() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let bytes = std::fs::read(dir.join("buffer_lanes16.dxil")).unwrap();
    let mut program = iridia_dxil::ir::lift(&bytes).unwrap();
    // the lift's control flow graph, structured as the executor takes it
    iridia_passes::structurize(&mut program.entry).unwrap();
    iridia_passes::break_flags(&mut program.entry).unwrap();
    let exe = Executable::from_ir(&program).unwrap();
    let warp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("buffer_lanes16.warp.json")).unwrap())
            .unwrap();
    let mut buffers = BTreeMap::new();
    let mut ids = Vec::new();
    for register in 0..3u32 {
        let id = program
            .interface
            .resources
            .iter()
            .find(|r| {
                r.provenance
                    .d3d
                    .is_some_and(|d| d.class == D3dClass::UnorderedAccess && d.register == register)
            })
            .unwrap_or_else(|| panic!("no u{register}"))
            .id;
        let len = warp[format!("u{register}")].as_array().unwrap().len() as u32;
        buffers.insert(id, (0..len).map(|k| seed(register, k)).collect::<Vec<u32>>());
        ids.push(id);
    }
    let resources = Resources::default();
    let bound = exe.bind_resources(&resources).unwrap();
    exe.dispatch(&bound, &mut NoBackend, &mut buffers, [1, 1, 1], 32)
        .unwrap();
    let mut differ = Vec::new();
    for (register, id) in (0..).zip(&ids) {
        let expected: Vec<u32> = warp[format!("u{register}")]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| u32::try_from(w.as_u64().unwrap()).unwrap())
            .collect();
        for (k, (ours, theirs)) in buffers[id].iter().zip(&expected).enumerate() {
            if ours != theirs {
                differ.push(format!("u{register} word {k}: {ours:08x}, WARP {theirs:08x}"));
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} words differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}
