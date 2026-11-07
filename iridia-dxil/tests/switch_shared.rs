#![allow(clippy::unwrap_used, clippy::indexing_slicing)]
//! A switch three of whose cases go straight to the block after it (`switch_shared.hlsl`): dxc gives that block's phis
//! an entry for each of the three edges, the same value on each (`phi [7.0, %0]` three times). The IR lift makes one
//! target for the block, so every edge carries the same arguments; the program goes through `iridia_passes::prepare`
//! (whose `structurize` refuses a switch that gives one block two sets of arguments) and runs as the HLSL says.
use iridia_exec::{PixelShader, ProgramShader, Sample};

fn no_samples(_: Sample) -> Result<[f32; 4], String> {
    Err("no sampling in this program".into())
}

#[test]
fn every_edge_of_a_switch_to_one_block_carries_the_same_arguments() {
    let bytes = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/switch_shared.dxil"),
    )
    .unwrap();
    let lifted = iridia_dxil::ir::lift(&bytes).unwrap();
    // the switch's targets: a block named twice has one set of arguments
    let iridia_ir::Body::Cfg(blocks) = &lifted.entry.body else {
        panic!("not a graph")
    };
    let switch = blocks
        .iter()
        .find(|b| matches!(b.term, iridia_ir::Terminator::Switch { .. }))
        .unwrap();
    let targets = switch.term.targets();
    assert!(targets.len() >= 4, "{targets:?}");
    for t in &targets {
        for u in targets.iter().filter(|u| u.block == t.block) {
            assert_eq!(u.args, t.args, "block {}", t.block);
        }
    }
    let mut program = lifted;
    iridia_passes::prepare(&mut program).unwrap();
    let shader = ProgramShader::from_ir(program).unwrap();
    let a = 1.5f32;
    for s in 0..8u32 {
        let mut inputs = vec![[0u32; 4]; shader.input_count().max(1)];
        for v in &shader.interface().inputs {
            let d = v.provenance.d3d.unwrap();
            let lane = d.mask.trailing_zeros() as usize;
            inputs[d.register as usize][lane] = match v.label.name.as_str() {
                "S" => s,
                "A" => a.to_bits(),
                other => panic!("input {other}"),
            };
        }
        let out = shader.execute(&inputs, &[], &mut no_samples).unwrap();
        let want = match s {
            0 | 2 | 5 => [7.0, a * 2.0],
            1 => [a * 3.0, 4.0],
            _ => [a - 2.0, a * 2.0],
        };
        assert_eq!(
            out.outputs[0].map(f32::from_bits),
            [want[0], want[1], 0.0, 1.0],
            "s = {s}"
        );
    }
}
