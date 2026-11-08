#![allow(
    clippy::unwrap_used,
    reason = "a real-data gate: a broken input should stop it"
)]
//! The migration gate of shader model 3 on the IR, over every distinct vertex and pixel program of Shaun White
//! Skateboarding's three SDB2 bundles (the oracle set `d3d-disassembler/shaun-white/sdb2`):
//!
//! - each program lifts, IR-validates, prints, parses back to the same text, becomes a control flow graph and is structured
//!   again (each form running alike, bit for bit), and is written as WGSL for browsers by both writers (`iridia-naga` and
//!   `iridia-wgsl`), each text read back and validated by naga's front end (`harness::prepare`);
//! - it runs two ways on the same seeded inputs, constants and texels, and the output bits agree, a NaN with any NaN (IEEE
//!   754 leaves the sign and payload of an arithmetic result's NaN open, and the executors keep different ones: `-x + y`
//!   against `y - x`): B, the IR in `iridia-exec-ir`, and D, the IR compiled into `iridia-exec` (`ProgramShader::from_ir`).
//!   A program whose runs agree only so is counted in a class of its own;
//! - a sample of the programs (one in 128 by file-name hash, every one with `shader.graphics=1`) has the WGSL of both
//!   writers run on Direct3D 12's WARP adapter (`tests/warp`) and held to B's run on the same inputs, a quad of four lanes
//!   for a pixel program: no lane beyond 1e-3 relative on either writer, except at `pow` of a zero base and an exponent of 0 or less (a
//!   program that agrees in every lane once that case is given libm's value is counted in a class of its own,
//!   `warp::Agreement::PowDefined`), or where `warp::EXPECTED` names the program, its source hash and the reason proved
//!   for it. What the harness cannot bind is a named gap, counted.
//!
//! The migration from the naga-based lifter `iridia-d3d9` ran this gate against that lifter's programs in its executor
//! (interfaces, WGSL declarations, and the lifter's own run, A, and its WGSL read back, C); those columns are gone with
//! the lifter.
//!
//! Every failure is counted by class and the test fails on any. Without the oracle set it prints SKIP and passes.
mod harness;
mod inputs;
mod tally;
mod warp;

use harness::{Rng, Texels, View, ir_texels, seeded};
use inputs::{Game, OracleSet, oracle_tree};
use iridia_exec::ProgramShader;
use iridia_ir::interface::{Interface, ResourceId};
use tally::{Tally, class};

/// Seeded runs per program.
const RUNS: u64 = 16;

fn programs() -> Option<Vec<(String, Vec<u8>)>> {
    let tree = oracle_tree(OracleSet::D3dDisassembler, Game::ShaunWhite, "sdb2")?;
    // the programs at the top of the set (Microsoft's listings, `.asm`, beside them are not programs)
    Some(
        tree.names()
            .into_iter()
            .filter(|n| !n.contains('/') && (n.ends_with(".vs") || n.ends_with(".ps")))
            .map(|n| (n.to_owned(), tree.read(n).unwrap()))
            .collect(),
    )
}

/// The lift: the multiply rule is IEEE 754, as `iridia-exec` computes it; the default, zero wins, is held to Direct3D 9 on
/// Windows by the vkd3d cases.
fn lifted(bytes: &[u8]) -> Result<iridia_ir::Program, String> {
    let options = iridia_d3d9_ir::Options {
        multiply: iridia_ir::MulZero::Ieee,
        ..Default::default()
    };
    iridia_d3d9_ir::lift_with(bytes, &options).map_err(|e| format!("lift: {}", e.group()))
}

/// A run of the IR compiled into `iridia-exec` (D).
fn run_exec(
    shader: &ProgramShader,
    interface: &Interface,
    inputs: &[[u32; 4]],
    buffers: &[(u32, &[[u32; 4]])],
) -> Result<View, String> {
    harness::run_exec(shader, interface, inputs, buffers, &mut Texels)
}

/// A run of the IR (B) in `iridia-exec-ir`, and of its graph and the graph structured again, which must match it.
fn run_ir(
    p: &harness::Prepared,
    inputs: &[[u32; 4]],
    buffers: &[(ResourceId, &[[u32; 4]])],
) -> Result<View, String> {
    let mut sampler = ir_texels(&p.ir.interface);
    p.run(
        inputs,
        &mut harness::Bound {
            buffers,
            state: &[],
            sampler: &mut sampler,
            denormals: iridia_ir::Denormals::Any,
        },
    )
}

const AGREE: &str = "B = D on every run";
const AGREE_NAN: &str = "B = D on every run, NaN for NaN (a NaN's sign or payload differs)";
const KILLED: &str = "B = D, killed in both on every run (outputs never compared)";

/// One program through every check but WARP; the class it lands in and the example to show for it, and the prepared
/// program where it got that far.
fn gate(name: &str, bytes: &[u8], runs: &mut (usize, usize)) -> (String, String, Option<harness::Prepared>) {
    let fail = |what: String| (class(&what), format!("{name}: {what}"), None);
    let new = match lifted(bytes) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let prepared = match harness::prepare(new) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let sd = match ProgramShader::from_ir(prepared.ir.clone()) {
        Ok(d) => d,
        Err(e) => return fail(format!("D not compiled: {e}")),
    };
    let interface = &prepared.ir.interface;
    let mut rng = Rng::of(name);
    let mut compared = 0;
    let mut nan_only = false;
    for run in 0..RUNS {
        let i = seeded(interface, &mut rng);
        let rb = run_ir(&prepared, &i.slots, &i.by_id());
        let rd = run_exec(&sd, interface, &i.registers, &i.by_register());
        let (rb, rd) = match (rb, rd) {
            (Ok(b), Ok(d)) => (b, d),
            (b, d) => {
                return fail(format!("not run: B {:?}, D {:?}", b.err(), d.err()));
            }
        };
        if rb.killed && rd.killed {
            runs.1 += 1;
            continue;
        }
        runs.0 += 1;
        compared += 1;
        if !rb.equal(&rd) {
            nan_only = true;
        }
        if !rb.alike(&rd) {
            return (
                "differ: B and D".into(),
                format!("{name} run {run}\n   B {rb:?}\n   D {rd:?}"),
                None,
            );
        }
    }
    let outcome = match (compared, nan_only) {
        (0, _) => KILLED,
        (_, true) => AGREE_NAN,
        _ => AGREE,
    };
    (outcome.into(), name.into(), Some(prepared))
}

/// The WARP comparison's outcome of a program, noted in `tally`: a named gap or a class per writer.
fn on_warp(tally: &mut Tally, name: &str, prepared: &harness::Prepared) {
    let Some(outcome) = warp::compare_sampled(prepared, name, warp::SAMPLE) else {
        return;
    };
    for (what, failing, example) in outcome.classes() {
        tally.note(
            if failing { class(&what) } else { what },
            if failing { example } else { name.to_owned() },
        );
    }
}

#[test]
fn every_program_lifts_round_trips_and_runs_alike_in_both_executors_and_on_warp() {
    let Some(programs) = programs() else { return };
    let mut tally = Tally::default();
    let mut warp_tally = Tally::default();
    let mut runs = (0, 0);
    for (name, bytes) in &programs {
        let (class, example, prepared) = gate(name, bytes, &mut runs);
        tally.note(class, example);
        if let Some(prepared) = prepared {
            on_warp(&mut warp_tally, name, &prepared);
        }
    }
    println!(
        "{} programs; runs compared {}, runs killed in both (outputs not compared) {}",
        programs.len(),
        runs.0,
        runs.1
    );
    tally.print("outcomes");
    warp_tally.print("WARP outcomes");
    assert_eq!(
        tally.failures(&[AGREE, AGREE_NAN, KILLED]),
        0,
        "programs failed the gate"
    );
    assert_eq!(
        warp_tally.failing(|c| c.starts_with("WARP FAILED")),
        0,
        "programs failed on WARP"
    );
    assert_eq!(programs.len(), 7_306);
}
