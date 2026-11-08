#![allow(
    clippy::unwrap_used,
    reason = "a real-data gate: a broken input should stop it"
)]
//! The migration gate of shader model 3 on the IR, over every distinct vertex and pixel program of Shaun White
//! Skateboarding's three SDB2 bundles (the oracle set `d3d-disassembler/shaun-white/sdb2`), against `iridia-d3d9`, the
//! naga-based lifter it replaces:
//!
//! - each program lifts, IR-validates, prints, parses back to the same text, lowers and writes WGSL for browsers;
//! - its interface equals the older lifter's, and so do its WGSL declarations (structures, bindings, globals, the
//!   entry point's signature);
//! - it runs four ways on the same seeded inputs, constants and texels, and the output bits agree, a NaN with any NaN
//!   (IEEE 754 leaves the sign and payload of an arithmetic result's NaN open, and the executors keep different ones:
//!   `-x + y` against `y - x`): A, the older lift in `iridia-exec`; B, the IR in `iridia-exec-ir`; C, the new WGSL read
//!   back by naga's front end in `iridia-exec`; D, the IR compiled into `iridia-exec` itself
//!   (`ProgramShader::from_ir`). A program whose runs agree only so is counted in a class of its own;
//!
//! - a sample of the programs (one in 128 by file-name hash, every one with `shader.graphics=1`) has the WGSL of both
//!   writers run on Direct3D 12's WARP adapter (`tests/warp`) and held to B's run on the same inputs, a quad of four lanes
//!   for a pixel program: no lane beyond 1e-3 relative on either writer, except at `pow` of a zero base and an exponent of 0 or less (a
//!   program that agrees in every lane once that case is given libm's value is counted in a class of its own,
//!   `warp::Outcome::Ran`). What the harness cannot bind is a named gap, counted.
//!
//! Every failure is counted by class and the test fails on any. Without the oracle set it prints SKIP and passes.
mod harness;
mod inputs;
mod tally;
mod warp;

use harness::{Rng, Texels, View, ir_texels, round_trip, seeded};
use inputs::{Game, OracleSet, oracle_tree};
use iridia_exec::ProgramShader;
use iridia_ir::interface::{Interface, ResourceId, ResourceKind};
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

/// The IR pipeline: lifted, promoted and simplified. The lift takes the older lifter's multiply rule, IEEE 754, so the
/// two mean the same; the default, zero wins, is held to Direct3D 9 on Windows by the vkd3d cases.
fn lifted(bytes: &[u8]) -> Result<iridia_ir::Program, String> {
    let options = iridia_d3d9_ir::Options {
        multiply: iridia_ir::MulZero::Ieee,
        ..Default::default()
    };
    let mut p = iridia_d3d9_ir::lift_with(bytes, &options).map_err(|e| format!("lift: {}", e.group()))?;
    iridia_passes::promote(&mut p.entry).map_err(|e| format!("promote: {e}"))?;
    iridia_passes::simplify(&mut p.entry);
    Ok(p)
}

/// The structures, globals and entry-point signature; module diagnostic policy is not part of that interface.
fn declarations(wgsl: &str) -> String {
    let mut out = String::new();
    let mut depth = 0i64;
    for line in wgsl.lines() {
        if depth == 0 && line.starts_with("diagnostic(") {
            continue;
        }
        let in_fn = depth > 0;
        if line.starts_with("fn ") || !in_fn {
            out.push_str(line);
            out.push('\n');
        }
        if line.starts_with("fn ") || in_fn {
            depth += line.matches('{').count() as i64 - line.matches('}').count() as i64;
        }
    }
    out
}

#[test]
fn diagnostic_policy_is_not_a_declaration_but_bindings_and_signatures_are() {
    let text = "@group(0) @binding(0) var<uniform> cb0: vec4<f32>;\nfn main() {\n    return;\n}\n";
    assert_eq!(
        declarations(text),
        declarations(&format!("diagnostic(off, derivative_uniformity);\n{text}"))
    );
    assert_ne!(
        declarations(text),
        declarations(&text.replace("binding(0)", "binding(1)"))
    );
    assert_ne!(
        declarations(text),
        declarations(&text.replace("main()", "main(x: f32)"))
    );
}

/// A run of the older lift (A) or of the WGSL read back (C) in `iridia-exec`.
fn run_old(
    shader: &ProgramShader,
    interface: &Interface,
    inputs: &[[u32; 4]],
    buffers: &[(u32, &[[u32; 4]])],
) -> Result<View, String> {
    harness::run_naga(shader, interface, inputs, buffers, &mut Texels)
}

/// A run of the IR (B) in `iridia-exec-ir`.
fn run_ir(
    p: &iridia_ir::Program,
    inputs: &[[u32; 4]],
    buffers: &[(ResourceId, &[[u32; 4]])],
) -> Result<View, String> {
    let mut sampler = ir_texels(&p.interface);
    harness::run_ir(
        p,
        inputs,
        &mut harness::Bound {
            buffers,
            state: &[],
            sampler: &mut sampler,
            denormals: iridia_ir::Denormals::Any,
        },
    )
}

/// What WGSL text cannot say, put back from the older module by position: a name ending in a digit (naga's namer
/// writes `cb0_` for `cb0`), and the entry point's argument and result member names and bindings (WGSL cannot say a
/// varying has no interpolation, so naga's front end supplies the default). Anything else that differs is left for
/// the executor to refuse or to compute differently.
fn normalise(m: &mut naga::Module, old: &naga::Module) -> Result<(), String> {
    for (_, g) in m.global_variables.iter_mut() {
        if let Some(n) = &mut g.name
            && n.len() > 1
            && n.ends_with('_')
            && n[..n.len() - 1].ends_with(|c: char| c.is_ascii_digit())
        {
            n.pop();
        }
    }
    let (Some(ep), Some(oe)) = (m.entry_points.first_mut(), old.entry_points.first()) else {
        return Err("no entry point".into());
    };
    if ep.function.arguments.len() != oe.function.arguments.len() {
        return Err("argument count".into());
    }
    for (a, o) in ep.function.arguments.iter_mut().zip(&oe.function.arguments) {
        a.name.clone_from(&o.name);
        a.binding.clone_from(&o.binding);
    }
    let (Some(r), Some(or)) = (&ep.function.result, &oe.function.result) else {
        return Err("no result".into());
    };
    let (rt, ort) = (r.ty, or.ty);
    let naga::TypeInner::Struct { members: om, .. } = &old.types[ort].inner else {
        return Err("the older result is not a structure".into());
    };
    let mut t = m.types[rt].clone();
    let naga::TypeInner::Struct { members, .. } = &mut t.inner else {
        return Err("the result is not a structure".into());
    };
    if members.len() != om.len() {
        return Err("member count".into());
    }
    for (a, o) in members.iter_mut().zip(om) {
        a.name.clone_from(&o.name);
        a.binding.clone_from(&o.binding);
    }
    t.name.clone_from(&old.types[ort].name);
    m.types.replace(rt, t);
    Ok(())
}

/// The program the new WGSL is, for `iridia-exec`: the older lift's, with its module read from the WGSL and each
/// resource bound to the global of its name.
fn from_wgsl(old: &iridia::Program, wgsl: &str) -> Result<iridia::Program, String> {
    let mut module = naga::front::wgsl::parse_str(wgsl).map_err(|e| format!("naga's WGSL front end: {e}"))?;
    normalise(&mut module, &old.module)?;
    let mut bindings = Vec::new();
    for r in &old.interface.resources {
        let d = r.provenance.d3d.unwrap();
        let name = match r.kind {
            ResourceKind::ConstantBuffer { .. } => format!("cb{}", d.register),
            ResourceKind::Texture { .. } => format!("t{}", d.register),
            ResourceKind::Sampler { .. } => format!("s{}", d.register),
            _ => return Err("an unexpected resource".into()),
        };
        let g = module
            .global_variables
            .iter()
            .find(|(_, g)| g.name.as_deref() == Some(&name))
            .map(|(h, _)| h)
            .ok_or_else(|| format!("no global {name}"))?;
        bindings.push((r.id, g));
    }
    let mut p = old.clone();
    p.module = module;
    p.bindings = bindings;
    Ok(p)
}

const AGREE: &str = "A = B = C = D on every run";
const AGREE_NAN: &str = "A = B = C = D on every run, NaN for NaN (a NaN's sign or payload differs)";
const KILLED: &str = "A = B = C = D, killed in all four on every run (outputs never compared)";

/// One program through every check; the class it lands in and the example to show for it.
fn gate(name: &str, bytes: &[u8], runs: &mut (usize, usize)) -> (String, String) {
    let fail = |what: String| (class(&what), format!("{name}: {what}"));
    let new = match lifted(bytes) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    if let Err(e) = iridia_ir::validate(&new) {
        return fail(format!("IR validation: {}", e[0]));
    }
    if let Err(e) = round_trip(&new.entry) {
        return fail(e);
    }
    let wgsl = match iridia_naga::wgsl(&new, &iridia_naga::Profile::BROWSER) {
        Ok(w) => w,
        Err(e) => return fail(format!("WGSL: {e}")),
    };
    let old = iridia_d3d9::lift(bytes).unwrap();
    if old.interface != new.interface {
        return fail("interface differs from iridia-d3d9's".into());
    }
    if declarations(&iridia::emit::wgsl(&old).unwrap()) != declarations(&wgsl) {
        return fail("WGSL declarations differ from iridia-d3d9's".into());
    }
    let c = match from_wgsl(&old, &wgsl) {
        Ok(p) => p,
        Err(e) => return fail(format!("C not built: {e}")),
    };
    let (sa, sc) = match (ProgramShader::new(old.clone()), ProgramShader::new(c)) {
        (Ok(a), Ok(c)) => (a, c),
        (a, c) => {
            return fail(format!(
                "not compiled for iridia-exec: A {:?}, C {:?}",
                a.err().map(|e| e.to_string()),
                c.err().map(|e| e.to_string())
            ));
        }
    };
    let sd = match ProgramShader::from_ir(new.clone()) {
        Ok(d) => d,
        Err(e) => return fail(format!("D not compiled: {e}")),
    };
    let interface = &old.interface;
    let mut rng = Rng::of(name);
    let mut compared = 0;
    let mut nan_only = false;
    for run in 0..RUNS {
        let i = seeded(interface, &mut rng);
        let ra = run_old(&sa, interface, &i.registers, &i.by_register());
        let rb = run_ir(&new, &i.slots, &i.by_id());
        let rc = run_old(&sc, interface, &i.registers, &i.by_register());
        let rd = run_old(&sd, interface, &i.registers, &i.by_register());
        let (ra, rb, rc, rd) = match (ra, rb, rc, rd) {
            (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
            (a, b, c, d) => {
                return fail(format!(
                    "not run: A {:?}, B {:?}, C {:?}, D {:?}",
                    a.err(),
                    b.err(),
                    c.err(),
                    d.err()
                ));
            }
        };
        if ra.killed && rb.killed && rc.killed && rd.killed {
            runs.1 += 1;
            continue;
        }
        runs.0 += 1;
        compared += 1;
        if !(ra.equal(&rb) && rb.equal(&rc) && ra.equal(&rd)) {
            nan_only = true;
        }
        if !(ra.alike(&rb) && rb.alike(&rc) && ra.alike(&rd)) {
            return (
                format!(
                    "differ: A=B {} B=C {} A=C {} A=D {}",
                    ra.alike(&rb),
                    rb.alike(&rc),
                    ra.alike(&rc),
                    ra.alike(&rd)
                ),
                format!("{name} run {run}\n   A {ra:?}\n   B {rb:?}\n   C {rc:?}\n   D {rd:?}"),
            );
        }
    }
    let outcome = match (compared, nan_only) {
        (0, _) => KILLED,
        (_, true) => AGREE_NAN,
        _ => AGREE,
    };
    (outcome.into(), name.into())
}

/// The WARP comparison's outcome of a program, noted in `tally`: a named gap or a class per writer.
fn on_warp(tally: &mut Tally, name: &str, bytes: &[u8]) {
    let Ok(program) = lifted(bytes) else { return };
    let Ok(prepared) = harness::prepare(program) else {
        return;
    };
    let Some(outcome) = warp::compare_sampled(&prepared, name, warp::SAMPLE) else {
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
fn every_program_lifts_round_trips_and_runs_as_iridia_d3d9s() {
    let Some(programs) = programs() else { return };
    let mut tally = Tally::default();
    let mut warp_tally = Tally::default();
    let mut runs = (0, 0);
    for (name, bytes) in &programs {
        let (class, example) = gate(name, bytes, &mut runs);
        tally.note(class, example);
        on_warp(&mut warp_tally, name, bytes);
    }
    println!(
        "{} programs; runs compared {}, runs killed in all three (outputs not compared) {}",
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
