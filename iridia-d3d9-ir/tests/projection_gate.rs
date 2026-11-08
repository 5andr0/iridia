#![allow(
    clippy::unwrap_used,
    reason = "a real-data gate: a broken input should stop it"
)]
//! The derived programs of shader model 3 on the IR (`project`, `reencode`, `project_vertex_output`), over every distinct
//! vertex and pixel program of Shaun White Skateboarding's three SDB2 bundles (the oracle set
//! `d3d-disassembler/shaun-white/sdb2`):
//!
//! - up to three cuts of each pixel program (after an instruction that writes three or more lanes of a temporary) and
//!   every declared output of each vertex program are derived; each derivation either is a program, which validates and
//!   runs alike (the IR in `iridia-exec-ir`, compiled into `iridia-exec`) on the same seeded inputs, constants and
//!   texels, or is refused, by reason;
//! - each IR derivation is held to the whole program as well: a vertex output's selected lanes are what the whole program
//!   writes there, and a pixel program's `reencode` given what its `project` computes is the whole program;
//! - the harness sees a change: a projection with its lanes permuted differs from the cut's own on most programs.
//!
//! The migration from the naga-based lifter `iridia-d3d9` ran the first of these against that lifter's derivations (the
//! two either both refused or both gave a program, and the programs ran alike); that comparison is gone with the lifter.
//!
//! A vertex program with flow control is refused (the older slice dropped the flow control and kept the writes inside it;
//! the IR means the program). Every other refusal is counted by reason and fails the test, and so does every other
//! failure, counted by class. Without the oracle set it prints SKIP and passes.
mod harness;
mod inputs;
mod tally;

use harness::{Bound, Rng, Texels, View, ir_texels, lanes_of, run_exec, run_ir};
use inputs::{Game, OracleSet, oracle_tree};
use iridia_d3d9_ir::{Opcode, Options, Register, Shader};
use iridia_exec::ProgramShader;
use iridia_ir::{
    MulZero,
    interface::{Interface, ResourceId, ResourceKind},
};
use tally::{Tally, class};

/// Seeded runs per derived program.
const RUNS: u64 = 8;
/// Cuts per pixel program.
const CUTS: usize = 3;

fn programs() -> Option<Vec<(String, Vec<u8>)>> {
    let tree = oracle_tree(OracleSet::D3dDisassembler, Game::ShaunWhite, "sdb2")?;
    Some(
        tree.names()
            .into_iter()
            .filter(|n| !n.contains('/') && (n.ends_with(".vs") || n.ends_with(".ps")))
            .map(|n| (n.to_owned(), tree.read(n).unwrap()))
            .collect(),
    )
}

/// The multiply rule IEEE 754, as `iridia-exec` computes it.
fn ieee() -> Options {
    Options {
        multiply: MulZero::Ieee,
        ..Options::default()
    }
}

/// The inputs of one seeded run: by register (the IR compiled into `iridia-exec`) and by slot
/// (`iridia-exec-ir`), and the constant banks.
struct Inputs {
    registers: Vec<[u32; 4]>,
    slots: Vec<[u32; 4]>,
    banks: Vec<(ResourceId, u32, Vec<[u32; 4]>)>,
}

/// The rows of the constant buffer at a D3D register in an interface: what a program of it reads.
fn bank_rows(interface: &Interface, register: u32) -> u32 {
    interface
        .resources
        .iter()
        .filter_map(|r| match &r.kind {
            ResourceKind::ConstantBuffer { layout }
                if r.provenance.d3d.is_some_and(|d| d.register == register) =>
            {
                Some(layout.size.div_ceil(16))
            }
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// The inputs of one seeded run of a program of `interface` and of the programs of `peers` (derived from the same
/// bytecode): each constant bank has as many rows as the largest of them reads, so that every program is given what it
/// needs.
fn seeded(interface: &Interface, peers: &[&Interface], rng: &mut Rng) -> Inputs {
    let mut registers = vec![[0u32; 4]; 32];
    let mut slots = Vec::new();
    for v in &interface.inputs {
        let d = v.provenance.d3d.unwrap();
        let mut packed = [0u32; 4];
        for (k, l) in lanes_of(d.mask).into_iter().enumerate() {
            let x = rng.float().to_bits();
            registers[d.register as usize][l] = x;
            packed[k] = x;
        }
        slots.push(packed);
    }
    let mut banks = Vec::new();
    for r in &interface.resources {
        if let ResourceKind::ConstantBuffer { layout } = &r.kind {
            let register = r.provenance.d3d.unwrap().register;
            let rows = peers
                .iter()
                .map(|p| bank_rows(p, register))
                .fold(layout.size.div_ceil(16), u32::max);
            let data = (0..rows)
                .map(|_| [(); 4].map(|_| rng.float().to_bits()))
                .collect();
            banks.push((r.id, register, data));
        }
    }
    Inputs {
        registers,
        slots,
        banks,
    }
}

impl Inputs {
    fn by_register(&self) -> Vec<(u32, &[[u32; 4]])> {
        self.banks.iter().map(|(_, r, d)| (*r, d.as_slice())).collect()
    }
    fn by_id(&self) -> Vec<(ResourceId, &[[u32; 4]])> {
        self.banks.iter().map(|(i, _, d)| (*i, d.as_slice())).collect()
    }
}

fn run_compiled(shader: &ProgramShader, interface: &Interface, i: &Inputs) -> Result<View, String> {
    run_exec(shader, interface, &i.registers, &i.by_register(), &mut Texels)
}

fn run_new(p: &iridia_ir::Program, i: &Inputs) -> Result<View, String> {
    let mut sampler = ir_texels(&p.interface);
    run_ir(
        p,
        &i.slots,
        &mut Bound {
            buffers: &i.by_id(),
            state: &[],
            sampler: &mut sampler,
            denormals: iridia_ir::Denormals::Any,
        },
    )
}

const AGREE: &str = "the derivation runs alike in both executors on every run";
const FLOW: &str = "refused: a vertex program with flow control";

/// The cuts of a pixel program: after an instruction that writes three or more lanes of a temporary, the first three of
/// them, at most [`CUTS`] spread over the program.
fn cuts(shader: &Shader) -> Vec<iridia_d3d9_ir::Projection> {
    let all: Vec<_> = shader
        .instructions
        .iter()
        .filter_map(|i| {
            let d = i.destination?;
            let Register::Temporary(temporary) = d.register else {
                return None;
            };
            let lanes: Vec<u8> = (0u8..4).filter(|n| d.mask & (1 << n) != 0).take(3).collect();
            (lanes.len() == 3 && !matches!(i.opcode, Opcode::TexKill)).then(|| iridia_d3d9_ir::Projection {
                after_word: i.word,
                temporary,
                lanes: [lanes[0], lanes[1], lanes[2]],
            })
        })
        .collect();
    let step = all.len().div_ceil(CUTS).max(1);
    all.into_iter().step_by(step).collect()
}

/// One derivation: the class it lands in and the example to show for it.
fn compare(
    name: &str,
    what: &str,
    new: Result<iridia_ir::Program, String>,
    runs: &mut usize,
) -> (String, String) {
    let fail = |why: String| (class(&why), format!("{name} {what}: {why}"));
    let new = match new {
        Err(e) if e.contains("flow control") => return (FLOW.into(), format!("{name} {what}: {e}")),
        Err(e) => return (format!("refused: {}", class(&e)), format!("{name} {what}: {e}")),
        Ok(n) => n,
    };
    if let Err(e) = iridia_ir::validate(&new) {
        return fail(format!("IR validation: {}", e[0]));
    }
    let sd = match ProgramShader::from_ir(new.clone()) {
        Ok(d) => d,
        Err(e) => return fail(format!("not compiled for iridia-exec: {e}")),
    };
    let mut rng = Rng::of(&format!("{name} {what}"));
    for run in 0..RUNS {
        let i = seeded(&new.interface, &[], &mut rng);
        let (rb, rd) = match (run_new(&new, &i), run_compiled(&sd, &new.interface, &i)) {
            (Ok(b), Ok(d)) => (b, d),
            (b, d) => return fail(format!("not run: IR {:?}, compiled {:?}", b.err(), d.err())),
        };
        *runs += 1;
        if !rb.alike(&rd) {
            return (
                "differ: IR and compiled".into(),
                format!("{name} {what} run {run}\n   IR       {rb:?}\n   compiled {rd:?}"),
            );
        }
    }
    (AGREE.into(), format!("{name} {what}"))
}

fn new_ok<T>(r: Result<T, iridia_d3d9_ir::Error>) -> Result<T, String> {
    r.map_err(|e| e.to_string())
}

#[test]
fn the_pixel_programs_cuts_derive_and_run_alike_in_both_executors() {
    let Some(programs) = programs() else { return };
    let (mut tally, mut runs, mut derived) = (Tally::default(), 0usize, 0usize);
    for (name, bytes) in programs.iter().filter(|(n, _)| n.ends_with(".ps")) {
        let Ok(shader) = iridia_d3d9_ir::read(bytes) else {
            continue;
        };
        if shader.version.major != 3 {
            continue;
        }
        for cut in cuts(&shader) {
            let what = format!(
                "cut after word {} of r{} {:?}",
                cut.after_word, cut.temporary, cut.lanes
            );
            for tail in [false, true] {
                let new = if tail {
                    new_ok(iridia_d3d9_ir::reencode_with(bytes, cut, &ieee()))
                } else {
                    new_ok(iridia_d3d9_ir::project_with(bytes, cut, &ieee()))
                };
                derived += usize::from(new.is_ok());
                let what = format!("{} {what}", if tail { "reencode" } else { "project" });
                let (c, example) = compare(name, &what, new, &mut runs);
                tally.note(c, example);
            }
        }
    }
    println!("{derived} derived programs, {runs} runs compared");
    tally.print("outcomes");
    assert!(derived > 0);
    assert_eq!(tally.failures(&[AGREE]), 0, "derivations failed the gate");
}

#[test]
fn the_vertex_outputs_slices_derive_and_run_alike_in_both_executors() {
    let Some(programs) = programs() else { return };
    let (mut tally, mut runs, mut derived) = (Tally::default(), 0usize, 0usize);
    for (name, bytes) in programs.iter().filter(|(n, _)| n.ends_with(".vs")) {
        let Ok(shader) = iridia_d3d9_ir::read(bytes) else {
            continue;
        };
        if shader.version.major != 3 {
            continue;
        }
        for o in &shader.outputs {
            let Register::Output(register) = o.register else {
                continue;
            };
            let what = format!("output o{register} mask {:#x}", o.mask);
            let new = new_ok(iridia_d3d9_ir::project_vertex_output_with(
                bytes,
                register,
                o.mask,
                &ieee(),
            ));
            derived += usize::from(new.is_ok());
            let (c, example) = compare(name, &what, new, &mut runs);
            tally.note(c, example);
        }
    }
    println!("{derived} derived programs, {runs} runs compared");
    tally.print("outcomes");
    assert!(derived > 0);
    assert_eq!(tally.failures(&[AGREE, FLOW]), 0, "derivations failed the gate");
}

/// The whole program, lifted as the derivations are (the multiply rule IEEE 754).
fn whole(bytes: &[u8]) -> Option<iridia_ir::Program> {
    iridia_d3d9_ir::lift_with(bytes, &ieee()).ok()
}

/// What a run of `interface` wrote to a lane of the output at a register: from the output the register is declared as,
/// which the view keeps by slot (the clip position apart, in the lanes it was written in).
fn written_lane(view: &View, interface: &Interface, register: u32, lane: usize) -> Option<u32> {
    interface.outputs.iter().enumerate().find_map(|(slot, v)| {
        let d = v
            .provenance
            .d3d
            .filter(|d| d.register == register && d.mask & (1 << lane) != 0)?;
        match v.system {
            Some(iridia_ir::interface::SystemValue::ClipPosition) => {
                let packed = (d.mask & ((1 << lane) - 1)).count_ones() as usize;
                view.position.map(|p| p[packed])
            }
            _ => view.outputs.get(&slot).map(|l| l[lane]),
        }
    })
}

#[test]
fn a_vertex_outputs_selected_lanes_are_what_the_whole_program_writes_there() {
    let Some(programs) = programs() else { return };
    let (mut tally, mut compared) = (Tally::default(), 0usize);
    for (name, bytes) in programs.iter().filter(|(n, _)| n.ends_with(".vs")) {
        let Ok(shader) = iridia_d3d9_ir::read(bytes) else {
            continue;
        };
        let Some(full) = whole(bytes) else { continue };
        if shader.version.major != 3 {
            continue;
        }
        for o in &shader.outputs {
            let Register::Output(register) = o.register else {
                continue;
            };
            let Ok(slice) = iridia_d3d9_ir::project_vertex_output_with(bytes, register, o.mask, &ieee())
            else {
                continue;
            };
            let what = format!("{name} output o{register}");
            let mut rng = Rng::of(&what);
            let mut agree = true;
            let mut why = String::new();
            for _ in 0..RUNS {
                let i = seeded(&full.interface, &[&slice.interface], &mut rng);
                let j = Inputs {
                    registers: i.registers.clone(),
                    // the slice keeps the program's inputs: the same slots
                    slots: i.slots.clone(),
                    // the slice's constant buffers are its own resources, by the same registers
                    banks: i
                        .banks
                        .iter()
                        .filter_map(|(_, r, d)| {
                            let id = slice.interface.resources.iter().find(|x| {
                                matches!(x.kind, ResourceKind::ConstantBuffer { .. })
                                    && x.provenance.d3d.is_some_and(|p| p.register == *r)
                            })?;
                            Some((id.id, *r, d.clone()))
                        })
                        .collect(),
                };
                let (rf, rs) = match (run_new(&full, &i), run_new(&slice, &j)) {
                    (Ok(rf), Ok(rs)) => (rf, rs),
                    (a, b) => {
                        why = format!("not run: whole {:?}, slice {:?}", a.err(), b.err());
                        agree = false;
                        break;
                    }
                };
                let Some(got) = rs.outputs.get(&0) else {
                    why = "the slice wrote no colour".into();
                    agree = false;
                    break;
                };
                for lane in lanes_of(o.mask) {
                    let Some(w) = written_lane(&rf, &full.interface, register, lane) else {
                        why = format!("the whole program wrote no lane {lane}");
                        agree = false;
                        break;
                    };
                    let g = got[lane];
                    if !(w == g || f32::from_bits(w).is_nan() && f32::from_bits(g).is_nan()) {
                        why = format!("lane {lane}: whole {w:#x}, slice {g:#x}");
                        agree = false;
                    }
                }
                // the placeholders
                for lane in (0..4).filter(|l| o.mask & (1 << l) == 0) {
                    if got[lane] != [0.0f32, 0.0, 0.0, 1.0][lane].to_bits() {
                        why = format!("placeholder lane {lane}: {:#x}", got[lane]);
                        agree = false;
                    }
                }
                compared += 1;
            }
            tally.note(
                if agree {
                    "the slice's lanes are the whole program's".to_string()
                } else {
                    format!("the slice differs from the whole program: {}", class(&why))
                },
                format!("{what}: {why}"),
            );
        }
    }
    println!("{compared} runs compared");
    tally.print("outcomes");
    assert!(compared > 0);
    assert_eq!(tally.failures(&["the slice's lanes are the whole program's"]), 0);
}

#[test]
fn a_pixel_programs_reencode_given_what_project_computes_is_the_whole_program() {
    let Some(programs) = programs() else { return };
    let (mut tally, mut compared) = (Tally::default(), 0usize);
    for (name, bytes) in programs.iter().filter(|(n, _)| n.ends_with(".ps")) {
        let Ok(shader) = iridia_d3d9_ir::read(bytes) else {
            continue;
        };
        let Some(full) = whole(bytes) else { continue };
        if shader.version.major != 3 {
            continue;
        }
        for cut in cuts(&shader) {
            let what = format!(
                "{name} cut after word {} of r{} {:?}",
                cut.after_word, cut.temporary, cut.lanes
            );
            let (Ok(prefix), Ok(tail)) = (
                iridia_d3d9_ir::project_with(bytes, cut, &ieee()),
                iridia_d3d9_ir::reencode_with(bytes, cut, &ieee()),
            ) else {
                continue;
            };
            let derived = tail
                .interface
                .inputs
                .iter()
                .position(|v| v.label.name == "DERIVED_RGB")
                .expect("reencode's input");
            let mut rng = Rng::of(&what);
            let mut agree = true;
            for _ in 0..RUNS {
                let i = seeded(&full.interface, &[&prefix.interface, &tail.interface], &mut rng);
                let (Ok(rf), Ok(rp)) = (run_new(&full, &i), run_new(&prefix, &i)) else {
                    agree = false;
                    break;
                };
                let Some(rgb) = rp.outputs.get(&0) else {
                    // the prefix is killed
                    continue;
                };
                // the tail takes the same inputs, and the projected lanes in its new input
                let mut slots = i.slots.clone();
                slots.insert(derived, [rgb[0], rgb[1], rgb[2], 0]);
                let j = Inputs {
                    registers: Vec::new(),
                    slots,
                    banks: i.banks.clone(),
                };
                let Ok(rt) = run_new(&tail, &j) else {
                    agree = false;
                    break;
                };
                agree &= rf.alike(&rt);
                compared += 1;
            }
            tally.note(
                if agree {
                    "the reencoded program given the projected lanes is the whole program"
                } else {
                    "the reencoded program differs from the whole program"
                },
                what,
            );
        }
    }
    println!("{compared} runs compared");
    tally.print("outcomes");
    assert!(compared > 0);
    assert_eq!(
        tally.failures(&["the reencoded program given the projected lanes is the whole program"]),
        0
    );
}

#[test]
fn the_comparison_sees_a_projection_with_its_lanes_permuted() {
    let Some(programs) = programs() else { return };
    let (mut cuts_made, mut seen) = (0usize, 0usize);
    for (name, bytes) in programs.iter().filter(|(n, _)| n.ends_with(".ps")) {
        let Ok(shader) = iridia_d3d9_ir::read(bytes) else {
            continue;
        };
        if shader.version.major != 3 {
            continue;
        }
        let Some(cut) = cuts(&shader).into_iter().next() else {
            continue;
        };
        let permuted = iridia_d3d9_ir::Projection {
            lanes: [cut.lanes[1], cut.lanes[2], cut.lanes[0]],
            ..cut
        };
        let (Ok(plain), Ok(permuted)) = (
            iridia_d3d9_ir::project_with(bytes, cut, &ieee()),
            iridia_d3d9_ir::project_with(bytes, permuted, &ieee()),
        ) else {
            continue;
        };
        cuts_made += 1;
        let mut rng = Rng::of(name);
        let mut differs = false;
        for _ in 0..RUNS {
            let i = seeded(&permuted.interface, &[&plain.interface], &mut rng);
            if let (Ok(a), Ok(b)) = (run_new(&plain, &i), run_new(&permuted, &i)) {
                differs |= !a.alike(&b);
            }
        }
        seen += usize::from(differs);
    }
    println!("{seen} of {cuts_made} programs differ once the lanes are permuted");
    assert!(cuts_made > 0);
    assert!(seen * 4 >= cuts_made * 3, "{seen} of {cuts_made}");
}
