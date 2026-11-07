#![allow(
    clippy::unwrap_used,
    reason = "fixture files: a broken one should stop the test"
)]
//! One-feature programs of every shader model 1, 2 and 3 version, written as assembly in `tests/assembled/<name>.asm`
//! and assembled by Microsoft's `D3DAssemble` (`tests/oracles/assemble.py`, which writes `<name>.bin`, the token
//! stream, and `<name>.dis`, Microsoft's `D3DDisassemble` listing of it). For each:
//!
//! - `d3d9-reader`'s listing of the tokens is Microsoft's (but for the instruction slot estimate the listing ends
//!   with), so the instructions the lift reads are the ones assembled;
//! - the program lifts, validates, prints and parses back, and so does its control flow graph (`flatten`) and the
//!   graph structured again (`structurize`); and it writes WGSL for browsers;
//! - each run its comments state gives the outputs they state, worked by hand from the instruction definitions, in
//!   the IR, in its graph and in the graph structured again (bit for bit), and in the WGSL read back by naga's front
//!   end in `iridia-exec`;
//! - a program that reads pipeline state (a pixel 1.x texture stage's dimension and projection, its bump matrix) runs
//!   in the IR reading it, then specialised by each run's state every way; where the run states a stage's dimension,
//!   the program lifted knowing it, specialised by the rest, runs every way too.
//!
//! A fixture's comments, before its version token:
//!
//! - `; lift: <option> ...`, the lift's options: `mul=ieee`, `register_max=<m>`, `texture_max=<m>`,
//!   `quantize=<bits>,<floor|nearest_even|...>`, `texdim<n>=<2d|3d|cube>` (the host knows the stage's dimension);
//! - `; run: <input>=<lanes> ... -> <output>=<lanes> ...` or `-> kill`: inputs by register (`c<n>`, `i<n>`, `b<n>`),
//!   by varying label (`COLOR0`, `TEXCOORD1`, `POSITION0`, `VPOS0`, `VFACE0`), or pipeline state (`texdim<n>`,
//!   `projected<n>`, `bumpenvmat<n>`, `bumplscale<n>`, `bumploffset<n>`); outputs by varying label (`SV_Target0`,
//!   `SV_Depth0`, `POSITION0` the clip position, `COLOR0`, `TEXCOORD0`, `FOG0`, `PSIZE0`). Lanes are floats (`nan`,
//!   `inf`, `-0` included), `0x` bits, `true`/`false`, or integers for `i<n>`; an output's lanes are its own (those
//!   of `o1.zw` are two), as many as given, compared bit for bit but for a NaN's sign and payload;
//! - `; quad: <run> | <run> | <run> | <run>`, a pixel program's 2 x 2 quad run together, each lane's inputs and outputs
//!   as a `run:` states them (lanes 0 and 1 the top row, left to right; the lanes' constants and state the same): a
//!   derivative is a difference between lanes, coarse for `dsx` and `dsy` as `iridia-exec` takes D3D's, and an
//!   implicit-LOD sample is asked with the quad's coarse derivatives of its coordinate (`tex(...;grad:...)`; run
//!   alone, the level is the sampler's, `tex(...;auto)`). A program that takes a derivative runs as a quad.
mod fixtures;
mod harness;

use std::collections::BTreeMap;
use std::path::Path;

use harness::{Bound, Prepared, Texels, ir_texels, prepare};
use iridia_ir::interface::{Interface, ResourceId, ResourceKind, StateParam, SystemValue, TextureDim};

/// The lanes of a value list: floats, `0x` bits, `true` and `false`; or the texel of a sample the run should make,
/// `tex(<texture register>;<coordinate lanes>;<auto|bias:b|level:l|grad:dx lanes/dy lanes>)`, as the harness's texel
/// model gives it.
fn lanes(text: &str) -> Vec<u32> {
    if let Some(sample) = text.strip_prefix("tex(").and_then(|t| t.strip_suffix(')')) {
        let [register, coord, lod] = sample.split(';').collect::<Vec<_>>()[..] else {
            panic!("not a sample: {text}")
        };
        let floats = |t: &str| lanes(t).into_iter().map(f32::from_bits).collect::<Vec<_>>();
        let coord = floats(coord);
        let three = |t: &str| {
            let mut out = [0.0; 3];
            for (o, x) in out.iter_mut().zip(floats(t)) {
                *o = x;
            }
            out
        };
        let lod = match lod.split_once(':') {
            None if lod == "auto" => iridia_exec::ImageLod::Implicit,
            Some(("bias", b)) => iridia_exec::ImageLod::Bias(floats(b)[0]),
            Some(("level", l)) => iridia_exec::ImageLod::Level(floats(l)[0]),
            Some(("grad", g)) => {
                let (dx, dy) = g.split_once('/').unwrap();
                iridia_exec::ImageLod::Gradients([three(dx), three(dy)])
            }
            _ => panic!("not a level of detail: {lod}"),
        };
        return harness::texel(register.parse().unwrap(), &coord, lod)
            .map(f32::to_bits)
            .to_vec();
    }
    text.split(',')
        .map(|t| match t.trim() {
            "true" => 1,
            "false" => 0,
            "nan" => f32::NAN.to_bits(),
            "inf" => f32::INFINITY.to_bits(),
            "-inf" => f32::NEG_INFINITY.to_bits(),
            t if t.starts_with("0x") => u32::from_str_radix(&t[2..], 16).unwrap(),
            t => t
                .parse::<f32>()
                .unwrap_or_else(|_| panic!("not a lane: {t}"))
                .to_bits(),
        })
        .collect()
}

/// A fixture's statement of a run: one invocation, or a quad's four.
struct Run {
    line: String,
    lanes: Vec<Lane>,
}

/// One invocation of a run.
struct Lane {
    inputs: Vec<(String, Vec<u32>)>,
    /// `None` where the invocation kills.
    outputs: Option<Vec<(String, Vec<u32>)>>,
}

impl Run {
    /// The inputs the lanes share: the first lane's (`outputs` checks they all state the same constants and state).
    fn shared(&self) -> &Lane {
        &self.lanes[0]
    }
}

/// A fixture's comments: the lift's options and the runs.
struct Statement {
    options: iridia_d3d9_ir::Options,
    runs: Vec<Run>,
}

fn statement(source: &str) -> Statement {
    let options = fixtures::lift_options(source);
    let mut runs = Vec::new();
    for line in source.lines().map(str::trim) {
        let stated = if let Some(rest) = line.strip_prefix("; run:") {
            vec![rest]
        } else if let Some(rest) = line.strip_prefix("; quad:") {
            let stated: Vec<&str> = rest.split('|').collect();
            assert_eq!(stated.len(), 4, "a quad of other than four lanes: {line}");
            stated
        } else {
            continue;
        };
        let lane = |rest: &str| {
            let (given, expected) = rest.split_once("->").unwrap();
            let pairs = |text: &str| {
                text.split_whitespace()
                    .map(|p| {
                        let (name, value) = p.split_once('=').unwrap();
                        let value = match name.as_bytes() {
                            [b't', b'e', b'x', b'd', b'i', b'm', ..] => {
                                vec![fixtures::dim(value).state_code()]
                            }
                            // an integer constant's lanes are integers
                            [b'i', d, ..] if d.is_ascii_digit() => value
                                .split(',')
                                .map(|x| x.parse::<i32>().unwrap().cast_unsigned())
                                .collect(),
                            _ => lanes(value),
                        };
                        (name.to_owned(), value)
                    })
                    .collect::<Vec<_>>()
            };
            Lane {
                inputs: pairs(given),
                outputs: (expected.trim() != "kill").then(|| pairs(expected)),
            }
        };
        runs.push(Run {
            line: line.to_owned(),
            lanes: stated.into_iter().map(lane).collect(),
        });
    }
    Statement { options, runs }
}

fn label(v: &iridia_ir::interface::Varying) -> String {
    format!("{}{}", v.label.name, v.label.index)
}

/// A pipeline state resource by the name a run gives it.
fn state_param(name: &str) -> Option<StateParam> {
    let (kind, n) = name.split_at(name.find(|c: char| c.is_ascii_digit())?);
    let n = n.parse().ok()?;
    Some(match kind {
        "texdim" => StateParam::TextureDim(n),
        "projected" => StateParam::Projected(n),
        "bumpenvmat" => StateParam::BumpEnvMat(n),
        "bumplscale" => StateParam::BumpLScale(n),
        "bumploffset" => StateParam::BumpLOffset(n),
        _ => return None,
    })
}

/// A run's inputs bound to an interface: the slots, the banks by resource and by register, and the state.
struct Bindings {
    slots: Vec<[u32; 4]>,
    banks: Vec<(ResourceId, u32, Vec<[u32; 4]>)>,
    state: Vec<(ResourceId, [u32; 4])>,
}

fn bind(interface: &Interface, run: &Lane) -> Result<Bindings, String> {
    let mut slots = vec![[0u32; 4]; interface.inputs.len()];
    let mut words: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    let mut state = Vec::new();
    for (name, value) in &run.inputs {
        if let Some(param) = state_param(name) {
            // a state the program does not read is left out: the run states it for the forms that do
            if let Some(r) = interface
                .resources
                .iter()
                .find(|r| r.kind == ResourceKind::PipelineState(param))
            {
                let mut lanes = [0u32; 4];
                lanes[..value.len()].copy_from_slice(value);
                state.push((r.id, lanes));
            }
            continue;
        }
        let bank = match name.as_bytes() {
            [b'c', d, ..] if d.is_ascii_digit() => Some((0, 4)),
            [b'i', d, ..] if d.is_ascii_digit() => Some((1, 4)),
            [b'b', d, ..] if d.is_ascii_digit() => Some((2, 1)),
            _ => None,
        };
        if let Some((bank, per)) = bank {
            let n: usize = name[1..].parse().unwrap();
            let w = words.entry(bank).or_default();
            if w.len() < (n + 1) * per {
                w.resize((n + 1) * per, 0);
            }
            w[n * per..n * per + value.len().min(per)].copy_from_slice(&value[..value.len().min(per)]);
            continue;
        }
        let slot = interface
            .inputs
            .iter()
            .position(|v| label(v) == *name)
            .ok_or_else(|| format!("{name}: no such input"))?;
        slots[slot][..value.len()].copy_from_slice(value);
    }
    let mut banks = Vec::new();
    for r in &interface.resources {
        let (ResourceKind::ConstantBuffer { layout }, Some(d)) = (&r.kind, r.provenance.d3d) else {
            continue;
        };
        let rows = layout.size.div_ceil(16) as usize;
        let w = words.get(&d.register).cloned().unwrap_or_default();
        let data = (0..rows)
            .map(|row| [0, 1, 2, 3].map(|l| w.get(row * 4 + l).copied().unwrap_or(0)))
            .collect();
        banks.push((r.id, d.register, data));
    }
    Ok(Bindings { slots, banks, state })
}

/// Each output's own lanes by label: a varying's from its register's in the lanes its mask names (`o1.zw` gives two).
fn packed(interface: &Interface, view: &harness::View) -> BTreeMap<String, [u32; 4]> {
    let mut out = BTreeMap::new();
    for (slot, v) in interface.outputs.iter().enumerate() {
        let lanes = match v.system {
            Some(SystemValue::ClipPosition) => view.position.unwrap_or([0; 4]),
            None | Some(SystemValue::Target(_)) => {
                let register = view.outputs.get(&slot).copied().unwrap_or([0; 4]);
                let mut lanes = [0; 4];
                let mask = v.provenance.d3d.map_or(0b1111, |d| d.mask);
                for (k, l) in harness::lanes_of(mask).into_iter().enumerate() {
                    lanes[k] = register[l];
                }
                lanes
            }
            Some(_) => view.other.get(&slot).copied().unwrap_or([0; 4]),
        };
        out.insert(label(v), lanes);
    }
    out
}

/// Each lane's outputs by label, `None` where it kills.
type Outputs = Vec<Option<BTreeMap<String, [u32; 4]>>>;

/// A run's lanes bound to an interface: each lane's, which must share their constants and state.
fn bind_all(interface: &Interface, run: &Run) -> Result<Vec<Bindings>, String> {
    let all: Vec<Bindings> = run
        .lanes
        .iter()
        .map(|l| bind(interface, l))
        .collect::<Result<_, _>>()?;
    if all
        .iter()
        .any(|b| b.banks != all[0].banks || b.state != all[0].state)
    {
        return Err(format!(
            "{}: the quad's lanes state different constants or state",
            run.line
        ));
    }
    Ok(all)
}

/// Each lane's slots, as a quad's four.
fn quad(all: &[Bindings]) -> Result<[&[[u32; 4]]; 4], String> {
    let [a, b, c, d] = all else {
        return Err("a quad of other than four lanes".into());
    };
    Ok([&a.slots, &b.slots, &c.slots, &d.slots])
}

/// Each lane's outputs of a view.
fn lanes_of(interface: &Interface, views: &[harness::View]) -> Outputs {
    views
        .iter()
        .map(|v| (!v.killed).then(|| packed(interface, v)))
        .collect()
}

/// A run of every form of `p`: one invocation, or a quad.
fn outputs(p: &Prepared, run: &Run) -> Result<Outputs, String> {
    let interface = &p.ir.interface;
    let all = bind_all(interface, run)?;
    let b = &all[0];
    let by_id: Vec<_> = b.banks.iter().map(|(i, _, d)| (*i, d.as_slice())).collect();
    let by_register: Vec<_> = b.banks.iter().map(|(_, r, d)| (*r, d.as_slice())).collect();
    let mut sampler = ir_texels(interface);
    let mut bound = Bound {
        buffers: &by_id,
        state: &b.state,
        sampler: &mut sampler,
        denormals: iridia_ir::Denormals::Any,
    };
    let views = if all.len() == 1 {
        vec![p.run(&b.slots, &mut bound, &by_register, &mut Texels)?]
    } else {
        p.run_quad(quad(&all)?, &mut bound, &by_register, &mut Texels)?
            .to_vec()
    };
    Ok(lanes_of(interface, &views))
}

/// The IR run alone of a program with pipeline state that has no WGSL form (a texture stage of unknown dimension).
fn ir_outputs(p: &iridia_ir::Program, run: &Run) -> Result<Outputs, String> {
    let all = bind_all(&p.interface, run)?;
    let b = &all[0];
    let by_id: Vec<_> = b.banks.iter().map(|(i, _, d)| (*i, d.as_slice())).collect();
    let mut sampler = ir_texels(&p.interface);
    let mut bound = Bound {
        buffers: &by_id,
        state: &b.state,
        sampler: &mut sampler,
        denormals: iridia_ir::Denormals::Any,
    };
    let views = if all.len() == 1 {
        vec![harness::run_ir(p, &b.slots, &mut bound)?]
    } else {
        harness::run_ir_quad(p, quad(&all)?, &mut bound)?.to_vec()
    };
    Ok(lanes_of(&p.interface, &views))
}

/// Whether a run's outputs are what it states, lane by lane; what differs if not.
fn check(run: &Run, got: &Outputs) -> Result<(), String> {
    for (l, (lane, got)) in run.lanes.iter().zip(got).enumerate() {
        let at = if run.lanes.len() > 1 {
            format!("{} lane {l}", run.line)
        } else {
            run.line.clone()
        };
        check_lane(&at, lane, got)?;
    }
    Ok(())
}

fn check_lane(at: &str, lane: &Lane, got: &Option<BTreeMap<String, [u32; 4]>>) -> Result<(), String> {
    match (&lane.outputs, got) {
        (None, None) => Ok(()),
        (None, Some(g)) => Err(format!("{at}: expected a kill, got {g:x?}")),
        (Some(_), None) => Err(format!("{at}: killed")),
        (Some(want), Some(got)) => {
            for (name, lanes) in want {
                let have = got.get(name).ok_or_else(|| format!("{at}: no output {name}"))?;
                // a NaN's sign and payload are not stated by IEEE 754 but for the sign operations
                let alike = |(&x, &y): (&u32, &u32)| {
                    x == y || f32::from_bits(x).is_nan() && f32::from_bits(y).is_nan()
                };
                if !have[..lanes.len()].iter().zip(lanes).all(alike) {
                    return Err(format!(
                        "{at}: {name} is {:?} ({:x?}), expected {:?}",
                        have[..lanes.len()]
                            .iter()
                            .map(|&b| f32::from_bits(b))
                            .collect::<Vec<_>>(),
                        &have[..lanes.len()],
                        lanes.iter().map(|&b| f32::from_bits(b)).collect::<Vec<_>>()
                    ));
                }
            }
            Ok(())
        }
    }
}

/// Lines `D3DDisassemble` (d3dcompiler_47 10.0.26100.4654) prints for an instruction other than the one its tokens
/// hold, with the instruction they hold: `breakp p0.x` (opcode 96, its source `p0` with swizzle `.xxxx`) is listed as
/// `break p0.`, the name and lane of `break`'s predicate form misprinted.
const MISPRINTS: &[(&str, &str)] = &[("break p0.", "breakp p0.x")];

/// One fixture through every check.
fn fixture(folder: &Path, name: &str) -> Result<(), String> {
    let source = std::fs::read_to_string(folder.join(format!("{name}.asm"))).unwrap();
    if let Ok(err) = std::fs::read_to_string(folder.join(format!("{name}.err"))) {
        return Err(format!("Microsoft's assembler refuses it: {}", err.trim()));
    }
    let bytes =
        std::fs::read(folder.join(format!("{name}.bin"))).map_err(|_| "not assembled (run assemble.py)")?;
    let listing = std::fs::read_to_string(folder.join(format!("{name}.dis"))).unwrap();
    let ours = d3d9_reader::disassemble(&bytes).map_err(|e| format!("d3d9-reader's listing: {e}"))?;
    // the listings' instructions, without the instruction slot estimate the disassembler ends them with, and with
    // Microsoft's misprinted lines as the instructions they list
    let instructions = |l: &str| {
        l.replace("\r\n", "\n")
            .lines()
            .filter(|l| !l.starts_with("// approximately"))
            .map(|l| {
                let (indent, text) = l.split_at(l.len() - l.trim_start().len());
                match MISPRINTS.iter().find(|(printed, _)| *printed == text) {
                    Some((_, meant)) => format!("{indent}{meant}"),
                    None => l.to_owned(),
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            .trim_end()
            .to_owned()
    };
    if instructions(&ours) != instructions(&listing) {
        return Err(format!(
            "d3d9-reader's listing is not Microsoft's:\n{ours}\n---\n{listing}"
        ));
    }
    let statement = statement(&source);
    let ir = iridia_d3d9_ir::lift_with(&bytes, &statement.options).map_err(|e| format!("lift: {e}"))?;
    if statement.runs.is_empty() {
        return Err("states no run".into());
    }
    // an implicit-LOD sample runs alone too, its level the sampler's; a derivative has no value but in a quad
    let derivative = ir
        .entry
        .values
        .iter()
        .any(|d| matches!(d.op, iridia_ir::Op::Unary(iridia_ir::Unary::Derivative(..), _)));
    if derivative && statement.runs.iter().any(|r| r.lanes.len() != 4) {
        return Err("takes a derivative, and states a run that is not a quad".into());
    }
    let reads_state = ir
        .interface
        .resources
        .iter()
        .any(|r| matches!(r.kind, ResourceKind::PipelineState(_)));
    if !reads_state {
        let p = prepare(ir)?;
        for run in &statement.runs {
            check(run, &outputs(&p, run)?)?;
        }
        return Ok(());
    }
    // pipeline state (a pixel 1.x texture stage's dimension and projection, its bump matrix): the IR runs reading it,
    // then the program specialised by the run's state runs every way; where the run states a stage's dimension, so
    // does the program lifted knowing it, specialised by the rest
    let mut unknown = ir;
    iridia_passes::promote(&mut unknown.entry).map_err(|e| format!("promote: {e}"))?;
    for run in &statement.runs {
        check(run, &ir_outputs(&unknown, run)?)?;
        let specialised = specialise_by(unknown.clone(), run).map_err(|e| format!("specialised: {e}"))?;
        check(run, &outputs(&specialised, run)?).map_err(|e| format!("specialised: {e}"))?;
        let mut stated = statement.options.clone();
        for (name, value) in &run.shared().inputs {
            if let Some(n) = name.strip_prefix("texdim") {
                let d = [TextureDim::D1, TextureDim::D2, TextureDim::D3, TextureDim::Cube]
                    .into_iter()
                    .find(|d| d.state_code() == value[0])
                    .unwrap();
                stated.texture_dims.insert(n.parse().unwrap(), d);
            }
        }
        if stated.texture_dims != statement.options.texture_dims {
            let told = iridia_d3d9_ir::lift_with(&bytes, &stated)
                .map_err(|e| format!("lift knowing the dimension: {e}"))?;
            let told = specialise_by(told, run).map_err(|e| format!("lifted knowing the dimension: {e}"))?;
            check(run, &outputs(&told, run)?).map_err(|e| format!("lifted knowing the dimension: {e}"))?;
        }
    }
    Ok(())
}

/// `p` specialised by the pipeline state `run` gives, prepared to run every way.
fn specialise_by(mut p: iridia_ir::Program, run: &Run) -> Result<Prepared, String> {
    let known: Vec<(ResourceId, [u64; 4])> = bind(&p.interface, run.shared())?
        .state
        .iter()
        .map(|(id, lanes)| (*id, lanes.map(u64::from)))
        .collect();
    iridia_passes::specialise_state(&mut p, &known).map_err(|e| format!("specialise: {e}"))?;
    prepare(p)
}

#[test]
fn every_assembled_fixture_runs_as_its_comments_state() {
    let (folder, names) = (fixtures::assembled_folder(), fixtures::assembled());
    let mut failed = Vec::new();
    let mut versions = BTreeMap::<String, usize>::new();
    for name in &names {
        let version = std::fs::read_to_string(folder.join(format!("{name}.asm")))
            .unwrap()
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty() && !l.starts_with(';'))
            .unwrap_or_default()
            .to_owned();
        *versions.entry(version).or_default() += 1;
        if let Err(e) = fixture(&folder, name) {
            failed.push(format!("{name}: {e}"));
        }
    }
    println!("{} fixtures by version: {versions:?}", names.len());
    assert!(
        failed.is_empty(),
        "{} of {} failed:\n{}",
        failed.len(),
        names.len(),
        failed.join("\n")
    );
}
