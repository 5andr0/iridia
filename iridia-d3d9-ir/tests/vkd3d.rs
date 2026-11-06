#![allow(
    clippy::unwrap_used,
    reason = "a corpus test: a broken case file should stop it"
)]
//! vkd3d's HLSL shader tests at pixel shader models 2.0 and 3.0, as `tests/vkd3d/cases.json` holds them: the
//! programs Microsoft's compiler makes of the tests' shaders, and what each test expects a Direct3D 9 device on
//! Windows to draw (`tests/oracles/vkd3d_cases.py` reads the test files as vkd3d's own runner does; the file says
//! which runs it leaves out and why).
//!
//! Every program a case draws with lifts, validates, prints and parses back, and writes WGSL for browsers. A draw is
//! the runner's full-target triangle with its pass-through vertex shader, run here pixel by pixel: two ways, which must
//! agree, the IR in `iridia-exec-ir` and the WGSL read back by naga's front end in `iridia-exec`. A probe is checked
//! at its rectangle's corners, edge midpoints and centre, against the test's value within its ulps as vkd3d compares.
//!
//! A pixel takes the value of the last draw that wrote it, a clear's, or none: a probe of a pixel nothing wrote, or
//! that a draw this test cannot run decided, is counted and not checked, and so is one whose value depends on a
//! constant the test never set (the program is run with those constants filled two ways). Every outcome is counted
//! and the test fails on a program that does not lift or run, on two ways that disagree, and on a probe that differs.
mod fixtures;
mod harness;
mod tally;

use std::collections::BTreeMap;

use harness::{NoTextures, Prepared, constructs, prepare};
use iridia_ir::interface::{D3dClass, ResourceKind, SystemValue};
use serde_json::Value;
use tally::{Tally, class};

/// The render target vkd3d's runner draws into.
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

/// A case's step.
enum Step {
    /// The render target cleared, each lane as the device stores the clear colour.
    Clear([u32; 4]),
    /// Words written to a constant bank (0 float, 1 integer, 2 boolean) from a word offset.
    Write { bank: usize, at: usize, bits: Vec<u32> },
    /// A draw with a program of the file.
    Draw(String),
    /// A draw with a shader that is not a Direct3D 9 program of the file.
    Unrunnable(String),
    /// A probe: a rectangle (left, top, right, bottom), the lanes expected and the ulps allowed.
    Probe {
        rect: [u32; 4],
        expect: Vec<u32>,
        ulps: u32,
        line: u64,
    },
}

struct Case {
    name: String,
    selected: bool,
    steps: Vec<Step>,
}

struct Cases {
    programs: BTreeMap<String, Vec<u8>>,
    cases: Vec<Case>,
    left_out: usize,
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}

/// Direct3D 9's `ColorFill` of a float target: each lane through the 8-bit colour vkd3d's runner packs it as.
fn clear_lane(bits: u32) -> Result<u32, String> {
    let x = f32::from_bits(bits);
    if !(0.0..=1.0).contains(&x) {
        return Err(format!("a clear to {x}, outside the colour range"));
    }
    let byte = (x * 255.0) as u8;
    Ok((f32::from(byte) / 255.0).to_bits())
}

fn cases() -> Cases {
    let file = fixtures::vkd3d();
    let programs = fixtures::vkd3d_programs(&file);
    let mut cases = Vec::new();
    for c in file["cases"].as_array().unwrap() {
        let mut steps = Vec::new();
        for s in c["steps"].as_array().unwrap() {
            let line = s["line"].as_u64().unwrap();
            let step = if let Some(lanes) = s.get("clear") {
                match u32s(lanes)
                    .into_iter()
                    .map(clear_lane)
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(l) => Step::Clear(l.try_into().unwrap()),
                    Err(e) => Step::Unrunnable(e),
                }
            } else if let Some(bank) = s.get("write") {
                let bank = match bank.as_str().unwrap() {
                    "float" => 0,
                    "int" => 1,
                    _ => 2,
                };
                Step::Write {
                    bank,
                    at: usize::try_from(s["at"].as_u64().unwrap()).unwrap(),
                    bits: u32s(&s["bits"]),
                }
            } else if let Some(program) = s.get("draw") {
                Step::Draw(program.as_str().unwrap().into())
            } else if let Some(why) = s.get("unrunnable") {
                Step::Unrunnable(why.as_str().unwrap().into())
            } else {
                Step::Probe {
                    rect: u32s(&s["probe"]).try_into().unwrap(),
                    expect: u32s(&s["expect"]),
                    ulps: u32::try_from(s["ulps"].as_u64().unwrap()).unwrap(),
                    line,
                }
            };
            steps.push(step);
        }
        cases.push(Case {
            name: format!(
                "{}:{} at {}",
                c["file"].as_str().unwrap(),
                c["line"],
                c["model"].as_str().unwrap()
            ),
            selected: c["run"] == "selected",
            steps,
        });
    }
    Cases {
        programs,
        cases,
        left_out: file["left_out"].as_array().unwrap().len(),
    }
}

/// vkd3d's `compare_float`: equal within `ulps` steps of the float order, every NaN alike.
fn compare_float(got: u32, expected: u32, ulps: u32) -> bool {
    let order = |bits: u32| {
        let f = f32::from_bits(bits);
        let i = if f.is_nan() {
            ((bits & 0x7f80_0000) | 0x40_0000) as i32
        } else {
            bits as i32
        };
        if i < 0 { i32::MIN.wrapping_sub(i) } else { i }
    };
    (i64::from(order(got)) - i64::from(order(expected))).unsigned_abs() <= u64::from(ulps)
}

/// The pixels of a probe's rectangle that are checked: its corners, edge midpoints and centre.
fn pixels(rect: [u32; 4]) -> Vec<(u32, u32)> {
    let [l, t, r, b] = rect;
    let (r, b) = (r.saturating_sub(1).max(l), b.saturating_sub(1).max(t));
    let xs = [l, (l + r) / 2, r];
    let ys = [t, (t + b) / 2, b];
    let mut out: Vec<(u32, u32)> = ys.iter().flat_map(|&y| xs.iter().map(move |&x| (x, y))).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// A program of the file in every form it is run in, and the output slot of its colour 0.
struct Lifted {
    p: Prepared,
    target: usize,
}

/// What a draw needs of a program that it does not model, if anything.
fn unmodelled(p: &iridia_ir::Program) -> Option<String> {
    for v in &p.interface.inputs {
        if !matches!(v.system, Some(SystemValue::FragCoord | SystemValue::FrontFacing)) {
            return Some(format!(
                "reads {}{}, which the runner's vertex shader does not write",
                v.label.name, v.label.index
            ));
        }
    }
    if p.interface
        .resources
        .iter()
        .any(|r| matches!(r.kind, ResourceKind::Texture { .. }))
    {
        return Some("samples a texture (the cases' textures are not modelled)".into());
    }
    None
}

fn lift(bytes: &[u8]) -> Result<Lifted, String> {
    let ir = iridia_d3d9_ir::lift(bytes).map_err(|e| format!("lift: {e}"))?;
    let target = ir
        .interface
        .outputs
        .iter()
        .position(|v| v.system == Some(SystemValue::Target(0)))
        .ok_or("no colour 0 output")?;
    Ok(Lifted {
        p: prepare(ir)?,
        target,
    })
}

/// The constant banks as the test left them: each bank's words, which run past the words written being unset.
#[derive(Clone, Default)]
struct Banks([Vec<u32>; 3]);

impl Banks {
    /// vkd3d's `uniform_buffer_write`: the bank grows to a whole register past the write, the new words 0x7f7f7f7f.
    fn write(&mut self, bank: usize, at: usize, bits: &[u32]) {
        let words = &mut self.0[bank];
        let count = words.len().max(at + bits.len()).next_multiple_of(4);
        words.resize(count, 0x7f7f_7f7f);
        words[at..at + bits.len()].copy_from_slice(bits);
    }

    /// The rows of bank `bank` for a program that binds `rows` of them, the unset words `fill`; and whether any is.
    fn rows(&self, bank: usize, rows: usize, fill: u32) -> (Vec<[u32; 4]>, bool) {
        let words = &self.0[bank];
        let out: Vec<[u32; 4]> = (0..rows)
            .map(|r| [0, 1, 2, 3].map(|l| words.get(r * 4 + l).copied().unwrap_or(fill)))
            .collect();
        (out, rows * 4 > words.len())
    }
}

/// One pixel of a draw: the colour the program writes, `None` where it kills.
type Pixel = Result<Option<[u32; 4]>, String>;

/// One pixel of a draw, run every way the harness runs a program; an error where two disagree or one fails. With
/// `denormals` Flush only the IR runs, as `iridia-exec` keeps denormals.
fn run(l: &Lifted, banks: &Banks, x: u32, y: u32, denormals: iridia_ir::Denormals) -> Pixel {
    let p = &l.p;
    let once = |fill: u32| -> Result<(Option<[u32; 4]>, bool), String> {
        let mut unset = false;
        let mut by_id = Vec::new();
        let mut by_register = Vec::new();
        for r in &p.ir.interface.resources {
            let (ResourceKind::ConstantBuffer { layout }, Some(d)) = (&r.kind, r.provenance.d3d) else {
                continue;
            };
            assert_eq!(d.class, D3dClass::ConstantBuffer);
            let (rows, some_unset) = banks.rows(d.register as usize, layout.size.div_ceil(16) as usize, fill);
            unset |= some_unset;
            by_id.push((r.id, rows.clone()));
            by_register.push((d.register, rows));
        }
        let slots = |x: u32, y: u32| -> Result<Vec<[u32; 4]>, String> {
            p.ir.interface
                .inputs
                .iter()
                .map(|v| match v.system {
                    Some(SystemValue::FragCoord) => {
                        Ok([x as f32 + 0.5, y as f32 + 0.5, 0.0, 1.0].map(f32::to_bits))
                    }
                    Some(SystemValue::FrontFacing) => Ok([1, 0, 0, 0]),
                    _ => Err("an input the draw does not model".to_string()),
                })
                .collect()
        };
        let mut no_textures =
            |_: iridia_exec_ir::Sample| -> Result<[f32; 4], String> { Err("a texture sample".into()) };
        let buffers: Vec<_> = by_id.iter().map(|(i, r)| (*i, r.as_slice())).collect();
        let banks: Vec<_> = by_register.iter().map(|(i, r)| (*i, r.as_slice())).collect();
        let mut bound = harness::Bound {
            buffers: &buffers,
            state: &[],
            sampler: &mut no_textures,
            denormals,
        };
        let view = if iridia_exec_ir::needs_quad(&p.ir) {
            // the pixel's 2 x 2 quad, aligned to even coordinates as a rasterizer's are
            let (left, top) = (x & !1, y & !1);
            let quad = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dy)| slots(left + dx, top + dy));
            let [a, b, c, d] = quad;
            let (a, b, c, d) = (a?, b?, c?, d?);
            let views = p.run_quad([&a, &b, &c, &d], &mut bound, &banks, &mut NoTextures)?;
            let lane = (x & 1) + 2 * (y & 1);
            views[lane as usize].clone()
        } else {
            p.run(&slots(x, y)?, &mut bound, &banks, &mut NoTextures)?
        };
        let colour = (!view.killed).then(|| view.outputs.get(&l.target).copied().unwrap_or([0; 4]));
        Ok((colour, unset))
    };
    let (first, unset) = once(0)?;
    if unset && once(0x3f80_0000)?.0 != first {
        return Err("reads a constant the test never set".into());
    }
    Ok(first)
}

/// A pixel after `events`: the last colour a clear or a draw wrote, `None` where nothing did.
fn pixel(events: &[Event<'_>], x: u32, y: u32, denormals: iridia_ir::Denormals) -> Pixel {
    let mut value: Pixel = Ok(None);
    for e in events {
        match e {
            Event::Clear(c) => value = Ok(Some(*c)),
            Event::Unknown(why) => value = Err(format!("after {why}")),
            Event::Draw(p, b) => match run(p, b, x, y, denormals) {
                Ok(Some(colour)) => value = Ok(Some(colour)),
                Ok(None) => {}
                Err(e) => value = Err(e),
            },
        }
    }
    value
}

/// Whether `got` is `expect` within `ulps`, as vkd3d compares.
fn holds(got: [u32; 4], expect: &[u32], ulps: u32) -> bool {
    expect
        .iter()
        .zip(got)
        .all(|(&want, got)| compare_float(got, want, ulps))
}

/// What a draw at a step left: a program and the banks it ran with, a clear, or a draw this test cannot run.
enum Event<'a> {
    Clear([u32; 4]),
    Draw(&'a Lifted, Banks),
    Unknown(String),
}

const AGREE: &str = "probe holds";
const RUNS: &str = "lifts and runs";
const RUNS_BRANCHING: &str = "lifts and runs, with control flow, as lifted, as a graph and structured again";
const FLUSHED: &str = "probe holds with denormals flushed, which the program allows";

/// Whether every program the events draw with leaves `f32` denormals open.
fn open_denormals(events: &[Event<'_>]) -> bool {
    events.iter().all(|e| match e {
        Event::Draw(p, _) => p.p.ir.float.f32 == iridia_ir::Denormals::Any,
        _ => true,
    })
}

#[test]
fn every_case_draws_what_vkd3d_expects_of_direct3d_9() {
    let all = cases();
    let mut programs = Tally::default();
    // the programs with each construct, and how many of it there are in all
    let mut flow = Tally::default();
    let mut flow_count = BTreeMap::from([("if", 0), ("loop", 0), ("block", 0)]);
    let mut lifted = BTreeMap::new();
    for (name, bytes) in &all.programs {
        match lift(bytes) {
            Ok(p) => {
                let why = unmodelled(&p.p.ir);
                let [ifs, loops, blocks] = constructs(match &p.p.ir.entry.body {
                    iridia_ir::Body::Regions(r) => r,
                    iridia_ir::Body::Cfg(_) => unreachable!("the lifter builds regions"),
                });
                for (n, what) in [(ifs, "if"), (loops, "loop"), (blocks, "block")] {
                    if n > 0 {
                        flow.note(format!("{what}s"), format!("{name}: {n}"));
                        *flow_count.entry(what).or_default() += n;
                    }
                }
                let ran = if why.is_some() {
                    None
                } else if ifs + loops + blocks > 0 {
                    Some(RUNS_BRANCHING)
                } else {
                    Some(RUNS)
                };
                programs.note(
                    why.clone().unwrap_or_else(|| ran.unwrap_or_default().into()),
                    name,
                );
                if why.is_none() {
                    lifted.insert(name.clone(), p);
                }
            }
            Err(e) => programs.note(class(&e), format!("{name}: {e}")),
        }
    }
    let mut probes = Tally::default();
    let mut selected = Tally::default();
    for case in &all.cases {
        let mut banks = Banks::default();
        let mut events = Vec::new();
        for step in &case.steps {
            match step {
                Step::Clear(c) => events.push(Event::Clear(*c)),
                Step::Write { bank, at, bits } => banks.write(*bank, *at, bits),
                Step::Draw(program) => events.push(match lifted.get(program) {
                    Some(p) => Event::Draw(p, banks.clone()),
                    None => Event::Unknown(format!("a draw with {program}, which is not run")),
                }),
                Step::Unrunnable(why) => events.push(Event::Unknown(why.clone())),
                Step::Probe {
                    rect,
                    expect,
                    ulps,
                    line,
                } => {
                    let mut outcome = (AGREE.to_string(), String::new());
                    for (x, y) in pixels(*rect) {
                        assert!(x < WIDTH && y < HEIGHT, "{}: a probe past the target", case.name);
                        let value = pixel(&events, x, y, iridia_ir::Denormals::Any);
                        let at = format!("{} line {line} ({x}, {y})", case.name);
                        let differs = match value {
                            Ok(Some(got)) if !holds(got, expect, *ulps) && open_denormals(&events) => {
                                // the programs leave denormals open: a flushing run is as correct as a keeping one
                                match pixel(&events, x, y, iridia_ir::Denormals::Flush) {
                                    Ok(Some(flushed)) if holds(flushed, expect, *ulps) => {
                                        outcome = (FLUSHED.into(), at.clone());
                                        None
                                    }
                                    _ => Some(format!(
                                        "{at}: got {:?} (and with denormals flushed no closer), expected {:?} within \
                                         {ulps} ulps",
                                        got.map(f32::from_bits),
                                        expect.iter().map(|&b| f32::from_bits(b)).collect::<Vec<_>>()
                                    )),
                                }
                            }
                            Ok(Some(got)) => (!holds(got, expect, *ulps)).then(|| {
                                format!(
                                    "{at}: got {:?}, expected {:?} within {ulps} ulps",
                                    got.map(f32::from_bits),
                                    expect.iter().map(|&b| f32::from_bits(b)).collect::<Vec<_>>()
                                )
                            }),
                            Ok(None) => {
                                outcome = ("not checked: nothing wrote the pixel".into(), at);
                                break;
                            }
                            Err(e) => {
                                let class = if e.contains("disagree") {
                                    "the IR and the WGSL disagree".to_string()
                                } else {
                                    format!("not checked: {}", tally::class(&e))
                                };
                                outcome = (class, format!("{at}: {e}"));
                                break;
                            }
                        };
                        if let Some(d) = differs {
                            outcome = ("probe differs".into(), d);
                            break;
                        }
                    }
                    if case.selected {
                        selected.note(outcome.0.clone(), outcome.1.clone());
                    }
                    probes.note(outcome.0, outcome.1);
                }
            }
        }
    }
    println!(
        "{} programs, {} cases ({} runs the case file leaves out)",
        all.programs.len(),
        all.cases.len(),
        all.left_out
    );
    programs.print("programs");
    flow.print("programs with control flow, by construct");
    println!("constructs in all: {flow_count:?}");
    probes.print("probes");
    selected.print("probes of runs vkd3d's runner selects");
    let unmodelled = |c: &str| c.starts_with("reads ") || c.starts_with("samples a texture");
    assert_eq!(
        programs.failing(|c| c != RUNS && c != RUNS_BRANCHING && !unmodelled(c)),
        0,
        "programs failed"
    );
    assert_eq!(
        probes.failing(|c| c != AGREE && c != FLUSHED && !c.starts_with("not checked")),
        0,
        "probes failed"
    );
}
