//! The WARP half of the harness: a prepared program's WGSL, as each of the two writers (`iridia-naga`, `iridia-wgsl`) gives
//! it, run on Direct3D 12's WARP adapter through `tools.wgsl_run` (an executor iridia does not write) and held to the IR's run
//! in `iridia-exec-ir` on the same inputs, constants and texels: what each writer writes is what the IR means, and the two
//! writers mean the same. The harness is the one `iridia-dxbc`'s WARP comparison runs (`iridia-warp-client`), and so are its
//! classes of distance, its environment (every texture one texel, every sampler point-filtering) and the sample of programs
//! it runs by default: one in [`SAMPLE`] by a hash of the program's file name (`shader.graphics=1` runs every one).
//!
//! WARP computes `sin`, `exp` and the rest to Direct3D's tolerance, not the IR executor's exact value, so a lane is held to
//! 1e-3 relative ([`Class`]) on each writer, and both writers' outcomes are reported. A program the harness cannot bind is a
//! named gap, counted, not a pass.
//!
//! One case is not held to WARP: `pow` of a zero base and an exponent of 0 or less. Microsoft's `pow` page states no value
//! for it (the documented expansion `exp(y * log(x))` with `log(0)` = -FLT_MAX gives libm's, which is the IR's), `FPow` in
//! the IR is undefined there, and so is WGSL's `pow`; the lift leaves the zero base undefined, as both do (the crate documentation's special inputs section). A target compiles
//! it as it likes: `pow(0, -0.0)` is 1 in `iridia-exec-ir` and NaN on WARP (the compilers' `exp2(y * log2(x))`). A program
//! with a lane beyond tolerance whose WGSL has `pow` is therefore run again with `pow` replaced by
//! [`wgsl_run::substitute_pow`]'s (WGSL's everywhere but at that case, which tests/pow_helper.rs holds it to), and when
//! every lane of every output then agrees it is counted as [`Agreement::PowDefined`]; any lane that still differs fails the
//! program, as it would without the rerun.
//!
//! A program that differs for another reason the writers are not at fault for is in [`EXPECTED`], by name and source hash,
//! with the reason and where it was proved; a listed program that agrees fails the test.
//!
//! The IR run is bounded ([`STEPS`]): the seeded constants give a loop any count, and a program whose loop takes more is the
//! named gap [`SEEDED_LOOP_BOUND`].
#![allow(dead_code, reason = "each test uses the part it needs")]

use std::collections::BTreeMap;
use std::sync::OnceLock;

use iridia_ir::interface::{D3dSignature, Interface, ResourceId, ResourceKind, Stage, SystemValue};

use crate::harness::{self, Rng, View, lanes_of, quad, seeded};
use crate::inputs::{self, Tool};

use iridia_warp_client as wgsl_run;
use wgsl_run::{Class, Environment, Harness, Layout};

/// One program in this many (by a hash of its file name) is in the sample, unless `shader.graphics_every` says another number.
pub const SAMPLE: u64 = 128;

/// The configured harness every test of the binary shares; `None` (the test skips its WARP half) where `tools.wgsl_run` is
/// not set. The client takes the machine-wide lock on the adapter for each job (see [`Harness::run`]).
pub fn harness() -> Option<&'static Harness> {
    static SHARED: OnceLock<Option<Harness>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            Some(Harness::new(
                inputs::program(Tool::WgslRun)?,
                inputs::tool(Tool::WgslRunDxc),
            ))
        })
        .as_ref()
}

/// A program's file name, without its folder.
fn short(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
}

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Whether the program `name` is in the one-in-`every` sample, by a hash of its file name (`iridia-dxbc`'s choice).
pub fn in_sample(name: &str, every: u64) -> bool {
    let fnv = short(name).bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    mix64(fnv).is_multiple_of(every)
}

/// Whether the program `name` runs: `shader.graphics` is `1` (every one), `0` (none), or unset and the program is in the
/// one-in-`every` sample (`shader.graphics_every` says another number), or `shader.only` names it.
pub fn wanted(name: &str, every: u64) -> bool {
    let only = inputs::debug("shader.only");
    match inputs::debug("shader.graphics").as_deref() {
        None | Some("sample") if only.is_some() => {
            only.is_some_and(|o| o.split('|').any(|part| name.contains(part)))
        }
        None | Some("sample") => {
            let every = inputs::debug("shader.graphics_every")
                .map_or(every, |v| v.parse().expect("shader.graphics_every is a number"));
            in_sample(name, every)
        }
        Some("1") => only.is_none_or(|o| o.split('|').any(|part| name.contains(part))),
        Some("0") => false,
        Some(other) => panic!("shader.graphics is 0, 1 or sample, not {other}"),
    }
}

/// `interface` with each varying in the register of its slot (its lanes as before): Direct3D 9 puts two varyings in one
/// register (`v0` and `t0`), `vPos` and `vFace` have none, and the harness carries a program's inputs and outputs by register.
pub fn registered(interface: &Interface) -> Interface {
    let mut i = interface.clone();
    let signature = |n: usize, v: &iridia_ir::interface::Varying| D3dSignature {
        register: u32::try_from(n).expect("a slot fits 32 bits"),
        mask: v
            .provenance
            .d3d
            .map_or((1u8 << v.value.width.min(4)) - 1, |d| d.mask),
        read_write_mask: 0,
    };
    for (n, v) in i.inputs.iter_mut().enumerate() {
        v.provenance.d3d = Some(signature(n, v));
    }
    for (n, v) in i.outputs.iter_mut().enumerate() {
        v.provenance.d3d = Some(signature(n, v));
    }
    i
}

/// The register view of inputs given by slot: each varying's packed lanes put in the lanes its mask names, in the register
/// of its slot (`interface` is [`registered`]).
pub fn register_view(interface: &Interface, slots: &[[u32; 4]]) -> Vec<[u32; 4]> {
    let mut registers = vec![[0u32; 4]; slots.len().max(1)];
    for (n, (v, packed)) in interface.inputs.iter().zip(slots).enumerate() {
        for (k, l) in lanes_of(v.provenance.d3d.map_or(0, |d| d.mask))
            .into_iter()
            .enumerate()
        {
            registers[n][l] = packed[k];
        }
    }
    registers
}

/// Why the harness cannot run a program as [`Environment`] binds it, if it cannot.
pub fn gap(p: &iridia_ir::Program) -> Option<String> {
    if p.interface
        .resources
        .iter()
        .any(|r| matches!(r.kind, ResourceKind::PipelineState(_)))
    {
        return Some("pipeline state, which a WGSL program does not carry".into());
    }
    wgsl_run::gap(p)
}

/// One run: each invocation's input slots (a pixel program's four lanes of a quad, lanes 0 and 1 the top row; a vertex
/// program's one) and the constant banks by Direct3D register.
pub struct Case {
    pub lanes: Vec<Vec<[u32; 4]>>,
    pub banks: Vec<(u32, Vec<[u32; 4]>)>,
}

/// What the runs of one writer's WGSL on WARP gave, against the IR's.
#[derive(Debug, Default)]
pub struct Report {
    /// Invocations compared, and those killed on both sides (no output compared).
    pub runs: usize,
    pub killed: usize,
    /// Output lanes by their distance.
    pub lanes: BTreeMap<Class, usize>,
    /// The worst class of a compared lane, where any was.
    pub worst: Option<Class>,
}

pub enum Outcome {
    /// Not run, and why: a named gap.
    Gap(String),
    /// Run: each writer's report, and how far the program agrees with WARP.
    Ran {
        reports: Vec<(&'static str, Report)>,
        agreement: Agreement,
    },
    /// A failure: a harness or executor error, or a lane beyond 1e-3 on a writer.
    Failed(String),
}

/// How far a program that ran agrees with WARP.
#[derive(Clone, Copy, Debug)]
pub enum Agreement {
    /// Every lane within tolerance.
    Plain,
    /// Every lane within tolerance only once `pow` of a zero base and an exponent of 0 or less is given libm's value (the IR's)
    /// where WGSL leaves it undefined.
    PowDefined,
    /// A lane beyond tolerance, and [`EXPECTED`] says why.
    Expected(&'static str),
}

/// Programs whose lanes differ from WARP's beyond 1e-3 with the IR executor and both writers right: each by program name and
/// source hash, with why. A program with other source of the same name, or the same source under another name, is not listed,
/// and a listed program that agrees is a failure.
pub const EXPECTED: &[(&str, &str, &str)] = &[(
    "59300e4c07903e72",
    "sha256:ebc3f12a3a32dcf5109842f3841a9bcb3a9691363d2755e36356fbae24d14ff5",
    "x - x of an infinity: the IR gives NaN (IEEE), DXC and FXC fold it to 0 (tools/wgsl-run/probes/inf_minus_inf.py: `-r + r`, `r - r` and `r + (-r)` of r = 1/0 are 0 and `r / r` is 1 on WARP, `r + (-r)` written as a sign-bit flip is NaN); the WGSL of both writers is that expression, and WGSL's finite math assumption (CRD section 15.7.2: an implementation may assume no infinity or NaN) allows the fold",
)];

/// The reason `name` (of the program with source hash `hash`) may differ from WARP, if listed.
pub fn expected(name: &str, hash: &str) -> Option<&'static str> {
    EXPECTED
        .iter()
        .find(|(n, h, _)| *n == short(name) && *h == hash)
        .map(|(_, _, why)| *why)
}

/// The most loop iterations and block executions a lane takes in a program the comparison runs: the seeded constants give a
/// loop a count of anything a 32-bit word holds, and a run of millions of steps is not one WARP's job should be.
pub const STEPS: u64 = 1 << 16;

/// The gap of a program whose loop takes more than [`STEPS`] on the seeded constants (a float seed read as an integer loop
/// bound): one name for it in every harness, the evidence (the bound, the steps) beside it where a test prints more.
pub const SEEDED_LOOP_BOUND: &str = "seeded integer loop bound";

/// The IR's run of each invocation of `case` (`iridia-exec-ir`), in the environment WARP gets: every sample the texel of
/// its texture's register. `None` where a lane takes more than [`STEPS`].
fn reference(
    p: &iridia_ir::Program,
    case: &Case,
    texels: &BTreeMap<u32, [u32; 4]>,
) -> Result<Option<Vec<View>>, String> {
    let mut buffers: Vec<(ResourceId, &[[u32; 4]])> = Vec::new();
    for r in &p.interface.resources {
        if let (ResourceKind::ConstantBuffer { .. }, Some(d)) = (&r.kind, r.provenance.d3d)
            && let Some((_, rows)) = case.banks.iter().find(|(register, _)| *register == d.register)
        {
            buffers.push((r.id, rows.as_slice()));
        }
    }
    let register_of = |id: ResourceId| {
        p.interface
            .resources
            .iter()
            .find(|r| r.id == id)
            .and_then(|r| r.provenance.d3d)
            .map_or(u32::MAX, |d| d.register)
    };
    let mut sampler = |s: iridia_exec_ir::Sample| -> Result<[f32; 4], String> {
        let register = register_of(s.texture);
        texels
            .get(&register)
            .map(|t| t.map(f32::from_bits))
            .ok_or_else(|| format!("no texture at register {register}"))
    };
    let mut bound = harness::Bound {
        buffers: &buffers,
        state: &[],
        sampler: &mut sampler,
        denormals: iridia_ir::Denormals::Any,
    };
    if p.stage == Stage::Pixel {
        let [a, b, c, d] = case.lanes.as_slice() else {
            return Err("a pixel program runs as a quad of four lanes".into());
        };
        Ok(harness::run_ir_quad_within(p, [a, b, c, d], &mut bound, STEPS)?.map(|views| views.to_vec()))
    } else {
        case.lanes
            .iter()
            .map(|l| harness::run_ir_within(p, l, &mut bound, STEPS))
            .collect()
    }
}

/// The view of an invocation WARP ran: outputs by slot, `interface` being [`registered`].
fn warp_view(interface: &Interface, r: &wgsl_run::Registers) -> View {
    let (mut outputs, mut position) = (BTreeMap::new(), None);
    if r.returned {
        for (slot, v) in interface.outputs.iter().enumerate() {
            let Some(d) = v.provenance.d3d else { continue };
            let got = r.outputs.get(&d.register).copied().unwrap_or([0; 4]);
            let mut lanes = [0u32; 4];
            for l in lanes_of(d.mask) {
                lanes[l] = got[l];
            }
            match v.system {
                Some(SystemValue::ClipPosition) => position = Some(lanes),
                None | Some(SystemValue::Target(_)) => {
                    outputs.insert(slot, lanes);
                }
                Some(_) => {}
            }
        }
    }
    View {
        outputs,
        position,
        other: BTreeMap::new(),
        killed: !r.returned,
    }
}

/// Seeded cases per program.
pub const CASES: u64 = 3;

/// The cases a program runs: seeded inputs and constants, a quad for a pixel program. `interface` is the program's,
/// [`registered`] here: `vPos` and `vFace` have no register of their own, and the seeded inputs are by register.
pub fn cases(name: &str, interface: &Interface) -> Vec<Case> {
    let interface = &registered(interface);
    let mut rng = Rng::of(&format!("{name}/warp"));
    (0..CASES)
        .map(|_| {
            let i = seeded(interface, &mut rng);
            let lanes = if interface.stage == Stage::Pixel {
                quad(interface, &i.slots, &mut rng)
            } else {
                vec![i.slots.clone()]
            };
            Case {
                lanes,
                banks: i
                    .banks
                    .into_iter()
                    .map(|(_, register, rows)| (register, rows))
                    .collect(),
            }
        })
        .collect()
}

/// The named gap of a program outside the sample.
const OUTSIDE: &str = "outside the sample (shader.graphics=1 runs every program)";

/// The comparison of `p` (named `name`) on seeded [`cases`] if it is in the one-in-`every` sample, a named gap if it is not;
/// `None` where the harness is not configured.
pub fn compare_sampled(p: &harness::Prepared, name: &str, every: u64) -> Option<Outcome> {
    let harness = harness()?;
    if !wanted(name, every) {
        return Some(Outcome::Gap(OUTSIDE.into()));
    }
    Some(compare(harness, p, name, &cases(name, &p.ir.interface)))
}

impl Outcome {
    /// The classes the outcome counts under, each with whether it fails the test and an example: a named gap, a failure, or
    /// each writer's worst lane.
    pub fn classes(&self) -> Vec<(String, bool, String)> {
        match self {
            Outcome::Gap(why) => vec![(format!("WARP gap: {why}"), false, why.clone())],
            Outcome::Failed(why) => vec![(format!("WARP FAILED: {why}"), true, why.clone())],
            Outcome::Ran { reports, agreement } => reports
                .iter()
                .map(|(writer, report)| {
                    let what = match (report.worst, agreement) {
                        (None, _) => "every run killed on both sides: nothing compared".to_owned(),
                        (Some(worst), Agreement::Plain) => format!("worst lane {worst:?}"),
                        (Some(worst), Agreement::Expected(_)) => {
                            format!("worst lane {worst:?}, differs as EXPECTED says")
                        }
                        (Some(worst), Agreement::PowDefined) => format!(
                            "worst lane {worst:?} once `pow` of a zero base and an exponent of 0 or less is given libm's value"
                        ),
                    };
                    (format!("WARP, {writer}: {what}"), false, String::new())
                })
                .collect(),
        }
    }
}

/// Each writer's report and its first difference beyond tolerance, where there was one.
type Writers = (Vec<(&'static str, Report)>, Vec<String>);

/// Every program, with its WGSL by each writer, held to the IR's run on WARP over `cases`. `name` is the program's file
/// name (the sample and the table name it).
pub fn compare(harness: &Harness, p: &harness::Prepared, name: &str, cases: &[Case]) -> Outcome {
    if let Some(why) = gap(&p.ir) {
        return Outcome::Gap(why);
    }
    let registered = registered(&p.ir.interface);
    let layout = match Layout::naga(&registered) {
        Ok(l) => l,
        Err(e) => return Outcome::Failed(format!("the layout: {e}")),
    };
    let bindings: Vec<_> = match iridia_naga::BindingPlan::d3d(&p.ir.interface) {
        Ok(plan) => plan
            .bindings
            .iter()
            .map(|b| {
                (
                    b.resource,
                    wgsl_run::Bound {
                        group: b.group,
                        binding: b.binding,
                    },
                )
            })
            .collect(),
        Err(e) => return Outcome::Failed(format!("the binding plan: {e}")),
    };
    // the output lanes that are `fract(x * c)` of an amplifying constant, by slot and lane
    let amplified = {
        let mut q = p.ir.clone();
        q.interface = registered.clone();
        wgsl_run::amplified::outputs(&q)
    };
    // the textures: each register's texel, as `iridia-dxbc`'s comparison gives them
    let shapes = wgsl_run::textures(&p.ir.interface);
    let mut references = Vec::new();
    for (c, case) in cases.iter().enumerate() {
        let texels: BTreeMap<u32, [u32; 4]> = shapes
            .iter()
            .map(|(&register, &(_, _, s))| (register, wgsl_run::texel(register, c as u32, s)))
            .collect();
        match reference(&p.ir, case, &texels) {
            Ok(None) => return Outcome::Gap(SEEDED_LOOP_BOUND.into()),
            Ok(Some(views)) => references.push((
                views,
                Environment {
                    texels,
                    buffers: case.banks.clone(),
                    words: BTreeMap::new(),
                },
            )),
            Err(e) => return Outcome::Failed(format!("the IR run: {e}")),
        }
    }
    let show = inputs::debug("shader.lanes").is_some_and(|s| name.contains(&s));
    // the writers' WGSL run on WARP over every case, each writer's report and its first difference beyond tolerance; with
    // `pow_defined` the WGSL's `pow` gives a zero base and an exponent of 0 or less libm's value
    let run = |pow_defined: bool| -> Result<Writers, Outcome> {
        let mut reports = Vec::new();
        // each writer's first difference beyond tolerance: a program that fails fails on every writer it differs on
        let mut beyond: Vec<String> = Vec::new();
        for (writer, text) in &p.writers {
            let mut first: Option<String> = None;
            let text = match inputs::debug("shader.sin").as_deref() {
                None => text.clone(),
                Some("reference") => match wgsl_run::substitute_sin(text) {
                    Ok(t) => t,
                    Err(e) => return Err(Outcome::Gap(e)),
                },
                Some(other) => panic!("shader.sin is reference, not {other}"),
            };
            let text = match (inputs::debug("shader.pow").as_deref(), pow_defined) {
                (None, false) => text,
                (None | Some("defined"), _) => match wgsl_run::substitute_pow(&text) {
                    Ok(t) => t,
                    Err(e) => return Err(Outcome::Gap(e)),
                },
                (Some(other), _) => panic!("shader.pow is defined, not {other}"),
            };
            if let Some(dir) = inputs::debug("shader.dump") {
                let _ = std::fs::create_dir_all(&dir);
                let _ = std::fs::write(
                    std::path::Path::new(&dir).join(format!("{}.{writer}.wgsl", short(name))),
                    &text,
                );
            }
            if show {
                println!("{name} ({writer})\n{text}");
            }
            let mut report = Report::default();
            for (c, case) in cases.iter().enumerate() {
                let (views, env) = &references[c];
                let registers: Vec<Vec<[u32; 4]>> =
                    case.lanes.iter().map(|l| register_view(&registered, l)).collect();
                let invocations: Vec<&[[u32; 4]]> = registers.iter().map(Vec::as_slice).collect();
                let job = match wgsl_run::job(&registered, &text, &layout, &bindings, &invocations, env) {
                    Ok(j) => j,
                    Err(e) => return Err(Outcome::Failed(format!("{writer}: the job: {e}"))),
                };
                let reply = match harness.run(&job) {
                    Ok(r) => r,
                    Err(e) => return Err(Outcome::Failed(format!("{writer}: the harness: {e}"))),
                };
                let got = match wgsl_run::registers(&registered, &layout, &reply) {
                    Ok(g) => g,
                    Err(e) => return Err(Outcome::Failed(format!("{writer}: the reply: {e}"))),
                };
                for (k, (ir, r)) in views.iter().zip(&got).enumerate() {
                    let warp = warp_view(&registered, r);
                    if ir.killed != warp.killed {
                        report.worst = Some(Class::Beyond);
                        first.get_or_insert_with(|| {
                            format!(
                                "{name} ({writer}) case {c} lane {k}: the IR {} and WARP {}",
                                if ir.killed { "kills" } else { "returns" },
                                if warp.killed { "kills" } else { "returns" }
                            )
                        });
                        continue;
                    }
                    if ir.killed {
                        report.killed += 1;
                        continue;
                    }
                    report.runs += 1;
                    for (slot, v) in registered.outputs.iter().enumerate() {
                        let (a, b) = match v.system {
                            Some(SystemValue::ClipPosition) => (ir.position, warp.position),
                            None | Some(SystemValue::Target(_)) => {
                                (ir.outputs.get(&slot).copied(), warp.outputs.get(&slot).copied())
                            }
                            Some(_) => continue,
                        };
                        let (Some(a), Some(b)) = (a, b) else {
                            if a.is_some() != b.is_some() {
                                report.worst = Some(Class::Beyond);
                                first.get_or_insert_with(|| {
                                    format!("{name} ({writer}) case {c} lane {k}: output {slot} is not carried by one side")
                                });
                            }
                            continue;
                        };
                        let mask = v.provenance.d3d.map_or(0, |d| d.mask);
                        for lane in lanes_of(mask) {
                            let mut class = wgsl_run::classify(true, a[lane], b[lane]);
                            if class == Class::Beyond
                                && amplified
                                    .get(&(slot as u32, lane))
                                    .is_some_and(|x| x.explains(a[lane], b[lane]))
                            {
                                class = Class::Amplified;
                            }
                            *report.lanes.entry(class).or_default() += 1;
                            report.worst = report.worst.max(Some(class));
                            if class != Class::Exact && show {
                                println!(
                                    "{name} ({writer}) case {c} invocation {k} output {slot}.{lane}: {class:?} IR {:#010x} ({}) WARP {:#010x} ({})",
                                    a[lane],
                                    f32::from_bits(a[lane]),
                                    b[lane],
                                    f32::from_bits(b[lane])
                                );
                            }
                            if class == Class::Beyond {
                                first.get_or_insert_with(|| {
                                    format!(
                                        "{name} ({writer}) case {c} lane {k} output {slot}.{lane}: IR {:#010x} ({}) WARP {:#010x} ({})",
                                        a[lane],
                                        f32::from_bits(a[lane]),
                                        b[lane],
                                        f32::from_bits(b[lane])
                                    )
                                });
                            }
                        }
                    }
                }
            }
            beyond.extend(first);
            reports.push((*writer, report));
        }
        Ok((reports, beyond))
    };
    let hash =
        p.ir.provenance
            .source_hash
            .clone()
            .unwrap_or_else(|| "-".to_owned());
    let (mut reports, mut beyond) = match run(false) {
        Ok(r) => r,
        Err(o) => return o,
    };
    let mut pow_defined = false;
    if !beyond.is_empty() && p.writers.iter().any(|(_, text)| text.contains("pow(")) {
        // WGSL leaves `pow` undefined for a zero base and an exponent of 0 or less, and a target compiles it as it likes: a
        // program that then agrees in every lane differs from WARP in that case alone
        match run(true) {
            Ok((again, still)) if still.is_empty() => {
                (reports, beyond, pow_defined) = (again, still, true);
            }
            Ok(_) => {}
            Err(o) => return o,
        }
    }
    let excused = expected(name, &hash);
    match ((!beyond.is_empty()).then(|| beyond.join(" | ")), excused) {
        (Some(first), None) => Outcome::Failed(format!("{first} [{hash}]")),
        (None, Some(_)) if inputs::debug("shader.sin").is_none() && inputs::debug("shader.pow").is_none() => {
            Outcome::Failed(format!(
                "{name}: EXPECTED excuses a program that agrees with WARP in every lane: remove it from the table"
            ))
        }
        (Some(_), Some(why)) => Outcome::Ran {
            reports,
            agreement: Agreement::Expected(why),
        },
        (None, _) => Outcome::Ran {
            reports,
            agreement: if pow_defined {
                Agreement::PowDefined
            } else {
                Agreement::Plain
            },
        },
    }
}
