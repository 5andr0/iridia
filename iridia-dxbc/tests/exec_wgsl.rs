#![allow(clippy::unwrap_used)]
//! The WGSL `iridia_naga` writes, run on Direct3D 12's WARP adapter through wgpu (the harness `tools.wgsl_run`, an
//! executor iridia does not write), against the IR executor (`ProgramShader::from_ir`): each vertex and pixel program of
//! a corpus the IR lift takes, after the passes the gate's P column runs, on the same input registers and constant
//! buffers, in `exec_warp`'s environment (uniform textures, point-filtering samplers clamped, so a sample is independent
//! of its coordinate). A vertex program runs once per vector; a pixel program as a 2x2 quad per vector, its lanes' float
//! inputs stepping along x and y as a rasterizer's do, so its derivatives are the quad's own. Every lane the executor
//! writes is compared, classed by how far apart (Direct3D leaves `exp`, `log`, `sin`, `cos`, `rsqrt` and `rcp` to a few
//! ulp): one further apart than 1e-3 relative, a discard on one side only, or a lane the WGSL does not carry fails.
//!
//! Needs `tools.wgsl_run` (and `tools.wgsl_run_dxc`, the `dxcompiler.dll` that compiles the programs that need Shader
//! Model 6), and a corpus folder. `shader.diff_limit` caps the programs of a corpus.
mod common;
mod inputs;
mod wgsl_run;

use std::collections::BTreeMap;

use iridia_exec::{Dim, ImageLoad, ImageQuery, ImageSample, PixelShader, ProgramShader, QueryKind, Sampler};
use iridia_ir::interface::{SampleType, ScalarType, Stage, SystemValue, TextureDim};
use wgsl_run::{Bound, Class, Environment, Harness, Layout};

/// The vectors each program runs on.
const CASES: u32 = 3;

/// The lanes beyond 1e-3 a report shows of one program.
const SHOWN: usize = 4;

fn has_dxil(bytes: &[u8]) -> bool {
    let word = |at: usize| -> Option<usize> {
        let b = bytes.get(at..at.checked_add(4)?)?;
        usize::try_from(u32::from_le_bytes(b.try_into().ok()?)).ok()
    };
    let Some(count) = word(28) else {
        return false;
    };
    (0..count.min(64)).any(|k| {
        word(32 + 4 * k)
            .and_then(|at| bytes.get(at..at.checked_add(4)?))
            .is_some_and(|k| k == b"DXIL")
    })
}

/// The IR lift of a container after the passes every output takes (`iridia_passes::prepare`).
fn lifted(bytes: &[u8]) -> Result<iridia_ir::Program, String> {
    let mut p = if has_dxil(bytes) {
        iridia_dxil::ir::lift(bytes)
    } else {
        iridia_dxbc::ir::lift(bytes)
    }
    .map_err(|e| format!("the IR lift: {e}"))?;
    iridia_passes::prepare(&mut p)?;
    iridia_ir::validate(&p).map_err(|e| format!("{e:?}"))?;
    Ok(p)
}

/// Answers every request with the texture's or typed buffer view's uniform texel, as the harness's are; a load outside
/// the texture (4 texels a side, one level, a 3D texture 4 deep, an array 2 layers, a cube 6, a cube array 12) or past
/// the buffer's [`wgsl_run::BUFFER_ELEMENTS`] reads zeros. A structured or raw buffer view is its words, zeros past them.
struct Uniform {
    texels: BTreeMap<u32, [u32; 4]>,
    shapes: BTreeMap<u32, (TextureDim, bool, SampleType)>,
    buffers: BTreeMap<u32, SampleType>,
    /// Per structured or raw buffer view: its words and the words of an element.
    words: BTreeMap<u32, (Vec<u32>, u32)>,
}

/// The words of a structured or raw buffer view `slot` in vector `c`: [`wgsl_run::BUFFER_WORDS`] (a structured view's
/// whole elements within them) of floats from -0.25 to 1.75, as unnamed constants are seeded.
fn buffer_words(slot: u32, per: u32, c: u32) -> Vec<u32> {
    (0..wgsl_run::BUFFER_WORDS / per * per)
        .map(|w| {
            let h = common::mix(u64::from(c) << 44 ^ u64::from(slot) << 32 ^ u64::from(w) ^ 0x5b0f);
            (common::unit(h) * 2.0 - 0.25).to_bits()
        })
        .collect()
}

impl Uniform {
    fn layers(&self, slot: u32) -> u32 {
        match self.shapes.get(&slot) {
            Some((TextureDim::Cube, false, _)) => 6,
            Some((TextureDim::Cube, true, _)) => 12,
            Some((_, true, _)) => 2,
            _ => 1,
        }
    }
}

impl Sampler for Uniform {
    fn sample(&mut self, r: iridia_exec::Sample) -> Result<[f32; 4], String> {
        self.sample_image(ImageSample {
            texture: r.texture,
            sampler: r.sampler,
            dim: Dim::D2,
            coord: [r.uv[0], r.uv[1], 0.0],
            layer: r.layer,
            lod: iridia_exec::ImageLod::Implicit,
            offset: [0; 3],
            gather: None,
            depth_ref: None,
            depth: false,
        })
    }
    fn sample_image(&mut self, r: ImageSample<u32>) -> Result<[f32; 4], String> {
        let t = self
            .texels
            .get(&r.texture)
            .ok_or_else(|| format!("no texture t{}", r.texture))?;
        let c = t.map(f32::from_bits);
        Ok(match r.gather {
            Some(k) => [c[usize::from(k)]; 4],
            None => c,
        })
    }
    fn load(&mut self, r: iridia_exec::Load) -> Result<[u32; 4], String> {
        if self.buffers.contains_key(&r.texture) {
            let inside = u32::try_from(r.texel[0]).is_ok_and(|x| x < wgsl_run::BUFFER_ELEMENTS);
            return Ok(if inside { self.texels[&r.texture] } else { [0; 4] });
        }
        let (dim, arrayed, _) = *self
            .shapes
            .get(&r.texture)
            .ok_or_else(|| format!("no texture t{}", r.texture))?;
        let (layer, z) = match dim {
            TextureDim::D3 => (None, r.texel[2]),
            _ if arrayed => (Some(r.texel[2]), 0),
            _ => (None, 0),
        };
        self.load_image(ImageLoad {
            texture: r.texture,
            dim: Dim::D2,
            storage: false,
            texel: [r.texel[0], r.texel[1], z],
            layer,
            sample: None,
            mip: Some(r.mip),
        })
    }
    fn load_image(&mut self, r: ImageLoad<u32>) -> Result<[u32; 4], String> {
        let t = *self
            .texels
            .get(&r.texture)
            .ok_or_else(|| format!("no texture t{}", r.texture))?;
        let (dim, _, _) = self.shapes[&r.texture];
        let comps = match dim {
            TextureDim::D1 => 1,
            TextureDim::D2 => 2,
            TextureDim::D3 | TextureDim::Cube => 3,
        };
        let layers = self.layers(r.texture);
        let inside = r.mip.unwrap_or(0) == 0
            && r.texel[..comps].iter().all(|t| (0..4).contains(t))
            && r.layer.is_none_or(|l| (0..layers as i32).contains(&l));
        Ok(if inside { t } else { [0; 4] })
    }
    fn buffer_elements(&mut self, slot: u32) -> Option<u32> {
        if let Some((words, per)) = self.words.get(&slot) {
            return u32::try_from(words.len()).ok().map(|n| n / per);
        }
        self.buffers
            .contains_key(&slot)
            .then_some(wgsl_run::BUFFER_ELEMENTS)
    }
    fn structured(&mut self, slot: u32, element: u32, offset: u32) -> Result<u32, String> {
        let (words, per) = self
            .words
            .get(&slot)
            .ok_or_else(|| format!("no buffer t{slot}"))?;
        let at = element.checked_mul(*per).and_then(|w| w.checked_add(offset / 4));
        Ok(at.and_then(|w| words.get(w as usize)).copied().unwrap_or(0))
    }
    fn query_image(&mut self, r: ImageQuery<u32>) -> Result<[u32; 4], String> {
        let layers = self.layers(r.texture);
        Ok(match r.kind {
            QueryKind::Size if r.level.unwrap_or(0) > 0 => [0; 4],
            QueryKind::Size => [4, 4, 4, 0],
            QueryKind::Levels => [1, 0, 0, 0],
            QueryKind::Layers => [layers, 0, 0, 0],
            QueryKind::Samples => [1, 0, 0, 0],
        })
    }
}

#[derive(Default)]
struct Tally {
    programs: usize,
    stages: BTreeMap<String, usize>,
    compared: usize,
    runs: usize,
    discarded: usize,
    refused: BTreeMap<String, (usize, String)>,
    gaps: BTreeMap<String, (usize, String)>,
    unrun: BTreeMap<String, (usize, String)>,
    errors: BTreeMap<String, (usize, String)>,
    lanes: BTreeMap<Class, usize>,
    by_program: BTreeMap<String, Class>,
    worst: Vec<String>,
}

fn tally(map: &mut BTreeMap<String, (usize, String)>, why: &str, at: &str) {
    map.entry(common::reason(why)).or_insert((0, at.to_owned())).0 += 1;
}

/// A quad of vector `c`: each lane's own inputs (vectors `4c` to `4c + 3`), every interpolated float input then made the
/// first lane's plus a step of its own along x and y, the first lane's constants and facing, positions a pixel apart.
fn quad(b: &ProgramShader, c: u32) -> Vec<common::Case> {
    let mut lanes: Vec<common::Case> = (0..4).map(|k| common::case(b, c * 4 + k)).collect();
    let interface = b.interface();
    for v in &interface.inputs {
        let Some(sig) = v.provenance.d3d else { continue };
        let r = sig.register as usize;
        if r >= lanes[0].inputs.len() {
            continue;
        }
        let interpolated = v.system.is_none()
            && v.value.scalar == ScalarType::Float
            && v.interp
                .is_none_or(|i| i.mode != iridia_ir::interface::InterpMode::Flat);
        for lane in (0..4usize).filter(|l| sig.mask >> l & 1 != 0) {
            let base = lanes[0].inputs[r][lane];
            for (k, l) in lanes.iter_mut().enumerate().skip(1) {
                l.inputs[r][lane] = if interpolated {
                    let h = common::mix(u64::from(c) << 40 ^ (r as u64) << 8 ^ lane as u64 ^ 0x0dd);
                    let dx = (common::unit(h) - 0.5) / 16.0;
                    let dy = (common::unit(common::mix(u64::from(h) ^ 1)) - 0.5) / 16.0;
                    (f32::from_bits(base) + dx * (k & 1) as f32 + dy * (k >> 1) as f32).to_bits()
                } else {
                    base
                };
            }
            if v.system == Some(SystemValue::FragCoord) && lane < 2 {
                for (k, l) in lanes.iter_mut().enumerate() {
                    let at = if lane == 0 { k & 1 } else { k >> 1 };
                    l.inputs[r][lane] = (0.5 + at as f32).to_bits();
                }
            }
        }
    }
    let first = lanes[0].buffers.clone();
    for l in &mut lanes {
        l.buffers = first.clone();
    }
    lanes
}

fn run_program(harness: &Harness, name: &str, bytes: &[u8], t: &mut Tally) {
    t.programs += 1;
    let p = match lifted(bytes) {
        Ok(p) => p,
        Err(e) => {
            tally(&mut t.unrun, &e, name);
            return;
        }
    };
    let stage = p.interface.stage;
    if !matches!(stage, Stage::Vertex | Stage::Pixel) {
        let why = format!(
            "{} program: this test runs vertex and pixel programs",
            stage.with_article()
        );
        tally(&mut t.gaps, &why, name);
        return;
    }
    *t.stages.entry(format!("{stage:?}")).or_default() += 1;
    if let Some(why) = wgsl_run::gap(&p) {
        tally(&mut t.gaps, &why, name);
        return;
    }
    let wgsl = match iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA) {
        Ok(w) => w,
        Err(e) => {
            tally(&mut t.refused, &e.0, name);
            return;
        }
    };
    let layout = match Layout::naga(&p.interface) {
        Ok(l) => l,
        Err(e) => {
            tally(&mut t.errors, &format!("the layout: {e}"), name);
            return;
        }
    };
    let bindings: Vec<_> = iridia_naga::BindingPlan::d3d(&p.interface)
        .unwrap()
        .bindings
        .iter()
        .map(|b| {
            (
                b.resource,
                Bound {
                    group: b.group,
                    binding: b.binding,
                },
            )
        })
        .collect();
    let interface = p.interface.clone();
    let b = match ProgramShader::from_ir(p) {
        Ok(b) => b,
        Err(e) => {
            tally(&mut t.unrun, &format!("B: {e}"), name);
            return;
        }
    };
    let shapes = wgsl_run::textures(&interface);
    let buffers = wgsl_run::typed_buffers(&interface);
    let word_buffers = wgsl_run::word_buffers(&interface);
    let float_registers: BTreeMap<u32, bool> = interface
        .outputs
        .iter()
        .filter_map(|v| {
            v.provenance
                .d3d
                .map(|d| (d.register, v.value.scalar == ScalarType::Float))
        })
        .collect();
    let mut worst = Class::Exact;
    let mut compared = false;
    for c in 0..CASES {
        let lanes = if stage == Stage::Pixel {
            quad(&b, c)
        } else {
            vec![common::case(&b, c)]
        };
        let texels = shapes
            .iter()
            .map(|(&slot, &(_, _, s))| (slot, s))
            .chain(buffers.iter().map(|(&slot, &s)| (slot, s)))
            .map(|(slot, s)| (slot, wgsl_run::texel(slot, c, s)))
            .collect();
        let mut sampler = Uniform {
            texels,
            shapes: shapes.clone(),
            buffers: buffers.clone(),
            words: word_buffers
                .iter()
                .map(|(&slot, &per)| (slot, (buffer_words(slot, per, c), per)))
                .collect(),
        };
        let refs = common::refs(&lanes[0], &b);
        let executions: Result<Vec<iridia_exec::Execution>, String> =
            if stage == Stage::Pixel && b.needs_quad() {
                b.execute_quad(
                    [
                        &lanes[0].inputs,
                        &lanes[1].inputs,
                        &lanes[2].inputs,
                        &lanes[3].inputs,
                    ],
                    &refs,
                    &mut sampler,
                )
                .map(Vec::from)
                .map_err(|e| e.to_string())
            } else {
                lanes
                    .iter()
                    .map(|l| {
                        b.execute(&l.inputs, &refs, &mut sampler)
                            .map_err(|e| e.to_string())
                    })
                    .collect()
            };
        let executions = match executions {
            Ok(e) => e,
            Err(e) => {
                tally(&mut t.unrun, &format!("B: {e}"), name);
                continue;
            }
        };
        let env = Environment {
            texels: sampler.texels,
            buffers: lanes[0].buffers.clone(),
            words: sampler
                .words
                .into_iter()
                .map(|(slot, (words, _))| (slot, words))
                .collect(),
        };
        let invocations: Vec<&[[u32; 4]]> = lanes.iter().map(|l| l.inputs.as_slice()).collect();
        let job = match wgsl_run::job(&interface, &wgsl, &layout, &bindings, &invocations, &env) {
            Ok(j) => j,
            Err(e) => {
                tally(&mut t.errors, &format!("the job: {e}"), name);
                return;
            }
        };
        let reply = match harness.run(&job) {
            Ok(r) => r,
            Err(e) => {
                tally(&mut t.errors, &format!("the harness: {e}"), name);
                return;
            }
        };
        let got = match wgsl_run::registers(&interface, &layout, &reply) {
            Ok(g) => g,
            Err(e) => {
                tally(&mut t.errors, &format!("the reply: {e}"), name);
                return;
            }
        };
        for (k, (e, h)) in executions.iter().zip(&got).enumerate() {
            t.runs += 1;
            if e.discarded != !h.returned {
                worst = Class::Beyond;
                if t.worst.len() >= SHOWN {
                    continue;
                }
                t.worst.push(format!(
                    "{name} case {c} lane {k}: B {} and WARP {}",
                    if e.discarded { "discards" } else { "returns" },
                    if h.returned { "returns" } else { "discards" }
                ));
                continue;
            }
            if e.discarded {
                t.discarded += 1;
                continue;
            }
            compared = true;
            for (register, (o, &mask)) in e.outputs.iter().zip(&e.output_masks).enumerate() {
                let register = register as u32;
                for lane in (0..4usize).filter(|l| mask >> l & 1 != 0) {
                    let carried = h.masks.get(&register).is_some_and(|m| m >> lane & 1 != 0);
                    if !carried {
                        worst = Class::Beyond;
                        if t.worst.len() < SHOWN {
                            t.worst.push(format!(
                                "{name} case {c} lane {k}: o{register}.{lane} is not carried"
                            ));
                        }
                        continue;
                    }
                    let float = float_registers.get(&register).copied().unwrap_or(true);
                    let class = wgsl_run::classify(float, o[lane], h.outputs[&register][lane]);
                    *t.lanes.entry(class).or_default() += 1;
                    worst = worst.max(class);
                    if class == Class::Beyond && t.worst.len() < SHOWN {
                        t.worst.push(format!(
                            "{name} case {c} lane {k} o{register}.{lane}: B {:#010x} ({}) WARP {:#010x} ({})",
                            o[lane],
                            f32::from_bits(o[lane]),
                            h.outputs[&register][lane],
                            f32::from_bits(h.outputs[&register][lane])
                        ));
                    }
                }
            }
        }
    }
    if compared || worst == Class::Beyond {
        t.compared += 1;
        t.by_program.insert(name.to_owned(), worst);
    }
}

fn report(title: &str, t: &Tally) {
    println!(
        "{title}: {} programs ({:?}), {} compared, {} runs ({} discarded on both sides)",
        t.programs, t.stages, t.compared, t.runs, t.discarded
    );
    let total: usize = t.lanes.values().sum();
    for (c, n) in &t.lanes {
        println!("  {:12} {n:8} lanes of {total}", wgsl_run::CLASSES[*c as usize]);
    }
    let mut programs: BTreeMap<Class, usize> = BTreeMap::new();
    for c in t.by_program.values() {
        *programs.entry(*c).or_default() += 1;
    }
    for (c, n) in &programs {
        println!(
            "  programs whose worst lane is {:12} {n}",
            wgsl_run::CLASSES[*c as usize]
        );
    }
    for (title, map) in [
        ("not written (iridia_naga refuses)", &t.refused),
        ("not on the harness", &t.gaps),
        ("not run", &t.unrun),
        ("ERROR", &t.errors),
    ] {
        for (why, (n, at)) in map {
            println!("  {title} {n:6}  {why}  (e.g. {at})");
        }
    }
    let beyond: Vec<&str> = t
        .by_program
        .iter()
        .filter(|(_, c)| **c == Class::Beyond)
        .map(|(n, _)| n.rsplit(['/', '\\']).next().unwrap_or(n))
        .collect();
    if !beyond.is_empty() {
        println!("  programs beyond 1e-3: {}", beyond.join(" "));
    }
    for w in &t.worst {
        println!("  BEYOND {w}");
    }
}

/// The vertex and pixel programs of `corpus` through the harness, as many at a time as `[run] threads` allows.
fn corpus_over(corpus: &common::Corpus) {
    let Some(harness) = Harness::configured() else {
        println!(
            "tools.wgsl_run is not configured: skipped (build tools/wgsl-run with `cargo build --release` there, and set \
             tools.wgsl_run_dxc to the dxcompiler.dll of DXC 1.8.2502 or later)"
        );
        return;
    };
    if !harness.has_dxc() {
        println!(
            "tools.wgsl_run_dxc is not configured: FXC compiles every program, so those that need Shader Model 6 fail \
             (set it to the dxcompiler.dll of DXC 1.8.2502 or later, dxil.dll beside it)"
        );
    }
    let files = &corpus.names[..corpus.names.len().min(common::diff_limit())];
    let next = std::sync::atomic::AtomicUsize::new(0);
    let tallies: Vec<std::sync::Mutex<Tally>> = files.iter().map(|_| Default::default()).collect();
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(8);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(file) = files.get(i) else { break };
                    let bytes = corpus.read(file);
                    run_program(
                        &harness,
                        &corpus.at(file),
                        &bytes,
                        &mut tallies[i].lock().unwrap(),
                    );
                }
            });
        }
    });
    let mut t = Tally::default();
    for one in tallies {
        let one = one.into_inner().unwrap();
        t.programs += one.programs;
        t.compared += one.compared;
        t.runs += one.runs;
        t.discarded += one.discarded;
        for (k, n) in one.stages {
            *t.stages.entry(k).or_default() += n;
        }
        for (k, n) in one.lanes {
            *t.lanes.entry(k).or_default() += n;
        }
        for (mine, theirs) in [
            (&mut t.refused, one.refused),
            (&mut t.gaps, one.gaps),
            (&mut t.unrun, one.unrun),
            (&mut t.errors, one.errors),
        ] {
            for (why, (n, at)) in theirs {
                mine.entry(why).or_insert((0, at)).0 += n;
            }
        }
        t.by_program.extend(one.by_program);
        t.worst.extend(one.worst);
    }
    report(&corpus.name, &t);
    assert!(t.errors.is_empty(), "{} kinds of harness errors", t.errors.len());
    assert!(
        t.worst.is_empty(),
        "{} lanes or runs differ from WARP",
        t.worst.len()
    );
}

fn corpus(game: inputs::Game) {
    if let Some(corpus) = common::corpus(game) {
        corpus_over(&corpus);
    }
}

#[test]
fn sekiro_programs_run_as_naga_wgsl_on_warp() {
    corpus(inputs::Game::Sekiro);
}
#[test]
fn riders_republic_programs_run_as_naga_wgsl_on_warp() {
    corpus(inputs::Game::Riders);
}
#[test]
fn unity_programs_run_as_naga_wgsl_on_warp() {
    corpus(inputs::Game::Sledding);
}
#[test]
fn elden_programs_run_as_naga_wgsl_on_warp() {
    corpus(inputs::Game::Elden);
}

/// A folder of containers outside the configured corpora (`shader.outside`); without it, nothing to run.
#[test]
fn outside_programs_run_as_naga_wgsl_on_warp() {
    let Some(root) = inputs::debug("shader.outside").map(std::path::PathBuf::from) else {
        return;
    };
    corpus_over(&common::folder(&root));
}
