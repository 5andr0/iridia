//! Physical inputs and comparisons shared by the WARP corpus tests and the IR gate.
use crate::{common, inputs, wgsl_run};

mod quantized_shape;
mod word_types;

use std::collections::{BTreeMap, BTreeSet};

use iridia_exec::{
    Backend, Dim, Element, Executable, ImageLoad, ImageQuery, ImageSample, PixelShader, ProgramShader,
    QueryKind, Resources, Sampler,
};
use iridia_ir::interface::{
    BufferKind, ResourceId, ResourceKind, SampleType, ScalarType, Stage, StorageTarget, SystemValue,
    TextureDim,
};
use iridia_ir::{Body, Node, Region};
use wgsl_run::amplified::Source;
use wgsl_run::{Bound, Class, Environment, Harness, Layout};

/// The vectors each program runs on.
const CASES: u32 = 3;

/// The lanes beyond 1e-3 a report shows of one program.
const SHOWN: usize = 4;

/// The lanes beyond 1e-3 a report prints in all.
const WORST_SHOWN: usize = 40;

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
        // a depth texture: its depth is the first lane, and a comparison sample gives 1 where `depth_ref <= depth`
        // (`wgsl_run::COMPARE`) and 0 elsewhere, in every lane of a gather as of the sample
        if r.depth || r.depth_ref.is_some() {
            let depth = c[0];
            let compared = |reference: f32| if reference <= depth { 1.0 } else { 0.0 };
            let value = r.depth_ref.map_or(depth, compared);
            return Ok([value; 4]);
        }
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
    compute_class_changes: BTreeMap<String, BTreeMap<String, usize>>,
    compute_program_class_changes: BTreeMap<String, (Class, Class)>,
    compute_word_origins: BTreeMap<String, BTreeMap<word_types::WordType, usize>>,
    origin_controlled_vectors: usize,
    programs: usize,
    stages: BTreeMap<String, usize>,
    compared: usize,
    runs: usize,
    discarded: usize,
    refused: BTreeMap<String, (usize, String)>,
    gaps: BTreeMap<String, (usize, String)>,
    unrun: BTreeMap<String, (usize, String)>,
    errors: BTreeMap<String, (usize, String)>,
    /// Programs written for a browser and held to Tint, and those a browser cannot carry.
    tinted: BTreeMap<String, usize>,
    /// Programs whose every run discards on both sides, so no lane is compared.
    all_discarded: BTreeMap<String, (usize, String)>,
    /// Programs Tint refuses for a barrier whose uniformity WGSL's analysis cannot prove.
    unprovable: BTreeMap<String, (usize, String)>,
    browser_refused: BTreeMap<String, (usize, String)>,
    lanes: BTreeMap<Class, usize>,
    /// Per compute program, the words that differ from WARP that the program's own writes leave unordered, by what shows
    /// them so: WARP gives the word differently on two runs of the same job; more stores than store instructions (no
    /// loop) gave it different values, so two invocations stored to it; two stores gave it different values, which one
    /// invocation's loop could do (not shown to come from two).
    unordered: BTreeMap<String, [usize; 3]>,
    /// Per compute program that differs from WARP (beyond 1e-3, or in words its writes leave unordered): its source hash and
    /// why [`COMPUTE_EXPECTED`] lets it, where it does.
    differs: BTreeMap<String, (String, Option<&'static str>)>,
    /// Per vertex or pixel program with a lane beyond 1e-3 (or a discard or a missing lane): its source hash and why
    /// [`GRAPHICS_EXPECTED`] lets it, where it does.
    graphics_differs: BTreeMap<String, (String, Option<&'static str>)>,
    /// Per such program, why its best shown word is unordered, with the grade.
    unordered_why: BTreeMap<String, (usize, String)>,
    /// Per source of an amplified lane: the lanes and the programs.
    amplified: BTreeMap<Source, (usize, BTreeSet<String>)>,
    by_program: BTreeMap<String, Class>,
    /// Per compared program, its lanes by class.
    program_lanes: BTreeMap<String, BTreeMap<Class, usize>>,
    worst: Vec<String>,
}

/// Programs whose barrier Tint cannot prove is reached by every invocation, though it is: each by program name and source hash,
/// with why. A barrier-uniformity error on any program not listed here fails the run.
pub(crate) const UNPROVABLE_BARRIERS: &[(&str, &str, &str)] = &[
    (
        "GXLightOrganize16.cpo",
        "sha256:3de7407b369589024aecb01052cc686608f07b6e0e43276ba467f85f95a9b20d",
        UNIFORM_BY_ARITHMETIC,
    ),
    (
        "GXLightOrganize16Sdw.cpo",
        "sha256:4329a89cdca46549936eaa1008f27b18c6f16ec431c419a5293fb19725f45bb9",
        UNIFORM_BY_ARITHMETIC,
    ),
    (
        "GXLightOrganize8_noopt.cpo",
        "sha256:af75218b2d3179d480fde0c153e2365cced4f08a83c4fa1b587bd7214b143bca",
        UNIFORM_BY_ARITHMETIC,
    ),
    (
        "GXLightOrganize8Sdw_noopt.cpo",
        "sha256:8215d3fc4d1460ed64a01fa7ba6808d7061e5bf880af504a28267266ee633f2c",
        UNIFORM_BY_ARITHMETIC,
    ),
];

/// The reason of the entries of [`UNPROVABLE_BARRIERS`]. Open: an IR rewrite that cancels the thread-index offset (the
/// induction variable minus its start) would make these provable in WGSL.
const UNIFORM_BY_ARITHMETIC: &str = "counter starts at the thread index and the condition subtracts it again: uniform by arithmetic, not provable in WGSL";

/// The reason `name` (of the program with source hash `hash`) may fail with a barrier that is not provably uniform, if listed.
pub(crate) fn unprovable_barrier(name: &str, hash: &str) -> Option<&'static str> {
    UNPROVABLE_BARRIERS
        .iter()
        .find(|(n, h, _)| *n == short(name) && *h == hash)
        .map(|(_, _, why)| *why)
}

/// A program's file name, without its folder.
fn short(name: &str) -> &str {
    name.rsplit(['/', '\\']).next().unwrap_or(name)
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

/// The WGSL of a program in naga's dialect by the writer `shader.writer` names: `naga` (the default, `iridia_naga`) or `wgsl`
/// (`iridia_wgsl`).
fn write(p: &iridia_ir::Program) -> Result<String, String> {
    match inputs::debug("shader.writer").as_deref() {
        None | Some("naga") => iridia_naga::wgsl(p, &iridia_naga::Profile::NAGA).map_err(|e| e.0),
        Some("wgsl") => iridia_wgsl::wgsl(p, &iridia_wgsl::Profile::NAGA).map_err(|e| e.0),
        Some(other) => panic!("shader.writer is naga or wgsl, not {other}"),
    }
}

/// The program's WGSL as a browser reads it, by the writer `shader.writer` names.
fn write_browser(p: &iridia_ir::Program) -> Result<String, String> {
    match inputs::debug("shader.writer").as_deref() {
        None | Some("naga") => iridia_naga::wgsl(p, &iridia_naga::Profile::BROWSER).map_err(|e| e.0),
        _ => iridia_wgsl::wgsl(p, &iridia_wgsl::Profile::BROWSER).map_err(|e| e.0),
    }
}

/// `tools.tint`, where `shader.tint` is `1`.
fn tint() -> Option<&'static wgsl_run::Tint> {
    static TINT: std::sync::OnceLock<Option<wgsl_run::Tint>> = std::sync::OnceLock::new();
    TINT.get_or_init(|| {
        if inputs::debug("shader.tint").as_deref() == Some("1") {
            wgsl_run::tint()
        } else {
            None
        }
    })
    .as_ref()
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
    // Tint takes every stage a writer writes, apart from the harness (compute programs are WebGPU's too)
    if let Some(tint) = tint() {
        let at = |why: &str| format!("{stage:?}: {why}");
        match write_browser(&p) {
            Ok(text) => {
                *t.tinted.entry(format!("{stage:?}")).or_default() += 1;
                if let Err(e) = tint.check(name, &text) {
                    let hash = p.provenance.source_hash.as_deref().unwrap_or("-");
                    let barrier = e.contains("Barrier' must only be called from uniform control flow");
                    if barrier {
                        println!("barrier-uniformity {} {hash}", short(name));
                    }
                    // a barrier that is reached by every invocation, by arithmetic WGSL's analysis cannot follow, is a
                    // class of the source and counted; the programs of that class are named, and any other is a fault
                    match unprovable_barrier(name, hash).filter(|_| barrier) {
                        Some(why) => tally(&mut t.unprovable, &at(&format!("tint: {why}")), name),
                        None => tally(&mut t.errors, &at(&format!("tint: {e}")), name),
                    }
                }
            }
            Err(e) => tally(&mut t.browser_refused, &at(&e), name),
        }
    }
    if stage == Stage::Compute {
        let hash = p.provenance.source_hash.clone().unwrap_or_else(|| "-".to_owned());
        if compute_wanted(name, &hash) {
            run_compute(harness, name, &hash, p, t);
            return;
        }
    }
    if matches!(stage, Stage::Vertex | Stage::Pixel) && !compute_only() && !graphics_wanted(name) {
        tally(
            &mut t.gaps,
            "a vertex or pixel program outside the sample (shader.graphics=1 runs every one)",
            name,
        );
        return;
    }
    if !matches!(stage, Stage::Vertex | Stage::Pixel) || compute_only() {
        let why = if stage == Stage::Compute {
            "a compute program outside the sample (shader.compute=1 runs every one)".to_owned()
        } else {
            format!(
                "{} program: this test runs vertex and pixel programs",
                stage.with_article()
            )
        };
        if !compute_only() {
            tally(&mut t.gaps, &why, name);
        }
        return;
    }
    *t.stages.entry(format!("{stage:?}")).or_default() += 1;
    if let Some(why) = wgsl_run::gap(&p) {
        tally(&mut t.gaps, &why, name);
        return;
    }
    let wgsl = match write(&p) {
        Ok(w) => w,
        Err(e) => {
            tally(&mut t.refused, &e, name);
            return;
        }
    };
    let wgsl = match inputs::debug("shader.sin").as_deref() {
        None => wgsl,
        Some("reference") => match wgsl_run::substitute_sin(&wgsl) {
            Ok(w) => w,
            Err(e) => {
                tally(&mut t.gaps, &e, name);
                return;
            }
        },
        Some(other) => panic!("shader.sin is reference, not {other}"),
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
    let amplified = wgsl_run::amplified::outputs(&p);
    let downstream = wgsl_run::amplified::downstream(&p);
    let hash = p.provenance.source_hash.clone().unwrap_or_else(|| "-".to_owned());
    let excused = graphics_expected(name, &hash);
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
    let mut counts: BTreeMap<Class, usize> = BTreeMap::new();
    let mut compared = false;
    let mut killed = false;
    // `shader.dump`: a folder that takes the text of each program that runs, to set one writer's text against another's
    if let Some(dir) = inputs::debug("shader.dump") {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(
            std::path::Path::new(&dir).join(format!("{}.wgsl", short(name))),
            &wgsl,
        );
    }
    // `shader.lanes`: every lane that is not exact, of the programs whose name has this in it
    let show_lanes = inputs::debug("shader.lanes").is_some_and(|s| name.contains(&s));
    if show_lanes {
        println!(
            "{name}
{wgsl}"
        );
    }
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
                killed = true;
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
                    let warp = h.outputs[&register][lane];
                    let mut class = wgsl_run::classify(float, o[lane], warp);
                    if class == Class::Beyond
                        && let Some(a) = amplified.get(&(register, lane))
                        && a.explains(o[lane], warp)
                    {
                        class = Class::Amplified;
                        let (lanes, programs) = t.amplified.entry(a.source).or_default();
                        *lanes += 1;
                        programs.insert(short(name).to_owned());
                    }
                    *t.lanes.entry(class).or_default() += 1;
                    *counts.entry(class).or_default() += 1;
                    worst = worst.max(class);
                    if class != Class::Exact && show_lanes {
                        println!(
                            "{name} case {c} invocation {k} o{register}.{lane}: {class:?} B {:#010x} ({}) WARP {warp:#010x} ({}), inputs {:x?}",
                            o[lane],
                            f32::from_bits(o[lane]),
                            f32::from_bits(warp),
                            lanes[k].inputs
                        );
                    }
                    if class == Class::Beyond && t.worst.len() < SHOWN && excused.is_none() {
                        let below = downstream.get(&(register, lane)).map_or(String::new(), |s| {
                            format!(", downstream of an amplified {} (no bound to check)", s.name())
                        });
                        t.worst.push(format!(
                            "{name} case {c} lane {k} o{register}.{lane}: B {:#010x} ({}) WARP {warp:#010x} ({}){below}",
                            o[lane],
                            f32::from_bits(o[lane]),
                            f32::from_bits(warp)
                        ));
                    }
                }
            }
        }
    }
    if killed && !compared && worst != Class::Beyond {
        tally(
            &mut t.all_discarded,
            "every run discarded on both sides: nothing to compare",
            name,
        );
    }
    if worst == Class::Beyond {
        t.graphics_differs.insert(name.to_owned(), (hash, excused));
    } else if excused.is_some() && inputs::debug("shader.sin").is_none() {
        tally(
            &mut t.errors,
            "a program GRAPHICS_EXPECTED excuses agrees with WARP in every lane: remove it from the table",
            name,
        );
    }
    if compared || worst == Class::Beyond {
        t.compared += 1;
        t.by_program.insert(name.to_owned(), worst);
        t.program_lanes.insert(name.to_owned(), counts);
    }
}

/// Which compute programs run, as `shader.compute` says.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ComputeScope {
    /// `0`: none.
    Off,
    /// Unset: a fixed sample, and every program [`COMPUTE_EXPECTED`] names.
    Sample,
    /// `1`: every one.
    All,
    /// `sample`: the sample, and no vertex or pixel program.
    SampleOnly,
    /// `only`: every one, and no vertex or pixel program.
    Only,
}

fn compute_scope() -> ComputeScope {
    match inputs::debug("shader.compute").as_deref() {
        None => ComputeScope::Sample,
        Some("0") => ComputeScope::Off,
        Some("1") => ComputeScope::All,
        Some("sample") => ComputeScope::SampleOnly,
        Some("only") => ComputeScope::Only,
        Some(other) => panic!("shader.compute is 0, 1, sample or only, not {other}"),
    }
}

/// Whether nothing but compute programs runs (`shader.compute` is `sample` or `only`).
fn compute_only() -> bool {
    matches!(compute_scope(), ComputeScope::SampleOnly | ComputeScope::Only)
}

/// One compute program in this many (by a hash of its file name) is in the sample.
const COMPUTE_SAMPLE: u64 = 16;

/// One vertex or pixel program in this many (by a hash of its file name) is in the sample, unless `shader.graphics_every`
/// says another number: Elden Ring alone has some 48,000, and the sample is what a gate can afford of them.
const GRAPHICS_SAMPLE: u64 = 128;

/// Whether the program `name` is in the one-in-`every` sample, by a hash of its file name.
fn sampled(name: &str, every: u64) -> bool {
    let fnv = short(name).bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    common::mix64(fnv).is_multiple_of(every)
}

/// Whether the vertex or pixel program `name` runs: `shader.graphics` is `1` (every one), `0` (none), or unset and the
/// program is in the sample (or `shader.only` names programs, which are the ones to run).
fn graphics_wanted(name: &str) -> bool {
    match inputs::debug("shader.graphics").as_deref() {
        None | Some("sample") if inputs::debug("shader.only").is_some() => true,
        None | Some("sample") => {
            let every = inputs::debug("shader.graphics_every").map_or(GRAPHICS_SAMPLE, |v| {
                v.parse().expect("shader.graphics_every is a number")
            });
            sampled(name, every)
        }
        Some("1") => true,
        Some("0") => false,
        Some(other) => panic!("shader.graphics is 0, 1 or sample, not {other}"),
    }
}

/// Whether the compute program `name` (of source hash `hash`) runs.
fn compute_wanted(name: &str, hash: &str) -> bool {
    match compute_scope() {
        ComputeScope::Off => false,
        ComputeScope::All | ComputeScope::Only => true,
        ComputeScope::Sample | ComputeScope::SampleOnly => {
            compute_expected(name, hash).is_some() || sampled(name, COMPUTE_SAMPLE)
        }
    }
}

/// Programs whose words are known to differ from WARP's with the executor and the writers right: each by program name and
/// source hash, with why. A program with other source of the same name, or the same source under another name, is not
/// listed.
const COMPUTE_EXPECTED: &[(&str, &str, &str)] = &[
    (
        "1028.dxbc",
        "sha256:4a06a8ae41d5ef5131001e15c861d913077b09c6b6f030be8988bc93b491e11e",
        WARP_SEEDED_NONFINITE,
    ),
    // Generated from all 48 rows of the reviewed per-word proof (capture 81015009, 83 words).
    (
        "M[AMSN_V][Ov_N][HierarchicalPivotAnim_3rdLayer]_Animation.cpo",
        "sha256:405aedbace663895dc63d2a37969bb908c2c4ffbf15ecd85076428374ce74e3b",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "M[AMSN_V][Ov_N][HierarchicalPivotAnim_Child]_Animation.cpo",
        "sha256:ce95a820786e6ad4bb1352a812724313d075b6e7c055199ac2d800db8a5a0589",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_50107_w32187_Animation.cpo",
        "sha256:84dabbe985c7bfcd54faba4e0e3f940ed437e076a3493c61d0e98c09d1b11a7a",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010f_w312187_Animation.cpo",
        "sha256:bbaeb10f87c7f05579ebc0140363f20fe8ec30bcb5ce02b7dc751f1f7e17f898",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010f_w312987_Animation.cpo",
        "sha256:8901a0baeeca6cb6026e6b78ed10df5479c13777e5e941435425c0071a3b1a67",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010f_w3129a7_Animation.cpo",
        "sha256:acd6fd6f308d3a13295f37d781edb96cacb24aa626ec1dce6ef8f1726b0afc20",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010f_w3131c7_Animation.cpo",
        "sha256:31757b02312eeaa9fd6e93dfb238e8b2f1c45a3d4ca60983c618bcfddbe2eff9",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010f_w32187_Animation.cpo",
        "sha256:9f1b05b915f2a539ddf21fb7b7a1e65005eb9906692fe24910baa261fc1fac06",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010f_w321a7_Animation.cpo",
        "sha256:84862caa82620b214909f356deb4c48769d3aaf7281dc1b3c68f176b80466c25",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010f_w329a7_Animation.cpo",
        "sha256:2d5f8e0cdef91dbf8c53f6f859cb2e5f7bf703960681e3e8e902ff121b266b66",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010f_w33187_Animation.cpo",
        "sha256:811b2efc0b15ba667fc2d16e5b31a96bdb82c9c48e5686554242a990749afea4",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010f_w331e7_Animation.cpo",
        "sha256:f39f1ad25408257528b18e055adc1009069331702b716f861e7ffbb87996d3ad",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fb_w312187_Animation.cpo",
        "sha256:39d62a3f1e0647f82bca079af0cf07b7b095ee74483bb3beae58c98228261e11",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fb_w3129a7_Animation.cpo",
        "sha256:c89114a1a80ab38c4e37f93e89920faa275ae16e64a397ae2f8b362633ccec9b",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmo_w312187_Animation.cpo",
        "sha256:8e81fbd71711631b7de8c32dca492b24e4d39cad2b1c5e2c585ef1787f661c2f",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmo_w312987_Animation.cpo",
        "sha256:ac9ae718841f7f0996f8e901ca07de0cfe50d360e3c603082940ca7de63c5583",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmo_w3129a7_Animation.cpo",
        "sha256:d887f62f366434e2b3ada89c6ce75b347c2c6eb2924f3299fee3566feca4a171",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmo_w32187_Animation.cpo",
        "sha256:81fabadbca6fd9a64bd1d98856450f8155dff8e5306e755d60c0645662a12f69",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmo_w321a7_Animation.cpo",
        "sha256:56735a4f9e9fb53c25d90283177d7dcd2a9b09edfed6350fe0f32688c47f973c",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmo_w32987_Animation.cpo",
        "sha256:541163162f931d50f844c2d4b764cb09e5cb3edf3b9c6c2c6209e68359fc58de",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmo_w329a7_Animation.cpo",
        "sha256:4f91a8245ac2d6e7552799de201379a0603ded24a3e4f4d49dbb1c5a3967a16b",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmo_w33187_Animation.cpo",
        "sha256:d2b7e99d93aad1dc14ff63e3462b0a917d0f9990deab806d378ba8a52393f142",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmo_w331c7_Animation.cpo",
        "sha256:bc076a4288777a4c62928f8f9fe869a71b590913c0f38f184f19e64b3dc8bebd",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmo_w331e7_Animation.cpo",
        "sha256:0e18e767f8d247226800fc6cfa31a2d4db11398ec9018d07dc5d641c903de315",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010fdmof9_w32187_Animation.cpo",
        "sha256:5709be42c2f445be148757db14c69073223ff609a4266f421ba01ced094eedb6",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff1_w3129a7_Animation.cpo",
        "sha256:81e12838bccb4faed69182dc50bb1591b0d02804f33d16bd96eec606881296a2",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff4_w3129a7_Animation.cpo",
        "sha256:3a47a0c2b21e998972b541226be68e6191e7eb4f1923d91b07c5ffc269f15044",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff4_w3131c7_Animation.cpo",
        "sha256:b8174c78c16ecc8e28ed4b487dd6aa9a9a2c4d254955c7624a969e92a5c29735",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff4b_w312187_Animation.cpo",
        "sha256:b78f17ffe9adbc2e6b67a1eaaf5404f1a4c3770519db2da64d34ac724b882361",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff4b_w312987_Animation.cpo",
        "sha256:78a182b80a3fd0364aa9e686a0fe4afbd044494ba926166150bc5764c8fd2e0e",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff4b_w3129a7_Animation.cpo",
        "sha256:4352d07a657db7b28d2e68e496aac28cbb0b69f00280c767af50ac7848ff5dcf",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff8_w312187_Animation.cpo",
        "sha256:481ddc9d9b34ca955124e31e98193e2a9ddcca8b4aa991395b3eb255f72964e9",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff8b_w312187_Animation.cpo",
        "sha256:48c666b68d8c546fb1e1a387be84873842e54a23289a03d411cde7c91985c7e5",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff8b_w3129a7_Animation.cpo",
        "sha256:5dd89bea8dba216a9c8f4f61cc36a77675beb521273327fcb7da19255def808c",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff9_w3131c7_Animation.cpo",
        "sha256:684d1bf1672cc8fdd6a7e79def602ab160997b855b7323e9021c97487cae41b7",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff9_w33187_Animation.cpo",
        "sha256:8fe79617bfac0dde3e449c481b3a6a1b226e23f6c94ef9adee914bb7ff1b50f8",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff9b_w312187_Animation.cpo",
        "sha256:647f58b4632bfd90d0436ae726391918be46c125ff35c91dfcc5b100ade53cf4",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff9b_w312987_Animation.cpo",
        "sha256:749b9498a4fc682caf91fcf1c05c8ea5d7d6c74ad6ee5b7fda38f33c8e584ab2",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5010ff9b_w3129a7_Animation.cpo",
        "sha256:3d2fb83d2f75553c9407d8179e0b7b0590ce5a641762d23a18b160c01a0dcd88",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5030f_w31a187_Animation.cpo",
        "sha256:a1b5889f83e26d69efa2c121eafa7886cdbeac25f51b074dc308640aade50a8e",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5030fb_w31a187_Animation.cpo",
        "sha256:4c97a057e26c575bb164172d08b9024a8844320137ea48b647752cf1c38fa3ad",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5030fdmo_w31a187_Animation.cpo",
        "sha256:0f9d9578e362bf3e811f6a88f790c29359cae543876fd527094ecedf03ef1336",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5030ff8_w31a187_Animation.cpo",
        "sha256:4c833c0d1c6926c80d6a847f26a50b800504845391a0dd4648fb7f91445c1309",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_5030ff8b_w31a187_Animation.cpo",
        "sha256:8cd72a51f780202df2e96b0bc68a3f8746369cc9b32a9861d33a1da45a2dd65a",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_50f1f_w3636187_Animation.cpo",
        "sha256:6f93e9f1bd56b3b16cbbdf30a80f3d266a7b5201fb748e761060813aeff40e98",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_50f1f_w36369a7_Animation.cpo",
        "sha256:d469c7bb9bd9c4faaf1f21b77e8c0c1dae4a97d6378b408113c8505da41b4a4a",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_50f1f_w386087_Animation.cpo",
        "sha256:2ec059ecac2c5b6dc64cb957a92393cd37980ca886fb651eeb86fd843c6f17ee",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "SpeedTree_50f1f_w3ef8087_Animation.cpo",
        "sha256:5a7c725064f897e4792a8a42e8ee6ec0dcbe209aa4b75b64719d53045745fa51",
        WARP_QUANTIZED_FLOAT_BITS,
    ),
    (
        "1021.dxbc",
        "sha256:1e5fef301327669be749764f0463efea892e79ae3ec98255869a4ae67d62fcf1",
        UNORDERED_UNPROVEN,
    ),
    (
        "1022.dxbc",
        "sha256:1721d8b909772e2a80aa81c279583b8afcae48a1a6b0630f945b18e24fda13d0",
        WARP_UNSTABLE,
    ),
    (
        "1023.dxbc",
        "sha256:e4b8c5e4982aacd1238821a934a7b1188bc6ccbf72d9060b08b6dc18abc38724",
        WARP_UNSTABLE,
    ),
    (
        "1024.dxbc",
        "sha256:3958f23ecb90f75b90dc658f04cdd742923372885f476a9830554ecde96afe0c",
        WARP_UNSTABLE,
    ),
    (
        "1026.dxbc",
        "sha256:62d4aca061e9dbfc9aec3a706eb8822c8b5e83c125ae89cc7bed903c1d093a75",
        UNORDERED_UNPROVEN,
    ),
    (
        "1029.dxbc",
        "sha256:0a84f8bb618611e84a54b3e15c20fdb34f1d992b84e1754a81d4a6d08595d336",
        WARP_SIN_INFERRED,
    ),
    (
        "BitonicSortSinglePass512.cpo",
        "sha256:752ec5b2a57ebc63792422a7f84c7fd1f5890161d4c742f2d5bb278469510157",
        WARP_UNSTABLE,
    ),
    (
        "MatrixTranspose.cpo",
        "sha256:bd1e5cfcc36c067d643f98e6dd7aafd90750a5630ea5c0255fd1791dd616859f",
        SOURCE_ORDER,
    ),
    (
        "SpeedTree_5010ff8_w3129a7_Animation.cpo",
        "sha256:92453537c491729ea6fc2d31e295ea7daf1d3718947c410f71e19b7e6eefe6ac",
        WARP_FLOAT_BITS,
    ),
    (
        "SpeedTree_50777_w32082_Animation.cpo",
        "sha256:4f6accfeb935a49d4831fc9d58502b5ed9c14b7cea158920f247b921506ebf79",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777_w3636187_Animation.cpo",
        "sha256:bb9de44c08b9516581a9c2301798666daa89743675df5c3f7358d362944fe928",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777_w36361a7_Animation.cpo",
        "sha256:91d98722f42a100e78310813bfa2b0af7145383d6aee4f7859af67ef39f58108",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777_w36369a7_Animation.cpo",
        "sha256:6fd7d8b8edd9260af0d72e2bec628c698055444e2dec268caa6246d80cafab10",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777_w36b6187_Animation.cpo",
        "sha256:2997ea2b7e16a23eb7a7760c422592bd72d013db5b82b8a07074242454629416",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777_w36b68a6_Animation.cpo",
        "sha256:d154876e4bcf0eb3b44e9809ec63e1a5c66da33c0e807c97a0024680815a1ce1",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777_w3eb4087_Animation.cpo",
        "sha256:a1b8f61d332006a43c4d99eafd22c464fdd31a43d66f855359e4ab1f10aba001",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777_w3f7a187_Animation.cpo",
        "sha256:37ff1403b952be37bbf9c805d2c62aa83aec6333c8a24139782afb0e4d0670ae",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777f4_w3636187_Animation.cpo",
        "sha256:1f8fb6cbb672fe997a2ce848ed431c968201f62d58803a13ba781e389a3475ee",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777f4b_w32082_Animation.cpo",
        "sha256:b28bf3a81d57ef9a79f94893c408f127c8a43ff2bd19cdbfdbb8c1a72e68f258",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777f8_w3636187_Animation.cpo",
        "sha256:46cbfb7d22f8db85715483eadd7a954b41fccb926c367b3156a27046fb9a5317",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777f9_w3636187_Animation.cpo",
        "sha256:956a5a2adef636fec68aefe945e445281d1a988bfea31d5738f26da071515576",
        WARP_SIN,
    ),
    (
        "SpeedTree_50777f9_w3f7a187_Animation.cpo",
        "sha256:34c70278b5c976a55ebe2f771a4c6ff8215eac0bdc5a9d4617c4bdb3a95b4ae9",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w32187_Animation.cpo",
        "sha256:9d0798277dbcfa46b815d940585d51079b202c28efd67796cd72b926470cb34e",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w34002082_Animation.cpo",
        "sha256:cdad01c17a303437a3560214c70f90bd50238c46f8044ed59c664b25653414e9",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w340028a2_Animation.cpo",
        "sha256:4853a83bf373cca034e876938f36ecf7f7ff593f792e1ef410b2674f66972fc3",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w34636187_Animation.cpo",
        "sha256:c357b79e6a9e805a9b1524369b7e5100ef23732a42054f12107383ac98afcd66",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w34636987_Animation.cpo",
        "sha256:4bb38d2e6d835966a066f406e6048a2efac95e25a24ff7bbf162db406f8aef7b",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w346369a7_Animation.cpo",
        "sha256:8590b30c3d8dd38b6e05704d7966f689818928ea744d43843a288fecf4f1a79e",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w346b6187_Animation.cpo",
        "sha256:0f3c34ddf399689ed4bd18b6096158066782dd49770dafbdf3eea25cc4711f58",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w346b68a6_Animation.cpo",
        "sha256:030dd9fabadcfe75697b9ddd23f0a7e3af3996b516ff594cdd3412559a02c627",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w36002082_Animation.cpo",
        "sha256:947fc8513636779bc3af07392ddbb426756b70039f7b4b7b6c5b117868c581ee",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w3636187_Animation.cpo",
        "sha256:ebe6a690d05911590fe114fa4b440f4f0fac811059cf7c747d171fd05bb926a0",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w36361a7_Animation.cpo",
        "sha256:d0ebd1ee251f2818396fcba06623b8d0021883e88a980c7460ed8466143183bf",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077f_w36f7a187_Animation.cpo",
        "sha256:adab6720c2cfc15e8f362819a0ce3d17d6b518fb195aeeff66dfa105f1b017d2",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077ff8_w34636187_Animation.cpo",
        "sha256:43b73cb1b0046280653352a087298f9faf1eab771c2e5910e8224524b7ed8f1d",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077ff8_w346369a7_Animation.cpo",
        "sha256:90060fbf258df9947471be4393309ce554907966fe7aaf650c34a6711d5909a2",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077ff8_w346b68a6_Animation.cpo",
        "sha256:104bf4bf3933f793141da24e50ed4cb88fd575d158621fe777c5cd5999a7e13a",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077ff8b_w34002082_Animation.cpo",
        "sha256:77f6dd689a5dedeec7a4e7f87d5f1cdab333ac7b2462970af1fa163ad1ea3d59",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077ff8b_w34636187_Animation.cpo",
        "sha256:e03c18718c243ed68eb7c4765398e291ce19a91db0745522653ffd5a8608cc27",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077ff8b_w36002082_Animation.cpo",
        "sha256:dd7406734514af456914b6c390902c3b22a91251394fbea5a7e68c72098bf89d",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077ff9_w34636187_Animation.cpo",
        "sha256:02a4b69ca5b962a1fabd97b4d041d71d97ff0a28dcc3d4c96acec31916e47aeb",
        WARP_SIN,
    ),
    (
        "SpeedTree_5077ff9_w346369a7_Animation.cpo",
        "sha256:e34165771e4a67fa7d2ebcb308623d188b052a1c2dc1922eaf8c3a7044495cce",
        WARP_SIN,
    ),
];

/// The reason `name` (of source hash `hash`) may differ from WARP, if [`COMPUTE_EXPECTED`] lists it.
fn compute_expected(name: &str, hash: &str) -> Option<&'static str> {
    COMPUTE_EXPECTED
        .iter()
        .find(|(n, h, _)| *n == short(name) && *h == hash)
        .map(|(_, _, why)| *why)
}

/// Vertex and pixel programs whose lanes are known to be further than 1e-3 from WARP's with the executor and the writers
/// right: each by program name and source hash, with why. Its beyond lanes are counted and the program printed, and the
/// test does not fail on them; a program listed here that agrees in every lane fails the test so the table is pruned.
const GRAPHICS_EXPECTED: &[(&str, &str, &str)] = &[
    (
        "03688.dxbc",
        "sha256:41e1293f60bffa879a90426022d755da06ff4dd2fe8321afd5418d5d37ecb28b",
        CELL_NOISE_NATIVE,
    ),
    (
        "03689.dxbc",
        "sha256:e1b2b7878ccdea68c222b258aa14a8bb0d1984f40ddb5f576876e6a37f14c198",
        CELL_NOISE_NATIVE,
    ),
    (
        "04137.dxbc",
        "sha256:d7347a69c616f1d6ea5f7f7901a3d47d1596a72d169f688d1f98a6e0544b55bc",
        CELL_NOISE,
    ),
    (
        "04138.dxbc",
        "sha256:0878266da43fa7b43bb0b3d884b6dc39b9b2bdb5aac05b07300ad7d833d8bf5a",
        CELL_NOISE,
    ),
    (
        "04139.dxbc",
        "sha256:c8c9372b026306c99e604ddf64908875bf83e9a5b2b6100e0b9b7dd951364609",
        CELL_NOISE,
    ),
    (
        "04140.dxbc",
        "sha256:3f3a7418ac50d1a209a7bd18b515ea6148ffbc0979ff78bac6e792da0ed8107c",
        CELL_NOISE,
    ),
    (
        "04143.dxbc",
        "sha256:9c1ad19cd3dc8767b680d0a3e3febd6bf26c784538eb3ce73f450222012e878c",
        CELL_NOISE,
    ),
    (
        "04144.dxbc",
        "sha256:72f01bea1e779d02b5fd65b25f1a91e89ee7202079771760473dc81c4a592e16",
        CELL_NOISE,
    ),
    (
        "04145.dxbc",
        "sha256:f9e79078bcd6db62af1f04bd5b1d67b2b265b649896d8ec0b5328f323ec28c0c",
        CELL_NOISE,
    ),
    (
        "04146.dxbc",
        "sha256:4316761fa8a5352902dbf6ea55a49fa97c98eb5e41ce9f4ec51dfab5729f588d",
        CELL_NOISE,
    ),
    (
        "04147.dxbc",
        "sha256:2405df6b6eada3e01d100b47457d791c455ed0a4bb75269f6553ff315ee8fa5d",
        CELL_NOISE,
    ),
    (
        "04148.dxbc",
        "sha256:8f75def7f915427ca59229eb0e4baebb8170e435ba4222efbbf7a49771a2cf20",
        CELL_NOISE,
    ),
    (
        "04149.dxbc",
        "sha256:4e19324c029ad2404cef72c1ab2cce64992c6df8e927eb0eefa5c64bc81357e6",
        CELL_NOISE,
    ),
    (
        "04150.dxbc",
        "sha256:46eba4b4543479bf7926f277390565ca44c1ea7ebee1c316cabb0bc2e6b1adcb",
        CELL_NOISE,
    ),
    (
        "04151.dxbc",
        "sha256:e35d8815f3ba29be4cf11043aff4a706ff886cd59c17582b505f6e9bc06243cb",
        CELL_NOISE,
    ),
    (
        "04152.dxbc",
        "sha256:f62605d111d605eb79449141b91b5b6171b8f850548c28df30e73316af732166",
        CELL_NOISE,
    ),
    (
        "04153.dxbc",
        "sha256:cc8487e764e5fd228e4ac16591b3128a22b9f7f4377065857ab1494a5c02a7f3",
        CELL_NOISE,
    ),
    (
        "04154.dxbc",
        "sha256:b8a8bc33a84f2986d3395565199818db3025ec1d1d82ef5c6275df1b9000d2ec",
        CELL_NOISE,
    ),
    (
        "04155.dxbc",
        "sha256:34c24220b460c8cd3601fde6697cc7f912abc86aa5661d4819e26dffc8733e7b",
        CELL_NOISE,
    ),
    (
        "04156.dxbc",
        "sha256:5372a6d99ecd26f35a90a74a7036185b7be92b0f12d0521f3ae0089a99466fb2",
        CELL_NOISE,
    ),
    (
        "04161.dxbc",
        "sha256:da31221d9877ce55fa160842a5e331ba771761fa067098435a2488295cf0da72",
        CELL_NOISE,
    ),
    (
        "04162.dxbc",
        "sha256:da0d877c0bb9076a1536945728ad4027b6846d3cc349ba2f77ad8cbd4b952337",
        CELL_NOISE,
    ),
    (
        "04163.dxbc",
        "sha256:3163463e9dff393a3dada6090cc3bce491b53da7879f13aa2146ffdccc45e997",
        CELL_NOISE,
    ),
    (
        "04164.dxbc",
        "sha256:70195113297cf0f197f23bab70b8b450a4dd2fcf3675c8858611cf875d5b4acb",
        CELL_NOISE,
    ),
    (
        "04167.dxbc",
        "sha256:fd69b548f628ba234d9e0331d92cc85d737aecb8db98b0a3ba0135ce0b0d8180",
        CELL_NOISE,
    ),
    (
        "04168.dxbc",
        "sha256:ce9698c92b49034a77ed3a96a216eb8511e65ebb401e4391d6fa326de17d4081",
        CELL_NOISE,
    ),
    (
        "04169.dxbc",
        "sha256:62cd0ab29beac9d1040d30ac06d88fc1f00fe61dfab2d0a1aef73f53257954a0",
        CELL_NOISE,
    ),
    (
        "04170.dxbc",
        "sha256:3f255215235488e0c2c0d75199bec65e102f468bbab88476d8c08f449ad16406",
        CELL_NOISE,
    ),
    (
        "04171.dxbc",
        "sha256:3d5df3cfd5f035eaa12bc66cb412ec7477a4baae4d3e8bb0f2f48fb2d9eda0d0",
        CELL_NOISE,
    ),
    (
        "04172.dxbc",
        "sha256:5ef469fb526d73361a839a25ea24d7b385d65be748c3c05ab775b70639b4ea82",
        CELL_NOISE,
    ),
    (
        "04173.dxbc",
        "sha256:9c50e43ef9843ac1b91f3d0e49b656e23d4bc24c0181a4a0c459e26d03ca4eb2",
        CELL_NOISE,
    ),
    (
        "04174.dxbc",
        "sha256:9f7a19554c6dce81a308a16d797b1e81ddd153a1e260787c9ac62454d55ba18d",
        CELL_NOISE,
    ),
    (
        "04175.dxbc",
        "sha256:3955f5e7a020adf171dba97e043405e0c76a9ade69922ce9d3e9c3b83fda84dc",
        CELL_NOISE,
    ),
    (
        "04176.dxbc",
        "sha256:0f5d2ed7626bfe1b6e23e2651c06a0a4f52430043abe2afdaa3b70bff656c4fe",
        CELL_NOISE,
    ),
    (
        "04177.dxbc",
        "sha256:3fe7ebad8fdf47184e6c08466e0d7730c5d281f0163a1c4df845cc1177a0b667",
        CELL_NOISE,
    ),
    (
        "04178.dxbc",
        "sha256:195e496e69a36b95afa37b6201a46d446ec87d8dcd722a5b977dcb593439d423",
        CELL_NOISE,
    ),
    (
        "04179.dxbc",
        "sha256:07c3752683f9e3dc03328b04b73b58284630f99b1d5db81d77f7c2e900971f98",
        CELL_NOISE,
    ),
    (
        "04180.dxbc",
        "sha256:24feddc456810dd7f1742af583a045a9c8d53c0970f214258a6775dc61e82e67",
        CELL_NOISE,
    ),
    (
        "04183.dxbc",
        "sha256:456403b6bbdebe3c7031de76afdde11ef7f335a93831d34a407cb051de13d545",
        CELL_NOISE,
    ),
    (
        "04211.dxbc",
        "sha256:eb055b74556e8b5c5228ff3b85d6c80d2ef05f5b54a907bd5ff79e991ee2d468",
        CELL_NOISE,
    ),
];

/// The reason `name` (of source hash `hash`) may differ from WARP, if [`GRAPHICS_EXPECTED`] lists it.
fn graphics_expected(name: &str, hash: &str) -> Option<&'static str> {
    GRAPHICS_EXPECTED
        .iter()
        .find(|(n, h, _)| *n == short(name) && *h == hash)
        .map(|(_, _, why)| *why)
}

/// Cell noise as [`CELL_NOISE`] says, but `shader.sin=reference` leaves one lane (case 2 lane 0 o0.2: 0.0011223652 against
/// 0.0011211843, 1.2e-6 apart on a value of 1.1e-3), within one ulp of `sin` times 43758 (about 2.6e-3 on a hash), so the cause
/// is proven by another run: WARP running the program's own fxc bytecode gives the WGSL's values, not the executor's (the
/// header's account of 03688, `exec_warp`).
const CELL_NOISE_NATIVE: &str = "cell noise, WARP's sin past D3D's bound: proven by WARP running the program's own fxc bytecode (it gives the WGSL's values); one lane of 1.2e-6 stays under shader.sin=reference, within one ulp of sin times 43758";

/// Cell noise, `fract(sin(dot(cell, k)) * 43758.547)` hashes of integer cells, and what is computed from them: the sin
/// of a cell's argument is beyond D3D's -100 pi..100 pi bound for most cells (WGSL bounds it only within -pi..pi), where
/// WARP's is not the reference's (tools/wgsl-run/probes/sin_cells.py: for Unity program 04211's cells -3..3 and its two
/// constants, WARP's differs from the f64 reference by up to 5.9e-5 and the hash by more than 1e-3 for 95 of 98
/// arguments), and the cell's feature point moves with its hash. Each program listed with it was run with `shader.sin=reference`
/// (WARP's builtin `sin` swapped for one within 8.5e-8 of the reference) and then has no lane beyond 1e-3: proven by sin
/// substitution, not inferred.
const CELL_NOISE: &str = "cell noise, WARP's sin past D3D's bound: no lane beyond 1e-3 with shader.sin=reference (proven by sin substitution)";

/// `fract(abs(sin(x) * 100000))` hashes: WARP's `sin` is within D3D's absolute bound (0.0008 in -100 pi to 100 pi, none
/// beyond: Microsoft Learn's `sincos (sm4 - asm)`) or outside its range, and the shader scales the difference by 1e5; the
/// executor's `sin` is the f64 reference rounded (tools/wgsl-run/probes/sin_args.py measures it: 12 of its 16 arguments are inside the range).
const WARP_SIN: &str =
    "WARP's sin is within D3D's absolute bound or outside its range, scaled x1e5 by fract(abs(sin(x) * 1e5))";

/// Two invocations store different words to one word, so the result depends on an order the executor and WARP do not share.
const SOURCE_ORDER: &str =
    "invocations store different words to one word: the result is order-dependent in the source";

/// Float computations become integer words through bitcasts and permitted inexact conversions.
// D3D11.3 functional specification, 22.13.6 (normative; the shorter Learn page omits this latitude):
// https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm
const WARP_FLOAT_BITS: &str = "integer words built from float bits; upstream 1-3 ulp differences and u32->f32 rounding, \
where D3D11.3 22.13.6 recommends but does not require nearest-even (executor: nearest-even; WARP: toward zero); \
proven by lane capture 567964d0";

/// Every differing word of these name/hash rows was captured and mechanically checked (192 vectors, 83 words).
/// https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#22.13.6%20utof
const WARP_QUANTIZED_FLOAT_BITS: &str = "an upstream fadd difference within the float bound, quantized to up to \
256-unit steps by bitcast->utof, with WARP's utof faithfully rounded (D3D11.3 22.13.6); each final saturate \
conversion exactly accounts for the stored integer; proven per word by capture 81015009";

/// The original output words stay unchanged when the overflowing expression's operands/argument are captured.
/// https://www.w3.org/TR/2026/CRD-WGSL-20261106/#floating-point-evaluation
/// https://www.w3.org/TR/2026/CRD-WGSL-20261106/#indeterminate-values
const WARP_SEEDED_NONFINITE: &str = "exp2 overflows on our seeded inputs (dot 7.85e18), then Inf*0; \
WGSL 15.7.2 (Finite Math Assumption) and 8.2 leave non-finite intermediates indeterminate; captures \
e5ca67ac/0b022ba6/41499eda cover all 64 differing words with unchanged original outputs; the huge dot \
comes from our seeds, so seeding by use may retire this row (remove it when the stale check fires)";

/// WARP gives different words on different runs of the same job.
const WARP_UNSTABLE: &str = "WARP gives different words on different runs of the same job (intermittent)";

/// A word two stores gave different values, and the program differs from WARP there; WARP's own variation between runs was
/// not seen in the run that listed it.
const UNORDERED_UNPROVEN: &str =
    "two stores give one word different values and it differs from WARP there; WARP's own variation not seen";

/// Differences of a few thousandths in an FFT's cancelled lanes, in a program with sin and cos; cause inferred, not probed.
const WARP_SIN_INFERRED: &str =
    "inferred, not probed: twiddle factors from sin and cos, cancelled in the butterfly";

/// The vectors a compute program runs on: 0 and 2 seed the words no reflection names as floats, 1 as small integers, 3
/// the buffers' words as pairs of half floats from 0.25 to 2 (data a program unpacks).
const COMPUTE_CASES: u32 = 4;

/// How many times each compute job runs on WARP (`shader.compute_runs`, 2 by default, at least 2).
fn warp_runs() -> usize {
    inputs::debug("shader.compute_runs")
        .and_then(|s| s.parse().ok())
        .map_or(2, |n: usize| n.max(2))
}

/// The words of each buffer a compute program reads or writes: 16 K, more than any index the programs make of a
/// 2 x 1 x 1 dispatch reaches, so no access leaves the buffer (where the executor reads zeros and wgpu clamps).
const COMPUTE_WORDS: u32 = 16384;

/// The thread groups a compute program is dispatched over.
const COMPUTE_GROUPS: [u32; 3] = [2, 1, 1];

/// The words of compute buffer `id` in vector `c`: floats from -0.25 to 1.75 for vectors 0 and 2, integers 0 to 7 for 1
/// (an integer buffer's indices and counts stay inside the buffer, a float's bits would not).
fn compute_words(id: ResourceId, per: u32, c: u32) -> Vec<u32> {
    (0..COMPUTE_WORDS / per * per)
        .map(|w| {
            let h = common::mix(u64::from(c) << 44 ^ u64::from(id.0) << 32 ^ u64::from(w) ^ 0xc07e);
            match c {
                1 => h % 8,
                3 => {
                    // a half float of exponent -2 to 0 and a mantissa, twice
                    let half = |h: u32| (((h % 3) + 13) << 10) | ((h >> 8) & 1023);
                    half(h) | half(common::mix(u64::from(h) ^ 0x4a1f)) << 16
                }
                _ => (common::unit(h) * 2.0 - 0.25).to_bits(),
            }
        })
        .collect()
}

/// The rows of constant buffer `layout` in vector `c`, each word seeded as its reflection class says (bool, integer 0 to
/// 2, float from -0.25 to 1.75; a word no reflection names a float in vectors 0 and 2, an integer in 1).
fn compute_rows(id: ResourceId, layout: &iridia_ir::interface::BufferLayout, c: u32) -> Vec<[u32; 4]> {
    let mut classes = BTreeMap::new();
    for k in &layout.constants {
        common::classify(&k.ty, k.offset, &mut classes);
    }
    let unknown = if c == 1 { 2 } else { 3 };
    let rows = layout.size.div_ceil(16).max(1);
    (0..rows)
        .map(|r| {
            std::array::from_fn(|k| {
                let word = r * 4 + k as u32;
                let h = common::mix(u64::from(c) << 44 ^ u64::from(id.0) << 32 ^ u64::from(word) ^ 0x77);
                match classes.get(&word).copied().unwrap_or(unknown) {
                    1 => h & 1,
                    2 => h % 3,
                    _ => (common::unit(h) * 2.0 - 0.25).to_bits(),
                }
            })
        })
        .collect()
}

/// Why a compute program is not run here: its resources are not constant buffers and structured or raw buffers, or it
/// takes atomics, counters or wave operations.
fn compute_gap(p: &iridia_ir::Program) -> Option<String> {
    if let Some(why) = wgsl_run::gap_compute(p) {
        return Some(why);
    }
    for v in &p.entry.values {
        match v.op {
            iridia_ir::Op::Atomic { .. } | iridia_ir::Op::Counter(..) => {
                return Some("an atomic or a counter".into());
            }
            iridia_ir::Op::Wave(_) => return Some("a wave operation".into()),
            _ => {}
        }
    }
    None
}

/// How many instructions of a region store to each buffer, and whether it has a loop (inside which one instruction runs
/// more than once per invocation).
fn store_shape(r: &Region, stores: &mut BTreeMap<ResourceId, u32>) -> bool {
    let mut loops = false;
    for n in &r.nodes {
        match n {
            Node::StoreBuffer { resource, .. } => *stores.entry(resource.id).or_default() += 1,
            Node::Loop { .. } => loops = true,
            _ => {}
        }
        for inner in n.regions() {
            loops |= store_shape(inner, stores);
        }
    }
    loops
}

/// The elements a word buffer is divided in: a structured buffer's words an element.
fn words_per(kind: &ResourceKind) -> u32 {
    match kind {
        ResourceKind::Buffer(BufferKind::Structured { stride })
        | ResourceKind::Storage {
            target: StorageTarget::Buffer(BufferKind::Structured { stride }),
            ..
        } => (stride / 4).max(1),
        _ => 1,
    }
}

/// An executor backend that owns the written buffers (a compute program that asks for a sample or a load is not run), and
/// notes the words that stores gave two different values: the invocations run in an order the executor chooses and WARP
/// does not, so a word they disagree on is not the program's to say.
#[derive(Default)]
struct Memory {
    aliases: BTreeMap<ResourceId, word_types::StoreOrigin>,
    origins: BTreeMap<(ResourceId, u32), word_types::WordType>,
    words: BTreeMap<ResourceId, Vec<u32>>,
    /// Per word stored to: the first value, whether a later store gave another, and how many stores it had.
    stored: BTreeMap<(ResourceId, u32), (u32, bool, u32)>,
}
impl Backend for Memory {
    fn sample(&mut self, _: ImageSample<Element>) -> Result<[f32; 4], String> {
        Err("a texture sample".into())
    }
    fn store_buffer(&mut self, view: ResourceId, word: u32, value: [u32; 4], mask: u8) -> Result<(), String> {
        let (view, types) = self
            .aliases
            .get(&view)
            .map_or((view, [word_types::WordType::Unknown; 4]), |site| {
                (site.buffer, site.lanes)
            });
        let words = self.words.get_mut(&view).ok_or("a store to a buffer not given")?;
        for lane in (0..4u32).filter(|l| mask >> l & 1 != 0) {
            let at = word + lane;
            let v = value[lane as usize];
            // a store outside the buffer changes nothing
            let Some(slot) = words.get_mut(at as usize) else {
                continue;
            };
            *slot = v;
            let origin = types[lane as usize];
            self.origins
                .entry((view, at))
                .and_modify(|old| *old = old.merge(origin))
                .or_insert(origin);
            let e = self.stored.entry((view, at)).or_insert((v, false, 0));
            e.1 |= e.0 != v;
            e.2 += 1;
        }
        Ok(())
    }
    fn read_buffer(&mut self, view: ResourceId, word: u32, count: u8) -> Result<[u32; 4], String> {
        let view = self.aliases.get(&view).map_or(view, |site| site.buffer);
        let words = self.words.get(&view).ok_or("a read of a buffer not given")?;
        Ok(std::array::from_fn(|k| {
            if k < usize::from(count) {
                words.get(word as usize + k).copied().unwrap_or(0)
            } else {
                0
            }
        }))
    }
    fn buffer_words(&mut self, view: ResourceId) -> Result<u32, String> {
        let view = self.aliases.get(&view).map_or(view, |site| site.buffer);
        let words = self.words.get(&view).ok_or("a buffer not given")?;
        u32::try_from(words.len()).map_err(|_| "a buffer of too many words".into())
    }
}

/// The executed store's value origin decides the comparison: integers and unknown/mixed origins are exact.
fn compute_class(view: word_types::WordType, a: u32, b: u32) -> Class {
    wgsl_run::classify(view == word_types::WordType::Float32, a, b)
}

/// A compute program (the executor's `dispatch`) against the harness: every written buffer, every vector, word for word.
fn run_compute(harness: &Harness, name: &str, hash: &str, p: iridia_ir::Program, t: &mut Tally) {
    let excused = compute_expected(name, hash);
    *t.stages.entry("Compute".to_owned()).or_default() += 1;
    if excused == Some(WARP_QUANTIZED_FLOAT_BITS)
        && let Err(why) = quantized_shape::check(&p.entry)
    {
        tally(&mut t.errors, why, name);
        return;
    }
    if let Some(why) = compute_gap(&p) {
        tally(&mut t.gaps, &why, name);
        return;
    }
    let wgsl = match write(&p) {
        Ok(w) => w,
        Err(e) => {
            tally(&mut t.refused, &e, name);
            return;
        }
    };
    let bindings = match iridia_naga::BindingPlan::d3d(&p.interface) {
        Ok(plan) => plan.bindings,
        Err(e) => {
            tally(&mut t.errors, &format!("the binding plan: {e}"), name);
            return;
        }
    };
    let at = |id: ResourceId| {
        bindings
            .iter()
            .find(|b| b.resource == id)
            .map(|b| (b.group, b.binding))
    };
    let exe = match Executable::from_ir(&p) {
        Ok(e) => e,
        Err(e) => {
            tally(&mut t.unrun, &format!("B: {e}"), name);
            return;
        }
    };
    let (observed_program, aliases) =
        word_types::instrument(&p).expect("store-origin observation must preserve valid IR");
    let observer = Executable::from_ir(&observed_program).expect("store-origin observation must compile");
    if let Some(dir) = inputs::debug("shader.dump") {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(
            std::path::Path::new(&dir).join(format!("{}.ir", short(name))),
            format!(
                "{}\nStore origins:\n{aliases:#?}",
                iridia_ir::text::print(&p.entry)
            ),
        );
        let _ = std::fs::write(
            std::path::Path::new(&dir).join(format!("{}.wgsl", short(name))),
            &wgsl,
        );
    }
    let show_lanes = inputs::debug("shader.lanes").is_some_and(|s| name.contains(&s));
    let mut static_stores = BTreeMap::new();
    let loops = match &p.entry.body {
        Body::Regions(body) => store_shape(body, &mut static_stores),
        _ => true,
    };
    let mut worst = Class::Exact;
    let mut previous_worst = Class::Exact;
    let mut counts: BTreeMap<Class, usize> = BTreeMap::new();
    let mut changed = false;
    for c in 0..COMPUTE_CASES {
        let mut resources = Resources::default();
        let mut written: BTreeMap<ResourceId, Vec<u32>> = BTreeMap::new();
        let mut job_resources = Vec::new();
        for r in &p.interface.resources {
            let Some((group, binding)) = at(r.id) else {
                continue;
            };
            let words = match &r.kind {
                ResourceKind::ConstantBuffer { layout } => {
                    let rows = compute_rows(r.id, layout, c);
                    let words = rows.iter().flatten().copied().collect();
                    resources.constants.insert(r.id, rows);
                    words
                }
                ResourceKind::Buffer(_) => {
                    let words = compute_words(r.id, words_per(&r.kind), c);
                    resources.buffers.insert(r.id, words.clone());
                    words
                }
                ResourceKind::Storage { .. } => {
                    let words = compute_words(r.id, words_per(&r.kind), c);
                    written.insert(r.id, words.clone());
                    words
                }
                _ => continue,
            };
            if show_lanes {
                println!(
                    "{name} case {c} resource {} (group {group} binding {binding}) first words {:08x?}",
                    r.id.0,
                    &words[..words.len().min(16)]
                );
            }
            job_resources.push(
                serde_json::json!({ "group": group, "binding": binding, "buffer": { "words": words } }),
            );
        }
        let initial = written.clone();
        let mut memory = Memory {
            words: written,
            ..Memory::default()
        };
        let bound = match exe.bind_resources(&resources) {
            Ok(b) => b,
            Err(e) => {
                tally(&mut t.unrun, &format!("B: {e}"), name);
                return;
            }
        };
        if let Err(e) = exe.dispatch(&bound, &mut memory, &mut BTreeMap::new(), COMPUTE_GROUPS, 32) {
            tally(&mut t.unrun, &format!("B: {e}"), name);
            return;
        }
        let mut observed = Memory {
            words: initial.clone(),
            aliases: aliases.clone(),
            ..Memory::default()
        };
        let observed_bound = observer
            .bind_resources(&resources)
            .expect("observer resource bindings must match B");
        observer
            .dispatch(
                &observed_bound,
                &mut observed,
                &mut BTreeMap::new(),
                COMPUTE_GROUPS,
                32,
            )
            .expect("observer execution must match B");
        assert_eq!(
            memory.words, observed.words,
            "store-origin observation changed B's output: {name} vector {c}"
        );
        assert_eq!(
            memory.stored, observed.stored,
            "store-origin observation changed B's writes: {name} vector {c}"
        );
        t.origin_controlled_vectors += 1;
        for &origin in observed.origins.values() {
            *t.compute_word_origins
                .entry(name.to_owned())
                .or_default()
                .entry(origin)
                .or_default() += 1;
        }
        let written = &memory.words;
        let job = serde_json::json!({
            "wgsl": wgsl,
            "dispatch": COMPUTE_GROUPS,
            "resources": job_resources,
        });
        // the job more than once: a word WARP gives differently on a later run is one the program's own writes leave
        // unordered
        let mut runs = Vec::new();
        for _ in 0..warp_runs() {
            let reply = match harness.run(&job) {
                Ok(r) => r,
                Err(e) => {
                    tally(&mut t.errors, &format!("the harness: {e}"), name);
                    return;
                }
            };
            let words: BTreeMap<(u32, u32), Vec<u32>> = reply["resources"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|r| {
                            let words = r["words"]
                                .as_array()?
                                .iter()
                                .map(|w| w.as_u64().and_then(|w| u32::try_from(w).ok()))
                                .collect::<Option<Vec<u32>>>()?;
                            Some((
                                (r["group"].as_u64()? as u32, r["binding"].as_u64()? as u32),
                                words,
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default();
            runs.push(words);
        }
        let theirs = &runs[0];
        t.runs += 1;
        for (id, ours) in written {
            let Some((group, binding)) = at(*id) else { continue };
            let Some(warp) = theirs.get(&(group, binding)) else {
                tally(&mut t.errors, "the reply lacks a written buffer", name);
                return;
            };
            if warp.len() != ours.len() {
                tally(
                    &mut t.errors,
                    &format!(
                        "a written buffer of {} words in the reply, {} given",
                        warp.len(),
                        ours.len()
                    ),
                    name,
                );
                return;
            }
            changed |= ours != &initial[id] || warp != &initial[id];
            for (k, (&o, &w)) in ours.iter().zip(warp).enumerate() {
                let view = observed
                    .origins
                    .get(&(*id, k as u32))
                    .copied()
                    .unwrap_or(word_types::WordType::Unknown);
                let class = compute_class(view, o, w);
                // Shadow the retired heuristic only to count the policy change; it never chooses the verdict.
                let previous = wgsl_run::classify(o >= 0x0080_0000 || w >= 0x0080_0000, o, w);
                // a word two stores gave different values, or WARP gave differently on its second run
                let stores = memory.stored.get(&(*id, k as u32));
                let second = runs[1..]
                    .iter()
                    .filter_map(|r| r.get(&(group, binding)).and_then(|a| a.get(k)).copied())
                    .find(|&x| x != w);
                let why = match (stores, second) {
                    (_, Some(second)) => Some((
                        0,
                        format!(
                            "buffer {} word {k}: WARP gives {w:#x} on one run of the job and {second:#x} on another",
                            id.0
                        ),
                    )),
                    // each store instruction runs once per invocation outside a loop, so more stores than instructions
                    // to the buffer came from more than one invocation
                    (Some(&(first, true, n)), _)
                        if !loops && n > static_stores.get(id).copied().unwrap_or(0) =>
                    {
                        Some((
                            1,
                            format!(
                                "buffer {} word {k}: {n} stores from at least two invocations ({} store instructions to the buffer, no loop), the first {first:#x}, the last {o:#x}",
                                id.0,
                                static_stores.get(id).copied().unwrap_or(0)
                            ),
                        ))
                    }
                    (Some(&(first, true, n)), _) => Some((
                        2,
                        format!(
                            "buffer {} word {k}: {n} stores of different words, the first {first:#x}, the last {o:#x} (not shown to come from two invocations: {} store instructions to the buffer{})",
                            id.0,
                            static_stores.get(id).copied().unwrap_or(0),
                            if loops { ", the program has a loop" } else { "" }
                        ),
                    )),
                    _ => None,
                };
                if class != Class::Exact
                    && let Some((kind, why)) = why
                {
                    t.unordered.entry(name.to_owned()).or_default()[kind] += 1;
                    // the strongest reason a program has is the one it shows
                    let shown = t
                        .unordered_why
                        .entry(name.to_owned())
                        .or_insert((kind, why.clone()));
                    if kind < shown.0 {
                        *shown = (kind, why);
                    }
                    continue;
                }
                if previous != class {
                    *t.compute_class_changes
                        .entry(name.to_owned())
                        .or_default()
                        .entry(format!("{previous:?} -> {class:?}"))
                        .or_default() += 1;
                }
                previous_worst = previous_worst.max(previous);
                *t.lanes.entry(class).or_default() += 1;
                *counts.entry(class).or_default() += 1;
                worst = worst.max(class);
                if class != Class::Exact && show_lanes {
                    println!(
                        "{name} case {c} u{} word {k}: {class:?} B {o:#010x} ({}) WARP {w:#010x} ({})",
                        id.0,
                        f32::from_bits(o),
                        f32::from_bits(w)
                    );
                }
                if class == Class::Beyond && t.worst.len() < SHOWN && excused.is_none() {
                    t.worst.push(format!(
                        "{name} case {c} buffer {} word {k}: B {o:#010x} ({}) WARP {w:#010x} ({})",
                        id.0,
                        f32::from_bits(o),
                        f32::from_bits(w)
                    ));
                }
            }
        }
    }
    if previous_worst != worst {
        t.compute_program_class_changes
            .insert(name.to_owned(), (previous_worst, worst));
    }
    if worst == Class::Beyond || t.unordered.contains_key(name) {
        t.differs.insert(name.to_owned(), (hash.to_owned(), excused));
    } else if let Some(why) = excused
        && [
            WARP_SIN,
            SOURCE_ORDER,
            WARP_FLOAT_BITS,
            WARP_QUANTIZED_FLOAT_BITS,
            WARP_SEEDED_NONFINITE,
        ]
        .contains(&why)
    {
        // an intermittent excuse is exact on some runs; a deterministic one that is exact in every vector is stale
        tally(
            &mut t.errors,
            "a program COMPUTE_EXPECTED excuses agrees with WARP in every vector: remove it from the table",
            name,
        );
    }
    if !changed && worst != Class::Beyond {
        tally(
            &mut t.all_discarded,
            "no word of a written buffer changed on either side: nothing to compare",
            name,
        );
        return;
    }
    t.compared += 1;
    t.by_program.insert(name.to_owned(), worst);
    t.program_lanes.insert(name.to_owned(), counts);
}

fn report(title: &str, t: &Tally) {
    for (name, (previous, current)) in &t.compute_program_class_changes {
        println!("compute-program-class-change {name}: {previous:?} -> {current:?}");
    }
    for (name, changes) in &t.compute_class_changes {
        println!(
            "compute-class-change {name}: {} words {changes:?}",
            changes.values().sum::<usize>()
        );
    }
    if t.origin_controlled_vectors != 0 {
        println!(
            "store-origin control: {} vectors have identical original and observed B outputs and write histories",
            t.origin_controlled_vectors
        );
    }
    for (name, origins) in &t.compute_word_origins {
        if let Some(unknown) = origins.get(&word_types::WordType::Unknown) {
            println!(
                "compute-unknown-store-origin {name}: {unknown} actually written words across vectors; compared exactly"
            );
        }
    }
    // `shader.worst`: a file that takes each program's worst class, to set one writer's run against another's
    if let Some(path) = inputs::debug("shader.worst") {
        let lines: String = t
            .by_program
            .iter()
            .map(|(n, c)| {
                let lanes: Vec<String> = t.program_lanes.get(n).map_or_else(Vec::new, |m| {
                    m.iter()
                        .map(|(k, v)| format!("{}={v}", wgsl_run::CLASSES[*k as usize].trim()))
                        .collect()
                });
                format!(
                    "{}	{}	{}
",
                    short(n),
                    wgsl_run::CLASSES[*c as usize].trim(),
                    lanes.join(" ")
                )
            })
            .collect();
        let _ = std::fs::write(format!("{path}-{}.txt", title.replace(['/', '\\'], "-")), lines);
    }
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
        ("no run survives", &t.all_discarded),
        ("barrier not provably uniform", &t.unprovable),
        ("not for a browser", &t.browser_refused),
        ("ERROR", &t.errors),
    ] {
        for (why, (n, at)) in map {
            println!("  {title} {n:6}  {why}  (e.g. {at})");
        }
    }
    for (source, (lanes, programs)) in &t.amplified {
        println!(
            "  amplified rounding of a {}: {lanes} lanes in {} programs: {}",
            source.name(),
            programs.len(),
            programs.iter().map(String::as_str).collect::<Vec<_>>().join(" ")
        );
    }
    if !t.unordered.is_empty() {
        let sum = |k: usize| t.unordered.values().map(|w| w[k]).sum::<usize>();
        println!(
            "  words of {} compute programs differ from WARP where the program's own writes are unordered: {} that WARP gives differently on different runs of the job, {} that stores of more than one invocation gave different values, {} that two stores gave different values (not shown to be two invocations)",
            t.unordered.len(),
            sum(0),
            sum(1),
            sum(2)
        );
        for (n, (_, why)) in &t.unordered_why {
            let w = t.unordered[n];
            println!("    {} ({}/{}/{}): {why}", short(n), w[0], w[1], w[2]);
        }
    }
    for (n, (hash, why)) in &t.graphics_differs {
        println!(
            "  graphics program {} {hash}: {}",
            short(n),
            why.unwrap_or("NOT IN GRAPHICS_EXPECTED")
        );
    }
    for (n, (hash, why)) in &t.differs {
        println!(
            "  compute program {} {hash}: {}",
            short(n),
            why.unwrap_or("NOT IN COMPUTE_EXPECTED")
        );
    }
    let beyond: Vec<&str> = t
        .by_program
        .iter()
        .filter(|(_, c)| **c == Class::Beyond)
        .map(|(n, _)| short(n))
        .collect();
    if !beyond.is_empty() {
        println!("  programs beyond 1e-3: {}", beyond.join(" "));
    }
    for w in t.worst.iter().take(WORST_SHOWN) {
        println!("  BEYOND {w}");
    }
    if t.worst.len() > WORST_SHOWN {
        println!("  and {} more BEYOND lines", t.worst.len() - WORST_SHOWN);
    }
}

/// The vertex and pixel programs of `corpus` through the harness, as many at a time as `[run] threads` allows.
pub(crate) fn corpus_over(corpus: &common::Corpus) {
    let Some(harness) = wgsl_run::harness() else {
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
    // `shader.only`: `|`-separated parts of program names; only the programs with one of them in their name run
    let only: Vec<String> = inputs::debug("shader.only")
        .map(|s| s.split('|').map(str::to_owned).collect())
        .unwrap_or_default();
    let selected: Vec<String> = corpus
        .names
        .iter()
        .filter(|n| only.is_empty() || only.iter().any(|o| n.contains(o.as_str())))
        .cloned()
        .collect();
    let files = &selected[..selected.len().min(common::diff_limit())];
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
        for (k, n) in one.tinted {
            *t.tinted.entry(k).or_default() += n;
        }
        for (k, n) in one.stages {
            *t.stages.entry(k).or_default() += n;
        }
        for (k, n) in one.lanes {
            *t.lanes.entry(k).or_default() += n;
        }
        t.unordered.extend(one.unordered);
        t.differs.extend(one.differs);
        t.graphics_differs.extend(one.graphics_differs);
        t.compute_class_changes.extend(one.compute_class_changes);
        t.compute_program_class_changes
            .extend(one.compute_program_class_changes);
        t.compute_word_origins.extend(one.compute_word_origins);
        t.origin_controlled_vectors += one.origin_controlled_vectors;
        t.unordered_why.extend(one.unordered_why);
        for (source, (lanes, programs)) in one.amplified {
            let (n, all) = t.amplified.entry(source).or_default();
            *n += lanes;
            all.extend(programs);
        }
        for (mine, theirs) in [
            (&mut t.refused, one.refused),
            (&mut t.gaps, one.gaps),
            (&mut t.unrun, one.unrun),
            (&mut t.errors, one.errors),
            (&mut t.browser_refused, one.browser_refused),
            (&mut t.all_discarded, one.all_discarded),
            (&mut t.unprovable, one.unprovable),
        ] {
            for (why, (n, at)) in theirs {
                mine.entry(why).or_insert((0, at)).0 += n;
            }
        }
        t.by_program.extend(one.by_program);
        t.program_lanes.extend(one.program_lanes);
        t.worst.extend(one.worst);
    }
    report(&corpus.name, &t);
    for (stage, n) in &t.tinted {
        println!(
            "  {n:6} {stage} programs written for a browser, accepted by Tint unless an ERROR names them"
        );
    }
    assert!(t.errors.is_empty(), "{} kinds of harness errors", t.errors.len());
    assert!(
        t.worst.is_empty(),
        "{} lanes or runs differ from WARP",
        t.worst.len()
    );
}

pub(crate) fn corpus(game: inputs::Game) {
    if let Some(corpus) = common::corpus(game) {
        corpus_over(&corpus);
    }
}
