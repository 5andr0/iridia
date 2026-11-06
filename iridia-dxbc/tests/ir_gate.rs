#![allow(clippy::unwrap_used)]
//! The IR lifter against the lifter it replaces: every DXBC container of a corpus is lifted both ways, and each
//! vertex or pixel program both lift is run in the executor on the same deterministic inputs, constant buffers and
//! textures. Column A is the old lift, B the IR lift as lifted, P the IR lift after `promote`, `simplify` and
//! `break_flags`. Every written output lane, the masks and the discard must agree bit for bit (NaNs alike); a
//! program the IR lifter refuses is counted by reason, and a program the old lifter lifts but the IR lifter does not
//! is a lifted-count regression.
//! Needs the corpus oracle set's `sekiro/shaders`, `riders/shaders` or `sledding/shaders`. `IR_GATE_TRACE=1` names
//! each program as it starts.
mod inputs;

use std::collections::BTreeMap;

use inputs::Game;
use iridia::{BufferKind, ResourceId, ResourceKind, Stage, StorageTarget};
use iridia_exec::{
    Backend, BufferAtomic, Dim, Element, Executable, ImageAtomic, ImageLoad, ImageLod, ImageQuery,
    ImageSample, ImageStore, Load, PixelShader, ProgramShader, QueryKind, Resources, Sample, Sampler,
};

mod common;

/// The vectors each program is run on.
const CASES: u32 = 4;

/// Answers every request from a hash of everything it states, so two programs that ask the same agree and two that
/// ask differently (another coordinate, offset, level, component) almost surely do not.
#[derive(Default)]
struct Hashed {
    asked: Vec<String>,
}

fn hash_words(words: &[u32]) -> u64 {
    words
        .iter()
        .fold(0x1234_5678u64, |h, &w| common::mix64(h ^ u64::from(w)))
}

fn dim_word(d: Dim) -> u32 {
    match d {
        Dim::D1 => 1,
        Dim::D2 => 2,
        Dim::D3 => 3,
        Dim::Cube => 4,
    }
}

/// A float a request states, as the words it is hashed by: every NaN one pattern, since a sampler's answer does not
/// depend on a NaN's sign or payload, which IEEE 754 leaves open and two lifts compute differently.
fn bits(f: f32) -> u32 {
    canonical(f.to_bits())
}

fn texel_floats(h: u64) -> [f32; 4] {
    std::array::from_fn(|k| common::unit(common::mix(h ^ k as u64)) * 2.0 - 0.5)
}

impl Sampler for Hashed {
    fn sample(&mut self, r: Sample) -> Result<[f32; 4], String> {
        self.sample_image(ImageSample {
            texture: r.texture,
            sampler: r.sampler,
            dim: Dim::D2,
            coord: [r.uv[0], r.uv[1], 0.0],
            layer: r.layer,
            lod: ImageLod::from(r.lod),
            offset: [0; 3],
            gather: None,
            depth_ref: None,
            depth: false,
        })
    }
    fn sample_image(&mut self, r: ImageSample<u32>) -> Result<[f32; 4], String> {
        let mut w = vec![1, r.texture, r.sampler, dim_word(r.dim)];
        w.extend(r.coord.map(bits));
        w.push(r.layer.map_or(u32::MAX, bits));
        match r.lod {
            ImageLod::Implicit => w.push(10),
            ImageLod::Bias(b) => w.extend([11, bits(b)]),
            ImageLod::Level(l) => w.extend([12, bits(l)]),
            ImageLod::Gradients(g) => {
                w.push(13);
                w.extend(g.iter().flatten().copied().map(bits));
            }
            ImageLod::GradientsBias(g, b) => {
                w.extend([14, bits(b)]);
                w.extend(g.iter().flatten().copied().map(bits));
            }
        }
        w.extend(r.offset.map(i32::cast_unsigned));
        w.push(r.gather.map_or(u32::MAX, u32::from));
        w.push(r.depth_ref.map_or(u32::MAX, bits));
        w.push(u32::from(r.depth));
        self.asked.push(format!("{r:?}"));
        let mut t = texel_floats(hash_words(&w));
        if r.depth || r.depth_ref.is_some() {
            // a depth texture's one lane; the rest as a depth view reads them
            t = [t[0].clamp(0.0, 1.0), 0.0, 0.0, 1.0];
        }
        Ok(t)
    }
    fn load(&mut self, r: Load) -> Result<[u32; 4], String> {
        self.load_image(ImageLoad {
            texture: r.texture,
            dim: Dim::D2,
            storage: false,
            texel: r.texel,
            layer: None,
            sample: None,
            mip: Some(r.mip),
        })
    }
    fn load_image(&mut self, r: ImageLoad<u32>) -> Result<[u32; 4], String> {
        let mut w = vec![2, r.texture, dim_word(r.dim), u32::from(r.storage)];
        w.extend(r.texel.map(i32::cast_unsigned));
        w.push(r.layer.map_or(u32::MAX, i32::cast_unsigned));
        w.push(r.sample.map_or(u32::MAX, i32::cast_unsigned));
        w.push(r.mip.map_or(u32::MAX, i32::cast_unsigned));
        self.asked.push(format!("{r:?}"));
        Ok(texel_floats(hash_words(&w)).map(f32::to_bits))
    }
    fn query_image(&mut self, r: ImageQuery<u32>) -> Result<[u32; 4], String> {
        let w = [
            3,
            r.texture,
            dim_word(r.dim),
            r.kind as u32,
            r.level.unwrap_or(u32::MAX),
        ];
        self.asked.push(format!("{r:?}"));
        let h = common::mix(hash_words(&w));
        Ok(match r.kind {
            QueryKind::Size => {
                let shift = r.level.unwrap_or(0).min(8);
                [(256 >> shift).max(1), (128 >> shift).max(1), 4, 0]
            }
            QueryKind::Levels => [9, 0, 0, 0],
            QueryKind::Layers => [h % 6 + 1, 0, 0, 0],
            QueryKind::Samples => [4, 0, 0, 0],
        })
    }
    fn structured(&mut self, slot: u32, element: u32, offset: u32) -> Result<u32, String> {
        self.asked.push(format!("structured t{slot}[{element}]+{offset}"));
        Ok(common::mix(hash_words(&[4, slot, element, offset])) % 1024)
    }
    fn buffer_elements(&mut self, slot: u32) -> Option<u32> {
        let _ = slot;
        Some(64)
    }
}

/// What a run gives: each output register's written lanes (unwritten lanes zero), the masks and the discard; or
/// the error.
type Run = Result<Vec<Invocation>, String>;

/// One invocation's written output lanes (unwritten lanes zero), its masks and whether it was discarded; a discarded
/// invocation's outputs are not compared, since nothing it writes reaches a target.
type Invocation = (Vec<[u32; 4]>, Vec<u8>, bool);

fn run(shader: &ProgramShader, c: &[common::Case]) -> Run {
    run_asking(shader, c, &mut Hashed::default())
}

/// A run of one invocation, or of a quad's four (lanes 0 and 1 the top row) whose derivatives are the quad's own.
fn run_asking(shader: &ProgramShader, cases: &[common::Case], sampler: &mut Hashed) -> Run {
    let first = cases.first().ok_or("no test vector")?;
    let buffers = common::refs(first, shader);
    let executions = match cases {
        [one] => vec![shader.execute(&one.inputs, &buffers, sampler)],
        [a, b, c, d] => {
            match shader.execute_quad([&a.inputs, &b.inputs, &c.inputs, &d.inputs], &buffers, sampler) {
                Ok(q) => q.into_iter().map(Ok).collect(),
                Err(e) => vec![Err(e)],
            }
        }
        _ => {
            return Err(format!(
                "{} test vectors, where one or a quad's four are run",
                cases.len()
            ));
        }
    };
    let mut out = Vec::with_capacity(executions.len());
    for e in executions {
        let e = e.map_err(|e| e.to_string())?;
        let outputs = e
            .outputs
            .iter()
            .zip(&e.output_masks)
            .map(|(o, &m)| {
                std::array::from_fn(|k| {
                    if m >> k & 1 != 0 && !e.discarded {
                        canonical(o[k])
                    } else {
                        0
                    }
                })
            })
            .collect();
        out.push((outputs, e.output_masks, e.discarded));
    }
    Ok(out)
}

/// A NaN as one bit pattern: two lifts may produce different NaNs.
fn canonical(w: u32) -> u32 {
    if f32::from_bits(w).is_nan() {
        0x7fc0_0000
    } else {
        w
    }
}

/// The IR lift's program after the passes the cut-over runs.
fn passed(mut p: iridia_ir::Program) -> Result<iridia_ir::Program, String> {
    iridia_passes::promote(&mut p.entry)?;
    iridia_passes::simplify(&mut p.entry);
    iridia_passes::break_flags(&mut p.entry)?;
    iridia_ir::validate(&p).map_err(|e| format!("{e:?}"))?;
    Ok(p)
}

/// One program's comparison over `cases` vectors.
enum Verdict {
    /// Every case ran in A and gave what B gave.
    Agree,
    /// A case A itself cannot evaluate (its sampler request, a step limit): nothing to compare.
    AFails(String),
    /// The first case where B gives something else.
    Differs(String),
}

/// Test vector `c`: the first half seed the constants no reflection names as floats, the second as small integers
/// (a loop bound read from a stripped buffer). A program that takes derivatives of what it computes (`quad`) runs as a
/// quad of four vectors: each lane's own inputs, the constants, the facing of lane 0 and positions a pixel apart.
fn gate_case(a: &ProgramShader, c: u32, quad: bool) -> Vec<common::Case> {
    let unstated = if c < CASES / 2 {
        common::Unstated::Floats
    } else {
        common::Unstated::Ints
    };
    if !quad {
        return vec![common::case_with(a, c, unstated)];
    }
    let mut lanes: Vec<common::Case> = (0..4)
        .map(|k| common::case_with(a, c * 4 + k, unstated))
        .collect();
    let first = lanes[0].buffers.clone();
    let facing = a
        .interface()
        .inputs
        .iter()
        .filter(|v| v.system == Some(iridia::SystemValue::FrontFacing))
        .filter_map(|v| v.provenance.d3d.map(|d| d.register as usize))
        .collect::<Vec<_>>();
    let position = a
        .interface()
        .inputs
        .iter()
        .find(|v| v.system == Some(iridia::SystemValue::FragCoord))
        .and_then(|v| v.provenance.d3d.map(|d| d.register as usize));
    // an input the program does not read may lie past the registers the case seeds
    let seeded = lanes[0].inputs.len();
    let facing: Vec<usize> = facing.into_iter().filter(|&r| r < seeded).collect();
    let position = position.filter(|&r| r < seeded);
    let front: Vec<[u32; 4]> = facing.iter().map(|&r| lanes[0].inputs[r]).collect();
    for (k, lane) in lanes.iter_mut().enumerate() {
        lane.buffers = first.clone();
        for (&r, f) in facing.iter().zip(&front) {
            lane.inputs[r] = *f;
        }
        if let Some(r) = position {
            lane.inputs[r][0] = (0.5 + (k & 1) as f32).to_bits();
            lane.inputs[r][1] = (0.5 + (k >> 1) as f32).to_bits();
        }
    }
    lanes
}

/// A and B on every vector. A vector A cannot evaluate (a loop past the executor's budget on these inputs, a request
/// the sampler does not answer) is not compared, and B is not run on it; a program none of whose vectors A evaluates
/// is `AFails`.
fn compare(a: &ProgramShader, b: &ProgramShader, cases: u32) -> Verdict {
    let mut failed = None;
    let mut compared = 0;
    let quad = a.needs_quad() || b.needs_quad();
    for c in 0..cases {
        let case = gate_case(a, c, quad);
        let x = run(a, &case);
        if let Err(e) = &x {
            failed.get_or_insert_with(|| e.clone());
            continue;
        }
        let y = run(b, &case);
        if x != y {
            return Verdict::Differs(format!("case {c}: A {x:x?}\n  but {y:x?}"));
        }
        compared += 1;
    }
    match failed {
        Some(e) if compared == 0 => Verdict::AFails(e),
        _ => Verdict::Agree,
    }
}

#[derive(Default)]
struct Gate {
    programs: usize,
    /// Old-lift vertex and pixel programs the executor runs.
    a: usize,
    agree_b: usize,
    agree_p: usize,
    /// Programs the IR lifter refuses that A lifted, by reason, with the first.
    refused: BTreeMap<String, (usize, String)>,
    /// Programs B lifted the executor (or a pass) refuses, by reason.
    unrun: BTreeMap<String, (usize, String)>,
    /// Programs A cannot evaluate, by reason.
    a_fails: BTreeMap<String, (usize, String)>,
    /// Disagreements: program and the first difference.
    differ: Vec<String>,
    /// Programs the IR lifter lifts and A does not.
    only_ir: usize,
    /// Compute programs among those A runs.
    compute: usize,
}

fn tally(map: &mut BTreeMap<String, (usize, String)>, why: &str, at: &str) {
    map.entry(common::reason(why)).or_insert((0, at.to_string())).0 += 1;
}

fn gate(game: Game) {
    let Some((name, paths)) = common::corpus(game) else {
        return;
    };
    gate_over(&name, &paths, NOT_YET);
}

/// What the IR lifter does not lift yet that the old lifter does: the programs of the stages made of functions (hull,
/// domain, geometry, and a vertex program feeding tessellation). The gate fails on any other refusal.
const NOT_YET: &[&str] = &["program made of functions is not lifted to the IR yet"];

/// The gate over the containers `paths` (`name` in its report): a refusal is a failure unless its reason contains one
/// of `not_yet`.
fn gate_over(name: &str, paths: &[std::path::PathBuf], not_yet: &[&str]) {
    let limit = common::diff_limit();
    let trace = std::env::var_os("IR_GATE_TRACE").is_some();
    let mut g = Gate::default();
    for path in paths.iter().take(limit) {
        let at = path.display().to_string();
        let bytes = std::fs::read(path).unwrap();
        g.programs += 1;
        if trace {
            eprintln!("  {name} {}: {at}", g.programs);
        } else if g.programs.is_multiple_of(500) {
            eprintln!("  {name}: {} programs", g.programs);
        }
        let (old, p) = match (iridia_dxbc::lift(&bytes), iridia_dxbc::ir::lift(&bytes)) {
            (Ok(old), Ok(p)) => (old, p),
            (Ok(_), Err(e)) => {
                tally(&mut g.refused, &e.to_string(), &at);
                continue;
            }
            (Err(_), Ok(_)) => {
                g.only_ir += 1;
                continue;
            }
            (Err(_), Err(_)) => continue,
        };
        if old.interface.stage == Stage::Compute {
            compute(&mut g, &old, p, &at);
            continue;
        }
        let a = match ProgramShader::new(old) {
            Ok(a) => a,
            Err(e) => {
                tally(&mut g.a_fails, &format!("A not built: {e}"), &at);
                continue;
            }
        };
        g.a += 1;
        let b = match ProgramShader::from_ir(p.clone()) {
            Ok(b) => b,
            Err(e) => {
                tally(&mut g.unrun, &format!("B: {e}"), &at);
                continue;
            }
        };
        match compare(&a, &b, CASES) {
            Verdict::Agree => g.agree_b += 1,
            Verdict::AFails(e) => {
                tally(&mut g.a_fails, &e, &at);
                continue;
            }
            Verdict::Differs(d) => g.differ.push(format!("{at} (B)\n  {d}")),
        }
        let pp = match passed(p).and_then(|p| ProgramShader::from_ir(p).map_err(|e| e.to_string())) {
            Ok(pp) => pp,
            Err(e) => {
                tally(&mut g.unrun, &format!("P: {e}"), &at);
                continue;
            }
        };
        match compare(&a, &pp, CASES) {
            Verdict::Agree => g.agree_p += 1,
            Verdict::AFails(e) => {
                tally(&mut g.a_fails, &e, &at);
                continue;
            }
            Verdict::Differs(d) => g.differ.push(format!("{at} (P)\n  {d}")),
        }
    }
    println!(
        "{name}: {} programs, A runs {} ({} compute), B agrees {}, P agrees {}, {} differ, IR refuses {}, only the IR \
         lifts {}",
        g.programs,
        g.a,
        g.compute,
        g.agree_b,
        g.agree_p,
        g.differ.len(),
        g.refused.values().map(|(n, _)| n).sum::<usize>(),
        g.only_ir,
    );
    for (why, (n, e)) in &g.refused {
        println!("  refused {n:5}  {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.unrun {
        println!("  not run {n:5}  {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.a_fails {
        println!("  A fails {n:5}  {why}  (e.g. {e})");
    }
    for d in g.differ.iter().take(20) {
        println!("  DIFFER {d}");
    }
    assert!(g.differ.is_empty(), "{} programs differ", g.differ.len());
    assert!(
        g.unrun.is_empty(),
        "{} IR-lifted programs do not run",
        g.unrun.values().map(|(n, _)| n).sum::<usize>()
    );
    let regressions: Vec<&String> = g
        .refused
        .keys()
        .filter(|why| !not_yet.iter().any(|n| why.contains(n)))
        .collect();
    assert!(
        regressions.is_empty(),
        "the IR lifter refuses programs the old one lifts: {regressions:?}"
    );
}

/// Unordered-access texels by view, element, texel and layer: their four lanes.
type Texels = BTreeMap<(u32, u32, [u32; 3], Option<u32>), [u32; 4]>;

/// The memory a compute dispatch leaves: each read-write buffer's words, each texel its stores and atomic operations
/// reached, and each counter.
#[derive(Debug, PartialEq, Eq)]
struct Memory {
    buffers: BTreeMap<ResourceId, Vec<u32>>,
    elements: BTreeMap<(ResourceId, u32), Vec<u32>>,
    texels: Texels,
    counters: BTreeMap<(u32, u32), u32>,
}

impl Memory {
    /// The memory with every word that is a NaN as one pattern, as outputs are compared: a float a program stores
    /// keeps the NaN its arithmetic made, whose sign and payload IEEE 754 leaves open.
    fn canonical(mut self) -> Self {
        for words in self.buffers.values_mut().chain(self.elements.values_mut()) {
            for w in words.iter_mut() {
                *w = canonical(*w);
            }
        }
        for t in self.texels.values_mut() {
            *t = t.map(canonical);
        }
        self
    }
}

/// A compute run's backend: textures answered by the hash, unordered-access texels kept as the program leaves them
/// (a texel never written reads as its hash), counters starting from a hash of their view.
#[derive(Default)]
struct Group {
    hashed: Hashed,
    texels: Texels,
    counters: BTreeMap<(u32, u32), u32>,
}

/// An element of the interface as the one word the hashed answers take.
fn element_word(e: Element) -> u32 {
    e.id.0 | e.index << 16
}

impl Backend for Group {
    fn sample(&mut self, r: ImageSample<Element>) -> Result<[f32; 4], String> {
        self.hashed.sample_image(ImageSample {
            texture: element_word(r.texture),
            sampler: element_word(r.sampler),
            dim: r.dim,
            coord: r.coord,
            layer: r.layer,
            lod: r.lod,
            offset: r.offset,
            gather: r.gather,
            depth_ref: r.depth_ref,
            depth: r.depth,
        })
    }
    fn load(&mut self, r: ImageLoad<Element>) -> Result<[u32; 4], String> {
        if r.storage {
            let key = (
                r.texture.id.0,
                r.texture.index,
                r.texel.map(i32::cast_unsigned),
                r.layer.map(i32::cast_unsigned),
            );
            if let Some(t) = self.texels.get(&key) {
                return Ok(*t);
            }
        }
        self.hashed.load_image(ImageLoad {
            texture: element_word(r.texture),
            dim: r.dim,
            storage: r.storage,
            texel: r.texel,
            layer: r.layer,
            sample: r.sample,
            mip: r.mip,
        })
    }
    fn query(&mut self, r: ImageQuery<Element>) -> Result<[u32; 4], String> {
        self.hashed.query_image(ImageQuery {
            texture: element_word(r.texture),
            dim: r.dim,
            kind: r.kind,
            level: r.level,
        })
    }
    fn store_image(&mut self, r: ImageStore<ResourceId>) -> Result<(), String> {
        let key = (r.view.0, r.index, r.texel, r.layer);
        let mut t = self.texels.get(&key).copied().unwrap_or_default();
        for (to, from) in t.iter_mut().zip(r.value).take(usize::from(r.lanes)) {
            *to = from;
        }
        self.texels.insert(key, t);
        Ok(())
    }
    fn atomic_image(&mut self, r: ImageAtomic<ResourceId>) -> Result<u32, String> {
        let key = (r.view.0, r.index, r.texel, r.layer);
        let fresh = common::mix(hash_words(&[
            5, r.view.0, r.index, r.texel[0], r.texel[1], r.texel[2],
        ]));
        let t = self.texels.entry(key).or_insert([fresh, 0, 0, 0]);
        let old = t[0];
        t[0] = r.op.apply(old, r.value, r.compare, r.signed);
        Ok(old)
    }
    fn atomic_buffer(&mut self, r: BufferAtomic) -> Result<u32, String> {
        Err(format!(
            "resource {}: a buffer the run does not hold",
            r.view.id.0
        ))
    }
    fn counter(&mut self, view: Element, decrement: bool) -> Result<u32, String> {
        let c = self
            .counters
            .entry((view.id.0, view.index))
            .or_insert_with(|| common::mix(hash_words(&[6, view.id.0, view.index])) % 64);
        if decrement {
            *c = c.wrapping_sub(1);
            Ok(*c)
        } else {
            let before = *c;
            *c = c.wrapping_add(1);
            Ok(before)
        }
    }
}

/// The thread groups a compute program is dispatched over.
const GROUPS: [u32; 3] = [2, 1, 1];

/// The elements of an array of resources a compute run binds: all of a small array, 32 of a larger or unbounded
/// one (a program indexing past them reads a binding it was not given, which the run refuses).
fn bound_elements(count: u32) -> u32 {
    count.min(32)
}

/// What a compute run binds and changes, seeded by case: the constant and read-only buffers bound (an array's
/// elements each), the read-write ones the dispatch changes. Each buffer has 64 elements of small words, so an
/// address the program computes from one stays mostly inside.
struct Seeded {
    resources: Resources,
    buffers: BTreeMap<ResourceId, Vec<u32>>,
    elements: BTreeMap<(ResourceId, u32), Vec<u32>>,
}

fn seeded(program: &iridia::Program, c: u32) -> Seeded {
    let mut s = Seeded {
        resources: Resources::default(),
        buffers: BTreeMap::new(),
        elements: BTreeMap::new(),
    };
    for r in &program.interface.resources {
        let array = r.count != 1;
        let (kind, storage) = match &r.kind {
            ResourceKind::Buffer(k) => (k, false),
            ResourceKind::Storage {
                target: StorageTarget::Buffer(k),
                ..
            } => (k, true),
            ResourceKind::ConstantBuffer { layout } => {
                let rows = layout.size.div_ceil(16).max(1);
                for e in 0..bound_elements(r.count) {
                    let v = (0..rows)
                        .map(|row| {
                            std::array::from_fn(|k| {
                                common::mix(hash_words(&[7, c, r.id.0, e, row, k as u32])) % 97
                            })
                        })
                        .collect();
                    if array {
                        s.resources.constant_elements.insert((r.id, e), v);
                    } else {
                        s.resources.constants.insert(r.id, v);
                    }
                }
                continue;
            }
            _ => continue,
        };
        let words = match kind {
            BufferKind::Typed(_) => 64 * 4,
            BufferKind::Structured { stride } => 64 * stride.div_ceil(4),
            BufferKind::Raw => 256,
        };
        for e in 0..bound_elements(r.count) {
            let v: Vec<u32> = (0..words)
                .map(|w| common::mix(hash_words(&[8, c, r.id.0, e, w])) % 251)
                .collect();
            match (storage, array) {
                (true, true) => {
                    s.elements.insert((r.id, e), v);
                }
                (true, false) => {
                    s.buffers.insert(r.id, v);
                }
                (false, true) => {
                    s.resources.elements.insert((r.id, e), v);
                }
                (false, false) => {
                    s.resources.buffers.insert(r.id, v);
                }
            }
        }
    }
    s
}

/// One dispatch of `exe` (of `program`'s interface) on case `c`'s memory, and the memory it leaves.
fn dispatch(exe: &Executable, program: &iridia::Program, c: u32) -> Result<Memory, String> {
    let mut s = seeded(program, c);
    let bound = exe.bind_resources(&s.resources)?;
    let mut backend = Group::default();
    exe.dispatch_elements(&bound, &mut backend, &mut s.buffers, &mut s.elements, GROUPS, 32)?;
    Ok(Memory {
        buffers: s.buffers,
        elements: s.elements,
        texels: backend.texels,
        counters: backend.counters,
    }
    .canonical())
}

/// A compute program: A, B and P dispatched on the same seeded memory must leave the same memory.
fn compute(g: &mut Gate, old: &iridia::Program, p: iridia_ir::Program, at: &str) {
    let a = match Executable::new(old) {
        Ok(a) => a,
        Err(e) => return tally(&mut g.a_fails, &format!("A not built: {e}"), at),
    };
    g.a += 1;
    g.compute += 1;
    let columns = [("B", Ok(p.clone())), ("P", passed(p))];
    for (column, ir) in columns {
        let b = match ir.and_then(|p| Executable::from_ir(&p).map_err(|e| e.to_string())) {
            Ok(b) => b,
            Err(e) => return tally(&mut g.unrun, &format!("{column}: {e}"), at),
        };
        for c in 0..CASES {
            let x = dispatch(&a, old, c);
            if let Err(e) = &x {
                return tally(&mut g.a_fails, e, at);
            }
            let y = dispatch(&b, old, c);
            if x != y {
                let d = format!("case {c}: A {x:x?}\n  but {y:x?}");
                let d: String = d.chars().take(4000).collect();
                return g.differ.push(format!("{at} ({column})\n  {d}"));
            }
        }
        if column == "B" {
            g.agree_b += 1;
        } else {
            g.agree_p += 1;
        }
    }
}

/// Every DXBC fixture of the crate's tests (no game data needed).
#[test]
fn fixture_programs() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut paths = Vec::new();
    common::containers(&root, &mut paths);
    assert!(paths.len() >= 20, "{} fixtures", paths.len());
    gate_over("fixtures", &paths, NOT_YET);
}

/// The compute comparison catches a wrong lift: in each compute fixture the IR program's first integer add made a
/// subtract, the memory the dispatch leaves differs for at least one.
#[test]
fn a_planted_compute_bug_is_caught() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut paths = Vec::new();
    common::containers(&root, &mut paths);
    let mut planted = 0;
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        let (Ok(old), Ok(mut p)) = (iridia_dxbc::lift(&bytes), iridia_dxbc::ir::lift(&bytes)) else {
            continue;
        };
        if old.interface.stage != Stage::Compute {
            continue;
        }
        let Some(add) = p
            .entry
            .values
            .iter_mut()
            .find(|v| matches!(v.op, iridia_ir::Op::Binary(iridia_ir::Binary::IAdd, ..)))
        else {
            continue;
        };
        let iridia_ir::Op::Binary(op, ..) = &mut add.op else {
            unreachable!()
        };
        *op = iridia_ir::Binary::ISub;
        planted += 1;
        let mut g = Gate::default();
        compute(&mut g, &old, p, &path.display().to_string());
        if !g.differ.is_empty() {
            return;
        }
    }
    panic!("no planted subtract of {planted} compute programs was caught");
}

/// The quad runs catch a wrong derivative: with its first derivative taken along the other axis, a program of the
/// corpus whose derivatives need a quad no longer agrees.
#[test]
fn a_planted_derivative_bug_is_caught() {
    let Some((_, paths)) = common::corpus(Game::Sekiro) else {
        return;
    };
    let mut tried = 0;
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        let (Ok(old), Ok(mut p)) = (iridia_dxbc::lift(&bytes), iridia_dxbc::ir::lift(&bytes)) else {
            continue;
        };
        let (Ok(a), Ok(b)) = (ProgramShader::new(old), ProgramShader::from_ir(p.clone())) else {
            continue;
        };
        if !(a.needs_quad() || b.needs_quad()) {
            continue;
        }
        let Some(d) = p
            .entry
            .values
            .iter_mut()
            .find(|v| matches!(v.op, iridia_ir::Op::Unary(iridia_ir::Unary::Derivative(..), _)))
        else {
            continue;
        };
        let iridia_ir::Op::Unary(iridia_ir::Unary::Derivative(axis, _), _) = &mut d.op else {
            unreachable!()
        };
        *axis = match axis {
            iridia_ir::Axis::X => iridia_ir::Axis::Y,
            iridia_ir::Axis::Y => iridia_ir::Axis::X,
        };
        tried += 1;
        let planted = ProgramShader::from_ir(p).unwrap();
        if matches!(compare(&a, &planted, CASES), Verdict::Differs(_)) {
            return;
        }
    }
    panic!("no planted derivative of {tried} programs that need a quad was caught");
}

#[test]
fn sekiro_programs() {
    gate(Game::Sekiro);
}
#[test]
fn riders_republic_programs() {
    gate(Game::Riders);
}
#[test]
fn unity_programs() {
    gate(Game::Sledding);
}

/// The gate catches a wrong lift: the IR program's first float add made a subtract no longer agrees.
#[test]
fn a_planted_bug_is_caught() {
    let Some((_, paths)) = common::corpus(Game::Sekiro) else {
        return;
    };
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        let (Ok(old), Ok(mut p)) = (iridia_dxbc::lift(&bytes), iridia_dxbc::ir::lift(&bytes)) else {
            continue;
        };
        let Some(add) = p
            .entry
            .values
            .iter_mut()
            .find(|v| matches!(v.op, iridia_ir::Op::Binary(iridia_ir::Binary::FAdd, ..)))
        else {
            continue;
        };
        let iridia_ir::Op::Binary(op, ..) = &mut add.op else {
            unreachable!()
        };
        *op = iridia_ir::Binary::FSub;
        let (Ok(a), Ok(b)) = (ProgramShader::new(old), ProgramShader::from_ir(p)) else {
            continue;
        };
        // an add whose result no output reads is no plant: the first program the plant reaches is caught
        if matches!(compare(&a, &b, CASES), Verdict::Differs(_)) {
            return;
        }
    }
    panic!("no planted subtract was caught");
}

/// One program's two lifts and runs, for a disagreement the gate prints: `IR_GATE_ONE=<container> cargo test -p
/// iridia-dxbc --test ir_gate one -- --ignored --nocapture`.
#[test]
#[ignore = "a diagnostic: names its program in IR_GATE_ONE"]
fn one() {
    let path = std::env::var("IR_GATE_ONE").expect("IR_GATE_ONE names a container");
    let bytes = std::fs::read(&path).unwrap();
    let old = iridia_dxbc::lift(&bytes).unwrap();
    println!(
        "A (old lift) as WGSL:\n{}",
        iridia::emit::wgsl(&old).unwrap_or_else(|e| e.to_string())
    );
    let p = iridia_dxbc::ir::lift(&bytes).unwrap();
    println!("B (IR lift):\n{}", iridia_ir::text::print(&p.entry));
    let a = ProgramShader::new(old).unwrap();
    let b = ProgramShader::from_ir(p).unwrap();
    for c in 0..CASES {
        let case = gate_case(&a, c, a.needs_quad() || b.needs_quad());
        println!(
            "case {c}: inputs {:x?}",
            case.iter().map(|q| &q.inputs).collect::<Vec<_>>()
        );
        let (mut sa, mut sb) = (Hashed::default(), Hashed::default());
        let (x, y) = (run_asking(&a, &case, &mut sa), run_asking(&b, &case, &mut sb));
        println!("  A {x:x?}\n  B {y:x?}");
        for (k, (p, q)) in sa.asked.iter().zip(&sb.asked).enumerate() {
            if p != q {
                println!("  request {k} differs:\n    A {p}\n    B {q}");
            }
        }
        if c == 0 || x != y {
            println!("  A asked {:#?}\n  B asked {:#?}", sa.asked, sb.asked);
        }
        if sa.asked.len() != sb.asked.len() {
            println!("  A made {} requests, B {}", sa.asked.len(), sb.asked.len());
        }
    }
}
