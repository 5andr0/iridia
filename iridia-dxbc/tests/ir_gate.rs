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
use iridia::{
    BufferKind, FunctionSig, IoType, ItemList, ResourceId, ResourceKind, Stage, StageFacts, StorageTarget,
};
use iridia_exec::{
    Atomic, Backend, BufferAtomic, Dim, Element, Executable, ImageAtomic, ImageLoad, ImageLod, ImageQuery,
    ImageSample, ImageStore, Load, PixelShader, ProgramShader, QueryKind, Resources, Runner, Sample, Sampler,
    Store,
};

mod common;

/// The vectors each program is run on: the first two seed the constants no reflection names as floats, the next two
/// as small integers, and the last ([`ZEROED`]) is every constant and every unordered-access word 0.
const CASES: u32 = 5;

/// The vector whose constants, unordered-access memory, texel loads and buffer reads are all 0, as a cleared buffer
/// and an unset parameter are (its samples keep their hash): it reaches the code behind a guard against 0 or against
/// a constant (a debug pixel at the origin, a first use of a counter, a texel compared with an id).
const ZEROED: u32 = 4;

/// How vector `c` seeds the constants no reflection names (all of them, for [`ZEROED`]).
fn unstated(c: u32) -> common::Unstated {
    match c {
        0 | 1 => common::Unstated::Floats,
        ZEROED => common::Unstated::Zeros,
        _ => common::Unstated::Ints,
    }
}

/// Answers every request from a hash of everything it states, so two programs that ask the same agree and two that
/// ask differently (another coordinate, offset, level, component) almost surely do not.
#[derive(Default)]
struct Hashed {
    asked: Vec<String>,
    /// Every read request (a sample, a texel load, a query, a structured read) and atomic operation, in order, as the
    /// words its answer is hashed by: compared between the lifts, since a texel asked for differently is the
    /// program's own behaviour even where what it computes from it is dropped (a select that does not take it).
    reads: Vec<Vec<u32>>,
    /// The words the run's stores and atomic operations left, a texel never written read as its hash.
    written: Written,
    /// The element stride of each structured unordered-access view, by bind slot: a store's words are addressed by
    /// element and byte offset there, by element and lane in a typed buffer ([`Hashed::typed`]), by word in a raw one.
    strides: BTreeMap<u32, u32>,
    /// The bind slots of the typed unordered-access buffers.
    typed: std::collections::BTreeSet<u32>,
    /// A texel load answers small integers, not floats' bits: the vectors that seed the constants so (a count of
    /// lights read from a buffer of integers, which a float's bits make a loop of millions).
    ints: bool,
    /// Texel loads, buffer reads and an unordered-access word never written read 0 ([`ZEROED`]), not their hash.
    zeroed: bool,
}

impl Hashed {
    /// The sampler of test vector `c`.
    fn for_case(c: u32) -> Self {
        Hashed {
            ints: c >= 2,
            zeroed: c == ZEROED,
            ..Hashed::default()
        }
    }

    /// The sampler of test vector `c` for `shader`'s views.
    fn for_shader(shader: &ProgramShader, c: u32) -> Self {
        let resources = &shader.interface().resources;
        let strides = resources
            .iter()
            .filter_map(|r| match r.kind {
                ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Structured { stride }),
                    ..
                } => Some((r.provenance.d3d?.register, stride)),
                _ => None,
            })
            .collect();
        let typed = resources
            .iter()
            .filter_map(|r| match r.kind {
                ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Typed(_)),
                    ..
                } => Some(r.provenance.d3d?.register),
                _ => None,
            })
            .collect();
        Hashed {
            strides,
            typed,
            ..Hashed::for_case(c)
        }
    }

    /// The word of buffer view `view` a store or an atomic operation at `address` reaches `k` words in.
    fn buffer_word(&self, view: u32, address: [u32; 3], k: u32) -> Word {
        match self.strides.get(&view) {
            Some(&stride) => Word::Element {
                view,
                element: address[0],
                word: (address[1] / 4).wrapping_add(k),
                stride,
            },
            // a typed buffer's lanes are its element's, four words to an element
            None if self.typed.contains(&view) => Word::Buffer {
                view,
                word: address[0].wrapping_mul(4).wrapping_add(k),
            },
            None => Word::Buffer {
                view,
                word: address[0].wrapping_add(k),
            },
        }
    }
}

/// A 32-bit word of an unordered-access view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Word {
    /// Of a raw or typed buffer, by word.
    Buffer { view: u32, word: u32 },
    /// Of a structured buffer: an element's word.
    Element {
        view: u32,
        element: u32,
        word: u32,
        stride: u32,
    },
    /// A lane of a texel.
    Texel {
        view: u32,
        dim: u32,
        texel: [u32; 3],
        layer: Option<u32>,
        lane: u32,
    },
}

/// What a pixel or vertex run's stores and atomic operations do: each word written, with the values written to it in
/// the order they were (the last is what it holds). Each word's sequence is compared, not the order of writes to
/// different words: Direct3D orders an invocation's writes to a word, and leaves the order between the lanes of a quad
/// open; a record a program rewrites in place leaves the same last words with a write left out.
#[derive(Debug, Default, PartialEq, Eq)]
struct Written {
    writes: BTreeMap<Word, Vec<u32>>,
}

impl Written {
    /// Word `at` takes `v`.
    fn write(&mut self, at: Word, v: u32) {
        self.writes.entry(at).or_default().push(v);
    }

    /// What word `at` holds, where it was written.
    fn last(&self, at: &Word) -> Option<u32> {
        self.writes.get(at).and_then(|v| v.last()).copied()
    }

    /// An atomic operation's change of word `at` (`fresh` where never written): the word as it was.
    fn atomic(&mut self, at: Word, fresh: u32, change: impl FnOnce(u32) -> u32) -> u32 {
        let old = self.last(&at).unwrap_or(fresh);
        self.write(at, change(old));
        old
    }

    /// The words with every NaN as one pattern, as outputs are compared.
    fn canonical(mut self) -> Self {
        for w in self.writes.values_mut().flatten() {
            *w = canonical(*w);
        }
        self
    }
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
        self.reads.push(w);
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
        let h = hash_words(&w);
        self.reads.push(w);
        let mut t = if self.ints {
            std::array::from_fn(|k| common::mix(h ^ k as u64) % 4)
        } else {
            texel_floats(h).map(f32::to_bits)
        };
        if self.zeroed {
            // a cleared texture: a guard comparing a texel with a constant (an id lookup) holds
            t = [0; 4];
        }
        if r.storage {
            // what this run stored there
            for (lane, t) in (0..).zip(t.iter_mut()) {
                let at = Word::Texel {
                    view: r.texture,
                    dim: dim_word(r.dim),
                    texel: r.texel.map(i32::cast_unsigned),
                    layer: r.layer.map(i32::cast_unsigned),
                    lane,
                };
                if let Some(w) = self.written.last(&at) {
                    *t = w;
                }
            }
        }
        Ok(t)
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
        self.reads.push(w.to_vec());
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
    fn store(&mut self, r: Store) -> Result<(), String> {
        self.asked.push(format!("{r:?}"));
        for k in (0..4).filter(|k| r.mask >> k & 1 != 0) {
            let at = self.buffer_word(r.view, r.address, k);
            self.written.write(at, r.value[k as usize]);
        }
        Ok(())
    }
    fn store_image(&mut self, r: ImageStore<u32>) -> Result<(), String> {
        self.asked.push(format!("{r:?}"));
        for (lane, &v) in (0..u32::from(r.lanes)).zip(&r.value) {
            let at = Word::Texel {
                view: r.view,
                dim: dim_word(r.dim),
                texel: r.texel,
                layer: r.layer,
                lane,
            };
            self.written.write(at, v);
        }
        Ok(())
    }
    fn atomic(&mut self, r: Atomic) -> Result<u32, String> {
        self.asked.push(format!("{r:?}"));
        let [x, y, z] = r.address;
        self.reads.push(vec![
            8,
            r.view,
            x,
            y,
            z,
            r.op as u32,
            r.value,
            r.compare,
            u32::from(r.signed),
        ]);
        let at = self.buffer_word(r.view, r.address, 0);
        let fresh = if self.zeroed {
            0
        } else {
            common::mix(hash_words(&[8, r.view, r.address[0], r.address[1], r.address[2]]))
        };
        Ok(self
            .written
            .atomic(at, fresh, |old| r.op.apply(old, r.value, r.compare, r.signed)))
    }
    fn atomic_image(&mut self, r: ImageAtomic<u32>) -> Result<u32, String> {
        self.asked.push(format!("{r:?}"));
        let [x, y, z] = r.texel;
        let layer = r.layer.unwrap_or(u32::MAX);
        self.reads.push(vec![
            5,
            r.view,
            x,
            y,
            z,
            layer,
            r.op as u32,
            r.value,
            r.compare,
            u32::from(r.signed),
        ]);
        let at = Word::Texel {
            view: r.view,
            dim: dim_word(r.dim),
            texel: r.texel,
            layer: r.layer,
            lane: 0,
        };
        let fresh = if self.zeroed {
            0
        } else {
            common::mix(hash_words(&[5, r.view, r.texel[0], r.texel[1], r.texel[2]]))
        };
        Ok(self
            .written
            .atomic(at, fresh, |old| r.op.apply(old, r.value, r.compare, r.signed)))
    }
    fn structured(&mut self, slot: u32, element: u32, offset: u32) -> Result<u32, String> {
        self.asked.push(format!("structured t{slot}[{element}]+{offset}"));
        self.reads.push(vec![4, slot, element, offset]);
        if self.zeroed {
            return Ok(0);
        }
        Ok(common::mix(hash_words(&[4, slot, element, offset])) % 1024)
    }
    fn buffer_elements(&mut self, slot: u32) -> Option<u32> {
        let _ = slot;
        Some(64)
    }
}

/// What a run gives: each invocation's output lanes, masks and discard, the words its stores and atomic operations
/// left, and its read requests ([`Hashed::reads`]); or the error.
type Run = Result<(Vec<Invocation>, Written, Vec<Vec<u32>>), String>;

/// One invocation's written output lanes (unwritten lanes zero), its masks and whether it was discarded; a discarded
/// invocation's outputs are not compared, since nothing it writes reaches a target.
type Invocation = (Vec<[u32; 4]>, Vec<u8>, bool);

/// A run of test vector `c` (its invocation, or its quad's four).
fn run(shader: &ProgramShader, c: u32, cases: &[common::Case]) -> Run {
    run_asking(shader, cases, &mut Hashed::for_shader(shader, c))
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
    Ok((
        out,
        std::mem::take(&mut sampler.written).canonical(),
        std::mem::take(&mut sampler.reads),
    ))
}

/// A NaN as one bit pattern: two lifts may produce different NaNs.
fn canonical(w: u32) -> u32 {
    if f32::from_bits(w).is_nan() {
        0x7fc0_0000
    } else {
        w
    }
}

/// The IR lift's program after the passes the cut-over runs, on each of its functions.
fn passed(mut p: iridia_ir::Program) -> Result<iridia_ir::Program, String> {
    for f in std::iter::once(&mut p.entry).chain(p.patch.as_mut()) {
        iridia_passes::promote(f)?;
        iridia_passes::simplify(f);
        iridia_passes::break_flags(f)?;
    }
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

/// The inputs of a quad's lanes, in the order the gate tries them: a quad whose lanes take different paths to a
/// derivative or an implicit-level sample cannot be evaluated, and runs again with the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum QuadInputs {
    /// Each lane its own.
    Own,
    /// As a rasterizer interpolates them across a quad: each interpolated float input the first lane's plus a step
    /// of its own along x and along y (so its derivatives are those steps); the flat and integer ones the first
    /// lane's.
    Coherent,
    /// Every lane the first lane's (but its position): the derivatives of the inputs are 0.
    First,
}

/// Test vector `c`: the constants seeded as [`unstated`] says (a loop bound read from a stripped buffer needs small
/// integers), as [`Hashed`] then answers texel loads. A program that takes
/// derivatives of what it computes (`quad`) runs as a quad of four vectors: the lanes' inputs as `inputs` says, the
/// constants, the facing of lane 0 and positions a pixel apart.
fn gate_case(a: &ProgramShader, c: u32, quad: bool, inputs: QuadInputs) -> Vec<common::Case> {
    let unstated = unstated(c);
    if !quad {
        return vec![common::case_with(a, c, unstated)];
    }
    let own = |k: u32| if inputs == QuadInputs::Own { k } else { 0 };
    let mut lanes: Vec<common::Case> = (0..4)
        .map(|k| common::case_with(a, c * 4 + own(k), unstated))
        .collect();
    if inputs == QuadInputs::Coherent {
        for v in &a.interface().inputs {
            let interpolated = v.system.is_none()
                && v.value.scalar == iridia::ScalarType::Float
                && v.interp.is_none_or(|i| i.mode != iridia::InterpMode::Flat);
            let Some(sig) = v.provenance.d3d.filter(|_| interpolated) else {
                continue;
            };
            let r = sig.register as usize;
            if r >= lanes[0].inputs.len() {
                continue;
            }
            for lane in (0..4usize).filter(|l| sig.mask >> l & 1 != 0) {
                let h = common::mix(hash_words(&[11, c, sig.register, lane as u32]));
                let dx = (common::unit(h) - 0.5) / 16.0;
                let dy = (common::unit(common::mix(u64::from(h) ^ 1)) - 0.5) / 16.0;
                let base = f32::from_bits(lanes[0].inputs[r][lane]);
                for (k, l) in lanes.iter_mut().enumerate() {
                    let x = base + dx * (k & 1) as f32 + dy * (k >> 1) as f32;
                    l.inputs[r][lane] = x.to_bits();
                }
            }
        }
    }
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

/// A and B on every vector, and the furthest of the [`QuadInputs`] a quad needed. A quad whose lanes diverge at a
/// derivative or an implicit-level sample runs again with coherent inputs, then with every lane the first lane's. A
/// vector A cannot evaluate (a loop past the executor's budget on these inputs, a request the sampler does not
/// answer) is not compared, and B is not run on it: the third result is why, for each; a program none of whose
/// vectors A evaluates is `AFails`.
/// Where `textures`, every texture request (sample, texel load, query) B makes must be one A makes too: a texel the
/// program asks for differently is its own behaviour even where a select then drops it. A request A makes and B does
/// not is one for a value nothing uses, which B's compiler leaves out (buffer reads have no effect; a pass removes a
/// sample nothing reads, so P is held to its results only).
fn compare(
    a: &ProgramShader,
    b: &ProgramShader,
    cases: u32,
    textures: bool,
) -> (Verdict, QuadInputs, Vec<String>) {
    let mut skipped = Vec::new();
    let mut compared = 0;
    let mut furthest = QuadInputs::Own;
    let quad = a.needs_quad() || b.needs_quad();
    for c in 0..cases {
        let mut case = gate_case(a, c, quad, QuadInputs::Own);
        let mut x = run(a, c, &case);
        for next in [QuadInputs::Coherent, QuadInputs::First] {
            if !(quad && x.as_ref().is_err_and(|e| e.contains("diverge"))) {
                break;
            }
            case = gate_case(a, c, quad, next);
            x = run(a, c, &case);
            furthest = furthest.max(next);
        }
        if let Err(e) = &x {
            skipped.push(e.clone());
            continue;
        }
        let y = run(b, c, &case);
        let same = match (&x, &y) {
            (Ok((ix, wx, rx)), Ok((iy, wy, ry))) => {
                ix == iy && wx == wy && (!textures || texture_requests(ry).is_subset(&texture_requests(rx)))
            }
            _ => x == y,
        };
        if !same {
            return (
                Verdict::Differs(format!("case {c}: A {x:x?}\n  but {y:x?}")),
                furthest,
                skipped,
            );
        }
        compared += 1;
    }
    let verdict = match skipped.first() {
        Some(e) if compared == 0 => Verdict::AFails(e.clone()),
        _ => Verdict::Agree,
    };
    (verdict, furthest, skipped)
}

/// The distinct texture requests of a run's read requests ([`Hashed::reads`]): its samples, texel loads and queries.
fn texture_requests(reads: &[Vec<u32>]) -> std::collections::BTreeSet<&[u32]> {
    reads
        .iter()
        .filter(|r| matches!(r.first(), Some(1..=3)))
        .map(Vec::as_slice)
        .collect()
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
    /// Programs A cannot evaluate on any vector, by reason.
    a_fails: BTreeMap<String, (usize, String)>,
    /// The vectors A cannot evaluate of programs B agrees with on the others, by reason: not compared.
    skipped: BTreeMap<String, (usize, String)>,
    /// Disagreements: program and the first difference.
    differ: Vec<String>,
    /// Programs the IR lifter lifts and A does not, by A's reason.
    only_ir: BTreeMap<String, (usize, String)>,
    /// Containers neither lifter lifts (DXIL among them), by the IR lifter's reason.
    neither: BTreeMap<String, (usize, String)>,
    /// Compute programs among those A runs.
    compute: usize,
    /// Programs made of functions (hull, domain, geometry, a vertex program feeding a tessellator) among those A runs.
    functions: usize,
    /// Programs a quad of whose ran with coherent inputs at most, and with the first lane's in every lane.
    coherent: usize,
    identical: usize,
    /// Of the programs that ran coherent quads and agree, those that take no derivative on any vector with coherent
    /// inputs (their quads diverged at an implicit-level sample only), and those in which every lane taking one is
    /// discarded (a derivative reaches no output there): what the coherent vectors leave untested.
    coherent_no_derivative: usize,
    coherent_derivatives_discarded: usize,
}

fn tally(map: &mut BTreeMap<String, (usize, String)>, why: &str, at: &str) {
    map.entry(common::reason(why)).or_insert((0, at.to_string())).0 += 1;
}

impl Gate {
    /// Settle a column of a program whose compared vectors agree: `compared` the vectors A evaluated, `skipped` why it
    /// could not evaluate the others. With none compared A fails the program (`false`); otherwise the column agrees,
    /// and B's skipped vectors are counted.
    fn settle(&mut self, column: &str, compared: usize, skipped: &[String], at: &str) -> bool {
        if compared == 0 {
            tally(
                &mut self.a_fails,
                skipped.first().map_or("no vector", String::as_str),
                at,
            );
            return false;
        }
        if column == "B" {
            self.agree_b += 1;
            for e in skipped {
                tally(&mut self.skipped, e, at);
            }
        } else {
            self.agree_p += 1;
        }
        true
    }
}

fn gate(game: Game) {
    let Some((name, paths)) = common::corpus(game) else {
        return;
    };
    gate_over(&name, &paths, &[]);
}

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
            (Err(e), Ok(_)) => {
                tally(&mut g.only_ir, &e.to_string(), &at);
                continue;
            }
            (Err(_), Err(e)) => {
                tally(&mut g.neither, &e.to_string(), &at);
                continue;
            }
        };
        if old.interface.stage == Stage::Compute {
            compute(&mut g, &old, p, &at);
            continue;
        }
        match iridia::stage_functions(&old.interface) {
            Ok(Some(sigs)) => {
                functions(&mut g, &old, &sigs, p, &at);
                continue;
            }
            Ok(None) => {}
            Err(e) => {
                tally(&mut g.a_fails, &format!("A's functions: {e}"), &at);
                continue;
            }
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
        let (verdict, quads, skipped) = compare(&a, &b, CASES, true);
        match quads {
            QuadInputs::Own => {}
            QuadInputs::Coherent => g.coherent += 1,
            QuadInputs::First => g.identical += 1,
        }
        if quads == QuadInputs::Coherent && matches!(verdict, Verdict::Agree) {
            let runs = coherent_runs(&b);
            g.coherent_no_derivative += usize::from(!runs.derivatives);
            g.coherent_derivatives_discarded += usize::from(runs.derivatives && !runs.kept);
        }
        match verdict {
            Verdict::Agree => {
                g.agree_b += 1;
                for e in &skipped {
                    tally(&mut g.skipped, e, &at);
                }
            }
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
        match compare(&a, &pp, CASES, false).0 {
            Verdict::Agree => g.agree_p += 1,
            Verdict::AFails(e) => {
                tally(&mut g.a_fails, &e, &at);
                continue;
            }
            Verdict::Differs(d) => g.differ.push(format!("{at} (P)\n  {d}")),
        }
    }
    println!(
        "{name}: {} programs, A runs {} ({} compute, {} of functions), B agrees {}, P agrees {}, {} differ, IR \
         refuses {}, only the IR lifts {}, neither lifts {}",
        g.programs,
        g.a,
        g.compute,
        g.functions,
        g.agree_b,
        g.agree_p,
        g.differ.len(),
        g.refused.values().map(|(n, _)| n).sum::<usize>(),
        g.only_ir.values().map(|(n, _)| n).sum::<usize>(),
        g.neither.values().map(|(n, _)| n).sum::<usize>(),
    );
    println!(
        "  quads rerun with coherent inputs {}, with the first lane's in every lane {}; of the coherent, {} take no derivative on those vectors and {} discard every lane that takes one",
        g.coherent, g.identical, g.coherent_no_derivative, g.coherent_derivatives_discarded
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
    for (why, (n, e)) in &g.skipped {
        println!("  skipped {n:5}  vectors of agreeing programs A cannot evaluate: {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.only_ir {
        println!("  IR only {n:5}  A: {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.neither {
        println!("  neither {n:5}  {why}  (e.g. {e})");
    }
    for d in g.differ.iter().take(20) {
        println!("  DIFFER {d}");
    }
    // every one by name, past the first twenty in full
    for d in g.differ.iter().skip(20) {
        println!("  differs {}", d.lines().next().unwrap_or_default());
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
struct Group {
    hashed: Hashed,
    texels: Texels,
    counters: BTreeMap<(u32, u32), u32>,
}

impl Group {
    /// The backend of test vector `c`.
    fn for_case(c: u32) -> Self {
        Group {
            hashed: Hashed::for_case(c),
            texels: BTreeMap::new(),
            counters: BTreeMap::new(),
        }
    }
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
            if self.hashed.zeroed {
                return Ok([0; 4]);
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
        let fresh = if self.hashed.zeroed {
            0
        } else {
            common::mix(hash_words(&[
                5, r.view.0, r.index, r.texel[0], r.texel[1], r.texel[2],
            ]))
        };
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
        let c = self.counters.entry((view.id.0, view.index)).or_insert_with(|| {
            if self.hashed.zeroed {
                0
            } else {
                common::mix(hash_words(&[6, view.id.0, view.index])) % 64
            }
        });
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
                                if c == ZEROED {
                                    0
                                } else {
                                    common::mix(hash_words(&[7, c, r.id.0, e, row, k as u32])) % 97
                                }
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
            // the zeroed vector's read-write buffers start cleared
            let v: Vec<u32> = (0..words)
                .map(|w| {
                    if storage && c == ZEROED {
                        0
                    } else {
                        common::mix(hash_words(&[8, c, r.id.0, e, w])) % 251
                    }
                })
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
    let mut backend = Group::for_case(c);
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
        let (mut compared, mut skipped) = (0, Vec::new());
        for c in 0..CASES {
            let x = dispatch(&a, old, c);
            if let Err(e) = &x {
                skipped.push(e.clone());
                continue;
            }
            let y = dispatch(&b, old, c);
            if x != y {
                let d = format!("case {c}: A {x:x?}\n  but {y:x?}");
                let d: String = d.chars().take(4000).collect();
                return g.differ.push(format!("{at} ({column})\n  {d}"));
            }
            compared += 1;
        }
        if !g.settle(column, compared, &skipped, at) {
            return;
        }
    }
}

/// A lane of seeded words of a varying of `scalar`, as [`common::case_with`] seeds an input's: a float in about
/// [-0.25, 1.75), a small integer.
fn seeded_lane(scalar: iridia::ScalarType, h: u32) -> u32 {
    match scalar {
        iridia::ScalarType::Float => (common::unit(h) * 2.0 - 0.25).to_bits(),
        iridia::ScalarType::Uint => h % 4,
        iridia::ScalarType::Sint => (h % 5).wrapping_sub(2),
        _ => h & 1,
    }
}

/// The registers stage function `sig` of `program` takes on case `c`: each point of an array argument and each
/// varying seeded by its scalar type, and the values the pipeline gives within what the stage's facts allow (an
/// output point the hull program has, a domain location inside the patch, an instance the geometry program runs).
fn arguments(program: &iridia::Program, sig: &FunctionSig, c: u32) -> Vec<[u32; 4]> {
    let interface = &program.interface;
    let patch: &[iridia::Varying] = match &interface.facts {
        StageFacts::Hull(h) => &h.patch_constants,
        StageFacts::Domain(d) => &d.patch_constants,
        _ => &[],
    };
    let scalar = |list: ItemList, v: Option<usize>| {
        let varyings = match list {
            ItemList::Inputs => &interface.inputs[..],
            ItemList::Outputs => &interface.outputs[..],
            ItemList::PatchConstants => patch,
            ItemList::Pipeline => &[],
        };
        v.and_then(|v| varyings.get(v))
            .map_or(iridia::ScalarType::Uint, |v| v.value.scalar)
    };
    let seeded = |k: usize, point: u32, s: iridia::ScalarType| -> [u32; 4] {
        std::array::from_fn(|lane| {
            seeded_lane(s, common::mix(hash_words(&[9, c, k as u32, point, lane as u32])))
        })
    };
    let mut registers = Vec::new();
    for (k, item) in sig.arguments.iter().enumerate() {
        let h = common::mix(hash_words(&[10, c, k as u32]));
        match (item.ty, item.name.as_str()) {
            (IoType::Array(_, n), _) => {
                let s = scalar(item.list, item.varying);
                registers.extend((0..n).map(|point| seeded(k, point, s)));
            }
            (_, "SV_OutputControlPointID") => {
                let points = match &interface.facts {
                    StageFacts::Hull(h) => h.output_control_points.max(1),
                    _ => 1,
                };
                registers.push([c % points, 0, 0, 0]);
            }
            (_, "SV_PrimitiveID") => registers.push([c, 0, 0, 0]),
            (_, "SV_GSInstanceID") => {
                let instances = match &interface.facts {
                    StageFacts::Geometry(g) => g.instances.max(1),
                    _ => 1,
                };
                registers.push([c % instances, 0, 0, 0]);
            }
            (_, "SV_DomainLocation") => {
                let u = common::unit(h);
                let v = common::unit(common::mix(u64::from(h) ^ 1)) * (1.0 - u);
                let w = match &interface.facts {
                    StageFacts::Domain(d) if d.domain == iridia::TessDomain::Triangle => 1.0 - u - v,
                    _ => 0.0,
                };
                registers.push([u.to_bits(), v.to_bits(), w.to_bits(), 0]);
            }
            _ => registers.push(seeded(k, 0, scalar(item.list, item.varying))),
        }
    }
    registers
}

/// One run of stage function `exe` (of `program`'s interface, as `sig`) on case `c`'s arguments and memory: its
/// result's members (NaNs alike) and the memory it leaves.
fn call(
    exe: &Executable,
    program: &iridia::Program,
    sig: &FunctionSig,
    c: u32,
) -> Result<(Vec<[u32; 4]>, Memory), String> {
    let s = seeded(program, c);
    let bound = exe.bind_resources(&s.resources)?;
    let mut backend = Group::for_case(c);
    let registers = arguments(program, sig, c);
    let values = {
        let mut runner = Runner::<1>::new(exe, &bound, &mut backend, None);
        let [out] = runner.run([&registers])?;
        out.values
    };
    Ok((
        values.into_iter().map(|v| v.map(canonical)).collect(),
        Memory {
            buffers: s.buffers,
            elements: s.elements,
            texels: backend.texels,
            counters: backend.counters,
        }
        .canonical(),
    ))
}

/// A program made of functions: each function of A, B and P run on the same seeded arguments and memory must give
/// the same members and leave the same memory.
fn functions(g: &mut Gate, old: &iridia::Program, sigs: &[FunctionSig], p: iridia_ir::Program, at: &str) {
    let mut a = Vec::with_capacity(sigs.len());
    for sig in sigs {
        match Executable::function(old, sig.name) {
            Ok(e) => a.push(e),
            Err(e) => return tally(&mut g.a_fails, &format!("A not built: {e}"), at),
        }
    }
    g.a += 1;
    g.functions += 1;
    let columns = [("B", Ok(p.clone())), ("P", passed(p))];
    for (column, ir) in columns {
        let ir = match ir {
            Ok(ir) => ir,
            Err(e) => return tally(&mut g.unrun, &format!("{column}: {e}"), at),
        };
        // each function is compared on a vector at least, or A fails the program
        let mut skipped = Vec::new();
        for (sig, a) in sigs.iter().zip(&a) {
            let b = match Executable::function_from_ir(&ir, sig.name) {
                Ok(b) => b,
                Err(e) => return tally(&mut g.unrun, &format!("{column} {}: {e}", sig.name), at),
            };
            let mut compared = 0;
            for c in 0..CASES {
                let x = call(a, old, sig, c);
                if let Err(e) = &x {
                    skipped.push(format!("{}: {e}", sig.name));
                    continue;
                }
                let y = call(&b, old, sig, c);
                if x != y {
                    let d = format!("{} case {c}: A {x:x?}\n  but {y:x?}", sig.name);
                    let d: String = d.chars().take(4000).collect();
                    return g.differ.push(format!("{at} ({column})\n  {d}"));
                }
                compared += 1;
            }
            if compared == 0 {
                return tally(
                    &mut g.a_fails,
                    skipped.last().map_or("no vector", String::as_str),
                    at,
                );
            }
        }
        g.settle(column, 1, &skipped, at);
    }
}

/// Every DXBC fixture of the crate's tests (no game data needed).
#[test]
fn fixture_programs() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut paths = Vec::new();
    common::containers(&root, &mut paths);
    assert!(paths.len() >= 20, "{} fixtures", paths.len());
    gate_over("fixtures", &paths, &[]);
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
        if !swap_derivative_axis(&mut p, 0) {
            continue;
        }
        tried += 1;
        let planted = ProgramShader::from_ir(p).unwrap();
        if matches!(compare(&a, &planted, CASES, true).0, Verdict::Differs(_)) {
            return;
        }
    }
    panic!("no planted derivative of {tried} programs that need a quad was caught");
}

/// What `b` does on the gate's vectors with coherent quad inputs (a vector it cannot run does nothing, as [`compare`]
/// skips it).
#[derive(Clone, Copy)]
struct CoherentRuns {
    /// Whether it takes a screen derivative on any.
    derivatives: bool,
    /// Whether a lane of any takes a derivative and is kept (not discarded).
    kept: bool,
}

fn coherent_runs(b: &ProgramShader) -> CoherentRuns {
    let mut runs = CoherentRuns {
        derivatives: false,
        kept: false,
    };
    for c in 0..CASES {
        let case = gate_case(b, c, true, QuadInputs::Coherent);
        let buffers = common::refs(&case[0], b);
        let mut sampler = Hashed::for_shader(b, c);
        if let Ok(quad) = b.execute_quad(
            [&case[0].inputs, &case[1].inputs, &case[2].inputs, &case[3].inputs],
            &buffers,
            &mut sampler,
        ) {
            runs.derivatives |= quad.iter().any(|e| e.derivatives > 0);
            runs.kept |= quad.iter().any(|e| e.derivatives > 0 && !e.discarded);
        }
    }
    runs
}

/// The `n`-th derivative of the program's entry, in the order of its values, taken along the other axis; whether it has
/// one.
fn swap_derivative_axis(p: &mut iridia_ir::Program, n: usize) -> bool {
    let Some(axis) = p
        .entry
        .values
        .iter_mut()
        .filter_map(|v| match &mut v.op {
            iridia_ir::Op::Unary(iridia_ir::Unary::Derivative(axis, _), _) => Some(axis),
            _ => None,
        })
        .nth(n)
    else {
        return false;
    };
    *axis = match axis {
        iridia_ir::Axis::X => iridia_ir::Axis::Y,
        iridia_ir::Axis::Y => iridia_ir::Axis::X,
    };
    true
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

/// The gate over a folder of containers outside the configured corpora, named by the debug key `shader.outside` and
/// walked as the fixtures are; without it, nothing to run.
#[test]
fn outside_programs() {
    let Some(root) = inputs::debug("shader.outside").map(std::path::PathBuf::from) else {
        return;
    };
    let mut paths = Vec::new();
    common::containers(&root, &mut paths);
    gate_over(&root.display().to_string(), &paths, &[]);
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
        if matches!(compare(&a, &b, CASES, true).0, Verdict::Differs(_)) {
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
        let case = gate_case(&a, c, a.needs_quad() || b.needs_quad(), QuadInputs::Own);
        println!(
            "case {c}: inputs {:x?}",
            case.iter().map(|q| &q.inputs).collect::<Vec<_>>()
        );
        let (mut sa, mut sb) = (Hashed::for_shader(&a, c), Hashed::for_shader(&b, c));
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
