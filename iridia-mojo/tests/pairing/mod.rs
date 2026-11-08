//! A Direct3D 9 program and MonoGame's GLSL of it (MojoShader's translation, made by `tests/oracles/mojoshader.py`),
//! each lifted onto the IR and run on the same inputs, constants and texels: the D3D9 program by `iridia-d3d9-ir` is
//! the oracle of the GLSL's by `iridia_mojo::ir`. Each pair runs four ways:
//!
//! - `D`, the D3D9 program on the IR, in `iridia-exec-ir` (lifted with IEEE 754's multiply, which is GLSL's);
//! - `M`, the GLSL on the IR, in `iridia-exec-ir`; `P`, the same after `iridia_passes::prepare`;
//! - `X`, `P` compiled into `iridia-exec` (`ProgramShader::from_ir`), the varyings each in the register of its slot.
//!
//! `M`, `P` and `X` must agree with `D` on every lane `D` writes: the same bits, or a NaN in both (IEEE 754 leaves a
//! NaN's sign and payload open). A pixel program runs as a 2 x 2 quad, each lane on inputs of its own.
//!
//! MojoShader prints a `def` constant's floats as C's `%f` does, six decimals (`-0.00033333333` is `-0.000333`), so the
//! GLSL of a program whose constants need more holds other numbers. Such a pair is held to `D'`, the D3D9 program with
//! each `def` holding the number its GLSL prints, and counted apart.
#![allow(dead_code, reason = "each test uses the part it needs")]

use std::collections::{BTreeMap, BTreeSet};

use iridia_exec::{Dim, ImageLod, ImageSample, PixelShader, ProgramShader, Sample, Sampler};
use iridia_ir::interface::{
    D3dSignature, Interface, Label, ResourceId, ResourceKind, Stage, SystemValue, Varying,
};

/// One program of a producer's `programs.jsonl`.
#[derive(Debug, serde::Deserialize)]
pub struct Line {
    /// Its name in the set that holds its bytes.
    pub name: String,
    /// The SHA-256 of its bytes.
    pub sha256: String,
    /// MojoShader's refusal, where it made no GLSL.
    #[serde(default)]
    pub refused: Option<String>,
    /// The GLSL, as MonoGame edits MojoShader's.
    #[serde(default)]
    pub glsl: Option<String>,
    /// `vertex` or `pixel`.
    #[serde(default)]
    pub stage: Option<String>,
    /// The program's version (`ps_3_0`).
    #[serde(default)]
    pub target: Option<String>,
    /// What the effect's tables say of it.
    #[serde(default)]
    pub hints: Option<HintsLine>,
    /// MojoShader's uniforms, in the order of the uniform array's rows.
    #[serde(default)]
    pub uniforms: Vec<UniformLine>,
}

#[derive(Debug, serde::Deserialize)]
pub struct HintsLine {
    pub constants: Vec<ConstantLine>,
    pub samplers: Vec<SamplerLine>,
    pub attributes: Vec<AttributeLine>,
}

#[derive(Debug, serde::Deserialize)]
pub struct ConstantLine {
    pub name: String,
    pub offset: u32,
    pub bytes: u32,
    pub width: u8,
}

#[derive(Debug, serde::Deserialize)]
pub struct SamplerLine {
    pub glsl_name: String,
    pub slot: u32,
    pub name: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct AttributeLine {
    pub glsl_name: String,
    pub semantic: String,
    pub index: u32,
}

#[derive(Debug, serde::Deserialize)]
pub struct UniformLine {
    #[serde(rename = "type")]
    pub kind: String,
    pub register: u32,
    pub array_count: u32,
    pub constant: bool,
}

/// The lines of a `programs.jsonl`.
pub fn lines(text: &str) -> Vec<Line> {
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("a line of programs.jsonl: {e}")))
        .collect()
}

impl Line {
    pub fn hints(&self) -> iridia_mojo::Hints {
        let h = self.hints.as_ref();
        iridia_mojo::Hints {
            target: self.target.clone().unwrap_or_default(),
            constants: h
                .map(|h| {
                    h.constants
                        .iter()
                        .map(|c| iridia_mojo::Constant {
                            name: c.name.clone(),
                            offset: c.offset,
                            bytes: c.bytes,
                            width: c.width,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            samplers: h
                .map(|h| {
                    h.samplers
                        .iter()
                        .map(|s| iridia_mojo::Sampler {
                            glsl_name: s.glsl_name.clone(),
                            slot: s.slot,
                            name: s.name.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            attributes: h
                .map(|h| {
                    h.attributes
                        .iter()
                        .map(|a| iridia_mojo::Attribute {
                            glsl_name: a.glsl_name.clone(),
                            semantic: a.semantic.clone(),
                            index: a.index,
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    pub fn stage(&self) -> Stage {
        if self.stage.as_deref() == Some("vertex") {
            Stage::Vertex
        } else {
            Stage::Pixel
        }
    }

    /// The D3D9 register each row of the uniform array holds, in row order: MojoShader's float uniforms that are not
    /// `def` constants, an array taking as many rows as it has elements.
    pub fn rows(&self) -> Vec<u32> {
        let mut rows = Vec::new();
        for u in self.uniforms.iter().filter(|u| u.kind == "float" && !u.constant) {
            for k in 0..u.array_count.max(1) {
                rows.push(u.register + k);
            }
        }
        rows
    }
}

/// The lanes a write mask names, in order.
pub fn lanes_of(mask: u8) -> Vec<usize> {
    (0..4).filter(|n| mask & (1 << n) != 0).collect()
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn seed_of(name: &str) -> u64 {
    mix(name.bytes().fold(0u64, |h, b| mix(h ^ u64::from(b))))
}

/// A float from a hash: in -4..4, or one of the values that find edge cases, one time in eight.
fn float(r: u64) -> f32 {
    const SPECIAL: [f32; 10] = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-3, 100.0, -0.25, 3.0];
    if r.is_multiple_of(8) {
        SPECIAL[(r >> 8) as usize % SPECIAL.len()]
    } else {
        ((r >> 11) as f64 / (1u64 << 53) as f64 * 8.0 - 4.0) as f32
    }
}

/// Four lanes from a hash of `seed` and `what`.
fn lanes(seed: u64, what: u64) -> [u32; 4] {
    let h = mix(seed ^ mix(what));
    [0u64, 1, 2, 3].map(|i| float(mix(h ^ i)).to_bits())
}

/// The value of a varying in one run: by its label, so the two programs read the same. A blend index is what a vertex
/// stream holds there, an index into the program's bone array (0 to 3): past the array the two read other things, the
/// Direct3D 9 program the registers after it and the GLSL the rows MojoShader packed after it, or zero.
fn varying_value(seed: u64, label: &Label) -> [u32; 4] {
    let value = lanes(
        seed,
        seed_of(&label.name) ^ u64::from(label.index).wrapping_mul(0x1000_0001),
    );
    if label.name == "BLENDINDICES" {
        return value.map(|w| (mix(u64::from(w)) % 4) as f32).map(f32::to_bits);
    }
    value
}

/// The value of float constant register `r` in one run.
fn constant_value(seed: u64, r: u32) -> [u32; 4] {
    lanes(seed ^ 0xc0c0, u64::from(r))
}

thread_local! {
    /// Whether the texel model ignores how a sample picks its level (`pair` sets it to test one explanation of a
    /// difference).
    static LEVEL_BLIND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A texel from what the request names: the texture's register, the coordinate's bits and the level.
fn texel(texture: u32, coord: &[f32], lod: ImageLod) -> [f32; 4] {
    let mut h = mix(u64::from(texture) ^ 0x5151);
    for c in coord {
        h = mix(h ^ u64::from(c.to_bits()));
    }
    let lod = if LEVEL_BLIND.get() {
        ImageLod::Implicit
    } else {
        lod
    };
    h = match lod {
        ImageLod::Implicit => mix(h ^ 1),
        ImageLod::Bias(b) => mix(h ^ 2 ^ (u64::from(b.to_bits()) << 8)),
        ImageLod::Level(l) => mix(h ^ 3 ^ (u64::from(l.to_bits()) << 8)),
        ImageLod::Gradients(_) | ImageLod::GradientsBias(..) => mix(h ^ 4),
    };
    let mut out = [0.0; 4];
    for (i, o) in out.iter_mut().enumerate() {
        *o = ((mix(h ^ i as u64) >> 40) as f32) / (1u64 << 24) as f32;
    }
    out
}

/// The texel model for `iridia-exec`.
struct Texels;

impl Sampler for Texels {
    fn sample(&mut self, r: Sample) -> Result<[f32; 4], String> {
        Ok(texel(r.texture, &r.uv, r.lod.into()))
    }
    fn sample_image(&mut self, r: ImageSample<u32>) -> Result<[f32; 4], String> {
        if r.gather.is_some() || r.depth_ref.is_some() || r.offset != [0; 3] || r.layer.is_some() {
            return Err("an unexpected sample".into());
        }
        let lanes = match r.dim {
            Dim::D1 => 1,
            Dim::D2 => 2,
            Dim::D3 | Dim::Cube => 3,
        };
        Ok(texel(r.texture, &r.coord[..lanes], r.lod))
    }
}

/// The texel model for `iridia-exec-ir`, the texture named by its register in `interface`.
fn ir_texels(interface: &Interface) -> impl FnMut(iridia_exec_ir::Sample) -> Result<[f32; 4], String> + '_ {
    move |s: iridia_exec_ir::Sample| {
        let register = interface
            .resources
            .iter()
            .find(|r| r.id == s.texture)
            .and_then(|r| r.provenance.d3d)
            .map_or(u32::MAX, |d| d.register);
        let lod = match s.lod {
            iridia_exec_ir::Lod::Auto => ImageLod::Implicit,
            iridia_exec_ir::Lod::Bias(b) => ImageLod::Bias(b),
            iridia_exec_ir::Lod::Exact(l) => ImageLod::Level(l),
            iridia_exec_ir::Lod::Grad(dx, dy) => ImageLod::Gradients([dx, dy]),
            iridia_exec_ir::Lod::GradBias(dx, dy, b) => ImageLod::GradientsBias([dx, dy], b),
        };
        Ok(texel(register, &s.coord[..usize::from(s.lanes)], lod))
    }
}

/// Where an output goes, which pairs the two programs' outputs: the clip position, a target, or a varying's label.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Place {
    Position,
    Target(u32),
    Varying(String, u32),
    Other(String),
}

fn place(v: &Varying) -> Place {
    match v.system {
        Some(SystemValue::ClipPosition) => Place::Position,
        Some(SystemValue::Target(n)) => Place::Target(n),
        None => Place::Varying(v.label.name.clone(), v.label.index),
        Some(s) => Place::Other(format!("{s:?}")),
    }
}

/// The lanes a varying of `D` holds, in the order the program stores them.
fn varying_lanes(v: &Varying) -> Vec<usize> {
    match v.provenance.d3d {
        Some(d) => lanes_of(d.mask),
        None => (0..usize::from(v.value.width)).collect(),
    }
}

/// One invocation's results: each output's four lanes by place (lanes not stored 0), and whether it was killed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct View {
    outputs: BTreeMap<Place, [u32; 4]>,
    killed: bool,
}

impl View {
    /// The four lanes of a varying output (lanes not stored 0).
    pub fn varying(&self, name: &str, index: u32) -> [u32; 4] {
        self.outputs[&Place::Varying(name.to_owned(), index)]
    }

    /// The four lanes of color target `n`.
    pub fn target(&self, n: u32) -> [u32; 4] {
        self.outputs[&Place::Target(n)]
    }
}

/// Whether `m` agrees with the oracle `d` on every lane `d` stores (`lanes` by place): the same bits or a NaN in both;
/// with `exact`, the same bits.
fn agrees(d: &View, m: &View, lanes: &BTreeMap<Place, Vec<usize>>, exact: bool) -> bool {
    if d.killed && m.killed {
        return true;
    }
    d.killed == m.killed
        && d.outputs.iter().all(|(p, a)| {
            m.outputs.get(p).is_some_and(|b| {
                lanes.get(p).into_iter().flatten().all(|&l| {
                    a[l] == b[l] || !exact && f32::from_bits(a[l]).is_nan() && f32::from_bits(b[l]).is_nan()
                })
            })
        })
}

fn ir_view(interface: &Interface, run: iridia_exec_ir::Run) -> View {
    let mut outputs = BTreeMap::new();
    for (slot, v) in interface.outputs.iter().enumerate() {
        let got = run.outputs[slot].unwrap_or([0; 4]);
        let mut four = [0u32; 4];
        for (k, l) in varying_lanes(v).into_iter().enumerate() {
            four[l] = got[k];
        }
        outputs.insert(place(v), four);
    }
    View {
        outputs,
        killed: run.killed,
    }
}

/// A program's inputs by slot in one run.
fn slots(interface: &Interface, seed: u64) -> Vec<[u32; 4]> {
    interface
        .inputs
        .iter()
        .map(|v| {
            let value = varying_value(seed, &v.label);
            let mut packed = [0u32; 4];
            for (k, l) in varying_lanes(v).into_iter().enumerate() {
                packed[k] = value[l];
            }
            packed
        })
        .collect()
}

/// The constant buffers of `D` in one run: the float bank by register, the integer bank small counts (a loop runs as
/// many times as one says), the boolean bank words.
fn d3d9_banks(interface: &Interface, seed: u64) -> Vec<(ResourceId, u32, Vec<[u32; 4]>)> {
    let mut out = Vec::new();
    for r in &interface.resources {
        let (ResourceKind::ConstantBuffer { layout }, Some(d)) = (&r.kind, r.provenance.d3d) else {
            continue;
        };
        let rows = layout.size.div_ceil(16);
        let data = (0..rows)
            .map(|row| match d.register {
                0 => constant_value(seed, row),
                1 => lanes(seed ^ 0x1111, u64::from(row)).map(|w| w % 4),
                _ => lanes(seed ^ 0x2222, u64::from(row)).map(|w| w & 1),
            })
            .collect();
        out.push((r.id, d.register, data));
    }
    out
}

/// The uniform array of `M` in one run: each row the float register MojoShader put there.
fn glsl_banks(interface: &Interface, rows: &[u32], seed: u64) -> Vec<(ResourceId, u32, Vec<[u32; 4]>)> {
    let mut out = Vec::new();
    for r in &interface.resources {
        if let (ResourceKind::ConstantBuffer { .. }, Some(d)) = (&r.kind, r.provenance.d3d) {
            out.push((
                r.id,
                d.register,
                rows.iter().map(|&reg| constant_value(seed, reg)).collect(),
            ));
        }
    }
    out
}

/// The interface with each varying in a register of its own for `iridia-exec`'s register view: an input in the
/// register of its slot, a target in the register of its number, another output in the register of its slot.
fn with_registers(interface: &mut Interface) {
    let reg = |register: usize| D3dSignature {
        register: u32::try_from(register).unwrap_or(u32::MAX),
        mask: 0b1111,
        read_write_mask: 0,
    };
    for (n, v) in interface.inputs.iter_mut().enumerate() {
        v.provenance.d3d = Some(reg(n));
    }
    for (n, v) in interface.outputs.iter_mut().enumerate() {
        match v.system {
            None => v.provenance.d3d = Some(reg(n)),
            Some(SystemValue::Target(t)) => v.provenance.d3d = Some(reg(t as usize)),
            _ => {}
        }
    }
}

/// A run of a program in `iridia-exec` with `with_registers`' view: a vertex program alone, a pixel program's quad.
fn exec_views(
    shader: &ProgramShader,
    interface: &Interface,
    inputs: &[Vec<[u32; 4]>],
    banks: &[(ResourceId, u32, Vec<[u32; 4]>)],
) -> Result<Vec<View>, String> {
    let buffers: Vec<(u32, &[[u32; 4]])> = banks.iter().map(|(_, r, d)| (*r, d.as_slice())).collect();
    let view = |e: iridia_exec::Execution, position: Option<[u32; 4]>| {
        let mut outputs = BTreeMap::new();
        for (slot, v) in interface.outputs.iter().enumerate() {
            let got = match (v.system, v.provenance.d3d) {
                (Some(SystemValue::ClipPosition), _) => position.unwrap_or([0; 4]),
                (_, Some(d)) => e.outputs.get(d.register as usize).copied().unwrap_or([0; 4]),
                _ => [0; 4],
            };
            let _ = slot;
            outputs.insert(place(v), got);
        }
        View {
            outputs,
            killed: e.discarded,
        }
    };
    if interface.stage == Stage::Vertex {
        let (e, p) = shader
            .execute_vertex(&inputs[0], &buffers, &mut Texels)
            .map_err(|e| e.to_string())?;
        return Ok(vec![view(e, Some(p))]);
    }
    let quad = [&inputs[0][..], &inputs[1][..], &inputs[2][..], &inputs[3][..]];
    let runs = shader
        .execute_quad(quad, &buffers, &mut Texels)
        .map_err(|e| e.to_string())?;
    Ok(runs.into_iter().map(|e| view(e, None)).collect())
}

/// A run of an IR program in `iridia-exec-ir`: a vertex program alone, a pixel program's quad.
pub fn ir_views(
    p: &iridia_ir::Program,
    inputs: &[Vec<[u32; 4]>],
    banks: &[(ResourceId, u32, Vec<[u32; 4]>)],
) -> Result<Vec<View>, String> {
    let buffers: Vec<(ResourceId, &[[u32; 4]])> = banks.iter().map(|(i, _, d)| (*i, d.as_slice())).collect();
    let mut sampler = ir_texels(&p.interface);
    if p.stage == Stage::Vertex {
        let run = iridia_exec_ir::run(
            p,
            &mut iridia_exec_ir::Bound {
                inputs: &inputs[0],
                buffers: &buffers,
                state: &[],
                sampler: &mut sampler,
                denormals: iridia_ir::Denormals::Any,
            },
        )?;
        return Ok(vec![ir_view(&p.interface, run)]);
    }
    let shared = std::cell::RefCell::new(&mut sampler);
    let answer = |s: iridia_exec_ir::Sample| (shared.borrow_mut())(s);
    let [mut a, mut b, mut c, mut d] = [answer; 4];
    let lane = |inputs, sampler| iridia_exec_ir::Bound {
        inputs,
        buffers: &buffers,
        state: &[],
        sampler,
        denormals: iridia_ir::Denormals::Any,
    };
    let (mut b0, mut b1, mut b2, mut b3) = (
        lane(&inputs[0], &mut a),
        lane(&inputs[1], &mut b),
        lane(&inputs[2], &mut c),
        lane(&inputs[3], &mut d),
    );
    let runs = iridia_exec_ir::run_quad(p, [&mut b0, &mut b1, &mut b2, &mut b3])?;
    Ok(runs.into_iter().map(|r| ir_view(&p.interface, r)).collect())
}

/// Every run's outcome agrees, NaN for NaN.
pub const AGREE: &str = "agree on every run";
/// As [`AGREE`], a NaN's bits differing.
pub const AGREE_NAN: &str = "agree on every run, NaN for NaN (a NaN's sign or payload differs)";
/// Every run killed in both.
pub const KILLED: &str = "agree, killed on every run (outputs never compared)";

/// What one pair gives: its class, and the example to show for it. A pair that differs is counted by what MojoShader's
/// GLSL does that its program does not, when the program has such an instruction (`departures`).
pub fn pair(name: &str, bytes: &[u8], line: &Line, runs: u64) -> (String, String) {
    let (class, example) = pair_once(name, bytes, line, runs);
    if class != DIFFER {
        return (class, example);
    }
    let reasons = departures(bytes);
    if reasons.is_empty() {
        return (class, example);
    }
    // the level of a sample is the one departure a comparison can discount: the texel model without it must agree
    if reasons.contains(&LEVEL) {
        LEVEL_BLIND.set(true);
        let blind = pair_once(name, bytes, line, runs);
        LEVEL_BLIND.set(false);
        if !blind.0.starts_with("agree") && reasons.len() == 1 {
            return (class, example);
        }
    }
    (format!("{DEPARTS}{}", reasons.join("; ")), name.into())
}

fn pair_once(name: &str, bytes: &[u8], line: &Line, runs: u64) -> (String, String) {
    let fail = |class: String, detail: String| (class, format!("{name}: {detail}"));
    if let Some(r) = &line.refused {
        return fail(format!("MojoShader refused: {}", grouped(r)), r.clone());
    }
    let glsl = line.glsl.as_deref().unwrap_or_default();
    let stage = line.stage();
    let hints = line.hints();
    let options = iridia_d3d9_ir::Options {
        multiply: iridia_ir::MulZero::Ieee,
        ..Default::default()
    };
    let d = match iridia_d3d9_ir::lift_with(bytes, &options) {
        Ok(p) => p,
        Err(e) => return fail(format!("D3D9 lift refused: {}", e.group()), e.to_string()),
    };
    let m = match iridia_mojo::ir::lift(glsl, stage, &hints) {
        Ok(p) => p,
        Err(e) => return fail(format!("GLSL lift refused: {}", e.group()), e.to_string()),
    };
    let printed = match printed_definitions(bytes, glsl) {
        Ok(b) => b,
        Err(e) => return fail("the def constants are not the GLSL's".into(), e),
    };
    let d2 = if printed == bytes {
        None
    } else {
        match iridia_d3d9_ir::lift_with(&printed, &options) {
            Ok(p) => Some(p),
            Err(e) => return fail("D' not lifted".into(), e.to_string()),
        }
    };
    if let Err(problems) = iridia_ir::validate(&m) {
        return fail("IR validation".into(), problems[0].to_string());
    }
    let mut p = m.clone();
    if let Err(e) = iridia_passes::prepare(&mut p) {
        return fail("the passes refused".into(), e);
    }
    if let Err(problems) = iridia_ir::validate(&p) {
        return fail("IR validation after the passes".into(), problems[0].to_string());
    }
    // `from_ir` takes a prepared program
    let mut x = p.clone();
    with_registers(&mut x.interface);
    let x = match ProgramShader::from_ir(x) {
        Ok(s) => s,
        Err(e) => return fail("X not compiled".into(), e.to_string()),
    };
    let mut x_interface = m.interface.clone();
    with_registers(&mut x_interface);
    // the lanes D stores, by place, which M must have too
    let mut lanes = BTreeMap::new();
    for v in &d.interface.outputs {
        lanes.insert(place(v), varying_lanes(v));
    }
    let m_lanes: BTreeMap<Place, Vec<usize>> = m
        .interface
        .outputs
        .iter()
        .map(|v| (place(v), varying_lanes(v)))
        .collect();
    if let Some(missing) = lanes.keys().find(|p| !m_lanes.contains_key(p)) {
        return fail(
            format!("an output the GLSL does not write: {missing:?}"),
            String::new(),
        );
    }
    // the GLSL's fog is one float: a program that writes more of Direct3D 9's scalar oFog than its x has no GLSL for
    // them
    let mut narrowed = false;
    for (place, lanes) in &mut lanes {
        let held = &m_lanes[place];
        if lanes.iter().any(|l| !held.contains(l)) {
            if !matches!(place, Place::Varying(name, _) if name == "FOG") {
                return fail(
                    format!("an output the GLSL holds fewer lanes of: {place:?}"),
                    String::new(),
                );
            }
            lanes.retain(|l| held.contains(l));
            narrowed = true;
        }
    }
    let rows = line.rows();
    let declared = m.interface.resources.iter().find_map(|r| match &r.kind {
        ResourceKind::ConstantBuffer { layout } => Some(layout.size / 16),
        _ => None,
    });
    if declared.is_some_and(|n| n as usize != rows.len()) {
        return fail(
            "the uniform array's rows are not MojoShader's uniforms".into(),
            format!("{} rows, {} uniforms", declared.unwrap_or(0), rows.len()),
        );
    }
    let seed = seed_of(name);
    let invocations = if stage == Stage::Vertex { 1 } else { 4 };
    // whether every run of the IR forms (M, P, X) agrees with D, with D'; whether only NaN's bits
    let (mut with_d, mut with_d2) = (true, true);
    let (mut compared, mut nan_only) = (0, false);
    let mut example = String::new();
    // a run whose D reads a float register past its constant arrays by relative addressing is not compared (`OUTSIDE`)
    let apart = kept_arrays(bytes).and_then(|arrays| RelativeReads::of(&d, arrays));
    let mut outside = 0;
    for run in 0..runs {
        let seeds: Vec<u64> = (0..invocations).map(|i| mix(seed ^ (run * 4 + i))).collect();
        let d_inputs: Vec<_> = seeds.iter().map(|&s| slots(&d.interface, s)).collect();
        let m_inputs: Vec<_> = seeds.iter().map(|&s| slots(&m.interface, s)).collect();
        let d_banks = d3d9_banks(&d.interface, mix(seed ^ run));
        let m_banks = glsl_banks(&m.interface, &rows, mix(seed ^ run));
        let views = (
            ir_views(&d, &d_inputs, &d_banks),
            d2.as_ref()
                .map(|d2| ir_views(d2, &d_inputs, &d_banks))
                .transpose(),
            ir_views(&m, &m_inputs, &m_banks),
            ir_views(&p, &m_inputs, &m_banks),
            exec_views(&x, &x_interface, &m_inputs, &m_banks),
        );
        let (vd, vd2, vm, vp, vx) = match views {
            (Ok(d), Ok(d2), Ok(m), Ok(p), Ok(x)) => (d, d2, m, p, x),
            (d, d2, m, p, x) => {
                return fail(
                    "not run".into(),
                    format!(
                        "D {:?}, D' {:?}, M {:?}, P {:?}, X {:?}",
                        d.err(),
                        d2.err(),
                        m.err(),
                        p.err(),
                        x.err()
                    ),
                );
            }
        };
        if let Some(apart) = &apart {
            match ir_views(&apart.program, &d_inputs, &apart.banks(&d_banks)) {
                Ok(v) if v != vd => {
                    outside += 1;
                    continue;
                }
                Ok(_) => {}
                Err(e) => {
                    return fail(
                        "not run".into(),
                        format!("D with its relative reads past the arrays changed: {e}"),
                    );
                }
            }
        }
        let vd2 = vd2.unwrap_or_else(|| vd.clone());
        for (i, (d, d2)) in vd.iter().zip(&vd2).enumerate() {
            if d.killed && vm[i].killed {
                continue;
            }
            compared += 1;
            for (form, v) in [("M", &vm[i]), ("P", &vp[i]), ("X", &vx[i])] {
                let (one, two) = (agrees(d, v, &lanes, false), agrees(d2, v, &lanes, false));
                if !one && !two && example.is_empty() {
                    let d2 = if d2 == d {
                        String::new()
                    } else {
                        format!("\n   D' {d2:x?}")
                    };
                    example =
                        format!("{name} run {run} invocation {i}: {form}\n   D {d:x?}{d2}\n   {form} {v:x?}");
                }
                with_d &= one;
                with_d2 &= two;
                nan_only |= one && !agrees(d, v, &lanes, true);
            }
        }
        if !with_d && !with_d2 {
            break;
        }
    }
    let outcome = match (compared, with_d, with_d2) {
        (0, ..) if outside > 0 => OUTSIDE,
        (0, ..) => KILLED,
        (_, true, _) => {
            if nan_only {
                AGREE_NAN
            } else {
                AGREE
            }
        }
        (_, false, true) => PRINTED,
        _ => return (DIFFER.into(), example),
    };
    let mut outcome = outcome.to_owned();
    if outside > 0 && compared > 0 {
        outcome.push_str("; some runs read past the constant arrays (not compared)");
    }
    if narrowed {
        outcome.push_str("; the GLSL's fog is one float (the program's other fog lanes are not compared)");
    }
    (outcome, name.into())
}

/// The class of a pair that differs where the GLSL does what its Direct3D 9 program does not.
pub const DEPARTS: &str = "MojoShader's GLSL departs from the program: ";
pub const NRM: &str = "NRM of lanes other than xyz (GLSL's normalize takes the length of the lanes written, Direct3D 9's NRM of x, y and z)";
pub const LEVEL: &str = "TEXLDL's level dropped (the GLSL samples at the implicit level)";
pub const CMP: &str = "CMP or CND that reads a lane of its destination that an earlier lane of it wrote (MojoShader writes the lanes one after the other, Direct3D 9 reads the old register)";

/// What the instructions of a Direct3D 9 program do that MojoShader's GLSL of them does otherwise: the departures
/// whose instruction is in the program, each named once.
fn departures(bytes: &[u8]) -> Vec<&'static str> {
    use d3d9_reader::Opcode;
    let Ok(program) = d3d9_reader::read(bytes) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for i in program.instructions() {
        let reason = match (i.opcode, &i.destination) {
            (Opcode::Nrm, Some(d)) if d.mask != 0b0111 => NRM,
            (Opcode::TexLdl, _) => LEVEL,
            (Opcode::Cmp | Opcode::Cnd, Some(d)) => {
                let rereads = (0..4usize).filter(|l| d.mask >> l & 1 == 1).any(|later| {
                    i.sources.iter().any(|s| {
                        s.register == d.register
                            && usize::from(s.swizzle[later]) < later
                            && d.mask >> s.swizzle[later] & 1 == 1
                    })
                });
                if !rereads {
                    continue;
                }
                CMP
            }
            _ => continue,
        };
        if !out.contains(&reason) {
            out.push(reason);
        }
    }
    out
}

/// The outcome of a pair each of whose runs reads a float register past the program's constant arrays: through
/// relative addressing, an index the seeded inputs send out of its array. There Direct3D 9 reads the registers that
/// follow and MojoShader's GLSL the rows it packed otherwise, or zero: the translation keeps only what the arrays hold.
pub const OUTSIDE: &str = "read past the constant arrays on every run (relative addressing): not compared";

/// The float registers the constant arrays of a program's constant table hold, for each array its relative reads
/// address (the constant that holds the read's base register). `None` for a program with no relative read of a float
/// register, which no register past its arrays can reach.
fn kept_arrays(bytes: &[u8]) -> Option<BTreeSet<u32>> {
    let program = d3d9_reader::read(bytes).ok()?;
    let constant = |r: &d3d9_reader::Register| r.kind == d3d9_reader::RegisterType::Const;
    let bases: BTreeSet<u32> = program
        .instructions()
        .flat_map(|i| &i.sources)
        .filter(|s| constant(&s.register) && s.relative.is_some())
        .map(|s| s.register.number)
        .collect();
    if bases.is_empty() {
        return None;
    }
    let mut arrays = BTreeSet::new();
    for c in program.constants.iter().flat_map(|t| &t.constants) {
        let range = u32::from(c.register)..u32::from(c.register) + u32::from(c.count);
        // D3DXRS_FLOAT4
        if c.register_set == 2 && bases.iter().any(|b| range.contains(b)) {
            arrays.extend(range);
        }
    }
    Some(arrays)
}

/// A Direct3D 9 program whose relative reads of the float bank go through a bank of their own: where a run's reads
/// reach past the constant arrays it shows in the outputs, once that bank holds NaN there, which a read carries to what
/// it reaches (a product with 0 among it, where other numbers would leave only a zero's sign). Direct reads, of
/// registers the arrays do not hold, still see the real bank.
struct RelativeReads {
    program: iridia_ir::Program,
    /// The float bank and the new resource, the registers of the arrays.
    bank: ResourceId,
    apart: ResourceId,
    arrays: BTreeSet<u32>,
}

impl RelativeReads {
    fn of(d: &iridia_ir::Program, arrays: BTreeSet<u32>) -> Option<RelativeReads> {
        let float = d.interface.resources.iter().find(|r| {
            matches!(r.kind, ResourceKind::ConstantBuffer { .. })
                && r.provenance.d3d.is_some_and(|b| b.register == 0)
        })?;
        let apart = ResourceId(u32::try_from(d.interface.resources.len()).ok()?);
        let mut program = d.clone();
        let mut twin = float.clone();
        twin.id = apart;
        program.interface.resources.push(twin);
        let mut any = false;
        for def in &mut program.entry.values {
            if let iridia_ir::Op::LoadBuffer {
                resource, address, ..
            } = &mut def.op
                && resource.id == float.id
                && address.index.is_some()
            {
                resource.id = apart;
                any = true;
            }
        }
        any.then_some(RelativeReads {
            program,
            bank: float.id,
            apart,
            arrays,
        })
    }

    /// `banks` and, for the relative reads, the float bank with NaN in each register outside the arrays.
    fn banks(&self, banks: &[(ResourceId, u32, Vec<[u32; 4]>)]) -> Vec<(ResourceId, u32, Vec<[u32; 4]>)> {
        let mut out = banks.to_vec();
        if let Some((_, register, rows)) = banks.iter().find(|(id, ..)| *id == self.bank) {
            let rows = (0u32..)
                .zip(rows)
                .map(|(row, data)| {
                    if self.arrays.contains(&row) {
                        *data
                    } else {
                        [0x7fc0_0001; 4]
                    }
                })
                .collect();
            out.push((self.apart, *register, rows));
        }
        out
    }
}

/// Whether the three IR forms of `glsl` run alike: `P` (after the passes) and `X` (`P` compiled into `iridia-exec`)
/// against `M` (as lifted) on `runs` seeded runs, on every lane of every output, NaN for NaN; the uniform array's row
/// `n` holds float register `n`. What differs, or what the IR lift refuses or the forms fail to run, is the error.
pub fn forms_agree(glsl: &str, stage: Stage, hints: &iridia_mojo::Hints, runs: u64) -> Result<(), String> {
    let m = iridia_mojo::ir::lift(glsl, stage, hints).map_err(|e| format!("the IR lift refuses: {e}"))?;
    let mut p = m.clone();
    iridia_passes::prepare(&mut p)?;
    iridia_ir::validate(&p).map_err(|e| format!("after the passes: {e:?}"))?;
    let mut x = p.clone();
    with_registers(&mut x.interface);
    let registers = x.interface.clone();
    let x = ProgramShader::from_ir(x).map_err(|e| format!("X not compiled: {e}"))?;
    let rows: Vec<u32> = m
        .interface
        .resources
        .iter()
        .find_map(|r| match &r.kind {
            ResourceKind::ConstantBuffer { layout } => Some(0..layout.size / 16),
            _ => None,
        })
        .into_iter()
        .flatten()
        .collect();
    let lanes: BTreeMap<Place, Vec<usize>> = m
        .interface
        .outputs
        .iter()
        .map(|v| (place(v), varying_lanes(v)))
        .collect();
    let seed = seed_of(glsl);
    let invocations = if stage == Stage::Vertex { 1 } else { 4 };
    for run in 0..runs {
        let inputs: Vec<_> = (0..invocations)
            .map(|i| slots(&m.interface, mix(seed ^ (run * 4 + i))))
            .collect();
        let banks = glsl_banks(&m.interface, &rows, mix(seed ^ run));
        let vm = ir_views(&m, &inputs, &banks).map_err(|e| format!("M not run: {e}"))?;
        let forms = [
            ("P", ir_views(&p, &inputs, &banks)),
            ("X", exec_views(&x, &registers, &inputs, &banks)),
        ];
        for (form, views) in forms {
            let views = views.map_err(|e| format!("{form} not run: {e}"))?;
            for (i, (v, m)) in views.iter().zip(&vm).enumerate() {
                if !agrees(m, v, &lanes, false) {
                    return Err(format!(
                        "run {run} invocation {i}: {form}
   M {m:x?}
   {form} {v:x?}"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The outcome of a pair whose IR forms agree with `D'`, not `D`: MojoShader printed a `def` constant at less precision.
pub const PRINTED: &str = "agree with D' (the def constants as the GLSL prints them), not D";
/// The outcome of a pair whose IR forms agree with neither.
pub const DIFFER: &str = "differ";

/// The program `bytes` with each float `def` holding the numbers the GLSL's `const vec4 <stage>_c<N> = vec4(..)` prints,
/// read as the lift reads a literal.
fn printed_definitions(bytes: &[u8], glsl: &str) -> Result<Vec<u8>, String> {
    let program = d3d9_reader::read(bytes).map_err(|e| e.to_string())?;
    let mut constants = BTreeMap::new();
    for line in glsl.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("const vec4 ") else {
            continue;
        };
        let Some((name, value)) = rest.split_once(" = vec4(") else {
            continue;
        };
        let Some(n) = name.trim_start_matches(['p', 'v']).strip_prefix("s_c") else {
            continue;
        };
        let lanes: Vec<f32> = value
            .trim_end_matches(");")
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<Result<_, _>>()
            .map_err(|e| format!("{line}: {e}"))?;
        let lanes: [f32; 4] = lanes.try_into().map_err(|_| format!("{line}: not four lanes"))?;
        constants.insert(n.parse::<u32>().map_err(|e| format!("{line}: {e}"))?, lanes);
    }
    let mut out = bytes.to_vec();
    for def in program.definitions() {
        if !matches!(def.literal, d3d9_reader::Literal::Float(_)) {
            continue;
        }
        let n = def.destination.register.number;
        // a def the program never reads has no constant in the GLSL
        let Some(lanes) = constants.get(&n) else { continue };
        for (k, x) in lanes.iter().enumerate() {
            let at = (def.destination.word + 1 + k) * 4;
            out.get_mut(at..at + 4)
                .ok_or("a def past the program's end")?
                .copy_from_slice(&x.to_bits().to_le_bytes());
        }
    }
    Ok(out)
}

/// A reason with each run of digits as `N`.
pub fn grouped(reason: &str) -> String {
    let mut out = String::with_capacity(reason.len());
    let mut digits = false;
    for c in reason.chars() {
        if c.is_ascii_digit() {
            if !digits {
                out.push('N');
            }
            digits = true;
        } else {
            out.push(c);
            digits = false;
        }
    }
    out
}

/// Programs counted by class, with an example of each.
#[derive(Default)]
pub struct Tally(BTreeMap<String, (usize, String)>);

impl Tally {
    pub fn note(&mut self, class: String, example: String) {
        let e = self.0.entry(class).or_insert((0, example));
        e.0 += 1;
    }
    pub fn print(&self) {
        for (class, (n, example)) in &self.0 {
            println!("{n:6}  {class}\n        e.g. {example}");
        }
    }
    /// The programs in classes other than `ok`.
    pub fn failures(&self, ok: &[&str]) -> usize {
        self.0
            .iter()
            .filter(|(c, _)| !ok.contains(&c.as_str()))
            .map(|(_, (n, _))| n)
            .sum()
    }
    pub fn count(&self, class: &str) -> usize {
        self.0.get(class).map_or(0, |(n, _)| *n)
    }
    pub fn classes(&self) -> impl Iterator<Item = (&String, usize)> {
        self.0.iter().map(|(c, (n, _))| (c, *n))
    }
}
