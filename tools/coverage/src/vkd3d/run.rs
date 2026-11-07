//! Runs a case file's runs on the [`Device`]: every step in order, as vkd3d's runner takes them, with the programs
//! lifted through the [`front`](crate::front) ends and run by `iridia-exec`; each probe compared with what the test
//! expects as vkd3d compares it (tests/utils.h: `compare_float` and its kin, on the resource's bytes as they are).
//!
//! A probe is answered by the resource's contents. Where a step before it could not run (a program the compiler or
//! the lifter refused, an executor error, state this runner does not model), every resource that step would have
//! written is marked with why, and a probe of it is counted as that, not checked: until a clear or a new definition
//! of the resource, or a draw that writes every pixel of a target, replaces its contents.
use std::collections::BTreeMap;
use std::sync::Arc;

use iridia::{Interface, Program, ResourceKind, SystemValue};
use iridia_exec::draw::{
    AlphaTest, Compare as DrawCompare, Cull, DepthState, DrawShader, Linkage, PointSize, Primitive,
    Rasterizer, Surface, Viewport,
};
use iridia_exec::geometry::{GeometryShader, Topology};
use iridia_exec::tessellation::TessellationShader;
use iridia_exec::{PixelShader, ProgramShader};

use super::cases::{File, InputElement, Run, What};
use crate::device::format::{Class, named};
use crate::device::resource::{Counter, Dim, Kind, Resource};
use crate::device::sample::{Address, Compare, Reduction, Sampler};
use crate::device::{Device, Dispatch, SlotRequests};
use crate::front::{Form, Lifts, Programs, Ready, Stop};
use crate::stages::{Assembly, Bank, Stages, Stopped as StageStop};

/// The wave width compute programs run with: 32 lanes, as most Direct3D 12 hardware reports.
pub const WAVE: u32 = 32;

/// What a probe found.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Outcome {
    /// It holds.
    Pass,
    /// It holds with denormals flushed, where the programs leave them open.
    PassFlushed,
    /// It does not hold.
    Fail(String),
    /// A program the probe depends on was refused by iridia: the program's name and the lifter's message.
    Refused(String, String),
    /// A program was lifted and a later stage refused it, or the executors disagree.
    Failed(String, String),
    /// The executor stopped on a program the probe depends on.
    Error(String, String),
    /// The probe depends on something this runner does not model.
    NotModelled(String),
    /// Microsoft's compiler refused a shader the probe depends on.
    CompilerRefused(String),
}

impl Outcome {
    /// The outcome's class in reports.
    pub fn class(&self) -> &'static str {
        match self {
            Outcome::Pass | Outcome::PassFlushed => "pass",
            Outcome::Fail(_) => "fail",
            Outcome::Refused(..) => "refused",
            Outcome::Failed(..) => "failed",
            Outcome::Error(..) => "error",
            Outcome::NotModelled(_) => "not modelled",
            Outcome::CompilerRefused(_) => "compiler refused",
        }
    }

    /// Why, as a class of outcomes: the message without the program it names, or for a probe that does not hold, the
    /// test file, as one file's failures share a cause more often than one message's do.
    pub fn reason(&self, file: &str) -> String {
        match self {
            Outcome::Pass | Outcome::PassFlushed => String::new(),
            Outcome::Fail(_) => format!("a probe in {file} differs"),
            Outcome::NotModelled(m) | Outcome::CompilerRefused(m) => m.clone(),
            Outcome::Refused(_, m) | Outcome::Failed(_, m) | Outcome::Error(_, m) => m.clone(),
        }
    }

    /// The message.
    pub fn detail(&self) -> String {
        match self {
            Outcome::Pass => String::new(),
            Outcome::PassFlushed => "with denormals flushed".into(),
            Outcome::Fail(m) | Outcome::NotModelled(m) | Outcome::CompilerRefused(m) => m.clone(),
            Outcome::Refused(p, m) | Outcome::Failed(p, m) | Outcome::Error(p, m) => format!("{p}: {m}"),
        }
    }
}

/// A probe's result.
#[derive(Clone, Debug)]
pub struct Probe {
    /// The line of the test file.
    pub line: u64,
    /// Whether vkd3d's runner runs this section at this model.
    pub selected: bool,
    /// What it found.
    pub outcome: Outcome,
}

/// A run's results.
#[derive(Clone, Debug)]
pub struct RunResult {
    /// The test file.
    pub file: String,
    /// The runner.
    pub runner: String,
    /// The model.
    pub model: String,
    /// Each probe.
    pub probes: Vec<Probe>,
    /// The programs its steps ran, and whether each ran (`None`) or why not.
    pub programs: BTreeMap<String, Option<Outcome>>,
}

impl RunResult {
    /// Whether vkd3d's runner runs any of the run's probes (a section of the file it runs at this model): where it
    /// does, the run's class is that of those probes, and the others are the extended ones, the same expectations at
    /// a model vkd3d's own runs leave out, which need not hold on any device (a compiler that places registers or
    /// folds constants otherwise at that model).
    pub fn selected(&self) -> bool {
        self.probes.iter().any(|p| p.selected)
    }

    /// The probe the run's class comes from: the first that does not hold, failures first, among the probes vkd3d's
    /// runner runs or, where it runs none of the run's, among all; `None` where every one holds.
    pub fn deciding(&self) -> Option<&Probe> {
        self.deciding_among(!self.selected())
    }

    /// The class of the extended probes of a run vkd3d's runner runs (`None` where it has none, or runs none of the
    /// run's).
    pub fn extended_class(&self) -> Option<&'static str> {
        if !self.selected() || self.probes.iter().all(|p| p.selected) {
            return None;
        }
        Some(
            self.probes
                .iter()
                .filter(|p| !p.selected)
                .filter(|p| p.outcome.class() != "pass")
                .min_by_key(|p| Self::rank(&p.outcome))
                .map_or("pass", |p| p.outcome.class()),
        )
    }

    /// The deciding probe among the selected probes, or among all where `all`.
    fn deciding_among(&self, all: bool) -> Option<&Probe> {
        self.probes
            .iter()
            .filter(|p| all || p.selected)
            .filter(|p| Self::rank(&p.outcome) < 6)
            .min_by_key(|p| Self::rank(&p.outcome))
    }

    /// How an outcome ranks in deciding a class: failures first, passes last.
    fn rank(o: &Outcome) -> u8 {
        match o {
            Outcome::Fail(_) => 0,
            Outcome::Failed(..) => 1,
            Outcome::Error(..) => 2,
            Outcome::Refused(..) => 3,
            Outcome::NotModelled(_) => 4,
            Outcome::CompilerRefused(_) => 5,
            Outcome::Pass | Outcome::PassFlushed => 6,
        }
    }

    /// The run's class: `pass` where every probe holds, otherwise the [`deciding`](Self::deciding) probe's.
    pub fn class(&self) -> &'static str {
        self.deciding().map_or("pass", |p| p.outcome.class())
    }
}

/// A run's state besides the device.
#[derive(Clone, Default)]
struct State {
    input_layout: Vec<InputElement>,
    depth: Option<Compare>,
    /// Direct3D 9's alpha test.
    alpha_test: Option<AlphaTest>,
    /// Direct3D 9's user clip planes, by index.
    clip_planes: [Option<[f32; 4]>; 6],
    /// Direct3D 9's point size.
    point_size: Option<PointSize>,
    /// Direct3D 9's point sprites.
    point_sprite: bool,
    /// Direct3D 9's flat shade mode (`shade mode flat`), which only vkd3d's Direct3D 9 runs take.
    flat: bool,
    viewports: BTreeMap<u32, [f32; 4]>,
    sample_mask: Option<u32>,
    /// The fixed-function state set that this runner does not model, by name.
    unmodelled: BTreeMap<String, String>,
    /// Resources whose contents are not known, and why.
    taint: BTreeMap<(Kind, u32), Outcome>,
}

/// A comparison as the draw takes it.
fn draw_compare(c: Compare) -> DrawCompare {
    match c {
        Compare::Never => DrawCompare::Never,
        Compare::Less => DrawCompare::Less,
        Compare::Equal => DrawCompare::Equal,
        Compare::LessEqual => DrawCompare::LessEqual,
        Compare::Greater => DrawCompare::Greater,
        Compare::NotEqual => DrawCompare::NotEqual,
        Compare::GreaterEqual => DrawCompare::GreaterEqual,
        Compare::Always => DrawCompare::Always,
    }
}

/// A float the case file holds as its bits.
fn float_bits(v: &serde_json::Value) -> Option<f32> {
    v.as_u64().and_then(|b| u32::try_from(b).ok()).map(f32::from_bits)
}

/// vkd3d's `alpha test <func> <ref>` (`[func, ref]`, the reference absent for `always` and `never`): the d3d9 runner
/// sets `D3DRS_ALPHAREF` to `ref * 255` as an integer (truncated), which the test compares with as a fraction of 255.
fn alpha_test_of(value: &serde_json::Value) -> Result<AlphaTest, String> {
    let shown = || format!("the alpha test {value}");
    let func = value
        .get(0)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(shown)?;
    let compare = draw_compare(Compare::named(func)?);
    let reference = match value.get(1) {
        None | Some(serde_json::Value::Null) => 0.0,
        Some(r) => {
            let r = float_bits(r).ok_or_else(shown)?;
            ((r * 255.0) as u8) as f32 / 255.0
        }
    };
    Ok(AlphaTest { compare, reference })
}

/// vkd3d's `point-size <size> <min> <max>` (`[size, min, max]`).
fn point_size_of(value: &serde_json::Value) -> Result<PointSize, String> {
    let shown = || format!("the point size {value}");
    let lane = |i: usize| value.get(i).and_then(float_bits).ok_or_else(shown);
    Ok(PointSize {
        size: lane(0)?,
        min: lane(1)?,
        max: lane(2)?,
    })
}

/// vkd3d's `clip-plane <index> <plane>` or `clip-plane <index> disable` (`[index, plane or null]`).
fn clip_plane_of(value: &serde_json::Value) -> Result<(usize, Option<[f32; 4]>), String> {
    let shown = || format!("the clip plane {value}");
    let index = value
        .get(0)
        .and_then(serde_json::Value::as_u64)
        .and_then(|i| usize::try_from(i).ok())
        .filter(|&i| i < 6)
        .ok_or_else(shown)?;
    let plane = match value.get(1) {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Array(p)) => {
            let lanes: Vec<f32> = p
                .iter()
                .map(float_bits)
                .collect::<Option<_>>()
                .ok_or_else(shown)?;
            Some(<[f32; 4]>::try_from(lanes).map_err(|_| shown())?)
        }
        Some(_) => return Err(shown()),
    };
    Ok((index, plane))
}

fn kind_slot(target: &str) -> Result<(Kind, u32), String> {
    if target == "dsv" {
        return Ok((Kind::Dsv, 0));
    }
    let (kind, slot) = target
        .split_once('-')
        .ok_or_else(|| format!("the resource {target}"))?;
    Ok((
        Kind::named(kind)?,
        slot.parse().map_err(|_| format!("the resource {target}"))?,
    ))
}

/// vkd3d's `compare_float`: equal within `ulps` steps of the float order, every NaN alike.
pub fn compare_float(got: u32, expected: u32, ulps: u32) -> bool {
    let order = |bits: u32| {
        let i = if f32::from_bits(bits).is_nan() {
            ((bits & 0x7f80_0000) | 0x40_0000) as i32
        } else {
            bits as i32
        };
        if i < 0 { i32::MIN.wrapping_sub(i) } else { i }
    };
    (i64::from(order(got)) - i64::from(order(expected))).unsigned_abs() <= u64::from(ulps)
}

/// vkd3d's `compare_double`.
pub fn compare_double(got: u64, expected: u64, ulps: u32) -> bool {
    let order = |bits: u64| {
        let i = if f64::from_bits(bits).is_nan() {
            ((bits & 0xfff0_0000_0000_0000) | 0x8_0000_0000_0000) as i64
        } else {
            bits as i64
        };
        if i < 0 { i64::MIN.wrapping_sub(i) } else { i }
    };
    (i128::from(order(got)) - i128::from(order(expected))).unsigned_abs() <= u128::from(ulps)
}

/// The major shader model of a Direct3D 9 target (`vs_2_0`: 2).
fn shader_model(target: &str) -> Option<u32> {
    target.split('_').nth(1)?.chars().next()?.to_digit(10)
}

/// The bytes a probe of `kind` with `count` elements reads at each texel.
fn probe_size(kind: &str, count: u32) -> usize {
    match kind {
        "f32" | "u32" | "i32" => 4 * count as usize,
        "f64" | "u64" | "i64" => 8 * count as usize,
        "u8" => 1,
        _ => 4,
    }
}

/// How many bytes apart a probe of `kind` with `count` elements finds consecutive texels: vkd3d reads any vector (two
/// or four lanes) as a 16-byte `vec4`, `dvec2`, `uvec4` or `u64vec2` (`get_readback_vec4` and the rest, in its
/// tests/utils.h), so x steps 16 bytes whatever the lanes; a scalar steps its own size.
fn probe_step_bytes(kind: &str, count: u32) -> usize {
    if count > 1 { 16 } else { probe_size(kind, count) }
}

/// Whether the bytes `got` match a probe's expectation.
fn holds(kind: &str, count: u32, got: &[u8], expect: &[u32], ulps: u32) -> bool {
    let word = |i: usize| u32::from_le_bytes([got[i * 4], got[i * 4 + 1], got[i * 4 + 2], got[i * 4 + 3]]);
    match kind {
        "f32" => (0..count as usize).all(|i| compare_float(word(i), expect[i], ulps)),
        "u32" | "i32" => (0..count as usize).all(|i| word(i) == expect[i]),
        "f64" => (0..count as usize).all(|i| {
            let g = u64::from(word(2 * i)) | u64::from(word(2 * i + 1)) << 32;
            let e = u64::from(expect[2 * i]) | u64::from(expect[2 * i + 1]) << 32;
            compare_double(g, e, ulps)
        }),
        "u64" | "i64" => {
            (0..count as usize).all(|i| word(2 * i) == expect[2 * i] && word(2 * i + 1) == expect[2 * i + 1])
        }
        "u8" => got[0] == (expect[0] & 0xff) as u8,
        _ => false,
    }
}

/// How the probe's bytes print.
fn shown(kind: &str, count: u32, bytes: &[u8]) -> String {
    let word =
        |i: usize| u32::from_le_bytes([bytes[i * 4], bytes[i * 4 + 1], bytes[i * 4 + 2], bytes[i * 4 + 3]]);
    match kind {
        "f32" => format!(
            "{:?}",
            (0..count as usize)
                .map(|i| f32::from_bits(word(i)))
                .collect::<Vec<_>>()
        ),
        "f64" => format!(
            "{:?}",
            (0..count as usize)
                .map(|i| f64::from_bits(u64::from(word(2 * i)) | u64::from(word(2 * i + 1)) << 32))
                .collect::<Vec<_>>()
        ),
        "u8" => format!("{:#04x}", bytes[0]),
        _ => format!("{:x?}", (0..bytes.len() / 4).map(word).collect::<Vec<_>>()),
    }
}

/// vkd3d's `uniform_buffer_write`: the buffer grows to a whole row past the write, the new words 0x7f7f7f7f.
fn write_words(words: &mut Vec<u32>, at: usize, bits: &[u32]) {
    let count = words.len().max(at + bits.len()).next_multiple_of(4);
    words.resize(count, 0x7f7f_7f7f);
    words[at..at + bits.len()].copy_from_slice(bits);
}

/// The quad vkd3d's runner draws for `draw quad`: one triangle over the target.
const QUAD_VERTICES: usize = 3;

/// The lanes of a vertex input element, from its vertex buffer.
fn element_lanes(device: &Device, layout: &[InputElement], e: &InputElement, vertex: u32) -> [u32; 4] {
    let stride: u32 = layout.iter().filter(|x| x.slot == e.slot).map(|x| x.size).sum();
    let offset: u32 = layout
        .iter()
        .filter(|x| x.slot == e.slot)
        .take_while(|x| !std::ptr::eq(*x, e))
        .map(|x| x.size)
        .sum();
    let Some(vb) = device.resources.get(&(Kind::Vb, e.slot)) else {
        return [0, 0, 0, 1f32.to_bits()];
    };
    let at = (vertex * stride + offset) as usize;
    let Ok(format) = named(&e.format) else {
        return [0; 4];
    };
    vb.data
        .get(at..at + format.size as usize)
        .map_or([0, 0, 0, 1f32.to_bits()], |b| format.read(b))
}

/// The input registers of vertex `vertex` of instance `instance` for a vertex program, its elements read at buffer
/// element `first_vertex + vertex`: SV_VertexID and SV_InstanceID count from 0 in DXBC and DXIL, whatever vertex and
/// instance the draw starts at (vkd3d's instance-id.shader_test).
fn vertex_registers(
    shader: &ProgramShader,
    device: &Device,
    layout: &[InputElement],
    first_vertex: u32,
    vertex: u32,
    instance: u32,
) -> Result<Vec<[u32; 4]>, Outcome> {
    let interface = shader.interface();
    let count = interface
        .inputs
        .iter()
        .filter_map(|v| v.provenance.d3d.map(|d| d.register as usize + 1))
        .max()
        .unwrap_or(0);
    let mut registers = vec![[0u32; 4]; count.max(shader.input_count())];
    for v in &interface.inputs {
        let Some(d) = v.provenance.d3d else { continue };
        let lanes = match v.system {
            Some(SystemValue::VertexIndex) => [vertex, 0, 0, 0],
            Some(SystemValue::InstanceIndex) => [instance, 0, 0, 0],
            Some(other) => {
                return Err(Outcome::NotModelled(format!(
                    "a vertex input of the system value {other:?}"
                )));
            }
            None => match layout
                .iter()
                .find(|e| e.name.eq_ignore_ascii_case(&v.label.name) && e.index == v.label.index)
            {
                Some(e) => element_lanes(device, layout, e, first_vertex + vertex),
                // an input no element feeds reads 0 0 0 1
                None => [0, 0, 0, 1f32.to_bits()],
            },
        };
        let reg = &mut registers[d.register as usize];
        for (k, l) in (0..4).filter(|l| d.mask >> l & 1 == 1).enumerate() {
            reg[l] = lanes[k];
        }
    }
    Ok(registers)
}

/// How a draw's topology (`triangle list`, `3 control point patch list`, ...) makes primitives of its vertices.
fn assembly_of(topology: &str) -> Result<Assembly, Outcome> {
    Ok(match topology {
        "triangle list" => Assembly::Primitives(Topology::TriangleList),
        "triangle strip" => Assembly::Primitives(Topology::TriangleStrip),
        "point list" => Assembly::Primitives(Topology::PointList),
        other => match other
            .strip_suffix(" control point patch list")
            .map(str::parse::<u32>)
        {
            Some(Ok(n)) => Assembly::Patches(n),
            _ => return Err(Outcome::NotModelled(format!("a draw of a {other}"))),
        },
    })
}

/// The primitives a draw of `vertices` vertices makes with no hull, domain or geometry program, numbered from 0.
fn primitives_of(assembly: Assembly, vertices: usize) -> Result<Vec<Primitive>, Outcome> {
    let Assembly::Primitives(topology) = assembly else {
        return Err(Outcome::NotModelled(format!(
            "a draw of {assembly:?} with no hull and domain program"
        )));
    };
    let indices = (0..vertices)
        .map(u32::try_from)
        .collect::<Result<Vec<u32>, _>>()
        .map_err(|e| Outcome::NotModelled(e.to_string()))?;
    Primitive::assemble(topology, &indices, None).map_err(|e| Outcome::NotModelled(e.detail))
}

/// The program a step names for a stage, lifted.
fn ready(programs: &Programs<'_>, name: &str) -> Result<Arc<Ready>, Outcome> {
    programs.get(name).map_err(|e| match e {
        Stop::Refused(m) => Outcome::Refused(name.to_owned(), m),
        Stop::Failed(m) => Outcome::Failed(name.to_owned(), m),
    })
}

/// The constant buffers a program binds, by slot, from the device.
fn constant_rows(shader: &ProgramShader, device: &Device) -> Vec<(u32, Vec<[u32; 4]>)> {
    shader
        .constant_buffers()
        .iter()
        .map(|&(slot, rows)| (slot, device.constant_rows(slot, rows as usize)))
        .collect()
}

/// The sampler Direct3D 9 reads at a stage with no sampler declared: its defaults (`D3DSAMP_*`: wrap, point
/// filtering, no mip filter).
fn d3d9_default_sampler() -> Sampler {
    Sampler {
        min_linear: false,
        mag_linear: false,
        mip_linear: false,
        reduction: Reduction::Standard,
        address: [Address::Wrap; 3],
        border: [0.0; 4],
        bias: 0.0,
        lod_range: [0.0, 0.0],
    }
}

/// Run one run of the file. A run with a probe that does not hold, among whose programs one leaves it open whether
/// 32-bit denormals are flushed (Direct3D 9's), runs again with them flushed: where that run breaks no probe that
/// held and makes one hold that did not, its outcomes are the run's, each probe it made hold
/// [`Outcome::PassFlushed`]. Either rule is the program's, but one for the whole run, as a device has one.
pub fn run(programs: &Programs<'_>, run: &Run) -> RunResult {
    let mut kept = run_as(programs, run, false);
    let open = kept
        .programs
        .keys()
        .any(|name| programs.get(name).is_ok_and(|r| r.lifted.denormals_open()));
    if !open || !kept.probes.iter().any(|p| p.outcome.class() == "fail") {
        return kept;
    }
    let flushed = run_as(programs, run, true);
    adopt_flushed(&mut kept.probes, flushed.probes);
    kept
}

/// Take a run's probes with denormals flushed (`flushed`) for its probes with them kept (`kept`) where that breaks no
/// probe that held and makes one hold that did not; each probe it made hold is [`Outcome::PassFlushed`]. Whether it
/// took them.
fn adopt_flushed(kept: &mut [Probe], flushed: Vec<Probe>) -> bool {
    let pairs = || kept.iter().zip(&flushed);
    let breaks = pairs().any(|(k, f)| k.outcome.class() == "pass" && f.outcome.class() != "pass");
    let mends = pairs().any(|(k, f)| k.outcome.class() == "fail" && f.outcome.class() == "pass");
    if breaks || !mends || kept.len() != flushed.len() {
        return false;
    }
    for (k, f) in kept.iter_mut().zip(flushed) {
        if k.outcome.class() == "fail" {
            k.outcome = if f.outcome == Outcome::Pass {
                Outcome::PassFlushed
            } else {
                f.outcome
            };
        }
    }
    true
}

/// Run one run of the file, with denormals flushed where `flushed` and a program leaves them open.
fn run_as(programs: &Programs<'_>, run: &Run, flushed: bool) -> RunResult {
    let mut device = Device::default();
    let mut state = State::default();
    let d3d9 = run.runner == "d3d9";
    // Direct3D 9's point size as vkd3d's runner starts it: 1 pixel, at least 1, at most `FLT_MAX`
    if d3d9 {
        state.point_size = Some(PointSize {
            size: 1.0,
            min: 1.0,
            max: f32::MAX,
        });
    }
    let mut probes = Vec::new();
    let mut used: BTreeMap<String, Option<Outcome>> = BTreeMap::new();
    let mut selected = false;
    for step in &run.steps {
        match &step.what {
            What::Section { section } => selected = section.as_deref() == Some("selected"),
            What::Resource { resource } => {
                let key = (Kind::named(&resource.kind).unwrap_or(Kind::Srv), resource.slot);
                match Resource::new(resource) {
                    Ok(r) => {
                        device.resources.insert(key, r);
                        state.taint.remove(&key);
                    }
                    Err(e) => {
                        state
                            .taint
                            .insert(key, Outcome::NotModelled(format!("line {}: {e}", step.line)));
                    }
                }
            }
            What::Sampler { sampler } => match Sampler::of(sampler) {
                Ok(s) => {
                    device.samplers.insert(sampler.slot, s);
                }
                Err(e) => {
                    state.unmodelled.insert(format!("sampler {}", sampler.slot), e);
                }
            },
            What::InputLayout { input_layout } => state.input_layout = input_layout.clone(),
            // an empty section is vkd3d's default descriptors again: each register its own resource
            What::Descriptors { descriptors } => {
                device.descriptors = (!descriptors.is_empty()).then(|| descriptors.clone());
            }
            What::Write { write, at, bits } => {
                write_words(device.constants.entry(*write).or_default(), *at, bits);
            }
            What::State { state: s } => {
                for (name, value) in s {
                    match name.as_str() {
                        "depth" => {
                            state.depth = value.as_str().and_then(|c| Compare::named(c).ok());
                        }
                        "viewport" => {
                            let v = value.as_array().cloned().unwrap_or_default();
                            let index = v.first().and_then(serde_json::Value::as_u64).unwrap_or(0) as u32;
                            let rect: Vec<f32> = v
                                .get(1)
                                .and_then(serde_json::Value::as_array)
                                .map(|a| {
                                    a.iter()
                                        .map(|x| f32::from_bits(x.as_u64().unwrap_or(0) as u32))
                                        .collect()
                                })
                                .unwrap_or_default();
                            if let [x, y, w, h] = rect[..] {
                                state.viewports.insert(index, [x, y, w, h]);
                            }
                        }
                        "alpha_test" => match alpha_test_of(value) {
                            Ok(t) => {
                                state.alpha_test = (t.compare != DrawCompare::Always).then_some(t);
                                state.unmodelled.remove(name);
                            }
                            Err(e) => {
                                state.unmodelled.insert(name.clone(), e);
                            }
                        },
                        "clip_plane" => match clip_plane_of(value) {
                            Ok((index, plane)) => {
                                if let Some(slot) = state.clip_planes.get_mut(index) {
                                    *slot = plane;
                                }
                                state.unmodelled.remove(name);
                            }
                            Err(e) => {
                                state.unmodelled.insert(name.clone(), e);
                            }
                        },
                        "sample_mask" => {
                            let m = value.as_u64().unwrap_or(u64::from(u32::MAX)) as u32;
                            state.sample_mask = (m != 0).then_some(m);
                        }
                        "flat" => match value.as_bool() {
                            Some(on) => {
                                state.flat = on;
                                state.unmodelled.remove(name);
                            }
                            None => {
                                state.unmodelled.insert(name.clone(), value.to_string());
                            }
                        },
                        "point_sprite" => match value.as_bool() {
                            Some(on) => {
                                state.point_sprite = on;
                                state.unmodelled.remove(name);
                            }
                            None => {
                                state.unmodelled.insert(name.clone(), value.to_string());
                            }
                        },
                        "point_size" => match point_size_of(value) {
                            Ok(size) => {
                                state.point_size = Some(size);
                                state.unmodelled.remove(name);
                            }
                            Err(e) => {
                                state.unmodelled.insert(name.clone(), e);
                            }
                        },
                        "fog" if value.get("mode").and_then(|m| m.as_str()) == Some("disable") => {
                            state.unmodelled.remove(name);
                        }
                        _ => {
                            state.unmodelled.insert(name.clone(), value.to_string());
                        }
                    }
                }
            }
            What::Clear {
                clear,
                value,
                stencil,
            } => {
                let Ok(key) = kind_slot(clear) else { continue };
                if let Some(r) = device.resources.get_mut(&key) {
                    clear_resource(r, value, *stencil);
                    state.taint.remove(&key);
                }
            }
            What::Copy { copy } => {
                let (Ok(a), Ok(b)) = (kind_slot(&copy[0]), kind_slot(&copy[1])) else {
                    continue;
                };
                if let Some(src) = device.resources.get(&a).map(|r| r.data.clone()) {
                    if let Some(dst) = device.resources.get_mut(&b) {
                        let n = dst.data.len().min(src.len());
                        dst.data[..n].copy_from_slice(&src[..n]);
                    }
                    match state.taint.get(&a).cloned() {
                        Some(t) => {
                            state.taint.insert(b, t);
                        }
                        None => {
                            state.taint.remove(&b);
                        }
                    }
                }
            }
            What::Unrunnable { unrunnable } => {
                // the draw or dispatch does not happen: what it would have written is not known
                let why = Outcome::CompilerRefused(unrunnable.clone());
                taint_targets(&device, &mut state, &why);
            }
            What::Draw {
                draw,
                quad,
                topology,
                vertices,
                instances,
                first_vertex,
                // every element of vkd3d's input layouts is per vertex, and SV_InstanceID counts from 0: the first
                // instance changes nothing a program reads
                first_instance: _,
            } => {
                let shape = match (quad, topology) {
                    (Some(q), _) => Shape::Quad(q.clone()),
                    (None, Some(t)) => Shape::Buffers {
                        topology: t.clone(),
                        vertices: *vertices,
                        instances: (*instances).max(1),
                        first_vertex: *first_vertex,
                    },
                    (None, None) => Shape::Quad(vec![-2.0, -2.0, -2.0, 4.0, 4.0, -2.0]),
                };
                for name in draw.values() {
                    used.entry(name.clone()).or_insert(None);
                }
                // a Direct3D 9 draw again from the IR, on what the device and state were before, which must write
                // what the WGSL's draw writes
                let before = d3d9.then(|| (device.clone(), state.clone()));
                let mut done = draw_step(
                    programs,
                    &mut device,
                    &mut state,
                    draw,
                    &shape,
                    d3d9,
                    flushed,
                    Form::Lifted,
                );
                if let (Ok(()), Some((mut ir_device, mut ir_state))) = (&done, before) {
                    let ps = draw.get("ps").cloned().unwrap_or_default();
                    let ir_done = draw_step(
                        programs,
                        &mut ir_device,
                        &mut ir_state,
                        draw,
                        &shape,
                        d3d9,
                        flushed,
                        Form::Ir,
                    );
                    done = match ir_done {
                        Ok(()) => match drawn_otherwise(&device, &ir_device) {
                            None => Ok(()),
                            Some(d) => Err(Outcome::Failed(
                                ps,
                                format!("the IR draws otherwise than its WGSL: {d}"),
                            )),
                        },
                        Err(why) => Err(Outcome::Failed(ps, format!("the IR's draw: {}", why.detail()))),
                    };
                }
                if let Err(why) = done {
                    for name in draw.values() {
                        if let Some(slot) = used.get_mut(name)
                            && slot.is_none()
                            && why.class() != "not modelled"
                        {
                            *slot = Some(why.clone());
                        }
                    }
                    taint_targets(&device, &mut state, &why);
                }
            }
            What::Dispatch {
                dispatch,
                programs: stages,
            } => {
                let Some(name) = stages.get("cs") else { continue };
                used.entry(name.clone()).or_insert(None);
                if let Err(why) = dispatch_step(programs, &mut device, &state, name, *dispatch) {
                    if let Some(slot) = used.get_mut(name) {
                        *slot = Some(why.clone());
                    }
                    let uavs: Vec<(Kind, u32)> = device
                        .resources
                        .keys()
                        .filter(|k| k.0 == Kind::Uav)
                        .copied()
                        .collect();
                    for k in uavs {
                        state.taint.insert(k, why.clone());
                    }
                }
            }
            What::DispatchMesh { programs: stages, .. } => {
                for name in stages.values() {
                    used.entry(name.clone()).or_insert(None);
                }
                let why = Outcome::NotModelled("a mesh dispatch".into());
                taint_targets(&device, &mut state, &why);
            }
            What::Probe {
                probe,
                target,
                kind,
                count,
                expect,
                ulps,
                layer,
            } => {
                let outcome = probe_step(
                    &device, &state, target, *probe, kind, *count, expect, *ulps, *layer,
                );
                probes.push(Probe {
                    line: step.line,
                    selected,
                    outcome,
                });
            }
        }
    }
    RunResult {
        file: run.file.clone(),
        runner: run.runner.clone(),
        model: run.model.clone(),
        probes,
        programs: used,
    }
}

/// A draw's vertices.
enum Shape {
    /// The runner's triangle over the target, these clip-space x and y.
    Quad(Vec<f32>),
    /// The run's vertex buffers.
    Buffers {
        topology: String,
        vertices: u32,
        instances: u32,
        first_vertex: u32,
    },
}

/// Mark every render target, the depth buffer and every unordered-access view as not known.
fn taint_targets(device: &Device, state: &mut State, why: &Outcome) {
    for key in device.resources.keys() {
        if matches!(key.0, Kind::Rtv | Kind::Dsv | Kind::Uav) {
            state.taint.insert(*key, why.clone());
        }
    }
}

fn clear_resource(r: &mut Resource, value: &[u32], stencil: Option<u32>) {
    let size = r.format.size as usize;
    if size == 0 {
        return;
    }
    let mut texel = vec![0u8; size];
    let lanes = [0, 1, 2, 3].map(|k| value.get(k).copied().unwrap_or(0));
    if r.format.class == Class::Depth {
        texel[..4].copy_from_slice(&lanes[0].to_le_bytes());
        if size >= 8
            && let Some(s) = stencil
        {
            texel[4] = s as u8;
        }
    } else {
        r.format.write(lanes, 0xf, &mut texel);
    }
    for chunk in r.data.chunks_mut(size) {
        chunk.copy_from_slice(&texel);
    }
}

/// The pixels a probe's rectangle covers.
fn rect_pixels(rect: [i64; 4]) -> impl Iterator<Item = (i64, i64)> {
    let [l, t, r, b] = rect;
    (t..b).flat_map(move |y| (l..r).map(move |x| (x, y)))
}

#[allow(clippy::too_many_arguments)]
fn probe_step(
    device: &Device,
    state: &State,
    target: &str,
    rect: [i64; 4],
    kind: &str,
    count: u32,
    expect: &[u32],
    ulps: u32,
    layer: u32,
) -> Outcome {
    let key = match kind_slot(target) {
        Ok(k) => k,
        Err(e) => return Outcome::NotModelled(e),
    };
    if let Some(why) = state.taint.get(&key) {
        return why.clone();
    }
    let Some(r) = device.resources.get(&key) else {
        return Outcome::NotModelled(format!("a probe of {target}, which the run never defined"));
    };
    let size = probe_size(kind, count);
    let step = probe_step_bytes(kind, count);
    let [w, h, _] = r.level_dims(0);
    let row = if r.dim == Dim::Buffer {
        r.data.len()
    } else {
        w as usize * r.element_size() as usize
    };
    let base = if r.dim == Dim::Buffer {
        0
    } else {
        r.subresource(0, layer)
    };
    for (x, y) in rect_pixels(rect) {
        if r.dim != Dim::Buffer && (x >= i64::from(w) || y >= i64::from(h)) {
            return Outcome::NotModelled(format!("a probe at ({x}, {y}), past {target}"));
        }
        let at = base + y as usize * row + x as usize * step;
        let Some(got) = r.data.get(at..at + size) else {
            return Outcome::Fail(format!("({x}, {y}): past the end of {target}"));
        };
        if !holds(kind, count, got, expect, ulps) {
            let mut want = Vec::new();
            for w in expect {
                want.extend_from_slice(&w.to_le_bytes());
            }
            return Outcome::Fail(format!(
                "({x}, {y}): got {}, expected {}{}",
                shown(kind, count, got),
                shown(kind, count, &want[..size.min(want.len())]),
                if ulps > 0 {
                    format!(" within {ulps} ulps")
                } else {
                    String::new()
                }
            ));
        }
    }
    Outcome::Pass
}

/// The interface's resources this runner cannot bind yet, by name.
fn unbindable(interface: &Interface) -> Option<String> {
    for r in &interface.resources {
        if r.count != 1 && !matches!(r.kind, ResourceKind::ConstantBuffer { .. }) {
            return Some(format!("{} is an array of views", r.provenance.name));
        }
    }
    None
}

/// Why a draw or dispatch whose programs have these interfaces cannot be known: a view they read holds what a step
/// before could not run, and so does all they write.
fn unknown_read(device: &Device, state: &State, interfaces: &[&Interface]) -> Result<(), Outcome> {
    match device
        .views_read(interfaces)
        .iter()
        .find_map(|k| state.taint.get(k))
    {
        Some(why) => Err(why.clone()),
        None => Ok(()),
    }
}

/// Where two devices' render targets and depth buffers differ, after the same draw by two forms of its programs: the
/// first texel that does, in words; `None` where none does. A float target's lanes are alike where both are NaN
/// (IEEE 754 states no NaN's payload but through the sign operations, and two forms may keep different ones).
fn drawn_otherwise(a: &Device, b: &Device) -> Option<String> {
    for (key, r) in a
        .resources
        .iter()
        .filter(|(k, _)| matches!(k.0, Kind::Rtv | Kind::Dsv))
    {
        let Some(other) = b.resources.get(key) else {
            return Some(format!("{key:?} is only in the first"));
        };
        let size = (r.format.size as usize).max(1);
        let float = r.format.class == Class::Float;
        for (n, (x, y)) in r.data.chunks(size).zip(other.data.chunks(size)).enumerate() {
            let alike = x == y
                || float
                    && x.len() == y.len()
                    && x.chunks(4).zip(y.chunks(4)).all(|(p, q)| {
                        let word =
                            |w: &[u8]| f32::from_bits(u32::from_le_bytes(w.try_into().unwrap_or([0; 4])));
                        p == q || word(p).is_nan() && word(q).is_nan()
                    });
            if !alike {
                return Some(format!(
                    "{key:?} texel {n}: {:x?} and {:x?}",
                    r.format.read(x),
                    other.format.read(y)
                ));
            }
        }
    }
    None
}

/// Run a draw, its programs in `form`.
#[allow(clippy::too_many_arguments)]
fn draw_step(
    programs: &Programs<'_>,
    device: &mut Device,
    state: &mut State,
    stages: &BTreeMap<String, String>,
    shape: &Shape,
    d3d9: bool,
    flushed: bool,
    form: Form,
) -> Result<(), Outcome> {
    if let Some((name, value)) = state.unmodelled.iter().next() {
        return Err(Outcome::NotModelled(format!("the state {name} {value}")));
    }
    if stages.contains_key("hs") != stages.contains_key("ds") {
        return Err(Outcome::NotModelled(
            "a hull program without a domain program, or one without the other".into(),
        ));
    }
    let vs_name = stages
        .get("vs")
        .ok_or_else(|| Outcome::NotModelled("a draw with no vertex program".into()))?;
    let ps_name = stages
        .get("ps")
        .ok_or_else(|| Outcome::NotModelled("a draw with no pixel program".into()))?;
    let vs = ready(programs, vs_name)?;
    let ps = ready(programs, ps_name)?;
    let shader = |r: &'_ Ready, name: &str| -> Result<(), Outcome> {
        r.shader(form, flushed)
            .ok_or_else(|| Outcome::Error(name.to_owned(), format!("no {form:?} form")))?
            .as_ref()
            .map(|_| ())
            .map_err(|e| Outcome::Error(name.to_owned(), e.clone()))
    };
    shader(&vs, vs_name)?;
    shader(&ps, ps_name)?;
    let (Some(Ok(vsh)), Some(Ok(psh))) = (vs.shader(form, flushed), ps.shader(form, flushed)) else {
        return Err(Outcome::Error(vs_name.clone(), "no shader".into()));
    };
    // the stages between the vertex and the pixel program, from their lifted programs
    let mut between = Vec::new();
    for stage in ["hs", "ds", "gs"] {
        if let Some(name) = stages.get(stage) {
            between.push((stage, name, ready(programs, name)?));
        }
    }
    let program = |stage: &str| {
        between
            .iter()
            .find(|b| b.0 == stage)
            .map(|b| (b.1, &b.2.lifted.program))
    };
    let gs = program("gs")
        .map(|(name, p)| GeometryShader::new(Program::clone(p)).map_err(|e| Outcome::Error(name.clone(), e)))
        .transpose()?;
    let tessellation = match (program("hs"), program("ds")) {
        (Some((name, hull)), Some((_, domain))) => Some(
            TessellationShader::new(Program::clone(hull), Program::clone(domain))
                .map_err(|e| Outcome::Error(name.clone(), e))?,
        ),
        _ => None,
    };
    let mut interfaces = vec![vsh.interface(), psh.interface()];
    interfaces.extend(between.iter().map(|b| &b.2.lifted.program.interface));
    unknown_read(device, state, &interfaces)?;
    for (r, name) in [(&vs, vs_name), (&ps, ps_name)]
        .into_iter()
        .chain(between.iter().map(|b| (&b.2, b.1)))
    {
        if let Some(why) = unbindable(&r.lifted.program.interface) {
            return Err(Outcome::NotModelled(format!("{name}: {why}")));
        }
    }
    let staged = gs.is_some() || tessellation.is_some();
    let linked = if staged {
        None
    } else {
        // Direct3D 9 reads a pixel input the vertex program leaves unwritten; shader models 1 and 2 start their
        // colour registers white
        let linkage = if d3d9 {
            Linkage::Direct3D9 {
                colours_white: shader_model(&vs.lifted.program.provenance.target).is_some_and(|m| m < 3),
            }
        } else {
            Linkage::Strict
        };
        Some(
            DrawShader::with_linkage(vsh, psh, linkage)
                .map_err(|e| Outcome::Error(ps_name.clone(), e.to_string()))?,
        )
    };
    // the vertices and how they make primitives
    let mut draws: Vec<(Vec<Vec<[u32; 4]>>, Assembly)> = Vec::new();
    match shape {
        Shape::Quad(xy) => {
            let mut layout_device = device.clone();
            let mut data = Vec::new();
            for v in xy {
                data.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            let quad = Resource {
                kind: Kind::Vb,
                slot: 0,
                format: named("unknown").map_err(Outcome::NotModelled)?,
                dim: Dim::Buffer,
                width: data.len() as u32,
                height: 1,
                depth: 1,
                layers: 1,
                levels: 1,
                samples: 1,
                stride: None,
                raw: false,
                swizzle: [0, 1, 2, 3].map(crate::device::resource::Swizzle::Component),
                min_lod_clamp: 0.0,
                data,
                counter: Counter::None,
                default: true,
            };
            layout_device.resources.insert((Kind::Vb, 0), quad);
            let layout = vec![InputElement {
                slot: 0,
                format: "r32g32-float".into(),
                size: 8,
                name: "position".into(),
                index: 0,
            }];
            let vertices = (0..QUAD_VERTICES as u32)
                .map(|i| vertex_registers(vsh, &layout_device, &layout, 0, i, 0))
                .collect::<Result<Vec<_>, _>>()?;
            draws.push((vertices, Assembly::Primitives(Topology::TriangleList)));
        }
        Shape::Buffers {
            topology,
            vertices,
            instances,
            first_vertex,
        } => {
            let assembly = assembly_of(topology)?;
            for instance in 0..*instances {
                let vs_inputs = (0..*vertices)
                    .map(|i| vertex_registers(vsh, device, &state.input_layout, *first_vertex, i, instance))
                    .collect::<Result<Vec<_>, _>>()?;
                draws.push((vs_inputs, assembly));
            }
        }
    }
    // the targets
    let targets: Vec<(u32, (u32, u32))> = device
        .resources
        .iter()
        .filter(|(k, _)| k.0 == Kind::Rtv)
        .map(|(k, r)| (k.1, (r.width, r.height)))
        .collect();
    let dsv = device.resources.get(&(Kind::Dsv, 0)).map(|r| (r.width, r.height));
    let width = targets
        .iter()
        .map(|t| t.1.0)
        .chain(dsv.map(|d| d.0))
        .min()
        .unwrap_or(1);
    let height = targets
        .iter()
        .map(|t| t.1.1)
        .chain(dsv.map(|d| d.1))
        .min()
        .unwrap_or(1);
    for (_, (w, h)) in &targets {
        if (*w, *h) != (width, height) {
            return Err(Outcome::NotModelled("render targets of different sizes".into()));
        }
    }
    // the slices of the targets and the depth buffer, alike (the first level of each, which come first in a
    // resource's data)
    let layers = device
        .resources
        .iter()
        .filter(|(k, _)| k.0 == Kind::Rtv || k.0 == Kind::Dsv)
        .map(|(_, r)| r.layers)
        .collect::<std::collections::BTreeSet<u32>>();
    if layers.len() > 1 {
        return Err(Outcome::NotModelled("targets of different array sizes".into()));
    }
    let layers = layers.first().copied().unwrap_or(1).max(1);
    let texel_count = (width as usize) * (height as usize) * (layers as usize);
    let mut surface = Surface {
        width,
        height,
        layers,
        targets: Vec::new(),
        depth: None,
    };
    let rt_count = targets.iter().map(|t| t.0 + 1).max().unwrap_or(0) as usize;
    surface.targets = vec![None; rt_count];
    for (slot, _) in &targets {
        let r = &device.resources[&(Kind::Rtv, *slot)];
        if r.samples > 1 {
            return Err(Outcome::NotModelled("a multisampled target".into()));
        }
        let size = r.format.size as usize;
        let read = r
            .data
            .chunks(size)
            .take(texel_count)
            .map(|t| r.format.read(t))
            .collect();
        surface.targets[*slot as usize] = Some(read);
    }
    let depth = match device.resources.get(&(Kind::Dsv, 0)) {
        Some(r) => {
            surface.depth = Some(
                r.data
                    .chunks(r.format.size as usize)
                    .take(texel_count)
                    .map(|t| r.format.depth(t))
                    .collect(),
            );
            let compare = state
                .depth
                .ok_or_else(|| Outcome::NotModelled("a depth buffer with no depth function".into()))?;
            Some(DepthState {
                compare: draw_compare(compare),
                write: true,
            })
        }
        None => None,
    };
    // the viewports by index (one the test does not set has no area, as Direct3D's), the target's whole by default.
    // Direct3D 9 samples pixel (x, y) at (x, y), where a later API samples it at (x + 1/2, y + 1/2), so a Direct3D 9
    // viewport is half a pixel right and down of the one set (as wined3d offsets its positions); the IR's vPos is
    // the window position half a pixel back again, the pixel's whole coordinates.
    let centre = if d3d9 { 0.5 } else { 0.0 };
    let last = state.viewports.keys().max().map_or(0, |&i| i as usize + 1);
    let viewports = (0..last.max(1))
        .map(|i| {
            let [x, y, width, height] = match state.viewports.get(&(i as u32)) {
                Some(&rect) => rect,
                None if state.viewports.is_empty() => [0.0, 0.0, width as f32, height as f32],
                None => [0.0; 4],
            };
            Viewport {
                x: x + centre,
                y: y + centre,
                width,
                height,
                min_depth: 0.0,
                max_depth: 1.0,
            }
        })
        .collect();
    let rasterizer = Rasterizer {
        viewports,
        sample_mask: state.sample_mask.unwrap_or(u32::MAX),
        cull: Cull::None,
        front_counter_clockwise: false,
        depth,
        alpha_test: state.alpha_test,
        clip_planes: state.clip_planes,
        point_size: state.point_size,
        point_sprite: state.point_sprite,
        flat_colours: d3d9 && state.flat,
        // shader models 1 and 2 pass colours through registers of [0, 1] (`oD#`), 3 through general ones
        clamp_colours: d3d9 && shader_model(&vs.lifted.program.provenance.target).is_some_and(|m| m < 3),
    };
    let vs_constants = constant_rows(vsh, device);
    let ps_constants = constant_rows(psh, device);
    let vs_banks: Vec<Bank<'_>> = vs_constants.iter().map(|(s, r)| (*s, r.as_slice())).collect();
    let ps_banks: Vec<Bank<'_>> = ps_constants.iter().map(|(s, r)| (*s, r.as_slice())).collect();
    // the constant buffers of the stages between, by register (space 0: the runner binds no other)
    let words = device.constants.clone();
    let rows = move |register: u32, space: u32, n: usize| -> Vec<[u32; 4]> {
        let w = words
            .get(&register)
            .filter(|_| space == 0)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        (0..n)
            .map(|r| [0, 1, 2, 3].map(|l| w.get(r * 4 + l).copied().unwrap_or(0)))
            .collect()
    };
    let staged = Stages {
        vs: vsh,
        tessellation: tessellation.as_ref(),
        gs: gs.as_ref(),
        ps: psh,
    };
    if d3d9 {
        for slot in psh.samplers().iter().chain(vsh.samplers()) {
            device.samplers.entry(*slot).or_insert_with(d3d9_default_sampler);
        }
    }
    let mut requests = SlotRequests::new(device, &interfaces).map_err(Outcome::NotModelled)?;
    let mut written = 0usize;
    for (vertices, assembly) in &draws {
        let stats = match &linked {
            Some(linked) => {
                let primitives = primitives_of(*assembly, vertices.len())?;
                linked
                    .draw(
                        vertices,
                        &primitives,
                        &vs_banks,
                        &ps_banks,
                        &mut requests,
                        &rasterizer,
                        &mut surface,
                    )
                    .map_err(|e| Outcome::Error(ps_name.clone(), e.to_string()))?
            }
            None => staged
                .draw(
                    vertices,
                    *assembly,
                    [&vs_banks, &ps_banks],
                    &rows,
                    &mut requests,
                    &rasterizer,
                    &mut surface,
                )
                .map_err(|e| match e {
                    StageStop::NotModelled(m) => Outcome::NotModelled(m),
                    StageStop::Failed(stage, m) => {
                        Outcome::Error(stages.get(stage).cloned().unwrap_or_default(), m)
                    }
                })?,
        };
        written += stats.written;
    }
    let device = requests.device;
    // the targets back into the resources
    for (slot, texels) in surface.targets.iter().enumerate() {
        let Some(texels) = texels else { continue };
        let Some(r) = device.resources.get_mut(&(Kind::Rtv, slot as u32)) else {
            continue;
        };
        let size = r.format.size as usize;
        let format = r.format;
        for (t, bytes) in texels.iter().zip(r.data.chunks_mut(size)) {
            format.write(*t, 0xf, bytes);
        }
        if written >= texel_count {
            state.taint.remove(&(Kind::Rtv, slot as u32));
        }
    }
    if let (Some(depths), Some(r)) = (&surface.depth, device.resources.get_mut(&(Kind::Dsv, 0))) {
        let size = r.format.size as usize;
        for (d, bytes) in depths.iter().zip(r.data.chunks_mut(size)) {
            bytes[..4].copy_from_slice(&d.to_bits().to_le_bytes());
        }
    }
    Ok(())
}

/// Run a compute dispatch.
fn dispatch_step(
    programs: &Programs<'_>,
    device: &mut Device,
    state: &State,
    name: &str,
    groups: [u32; 3],
) -> Result<(), Outcome> {
    let cs = ready(programs, name)?;
    let exe = cs
        .executable
        .as_ref()
        .map_err(|e| Outcome::Error(name.to_owned(), e.clone()))?;
    let interface = &cs.lifted.program.interface;
    unknown_read(device, state, &[interface])?;
    let resources = crate::device::constant_buffers(interface, |register, space, rows| {
        (space == 0 && device.constants.contains_key(&register)).then(|| device.constant_rows(register, rows))
    });
    crate::device::dispatch(exe, interface, device, &resources, groups, WAVE).map_err(|e| match e {
        Dispatch::Unbindable(m) => Outcome::NotModelled(format!("{name}: {m}")),
        Dispatch::Failed(m) => Outcome::Error(name.to_owned(), m),
    })
}

/// Run every run of a file, `threads` at a time.
pub fn run_all(file: &File, threads: usize) -> (Vec<RunResult>, Lifts) {
    let programs = Programs::new(|name: &str| super::cases::program(file, name));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(r) = file.cases.get(i) else { break };
                    let done = run(&programs, r);
                    if let Ok(mut all) = results.lock() {
                        all.push((i, done));
                    }
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap_or_default();
    results.sort_by_key(|(i, _)| *i);
    (results.into_iter().map(|(_, r)| r).collect(), programs.lifted())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probes_of(outcomes: &[Outcome]) -> Vec<Probe> {
        outcomes
            .iter()
            .zip(1..)
            .map(|(outcome, line)| Probe {
                line,
                selected: true,
                outcome: outcome.clone(),
            })
            .collect()
    }

    #[test]
    fn a_run_is_classed_by_the_probes_vkd3d_runs() {
        let fail = || Outcome::Fail("differs".into());
        let run = |probes: Vec<Probe>| RunResult {
            file: "t.shader_test".into(),
            runner: "d3d12".into(),
            model: "5.1".into(),
            probes,
            programs: BTreeMap::new(),
        };
        // a selected probe holds and an extended one does not: the run passes, its extended probes fail
        let mut probes = probes_of(&[Outcome::Pass, fail()]);
        probes[1].selected = false;
        let r = run(probes);
        assert!(r.selected());
        assert_eq!((r.class(), r.extended_class()), ("pass", Some("fail")));
        // a selected probe fails: the run fails, whatever the extended ones
        let mut probes = probes_of(&[fail(), Outcome::Pass]);
        probes[1].selected = false;
        let r = run(probes);
        assert_eq!((r.class(), r.extended_class()), ("fail", Some("pass")));
        // vkd3d runs none of the run's probes: classed by all of them, none apart
        let mut probes = probes_of(&[Outcome::Pass, fail()]);
        probes.iter_mut().for_each(|p| p.selected = false);
        let r = run(probes);
        assert!(!r.selected());
        assert_eq!((r.class(), r.extended_class()), ("fail", None));
    }

    #[test]
    fn a_flushed_run_is_taken_only_where_it_breaks_nothing() {
        let fail = || Outcome::Fail("differs".into());
        // it mends a probe and breaks none: taken, the mended probe saying so
        let mut kept = probes_of(&[Outcome::Pass, fail(), fail()]);
        assert!(adopt_flushed(
            &mut kept,
            probes_of(&[Outcome::Pass, Outcome::Pass, fail()])
        ));
        let got: Vec<_> = kept.iter().map(|p| p.outcome.clone()).collect();
        assert_eq!(got, [Outcome::Pass, Outcome::PassFlushed, fail()]);
        // it mends one and breaks one: one rule for the run, so not taken
        let mut kept = probes_of(&[Outcome::Pass, fail()]);
        assert!(!adopt_flushed(&mut kept, probes_of(&[fail(), Outcome::Pass])));
        assert_eq!(kept[1].outcome, fail());
        // it mends none
        let mut kept = probes_of(&[Outcome::Pass, fail()]);
        assert!(!adopt_flushed(&mut kept, probes_of(&[Outcome::Pass, fail()])));
    }
}
