//! Runs the offload-test-suite's tests as lit runs them for DirectX with DXC: each RUN line in order, the offloader's
//! part (`offload`) on the CPU [`Device`] through `iridia-exec`, and what the line pipes its output to (FileCheck,
//! imgdiff) as those tools would. A test's REQUIRES, UNSUPPORTED and XFAIL lines are taken against [`features`], the
//! device iridia's executor is.
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use crate::shard::Shard;
use iridia::{Interface, ResourceKind as IrKind, SystemValue};
use iridia_exec::amplification::AmplificationShader;
use iridia_exec::draw::{
    Compare as DepthCompare, Cull, DepthState, DrawShader, Primitive, Rasterizer, Surface,
    Viewport as DrawViewport,
};
use iridia_exec::geometry::Topology;
use iridia_exec::mesh::MeshShader;
use iridia_exec::{PixelShader, ProgramShader};
use serde_json::Value;
use sha2::Digest as _;

use super::cases::{Case, File};
use super::features::{applies, features, not_for_direct3d};
use super::filecheck::{self, Directive, Kind as CheckKind};
use super::image;
use super::pipeline::{
    self, Buffer, DataFormat, DescriptorKind, Pipeline, PipelineKind, ResourceKind, RootParameter,
};
use super::rts0::{self, Parameter, RangeKind, RootSignature};
use crate::device::format::{Format, named};
use crate::device::resource::{Counter, Dim, Kind, Resource, Swizzle};
use crate::device::sample::{Address, Compare, Reduction, Sampler};
use crate::device::{Device, Dispatch, SlotRequests};
use crate::front::{self, Code, Programs, Ready, Stop};
use crate::stages::{Assembly, Stages, Stopped as StageStop, mesh_draw};
use crate::vkd3d::cases::Descriptor;

/// The wave width compute programs run with (the device's `WaveLaneCountMin` and `Max`).
pub const WAVE: u32 = 32;

/// The device's description, as `Configuring execution on device:` prints it.
const DESCRIPTION: &str = "iridia-exec (CPU reference executor)";

/// What a test came to.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum Outcome {
    /// Every RUN line did what the test expects.
    Pass,
    /// A line failed, and the test is expected to fail here (XFAIL).
    XFail(String),
    /// A line failed.
    Fail(String),
    /// Every line passed where the test expects a failure (XPASS, which lit counts as a failure).
    XPass,
    /// lit would run it on no Direct3D 12 device in this configuration ([`not_for_direct3d`]): it is for another API
    /// or another compiler.
    NotApplicable(String),
    /// iridia's lifter or executor refused one of its programs.
    Refused(String),
    /// It asks something of the offloader this runner does not model, or a feature a Direct3D 12 device can have and
    /// iridia's does not (a `REQUIRES` or `UNSUPPORTED` lit decides by [`features`]).
    NotModelled(String),
    /// dxc refused to compile a shader the test expects to compile.
    CompilerRefused(String),
}

impl Outcome {
    /// Its class's name, as the table shows it.
    pub fn class(&self) -> &'static str {
        match self {
            Outcome::Pass => "pass",
            Outcome::XFail(_) => "xfail",
            Outcome::Fail(_) => "fail",
            Outcome::XPass => "xpass",
            Outcome::NotApplicable(_) => "not-applicable",
            Outcome::Refused(_) => "refused",
            Outcome::NotModelled(_) => "not-modelled",
            Outcome::CompilerRefused(_) => "compiler-refused",
        }
    }

    /// Why, in a line.
    pub fn detail(&self) -> String {
        match self {
            Outcome::Pass | Outcome::XPass => String::new(),
            Outcome::XFail(m)
            | Outcome::Fail(m)
            | Outcome::NotApplicable(m)
            | Outcome::Refused(m)
            | Outcome::NotModelled(m)
            | Outcome::CompilerRefused(m) => m.lines().next().unwrap_or("").to_string(),
        }
    }
}

/// A test's result.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TestResult {
    /// The test's path under test/.
    pub test: String,
    /// What it came to.
    pub outcome: Outcome,
    /// The programs it ran, by digest.
    pub programs: Vec<String>,
    /// How long it took ([`run_all_with`] times it): not kept in a part, as it differs from run to run.
    #[serde(skip)]
    pub took: std::time::Duration,
}

/// Why a RUN line could not be taken as the real tools would take it.
enum Stopped {
    /// The program `0` was refused by iridia.
    Refused(String),
    /// A part of the offloader this runner does not model.
    NotModelled(String),
}

/// What the offloader did: its exit status and output.
struct Offloaded {
    ok: bool,
    stdout: String,
    stderr: String,
    /// The render target `-r` names, as 16-bit RGBA.
    image: Option<image::Rgba16>,
}

fn not_modelled(m: impl Into<String>) -> Stopped {
    Stopped::NotModelled(m.into())
}

/// The program `name`, lifted and made runnable.
fn ready(programs: &Programs<'_>, name: &str) -> Result<Arc<Ready>, Stopped> {
    programs.get(name).map_err(|e| match e {
        Stop::Refused(m) => Stopped::Refused(format!("{name}: {m}")),
        Stop::Failed(m) => Stopped::Refused(format!("{name}: {m}")),
    })
}

/// A view the programs read or write: the resource and element of the pipeline it shows.
#[derive(Clone, Debug)]
struct View {
    class: DescriptorKind,
    register: u32,
    space: u32,
    buffer: String,
    element: usize,
    kind: ResourceKind,
    /// Whether it has a hidden counter (`HasCounter`).
    counted: bool,
}

/// Where everything the pipeline binds is, by register.
#[derive(Clone, Debug, Default)]
struct Layout {
    views: Vec<View>,
    /// Samplers: register, space, the sampler.
    samplers: Vec<(u32, u32, Sampler)>,
    /// Root constants: register, space, the words.
    constants: Vec<(u32, u32, Vec<u32>)>,
}

/// One descriptor of the heap the offloader fills: a view, or (in the sampler heap) a sampler.
#[derive(Clone, Debug)]
enum Slot {
    /// A view: the buffer, the element of it, the kind, whether it has a counter.
    View(String, usize, ResourceKind, bool),
    Sampler(String),
}

/// The two descriptor heaps as `buildDescriptorTables` fills them: every resource of every set in order, an array
/// element by element, samplers in their own heap; a resource with a `HeapIndex` at that index.
fn heaps(p: &Pipeline) -> (BTreeMap<u32, Slot>, BTreeMap<u32, Slot>) {
    let (mut views, mut samplers) = (BTreeMap::new(), BTreeMap::new());
    let (mut next_view, mut next_sampler) = (0u32, 0u32);
    for set in &p.sets {
        for r in set {
            if r.kind == ResourceKind::Sampler {
                let at = r.heap_index.unwrap_or_else(|| {
                    next_sampler += 1;
                    next_sampler - 1
                });
                samplers.insert(at, Slot::Sampler(r.name.clone()));
                continue;
            }
            let count = p.buffer(&r.name).map_or(1, |b| b.array_size);
            for e in 0..count {
                // every element of an array with a heap index goes to that one index, as the offloader writes it
                let at = r.heap_index.unwrap_or_else(|| {
                    next_view += 1;
                    next_view - 1
                });
                views.insert(at, Slot::View(r.name.clone(), e as usize, r.kind, r.has_counter));
            }
        }
    }
    (views, samplers)
}

fn sampler_of(p: &Pipeline, name: &str) -> Result<Sampler, Stopped> {
    let s = p
        .samplers
        .iter()
        .find(|s| s.name == name)
        .ok_or_else(|| not_modelled(format!("the sampler {name}")))?;
    let address = match s.address.as_str() {
        "Clamp" => Address::Clamp,
        "Repeat" => Address::Wrap,
        "Mirror" => Address::Mirror,
        "Border" => Address::Border,
        _ => Address::MirrorOnce,
    };
    let compare = |op: &str| match op {
        "Never" => Compare::Never,
        "Less" => Compare::Less,
        "Equal" => Compare::Equal,
        "LessEqual" => Compare::LessEqual,
        "Greater" => Compare::Greater,
        "NotEqual" => Compare::NotEqual,
        "GreaterEqual" => Compare::GreaterEqual,
        _ => Compare::Always,
    };
    Ok(Sampler {
        min_linear: s.min_filter == "Linear",
        mag_linear: s.mag_filter == "Linear",
        // getDXFilterMode always filters mips by point
        mip_linear: false,
        reduction: if s.kind == "SamplerComparison" {
            Reduction::Comparison(compare(&s.comparison_op))
        } else {
            Reduction::Standard
        },
        address: [address; 3],
        border: [0.0; 4],
        bias: s.mip_lod_bias,
        lod_range: [s.min_lod, s.max_lod],
    })
}

/// A static sampler of a root signature as a sampler (`D3D12_FILTER` bits: min 4, mag 2, mip 0; reduction 7).
fn static_sampler(s: &rts0::StaticSampler) -> Sampler {
    let address = |a: u32| match a {
        1 => Address::Wrap,
        2 => Address::Mirror,
        3 => Address::Clamp,
        4 => Address::Border,
        _ => Address::MirrorOnce,
    };
    let compare = match s.comparison {
        1 => Compare::Never,
        2 => Compare::Less,
        3 => Compare::Equal,
        4 => Compare::LessEqual,
        5 => Compare::Greater,
        6 => Compare::NotEqual,
        7 => Compare::GreaterEqual,
        _ => Compare::Always,
    };
    let reduction = match (s.filter >> 7) & 3 {
        1 => Reduction::Comparison(compare),
        2 => Reduction::Minimum,
        3 => Reduction::Maximum,
        _ => Reduction::Standard,
    };
    // D3D12_STATIC_BORDER_COLOR: transparent black, opaque black, opaque white (and their uint forms)
    let border = match s.border {
        1 | 3 => [0.0, 0.0, 0.0, 1.0],
        2 | 4 => [1.0; 4],
        _ => [0.0; 4],
    };
    Sampler {
        min_linear: s.filter >> 4 & 1 == 1,
        mag_linear: s.filter >> 2 & 1 == 1,
        mip_linear: s.filter & 1 == 1,
        reduction,
        address: s.address.map(address),
        border,
        bias: s.mip_lod_bias,
        lod_range: [s.min_lod, s.max_lod],
    }
}

/// The views of one descriptor table: its ranges over the heap from `base` (`D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND`
/// following the previous range, an unbounded range running to the heap's end).
fn table_views(
    ranges: &[rts0::Range],
    base: u32,
    heap: &BTreeMap<u32, Slot>,
    out: &mut Layout,
    p: &Pipeline,
) -> Result<(), Stopped> {
    let end = heap.keys().next_back().map_or(0, |k| k + 1);
    let mut cursor = 0u32;
    for r in ranges {
        let start = if r.offset == u32::MAX { cursor } else { r.offset };
        let count = if r.count == u32::MAX {
            end.saturating_sub(base + start)
        } else {
            r.count
        };
        for j in 0..count {
            match heap.get(&(base + start + j)) {
                Some(Slot::View(name, element, kind, counted)) if r.kind != RangeKind::Sampler => {
                    out.views.push(View {
                        class: kind.descriptor(),
                        register: r.base + j,
                        space: r.space,
                        buffer: name.clone(),
                        element: *element,
                        kind: *kind,
                        counted: *counted,
                    })
                }
                Some(Slot::Sampler(name)) if r.kind == RangeKind::Sampler => {
                    out.samplers.push((r.base + j, r.space, sampler_of(p, name)?));
                }
                _ => {}
            }
        }
        cursor = start.saturating_add(count);
    }
    Ok(())
}

/// Where the pipeline's resources are by register, as the offloader binds them: with no root signature in the program
/// by each resource's `DirectXBinding` (an array's elements at consecutive registers); with one, through its
/// descriptor tables, root constants, root descriptors and static samplers.
fn layout(p: &Pipeline, rs: Option<&RootSignature>) -> Result<Layout, Stopped> {
    let mut out = Layout::default();
    let Some(rs) = rs else {
        if !p.root_parameters.is_empty() {
            return Err(not_modelled(
                "RootParameters for a program with no root signature",
            ));
        }
        for set in &p.sets {
            for r in set {
                let Some((register, space)) = r.dx_binding else {
                    return Err(not_modelled(format!("{}: a resource by HeapIndex only", r.name)));
                };
                if r.kind == ResourceKind::Sampler {
                    out.samplers.push((register, space, sampler_of(p, &r.name)?));
                    continue;
                }
                let count = p.buffer(&r.name).map_or(1, |b| b.array_size);
                for e in 0..count {
                    out.views.push(View {
                        class: r.kind.descriptor(),
                        register: register + e,
                        space,
                        buffer: r.name.clone(),
                        element: e as usize,
                        kind: r.kind,
                        counted: r.has_counter,
                    });
                }
            }
        }
        return Ok(out);
    };
    let (view_heap, sampler_heap) = heaps(p);
    for s in &rs.samplers {
        out.samplers.push((s.register, s.space, static_sampler(s)));
    }
    if p.root_parameters.is_empty() {
        let (mut handle, mut sampler_handle) = (0u32, 0u32);
        for param in &rs.parameters {
            let Parameter::Table(ranges) = param else {
                return Err(not_modelled(
                    "Root Signatures that contain constants and inline descriptors require custom RootParameters to be \
                     defined.",
                ));
            };
            let count: u32 = ranges.iter().map(|r| r.count).fold(0, u32::saturating_add);
            if ranges.first().is_some_and(|r| r.kind == RangeKind::Sampler) {
                table_views(ranges, sampler_handle, &sampler_heap, &mut out, p)?;
                sampler_handle = sampler_handle.saturating_add(count);
            } else {
                table_views(ranges, handle, &view_heap, &mut out, p)?;
                handle = handle.saturating_add(count);
            }
        }
        return Ok(out);
    }
    if p.sets.iter().flatten().any(|r| r.kind == ResourceKind::Sampler) {
        return Err(not_modelled(
            "Descriptor tables containing samplers are not yet supported with explicit RootParameters.",
        ));
    }
    let (mut handle, mut table, mut constant_offset) = (0u32, 0usize, 0usize);
    for (i, param) in p.root_parameters.iter().enumerate() {
        let rsp = rs.parameters.get(i).ok_or_else(|| {
            not_modelled(format!("root parameter {i}: not in the program's root signature"))
        })?;
        match (param, rsp) {
            (RootParameter::DescriptorTable, Parameter::Table(ranges)) => {
                table_views(ranges, handle, &view_heap, &mut out, p)?;
                // the offloader steps the handle by the set's resources, not their elements
                handle += p.sets.get(table).map_or(0, Vec::len) as u32;
                table += 1;
            }
            (RootParameter::Constant(name), Parameter::Constants(register, space, count)) => {
                let b = p
                    .buffer(name)
                    .ok_or_else(|| not_modelled(format!("the buffer {name}")))?;
                if b.array_size != 1 {
                    return Err(not_modelled("Root constant cannot refer to resource arrays."));
                }
                let values: Vec<u32> = b.data[0][..b.size]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| u32::from_le_bytes(*c))
                    .collect();
                // SetComputeRoot32BitConstants at the running offset, as the offloader calls it
                let mut words = vec![0u32; *count as usize];
                for (k, v) in values.iter().enumerate() {
                    if let Some(w) = words.get_mut(constant_offset + k) {
                        *w = *v;
                    }
                }
                constant_offset += values.len();
                out.constants.push((*register, *space, words));
            }
            (RootParameter::RootDescriptor(name, kind), Parameter::Descriptor(class, register, space)) => {
                let wanted = match kind.descriptor() {
                    DescriptorKind::Srv => RangeKind::Srv,
                    DescriptorKind::Uav => RangeKind::Uav,
                    _ => RangeKind::Cbv,
                };
                if wanted != *class {
                    return Err(not_modelled(format!(
                        "root parameter {i}: a {kind:?} at a {class:?} parameter"
                    )));
                }
                if p.buffer(name).is_some_and(|b| b.array_size != 1) {
                    return Err(not_modelled("Root descriptor cannot refer to resource arrays."));
                }
                out.views.push(View {
                    class: kind.descriptor(),
                    register: *register,
                    space: *space,
                    buffer: name.clone(),
                    element: 0,
                    kind: *kind,
                    // a root descriptor has no counter (D3D12_ROOT_DESCRIPTOR names no counter resource)
                    counted: false,
                });
            }
            _ => {
                return Err(not_modelled(format!(
                    "root parameter {i}: not the kind the root signature has there"
                )));
            }
        }
    }
    Ok(out)
}

/// The device format `toFormat` gives a buffer's data format and channel count.
fn format_of(format: DataFormat, channels: i64) -> Result<Format, Stopped> {
    let name = match (format, channels) {
        (DataFormat::Float32, 1) => "r32-float",
        (DataFormat::Float32, 2) => "r32g32-float",
        (DataFormat::Float32, 3) => "r32g32b32-float",
        (DataFormat::Float32, 4) => "r32g32b32a32-float",
        (DataFormat::Int32, 1) => "r32-sint",
        (DataFormat::Int32, 2) => "r32g32-sint",
        (DataFormat::Int32, 4) => "r32g32b32a32-sint",
        (DataFormat::UInt32, 1) => "r32-uint",
        (DataFormat::UInt32, 2) => "r32g32-uint",
        (DataFormat::UInt32, 4) => "r32g32b32a32-uint",
        (DataFormat::Depth32, 1) => "d32-float",
        (f, c) => {
            return Err(not_modelled(format!(
                "a typed view of {} x {c}, a format the device does not have",
                f.name()
            )));
        }
    };
    named(name).map_err(Stopped::NotModelled)
}

/// The subresources of a texture of `b` in the offloader's order (each slice's whole mip chain, slice after slice)
/// and in the device's (each level's every layer, level after level): the byte ranges, matched.
fn subresources(
    element: usize,
    levels: u32,
    layers: u32,
    dims: impl Fn(u32) -> [u32; 3],
) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>)> {
    let level_bytes = |l: u32| {
        let [w, h, d] = dims(l);
        w as usize * h as usize * d as usize * element
    };
    let chain: usize = (0..levels).map(level_bytes).sum();
    let mut out = Vec::new();
    for s in 0..layers {
        let mut at = s as usize * chain;
        for l in 0..levels {
            let n = level_bytes(l);
            let before: usize = (0..l).map(|k| level_bytes(k) * layers as usize).sum();
            let device = before + s as usize * n;
            out.push((at..at + n, device..device + n));
            at += n;
        }
    }
    out
}

/// The device resource a view of the pipeline's buffer makes.
fn resource_of(p: &Pipeline, v: &View, kind: Kind, slot: u32) -> Result<Resource, Stopped> {
    let b = p
        .buffer(&v.buffer)
        .ok_or_else(|| not_modelled(format!("the buffer {}", v.buffer)))?;
    let data = b.data.get(v.element).cloned().unwrap_or_default();
    let unknown = named("unknown").map_err(Stopped::NotModelled)?;
    let mut r = Resource {
        kind,
        slot,
        format: unknown,
        dim: Dim::Buffer,
        width: 0,
        height: 1,
        depth: 1,
        layers: 1,
        levels: 1,
        samples: 1,
        stride: None,
        raw: false,
        swizzle: [0, 1, 2, 3].map(Swizzle::Component),
        min_lod_clamp: 0.0,
        data: Vec::new(),
        // the offloader uploads the buffer's bytes only: the counter after them starts as the zeroed resource's
        counter: if v.counted {
            Counter::Separate(0)
        } else {
            Counter::None
        },
        default: false,
    };
    use ResourceKind as K;
    match v.kind {
        K::StructuredBuffer | K::RWStructuredBuffer => {
            let stride = b.element_size().max(1);
            r.stride = Some(stride as u32);
            r.width = (b.size / stride) as u32;
            r.data = data;
        }
        K::ByteAddressBuffer | K::RWByteAddressBuffer | K::ConstantBuffer => {
            r.raw = true;
            r.width = (b.size / 4) as u32;
            r.data = data;
        }
        K::Buffer | K::RWBuffer => {
            r.format = format_of(b.format, b.channels)?;
            r.width = (b.size / r.format.size.max(1) as usize) as u32;
            r.data = data;
        }
        K::Sampler | K::AccelerationStructure | K::SampledTexture2D => {
            return Err(not_modelled(format!("a {:?} view", v.kind)));
        }
        _ => {
            r.format = format_of(b.format, b.channels)?;
            let o = b.output_props;
            r.width = o.width.max(1) as u32;
            r.height = o.height.max(1) as u32;
            r.levels = o.mip_levels.max(1) as u32;
            let slices = o.array_slices.max(1) as u32;
            match v.kind {
                K::Texture3D | K::RWTexture3D => {
                    r.dim = Dim::D3;
                    r.depth = o.depth.max(1) as u32;
                }
                K::TextureCube | K::TextureCubeArray => {
                    r.dim = Dim::Cube;
                    r.layers = slices;
                }
                _ => {
                    r.dim = Dim::D2;
                    r.layers = slices;
                }
            }
            r.data = vec![0; r.total_bytes()];
            let element = r.format.size as usize;
            for (from, to) in subresources(element, r.levels, r.layers, |l| r.level_dims(l)) {
                if let (Some(src), true) = (data.get(from), to.end <= r.data.len()) {
                    r.data[to].copy_from_slice(src);
                }
            }
        }
    }
    Ok(r)
}

/// Put a device resource's bytes back into the pipeline's buffer, as the offloader reads a UAV back.
fn read_back(p: &mut Pipeline, v: &View, r: &Resource) {
    let Some(b) = p.buffer_mut(&v.buffer) else { return };
    if v.counted {
        // `Counters`, one per element of the buffer, as `readBack` pushes them element by element
        let n = b.data.len();
        b.counters.resize(n, 0);
        if let Some(c) = b.counters.get_mut(v.element) {
            *c = r.count();
        }
    }
    let Some(data) = b.data.get_mut(v.element) else {
        return;
    };
    if r.dim == Dim::Buffer {
        let n = data.len().min(r.data.len());
        data[..n].copy_from_slice(&r.data[..n]);
        return;
    }
    let element = r.format.size as usize;
    for (to, from) in subresources(element, r.levels, r.layers, |l| r.level_dims(l)) {
        if let (Some(dst), Some(src)) = (data.get_mut(to), r.data.get(from)) {
            dst.copy_from_slice(src);
        }
    }
}

/// Constant-buffer views and root constants by register and space, as words.
type Rows = BTreeMap<(u32, u32), Vec<u32>>;

/// The device a layout makes: each view a resource at its own slot, the descriptor ranges naming them by register
/// and space; constant-buffer views and root constants by register and space, as rows.
fn device_of(p: &Pipeline, l: &Layout) -> Result<(Device, Rows), Stopped> {
    let mut device = Device {
        descriptors: Some(Vec::new()),
        ..Device::default()
    };
    let mut constants = BTreeMap::new();
    for (i, v) in l.views.iter().enumerate() {
        let slot = i as u32;
        match v.class {
            DescriptorKind::Cbv => {
                let b = p
                    .buffer(&v.buffer)
                    .ok_or_else(|| not_modelled(format!("the buffer {}", v.buffer)))?;
                let words = b
                    .data
                    .get(v.element)
                    .map(|d| {
                        d.chunks(4)
                            .map(|c| {
                                let mut w = [0u8; 4];
                                w[..c.len()].copy_from_slice(c);
                                u32::from_le_bytes(w)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                constants.insert((v.register, v.space), words);
            }
            DescriptorKind::Srv | DescriptorKind::Uav => {
                let kind = if v.class == DescriptorKind::Uav {
                    Kind::Uav
                } else {
                    Kind::Srv
                };
                let r = resource_of(p, v, kind, slot)?;
                device.resources.insert((kind, slot), r);
                if let Some(d) = device.descriptors.as_mut() {
                    d.push(Descriptor {
                        kind: if kind == Kind::Uav { "uav" } else { "srv" }.into(),
                        register: v.register,
                        count: 1,
                        space: v.space,
                        target: slot,
                    });
                }
            }
            DescriptorKind::Sampler => {}
        }
    }
    for (register, space, words) in &l.constants {
        constants.insert((*register, *space), words.clone());
    }
    for (register, space, s) in &l.samplers {
        if *space != 0 {
            return Err(not_modelled("a sampler in a register space other than 0"));
        }
        device.samplers.insert(*register, *s);
    }
    Ok((device, constants))
}

/// The interface's resources the device does not bind, by name.
fn unbindable(interface: &Interface) -> Option<String> {
    for r in &interface.resources {
        if r.count != 1 && !matches!(r.kind, IrKind::ConstantBuffer { .. }) {
            return Some(format!("{} is an array of {} views", r.provenance.name, r.count));
        }
        if r.provenance.d3d.is_none() && r.provenance.counter_of.is_none() {
            return Some(format!(
                "{} has no register (a descriptor-heap index?)",
                r.provenance.name
            ));
        }
    }
    None
}

/// Run a compute pipeline.
fn compute(
    p: &mut Pipeline,
    cs: &Ready,
    name: &str,
    rs: Option<&RootSignature>,
) -> Result<Result<(), String>, Stopped> {
    let exe = cs
        .executable
        .as_ref()
        .map_err(|e| Stopped::Refused(format!("{name}: {e}")))?;
    let interface = cs.lifted.code.interface();
    let l = layout(p, rs)?;
    let (mut device, constants) = device_of(p, &l)?;
    let resources = crate::device::constant_buffers(interface, |register, space, rows| {
        let words = constants.get(&(register, space))?;
        Some(
            (0..rows)
                .map(|row| [0, 1, 2, 3].map(|k| words.get(row * 4 + k).copied().unwrap_or(0)))
                .collect(),
        )
    });
    match crate::device::dispatch(exe, interface, &mut device, &resources, p.dispatch, WAVE) {
        Ok(()) => {}
        Err(Dispatch::Unbindable(m)) => return Err(not_modelled(format!("{name}: {m}"))),
        Err(Dispatch::Failed(m)) => return Ok(Err(format!("{name}: {m}"))),
    }
    for (i, v) in l.views.iter().enumerate() {
        if v.class == DescriptorKind::Uav
            && let Some(r) = device.resources.get(&(Kind::Uav, i as u32))
        {
            read_back(p, v, r);
        }
    }
    Ok(Ok(()))
}

/// The input registers of vertex `vertex` for a vertex program, from the pipeline's vertex buffer: each attribute at
/// its offset in a vertex of the attributes' sizes added up (`getVertexStride`); an input no attribute feeds reads
/// 0 0 0 1.
fn vertex_inputs(vsh: &ProgramShader, p: &Pipeline, vertex: u32) -> Result<Vec<[u32; 4]>, Stopped> {
    let interface = vsh.interface();
    let count = interface
        .inputs
        .iter()
        .filter_map(|v| v.provenance.d3d.map(|d| d.register as usize + 1))
        .max()
        .unwrap_or(0);
    let mut registers = vec![[0u32; 4]; count.max(vsh.input_count())];
    let stride = p.bindings.vertex_stride();
    let vb = p.buffer(&p.bindings.vertex_buffer).and_then(|b| b.data.first());
    for v in &interface.inputs {
        let Some(d) = v.provenance.d3d else { continue };
        let lanes = match v.system {
            Some(SystemValue::VertexIndex) => [vertex, 0, 0, 0],
            Some(SystemValue::InstanceIndex) => [0; 4],
            Some(other) => {
                return Err(not_modelled(format!(
                    "a vertex input of the system value {other:?}"
                )));
            }
            None => {
                let full = format!("{}{}", v.label.name, v.label.index);
                let attr = p.bindings.vertex_attributes.iter().find(|a| {
                    a.name.eq_ignore_ascii_case(&full)
                        || (v.label.index == 0 && a.name.eq_ignore_ascii_case(&v.label.name))
                });
                match (attr, vb) {
                    (Some(a), Some(data)) => {
                        let size = a.format.size();
                        let mut out = [0, 0, 0, 1f32.to_bits()];
                        if !a.format.is_float() {
                            out[3] = 1;
                        }
                        for (k, lane) in out.iter_mut().enumerate().take(a.channels.clamp(0, 4) as usize) {
                            let at = vertex as usize * stride + a.offset as usize + k * size;
                            let Some(bytes) = data.get(at..at + size) else {
                                continue;
                            };
                            *lane = match (a.format, size) {
                                (DataFormat::Float16, 2) => {
                                    half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]))
                                        .to_f32()
                                        .to_bits()
                                }
                                (_, 4) => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                                _ => {
                                    return Err(not_modelled(format!(
                                        "a vertex attribute of {} x {}",
                                        a.format.name(),
                                        a.channels
                                    )));
                                }
                            };
                        }
                        out
                    }
                    _ => [0, 0, 0, 1f32.to_bits()],
                }
            }
        };
        let reg = &mut registers[d.register as usize];
        for (k, l) in (0..4).filter(|l| d.mask >> l & 1 == 1).enumerate() {
            reg[l] = lanes[k];
        }
    }
    Ok(registers)
}

/// Run a raster pipeline of a vertex and a pixel program, with hull and domain programs and a geometry program where
/// it names them (through [`Stages`]), as the offloader's D3D12 pipeline state has it: no culling, counter-clockwise
/// triangles front facing, a depth buffer cleared to 1 and tested `less`, one viewport over the render target, the
/// target loaded with its buffer's contents, and a draw of `VertexCount` vertices (or the vertex buffer's) in the
/// pipeline's topology.
fn raster(
    p: &mut Pipeline,
    ready: &[(String, Arc<Ready>)],
    rs: Option<&RootSignature>,
) -> Result<Result<(), String>, Stopped> {
    let stage = |s: &str| {
        p.shaders
            .iter()
            .zip(ready)
            .find(|(sh, _)| sh.stage == s)
            .map(|(_, r)| r.clone())
    };
    let (Some((vs_name, vs)), Some((ps_name, ps))) = (stage("Vertex"), stage("Pixel")) else {
        return Err(not_modelled(
            "a raster pipeline without a vertex and a pixel program",
        ));
    };
    let (hull, domain, geometry) = (stage("Hull"), stage("Domain"), stage("Geometry"));
    if hull.is_some() != domain.is_some() {
        return Err(not_modelled(
            "a hull program without a domain program, or one without the other",
        ));
    }
    let assembly = match p.bindings.topology.as_str() {
        "TriangleList" => Assembly::Primitives(Topology::TriangleList),
        "LineList" => Assembly::Primitives(Topology::LineList),
        "PointList" => Assembly::Primitives(Topology::PointList),
        "PatchList" => Assembly::Patches(
            p.bindings
                .patch_control_points
                .ok_or_else(|| not_modelled("a patch list with no PatchControlPoints"))?,
        ),
        other => return Err(not_modelled(format!("a draw of a {other}"))),
    };
    let RasterTarget {
        name: target,
        format,
        mut surface,
        rasterizer,
    } = target_of(p)?;
    let vsh = vs
        .shader
        .as_ref()
        .map_err(|e| Stopped::Refused(format!("{vs_name}: {e}")))?;
    let psh = ps
        .shader
        .as_ref()
        .map_err(|e| Stopped::Refused(format!("{ps_name}: {e}")))?;
    let between: Vec<&(String, Arc<Ready>)> = [&hull, &domain, &geometry].into_iter().flatten().collect();
    for (r, name) in [(&vs, &vs_name), (&ps, &ps_name)]
        .into_iter()
        .chain(between.iter().map(|(name, r)| (r, name)))
    {
        if let Some(why) = unbindable(r.lifted.code.interface()) {
            return Err(not_modelled(format!("{name}: {why}")));
        }
    }
    let gs = match &geometry {
        Some((name, r)) => match r.lifted.code.geometry() {
            Ok(g) => Some(g),
            Err(e) => return Ok(Err(format!("{name}: {e}"))),
        },
        None => None,
    };
    let tessellation = match (&hull, &domain) {
        (Some((name, h)), Some((_, d))) => match Code::tessellation(&h.lifted.code, &d.lifted.code) {
            Ok(t) => Some(t),
            Err(e) => return Ok(Err(format!("{name}: {e}"))),
        },
        _ => None,
    };
    // a vertex and a pixel program alone link directly; with stages between them, through `Stages`
    let linked = if between.is_empty() {
        match DrawShader::new(vsh, psh) {
            Ok(l) => Some(l),
            Err(e) => return Ok(Err(format!("{ps_name}: {e}"))),
        }
    } else {
        None
    };
    let stride = p.bindings.vertex_stride();
    let count = match p.vertex_count {
        Some(n) => n,
        None => {
            let vb = p
                .buffer(&p.bindings.vertex_buffer)
                .ok_or_else(|| not_modelled("a draw with no vertex buffer and no VertexCount"))?;
            if stride == 0 {
                return Err(not_modelled("a vertex buffer with no attributes"));
            }
            (vb.size / stride) as u32
        }
    };
    let vertices = (0..count)
        .map(|i| vertex_inputs(vsh, p, i))
        .collect::<Result<Vec<_>, _>>()?;
    let indices: Vec<u32> = (0..count).collect();
    let l = layout(p, rs)?;
    let (mut device, constants) = device_of(p, &l)?;
    samplers_in_space_0(&l)?;
    let (vs_constants, ps_constants) = (banks_of(vsh, &constants), banks_of(psh, &constants));
    let vs_banks: Vec<(u32, &[[u32; 4]])> = vs_constants.iter().map(|(s, r)| (*s, r.as_slice())).collect();
    let ps_banks: Vec<(u32, &[[u32; 4]])> = ps_constants.iter().map(|(s, r)| (*s, r.as_slice())).collect();
    let mut interfaces = vec![vsh.interface(), psh.interface()];
    interfaces.extend(between.iter().map(|(_, r)| r.lifted.code.interface()));
    let mut requests = SlotRequests::new(&mut device, &interfaces).map_err(Stopped::NotModelled)?;
    match &linked {
        Some(linked) => {
            let Assembly::Primitives(topology) = assembly else {
                return Err(not_modelled("a patch list with no hull and domain program"));
            };
            let primitives =
                Primitive::assemble(topology, &indices, None).map_err(|e| not_modelled(e.detail))?;
            if let Err(e) = linked.draw(
                &vertices,
                &primitives,
                &vs_banks,
                &ps_banks,
                &mut requests,
                &rasterizer,
                &mut surface,
            ) {
                return Ok(Err(format!("{ps_name}: {e}")));
            }
        }
        None => {
            let rows = rows_of(&constants);
            let staged = Stages {
                vs: vsh,
                tessellation: tessellation.as_ref(),
                gs: gs.as_ref(),
                ps: psh,
            };
            match staged.draw(
                &vertices,
                assembly,
                [&vs_banks, &ps_banks],
                &rows,
                &mut requests,
                &rasterizer,
                &mut surface,
            ) {
                Ok(_) => {}
                Err(StageStop::NotModelled(m)) => return Err(not_modelled(m)),
                Err(StageStop::Failed(stage, m)) => {
                    let name = match stage {
                        "vs" => &vs_name,
                        "hs" => hull.as_ref().map_or(&vs_name, |h| &h.0),
                        "ds" => domain.as_ref().map_or(&vs_name, |d| &d.0),
                        "gs" => geometry.as_ref().map_or(&vs_name, |g| &g.0),
                        _ => &ps_name,
                    };
                    return Ok(Err(format!("{name}: {m}")));
                }
            }
        }
    }
    write_back(p, &target, format, &surface, &l, &device)?;
    Ok(Ok(()))
}

/// Run a mesh pipeline of an optional amplification, a mesh and a pixel program as the offloader's D3D12 mesh pipeline
/// state has it: `DispatchGroupCount` thread groups of the mesh program, or of the amplification program that launches
/// the mesh program's, the primitives they write drawn with the pixel program, with a depth buffer, viewport and
/// target as [`raster`]'s but the default rasterizer state (`CD3DX12_RASTERIZER_DESC`: back faces culled, clockwise
/// triangles front facing), where the traditional pipeline culls none. With an amplification program, it and the mesh
/// program run on the IR ([`Programs::ir`]).
fn mesh_raster(
    p: &mut Pipeline,
    programs: &Programs<'_>,
    digests: &[String],
    rs: Option<&RootSignature>,
) -> Result<Result<(), String>, Stopped> {
    let stage = |s: &str| {
        p.shaders
            .iter()
            .zip(digests)
            .find(|(sh, _)| sh.stage == s)
            .map(|(_, name)| name.clone())
    };
    let (Some(ms_name), Some(ps_name)) = (stage("Mesh"), stage("Pixel")) else {
        return Err(not_modelled("a mesh pipeline without a mesh and a pixel program"));
    };
    let ps = ready(programs, &ps_name)?;
    let lift_ir = |name: &String| {
        programs.ir(name).map_err(|e| match e {
            Stop::Refused(m) | Stop::Failed(m) => Stopped::Refused(format!("{name}: {m}")),
        })
    };
    let (amplification, mesh) = match stage("Amplification") {
        Some(as_name) => {
            let a = match AmplificationShader::from_ir(lift_ir(&as_name)?.as_ref()) {
                Ok(a) => a,
                Err(e) => return Ok(Err(format!("{as_name}: {e}"))),
            };
            let m = match MeshShader::from_ir(lift_ir(&ms_name)?.as_ref()) {
                Ok(m) => m,
                Err(e) => return Ok(Err(format!("{ms_name}: {e}"))),
            };
            (Some((as_name, a)), m)
        }
        None => {
            let ms = ready(programs, &ms_name)?;
            match ms.lifted.code.mesh() {
                Ok(m) => (None, m),
                Err(e) => return Ok(Err(format!("{ms_name}: {e}"))),
            }
        }
    };
    let RasterTarget {
        name: target,
        format,
        mut surface,
        mut rasterizer,
    } = target_of(p)?;
    rasterizer.cull = Cull::Back;
    rasterizer.front_counter_clockwise = false;
    let psh = ps
        .shader
        .as_ref()
        .map_err(|e| Stopped::Refused(format!("{ps_name}: {e}")))?;
    let mut named = vec![(mesh.interface(), &ms_name), (psh.interface(), &ps_name)];
    if let Some((as_name, a)) = &amplification {
        named.push((a.interface(), as_name));
    }
    for (interface, name) in &named {
        if let Some(why) = unbindable(interface) {
            return Err(not_modelled(format!("{name}: {why}")));
        }
    }
    let l = layout(p, rs)?;
    let (mut device, constants) = device_of(p, &l)?;
    samplers_in_space_0(&l)?;
    let ps_constants = banks_of(psh, &constants);
    let ps_banks: Vec<(u32, &[[u32; 4]])> = ps_constants.iter().map(|(s, r)| (*s, r.as_slice())).collect();
    let rows = rows_of(&constants);
    let interfaces: Vec<&iridia::Interface> = named.iter().map(|(i, _)| *i).collect();
    let mut requests = SlotRequests::new(&mut device, &interfaces).map_err(Stopped::NotModelled)?;
    match mesh_draw(
        amplification.as_ref().map(|(_, a)| a),
        &mesh,
        psh,
        p.dispatch,
        WAVE,
        &ps_banks,
        &rows,
        &mut requests,
        &rasterizer,
        &mut surface,
    ) {
        Ok(_) => {}
        Err(StageStop::NotModelled(m)) => return Err(not_modelled(m)),
        Err(StageStop::Failed(stage, m)) => {
            let name = match (stage, &amplification) {
                ("as", Some((as_name, _))) => as_name,
                ("ms", _) => &ms_name,
                _ => &ps_name,
            };
            return Ok(Err(format!("{name}: {m}")));
        }
    }
    write_back(p, &target, format, &surface, &l, &device)?;
    Ok(Ok(()))
}

/// The render target a raster pipeline draws into and the state it draws with, as the offloader's D3D12 pipeline
/// state has them (see [`raster`]).
struct RasterTarget {
    /// The render target's buffer.
    name: String,
    format: Format,
    /// The target, loaded with its buffer's contents, and a depth buffer cleared to 1.
    surface: Surface,
    rasterizer: Rasterizer,
}

/// The render target of a raster pipeline, which draws through one viewport with no stencil test: `SampleCount`
/// samples a pixel, each loaded with the buffer's texel.
fn target_of(p: &Pipeline) -> Result<RasterTarget, Stopped> {
    let samples = p.bindings.sample_count.max(1);
    if p.bindings.viewports.len() > 1 {
        return Err(not_modelled("more than one viewport"));
    }
    if p.stencil.as_ref().is_some_and(|s| s.enabled) {
        return Err(not_modelled("a stencil test"));
    }
    let rt = p
        .buffer(&p.bindings.render_target)
        .ok_or_else(|| not_modelled("a raster pipeline with no render target"))?;
    let format = format_of(rt.format, rt.channels)?;
    let (width, height) = (
        rt.output_props.width.max(1) as u32,
        rt.output_props.height.max(1) as u32,
    );
    let texels: Vec<[u32; 4]> = rt.data[0]
        .chunks(format.size as usize)
        .take((width * height) as usize)
        .flat_map(|t| std::iter::repeat_n(format.read(t), samples as usize))
        .collect();
    let surface = Surface {
        width,
        height,
        layers: 1,
        samples,
        targets: vec![Some(texels)],
        depth: Some(vec![1.0; (width * height * samples) as usize]),
    };
    let [vx, vy, vw, vh, z0, z1] = p
        .bindings
        .viewports
        .first()
        .map_or([0.0, 0.0, width as f32, height as f32, 0.0, 1.0], |v| {
            [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth]
        });
    let rasterizer = Rasterizer {
        viewports: vec![DrawViewport {
            x: vx,
            y: vy,
            width: vw,
            height: vh,
            min_depth: z0,
            max_depth: z1,
        }],
        sample_mask: u32::MAX,
        cull: Cull::None,
        front_counter_clockwise: true,
        depth: Some(DepthState {
            compare: DepthCompare::Less,
            write: true,
            bounds: None,
        }),
        alpha_test: None,
        clip_planes: [None; 6],
        point_size: None,
        point_sprite: false,
        flat_colours: false,
        clamp_colours: false,
        fog: None,
    };
    Ok(RasterTarget {
        name: p.bindings.render_target.clone(),
        format,
        surface,
        rasterizer,
    })
}

/// Write a draw's target back to its buffer `target`, a multisampled one resolved as the offloader resolves it before
/// it reads it back (`ResolveSubresource`), and read back every read-write view of the layout `l`.
fn write_back(
    p: &mut Pipeline,
    target: &str,
    format: Format,
    surface: &Surface,
    l: &Layout,
    device: &Device,
) -> Result<(), Stopped> {
    if let (Some(Some(texels)), Some(rt)) = (surface.targets.first(), p.buffer_mut(target)) {
        let samples = surface.samples.max(1) as usize;
        for (pixel, bytes) in texels
            .chunks(samples)
            .zip(rt.data[0].chunks_mut(format.size as usize))
        {
            let texel = match pixel {
                [one] => *one,
                _ => format.resolve(pixel).map_err(Stopped::NotModelled)?,
            };
            format.write(texel, 0xf, bytes);
        }
    }
    for (i, v) in l.views.iter().enumerate() {
        if v.class == DescriptorKind::Uav
            && let Some(r) = device.resources.get(&(Kind::Uav, i as u32))
        {
            read_back(p, v, r);
        }
    }
    Ok(())
}

/// Refuse a layout with a sampler in a register space other than 0, which a draw's sampler slots do not name.
fn samplers_in_space_0(l: &Layout) -> Result<(), Stopped> {
    if l.samplers.iter().any(|(_, space, _)| *space != 0) {
        return Err(not_modelled("a sampler in a register space other than 0"));
    }
    Ok(())
}

/// The constant buffer banks a stage's program `sh` reads, from the pipeline's `constants` (space 0).
fn banks_of(sh: &ProgramShader, constants: &Rows) -> Vec<(u32, Vec<[u32; 4]>)> {
    sh.constant_buffers()
        .iter()
        .map(|&(slot, rows)| {
            let words = constants.get(&(slot, 0)).map(Vec::as_slice).unwrap_or(&[]);
            let rows = (0..rows as usize)
                .map(|r| [0, 1, 2, 3].map(|k| words.get(r * 4 + k).copied().unwrap_or(0)))
                .collect();
            (slot, rows)
        })
        .collect()
}

/// The pipeline's `constants` as a stage between reads them: `n` rows of the buffer at `register` in `space`, zeros
/// past what it holds.
fn rows_of(constants: &Rows) -> impl Fn(u32, u32, usize) -> Vec<[u32; 4]> + '_ {
    move |register: u32, space: u32, n: usize| -> Vec<[u32; 4]> {
        let words = constants
            .get(&(register, space))
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        (0..n)
            .map(|r| [0, 1, 2, 3].map(|k| words.get(r * 4 + k).copied().unwrap_or(0)))
            .collect()
    }
}

/// Run the offloader on a pipeline and programs.
fn offload(
    programs: &Programs<'_>,
    step: &Value,
    digests: &[String],
    render: Option<&str>,
) -> Result<Offloaded, Stopped> {
    let fail = |stderr: String| Offloaded {
        ok: false,
        stdout: String::new(),
        stderr,
        image: None,
    };
    if let Some(e) = step.get("pipeline_error").and_then(Value::as_str) {
        return Err(not_modelled(format!(
            "the pipeline YAML, which the case reader could not read: {e}"
        )));
    }
    let Some(desc) = step.get("pipeline") else {
        return Err(not_modelled("an offload step with no pipeline"));
    };
    let mut p = match pipeline::read(desc) {
        Ok(p) => p,
        Err(e) => return Ok(fail(format!("gpu-exec: error: {e}\n"))),
    };
    if let Some(what) = p.unmodelled.first() {
        return Err(not_modelled(format!("a pipeline with {what}")));
    }
    if digests.len() != p.shaders.len() && p.kind != PipelineKind::RayTracing {
        return Ok(fail(format!(
            "gpu-exec: error: Pipeline description expects {} shader(s) {} provided\n",
            p.shaders.len(),
            digests.len()
        )));
    }
    let mut stdout = format!("Using DirectX API\nConfiguring execution on device: {DESCRIPTION}\n");
    let result = match p.kind {
        PipelineKind::Compute => {
            let name = &digests[0];
            let cs = ready(programs, name)?;
            let bytes = programs.bytes(name).map_err(Stopped::NotModelled)?;
            let rs = rts0::of(&bytes).map_err(Stopped::NotModelled)?;
            stdout.push_str("Descriptor heap created.\nCommand buffer created.\nBuffers created.\n");
            let r = compute(&mut p, &cs, name, rs.as_ref())?;
            stdout.push_str("Compute Pipeline created.\nCompute command list created.\n");
            r
        }
        PipelineKind::TraditionalRaster => {
            let mut ready_all = Vec::new();
            for name in digests {
                ready_all.push((name.clone(), ready(programs, name)?));
            }
            let bytes = programs.bytes(&digests[0]).map_err(Stopped::NotModelled)?;
            let rs = rts0::of(&bytes).map_err(Stopped::NotModelled)?;
            stdout.push_str("Descriptor heap created.\nCommand buffer created.\nBuffers created.\n");
            stdout.push_str("Render pass created.\nTraditional Raster Pipeline created.\n");
            let r = raster(&mut p, &ready_all, rs.as_ref())?;
            stdout.push_str("Graphics command list created complete.\n");
            r
        }
        PipelineKind::MeshShaderRaster => {
            let bytes = programs.bytes(&digests[0]).map_err(Stopped::NotModelled)?;
            let rs = rts0::of(&bytes).map_err(Stopped::NotModelled)?;
            stdout.push_str(
                "Descriptor heap created.
Command buffer created.
Buffers created.
",
            );
            stdout.push_str(
                "Render pass created.
Mesh Shader Pipeline created.
",
            );
            let r = mesh_raster(&mut p, programs, digests, rs.as_ref())?;
            stdout.push_str(
                "Graphics command list created complete.
",
            );
            r
        }
        PipelineKind::RayTracing => return Err(not_modelled("a ray tracing pipeline")),
    };
    if let Err(e) = result {
        return Ok(fail(format!("gpu-exec: error: {e}\n")));
    }
    stdout.push_str("Compute commands executed.\nRead data back.\n");
    // every element of a read-write resource with a counter reads its counter back, the ones no view bound too
    let counted: Vec<String> = p
        .sets
        .iter()
        .flatten()
        .filter(|r| r.has_counter && r.kind.descriptor() == DescriptorKind::Uav)
        .map(|r| r.name.clone())
        .collect();
    for name in counted {
        if let Some(b) = p.buffer_mut(&name) {
            let n = b.data.len();
            b.counters.resize(n, 0);
        }
    }
    let mut stderr = String::new();
    for r in &p.results {
        let (Some(a), Some(e)) = (p.buffer(&r.actual), p.buffer(&r.expected)) else {
            continue;
        };
        if let Some(text) = super::check::verify(r, a, e) {
            stderr.push_str(&text);
            stderr.push('\n');
        }
    }
    if !stderr.is_empty() {
        return Ok(Offloaded {
            ok: false,
            stdout,
            stderr,
            image: None,
        });
    }
    if let Some(target) = render {
        let b = p
            .buffer(target)
            .ok_or_else(|| not_modelled(format!("No descriptor with name {target}")))?;
        return Ok(Offloaded {
            ok: true,
            stdout,
            stderr,
            image: Some(rgba16_of(b)?),
        });
    }
    stdout.push_str(&pipeline::print(&p));
    Ok(Offloaded {
        ok: true,
        stdout,
        stderr,
        image: None,
    })
}

/// A render target buffer as `writePNG` stores it: each float channel as 16 bits, three or four channels.
fn rgba16_of(b: &Buffer) -> Result<image::Rgba16, Stopped> {
    if b.format != DataFormat::Float32 || !(3..=4).contains(&b.channels) {
        return Err(not_modelled(format!(
            "an image of {} x {}, which writePNG is not given in the suite",
            b.format.name(),
            b.channels
        )));
    }
    let c = b.channels as usize;
    let data = b.data.first().map(Vec::as_slice).unwrap_or(&[]);
    let pixels = data
        .chunks_exact(4 * c)
        .map(|px| {
            let ch = |k: usize| f32::from_le_bytes([px[4 * k], px[4 * k + 1], px[4 * k + 2], px[4 * k + 3]]);
            [
                image::to_u16(ch(0)),
                image::to_u16(ch(1)),
                image::to_u16(ch(2)),
                if c == 4 { image::to_u16(ch(3)) } else { 65535 },
            ]
        })
        .collect();
    Ok(image::Rgba16 {
        width: b.output_props.width.max(0) as u32,
        height: b.output_props.height.max(0) as u32,
        pixels,
    })
}

/// The directives a FileCheck line's prefixes select, in file order.
fn directives(case: &Case, prefixes: &[String]) -> Result<Vec<Directive>, String> {
    let mut out = Vec::new();
    for c in &case.checks {
        if !prefixes.contains(&c.prefix) {
            continue;
        }
        let kind = CheckKind::named(&c.kind)
            .ok_or_else(|| format!("the FileCheck directive {}-{}", c.prefix, c.kind))?;
        out.push(Directive {
            kind,
            pattern: c.pattern.clone(),
        });
    }
    Ok(out)
}

/// Run one test.
pub fn run(file: &File, programs: &Programs<'_>, case: &Case, golden: Option<&Path>) -> TestResult {
    let features = features();
    let mut used = Vec::new();
    let skipped = not_for_direct3d(&case.requires, &case.unsupported).and_then(|none| match none {
        Some(why) => Ok(Err(Outcome::NotApplicable(why))),
        None => applies(&case.requires, &case.unsupported, &case.xfail, &features).map(Ok),
    });
    let outcome = match skipped {
        Err(e) => Outcome::NotModelled(format!("a lit expression: {e}")),
        Ok(Err(not_applicable)) => not_applicable,
        Ok(Ok((Some(why), _))) => {
            Outcome::NotModelled(format!("{why}: a feature iridia's device does not have"))
        }
        Ok(Ok((None, xfail))) => {
            let ran = run_lines(file, programs, case, golden, &mut used);
            match (ran, xfail) {
                (Ok(()), false) => Outcome::Pass,
                (Ok(()), true) => Outcome::XPass,
                (Err(Line::Failed(m)), true) => Outcome::XFail(m),
                (Err(Line::Failed(m)), false) => Outcome::Fail(m),
                (Err(Line::Compiler(m)), true) => Outcome::XFail(m),
                (Err(Line::Compiler(m)), false) => Outcome::CompilerRefused(m),
                (Err(Line::Stopped(Stopped::Refused(m))), _) => Outcome::Refused(m),
                (Err(Line::Stopped(Stopped::NotModelled(m))), _) => Outcome::NotModelled(m),
            }
        }
    };
    TestResult {
        test: case.test.clone(),
        outcome,
        programs: used,
        took: std::time::Duration::ZERO,
    }
}

/// How a RUN line went wrong.
enum Line {
    /// It failed as lit sees a failure.
    Failed(String),
    /// dxc refused a shader.
    Compiler(String),
    /// It could not be taken.
    Stopped(Stopped),
}

fn run_lines(
    file: &File,
    programs: &Programs<'_>,
    case: &Case,
    golden: Option<&Path>,
    used: &mut Vec<String>,
) -> Result<(), Line> {
    let mut last_image: Option<image::Rgba16> = None;
    for step in &case.steps {
        let expect_failure = step
            .get("expect_failure")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if step.get("compile").is_some() {
            let compiled = step.get("program").and_then(Value::as_str).is_some();
            match (compiled, expect_failure) {
                (true, false) | (false, true) => {}
                (true, true) => {
                    return Err(Line::Failed(
                        "dxc compiled a shader the test expects it to refuse".into(),
                    ));
                }
                (false, false) => {
                    let why = step.get("refused").and_then(Value::as_str).unwrap_or("");
                    return Err(Line::Compiler(format!(
                        "dxc: {}",
                        why.lines().next().unwrap_or("")
                    )));
                }
            }
        } else if step.get("offload").is_some() {
            let digests: Vec<String> = step
                .get("programs")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|d| d.as_str().map(String::from)).collect())
                .unwrap_or_default();
            if digests.len() != step.get("programs").and_then(Value::as_array).map_or(0, Vec::len) {
                return Err(Line::Failed(
                    "the offloader was given an object no compile made".into(),
                ));
            }
            used.extend(digests.iter().cloned());
            let render = step.get("render").and_then(Value::as_str);
            let done = offload(programs, step, &digests, render).map_err(Line::Stopped)?;
            match (done.ok, expect_failure) {
                (true, true) => {
                    return Err(Line::Failed(
                        "the offloader succeeded where the test expects it to fail".into(),
                    ));
                }
                (false, false) => return Err(Line::Failed(done.stderr)),
                _ => {}
            }
            if let Some(prefixes) = step.get("filecheck").and_then(Value::as_array) {
                let prefixes: Vec<String> = prefixes
                    .iter()
                    .filter_map(|p| p.as_str().map(String::from))
                    .collect();
                let wanted = directives(case, &prefixes).map_err(|e| Line::Stopped(not_modelled(e)))?;
                let mut input = done.stdout.clone();
                if step.get("stderr").and_then(Value::as_bool).unwrap_or(false) {
                    input.push_str(&done.stderr);
                }
                filecheck::check(&input, &wanted).map_err(|e| Line::Failed(format!("FileCheck: {e}")))?;
            }
            last_image = done.image;
        } else if step.get("imgdiff").is_some() {
            let Some(rel) = step.get("imgdiff").and_then(Value::as_str) else {
                return Err(Line::Stopped(not_modelled("an imgdiff with no golden image")));
            };
            let golden =
                golden.ok_or_else(|| Line::Stopped(not_modelled("no checkout of offload-golden-images")))?;
            let bytes = std::fs::read(golden.join(rel))
                .map_err(|e| Line::Stopped(not_modelled(format!("the golden image {rel}: {e}"))))?;
            let digest = sha2::Sha256::digest(&bytes);
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            if file.images.get(rel) != Some(&hex) {
                return Err(Line::Stopped(not_modelled(format!(
                    "the golden image {rel} is not the one the case file names"
                ))));
            }
            let expected = image::decode_png(&bytes).map_err(|e| Line::Stopped(not_modelled(e)))?;
            let actual = last_image
                .take()
                .ok_or_else(|| Line::Failed("imgdiff: no image was rendered".into()))?;
            if (expected.width, expected.height) != (actual.width, actual.height) {
                return Err(Line::Failed("imgdiff: error: Image sizes do not match".into()));
            }
            let rules = match step.get("rules") {
                Some(r) => image::rules(r).map_err(|e| Line::Stopped(not_modelled(e)))?,
                None => Vec::new(),
            };
            let e: Vec<[u8; 3]> = expected.pixels.iter().map(|p| image::to_srgb8(*p)).collect();
            let a: Vec<[u8; 3]> = actual.pixels.iter().map(|p| image::to_srgb8(*p)).collect();
            image::compare(&e, &a, &rules).map_err(Line::Failed)?;
        } else if let Some(what) = step.get("unmodelled").and_then(Value::as_str) {
            return Err(Line::Stopped(not_modelled(format!("the RUN line `{what}`"))));
        } else {
            return Err(Line::Stopped(not_modelled(
                "a step of a kind this runner does not know",
            )));
        }
    }
    Ok(())
}

/// Run every test of a file, `threads` at a time, its programs through `path`.
pub fn run_all(file: &File, threads: usize, golden: Option<&Path>, path: front::Path) -> Vec<TestResult> {
    let programs = Programs::new(path, |name: &str| {
        let text = file
            .programs
            .get(name)
            .ok_or_else(|| format!("no program {name}"))?;
        pipeline::base64(text)
    });
    run_all_with(file, threads, golden, &programs)
}

/// [`run_all`] with the caller's programs, to read what was lifted afterwards.
pub fn run_all_with(
    file: &File,
    threads: usize,
    golden: Option<&Path>,
    programs: &Programs<'_>,
) -> Vec<TestResult> {
    run_shard_with(file, threads, golden, programs, Shard::WHOLE)
}

/// [`run_all_with`] for the tests of one shard of the file.
pub fn run_shard_with(
    file: &File,
    threads: usize,
    golden: Option<&Path>,
    programs: &Programs<'_>,
    shard: Shard,
) -> Vec<TestResult> {
    let mine = shard.cases(file.cases.len());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&i) = mine.get(k) else { break };
                    let Some(c) = file.cases.get(i) else { break };
                    let start = std::time::Instant::now();
                    let mut done = run(file, programs, c, golden);
                    done.took = start.elapsed();
                    if let Ok(mut all) = results.lock() {
                        all.push((i, done));
                    }
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap_or_default();
    results.sort_by_key(|(i, _)| *i);
    results.into_iter().map(|(_, r)| r).collect()
}
