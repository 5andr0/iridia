//! An offload-test-suite pipeline (include/Support/Pipeline.h) as the offloader reads it from YAML (here, the case
//! file's JSON of the YAML, every scalar a string and every buffer's `Data` its bytes) and prints it back after a run
//! (lib/Support/Pipeline.cpp's mappings, in their order, through [`yaml::Output`](super::yaml::Output)).
use serde_json::Value;

use super::yaml::{Output, format_g};

/// A buffer's element format (`DataFormat`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs, reason = "the offloader's names")]
pub enum DataFormat {
    Hex8,
    Hex16,
    Hex32,
    Hex64,
    UInt16,
    UInt32,
    UInt64,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Depth32,
    Bool,
}

const DATA_FORMATS: [(&str, DataFormat); 15] = [
    ("Hex8", DataFormat::Hex8),
    ("Hex16", DataFormat::Hex16),
    ("Hex32", DataFormat::Hex32),
    ("Hex64", DataFormat::Hex64),
    ("UInt16", DataFormat::UInt16),
    ("UInt32", DataFormat::UInt32),
    ("UInt64", DataFormat::UInt64),
    ("Int16", DataFormat::Int16),
    ("Int32", DataFormat::Int32),
    ("Int64", DataFormat::Int64),
    ("Float16", DataFormat::Float16),
    ("Float32", DataFormat::Float32),
    ("Float64", DataFormat::Float64),
    ("Depth32", DataFormat::Depth32),
    ("Bool", DataFormat::Bool),
];

impl DataFormat {
    /// Its name.
    pub fn name(self) -> &'static str {
        DATA_FORMATS
            .iter()
            .find(|(_, f)| *f == self)
            .map_or("?", |(n, _)| n)
    }

    /// The size of one scalar (`getFormatSize`).
    pub fn size(self) -> usize {
        match self {
            DataFormat::Hex8 => 1,
            DataFormat::Hex16 | DataFormat::UInt16 | DataFormat::Int16 | DataFormat::Float16 => 2,
            DataFormat::Hex32
            | DataFormat::UInt32
            | DataFormat::Int32
            | DataFormat::Float32
            | DataFormat::Depth32
            | DataFormat::Bool => 4,
            DataFormat::Hex64 | DataFormat::UInt64 | DataFormat::Int64 | DataFormat::Float64 => 8,
        }
    }

    /// Whether it is a floating-point format (`isFloatingPointFormat`).
    pub fn is_float(self) -> bool {
        matches!(
            self,
            DataFormat::Float16 | DataFormat::Float32 | DataFormat::Float64 | DataFormat::Depth32
        )
    }

    /// One scalar as the YAML writer prints it (`ScalarTraits<T>::output` of the type `setData` reads it as).
    pub fn print(self, bytes: &[u8]) -> String {
        let mut b = [0u8; 8];
        b[..bytes.len()].copy_from_slice(bytes);
        let u = u64::from_le_bytes(b);
        match self {
            DataFormat::Hex8
            | DataFormat::Hex16
            | DataFormat::Hex32
            | DataFormat::Hex64
            | DataFormat::Float16 => {
                format!("0x{u:X}")
            }
            DataFormat::UInt16 | DataFormat::UInt32 | DataFormat::UInt64 | DataFormat::Bool => u.to_string(),
            DataFormat::Int16 => (u as u16 as i16).to_string(),
            DataFormat::Int32 => (u as u32 as i32).to_string(),
            DataFormat::Int64 => (u as i64).to_string(),
            DataFormat::Float32 | DataFormat::Depth32 => format_g(f64::from(f32::from_bits(u as u32))),
            DataFormat::Float64 => format_g(f64::from_bits(u)),
        }
    }
}

/// A texture's shape (`OutputProperties`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct OutputProps {
    pub height: i64,
    pub width: i64,
    pub depth: i64,
    pub mip_levels: i64,
    pub array_slices: i64,
}

/// A buffer (`CPUBuffer`).
#[derive(Clone, Debug)]
pub struct Buffer {
    /// Its name.
    pub name: String,
    /// Its scalars' format.
    pub format: DataFormat,
    /// Scalars per element.
    pub channels: i64,
    /// A structure's stride, 0 for none.
    pub stride: i64,
    /// How many buffers it is (an array of resources).
    pub array_size: u32,
    /// The bytes of each.
    pub data: Vec<Vec<u8>>,
    /// The size of each, in bytes.
    pub size: usize,
    /// Its texture shape.
    pub output_props: OutputProps,
    /// The counters read back, one per buffer.
    pub counters: Vec<u32>,
}

impl Buffer {
    /// `getElementSize`.
    pub fn element_size(&self) -> usize {
        if self.stride > 0 {
            self.stride as usize
        } else {
            self.format.size() * self.channels.max(0) as usize
        }
    }
}

/// A resource's kind (`ResourceKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs, reason = "the offloader's names")]
pub enum ResourceKind {
    Buffer,
    StructuredBuffer,
    ByteAddressBuffer,
    Texture1D,
    RWTexture1D,
    Texture1DArray,
    RWTexture1DArray,
    Texture2D,
    RWBuffer,
    RWStructuredBuffer,
    RWByteAddressBuffer,
    RWTexture2D,
    ConstantBuffer,
    Sampler,
    SampledTexture2D,
    AccelerationStructure,
    Texture2DArray,
    RWTexture2DArray,
    Texture3D,
    RWTexture3D,
    TextureCube,
    TextureCubeArray,
}

const RESOURCE_KINDS: [(&str, ResourceKind); 22] = [
    ("Buffer", ResourceKind::Buffer),
    ("StructuredBuffer", ResourceKind::StructuredBuffer),
    ("ByteAddressBuffer", ResourceKind::ByteAddressBuffer),
    ("Texture1D", ResourceKind::Texture1D),
    ("RWTexture1D", ResourceKind::RWTexture1D),
    ("Texture1DArray", ResourceKind::Texture1DArray),
    ("RWTexture1DArray", ResourceKind::RWTexture1DArray),
    ("Texture2D", ResourceKind::Texture2D),
    ("RWBuffer", ResourceKind::RWBuffer),
    ("RWStructuredBuffer", ResourceKind::RWStructuredBuffer),
    ("RWByteAddressBuffer", ResourceKind::RWByteAddressBuffer),
    ("RWTexture2D", ResourceKind::RWTexture2D),
    ("ConstantBuffer", ResourceKind::ConstantBuffer),
    ("Sampler", ResourceKind::Sampler),
    ("SampledTexture2D", ResourceKind::SampledTexture2D),
    ("AccelerationStructure", ResourceKind::AccelerationStructure),
    ("Texture2DArray", ResourceKind::Texture2DArray),
    ("RWTexture2DArray", ResourceKind::RWTexture2DArray),
    ("Texture3D", ResourceKind::Texture3D),
    ("RWTexture3D", ResourceKind::RWTexture3D),
    ("TextureCube", ResourceKind::TextureCube),
    ("TextureCubeArray", ResourceKind::TextureCubeArray),
];

/// A descriptor's class (`DescriptorKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorKind {
    /// An unordered-access view.
    Uav,
    /// A shader-resource view.
    Srv,
    /// A constant-buffer view.
    Cbv,
    /// A sampler.
    Sampler,
}

impl ResourceKind {
    /// Its name.
    pub fn name(self) -> &'static str {
        RESOURCE_KINDS
            .iter()
            .find(|(_, k)| *k == self)
            .map_or("?", |(n, _)| n)
    }

    /// `getDescriptorKind`.
    pub fn descriptor(self) -> DescriptorKind {
        use ResourceKind as K;
        match self {
            K::RWStructuredBuffer
            | K::RWBuffer
            | K::RWByteAddressBuffer
            | K::RWTexture1D
            | K::RWTexture1DArray
            | K::RWTexture2D
            | K::RWTexture2DArray
            | K::RWTexture3D => DescriptorKind::Uav,
            K::ConstantBuffer => DescriptorKind::Cbv,
            K::Sampler => DescriptorKind::Sampler,
            _ => DescriptorKind::Srv,
        }
    }

    /// `isTexture`.
    pub fn is_texture(self) -> bool {
        use ResourceKind as K;
        matches!(
            self,
            K::Texture1D
                | K::RWTexture1D
                | K::Texture1DArray
                | K::RWTexture1DArray
                | K::Texture2D
                | K::RWTexture2D
                | K::Texture2DArray
                | K::RWTexture2DArray
                | K::Texture3D
                | K::RWTexture3D
                | K::TextureCube
                | K::TextureCubeArray
                | K::SampledTexture2D
        )
    }
}

/// A resource of a descriptor set (`Resource`).
#[derive(Clone, Debug)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct Resource {
    pub name: String,
    pub kind: ResourceKind,
    pub has_counter: bool,
    pub tiles_mapped: Option<u32>,
    pub is_reserved: bool,
    /// `DirectXBinding`: register and space.
    pub dx_binding: Option<(u32, u32)>,
    /// `VulkanBinding`: binding and counter binding.
    pub vk_binding: Option<(u32, Option<u32>)>,
    pub heap_index: Option<u32>,
}

/// A sampler (`YAMLSampler`).
#[derive(Clone, Debug)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct SamplerDesc {
    pub name: String,
    pub kind: String,
    pub min_filter: String,
    pub mag_filter: String,
    pub address: String,
    pub min_lod: f32,
    pub max_lod: f32,
    pub mip_lod_bias: f32,
    pub comparison_op: String,
}

/// A comparison rule (`Rule`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs, reason = "the offloader's names")]
pub enum Rule {
    BufferExact,
    BufferFloatULP,
    BufferFloatEpsilon,
}

/// A result to check (`Result`).
#[derive(Clone, Debug)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct Check {
    pub name: String,
    pub rule: Rule,
    pub actual: String,
    pub expected: String,
    pub denorm: String,
    pub ulpt: u32,
    pub epsilon: f64,
    pub zero_tolerance: f64,
}

/// A vertex attribute (`VertexAttribute`).
#[derive(Clone, Debug)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct VertexAttribute {
    pub format: DataFormat,
    pub channels: i64,
    pub offset: i64,
    pub name: String,
}

/// A viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

/// A scissor rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct Scissor {
    pub x: i64,
    pub y: i64,
    pub width: u32,
    pub height: u32,
}

/// The raster bindings (`IOBindings`).
#[derive(Clone, Debug, Default)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct Bindings {
    pub vertex_buffer: String,
    pub vertex_attributes: Vec<VertexAttribute>,
    pub render_target: String,
    pub topology: String,
    pub patch_control_points: Option<u32>,
    pub sample_count: u32,
    pub viewports: Vec<Viewport>,
    pub scissors: Vec<Scissor>,
}

impl Bindings {
    /// `getVertexStride`: the attributes' sizes added up.
    pub fn vertex_stride(&self) -> usize {
        self.vertex_attributes
            .iter()
            .map(|a| a.format.size() * a.channels.max(0) as usize)
            .sum()
    }
}

/// A root parameter (`dx::RootParameter`).
#[derive(Clone, Debug)]
pub enum RootParameter {
    /// Root constants from a buffer.
    Constant(String),
    /// A descriptor table (the next descriptor set).
    DescriptorTable,
    /// A root descriptor: a buffer's name and kind.
    RootDescriptor(String, ResourceKind),
}

/// A push-constant value (`PushConstantValue`).
#[derive(Clone, Debug)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct PushConstantValue {
    pub format: DataFormat,
    pub offset: u32,
    pub data: Vec<u8>,
}

/// A push-constant block (`PushConstantBlock`).
#[derive(Clone, Debug)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct PushConstantBlock {
    pub stage: String,
    pub values: Vec<PushConstantValue>,
}

/// A shader of the pipeline.
#[derive(Clone, Debug)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct Shader {
    pub stage: String,
    pub entry: String,
    /// `SpecializationConstants`: id, type, value.
    pub specialization: Vec<(u32, String, String)>,
}

/// What a pipeline runs (`ShaderPipelineKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineKind {
    /// One compute program.
    Compute,
    /// Vertex, optional hull, domain and geometry, and pixel programs.
    TraditionalRaster,
    /// Optional amplification, mesh and pixel programs.
    MeshShaderRaster,
    /// Ray-tracing programs.
    RayTracing,
}

/// The stencil state.
#[derive(Clone, Debug)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct Stencil {
    pub enabled: bool,
    pub clear_value: u32,
    pub comparison_op: String,
}

/// A pipeline.
#[derive(Clone, Debug)]
#[allow(missing_docs, reason = "the offloader's names")]
pub struct Pipeline {
    pub kind: PipelineKind,
    pub shaders: Vec<Shader>,
    pub root_parameters: Vec<RootParameter>,
    pub shading_rate: String,
    pub stencil: Option<Stencil>,
    pub buffers: Vec<Buffer>,
    pub samplers: Vec<SamplerDesc>,
    pub results: Vec<Check>,
    pub sets: Vec<Vec<Resource>>,
    pub bindings: Bindings,
    pub push_constants: Vec<PushConstantBlock>,
    pub dispatch: [u32; 3],
    pub vertex_count: Option<u32>,
    /// What the pipeline asks that this reader does not model (acceleration structures, ray-tracing tables).
    pub unmodelled: Vec<String>,
}

impl Pipeline {
    /// The buffer named `name`.
    pub fn buffer(&self, name: &str) -> Option<&Buffer> {
        self.buffers.iter().find(|b| b.name == name)
    }

    /// The buffer named `name`, to write.
    pub fn buffer_mut(&mut self, name: &str) -> Option<&mut Buffer> {
        self.buffers.iter_mut().find(|b| b.name == name)
    }
}

// Reading.

fn text<'a>(v: &'a Value, what: &str) -> Result<&'a str, String> {
    v.as_str().ok_or_else(|| format!("{what}: not a scalar"))
}

fn get<'a>(m: &'a Value, key: &str) -> Option<&'a Value> {
    m.get(key)
}

fn required<'a>(m: &'a Value, key: &str) -> Result<&'a Value, String> {
    get(m, key).ok_or_else(|| format!("missing required key '{key}'"))
}

/// `StringRef::getAsInteger` with radix 0.
pub fn llvm_integer(s: &str) -> Option<i128> {
    let t = s.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t),
    };
    let lower = body.to_ascii_lowercase();
    let v = if let Some(h) = lower.strip_prefix("0x") {
        i128::from_str_radix(h, 16).ok()?
    } else if let Some(b) = lower.strip_prefix("0b") {
        i128::from_str_radix(b, 2).ok()?
    } else if let Some(o) = lower.strip_prefix("0o") {
        i128::from_str_radix(o, 8).ok()?
    } else if lower.len() > 1 && lower.starts_with('0') {
        i128::from_str_radix(&lower[1..], 8).ok()?
    } else {
        lower.parse::<i128>().ok()?
    };
    Some(if neg { -v } else { v })
}

fn uint(v: &Value, what: &str) -> Result<u32, String> {
    let s = text(v, what)?;
    llvm_integer(s)
        .filter(|&n| (0..=i128::from(u32::MAX)).contains(&n))
        .map(|n| n as u32)
        .ok_or_else(|| format!("{what}: invalid number '{s}'"))
}

fn int(v: &Value, what: &str) -> Result<i64, String> {
    let s = text(v, what)?;
    llvm_integer(s)
        .filter(|&n| (i128::from(i32::MIN)..=i128::from(i32::MAX)).contains(&n))
        .map(|n| n as i64)
        .ok_or_else(|| format!("{what}: invalid number '{s}'"))
}

fn float(v: &Value, what: &str) -> Result<f64, String> {
    let s = text(v, what)?;
    let t = s.trim();
    match t {
        ".inf" | ".Inf" | ".INF" | "+.inf" | "+.Inf" | "+.INF" => return Ok(f64::INFINITY),
        "-.inf" | "-.Inf" | "-.INF" => return Ok(f64::NEG_INFINITY),
        ".nan" | ".NaN" | ".NAN" => return Ok(f64::NAN),
        _ => {}
    }
    t.parse::<f64>()
        .map_err(|_| format!("{what}: invalid floating point number '{s}'"))
}

fn boolean(v: &Value, what: &str) -> Result<bool, String> {
    match text(v, what)? {
        "true" | "True" | "TRUE" | "yes" | "Yes" | "YES" | "on" | "On" | "ON" | "y" | "Y" | "1" => Ok(true),
        "false" | "False" | "FALSE" | "no" | "No" | "NO" | "off" | "Off" | "OFF" | "n" | "N" | "0" => {
            Ok(false)
        }
        s => Err(format!("{what}: invalid boolean '{s}'")),
    }
}

fn opt<T>(
    m: &Value,
    key: &str,
    default: T,
    read: impl Fn(&Value, &str) -> Result<T, String>,
) -> Result<T, String> {
    match get(m, key) {
        Some(v) => read(v, key),
        None => Ok(default),
    }
}

fn list<'a>(m: &'a Value, key: &str) -> Result<&'a [Value], String> {
    match get(m, key) {
        None => Ok(&[]),
        Some(Value::Array(a)) => Ok(a),
        // an empty flow sequence written `[]` reads as an empty list; a bare key reads as an empty scalar
        Some(Value::String(s)) if s.is_empty() => Ok(&[]),
        Some(_) => Err(format!("{key}: not a sequence")),
    }
}

fn data_format(v: &Value) -> Result<DataFormat, String> {
    let s = text(v, "Format")?;
    DATA_FORMATS
        .iter()
        .find(|(n, _)| *n == s)
        .map(|(_, f)| *f)
        .ok_or_else(|| format!("unknown enumerated scalar '{s}'"))
}

fn resource_kind(v: &Value) -> Result<ResourceKind, String> {
    let s = text(v, "Kind")?;
    RESOURCE_KINDS
        .iter()
        .find(|(n, _)| *n == s)
        .map(|(_, k)| *k)
        .ok_or_else(|| format!("unknown enumerated scalar '{s}'"))
}

fn enumerated(v: &Value, what: &str, names: &[&str]) -> Result<String, String> {
    let s = text(v, what)?;
    if names.contains(&s) {
        Ok(s.to_string())
    } else {
        Err(format!("unknown enumerated scalar '{s}'"))
    }
}

/// Base64, as the case file writes buffer bytes.
pub fn base64(s: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc = 0u32;
    let mut bits = 0;
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => return Err(format!("base64: byte {c:#x}")),
        };
        acc = acc << 6 | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

const STAGES: [&str; 14] = [
    "Compute",
    "Vertex",
    "Hull",
    "Domain",
    "Geometry",
    "Pixel",
    "Amplification",
    "Mesh",
    "RayGeneration",
    "Miss",
    "ClosestHit",
    "AnyHit",
    "Intersection",
    "Callable",
];
const RAY_STAGES: [&str; 6] = [
    "RayGeneration",
    "Miss",
    "ClosestHit",
    "AnyHit",
    "Intersection",
    "Callable",
];
const COMPARES: [&str; 8] = [
    "Never",
    "Less",
    "Equal",
    "LessEqual",
    "Greater",
    "NotEqual",
    "GreaterEqual",
    "Always",
];
const TOPOLOGIES: [&str; 4] = ["TriangleList", "PointList", "PatchList", "LineList"];
const RATES: [&str; 7] = ["1x1", "1x2", "2x1", "2x2", "2x4", "4x2", "4x4"];

fn buffer(b: &Value) -> Result<Buffer, String> {
    let name = text(required(b, "Name")?, "Name")?.to_string();
    let format = data_format(required(b, "Format")?)?;
    let channels = opt(b, "Channels", 1, int)?;
    let stride = opt(b, "Stride", 0, int)?;
    let array_size = opt(b, "ArraySize", 1, uint)?;
    if stride != 0 && channels != 1 {
        return Err("Cannot set a structure stride and more than one channel.".into());
    }
    if get(b, "ZeroInitSize").is_some() {
        return Err("invalid key 'ZeroInitSize' - did you mean 'FillSize'?".into());
    }
    let fill_size = opt(b, "FillSize", 0, int)?;
    let fill_value = match get(b, "FillValue") {
        Some(v) => Some(base64(text(v, "FillValue")?)?),
        None => None,
    };
    let (data, size) = if fill_size > 0 {
        // `setData`: FillSize scalars of FillValue, of which the buffer keeps the first FillSize bytes
        let one = fill_value.unwrap_or_else(|| vec![0; format.size()]);
        let bytes: Vec<u8> = one.iter().copied().cycle().take(fill_size as usize).collect();
        (
            (0..array_size).map(|_| bytes.clone()).collect(),
            fill_size as usize,
        )
    } else {
        if fill_value.as_ref().is_some_and(|v| v.iter().any(|&b| b != 0)) {
            return Err("'FillValue' specified without 'FillSize'".into());
        }
        let d = required(b, "Data")?;
        if array_size == 1 {
            let bytes = base64(text(d, "Data")?)?;
            let n = bytes.len();
            (vec![bytes], n)
        } else {
            let arrays = d
                .as_array()
                .ok_or("Data: an array of buffers needs a list of lists")?
                .iter()
                .map(|x| base64(text(x, "Data")?))
                .collect::<Result<Vec<_>, _>>()?;
            let n = arrays.last().map_or(0, Vec::len);
            if arrays.iter().any(|a| a.len() != n) {
                return Err("All buffers must have the same size.".into());
            }
            if arrays.len() != array_size as usize {
                return Err(format!("Expected {array_size} buffers, found {}", arrays.len()));
            }
            (arrays, n)
        }
    };
    let output_props = match get(b, "OutputProps") {
        Some(p) => OutputProps {
            height: int(required(p, "Height")?, "Height")?,
            width: int(required(p, "Width")?, "Width")?,
            depth: int(required(p, "Depth")?, "Depth")?,
            mip_levels: opt(p, "MipLevels", 1, int)?,
            array_slices: opt(p, "ArraySlices", 1, int)?,
        },
        None => OutputProps {
            mip_levels: 1,
            array_slices: 1,
            ..OutputProps::default()
        },
    };
    let out = Buffer {
        name,
        format,
        channels,
        stride,
        array_size,
        data,
        size,
        output_props,
        counters: Vec::new(),
    };
    if output_props.width > 0 {
        let p = output_props;
        if p.array_slices < 1 {
            return Err(format!(
                "Buffer '{}' has an invalid ArraySlices count ({}); must be at least 1",
                out.name, p.array_slices
            ));
        }
        let (mut w, mut h, mut d) = (p.width as u64, p.height as u64, p.depth as u64);
        let mut chain = 0u64;
        for _ in 0..p.mip_levels {
            chain += w * h * d * out.element_size() as u64;
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            d = (d / 2).max(1);
        }
        let expected = chain * p.array_slices as u64;
        if out.size as u64 != expected {
            return Err(format!(
                "Buffer '{}' size ({}) does not match OutputProps dimensions ({expected})",
                out.name, out.size
            ));
        }
    }
    Ok(out)
}

fn resource(r: &Value) -> Result<Resource, String> {
    let name = text(required(r, "Name")?, "Name")?.to_string();
    let kind = resource_kind(required(r, "Kind")?)?;
    let dx_binding = match get(r, "DirectXBinding") {
        Some(b) => Some((
            uint(required(b, "Register")?, "Register")?,
            uint(required(b, "Space")?, "Space")?,
        )),
        None => None,
    };
    let vk_binding = match get(r, "VulkanBinding") {
        Some(b) => Some((
            uint(required(b, "Binding")?, "Binding")?,
            match get(b, "CounterBinding") {
                Some(c) => Some(uint(c, "CounterBinding")?),
                None => None,
            },
        )),
        None => None,
    };
    let heap_index = match get(r, "HeapIndex") {
        Some(v) => Some(uint(v, "HeapIndex")?),
        None => None,
    };
    if dx_binding.is_some() == heap_index.is_some() {
        return Err(format!(
            "Resource '{name}' must specify exactly one of 'DirectXBinding' or 'HeapIndex'."
        ));
    }
    Ok(Resource {
        name,
        kind,
        has_counter: opt(r, "HasCounter", false, boolean)?,
        tiles_mapped: match get(r, "TilesMapped") {
            Some(v) => Some(uint(v, "TilesMapped")?),
            None => None,
        },
        is_reserved: opt(r, "IsReserved", false, boolean)?,
        dx_binding,
        vk_binding,
        heap_index,
    })
}

fn sampler(s: &Value) -> Result<SamplerDesc, String> {
    let filters = ["Nearest", "Linear"];
    let addresses = ["Clamp", "Repeat", "Mirror", "Border", "MirrorOnce"];
    let e = |key: &str, default: &str, names: &[&str]| -> Result<String, String> {
        match get(s, key) {
            Some(v) => enumerated(v, key, names),
            None => Ok(default.to_string()),
        }
    };
    let out = SamplerDesc {
        name: text(required(s, "Name")?, "Name")?.to_string(),
        kind: e("Kind", "Sampler", &["Sampler", "SamplerComparison"])?,
        min_filter: e("MinFilter", "Linear", &filters)?,
        mag_filter: e("MagFilter", "Linear", &filters)?,
        address: e("Address", "Clamp", &addresses)?,
        min_lod: opt(s, "MinLOD", 0.0, float)? as f32,
        max_lod: opt(s, "MaxLOD", f64::from(f32::MAX), float)? as f32,
        mip_lod_bias: opt(s, "MipLODBias", 0.0, float)? as f32,
        comparison_op: e("ComparisonOp", "Never", &COMPARES)?,
    };
    if out.kind == "Sampler" && out.comparison_op != "Never" {
        return Err("Sampler kind 'Sampler' requires ComparisonOp: Never".into());
    }
    if out.kind == "SamplerComparison" && out.comparison_op == "Never" {
        return Err("Sampler kind 'SamplerComparison' requires ComparisonOp other than Never".into());
    }
    Ok(out)
}

fn check(r: &Value) -> Result<Check, String> {
    let rule = match text(required(r, "Rule")?, "Rule")? {
        "BufferExact" => Rule::BufferExact,
        "BufferFloatULP" => Rule::BufferFloatULP,
        "BufferFloatEpsilon" => Rule::BufferFloatEpsilon,
        s => return Err(format!("unknown enumerated scalar '{s}'")),
    };
    let denorm = |r: &Value| -> Result<String, String> {
        match get(r, "DenormMode") {
            Some(v) => enumerated(v, "DenormMode", &["Any", "FTZ", "Preserve"]),
            None => Ok("Any".into()),
        }
    };
    let mut out = Check {
        name: text(required(r, "Result")?, "Result")?.to_string(),
        rule,
        actual: text(required(r, "Actual")?, "Actual")?.to_string(),
        expected: text(required(r, "Expected")?, "Expected")?.to_string(),
        denorm: "Any".into(),
        ulpt: 0,
        epsilon: 0.0,
        zero_tolerance: 0.0,
    };
    match rule {
        Rule::BufferFloatULP => {
            out.ulpt = uint(required(r, "ULPT")?, "ULPT")?;
            out.denorm = denorm(r)?;
            out.zero_tolerance = opt(r, "ZeroTolerance", 0.0, float)?;
        }
        Rule::BufferFloatEpsilon => {
            out.epsilon = float(required(r, "Epsilon")?, "Epsilon")?;
            out.denorm = denorm(r)?;
        }
        Rule::BufferExact => {}
    }
    Ok(out)
}

/// Read a pipeline as the offloader's YAML reader does, with its errors.
pub fn read(p: &Value) -> Result<Pipeline, String> {
    let mut shaders = Vec::new();
    for s in list(p, "Shaders")? {
        let stage = enumerated(required(s, "Stage")?, "Stage", &STAGES)?;
        let mut specialization = Vec::new();
        for c in list(s, "SpecializationConstants")? {
            specialization.push((
                uint(required(c, "ConstantID")?, "ConstantID")?,
                text(required(c, "Type")?, "Type")?.to_string(),
                text(required(c, "Value")?, "Value")?.to_string(),
            ));
        }
        shaders.push(Shader {
            stage,
            entry: text(required(s, "Entry")?, "Entry")?.to_string(),
            specialization,
        });
    }
    required(p, "Shaders")?;
    let mut root_parameters = Vec::new();
    if let Some(dx) = get(p, "RuntimeSettings").and_then(|r| get(r, "DirectX")) {
        for r in list(dx, "RootParameters")? {
            root_parameters.push(match text(required(r, "Kind")?, "Kind")? {
                "Constant" => RootParameter::Constant(text(required(r, "Name")?, "Name")?.to_string()),
                "DescriptorTable" => RootParameter::DescriptorTable,
                "RootDescriptor" => {
                    let res = required(r, "Resource")?;
                    let kind = resource_kind(required(res, "Kind")?)?;
                    if !matches!(
                        kind,
                        ResourceKind::StructuredBuffer
                            | ResourceKind::RWStructuredBuffer
                            | ResourceKind::ByteAddressBuffer
                            | ResourceKind::RWByteAddressBuffer
                            | ResourceKind::ConstantBuffer
                    ) {
                        return Err("Root descriptors must be raw resource types.".into());
                    }
                    RootParameter::RootDescriptor(text(required(res, "Name")?, "Name")?.to_string(), kind)
                }
                s => return Err(format!("unknown enumerated scalar '{s}'")),
            });
        }
    }
    let shading_rate = match get(p, "ShadingRate") {
        Some(v) => enumerated(v, "ShadingRate", &RATES)?,
        None => "1x1".into(),
    };
    let stencil = match get(p, "StencilState") {
        Some(s) => {
            let out = Stencil {
                enabled: opt(s, "Enabled", false, boolean)?,
                clear_value: opt(s, "ClearValue", 0, uint)?,
                comparison_op: match get(s, "ComparisonOp") {
                    Some(v) => enumerated(v, "ComparisonOp", &COMPARES)?,
                    None => "Equal".into(),
                },
            };
            if out.clear_value > 255 {
                return Err("StencilState.ClearValue must be in the range 0..255".into());
            }
            Some(out)
        }
        None => None,
    };
    required(p, "Buffers")?;
    let buffers = list(p, "Buffers")?
        .iter()
        .map(buffer)
        .collect::<Result<Vec<_>, _>>()?;
    let samplers = list(p, "Samplers")?
        .iter()
        .map(sampler)
        .collect::<Result<Vec<_>, _>>()?;
    let results = list(p, "Results")?
        .iter()
        .map(check)
        .collect::<Result<Vec<_>, _>>()?;
    required(p, "DescriptorSets")?;
    let mut sets = Vec::new();
    for d in list(p, "DescriptorSets")? {
        sets.push(
            list(d, "Resources")?
                .iter()
                .map(resource)
                .collect::<Result<Vec<_>, _>>()?,
        );
    }
    let mut bindings = Bindings {
        topology: "TriangleList".into(),
        sample_count: 1,
        ..Bindings::default()
    };
    if let Some(b) = get(p, "Bindings") {
        bindings.vertex_buffer = opt(b, "VertexBuffer", String::new(), |v, w| {
            text(v, w).map(str::to_string)
        })?;
        for a in list(b, "VertexAttributes")? {
            bindings.vertex_attributes.push(VertexAttribute {
                format: data_format(required(a, "Format")?)?,
                channels: int(required(a, "Channels")?, "Channels")?,
                offset: int(required(a, "Offset")?, "Offset")?,
                name: text(required(a, "Name")?, "Name")?.to_string(),
            });
        }
        bindings.render_target = opt(b, "RenderTarget", String::new(), |v, w| {
            text(v, w).map(str::to_string)
        })?;
        if let Some(t) = get(b, "Topology") {
            bindings.topology = enumerated(t, "Topology", &TOPOLOGIES)?;
        }
        bindings.patch_control_points = match get(b, "PatchControlPoints") {
            Some(v) => Some(uint(v, "PatchControlPoints")?),
            None => None,
        };
        bindings.sample_count = opt(b, "SampleCount", 1, uint)?;
        for v in list(b, "Viewports")? {
            bindings.viewports.push(Viewport {
                x: opt(v, "X", 0.0, float)? as f32,
                y: opt(v, "Y", 0.0, float)? as f32,
                width: float(required(v, "Width")?, "Width")? as f32,
                height: float(required(v, "Height")?, "Height")? as f32,
                min_depth: opt(v, "MinDepth", 0.0, float)? as f32,
                max_depth: opt(v, "MaxDepth", 1.0, float)? as f32,
            });
        }
        for s in list(b, "Scissors")? {
            bindings.scissors.push(Scissor {
                x: opt(s, "X", 0, int)?,
                y: opt(s, "Y", 0, int)?,
                width: uint(required(s, "Width")?, "Width")?,
                height: uint(required(s, "Height")?, "Height")?,
            });
        }
        if bindings.viewports.len() != bindings.scissors.len() {
            return Err(format!(
                "Bindings: 'Scissors' has {} entries but there are {} viewports; they must match.",
                bindings.scissors.len(),
                bindings.viewports.len()
            ));
        }
    }
    let mut push_constants = Vec::new();
    for b in list(p, "PushConstants")? {
        let mut values = Vec::new();
        for v in list(b, "Values")? {
            values.push(PushConstantValue {
                format: data_format(required(v, "Format")?)?,
                offset: uint(required(v, "Offset")?, "Offset")?,
                data: base64(text(required(v, "Data")?, "Data")?)?,
            });
        }
        push_constants.push(PushConstantBlock {
            stage: enumerated(required(b, "Stage")?, "Stage", &STAGES)?,
            values,
        });
    }
    let mut unmodelled = Vec::new();
    for key in [
        "AccelerationStructures",
        "RayTracingPipelineConfig",
        "HitGroups",
        "ShaderBindingTable",
    ] {
        if get(p, key).is_some() {
            unmodelled.push(key.to_string());
        }
    }
    let ray = shaders.iter().any(|s| RAY_STAGES.contains(&s.stage.as_str()));
    let has = |stage: &str| shaders.iter().any(|s| s.stage == stage);
    for (i, s) in shaders.iter().enumerate() {
        if !RAY_STAGES.contains(&s.stage.as_str()) && shaders[..i].iter().any(|t| t.stage == s.stage) {
            return Err("Pipeline has multiple shaders of the same type.".into());
        }
    }
    let kind = if ray {
        PipelineKind::RayTracing
    } else if has("Compute") {
        if shaders.len() != 1 {
            return Err("Compute pipeline must have exactly one compute shader.".into());
        }
        PipelineKind::Compute
    } else if has("Mesh") {
        PipelineKind::MeshShaderRaster
    } else if has("Vertex") {
        PipelineKind::TraditionalRaster
    } else {
        return Err("Pipeline has no compute, vertex, mesh or ray-generation shader.".into());
    };
    if stencil.is_some()
        && !matches!(
            kind,
            PipelineKind::TraditionalRaster | PipelineKind::MeshShaderRaster
        )
    {
        return Err("StencilState is only valid on a raster pipeline".into());
    }
    let (mut dispatch, mut vertex_count) = ([1, 1, 1], None);
    if let Some(d) = get(p, "DispatchParameters") {
        let groups = list(d, "DispatchGroupCount")?;
        for (k, g) in groups.iter().enumerate().take(3) {
            dispatch[k] = uint(g, "DispatchGroupCount")?;
        }
        vertex_count = match get(d, "VertexCount") {
            Some(v) => Some(uint(v, "VertexCount")?),
            None => None,
        };
    }
    match kind {
        PipelineKind::Compute | PipelineKind::MeshShaderRaster | PipelineKind::RayTracing
            if vertex_count.is_some() =>
        {
            return Err(
                "DispatchParameters.VertexCount set on a pipeline that is not a TraditionalRaster one."
                    .into(),
            );
        }
        PipelineKind::TraditionalRaster if dispatch != [1, 1, 1] => {
            return Err(
                "DispatchParameters.DispatchGroupCount set on a TraditionalRaster pipeline. Only allowed on a \
                 Compute pipeline."
                    .into(),
            );
        }
        _ => {}
    }
    let out = Pipeline {
        kind,
        shaders,
        root_parameters,
        shading_rate,
        stencil,
        buffers,
        samplers,
        results,
        sets,
        bindings,
        push_constants,
        dispatch,
        vertex_count,
        unmodelled,
    };
    // the names the mapping resolves
    for set in &out.sets {
        for r in set {
            match r.kind {
                ResourceKind::AccelerationStructure => {}
                ResourceKind::Sampler => {
                    if !out.samplers.iter().any(|s| s.name == r.name) {
                        return Err(format!("Referenced sampler {} not found!", r.name));
                    }
                }
                _ => {
                    if r.kind == ResourceKind::SampledTexture2D
                        && !out.samplers.iter().any(|s| s.name == r.name)
                    {
                        return Err(format!("Referenced sampler {} not found!", r.name));
                    }
                    if out.buffer(&r.name).is_none() {
                        return Err(format!("Referenced buffer {} not found!", r.name));
                    }
                }
            }
        }
    }
    for r in &out.results {
        for name in [&r.actual, &r.expected] {
            if out.buffer(name).is_none() {
                return Err(format!("Reference buffer {name} not found!"));
            }
        }
        if matches!(r.rule, Rule::BufferFloatULP | Rule::BufferFloatEpsilon) {
            let (a, e) = (out.buffer(&r.actual), out.buffer(&r.expected));
            if let (Some(a), Some(e)) = (a, e) {
                if !a.format.is_float() || !e.format.is_float() {
                    return Err("BufferFloat only accepts Float buffers".into());
                }
                if a.format != e.format {
                    return Err("Buffers must have the same type".into());
                }
            }
        }
    }
    let tables = out
        .root_parameters
        .iter()
        .filter(|r| matches!(r, RootParameter::DescriptorTable))
        .count();
    if !out.root_parameters.is_empty() && tables != out.sets.len() {
        return Err(format!(
            "Expected {} DescriptorTable root parameters, found {tables}",
            out.sets.len()
        ));
    }
    for r in &out.root_parameters {
        let name = match r {
            RootParameter::Constant(n) | RootParameter::RootDescriptor(n, _) => n,
            RootParameter::DescriptorTable => continue,
        };
        if out.buffer(name).is_none() {
            return Err(format!("Referenced buffer in root parameter {name} not found!"));
        }
    }
    if !out.bindings.vertex_buffer.is_empty() && out.buffer(&out.bindings.vertex_buffer).is_none() {
        return Err(format!(
            "Referenced vertex buffer {} not found!",
            out.bindings.vertex_buffer
        ));
    }
    if !out.bindings.render_target.is_empty() && out.buffer(&out.bindings.render_target).is_none() {
        return Err(format!(
            "Referenced render target buffer {} not found!",
            out.bindings.render_target
        ));
    }
    Ok(out)
}

// Printing.

fn print_data(o: &mut Output, b: &Buffer) {
    let n = b.format.size();
    let values = |d: &[u8]| -> Vec<String> {
        d[..b.size.min(d.len())]
            .chunks_exact(n)
            .map(|c| b.format.print(c))
            .collect()
    };
    if o.key("Data", true, false) {
        if b.array_size == 1 {
            o.flow_values(&values(b.data.last().map_or(&[][..], Vec::as_slice)));
        } else {
            o.begin_sequence();
            for d in &b.data {
                o.flow_values(&values(d));
                o.post_element();
            }
            o.end_sequence();
        }
        o.post_key();
    }
}

/// A buffer as `MappingTraits<CPUBuffer>` writes it.
pub fn print_buffer(o: &mut Output, b: &Buffer) {
    o.begin_mapping();
    o.string("Name", &b.name);
    o.enumeration("Format", b.format.name(), true, false);
    o.defaulted("Channels", &b.channels.to_string(), b.channels == 1);
    o.defaulted("Stride", &b.stride.to_string(), b.stride == 0);
    o.defaulted("ArraySize", &b.array_size.to_string(), b.array_size == 1);
    if !b.counters.is_empty() {
        if b.array_size == 1 {
            o.field("Counter", &b.counters[b.counters.len() - 1].to_string());
        } else if o.key("Counters", true, false) {
            o.flow_values(&b.counters.iter().map(u32::to_string).collect::<Vec<_>>());
            o.post_key();
        }
    }
    print_data(o, b);
    if o.key("OutputProps", false, false) {
        let p = b.output_props;
        o.begin_mapping();
        o.field("Height", &p.height.to_string());
        o.field("Width", &p.width.to_string());
        o.field("Depth", &p.depth.to_string());
        o.defaulted("MipLevels", &p.mip_levels.to_string(), p.mip_levels == 1);
        o.defaulted("ArraySlices", &p.array_slices.to_string(), p.array_slices == 1);
        o.end_mapping();
        o.post_key();
    }
    o.end_mapping();
}

/// A buffer printed as its own document (`yaml::Output << CPUBuffer`, as `verifyResult` prints one).
pub fn buffer_document(b: &Buffer) -> String {
    let mut o = Output::new();
    o.begin_documents();
    print_buffer(&mut o, b);
    o.end_documents();
    o.finish()
}

fn sequence<T>(o: &mut Output, key: &str, items: &[T], required: bool, each: impl Fn(&mut Output, &T)) {
    if !required && o.can_elide_empty_sequence() && items.is_empty() {
        return;
    }
    if o.key(key, required, false) {
        o.begin_sequence();
        for item in items {
            each(o, item);
            o.post_element();
        }
        o.end_sequence();
        o.post_key();
    }
}

fn mapping(o: &mut Output, key: &str, required: bool, body: impl FnOnce(&mut Output)) {
    if o.key(key, required, false) {
        o.begin_mapping();
        body(o);
        o.end_mapping();
        o.post_key();
    }
}

/// The pipeline as `yaml::Output << Pipeline` writes it.
pub fn print(p: &Pipeline) -> String {
    let mut o = Output::new();
    o.begin_documents();
    o.begin_mapping();
    sequence(&mut o, "Shaders", &p.shaders, true, |o, s| {
        o.begin_mapping();
        o.enumeration("Stage", &s.stage, true, false);
        o.string("Entry", &s.entry);
        sequence(
            o,
            "SpecializationConstants",
            &s.specialization,
            false,
            |o, (id, ty, value)| {
                o.begin_mapping();
                o.field("ConstantID", &id.to_string());
                o.enumeration("Type", ty, true, false);
                o.string("Value", value);
                o.end_mapping();
            },
        );
        o.end_mapping();
    });
    mapping(&mut o, "RuntimeSettings", false, |o| {
        mapping(o, "DirectX", false, |o| {
            sequence(o, "RootParameters", &p.root_parameters, false, |o, r| {
                o.begin_mapping();
                match r {
                    RootParameter::Constant(name) => {
                        o.enumeration("Kind", "Constant", true, false);
                        o.string("Name", name);
                    }
                    RootParameter::DescriptorTable => o.enumeration("Kind", "DescriptorTable", true, false),
                    RootParameter::RootDescriptor(name, kind) => {
                        o.enumeration("Kind", "RootDescriptor", true, false);
                        mapping(o, "Resource", true, |o| {
                            o.string("Name", name);
                            o.enumeration("Kind", kind.name(), true, false);
                            o.defaulted("HasCounter", "false", true);
                            o.field("IsReserved", "false");
                            mapping(o, "DirectXBinding", false, |o| {
                                o.field("Register", "0");
                                o.field("Space", "0");
                            });
                        });
                    }
                }
                o.end_mapping();
            });
        });
    });
    o.enumeration("ShadingRate", &p.shading_rate, false, p.shading_rate == "1x1");
    if let Some(s) = &p.stencil {
        mapping(&mut o, "StencilState", false, |o| {
            o.defaulted("Enabled", if s.enabled { "true" } else { "false" }, !s.enabled);
            o.defaulted("ClearValue", &s.clear_value.to_string(), s.clear_value == 0);
            o.enumeration(
                "ComparisonOp",
                &s.comparison_op,
                false,
                s.comparison_op == "Equal",
            );
        });
    }
    sequence(&mut o, "Buffers", &p.buffers, true, print_buffer);
    sequence(&mut o, "Samplers", &p.samplers, false, |o, s| {
        o.begin_mapping();
        o.string("Name", &s.name);
        o.enumeration("Kind", &s.kind, false, false);
        o.enumeration("MinFilter", &s.min_filter, false, false);
        o.enumeration("MagFilter", &s.mag_filter, false, false);
        o.enumeration("Address", &s.address, false, false);
        o.defaulted("MinLOD", &format_g(f64::from(s.min_lod)), false);
        o.defaulted("MaxLOD", &format_g(f64::from(s.max_lod)), false);
        o.defaulted("MipLODBias", &format_g(f64::from(s.mip_lod_bias)), false);
        o.enumeration("ComparisonOp", &s.comparison_op, false, false);
        o.end_mapping();
    });
    sequence(&mut o, "Results", &p.results, false, |o, r| {
        o.begin_mapping();
        o.string("Result", &r.name);
        let rule = match r.rule {
            Rule::BufferExact => "BufferExact",
            Rule::BufferFloatULP => "BufferFloatULP",
            Rule::BufferFloatEpsilon => "BufferFloatEpsilon",
        };
        o.enumeration("Rule", rule, true, false);
        o.string("Actual", &r.actual);
        o.string("Expected", &r.expected);
        match r.rule {
            Rule::BufferFloatULP => {
                o.field("ULPT", &r.ulpt.to_string());
                o.enumeration("DenormMode", &r.denorm, false, false);
                o.defaulted(
                    "ZeroTolerance",
                    &format_g(r.zero_tolerance),
                    r.zero_tolerance == 0.0,
                );
            }
            Rule::BufferFloatEpsilon => {
                o.field("Epsilon", &format_g(r.epsilon));
                o.enumeration("DenormMode", &r.denorm, false, false);
            }
            Rule::BufferExact => {}
        }
        o.end_mapping();
    });
    sequence(&mut o, "DescriptorSets", &p.sets, true, |o, set| {
        o.begin_mapping();
        sequence(o, "Resources", set, true, |o, r| {
            o.begin_mapping();
            o.string("Name", &r.name);
            o.enumeration("Kind", r.kind.name(), true, false);
            o.defaulted(
                "HasCounter",
                if r.has_counter { "true" } else { "false" },
                !r.has_counter,
            );
            if let Some(t) = r.tiles_mapped {
                o.defaulted("TilesMapped", &t.to_string(), false);
            }
            o.defaulted("IsReserved", if r.is_reserved { "true" } else { "false" }, false);
            if let Some((register, space)) = r.dx_binding {
                mapping(o, "DirectXBinding", false, |o| {
                    o.field("Register", &register.to_string());
                    o.field("Space", &space.to_string());
                });
            }
            if let Some((binding, counter)) = r.vk_binding {
                mapping(o, "VulkanBinding", false, |o| {
                    o.field("Binding", &binding.to_string());
                    if let Some(c) = counter {
                        o.defaulted("CounterBinding", &c.to_string(), false);
                    }
                });
            }
            if let Some(h) = r.heap_index {
                o.defaulted("HeapIndex", &h.to_string(), false);
            }
            o.end_mapping();
        });
        o.end_mapping();
    });
    let b = &p.bindings;
    mapping(&mut o, "Bindings", false, |o| {
        o.string("VertexBuffer", &b.vertex_buffer);
        sequence(o, "VertexAttributes", &b.vertex_attributes, false, |o, a| {
            o.begin_mapping();
            o.enumeration("Format", a.format.name(), true, false);
            o.field("Channels", &a.channels.to_string());
            o.field("Offset", &a.offset.to_string());
            o.string("Name", &a.name);
            o.end_mapping();
        });
        o.string("RenderTarget", &b.render_target);
        o.enumeration("Topology", &b.topology, false, b.topology == "TriangleList");
        if let Some(n) = b.patch_control_points {
            o.defaulted("PatchControlPoints", &n.to_string(), false);
        }
        o.defaulted("SampleCount", &b.sample_count.to_string(), b.sample_count == 1);
        sequence(o, "Viewports", &b.viewports, false, |o, v| {
            o.begin_mapping();
            o.defaulted("X", &format_g(f64::from(v.x)), v.x == 0.0);
            o.defaulted("Y", &format_g(f64::from(v.y)), v.y == 0.0);
            o.field("Width", &format_g(f64::from(v.width)));
            o.field("Height", &format_g(f64::from(v.height)));
            o.defaulted("MinDepth", &format_g(f64::from(v.min_depth)), v.min_depth == 0.0);
            o.defaulted("MaxDepth", &format_g(f64::from(v.max_depth)), v.max_depth == 1.0);
            o.end_mapping();
        });
        sequence(o, "Scissors", &b.scissors, false, |o, s| {
            o.begin_mapping();
            o.defaulted("X", &s.x.to_string(), s.x == 0);
            o.defaulted("Y", &s.y.to_string(), s.y == 0);
            o.field("Width", &s.width.to_string());
            o.field("Height", &s.height.to_string());
            o.end_mapping();
        });
    });
    sequence(&mut o, "PushConstants", &p.push_constants, false, |o, block| {
        o.begin_mapping();
        o.enumeration("Stage", &block.stage, true, false);
        sequence(o, "Values", &block.values, true, |o, v| {
            o.begin_mapping();
            o.enumeration("Format", v.format.name(), true, false);
            o.field("Offset", &v.offset.to_string());
            if o.key("Data", true, false) {
                let n = v.format.size();
                o.flow_values(
                    &v.data
                        .chunks_exact(n)
                        .map(|c| v.format.print(c))
                        .collect::<Vec<_>>(),
                );
                o.post_key();
            }
            o.end_mapping();
        });
        o.end_mapping();
    });
    // AccelerationStructures: an empty mapping of two elided sequences
    mapping(&mut o, "AccelerationStructures", false, |_| {});
    mapping(&mut o, "DispatchParameters", false, |o| {
        if o.key("DispatchGroupCount", false, false) {
            o.flow_values(&p.dispatch.map(|d| d.to_string()));
            o.post_key();
        }
        if let Some(n) = p.vertex_count {
            o.defaulted("VertexCount", &n.to_string(), false);
        }
    });
    o.end_mapping();
    o.end_documents();
    o.finish()
}

#[cfg(test)]
mod tests {
    use super::super::yaml::{Quote, needs_quotes};
    use super::*;

    /// Whether a scalar would be quoted.
    fn quoted(s: &str) -> bool {
        needs_quotes(s, true) != Quote::None
    }

    fn simple() -> Value {
        serde_json::json!({
            "Shaders": [{"Stage": "Compute", "Entry": "main"}],
            "Buffers": [
                {"Name": "In", "Format": "Float32", "Channels": "4",
                 "Data": "AAAAQAAAgEAAAMBAAAAAQQ=="},
                {"Name": "Out", "Format": "Float32", "Channels": "4", "FillSize": "16"}
            ],
            "DescriptorSets": [{"Resources": [
                {"Name": "In", "Kind": "RWBuffer", "DirectXBinding": {"Register": "0", "Space": "0"},
                 "VulkanBinding": {"Binding": "0"}},
                {"Name": "Out", "Kind": "RWBuffer", "DirectXBinding": {"Register": "1", "Space": "0"},
                 "VulkanBinding": {"Binding": "1"}}
            ]}]
        })
    }

    #[test]
    fn reads_and_prints_a_compute_pipeline() {
        let mut p = read(&simple()).expect("reads");
        assert_eq!(p.kind, PipelineKind::Compute);
        assert_eq!(p.buffers[1].size, 16);
        p.buffers[1].data[0] = [4.0f32, 16.0, 36.0, 64.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let text = print(&p);
        assert!(text.contains("Data:            [ 2, 4, 6, 8 ]"), "{text}");
        assert!(text.contains("Data:            [ 4, 16, 36, 64 ]"), "{text}");
        assert!(
            text.contains("RuntimeSettings:\n  DirectX:         {}\n"),
            "{text}"
        );
        assert!(
            text.contains("    OutputProps:\n      Height:          0\n"),
            "{text}"
        );
        assert!(
            text.contains("Bindings:\n  VertexBuffer:    ''\n  RenderTarget:    ''\n"),
            "{text}"
        );
        assert!(
            text.contains("DispatchParameters:\n  DispatchGroupCount: [ 1, 1, 1 ]\n"),
            "{text}"
        );
        assert!(text.contains("        IsReserved:      false\n"), "{text}");
        assert!(text.starts_with("---\nShaders:\n  - Stage:           Compute\n    Entry:           main\n"));
        assert!(text.ends_with("...\n"));
    }

    #[test]
    fn long_flow_sequences_wrap() {
        let mut p = read(&simple()).expect("reads");
        p.buffers[1].size = 64;
        p.buffers[1].data[0] = (0..16).flat_map(|i| (i as f32 * 1000.5).to_le_bytes()).collect();
        let text = print(&p);
        // a wrapped line starts two columns in from the `[`
        let data = text.lines().find(|l| l.contains("1000.5")).expect("data line");
        let next = text
            .lines()
            .skip_while(|l| *l != data)
            .nth(1)
            .expect("wrapped line");
        assert!(data.len() > 70, "{data}");
        assert!(
            next.starts_with(&" ".repeat(23)) && !next.starts_with(&" ".repeat(24)),
            "{next}"
        );
        assert!(!quoted("Out"));
    }

    #[test]
    fn integers_read_as_llvm_reads_them() {
        assert_eq!(llvm_integer("0x10"), Some(16));
        assert_eq!(llvm_integer("010"), Some(8));
        assert_eq!(llvm_integer("-5"), Some(-5));
        assert_eq!(llvm_integer("0b11"), Some(3));
    }
}
