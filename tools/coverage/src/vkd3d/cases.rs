//! The case files `suites/vkd3d/vkd3d_cases.py` writes: vkd3d's HLSL shader tests read as runs of a Windows runner,
//! each step of a run in order, with the programs Microsoft's compilers made of the tests' shaders.
use std::collections::BTreeMap;

use serde::Deserialize;

/// The case-file format this reader takes (`format` in the file).
pub const FORMAT: u32 = 2;

/// A case file.
#[derive(Debug, Deserialize)]
pub struct File {
    /// The case-file format.
    pub format: u32,
    /// Where the cases come from: vkd3d's commit and the runners read.
    pub source: String,
    /// The programs by name (the first 16 hex digits of the SHA-1 of their bytes), as hex.
    pub programs: BTreeMap<String, String>,
    /// Where each program's shader is in the test files (`file:line`).
    #[serde(default)]
    pub origins: BTreeMap<String, Vec<String>>,
    /// The runs that read a probe.
    pub cases: Vec<Run>,
    /// The runs left out, and why.
    pub left_out: Vec<LeftOut>,
}

/// A run the reader left out, or stopped part way.
#[derive(Debug, Deserialize)]
pub struct LeftOut {
    /// The test file.
    pub file: String,
    /// The runner.
    pub runner: String,
    /// The shader model.
    pub model: String,
    /// Why.
    pub reason: String,
    /// The line it stopped at, where it ran up to one.
    pub line: Option<u64>,
}

/// One test file at one shader model on one runner.
#[derive(Debug, Deserialize)]
pub struct Run {
    /// The test file.
    pub file: String,
    /// The runner: `d3d9`, `d3d11`, `d3d12` or `d3d12-dxc`.
    pub runner: String,
    /// The shader model, `2.0` to `6.6`.
    pub model: String,
    /// What the run does, in order.
    pub steps: Vec<Step>,
}

/// A step of a run, with the line of the test file it comes from.
#[derive(Debug)]
pub struct Step {
    /// The line of the test file.
    pub line: u64,
    /// What it does.
    pub what: What,
}

/// The key that names each kind of step, in the order they are looked for.
const KINDS: [&str; 15] = [
    "section",
    "resource",
    "sampler",
    "input_layout",
    "descriptors",
    "write",
    "draw",
    "dispatch",
    "dispatch_mesh",
    "unrunnable",
    "clear",
    "copy",
    "probe",
    "state",
    "line",
];

impl<'de> Deserialize<'de> for Step {
    /// A step is an object whose kind is the one of `KINDS` it has as a key.
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let map = serde_json::Map::<String, serde_json::Value>::deserialize(d)?;
        let line = map
            .get("line")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| D::Error::custom("a step with no line"))?;
        let kind = KINDS
            .iter()
            .find(|k| **k != "line" && map.contains_key(**k))
            .ok_or_else(|| D::Error::custom(format!("line {line}: a step of no kind this reader knows")))?;
        let mut tagged = serde_json::Map::new();
        tagged.insert((*kind).to_owned(), serde_json::Value::Object(map));
        let what = serde_json::from_value(serde_json::Value::Object(tagged))
            .map_err(|e| D::Error::custom(format!("line {line}: {e}")))?;
        Ok(Step { line, what })
    }
}

/// What a step does: the variant is named by the step's key, its fields are the step's.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum What {
    /// A [test] section starts: `selected` where vkd3d's runner runs it at this model, `extended` where it runs only
    /// here, `null` where neither does (its directives are not read).
    Section {
        /// Which.
        section: Option<String>,
    },
    /// A resource is defined (replacing one of its type and slot).
    Resource {
        /// The resource.
        resource: ResourceDesc,
    },
    /// A sampler is defined.
    Sampler {
        /// The sampler.
        sampler: SamplerDesc,
    },
    /// The input layout of the vertex buffers.
    InputLayout {
        /// Its elements.
        input_layout: Vec<InputElement>,
    },
    /// The descriptor ranges: which resource a register range reads.
    Descriptors {
        /// The ranges.
        descriptors: Vec<Descriptor>,
    },
    /// Words written to a constant buffer.
    Write {
        /// The constant buffer's slot (Direct3D 9: 0 float, 1 integer, 2 boolean constants).
        write: u32,
        /// The first word written.
        at: usize,
        /// The words.
        bits: Vec<u32>,
    },
    /// A draw.
    Draw {
        /// The program of each stage (`vs`, `hs`, `ds`, `gs`, `ps`).
        draw: BTreeMap<String, String>,
        /// The draw's vertices are the runner's quad: a triangle over the target, these clip-space x and y.
        #[serde(default)]
        quad: Option<Vec<f32>>,
        /// The topology of a draw of the run's vertex buffers.
        #[serde(default)]
        topology: Option<String>,
        /// How many vertices.
        #[serde(default)]
        vertices: u32,
        /// How many instances.
        #[serde(default)]
        instances: u32,
        /// The first vertex.
        #[serde(default)]
        first_vertex: u32,
        /// The first instance.
        #[serde(default)]
        first_instance: u32,
    },
    /// A compute dispatch.
    Dispatch {
        /// The thread groups in x, y and z.
        dispatch: [u32; 3],
        /// The compute program (`cs`).
        programs: BTreeMap<String, String>,
    },
    /// A mesh dispatch.
    DispatchMesh {
        /// The thread groups in x, y and z.
        dispatch_mesh: [u32; 3],
        /// The programs (`as`, `ms`, `ps`).
        programs: BTreeMap<String, String>,
    },
    /// A draw or dispatch whose program the compiler refused.
    Unrunnable {
        /// Which shader, and why.
        unrunnable: String,
    },
    /// A target cleared.
    Clear {
        /// The target (`rtv-<n>`, `dsv`, `uav-<n>`).
        clear: String,
        /// The value's words.
        value: Vec<u32>,
        /// A depth-stencil target's stencil value.
        #[serde(default)]
        stencil: Option<u32>,
    },
    /// A resource copied to another.
    Copy {
        /// The source and the destination.
        copy: [String; 2],
    },
    /// A check of what a resource holds.
    Probe {
        /// The rectangle: left, top, right, bottom (right and bottom past it).
        probe: [i64; 4],
        /// The resource read (`rtv-<n>`, `uav-<n>`, `dsv`).
        target: String,
        /// The literal's type: `f32`, `f64`, `u32`, `i32`, `u8`, `u16`, `i16`, `u64` or `i64`.
        kind: String,
        /// How many elements the literal has.
        count: u32,
        /// The literal's words.
        expect: Vec<u32>,
        /// The steps of the float order a value may be from it.
        #[serde(default)]
        ulps: u32,
        /// The array layer.
        #[serde(default)]
        layer: u32,
    },
    /// Fixed-function state.
    State {
        /// The state, as the reader names it.
        state: serde_json::Map<String, serde_json::Value>,
    },
}

/// A resource of a run.
#[derive(Clone, Debug, Deserialize)]
pub struct ResourceDesc {
    /// `rtv`, `dsv`, `srv`, `uav` or `vb`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Its slot.
    pub slot: u32,
    /// Its format (`unknown` for a structured or raw buffer, absent for a vertex buffer).
    #[serde(default)]
    pub format: Option<String>,
    /// `buffer`, `2d`, `3d` or `cube`.
    #[serde(default)]
    pub dimension: Option<String>,
    /// Width in texels (bytes for a vertex buffer, elements for a buffer).
    #[serde(default)]
    pub width: u32,
    /// Height.
    #[serde(default = "one")]
    pub height: u32,
    /// Depth.
    #[serde(default = "one")]
    pub depth: u32,
    /// Array layers (6 for a cube).
    #[serde(default = "one")]
    pub layers: u32,
    /// Mip levels.
    #[serde(default = "one")]
    pub levels: u32,
    /// Samples per texel.
    #[serde(default = "one")]
    pub samples: u32,
    /// A structured buffer's element size.
    #[serde(default)]
    pub stride: Option<u32>,
    /// A raw (byte-address) buffer.
    #[serde(default)]
    pub raw: bool,
    /// A counter buffer.
    #[serde(default)]
    pub counter: bool,
    /// The texture's component mapping (`rgba` by default).
    #[serde(default)]
    pub swizzle: Option<String>,
    /// The least level of detail sampled, as f32 bits.
    #[serde(default)]
    pub min_lod_clamp: Option<u32>,
    /// The resource's words: every level of every layer, in order.
    #[serde(default)]
    pub data: Vec<u32>,
    /// A target the runner makes itself.
    #[serde(default)]
    pub default: bool,
}

fn one() -> u32 {
    1
}

/// A sampler of a run.
#[derive(Clone, Debug, Deserialize)]
pub struct SamplerDesc {
    /// Its slot.
    pub slot: u32,
    /// Its minification, magnification and mip filters, `point` or `linear` each.
    pub filter: String,
    /// The reduction: `standard`, `comparison`, `minimum` or `maximum`.
    #[serde(default)]
    pub reduction: Option<String>,
    /// The addressing along u, v and w.
    pub address: [String; 3],
    /// A comparison sampler's function.
    #[serde(default)]
    pub comparison: Option<String>,
}

/// An element of the input layout.
#[derive(Clone, Debug, Deserialize)]
pub struct InputElement {
    /// The vertex buffer slot.
    pub slot: u32,
    /// The element's format.
    pub format: String,
    /// Its size in bytes.
    pub size: u32,
    /// The semantic name.
    pub name: String,
    /// The semantic index.
    pub index: u32,
}

/// A descriptor range.
#[derive(Clone, Debug, Deserialize)]
pub struct Descriptor {
    /// `srv` or `uav`.
    #[serde(rename = "type")]
    pub kind: String,
    /// The first register.
    pub register: u32,
    /// How many registers.
    pub count: u32,
    /// The register space.
    pub space: u32,
    /// The resource slot of the first register.
    pub target: u32,
}

/// Read a case file.
pub fn read(text: &str) -> Result<File, String> {
    let file: File = serde_json::from_str(text).map_err(|e| format!("the case file: {e}"))?;
    if file.format != FORMAT {
        return Err(format!(
            "the case file is format {}, this reader takes {FORMAT}",
            file.format
        ));
    }
    Ok(file)
}

/// A program's bytes.
pub fn program(file: &File, name: &str) -> Result<Vec<u8>, String> {
    let hex = file
        .programs
        .get(name)
        .ok_or_else(|| format!("program {name} is not in the file"))?;
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(hex.get(i..i + 2).unwrap_or(""), 16).map_err(|e| format!("{name}: {e}")))
        .collect()
}
