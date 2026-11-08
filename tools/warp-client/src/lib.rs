//! The client of the WGSL execution harness (tools/wgsl-run, `tools.wgsl_run` in a test's configuration), for the tests that
//! hold a writer's WGSL to the IR executor: a WGSL program's entry point run on Direct3D 12's WARP adapter through wgpu, an
//! executor iridia does not write. Jobs and replies are tools/wgsl-run/README.md's JSON lines; [`Harness`] keeps its
//! processes and restarts one that timed out. [`Layout`] is how a program's entry point carries its interface, [`job`] makes
//! a job of a program's WGSL and the executor's register files, and [`registers`] reads a reply back into register files to
//! compare; [`classify`] says how far apart two lanes are, and [`amplified`] which differences a program's own arithmetic
//! explains. The tests find the tools in their own configuration (`primase_testdata::inputs!`) and hand them in.
//!
//! A test-support crate: it is not published, and only tests depend on it.

use std::collections::BTreeMap;
use std::io::{BufRead as _, BufReader, Write as _};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Stdio};
use std::sync::{Mutex, mpsc};
use std::time::Duration;

use iridia_ir::interface::{
    BufferKind, Interface, ResourceKind, SampleType, ScalarType, SystemValue, TextureDim, Value, Varying,
};

pub mod amplified;

/// How long one job may take, compiles included, before its process is killed.
const LIMIT: Duration = Duration::from_secs(120);

/// Dawn's `tint` (`tools.tint`): the WGSL compiler of Chrome, which reads a text and checks it as a browser would.
#[derive(Debug)]
pub struct Tint {
    program: primase_testdata::Program,
    keep: Option<PathBuf>,
}

impl Tint {
    /// The compiler `program` is. A text it refuses is kept, with what it said, in `keep` where there is one.
    pub fn new(program: primase_testdata::Program, keep: Option<PathBuf>) -> Tint {
        Tint { program, keep }
    }

    /// Whether Tint accepts `wgsl`: `Err` has its first diagnostic. `id` names the temporary file (unique per call).
    pub fn check(&self, id: &str, wgsl: &str) -> Result<(), String> {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let folder = std::env::temp_dir().join(format!("iridia-tint-{}", std::process::id()));
        std::fs::create_dir_all(&folder).map_err(|e| format!("making {}: {e}", folder.display()))?;
        let file = folder.join(format!(
            "{n}-{}.wgsl",
            id.rsplit(['/', '\\']).next().unwrap_or(id)
        ));
        std::fs::write(&file, wgsl).map_err(|e| format!("writing {}: {e}", file.display()))?;
        let mut command = self.program.command();
        command.args(["--format", "wgsl"]).path_arg(&file);
        let out = command
            .command()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| format!("starting {}: {e}", self.program.path().display()))?;
        let text = String::from_utf8_lossy(&out.stderr);
        if !out.status.success()
            && let Some(keep) = &self.keep
        {
            let _ = std::fs::create_dir_all(keep);
            let name = file.file_name().unwrap_or_default();
            let _ = std::fs::copy(&file, keep.join(name));
            let _ = std::fs::write(keep.join(name).with_extension("tint.txt"), &*text);
        }
        let _ = std::fs::remove_file(&file);
        if out.status.success() {
            return Ok(());
        }
        Err(text
            .lines()
            .find(|l| l.contains("error"))
            .or_else(|| text.lines().find(|l| !l.trim().is_empty()))
            .unwrap_or("tint failed without a message")
            .to_owned())
    }
}

/// The harness's processes, each used by one job at a time.
#[derive(Debug)]
pub struct Harness {
    program: primase_testdata::Program,
    /// `tools.wgsl_run_dxc`, the `dxcompiler.dll` DXC compiles with (the programs that need Shader Model 6: subgroup
    /// operations, 16-bit and 64-bit types); FXC where it is not configured.
    dxc: Option<PathBuf>,
    idle: Mutex<Vec<Process>>,
}

#[derive(Debug)]
struct Process {
    child: Child,
    stdin: ChildStdin,
    replies: mpsc::Receiver<String>,
}

impl Drop for Harness {
    fn drop(&mut self) {
        for mut p in self
            .idle
            .lock()
            .map(|mut v| std::mem::take(&mut *v))
            .unwrap_or_default()
        {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
    }
}

impl Harness {
    /// The harness `program` is (`tools.wgsl_run`), compiling with the `dxcompiler.dll` `dxc` names where there is one
    /// (`tools.wgsl_run_dxc`). Its processes start with the first job.
    pub fn new(program: primase_testdata::Program, dxc: Option<PathBuf>) -> Harness {
        Harness {
            program,
            dxc,
            idle: Mutex::new(Vec::new()),
        }
    }

    /// Whether DXC compiles (`tools.wgsl_run_dxc` is configured); FXC does otherwise.
    pub fn has_dxc(&self) -> bool {
        self.dxc.is_some()
    }

    fn spawn(&self) -> Result<Process, String> {
        let mut command = self.program.command();
        if let Some(dxc) = &self.dxc {
            command.arg("--dxc").path_arg(dxc);
        }
        command.args(["--timeout", "30"]);
        let mut child = command
            .command()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("starting {}: {e}", self.program.path().display()))?;
        let stdin = child.stdin.take().ok_or("no standard input")?;
        let stdout = child.stdout.take().ok_or("no standard output")?;
        let (send, replies) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if send.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Process {
            child,
            stdin,
            replies,
        })
    }

    /// Run one job: its reply, or why there is none (the process's reply is the harness's error where it gives one). The
    /// machine-wide lock on the adapter (`primase_testdata::warp_exclusive`) is held for the job and no longer, so callers
    /// never take it and a long run of jobs does not shut other WARP users out.
    pub fn run(&self, job: &serde_json::Value) -> Result<serde_json::Value, String> {
        let _exclusive = primase_testdata::warp_exclusive();
        let idle = self.idle.lock().ok().and_then(|mut v| v.pop());
        let mut p = match idle {
            Some(p) => p,
            None => self.spawn()?,
        };
        let line = serde_json::to_string(job).map_err(|e| e.to_string())?;
        if let Err(e) = writeln!(p.stdin, "{line}").and_then(|()| p.stdin.flush()) {
            let _ = p.child.kill();
            return Err(format!("writing a job: {e}"));
        }
        match p.replies.recv_timeout(LIMIT) {
            Ok(reply) => {
                let v: serde_json::Value =
                    serde_json::from_str(&reply).map_err(|e| format!("the reply {reply:?}: {e}"))?;
                // a process that exited (a timed-out job's) is not kept
                if matches!(p.child.try_wait(), Ok(None)) {
                    if let Ok(mut idle) = self.idle.lock() {
                        idle.push(p);
                    }
                } else {
                    let _ = p.child.wait();
                }
                match v.get("error").and_then(|e| e.as_str()) {
                    Some(e) => Err(e.to_owned()),
                    None => Ok(v),
                }
            }
            Err(_) => {
                let _ = p.child.kill();
                let _ = p.child.wait();
                Err(format!("no reply within {} s", LIMIT.as_secs()))
            }
        }
    }
}

/// One entry point argument or result: the varying it carries, by index into the inputs or outputs, and the value the
/// module carries for it (a built-in's own type; a pixel target's four lanes).
#[derive(Clone, Debug)]
pub enum Carried {
    Varying(usize, Value),
    /// A vertex program's clip distances: one array of every clip distance output's lanes, in output order.
    ClipDistances(Vec<usize>),
}

/// How an entry point carries a program's interface, in WGSL order: its arguments and its result members.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub arguments: Vec<Carried>,
    pub results: Vec<Carried>,
}

impl Layout {
    /// The layout `iridia_naga`'s WGSL has (`iridia_naga::entry_io`).
    pub fn naga(interface: &Interface) -> Result<Layout, String> {
        let io = iridia_naga::entry_io(interface)?;
        let clip = |interface: &Interface| -> Vec<usize> {
            interface
                .outputs
                .iter()
                .enumerate()
                .filter(|(_, v)| v.system == Some(SystemValue::ClipDistance))
                .map(|(n, _)| n)
                .collect()
        };
        Ok(Layout {
            arguments: io
                .arguments
                .iter()
                .map(|a| Carried::Varying(a.varying, a.value))
                .collect(),
            results: io
                .results
                .iter()
                .map(|r| {
                    // the one item carrying every clip distance names the first of them
                    if interface.outputs[r.varying].system == Some(SystemValue::ClipDistance) {
                        Carried::ClipDistances(clip(interface))
                    } else {
                        Carried::Varying(r.varying, r.value)
                    }
                })
                .collect(),
        })
    }
}

/// The lanes of a D3D register a varying is on, in ascending order; none for a varying with no register.
fn lanes(v: &Varying) -> Vec<usize> {
    v.provenance
        .d3d
        .map(|s| (0..4).filter(|b| s.mask >> b & 1 != 0).collect())
        .unwrap_or_default()
}

/// The words of an argument carrying input `v` as `carried`, from the input register file: lane `j` is the `j`th lane
/// of its register's mask (zero past them and for a varying with no register), a flag 1 where its word is not zero.
fn argument_words(v: &Varying, carried: Value, registers: &[[u32; 4]]) -> Result<Vec<u32>, String> {
    if carried.scalar == ScalarType::Double {
        return Err(format!("{}{} is f64", v.label.name, v.label.index));
    }
    let register = v.provenance.d3d.and_then(|s| registers.get(s.register as usize));
    let lanes = lanes(v);
    Ok((0..usize::from(carried.width))
        .map(|j| {
            let w = match (register, lanes.get(j)) {
                (Some(r), Some(&l)) => r[l],
                _ => 0,
            };
            if carried.scalar == ScalarType::Bool {
                u32::from(w != 0)
            } else {
                w
            }
        })
        .collect())
}

/// Where the program binds a resource, as a plan says.
#[derive(Clone, Copy, Debug)]
pub struct Bound {
    pub group: u32,
    pub binding: u32,
}

/// How many elements a typed buffer view has: every one its register's texel.
pub const BUFFER_ELEMENTS: u32 = 64;

/// The environment a run takes, `exec_warp`'s: each texture uniform (every texel of every layer the same), 4 texels a
/// side with one level, and every sampler point-filtering with coordinates clamped, so a sample is the texel whatever
/// its coordinate; each typed buffer view [`BUFFER_ELEMENTS`] elements of its texel; the constant buffers' rows.
#[derive(Debug)]
pub struct Environment {
    /// Per texture and typed buffer view, by its D3D register: its texel's lanes' bits.
    pub texels: BTreeMap<u32, [u32; 4]>,
    /// Per constant buffer, by its D3D register: its rows.
    pub buffers: Vec<(u32, Vec<[u32; 4]>)>,
    /// Per structured or raw buffer view, by its D3D register: its words.
    pub words: BTreeMap<u32, Vec<u32>>,
}

/// Why the harness cannot run a program, or cannot bind it as [`Environment`] makes it: `None` when it can.
pub fn gap(program: &iridia_ir::Program) -> Option<String> {
    let needs = iridia_ir::target::requirements(program);
    if needs
        .iter()
        .any(|r| r.capability == iridia_ir::target::Capability::Float64)
    {
        return Some("f64: WARP through wgpu has no SHADER_F64".into());
    }
    for r in &program.interface.resources {
        if r.count != 1 && !matches!(r.kind, ResourceKind::PipelineState(_)) {
            return Some("a resource array: this test binds none yet".into());
        }
        let why = match &r.kind {
            ResourceKind::ConstantBuffer { .. } | ResourceKind::PipelineState(_) => continue,
            ResourceKind::Texture {
                multisampled: true, ..
            } => "a multisampled texture",
            ResourceKind::Texture { .. } => continue,
            ResourceKind::Sampler { .. } => continue,
            ResourceKind::Buffer(_) => continue,
            ResourceKind::Storage { .. } => "an unordered-access view",
            _ => "a resource of another kind",
        };
        return Some(why.into());
    }
    None
}

/// Why the harness cannot run a compute program as [`job`]-less callers bind it: it needs a resource other than constant
/// buffers and structured or raw buffers (read or written), an array of resources, or f64.
pub fn gap_compute(program: &iridia_ir::Program) -> Option<String> {
    let needs = iridia_ir::target::requirements(program);
    if needs
        .iter()
        .any(|r| r.capability == iridia_ir::target::Capability::Float64)
    {
        return Some("f64: WARP through wgpu has no SHADER_F64".into());
    }
    for r in &program.interface.resources {
        if r.count != 1 && !matches!(r.kind, ResourceKind::PipelineState(_)) {
            return Some("a resource array".into());
        }
        let why = match &r.kind {
            ResourceKind::ConstantBuffer { .. } => continue,
            ResourceKind::Buffer(BufferKind::Structured { .. } | BufferKind::Raw) => continue,
            ResourceKind::Storage {
                target:
                    iridia_ir::interface::StorageTarget::Buffer(BufferKind::Structured { .. } | BufferKind::Raw),
                ..
            } => continue,
            ResourceKind::Buffer(BufferKind::Typed(_)) => "a typed buffer view",
            ResourceKind::Storage { .. } => "an unordered-access view that is not a structured or raw buffer",
            ResourceKind::Texture { .. } => "a texture",
            ResourceKind::Sampler { .. } => "a sampler",
            _ => "a resource of another kind",
        };
        return Some(why.into());
    }
    None
}

/// The job running `wgsl` (whose entry point carries `program`'s interface as `layout`, its resources bound where
/// `bindings` says) on `invocations`, each one's input register file, in `env`.
pub fn job(
    interface: &Interface,
    wgsl: &str,
    layout: &Layout,
    bindings: &[(iridia_ir::interface::ResourceId, Bound)],
    invocations: &[&[[u32; 4]]],
    env: &Environment,
) -> Result<serde_json::Value, String> {
    let mut args = Vec::new();
    for registers in invocations {
        let mut words = Vec::new();
        for a in &layout.arguments {
            let Carried::Varying(n, carried) = a else {
                return Err("clip distances as an argument".into());
            };
            let v = interface.inputs.get(*n).ok_or("an argument of no input")?;
            words.push(argument_words(v, *carried, registers)?);
        }
        args.push(words);
    }
    let mut resources = Vec::new();
    for r in &interface.resources {
        let Some((_, at)) = bindings.iter().find(|(id, _)| *id == r.id) else {
            continue;
        };
        let register = r.provenance.d3d.map(|d| d.register).unwrap_or(0);
        let mut entry = serde_json::json!({ "group": at.group, "binding": at.binding });
        match &r.kind {
            ResourceKind::ConstantBuffer { .. } => {
                let rows = env
                    .buffers
                    .iter()
                    .find(|(slot, _)| *slot == register)
                    .map(|(_, rows)| rows.as_slice())
                    .unwrap_or(&[]);
                let words: Vec<u32> = rows.iter().flatten().copied().collect();
                entry["buffer"] = serde_json::json!({ "words": words });
            }
            ResourceKind::Texture { .. } => {
                let texel = env.texels.get(&register).copied().unwrap_or_default();
                entry["texture"] = serde_json::json!({ "texel": texel });
            }
            ResourceKind::Buffer(BufferKind::Typed(_)) => {
                let texel = env.texels.get(&register).copied().unwrap_or_default();
                let words: Vec<u32> = (0..BUFFER_ELEMENTS).flat_map(|_| texel).collect();
                entry["buffer"] = serde_json::json!({ "words": words });
            }
            ResourceKind::Buffer(_) => {
                let words = env
                    .words
                    .get(&register)
                    .ok_or_else(|| format!("no words for t{register}"))?;
                entry["buffer"] = serde_json::json!({ "words": words });
            }
            // a comparison sampler compares with `COMPARE`, which the test's sampler (`Uniform`) applies the same way
            ResourceKind::Sampler { comparison: true } => {
                entry["sampler"] = serde_json::json!({ "compare": COMPARE })
            }
            ResourceKind::Sampler { .. } => entry["sampler"] = serde_json::json!({}),
            _ => continue,
        }
        resources.push(entry);
    }
    Ok(serde_json::json!({ "wgsl": wgsl, "invocations": args, "resources": resources }))
}

/// What an invocation of the harness gave, as the executor's register files: the output registers' lanes (by D3D
/// register), the lanes it carried, and whether it returned (a discarded pixel did not).
#[derive(Debug, Default)]
pub struct Registers {
    pub outputs: BTreeMap<u32, [u32; 4]>,
    pub masks: BTreeMap<u32, u8>,
    pub returned: bool,
}

/// The reply's invocations as register files.
pub fn registers(
    interface: &Interface,
    layout: &Layout,
    reply: &serde_json::Value,
) -> Result<Vec<Registers>, String> {
    let invocations = reply["invocations"]
        .as_array()
        .ok_or("a reply without invocations")?;
    let mut out = Vec::new();
    for inv in invocations {
        let mut r = Registers {
            returned: inv["returned"].as_bool().unwrap_or(false),
            ..Registers::default()
        };
        let results = inv["results"].as_array().cloned().unwrap_or_default();
        if r.returned {
            if results.len() != layout.results.len() {
                return Err(format!(
                    "{} results, where the entry point has {}",
                    results.len(),
                    layout.results.len()
                ));
            }
            for (carried, words) in layout.results.iter().zip(&results) {
                let words: Vec<u32> = words
                    .as_array()
                    .ok_or("a result that is not words")?
                    .iter()
                    .map(|w| w.as_u64().and_then(|w| u32::try_from(w).ok()).ok_or("a word"))
                    .collect::<Result<_, _>>()?;
                let slots: Vec<usize> = match carried {
                    Carried::Varying(n, _) => vec![*n],
                    Carried::ClipDistances(slots) => slots.clone(),
                };
                let mut at = 0;
                for n in slots {
                    let v = interface.outputs.get(n).ok_or("a result of no output")?;
                    let Some(sig) = v.provenance.d3d else {
                        at += usize::from(v.value.width);
                        continue;
                    };
                    for (j, lane) in lanes(v).into_iter().enumerate().take(usize::from(v.value.width)) {
                        let w = *words.get(at + j).ok_or("a result shorter than its output")?;
                        r.outputs.entry(sig.register).or_default()[lane] = w;
                        *r.masks.entry(sig.register).or_default() |= 1 << lane;
                    }
                    at += usize::from(v.value.width);
                }
            }
        }
        out.push(r);
    }
    Ok(out)
}

/// How far apart two lanes are (`exec_warp`'s classes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Exact,
    BothNan,
    Ulps4,
    Relative1e5,
    Relative1e3,
    /// Further apart than 1e-3 relative, but within 2^-20 of each other where either is below 2^-10 in magnitude: a
    /// cancellation near zero that a contracted `mad` leaves a residue of (Direct3D lets `mad` fuse or not), counted on
    /// its own so it stays visible.
    NearZero,
    /// Further apart than 1e-3 relative, but a lane that is `fract(x * c)` of an amplifying constant, the difference
    /// within what WGSL lets `x` be off by ([`amplified::Amplified::explains`]); [`classify`] never gives it, a test
    /// that reads the program's data flow does.
    Amplified,
    Beyond,
}

pub const CLASSES: [&str; 8] = [
    "exact",
    "both NaN",
    "<= 4 ulp",
    "<= 1e-5",
    "<= 1e-3",
    "near zero",
    "amplified",
    "BEYOND 1e-3",
];

/// How far apart lanes `a` and `b` are: exact for integers; for floats, a few units in the last place, a relative
/// distance (Direct3D leaves `exp`, `log`, `sin`, `cos`, `rsqrt` and `rcp` within a few ulp), or near zero.
pub fn classify(float: bool, a: u32, b: u32) -> Class {
    if a == b {
        return Class::Exact;
    }
    if !float {
        return Class::Beyond;
    }
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    if x.is_nan() && y.is_nan() {
        return Class::BothNan;
    }
    if x == y {
        return Class::Exact;
    }
    if x.is_nan() || y.is_nan() {
        return Class::Beyond;
    }
    let key = |w: u32| if w >> 31 != 0 { !w } else { w | 0x8000_0000 };
    if key(a).abs_diff(key(b)) <= 4 {
        return Class::Ulps4;
    }
    let (x, y) = (f64::from(x), f64::from(y));
    let relative = (x - y).abs() / x.abs().max(y.abs());
    if relative <= 1e-5 {
        Class::Relative1e5
    } else if relative <= 1e-3 {
        Class::Relative1e3
    } else if x.abs().min(y.abs()) < 2f64.powi(-10) && (x - y).abs() <= 2f64.powi(-20) {
        Class::NearZero
    } else {
        Class::Beyond
    }
}

/// The comparison function of every comparison sampler the harness binds: a sample passes where `depth_ref <= depth`.
pub const COMPARE: &str = "less_equal";

/// The texel `exec_warp` gives texture `slot` in vector `c`: each lane `k` `((slot * 3 + k + c) % 8 + 1) / 8` as a float
/// for a float texture, that numerator as an integer for an integer one.
pub fn texel(slot: u32, c: u32, sample: SampleType) -> [u32; 4] {
    std::array::from_fn(|k| {
        let n = (slot * 3 + k as u32 + c) % 8 + 1;
        match sample {
            SampleType::Sint | SampleType::Uint => n,
            _ => (n as f32 / 8.0).to_bits(),
        }
    })
}

/// The texture shapes of an interface by D3D register: dimension and arrayed.
pub fn textures(interface: &Interface) -> BTreeMap<u32, (TextureDim, bool, SampleType)> {
    interface
        .resources
        .iter()
        .filter_map(|r| match (&r.kind, r.provenance.d3d) {
            (
                ResourceKind::Texture {
                    dim, arrayed, sample, ..
                },
                Some(d),
            ) => Some((d.register, (*dim, *arrayed, *sample))),
            _ => None,
        })
        .collect()
}

/// The typed buffer views of an interface by D3D register, with the sample type of their texels.
pub fn typed_buffers(interface: &Interface) -> BTreeMap<u32, SampleType> {
    interface
        .resources
        .iter()
        .filter_map(|r| match (&r.kind, r.provenance.d3d) {
            (ResourceKind::Buffer(BufferKind::Typed(v)), Some(d)) => Some((
                d.register,
                match v.scalar {
                    ScalarType::Sint => SampleType::Sint,
                    ScalarType::Uint => SampleType::Uint,
                    _ => SampleType::Float,
                },
            )),
            _ => None,
        })
        .collect()
}

/// How many words a structured or raw buffer view has (a structured one's whole elements within them).
pub const BUFFER_WORDS: u32 = 256;

/// The structured and raw buffer views of an interface by D3D register, with the words of an element (a raw view's
/// element is a word).
pub fn word_buffers(interface: &Interface) -> BTreeMap<u32, u32> {
    interface
        .resources
        .iter()
        .filter_map(|r| match (&r.kind, r.provenance.d3d) {
            (ResourceKind::Buffer(BufferKind::Structured { stride }), Some(d)) => {
                Some((d.register, (stride / 4).max(1)))
            }
            (ResourceKind::Buffer(BufferKind::Raw), Some(d)) => Some((d.register, 1)),
            _ => None,
        })
        .collect()
}

/// An accurate `sin` and `sin4` (wgsl_run/reference_sin.wgsl), what `shader.sin=reference` swaps for the builtin.
pub const REFERENCE_SIN: &str = include_str!("reference_sin.wgsl");

/// `wgsl` with each `sin` of an `f32` or `vec4<f32>` (as the writers emit it: `sin(bitcast<..>(..))`) made `iridia_sin` or
/// `iridia_sin4` and their definitions added; a `sin` of another form, or a `cos`, is an error, so a run that says it swapped
/// `sin` did.
pub fn substitute_sin(wgsl: &str) -> Result<String, String> {
    let mut out = String::with_capacity(wgsl.len() + REFERENCE_SIN.len());
    let mut rest = wgsl;
    while let Some(at) = rest.find("sin(") {
        let (before, after) = rest.split_at(at);
        out.push_str(before);
        let named = before
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let args = &after["sin(".len()..];
        if named {
            out.push_str("sin(");
        } else if args.starts_with("bitcast<vec4<f32>>(") {
            out.push_str("iridia_sin4(");
        } else if args.starts_with("bitcast<f32>(") {
            out.push_str("iridia_sin(");
        } else {
            return Err("a sin of a form shader.sin=reference does not swap".to_owned());
        }
        rest = args;
    }
    out.push_str(rest);
    if out.contains(" cos(") || out.contains("(cos(") {
        return Err("a cos, which shader.sin=reference does not swap".to_owned());
    }
    out.push_str(REFERENCE_SIN);
    Ok(out)
}
