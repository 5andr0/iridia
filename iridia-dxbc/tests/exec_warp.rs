#![allow(clippy::unwrap_used)]
//! The program executor against Microsoft's WARP rasterizer (Direct3D 11, Shader Model 5), the oracle that is not ours:
//! each pixel program of a corpus the IR lift takes is rendered by WARP on one pixel, with the same input registers,
//! constant buffers and uniform textures (of the dimension the program declares: 1D, 2D, 3D, cube, arrays, cube
//! arrays), and run by [`ProgramShader::from_ir`]; every lane both write is compared. Where the older lift takes the
//! program too, it runs on the same cases beside the IR's, which it must agree with bit for bit.
//!
//! Needs `tools.warp` (the oracle program) and `tools.fxc` (Microsoft's `fxc`, for the oracle's vertex shader), and a
//! corpus folder (the corpus oracle set's `sekiro/shaders`, `riders/shaders`, `sledding/shaders`); it works in
//! `warp` under the scratch folder. By default only the programs with a texture that is not a plain 2D one run (1D, 3D, cube
//! and arrayed shapes: the dimension plumbing); the debug key `shader.warp_all=1` runs every program the
//! oracle can bind.
//!
//! Textures are uniform and every sampler point-filters with clamp, so a sample cannot depend on its coordinate: the
//! gate checks the arithmetic around it, the dimension plumbing and the declared shapes, not the sampler. Direct3D
//! leaves `exp`, `log`, `sin`, `cos`, `rsqrt` and `rcp` within a few units in the last place: lanes are classed by how
//! far apart they are, and only a lane further apart than 1e-3 relative fails.
mod common;
mod inputs;
use common::case;
use inputs::{Game, Tool, debug, tool};
use iridia::{
    Interface, InterpMode, InterpSampling, ResourceKind, SampleType, ScalarType, StorageTarget, SystemValue,
    TextureDim,
};
use iridia_exec::{Dim, ImageLoad, ImageQuery, ImageSample, ProgramShader, QueryKind};
use iridia_exec::{PixelShader, Sampler};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn hex(words: &[u32]) -> String {
    words
        .iter()
        .map(|w| format!("{w:08x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The oracle's name for a texture's shape, and its layer count.
fn oracle_dimension(dim: TextureDim, arrayed: bool) -> (&'static str, u32) {
    match (dim, arrayed) {
        (TextureDim::D1, false) => ("1d", 1),
        (TextureDim::D1, true) => ("1darray", 2),
        (TextureDim::D2, false) => ("2d", 1),
        (TextureDim::D2, true) => ("2darray", 2),
        (TextureDim::D3, _) => ("3d", 1),
        (TextureDim::Cube, false) => ("cube", 6),
        (TextureDim::Cube, true) => ("cubearray", 12),
    }
}

/// Whether a region of the IR holds a loop.
fn loops(r: &iridia_ir::Region) -> bool {
    r.nodes
        .iter()
        .any(|n| matches!(n, iridia_ir::Node::Loop { .. }) || n.regions().into_iter().any(loops))
}

/// Why the oracle cannot bind a program, or `None` when it can.
fn unbindable(program: &iridia_ir::Program) -> Option<String> {
    for r in &program.interface.resources {
        match &r.kind {
            ResourceKind::Buffer(_) | ResourceKind::Storage { .. } => {
                return Some(
                    match &r.kind {
                        ResourceKind::Storage {
                            target: StorageTarget::Image { .. },
                            ..
                        } => "an unordered-access image",
                        _ => "a buffer view",
                    }
                    .into(),
                );
            }
            ResourceKind::Texture {
                multisampled, sample, ..
            } if *multisampled || *sample != SampleType::Float => {
                return Some("a multisampled, depth or integer texture".into());
            }
            ResourceKind::Sampler { comparison: true } => return Some("a comparison sampler".into()),
            _ => {}
        }
    }
    // a loop whose bound is a constant of unknown type could run away on any value the test makes up
    let looped = match &program.entry.body {
        iridia_ir::Body::Regions(r) => loops(r),
        // a graph's loops are its back edges: taken as one
        iridia_ir::Body::Cfg(_) => true,
    };
    if looped
        && program.interface.resources.iter().any(
            |r| matches!(&r.kind, ResourceKind::ConstantBuffer { layout } if layout.constants.is_empty()),
        )
    {
        return Some("a loop and constants of unknown type".into());
    }
    if program.interface.inputs.iter().any(|v| {
        v.provenance.d3d.is_none()
            && !matches!(
                v.system,
                Some(SystemValue::PrimitiveIndex | SystemValue::Coverage)
            )
    }) {
        return Some("an input without a register".into());
    }
    None
}

/// The interpolation modifiers of an input in HLSL.
fn modifiers(v: &iridia::Varying) -> String {
    match (v.value.scalar, v.interp) {
        (ScalarType::Float, Some(i)) => {
            let mut m = String::new();
            match i.mode {
                InterpMode::Flat => m += "nointerpolation ",
                InterpMode::Linear => m += "noperspective ",
                InterpMode::Perspective => {}
            }
            match i.sampling {
                InterpSampling::Centroid => m += "centroid ",
                InterpSampling::Sample => m += "sample ",
                InterpSampling::Center => {}
            }
            m
        }
        (ScalarType::Float, None) => String::new(),
        _ => "nointerpolation ".into(),
    }
}

/// The vertex shader that feeds a program's non-system inputs from `cbuffer Case : register(b13) { uint4 r[32]; }`.
///
/// The pixel stage reads the vertex stage's outputs lane for lane, so the vertex shader must pack them where the pixel
/// shader's signature has them: members are declared register by register, in lane order, with a padding member for
/// every lane and register nothing uses (the compiler packs a later scalar into an earlier register's free lanes
/// otherwise). Output register 0 is the position, so a signature that starts at register 0 is shifted up by one.
/// `None` for a signature this cannot reproduce (an element on lanes that are not consecutive).
fn vertex_hlsl(interface: &Interface) -> Option<String> {
    let mut inputs: Vec<_> = interface.inputs.iter().filter(|v| v.system.is_none()).collect();
    inputs.sort_by_key(|v| v.provenance.d3d.map(|s| (s.register, s.mask)));
    let shift = u32::from(
        inputs
            .iter()
            .any(|v| v.provenance.d3d.is_some_and(|s| s.register == 0)),
    );
    let last = inputs
        .iter()
        .filter_map(|v| v.provenance.d3d)
        .map(|s| s.register + shift)
        .max()
        .unwrap_or(0);
    let mut members = String::new();
    let mut body = String::new();
    for register in 1..=last {
        let mut cursor = 0u32;
        // elements that share a register share their interpolation (the compiler packs by it), so the register has one
        // style: the one of the first element the program reads, and an element it does not read (whose interpolation
        // its signature does not state) takes it too
        let in_register =
            |v: &iridia::Varying| v.provenance.d3d.is_some_and(|s| s.register + shift == register);
        let style = inputs
            .iter()
            .copied()
            .filter(|v| in_register(v))
            .find(|v| v.used)
            .or_else(|| inputs.iter().copied().find(|v| in_register(v)))
            .map(modifiers)
            .unwrap_or_default();
        for (n, v) in inputs.iter().enumerate() {
            let sig = v.provenance.d3d?;
            if sig.register + shift != register {
                continue;
            }
            let (first, width) = (sig.mask.trailing_zeros(), sig.mask.count_ones());
            if u32::from(sig.mask) != ((1u32 << width) - 1) << first {
                return None;
            }
            if first > cursor {
                members += &format!(
                    "  {style}float{} pad{register}_{cursor} : PADDING{register}{cursor};
",
                    first - cursor
                );
            }
            let (ty, conv) = match v.value.scalar {
                ScalarType::Uint => ("uint", ""),
                ScalarType::Sint => ("int", "asint"),
                _ => ("float", "asfloat"),
            };
            let name = if width == 1 {
                ty.to_string()
            } else {
                format!("{ty}{width}")
            };
            let swizzle: String = (first..first + width)
                .map(|b| ["x", "y", "z", "w"][b as usize])
                .collect();
            let own = if v.used || ty != "float" {
                modifiers(v)
            } else {
                style.clone()
            };
            members += &format!(
                "  {own}{name} f{n} : {}{};
",
                v.label.name, v.label.index
            );
            body += &format!(
                "  o.f{n} = {conv}(r[{}].{swizzle});
",
                sig.register
            );
            cursor = first + width;
        }
        if cursor < 4 {
            members += &format!(
                "  {style}float{} pad{register}_{cursor} : PADDING{register}{cursor};
",
                4 - cursor
            );
        }
    }
    Some(format!(
        "cbuffer Case : register(b13) {{ uint4 r[32]; }};
struct VSOut {{
  float4 pos : SV_Position;
{members}}};
VSOut main(uint id : SV_VertexID) {{
  VSOut o;
  o.pos = float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0.5, 1);
{body}  return o;
}}
"
    ))
}

/// Answers every request with the slot's uniform colour, as the oracle's textures do; a load outside the texture, or at
/// another mip level than the only one, reads zeros.
struct Uniform {
    colours: BTreeMap<u32, [f32; 4]>,
    /// Texture slot -> (dimension, layers)
    shapes: BTreeMap<u32, (TextureDim, u32)>,
    samples: usize,
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
        self.samples += 1;
        // a gather returns the component of each of four texels: of a uniform texture, that component four times
        let c = self
            .colours
            .get(&r.texture)
            .ok_or_else(|| format!("no texture t{}", r.texture))?;
        Ok(match r.gather {
            Some(k) => [c[usize::from(k)]; 4],
            None => *c,
        })
    }
    fn load(&mut self, r: iridia_exec::Load) -> Result<[u32; 4], String> {
        // a plain load: x, y and an array layer or 3D slice in the third lane
        let (dim, layers) = self.shapes[&r.texture];
        let (layer, z) = match dim {
            TextureDim::D3 => (None, r.texel[2]),
            _ if layers > 1 => (Some(r.texel[2]), 0),
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
        let c = self
            .colours
            .get(&r.texture)
            .ok_or_else(|| format!("no texture t{}", r.texture))?;
        let (dim, layers) = self.shapes[&r.texture];
        let comps = match dim {
            TextureDim::D1 => 1,
            TextureDim::D2 => 2,
            TextureDim::D3 | TextureDim::Cube => 3,
        };
        let inside = r.mip.unwrap_or(0) == 0
            && r.texel[..comps].iter().all(|t| (0..4).contains(t))
            && r.layer.is_none_or(|l| (0..layers as i32).contains(&l));
        Ok(if inside { c.map(f32::to_bits) } else { [0; 4] })
    }
    fn query_image(&mut self, r: ImageQuery<u32>) -> Result<[u32; 4], String> {
        let (_, layers) = self.shapes[&r.texture];
        Ok(match r.kind {
            // the oracle's textures are 4 texels a side with one level: a level past it has no size
            QueryKind::Size if r.level.unwrap_or(0) > 0 => [0; 4],
            QueryKind::Size => [4, 4, 4, 0],
            QueryKind::Levels => [1, 0, 0, 0],
            QueryKind::Layers => [layers, 0, 0, 0],
            QueryKind::Samples => [1, 0, 0, 0],
        })
    }
}

/// How far apart two lanes are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    Exact,
    BothNan,
    Ulps4,
    Relative1e5,
    Relative1e3,
    Beyond,
}
const CLASSES: [&str; 6] = [
    "exact",
    "both NaN",
    "<= 4 ulp",
    "<= 1e-5",
    "<= 1e-3",
    "BEYOND 1e-3",
];

fn classify(float: bool, a: u32, b: u32) -> Class {
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
        return Class::Exact; // +0 and -0
    }
    if x.is_nan() || y.is_nan() {
        return Class::Beyond;
    }
    // a difference of a few units in the last place
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
    } else {
        Class::Beyond
    }
}

#[derive(Default)]
struct Tally {
    programs: usize,
    lifted: usize,
    selected: usize,
    compared: usize,
    discarded_runs: usize,
    unbindable: BTreeMap<String, usize>,
    errors: BTreeMap<String, usize>,
    lanes: BTreeMap<Class, usize>,
    worst: Vec<String>,
    by_program: BTreeMap<String, Class>,
    /// Compared programs by the shape of each of their textures (a program counts once per shape).
    shapes: BTreeMap<&'static str, usize>,
    /// The pixel programs the IR lift refuses (by why).
    ir_refused: BTreeMap<String, usize>,
    /// The older lift's runs that agree with the IR lift's bit for bit, and the programs it does not lift (by why).
    older_agree: usize,
    older_refused: BTreeMap<String, usize>,
    /// The runs where the older lift's outputs are not the IR lift's.
    older_differ: Vec<String>,
}

/// What a run of a pixel program leaves: each written output lane (a NaN as one pattern), or that it discarded.
fn observed(e: &iridia_exec::Execution) -> Option<Vec<[u32; 4]>> {
    let canonical = |w: u32| {
        if f32::from_bits(w).is_nan() {
            0x7fc0_0000
        } else {
            w
        }
    };
    (!e.discarded).then(|| {
        e.outputs
            .iter()
            .zip(&e.output_masks)
            .map(|(o, &m)| std::array::from_fn(|k| if m >> k & 1 != 0 { canonical(o[k]) } else { 0 }))
            .collect()
    })
}

fn run_program(
    name: &str,
    bytes: &[u8],
    cases: u32,
    all: bool,
    tools: &(PathBuf, PathBuf, inputs::OwnScratch),
    t: &mut Tally,
) {
    let (fxc, warp, scratch) = tools;
    t.programs += 1;
    if iridia_dxbc::signatures(bytes).map(|i| i.stage).ok() != Some(iridia::Stage::Pixel) {
        return;
    }
    let program = match iridia_dxbc::ir::lift(bytes) {
        Ok(p) => p,
        Err(e) => {
            *t.ir_refused.entry(common::reason(&e.to_string())).or_default() += 1;
            return;
        }
    };
    t.lifted += 1;
    let exotic = program.interface.resources.iter().any(|r| {
        matches!(
            r.kind,
            ResourceKind::Texture { dim, arrayed, .. } if dim != TextureDim::D2 || arrayed
        )
    });
    if !exotic && !all {
        return;
    }
    if let Some(why) = unbindable(&program) {
        *t.unbindable.entry(why).or_default() += 1;
        return;
    }
    t.selected += 1;
    let Some(vertex) = vertex_hlsl(&program.interface) else {
        *t.unbindable
            .entry("inputs on lanes the vertex shader cannot pack alike".into())
            .or_default() += 1;
        return;
    };
    let shader = match ProgramShader::from_ir(program) {
        Ok(s) => s,
        Err(e) => {
            *t.errors
                .entry(format!("ProgramShader::from_ir: {e}"))
                .or_default() += 1;
            return;
        }
    };
    // the older lift of the same program, where it takes it, run on the same cases beside it
    let older = match iridia_dxbc::lift(bytes)
        .map_err(|e| e.to_string())
        .and_then(|p| ProgramShader::new(p).map_err(|e| e.to_string()))
    {
        Ok(s) => Some(s),
        Err(e) => {
            *t.older_refused.entry(common::reason(&e)).or_default() += 1;
            None
        }
    };
    let tag = Path::new(name)
        .file_stem()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let hlsl_path = scratch.join(format!("{tag}.vs.hlsl"));
    let vs_path = scratch.join(format!("{tag}.vs.cso"));
    let ps_path = scratch.join(format!("{tag}.ps.dxbc"));
    std::fs::write(&hlsl_path, vertex).unwrap();
    std::fs::write(&ps_path, bytes).unwrap();
    let compiled = std::process::Command::new(fxc)
        .args(["/nologo", "/T", "vs_5_0", "/E", "main", "/Fo"])
        .arg(&vs_path)
        .arg(&hlsl_path)
        .output()
        .unwrap();
    if !compiled.status.success() {
        let first = String::from_utf8_lossy(&compiled.stderr)
            .lines()
            .next()
            .unwrap_or("")
            .to_owned();
        *t.errors.entry(format!("fxc: {first}")).or_default() += 1;
        return;
    }
    let interface = shader.interface().clone();
    let targets: Vec<(u32, u32, u8, ScalarType)> = interface
        .outputs
        .iter()
        .filter_map(|v| match (v.system, v.provenance.d3d) {
            (Some(SystemValue::Target(n)), Some(sig)) => Some((n, sig.register, sig.mask, v.value.scalar)),
            _ => None,
        })
        .collect();
    let textures: Vec<(u32, TextureDim, bool)> = interface
        .resources
        .iter()
        .filter_map(|r| match (&r.kind, r.provenance.d3d) {
            (ResourceKind::Texture { dim, arrayed, .. }, Some(b)) => Some((b.register, *dim, *arrayed)),
            _ => None,
        })
        .collect();
    let mut text = String::new();
    let mut ours = Vec::new();
    for c in 0..cases {
        let case = case(&shader, c);
        let front = if c >= 2 {
            !c.is_multiple_of(3)
        } else {
            c.is_multiple_of(2)
        };
        text += &format!(
            "case c{c}\napi d3d11\nps {}\nvs {}\nfront {}\n",
            ps_path.display(),
            vs_path.display(),
            u8::from(front)
        );
        for (r, w) in case.inputs.iter().enumerate() {
            text += &format!("input {r} {}\n", hex(w));
        }
        for (slot, rows) in &case.buffers {
            text += &format!("cb {slot} {}\n", rows.len());
            for row in rows {
                text += &format!("row {}\n", hex(row));
            }
        }
        let mut colours = BTreeMap::new();
        let mut shapes = BTreeMap::new();
        for &(slot, dim, arrayed) in &textures {
            let v = |k: u32| {
                if std::env::var("EXP_TEX").is_ok() {
                    0.5
                } else {
                    ((slot * 3 + k + c) % 8 + 1) as f32 / 8.0
                }
            };
            let colour = [v(0), v(1), v(2), v(3)];
            let (dimension, layers) = oracle_dimension(dim, arrayed);
            text += &format!("texture {slot} {} {dimension}\n", hex(&colour.map(f32::to_bits)));
            colours.insert(slot, colour);
            shapes.insert(slot, (dim, layers));
        }
        for &(n, _, _, scalar) in &targets {
            text += &format!(
                "target {n} {}\n",
                match scalar {
                    ScalarType::Uint => "uint",
                    ScalarType::Sint => "sint",
                    _ => "float",
                }
            );
        }
        text += "end\n";
        let mut sampler = Uniform {
            colours,
            shapes,
            samples: 0,
        };
        let zero = vec![[[0u32; 4]; 2]; case.inputs.len()];
        let refs = common::refs(&case, &shader);
        let a = shader.execute_with_derivatives(&case.inputs, &zero, &refs, &mut sampler);
        if let (Some(b), Ok(a)) = (&older, &a) {
            sampler.samples = 0;
            match b.execute_with_derivatives(&case.inputs, &zero, &refs, &mut sampler) {
                Ok(b) if observed(&b) == observed(a) => t.older_agree += 1,
                b => t.older_differ.push(format!(
                    "{name} case {c}: IR lift {:x?}, older lift {:x?}",
                    observed(a),
                    b.map(|b| observed(&b))
                )),
            }
        }
        ours.push(a);
    }
    let cases_path = scratch.join(format!("{tag}.cases.txt"));
    let results_path = scratch.join(format!("{tag}.warp.jsonl"));
    std::fs::write(&cases_path, text).unwrap();
    let _oracle = inputs::warp_exclusive();
    let run = std::process::Command::new(warp)
        .arg(&cases_path)
        .arg(&results_path)
        .output()
        .unwrap();
    if !run.status.success() {
        *t.errors
            .entry(format!(
                "warp: {}",
                String::from_utf8_lossy(&run.stderr).lines().next().unwrap_or("")
            ))
            .or_default() += 1;
        return;
    }
    let mut worst = Class::Exact;
    let results = std::fs::read_to_string(&results_path).unwrap();
    for (c, line) in results.lines().enumerate() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        if let Some(e) = v["error"].as_str() {
            *t.errors.entry(format!("WARP: {e}")).or_default() += 1;
            continue;
        }
        let execution = match &ours[c] {
            Ok(e) => e,
            // a scalar run cannot derive a computed value (WARP's one pixel has no neighbours either): not a comparison
            Err(e) if e.detail.contains("derivative of a computed value") => {
                *t.unbindable
                    .entry("a derivative of a computed value".into())
                    .or_default() += 1;
                continue;
            }
            Err(e) => {
                *t.errors
                    .entry(format!(
                        "executor: {}",
                        e.detail.chars().take(80).collect::<String>()
                    ))
                    .or_default() += 1;
                continue;
            }
        };
        if execution.discarded {
            t.discarded_runs += 1;
            continue;
        }
        t.compared += 1;
        for (k, &(_, register, mask, scalar)) in targets.iter().enumerate() {
            for lane in (0..4usize).filter(|l| mask >> l & 1 != 0) {
                let warp_bits =
                    u32::from_str_radix(v["targets"][k][lane].as_str().unwrap_or(""), 16).unwrap();
                let our_bits = execution.outputs[register as usize][lane];
                let class = classify(scalar == ScalarType::Float, our_bits, warp_bits);
                *t.lanes.entry(class).or_default() += 1;
                worst = worst.max(class);
                if class == Class::Beyond && t.worst.len() < 30 {
                    t.worst.push(format!(
                        "{name} case {c} t{k}.{lane}: executor {our_bits:#010x} ({}) WARP {warp_bits:#010x} ({})",
                        f32::from_bits(our_bits),
                        f32::from_bits(warp_bits)
                    ));
                }
            }
        }
    }
    t.by_program.insert(name.to_owned(), worst);
    let mut seen = std::collections::BTreeSet::new();
    for &(_, dim, arrayed) in &textures {
        if seen.insert(oracle_dimension(dim, arrayed).0) {
            *t.shapes.entry(oracle_dimension(dim, arrayed).0).or_default() += 1;
        }
    }
}

fn report(title: &str, t: &Tally) {
    println!(
        "{title}: {} programs, {} lifted, {} selected, {} compared runs ({} discarded runs skipped)",
        t.programs, t.lifted, t.selected, t.compared, t.discarded_runs
    );
    let total: usize = t.lanes.values().sum();
    for (c, n) in &t.lanes {
        println!("  {:12} {n:7} lanes of {total}", CLASSES[*c as usize]);
    }
    let mut programs: BTreeMap<Class, usize> = BTreeMap::new();
    for c in t.by_program.values() {
        *programs.entry(*c).or_default() += 1;
    }
    for (c, n) in &programs {
        println!("  programs whose worst lane is {:12} {n}", CLASSES[*c as usize]);
    }
    for (shape, n) in &t.shapes {
        println!("  compared programs with a {shape} texture: {n}");
    }
    for (why, n) in &t.unbindable {
        println!("  unbindable {n:5}  {why}");
    }
    for (why, n) in &t.errors {
        println!("  ERROR {n:5}  {why}");
    }
    for w in &t.worst {
        println!("  BEYOND {w}");
    }
    for (why, n) in &t.ir_refused {
        println!("  IR refuses {n:5}  {why}");
    }
    println!(
        "  the older lift: {} runs agree with the IR lift's",
        t.older_agree
    );
    for (why, n) in &t.older_refused {
        println!("  the older lift refuses {n:5}  {why}");
    }
    for d in t.older_differ.iter().take(20) {
        println!("  OLDER LIFT DIFFERS {d}");
    }
}

fn corpus(game: Game) {
    if let Some((name, paths)) = common::corpus(game) {
        corpus_over(&name, &paths);
    }
}

/// The pixel programs of the containers `paths` (`name` in the report) against WARP, the older lift beside the IR's.
fn corpus_over(name: &str, paths: &[PathBuf]) {
    let (Some(fxc), Some(warp)) = (tool(Tool::Fxc), tool(Tool::Warp)) else {
        return;
    };
    let scratch = inputs::own_scratch("warp");
    let all = debug("shader.warp_all").is_some_and(|v| v == "1");
    let tools = (fxc, warp, scratch);
    let mut t = Tally::default();
    for path in paths.iter().take(common::diff_limit()) {
        let bytes = std::fs::read(path).unwrap();
        run_program(&path.display().to_string(), &bytes, 3, all, &tools, &mut t);
    }
    report(name, &t);
    // Direct3D leaves exp2, log2, sin, cos and rsqrt to the implementation within a few ulp, and a program that raises
    // a value to a large power through them turns that into more than 1e-3; only the textures that are not plain 2D are
    // asserted, the everything-bindable run (the debug key `shader.warp_all=1`) reports
    assert!(
        all || t.worst.is_empty(),
        "{} lanes are further than 1e-3 from WARP",
        t.worst.len()
    );
    assert!(t.errors.is_empty(), "{} programs errored", t.errors.len());
    assert!(
        t.ir_refused.is_empty(),
        "the IR lift refuses {} pixel programs",
        t.ir_refused.len()
    );
    assert!(
        t.older_differ.is_empty(),
        "{} runs of the older lift differ from the IR lift's",
        t.older_differ.len()
    );
}

#[test]
fn sekiro_programs_agree_with_warp() {
    corpus(Game::Sekiro);
}
#[test]
fn riders_republic_programs_agree_with_warp() {
    corpus(Game::Riders);
}
#[test]
fn unity_programs_agree_with_warp() {
    corpus(Game::Sledding);
}

/// A folder of containers outside the configured corpora, named by the debug key `shader.outside`; without it, nothing
/// to run.
#[test]
fn outside_programs_agree_with_warp() {
    let Some(root) = debug("shader.outside").map(PathBuf::from) else {
        return;
    };
    let mut paths = Vec::new();
    common::containers(&root, &mut paths);
    corpus_over(&root.display().to_string(), &paths);
}

/// A pixel program asking `resinfo` of a texture of each shape the oracle binds, at the level its input chooses (0, or
/// 1: past the oracle's one level, where the size is zero); `UINT` asks for integers.
const RESINFO_HLSL: &str = r#"
Texture1D<float4> t0 : register(t0);
Texture1DArray<float4> t1 : register(t1);
Texture2D<float4> t2 : register(t2);
Texture2DArray<float4> t3 : register(t3);
Texture3D<float4> t4 : register(t4);
TextureCube<float4> t5 : register(t5);
#ifdef UINT
#define T uint
#define V uint4
#else
#define T float
#define V float4
#endif
struct Out {
    V a : SV_Target0; V b : SV_Target1; V c : SV_Target2; V d : SV_Target3; V e : SV_Target4; V f : SV_Target5;
};
Out main(float4 p : SV_Position, float4 lv : LEVEL) {
    uint l = lv.x > 0.75 ? 1 : 0;
    Out o;
    T w, h, d, n;
    t0.GetDimensions(l, w, n); o.a = V(w, 0, 0, n);
    t1.GetDimensions(l, w, d, n); o.b = V(w, d, 0, n);
    t2.GetDimensions(l, w, h, n); o.c = V(w, h, 0, n);
    t3.GetDimensions(l, w, h, d, n); o.d = V(w, h, d, n);
    t4.GetDimensions(l, w, h, d, n); o.e = V(w, h, d, n);
    t5.GetDimensions(l, w, h, n); o.f = V(w, h, 0, n);
    return o;
}
"#;

/// `bytes` with its instructions' tokens changed by `change` (each instruction, in order, and the program's words),
/// and the container's digest repaired.
fn patched(bytes: &[u8], mut change: impl FnMut(&dxbc_reader::decode::Instruction, &mut [u32])) -> Vec<u8> {
    let parts = dxcontainer::chunks(bytes).unwrap();
    let code = parts
        .iter()
        .find(|c| matches!(&c.kind, b"SHDR" | b"SHEX"))
        .unwrap();
    let mut words: Vec<u32> = code
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let decoded = dxbc_reader::decode::decode(&words).unwrap();
    for ins in &decoded.instructions {
        change(ins, &mut words);
    }
    let data: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut out = dxcontainer::replace_part(bytes, code.kind, &data).unwrap();
    dxcontainer::repair_digest(&mut out).unwrap();
    out
}

/// `bytes` with every `resinfo` instruction's return type (its opcode token's bits 11 and 12) made `ret`; how many it
/// changed.
fn with_resinfo_return(bytes: &[u8], ret: u32) -> (Vec<u8>, usize) {
    let mut changed = 0;
    let out = patched(bytes, |ins, words| {
        if ins.name() == "resinfo" {
            words[ins.word] = (words[ins.word] & !(3 << 11)) | ret << 11;
            changed += 1;
        }
    });
    (out, changed)
}

/// `resinfo` against WARP, in each of its three return types: float and uint as `fxc` writes them, and `_rcpFloat`,
/// which no compiler writes (the float program with its instructions' return type changed): the reciprocals of the
/// size's lanes, the layers and the level count as they are. Each program runs through both lifts.
#[test]
fn resinfo_agrees_with_warp() {
    let (Some(fxc), Some(warp)) = (tool(Tool::Fxc), tool(Tool::Warp)) else {
        return;
    };
    let scratch = inputs::own_scratch("warp-resinfo");
    let source = scratch.join("resinfo.hlsl");
    std::fs::write(&source, RESINFO_HLSL).unwrap();
    let compile = |name: &str, defines: &[&str]| -> Vec<u8> {
        let out = scratch.join(format!("{name}.dxbc"));
        let mut command = std::process::Command::new(&fxc);
        command.args(["/nologo", "/T", "ps_5_0", "/E", "main"]);
        for d in defines {
            command.args(["/D", d]);
        }
        let compiled = command.arg("/Fo").arg(&out).arg(&source).output().unwrap();
        assert!(
            compiled.status.success(),
            "fxc: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        std::fs::read(&out).unwrap()
    };
    let float = compile("resinfo_float", &[]);
    let uint = compile("resinfo_uint", &["UINT=1"]);
    let (rcp, changed) = with_resinfo_return(&float, 1);
    assert_eq!(changed, 6, "the float program's resinfo instructions");
    let returns = |bytes: &[u8]| -> Vec<u32> {
        let parts = dxcontainer::chunks(bytes).unwrap();
        let code = parts
            .iter()
            .find(|c| matches!(&c.kind, b"SHDR" | b"SHEX"))
            .unwrap();
        let words: Vec<u32> = code
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        dxbc_reader::decode::decode(&words)
            .unwrap()
            .instructions
            .iter()
            .filter(|i| i.name() == "resinfo")
            .map(|i| i.control & 3)
            .collect()
    };
    assert_eq!(returns(&float), [0; 6], "fxc's float resinfo");
    assert_eq!(returns(&uint), [2; 6], "fxc's uint resinfo");
    assert_eq!(returns(&rcp), [1; 6], "the patched resinfo");
    let tools = (fxc, warp, scratch);
    let mut t = Tally::default();
    for (name, bytes) in [
        ("resinfo_float.dxbc", &float),
        ("resinfo_uint.dxbc", &uint),
        ("resinfo_rcp.dxbc", &rcp),
    ] {
        if let Err(e) = iridia_dxbc::lift(bytes) {
            panic!("{name}: {e}");
        }
        run_program(name, bytes, 4, true, &tools, &mut t);
    }
    report("resinfo", &t);
    assert_eq!(t.selected, 3, "every program binds");
    assert!(t.compared >= 6, "{} runs compared", t.compared);
    assert!(t.errors.is_empty(), "{} programs errored", t.errors.len());
    assert!(
        t.worst.is_empty(),
        "{} lanes are further than 1e-3 from WARP",
        t.worst.len()
    );
    assert!(t.ir_refused.is_empty(), "the IR lift refuses");
    assert!(
        t.older_differ.is_empty(),
        "{} runs of the older lift differ from the IR lift's",
        t.older_differ.len()
    );
}

/// Two switches whose cases end with `break`, for [`switches_agree_with_warp_and_a_falling_case_is_refused`] to make
/// some fall into the next: through an `if` with a `break` of its own, through a loop with one, a chain into the
/// default and off the end.
const SWITCH_HLSL: &str = r#"
float4 main(float4 pos : SV_Position, float4 s : SELECT) : SV_Target {
    float4 r = s * 1.5;
    uint k = (uint)(abs(s.x) * 37.0) % 7;
    [forcecase] switch (k) {
    case 0: r.x += 1; break;
    case 1: r.y += 2; if (s.y > 0.5) { r.z += 3; break; } break;
    case 2: r.w *= 2; break;
    case 3: r.x -= 5; break;
    default: r.yz += 7; break;
    case 5: r.w += s.z; break;
    }
    [forcecase] switch ((uint)(abs(s.w) * 13.0) & 3) {
    case 0: r *= 0.5; break;
    case 1: [loop] for (uint j = 0; j < k; j++) { r.x += j; if (r.x > 9) break; } break;
    case 2: r.y -= 1; break;
    }
    return r;
}
"#;

/// The `break`s of [`SWITCH_HLSL`]'s program (in order, from 1) made `nop`: case 0 falls into case 1, case 2 into 3,
/// case 5 into the default and on off the end of the switch; in the second switch case 0 into 1 and 1 into 2. HLSL
/// refuses a case that falls into the next after code of its own, and so does Direct3D 11 (the Direct3D 11.3
/// specification, 22.7.18: Direct3D 10 allowed it); WARP runs such a case as if it ended with `break`, in a
/// `ps_4_0` program and a `ps_5_0` one alike.
const FALLING: [usize; 6] = [1, 4, 6, 7, 8, 10];

/// [`SWITCH_HLSL`] as `ps_4_0` and `ps_5_0` against WARP, through both lifts; with [`FALLING`]'s breaks made `nop`,
/// the IR lift refuses each by name.
#[test]
fn switches_agree_with_warp_and_a_falling_case_is_refused() {
    let (Some(fxc), Some(warp)) = (tool(Tool::Fxc), tool(Tool::Warp)) else {
        return;
    };
    let scratch = inputs::own_scratch("warp-switch");
    let source = scratch.join("switch.hlsl");
    std::fs::write(&source, SWITCH_HLSL).unwrap();
    let compile = |profile: &str| -> Vec<u8> {
        let out = scratch.join(format!("switch_{profile}.dxbc"));
        let compiled = std::process::Command::new(&fxc)
            .args(["/nologo", "/T", profile, "/E", "main", "/Fo"])
            .arg(&out)
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "fxc: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        std::fs::read(&out).unwrap()
    };
    let falling = |breaks: &[u8]| -> Vec<u8> {
        let nop = 58;
        let mut seen = 0;
        let out = patched(breaks, |ins, words| {
            if ins.name() == "break" {
                seen += 1;
                if FALLING.contains(&seen) {
                    words[ins.word] = (words[ins.word] & !0x7ff) | nop;
                }
            }
        });
        assert_eq!(seen, 12, "the program's breaks");
        out
    };
    let (breaks4, breaks5) = (compile("ps_4_0"), compile("ps_5_0"));
    for falling in [falling(&breaks4), falling(&breaks5)] {
        match iridia_dxbc::ir::lift(&falling) {
            Err(e) => assert!(e.to_string().contains("which Direct3D 11 does not allow"), "{e}"),
            Ok(_) => panic!("the IR lift takes a case that falls into the next after code of its own"),
        }
    }
    let tools = (fxc, warp, scratch);
    let mut t = Tally::default();
    for (name, bytes) in [
        ("switch_breaks4.dxbc", &breaks4),
        ("switch_breaks5.dxbc", &breaks5),
    ] {
        if let Err(e) = iridia_dxbc::lift(bytes) {
            panic!("{name}: {e}");
        }
        run_program(name, bytes, 48, true, &tools, &mut t);
    }
    report("switch", &t);
    assert_eq!(t.selected, 2, "every program binds");
    assert!(t.compared >= 96, "{} runs compared", t.compared);
    assert!(t.errors.is_empty(), "{} programs errored", t.errors.len());
    assert!(
        t.worst.is_empty(),
        "{} lanes are further than 1e-3 from WARP",
        t.worst.len()
    );
    assert!(t.ir_refused.is_empty(), "the IR lift refuses: {:?}", t.ir_refused);
    assert!(
        t.older_differ.is_empty(),
        "{} runs of the older lift differ from the IR lift's",
        t.older_differ.len()
    );
}

/// A pixel program of doubles: `fxc` writes every double-precision instruction of Shader Model 5 for it (`dadd`,
/// `dmul`, `ddiv`, `dfma`, `drcp`, `dmin`, `dmax`, `dmov`, `dmovc`, the four comparisons, and the conversions both
/// ways), the doubles' bits written out as integers and their conversions to float.
const DOUBLES_HLSL: &str = r#"
struct Out { float4 f : SV_Target0; uint4 a : SV_Target1; uint4 b : SV_Target2; uint4 c : SV_Target3; };
Out main(float4 p : SV_Position, float4 s : SELECT) {
    double x = (double)s.x * 10000.0L - 3.0L;
    double y = (double)s.y / ((double)s.z - 0.5L);
    double z = fma(x, y, (double)s.w);
    double m = min(x, y);
    double big = max(x, z);
    double r = rcp(y);
    double sel = s.w > 0.5 ? x : -y;
    int i = (int)x;
    uint u = (uint)(x * x);
    double fromi = (double)(int)(s.x * 100.0);
    double fromu = (double)(uint)(s.y * 100.0);
    double acc = x;
    [loop] for (int k = 0; k < (int)(s.z * 3.0); k++) { acc = acc * 0.5L + y; }
    big = max(big, acc);
    Out o;
    o.f = float4((float)x, (float)z, (float)m, (float)r);
    asuint(z, o.a.x, o.a.y);
    asuint(big, o.a.z, o.a.w);
    asuint(y, o.b.x, o.b.y);
    asuint(sel, o.b.z, o.b.w);
    uint flags = (x == y ? 1u : 0u) | (x >= z ? 2u : 0u) | (x < y ? 4u : 0u) | (x != z ? 8u : 0u);
    o.c = uint4(asuint(i), u, flags, asuint((float)(fromi + fromu)));
    return o;
}
"#;

/// [`DOUBLES_HLSL`] against WARP through the IR lift: each double's bits exact, each conversion to float within the
/// comparison's tolerance (`drcp` is within one unit in the last place). The older lift refuses the program.
#[test]
fn doubles_agree_with_warp() {
    let (Some(fxc), Some(warp)) = (tool(Tool::Fxc), tool(Tool::Warp)) else {
        return;
    };
    let scratch = inputs::own_scratch("warp-doubles");
    let source = scratch.join("doubles.hlsl");
    std::fs::write(&source, DOUBLES_HLSL).unwrap();
    let out = scratch.join("doubles.dxbc");
    let compiled = std::process::Command::new(&fxc)
        .args(["/nologo", "/T", "ps_5_0", "/E", "main", "/Fo"])
        .arg(&out)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "fxc: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let bytes = std::fs::read(&out).unwrap();
    let parts = dxcontainer::chunks(&bytes).unwrap();
    let code = parts
        .iter()
        .find(|c| matches!(&c.kind, b"SHDR" | b"SHEX"))
        .unwrap();
    let words: Vec<u32> = code
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let names: std::collections::BTreeSet<&str> = dxbc_reader::decode::decode(&words)
        .unwrap()
        .instructions
        .iter()
        .map(|i| i.name())
        .collect();
    for name in [
        "dadd", "dmul", "ddiv", "dfma", "drcp", "dmin", "dmax", "dmov", "dmovc", "deq", "dge", "dlt", "dne",
        "dtof", "dtoi", "dtou", "ftod", "itod", "utod",
    ] {
        assert!(names.contains(name), "fxc wrote no {name}");
    }
    assert!(iridia_dxbc::lift(&bytes).is_err(), "the older lift takes doubles");
    let tools = (fxc, warp, scratch);
    let mut t = Tally::default();
    run_program("doubles.dxbc", &bytes, 64, true, &tools, &mut t);
    report("doubles", &t);
    assert_eq!(t.selected, 1, "the program binds");
    assert!(t.compared >= 64, "{} runs compared", t.compared);
    assert!(t.errors.is_empty(), "{} programs errored", t.errors.len());
    assert!(
        t.worst.is_empty(),
        "{} lanes are further than 1e-3 from WARP",
        t.worst.len()
    );
    assert!(t.ir_refused.is_empty(), "the IR lift refuses: {:?}", t.ir_refused);
}

/// A pixel program that moves a denormal float or a zero (the input's bits, either sign) through `mov` with each
/// source modifier (`|x|`, `-x`, `-|x|`) and `mov_sat`, its words written out as integers.
const DENORMAL_MOVES_HLSL: &str = r#"
uint4 main(float4 p : SV_Position, nointerpolation uint4 b : BITS) : SV_Target {
    float x = asfloat(b.x | (b.y & 1) << 31);
    return uint4(asuint(abs(x)), asuint(-x), asuint(-abs(x)), asuint(saturate(x)));
}
"#;

/// [`DENORMAL_MOVES_HLSL`] against WARP through both lifts, every lane exact: which moves keep a denormal and which
/// flush it are Direct3D's.
#[test]
fn denormals_through_moves_agree_with_warp() {
    let (Some(fxc), Some(warp)) = (tool(Tool::Fxc), tool(Tool::Warp)) else {
        return;
    };
    let scratch = inputs::own_scratch("warp-denormal-moves");
    let source = scratch.join("denormal.hlsl");
    std::fs::write(&source, DENORMAL_MOVES_HLSL).unwrap();
    let out = scratch.join("denormal.dxbc");
    let compiled = std::process::Command::new(&fxc)
        .args(["/nologo", "/T", "ps_5_0", "/E", "main", "/Fo"])
        .arg(&out)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "fxc: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let bytes = std::fs::read(&out).unwrap();
    let tools = (fxc, warp, scratch);
    let mut t = Tally::default();
    run_program("denormal.dxbc", &bytes, 16, true, &tools, &mut t);
    report("denormal moves", &t);
    assert_eq!(t.selected, 1, "the program binds");
    assert!(t.compared >= 16, "{} runs compared", t.compared);
    assert!(t.errors.is_empty(), "{:?}", t.errors);
    assert!(
        t.lanes.keys().all(|c| *c == Class::Exact),
        "lanes not exact: {:?}",
        t.lanes
    );
    assert!(
        t.older_differ.is_empty(),
        "the older lift is not the IR lift: {:#?}",
        t.older_differ
    );
}

/// A compute program taking signed and unsigned minimums and maximums of one raw buffer's words: four threads, each
/// with its own negated index.
const MIXED_MIN_MAX_HLSL: &str = r#"
RWByteAddressBuffer b : register(u0);
[numthreads(4, 1, 1)]
void main(uint t : SV_DispatchThreadID) {
    int v = -(int)t;
    b.InterlockedMin(0, v);
    b.InterlockedMin(4, (uint)v);
    b.InterlockedMax(8, v);
    b.InterlockedMax(12, (uint)v);
}
"#;

/// A backend for a compute program that holds its memory and samples nothing.
struct Holding;

impl iridia_exec::Backend for Holding {
    fn sample(&mut self, _: iridia_exec::ImageSample<iridia_exec::Element>) -> Result<[f32; 4], String> {
        Err("the program samples nothing".into())
    }
}

/// [`MIXED_MIN_MAX_HLSL`] through the IR lift on words that start at 5: `imin` and `imax` compare as signed, `umin`
/// and `umax` as unsigned (-1 is the largest), as their definitions state (the Direct3D 11.3 specification, 22.17).
/// The older lift refuses it by name: its atomics compare as the memory's one type.
#[test]
fn a_buffer_takes_signed_and_unsigned_minimums() {
    let Some(fxc) = tool(Tool::Fxc) else {
        return;
    };
    let scratch = inputs::own_scratch("mixed-min-max");
    let source = scratch.join("minmax.hlsl");
    std::fs::write(&source, MIXED_MIN_MAX_HLSL).unwrap();
    let out = scratch.join("minmax.dxbc");
    let compiled = std::process::Command::new(&fxc)
        .args(["/nologo", "/T", "cs_5_0", "/E", "main", "/Fo"])
        .arg(&out)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "fxc: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let bytes = std::fs::read(&out).unwrap();
    let refused = iridia_dxbc::lift(&bytes).unwrap_err().to_string();
    assert!(refused.contains("on memory whose atomics are"), "{refused}");
    let program = iridia_dxbc::ir::lift(&bytes).unwrap();
    let exe = iridia_exec::Executable::from_ir(&program).unwrap();
    let id = program.interface.resources[0].id;
    let resources = iridia_exec::Resources::default();
    let bound = exe.bind_resources(&resources).unwrap();
    let mut buffers = BTreeMap::from([(id, vec![5u32; 4])]);
    exe.dispatch(&bound, &mut Holding, &mut buffers, [1, 1, 1], 4)
        .unwrap();
    assert_eq!(buffers[&id], [(-3i32).cast_unsigned(), 0, 5, u32::MAX]);
}

/// A pixel program that reads its front-facing flag as an integer: Direct3D gives all ones for a front face, so its
/// bits survive an `and` with a float's.
const FRONT_FACE_HLSL: &str = r#"
uint4 main(float4 p : SV_Position, uint f : SV_IsFrontFace) : SV_Target {
    return uint4(f, f & 0x3f800000, f >> 31, 1);
}
"#;

/// [`FRONT_FACE_HLSL`] against WARP through both lifts, on front and back faces: the flag's word is Direct3D's.
#[test]
fn a_front_face_flag_is_all_ones() {
    let (Some(fxc), Some(warp)) = (tool(Tool::Fxc), tool(Tool::Warp)) else {
        return;
    };
    let scratch = inputs::own_scratch("warp-front-face");
    let source = scratch.join("front.hlsl");
    std::fs::write(&source, FRONT_FACE_HLSL).unwrap();
    let out = scratch.join("front.dxbc");
    let compiled = std::process::Command::new(&fxc)
        .args(["/nologo", "/T", "ps_5_0", "/E", "main", "/Fo"])
        .arg(&out)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "fxc: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let bytes = std::fs::read(&out).unwrap();
    let tools = (fxc, warp, scratch);
    let mut t = Tally::default();
    run_program("front.dxbc", &bytes, 8, true, &tools, &mut t);
    report("front face", &t);
    assert_eq!(t.selected, 1, "the program binds");
    assert!(t.compared >= 8, "{} runs compared", t.compared);
    assert!(t.errors.is_empty(), "{:?}", t.errors);
    assert!(t.worst.is_empty(), "the IR lift is not WARP: {:#?}", t.worst);
    assert!(
        t.older_differ.is_empty(),
        "the older lift is not the IR lift: {:#?}",
        t.older_differ
    );
}

/// A pixel program that writes unordered-access memory before and after a `discard` it may take, each lane of its quad
/// at its own addresses (lane `v.y`), and takes a derivative after it, so a discarded lane runs on as a helper.
const DISCARD_HLSL: &str = r#"
RWStructuredBuffer<uint> u : register(u1);
RWTexture2D<uint> t : register(u2);
float4 main(float4 pos : SV_Position, float2 v : TEXCOORD0) : SV_Target {
    uint i = (uint)v.y;
    u[i * 4] = 1;
    t[uint2(i, 0)] = 1;
    uint before;
    InterlockedAdd(u[i * 4 + 1], 1, before);
    if (v.x > 0.5) {
        discard;
    }
    u[i * 4 + 2] = 2;
    t[uint2(i, 1)] = 2;
    uint after;
    InterlockedAdd(u[i * 4 + 3], 1, after);
    InterlockedMax(t[uint2(i, 2)], 5);
    return float4(before, after, ddx(v.x), 1);
}
"#;

/// Every store (the lanes it writes: `store_uav_typed` writes all four, the view's format takes its own) and atomic
/// operation a run asks for, in order; an atomic's word was 7. `memory` is what the buffer stores leave, by view,
/// address and lane, however a lift splits them into calls.
#[derive(Default)]
struct Effects(Vec<String>, BTreeMap<(u32, [u32; 3], usize), u32>);

impl Sampler for Effects {
    fn sample(&mut self, _: iridia_exec::Sample) -> Result<[f32; 4], String> {
        Ok([0.0; 4])
    }
    fn store(&mut self, r: iridia_exec::Store) -> Result<(), String> {
        let lanes: Vec<u32> = (0..4)
            .filter(|k| r.mask >> k & 1 != 0)
            .map(|k| r.value[k])
            .collect();
        self.0
            .push(format!("store u{} {:?} {lanes:?}", r.view, r.address));
        for k in (0..4).filter(|k| r.mask >> k & 1 != 0) {
            self.1.insert((r.view, r.address, k), r.value[k]);
        }
        Ok(())
    }
    fn store_image(&mut self, r: iridia_exec::ImageStore<u32>) -> Result<(), String> {
        let lanes = &r.value[..usize::from(r.lanes)];
        self.0.push(format!("store u{} {:?} {lanes:?}", r.view, r.texel));
        Ok(())
    }
    fn atomic(&mut self, r: iridia_exec::Atomic) -> Result<u32, String> {
        self.0
            .push(format!("{:?} u{} {:?} {}", r.op, r.view, r.address, r.value));
        Ok(7)
    }
    fn atomic_image(&mut self, r: iridia_exec::ImageAtomic<u32>) -> Result<u32, String> {
        self.0
            .push(format!("{:?} u{} {:?} {}", r.op, r.view, r.texel, r.value));
        Ok(7)
    }
}

/// A pixel discarded changes no memory after its `discard` and keeps what it changed before (the Direct3D 11.3
/// specification, 16.13), in both lifts: the program writes a structured buffer, a texture and atomics either side.
#[test]
fn a_discarded_pixel_changes_no_memory_after_it() {
    let Some(fxc) = tool(Tool::Fxc) else {
        return;
    };
    let scratch = inputs::own_scratch("discard-memory");
    let source = scratch.join("discard.hlsl");
    std::fs::write(&source, DISCARD_HLSL).unwrap();
    let out = scratch.join("discard.dxbc");
    let compiled = std::process::Command::new(&fxc)
        .args(["/nologo", "/T", "ps_5_0", "/E", "main", "/Fo"])
        .arg(&out)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "fxc: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let bytes = std::fs::read(&out).unwrap();
    let lifts = [
        (
            "older",
            ProgramShader::new(iridia_dxbc::lift(&bytes).unwrap()).unwrap(),
        ),
        (
            "IR",
            ProgramShader::from_ir(iridia_dxbc::ir::lift(&bytes).unwrap()).unwrap(),
        ),
    ];
    // what lane `i` does before its discard and after it
    let before = |i: u32| {
        [
            format!("store u1 [{}, 0, 0] [1]", i * 4),
            format!("store u2 [{i}, 0, 0] [1, 1, 1, 1]"),
            format!("Add u1 [{}, 0, 0] 1", i * 4 + 1),
        ]
    };
    let after = |i: u32| {
        [
            format!("store u1 [{}, 0, 0] [2]", i * 4 + 2),
            format!("store u2 [{i}, 1, 0] [2, 2, 2, 2]"),
            format!("Add u1 [{}, 0, 0] 1", i * 4 + 3),
            format!("Max u2 [{i}, 2, 0] 5"),
        ]
    };
    for (name, shader) in &lifts {
        assert!(shader.needs_quad(), "{name} lift: the program runs as a quad");
        // lanes 2 and 3 discard
        let xs = [0.25f32, 0.25, 0.75, 0.75];
        let lane = |i: usize| {
            let mut inputs = vec![[0u32; 4]; shader.input_count().max(2)];
            inputs[1] = [xs[i].to_bits(), (i as f32).to_bits(), 0, 0];
            inputs
        };
        let quad = [lane(0), lane(1), lane(2), lane(3)];
        let mut effects = Effects::default();
        let e = shader
            .execute_quad([&quad[0], &quad[1], &quad[2], &quad[3]], &[], &mut effects)
            .unwrap();
        let discarded: Vec<bool> = e.iter().map(|e| e.discarded).collect();
        assert_eq!(discarded, [false, false, true, true], "{name} lift");
        let mut want: Vec<String> = (0..4u32).flat_map(before).collect();
        want.extend((0..2u32).flat_map(after));
        want.sort();
        let mut got = effects.0.clone();
        got.sort();
        assert_eq!(got, want, "{name} lift: {:#?}", effects.0);
        println!(
            "{name} lift: {} effects, none of a discarded lane's after its discard",
            got.len()
        );
    }
}

/// A vertex program with two clip distances and a cull distance, and an output after them.
const DISTANCES_HLSL: &str = r#"
struct Out {
    float4 pos : SV_Position;
    float2 clip : SV_ClipDistance0;
    float cull : SV_CullDistance0;
    float4 t : TEXCOORD0;
};
Out main(float4 p : POSITION, float4 q : TEXCOORD0) {
    Out o;
    o.pos = p;
    o.clip = float2(1, 2);
    o.cull = 3;
    o.t = q;
    return o;
}
"#;

/// The register view of a program whose clip distances are an array of several: the outputs after them are their own
/// (each distance one result member), in both lifts.
#[test]
fn outputs_after_several_clip_distances_are_their_own() {
    let Some(fxc) = tool(Tool::Fxc) else {
        return;
    };
    let scratch = inputs::own_scratch("clip-distances");
    let source = scratch.join("distances.hlsl");
    std::fs::write(&source, DISTANCES_HLSL).unwrap();
    let out = scratch.join("distances.dxbc");
    let compiled = std::process::Command::new(&fxc)
        .args(["/nologo", "/T", "vs_5_0", "/E", "main", "/Fo"])
        .arg(&out)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "fxc: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let bytes = std::fs::read(&out).unwrap();
    let lifts = [
        (
            "older",
            ProgramShader::new(iridia_dxbc::lift(&bytes).unwrap()).unwrap(),
        ),
        (
            "IR",
            ProgramShader::from_ir(iridia_dxbc::ir::lift(&bytes).unwrap()).unwrap(),
        ),
    ];
    let q = [5.0f32, 6.0, 7.0, 8.0].map(f32::to_bits);
    for (name, shader) in &lifts {
        let texcoord = shader
            .interface()
            .outputs
            .iter()
            .find(|v| v.system.is_none())
            .and_then(|v| v.provenance.d3d)
            .expect("a TEXCOORD output")
            .register as usize;
        let mut inputs = vec![[0u32; 4]; shader.input_count().max(2)];
        inputs[0] = [1.0f32, 2.0, 3.0, 1.0].map(f32::to_bits);
        inputs[1] = q;
        let (e, position) = shader
            .execute_vertex(
                &inputs,
                &[],
                &mut |_: iridia_exec::Sample| -> Result<[f32; 4], String> { Ok([0.0; 4]) },
            )
            .unwrap();
        assert_eq!(e.outputs[texcoord], q, "{name} lift: TEXCOORD0 in o{texcoord}");
        assert_eq!(
            position,
            [1.0f32, 2.0, 3.0, 1.0].map(f32::to_bits),
            "{name} lift: the position"
        );
        println!("{name} lift: o{texcoord} = {:x?}", e.outputs[texcoord]);
    }
}

/// A pixel program that stores an element of two typed buffers.
const TYPED_STORE_HLSL: &str = r#"
RWBuffer<int4> u : register(u1);
RWBuffer<uint> v : register(u2);
float4 main(float4 pos : SV_Position) : SV_Target {
    u[1] = int4(-15, 16, -17, 18);
    v[3] = 7;
    InterlockedAdd(v[5], 3);
    return float4(1, 0, 0, 1);
}
"#;

/// Two texture views of a Shader Model 5.1 program at registers other than their ids (`U0[3:3]`, `U1[5:5]`): each
/// store reaches the view at its register, in both lifts and at 5.0 alike.
const UAV_REGISTERS_HLSL: &str = r#"
RWTexture2D<float> u : register(u3);
RWTexture2D<float4> v : register(u5);
float4 main() : SV_Target {
    u[uint2(0, 1)] = 0.5;
    v[uint2(0, 0)].yxwz = float4(1, 2, 3, 4);
    return 0;
}
"#;

#[test]
fn a_view_is_stored_at_its_register_not_its_id() {
    let Some(fxc) = tool(Tool::Fxc) else {
        return;
    };
    let scratch = inputs::own_scratch("uav-registers");
    let source = scratch.join("uav.hlsl");
    std::fs::write(&source, UAV_REGISTERS_HLSL).unwrap();
    let want = [
        format!("store u3 [0, 1, 0] {:?}", [0.5f32.to_bits(); 4]),
        format!(
            "store u5 [0, 0, 0] {:?}",
            [2.0f32, 1.0, 4.0, 3.0].map(f32::to_bits)
        ),
    ];
    for model in ["ps_5_0", "ps_5_1"] {
        let out = scratch.join(format!("uav.{model}.dxbc"));
        let compiled = std::process::Command::new(&fxc)
            .args(["/nologo", "/T", model, "/E", "main", "/Fo"])
            .arg(&out)
            .arg(&source)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "fxc: {}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let bytes = std::fs::read(&out).unwrap();
        let lifts = [
            (
                "older",
                ProgramShader::new(iridia_dxbc::lift(&bytes).unwrap()).unwrap(),
            ),
            (
                "IR",
                ProgramShader::from_ir(iridia_dxbc::ir::lift(&bytes).unwrap()).unwrap(),
            ),
        ];
        for (name, shader) in &lifts {
            let inputs = vec![[0u32; 4]; shader.input_count().max(1)];
            let mut effects = Effects::default();
            shader.execute(&inputs, &[], &mut effects).unwrap();
            assert_eq!(effects.0, want, "{model}, {name} lift");
            println!("{model}, {name} lift: {:?}", effects.0);
        }
    }
}

/// A pixel program's store to a typed buffer reaches its sampler as the element and the element's lanes ([`Store`]'s
/// typed address), in both lifts: `u[1]` is element 1, not word 4; an atomic on `v[5]` is element 5.
///
/// [`Store`]: iridia_exec::Store
#[test]
fn a_typed_buffer_store_is_its_element() {
    let Some(fxc) = tool(Tool::Fxc) else {
        return;
    };
    let scratch = inputs::own_scratch("typed-store");
    let source = scratch.join("typed.hlsl");
    std::fs::write(&source, TYPED_STORE_HLSL).unwrap();
    let out = scratch.join("typed.dxbc");
    let compiled = std::process::Command::new(&fxc)
        .args(["/nologo", "/T", "ps_5_0", "/E", "main", "/Fo"])
        .arg(&out)
        .arg(&source)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "fxc: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let bytes = std::fs::read(&out).unwrap();
    let lifts = [
        (
            "older",
            ProgramShader::new(iridia_dxbc::lift(&bytes).unwrap()).unwrap(),
        ),
        (
            "IR",
            ProgramShader::from_ir(iridia_dxbc::ir::lift(&bytes).unwrap()).unwrap(),
        ),
    ];
    let int = |x: i32| x.cast_unsigned();
    let mut want = BTreeMap::new();
    for (k, v) in [int(-15), 16, int(-17), 18].into_iter().enumerate() {
        want.insert((1, [1, 0, 0], k), v);
        want.insert((2, [3, 0, 0], k), 7);
    }
    for (name, shader) in &lifts {
        let inputs = vec![[0u32; 4]; shader.input_count().max(1)];
        let mut effects = Effects::default();
        shader.execute(&inputs, &[], &mut effects).unwrap();
        assert_eq!(effects.1, want, "{name} lift: {:?}", effects.0);
        let atomics: Vec<&String> = effects.0.iter().filter(|e| !e.starts_with("store")).collect();
        assert_eq!(atomics, ["Add u2 [5, 0, 0] 3"], "{name} lift");
        println!("{name} lift: {:?}", effects.0);
    }
}
