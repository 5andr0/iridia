#![allow(clippy::unwrap_used)]
//! The executor's ray queries, lifted onto the IR ([`iridia_dxil::ir`]), against Microsoft's WARP rasterizer
//! (Direct3D 12, DXR 1.1 inline ray tracing), the oracle that is not ours: the same scene (triangles, instances with
//! transforms, masks and flags) and the same rays go through a fixture on WARP and on the executor, and what each ray
//! leaves is compared: the integer fields (status, instance index and id, hit-group contribution, geometry, primitive,
//! facing, flags) exactly, the floats (distance, barycentrics, the ray in object space, the instance's matrices) to a
//! relative 1e-4 (the arithmetic of a hardware or software traversal is its own).
//!
//! Needs the WARP oracle program with the `d3d12rt` case form (the configured `Tool::Warp`) and the scratch folder. The
//! scenes and rays are made from fixed seeds.
//! WARP has returned different hits for identical cases. Run it twice, and only when it disagrees with itself run
//! a third time: report each unstable ray and compare its two agreeing answers with the executor. No pair agreeing
//! is a failure, as is any stable WARP answer that differs from the executor. Words after a miss are not written.
mod inputs;

use inputs::{Tool, own_scratch, program};
use iridia::D3dClass;
use iridia_exec::{
    Backend, Element, Executable, Geometry, ImageSample, Instance, InstanceFlags, Resources, Scene,
};
use std::collections::BTreeMap;
use std::path::Path;

struct NoBackend;
impl Backend for NoBackend {
    fn sample(&mut self, _: ImageSample<Element>) -> Result<[f32; 4], String> {
        Err("no sampling in this fixture".into())
    }
}

/// A small deterministic generator.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// A float in [lo, hi).
    fn float(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next() as f32 / (1u64 << 31) as f32)
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

fn bits(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}

/// A scene from a seed: bottom levels of several geometries of random triangles (some opaque), and instances with a
/// rotation about z, a scale and a translation, masks, ids and flags.
/// Eight spheres (centre, radius) the procedural primitives stand for.
fn spheres() -> Vec<[f32; 4]> {
    let mut r = Lcg(31337);
    (0..8)
        .map(|_| {
            [
                r.float(-2.5, 2.5),
                r.float(-2.5, 2.5),
                r.float(-2.5, 2.5),
                r.float(0.5, 1.5),
            ]
        })
        .collect()
}

fn scene(seed: u64, with_instance_flags: bool, procedural: bool) -> Scene {
    let mut r = Lcg(seed);
    let mut bottoms = Vec::new();
    for _ in 0..2 {
        let mut geometries = Vec::new();
        for _ in 0..(1 + r.below(3)) {
            let n = 6 + r.below(10);
            let mut positions = Vec::new();
            let centre = [r.float(-2.0, 2.0), r.float(-2.0, 2.0), r.float(-2.0, 2.0)];
            for _ in 0..n * 3 {
                positions.push([
                    centre[0] + r.float(-2.5, 2.5),
                    centre[1] + r.float(-2.5, 2.5),
                    centre[2] + r.float(-2.5, 2.5),
                ]);
            }
            geometries.push(Geometry {
                positions,
                boxes: Vec::new(),
                indices: None,
                opaque: r.below(2) == 0,
            });
        }
        if procedural {
            // four boxes, each bounding sphere k with room to spare, so that some rays enter a box and miss the sphere
            let boxes = spheres()
                .iter()
                .take(4)
                .map(|s| {
                    let slack = 1.3 * s[3];
                    [
                        [s[0] - slack, s[1] - slack, s[2] - slack],
                        [s[0] + slack, s[1] + slack, s[2] + slack],
                    ]
                })
                .collect();
            geometries.push(Geometry {
                positions: Vec::new(),
                boxes,
                indices: None,
                opaque: r.below(2) == 0,
            });
        }
        bottoms.push(geometries);
    }
    let mut instances = Vec::new();
    for k in 0..3 {
        let angle = r.float(0.0, std::f32::consts::TAU);
        let scale = r.float(0.6, 1.6);
        let (s, c) = angle.sin_cos();
        let t = [r.float(-3.0, 3.0), r.float(-3.0, 3.0), r.float(-3.0, 3.0)];
        let transform = [
            [c * scale, -s * scale, 0.0, t[0]],
            [s * scale, c * scale, 0.0, t[1]],
            [0.0, 0.0, scale, t[2]],
        ];
        let flags = if with_instance_flags {
            let f = r.below(8);
            InstanceFlags {
                cull_disable: f & 1 != 0,
                front_counter_clockwise: f & 2 != 0,
                // DXR's FORCE_OPAQUE and FORCE_NON_OPAQUE instance flags are mutually exclusive.
                force_opaque: f & 4 != 0 && f != 7,
                force_non_opaque: f == 7,
            }
        } else {
            InstanceFlags::default()
        };
        instances.push(Instance {
            bottom: (r.below(2)) as usize,
            transform,
            id: 100 + 7 * k,
            mask: [0xff, 0x01, 0x02][r.below(3) as usize],
            contribution: 10 * k + r.below(5),
            flags,
        });
    }
    Scene { bottoms, instances }
}

/// 64 rays: origin, direction, flags, mask. With `outside`, every origin is far from the scene (further than any
/// instance of it reaches), so that no ray starts inside a procedural primitive's box: WARP offers no candidate for a box
/// the ray starts in (its entry before the ray's t min), which DXR does not say, so the gate leaves such rays out.
fn rays(seed: u64, outside: bool) -> Vec<RayDesc> {
    let mut r = Lcg(seed);
    let flag_sets = [
        0u32,
        0x01,
        0x02,
        0x10,
        0x20,
        0x40,
        0x80,
        0x100,
        0x10 | 0x80,
        0x20 | 0x40,
        0x01 | 0x20,
    ];
    (0..64)
        .map(|_| {
            let mut origin = [r.float(-5.0, 5.0), r.float(-5.0, 5.0), r.float(-5.0, 5.0)];
            if outside {
                // push the origin out along itself to a distance of 16 or more
                let length = origin.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-3);
                origin = origin.map(|x| x / length * 16.0);
            }
            let target = [r.float(-3.0, 3.0), r.float(-3.0, 3.0), r.float(-3.0, 3.0)];
            let direction = [
                target[0] - origin[0],
                target[1] - origin[1],
                target[2] - origin[2],
            ];
            (
                origin,
                direction,
                flag_sets[r.below(flag_sets.len() as u32) as usize],
                [0xff, 0xff, 0x01, 0x02, 0x03][r.below(5) as usize],
            )
        })
        .collect()
}

/// Whether two floats' bits agree to a relative 1e-4 (of 1e-2 at least).
fn close(x: u32, y: u32) -> bool {
    let (x, y) = (f32::from_bits(x), f32::from_bits(y));
    (x - y).abs() <= 1e-4 * x.abs().max(y.abs()).max(1e-2)
}

#[test]
fn ray_queries_agree_with_warp() {
    hits("ray_query_grid", false, Cs::Fixture);
}

#[test]
fn procedural_primitives_agree_with_warp() {
    hits("ray_query_procedural", true, Cs::Fixture);
}

/// The same scenes and rays through the program's naga module (iridia-naga's lowering) written as HLSL by stock naga: the
/// module's ray queries as naga's HLSL writer, the one wgpu runs on Direct3D 12, gives them.
#[test]
fn ray_queries_through_naga_hlsl_agree_with_warp() {
    hits("ray_query_grid", false, Cs::NagaHlsl);
}

#[test]
fn procedural_primitives_through_naga_hlsl_agree_with_warp() {
    hits("ray_query_procedural", true, Cs::NagaHlsl);
}

/// `ray_query_grid.hlsl` and `ray_query_procedural.hlsl`: the committed hit of every ray, ten words.
fn hits(fixture_name: &str, procedural: bool, cs: Cs) {
    let Some(run) = gate(fixture_name, procedural, 10, cs) else {
        return;
    };
    let (mut hits, mut misses, mut mismatches) = (0, 0, Vec::new());
    for (name, rays, a, b) in run.rays() {
        let i = rays.0;
        let (o, d, f, m) = rays.1;
        if a[0] != b[0] {
            mismatches.push(format!(
                "{name} ray {i} (origin {o:?}, direction {d:?}, flags {f:#x}, mask {m:#x}): status executor {}, WARP {}",
                a[0], b[0]
            ));
            continue;
        }
        if a[0] == 0 {
            misses += 1;
            continue;
        }
        hits += 1;
        // a triangle also has barycentrics and a facing; a procedural hit has neither (the words are 0)
        let same = [2, 3, 4, 5, 6, 9].iter().all(|&k| a[k] == b[k])
            && close(a[1], b[1])
            && (a[0] != 1 || (close(a[7], b[7]) && close(a[8], b[8])));
        if !same {
            mismatches.push(format!(
                "{name} ray {i} (origin {o:?}, direction {d:?}, flags {f:#x}, mask {m:#x}):\n    executor {a:08x?}\n    WARP     {b:08x?}"
            ));
        }
    }
    println!(
        "{fixture_name}: {} rays: {hits} hits and {misses} misses agree with WARP, {} differ",
        12 * 64,
        mismatches.len()
    );
    for m in &mismatches {
        println!("MISMATCH {m}");
    }
    assert!(hits > 0 && mismatches.is_empty());
}

/// `ray_query_object.hlsl`: what a query reports of the ray it was traced with (its flags joined with the query's
/// declared `RAY_FLAG_SKIP_CLOSEST_HIT_SHADER`, origin, direction, t min), of the committed hit's instance (the ray in
/// its object space, its matrices), and of every procedural candidate a query of `RAY_FLAG_SKIP_TRIANGLES` that commits
/// nothing is offered (whether it is non-opaque, the ray in its instance's object space).
#[test]
fn object_space_rays_and_candidate_opacity_agree_with_warp() {
    object_space("ray_query_object", Cs::Fixture);
}

/// `ray_query_object` without the candidate's opacity, which iridia-naga does not lower ([`fixture_dxil`]), through the
/// program's naga module written as HLSL.
#[test]
fn object_space_rays_through_naga_hlsl_agree_with_warp() {
    object_space("ray_query_object_fields", Cs::NagaHlsl);
}

fn object_space(fixture_name: &str, cs: Cs) {
    let Some(run) = gate(fixture_name, true, 132, cs) else {
        return;
    };
    // the fixture with the candidate's opacity tells a non-opaque primitive from an opaque one; without it every candidate
    // is a 1
    let opacity = fixture_name == "ray_query_object";
    let (mut hits, mut candidates, mut non_opaque, mut mismatches) = (0, 0, 0, Vec::new());
    for (name, rays, a, b) in run.rays() {
        let i = rays.0;
        let (o, d, f, m) = rays.1;
        let mut differ = Vec::new();
        if a[0] != b[0] || a[1] != b[1] {
            differ.push("status or flags");
        }
        if !(2..9).all(|k| close(a[k], b[k])) {
            differ.push("world ray");
        }
        if a[0] == b[0] && a[0] != 0 {
            hits += 1;
            if !(9..40).all(|k| close(a[k], b[k])) || a[40..43] != b[40..43] {
                differ.push("committed hit, its object ray or matrices");
            }
        }
        for slot in (48..132).step_by(7) {
            if a[slot] != b[slot] {
                differ.push("candidate offered or opacity");
            } else if a[slot] != 0 {
                candidates += 1;
                non_opaque += usize::from(a[slot] == 2);
                if !(slot + 1..slot + 7).all(|k| close(a[k], b[k])) {
                    differ.push("candidate object ray");
                }
            }
        }
        if !differ.is_empty() {
            mismatches.push(format!(
                "{name} ray {i} (origin {o:?}, direction {d:?}, flags {f:#x}, mask {m:#x}): {}:\n    executor {a:08x?}\n    WARP     {b:08x?}",
                differ.join(", ")
            ));
        }
    }
    println!(
        "ray_query_object: {} rays, {hits} hits and {candidates} procedural candidates ({non_opaque} non-opaque) \
         agree with WARP, {} differ",
        12 * 64,
        mismatches.len()
    );
    for m in &mismatches {
        println!("MISMATCH {m}");
    }
    assert!(
        hits > 0
            && candidates > 0
            && (!opacity || (candidates > non_opaque && non_opaque > 0))
            && mismatches.is_empty()
    );
}

/// A ray's origin, direction, flags and mask.
type RayDesc = ([f32; 3], [f32; 3], u32, u32);

/// A ray of a case: its number, and the ray.
type Ray = (usize, RayDesc);

/// One fixture's cases run on the executor and on WARP: `words` words a ray of each.
struct Run {
    words: usize,
    ours: BTreeMap<String, Vec<u32>>,
    theirs: BTreeMap<String, Vec<u32>>,
    cases: Vec<(String, Vec<RayDesc>)>,
}

impl Run {
    /// Each ray of each case: the case, the ray, and the words the executor and WARP left for it.
    fn rays(&self) -> Vec<(&str, Ray, &[u32], &[u32])> {
        let mut out = Vec::new();
        for (name, rays) in &self.cases {
            let (mine, theirs) = (&self.ours[name], &self.theirs[name]);
            for (i, ray) in rays.iter().enumerate() {
                let at = self.words * i..self.words * (i + 1);
                out.push((name.as_str(), (i, *ray), &mine[at.clone()], &theirs[at]));
            }
        }
        out
    }
}

/// The DXIL of a fixture: the committed `tests/fixtures/<name>.dxil`, or for `ray_query_object_fields` the committed
/// `ray_query_object.hlsl` with the candidate procedural primitive's opacity left out (`CandidateProceduralPrimitiveNonOpaque`
/// is the one query field iridia-naga does not lower), compiled by `dxc` here so that no second fixture can go stale.
fn fixture_dxil(name: &str, dxc: Option<&inputs::Program>, scratch: &Path) -> std::path::PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    if name != "ray_query_object_fields" {
        return dir.join(format!("{name}.dxil"));
    }
    let text = std::fs::read_to_string(dir.join("ray_query_object.hlsl")).unwrap();
    let from = "results[slot] = p.CandidateProceduralPrimitiveNonOpaque() ? 2 : 1;";
    assert!(
        text.contains(from),
        "ray_query_object.hlsl has changed under this test"
    );
    let hlsl = scratch.join(format!("{name}.hlsl"));
    let dxil = scratch.join(format!("{name}.dxil"));
    std::fs::write(&hlsl, text.replace(from, "results[slot] = 1;")).unwrap();
    let out = dxc
        .unwrap()
        .command()
        .args(["-T", "cs_6_5", "-E", "main", "-Fo"])
        .path_arg(&dxil)
        .path_arg(&hlsl)
        .output_within(inputs::COMPILE_LIMIT)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        out.status.success(),
        "dxc on {}: {}",
        hlsl.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    dxil
}

/// What a case file has WARP run.
#[derive(Clone, Copy)]
enum Cs {
    /// The fixture's own DXIL.
    Fixture,
    /// The program's naga module as iridia-naga lowers it, written as HLSL by stock naga (the writer wgpu runs on
    /// Direct3D 12) and compiled by `tools.dxc`.
    NagaHlsl,
}

impl Cs {
    fn label(self) -> &'static str {
        match self {
            Cs::Fixture => "fixture",
            Cs::NagaHlsl => "naga-hlsl",
        }
    }
}

/// A validated naga module written as Shader Model 6.5 HLSL, each resource in the register of the program it came from,
/// and compiled to DXIL by `dxc`; the path of the DXIL. The binding plan says which group and binding of the module a
/// resource has.
fn hlsl_dxil(
    module: &naga_hlsl::Module,
    info: &naga_hlsl::valid::ModuleInfo,
    interface: &iridia_ir::interface::Interface,
    plan: &iridia_naga::BindingPlan,
    dxc: &inputs::Program,
    scratch: &Path,
    label: &str,
) -> std::path::PathBuf {
    use naga_hlsl::back::hlsl;
    let mut options = hlsl::Options {
        shader_model: hlsl::ShaderModel::V6_5,
        ..hlsl::Options::default()
    };
    for b in &plan.bindings {
        let d3d = interface
            .resources
            .iter()
            .find(|r| r.id == b.resource)
            .and_then(|r| r.provenance.d3d)
            .unwrap();
        options.binding_map.insert(
            naga_hlsl::ResourceBinding {
                group: b.group,
                binding: b.binding,
            },
            hlsl::BindTarget {
                space: u8::try_from(d3d.space).unwrap(),
                register: d3d.register,
                binding_array_size: None,
                dynamic_storage_buffer_offsets_index: None,
                restrict_indexing: false,
            },
        );
    }
    let pipeline = hlsl::PipelineOptions::default();
    let mut text = String::new();
    hlsl::Writer::new(&mut text, &options, &pipeline)
        .write(module, info, None)
        .unwrap();
    let hlsl_file = scratch.join(format!("{label}.hlsl"));
    let dxil = scratch.join(format!("{label}.dxil"));
    std::fs::write(&hlsl_file, text).unwrap();
    // `-HV 2018`: naga's HLSL uses `&&` on vectors, which HLSL 2021 refuses
    let out = dxc
        .command()
        .args(["-T", "cs_6_5", "-E", "main", "-HV", "2018", "-Fo"])
        .path_arg(&dxil)
        .path_arg(&hlsl_file)
        .output_within(inputs::COMPILE_LIMIT)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        out.status.success(),
        "dxc on {}: {}",
        hlsl_file.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    dxil
}

/// The fixture lifted onto the IR and prepared as the executor takes it.
fn on_ir(bytes: &[u8]) -> iridia_ir::Program {
    let mut program = iridia_dxil::ir::lift(bytes).unwrap();
    iridia_passes::prepare(&mut program).unwrap();
    program
}

/// Twelve scenes of 64 rays each through `fixture_name` on the executor and on WARP, or `None` without WARP.
fn gate(fixture_name: &str, procedural: bool, words: usize, cs: Cs) -> Option<Run> {
    let warp = program(Tool::Warp)?;
    let derived = fixture_name == "ray_query_object_fields";
    let dxc = match cs {
        Cs::NagaHlsl => Some(program(Tool::Dxc)?),
        Cs::Fixture if derived => Some(program(Tool::Dxc)?),
        Cs::Fixture => None,
    };
    let scratch = own_scratch("warp");
    let fixture = fixture_dxil(fixture_name, dxc.as_ref(), &scratch);
    let bytes = std::fs::read(&fixture).unwrap();
    let program = on_ir(&bytes);
    let label = format!("{fixture_name}.{}", cs.label());
    // what WARP runs: the fixture itself, or the program as the other route gives it
    let shader = match (cs, &dxc) {
        (Cs::NagaHlsl, Some(dxc)) => {
            let profile = iridia_naga::Profile::NAGA;
            let plan = iridia_naga::BindingPlan::new(&program.interface, profile.bindings).unwrap();
            let module = iridia_naga::module(&program, &plan).unwrap();
            let info = iridia_naga::validate(&module, &profile).unwrap();
            hlsl_dxil(&module, &info, &program.interface, &plan, dxc, &scratch, &label)
        }
        _ => fixture.clone(),
    };
    let exe = Executable::from_ir(&program).unwrap();
    let find = |class: Option<D3dClass>| {
        program
            .interface
            .resources
            .iter()
            .find(|r| match class {
                Some(c) => r.provenance.d3d.is_some_and(|d| d.class == c && d.register == 0),
                None => matches!(r.kind, iridia::ResourceKind::AccelerationStructure),
            })
            .unwrap()
            .id
    };
    let (structure, results, constants) = (
        find(None),
        find(Some(D3dClass::UnorderedAccess)),
        find(Some(D3dClass::ConstantBuffer)),
    );

    let mut text = String::new();
    let mut ours = BTreeMap::new();
    let mut cases = Vec::new();
    for (seed, instance_flags) in (1u64..=12).map(|seed| (seed, seed > 3)) {
        let name = format!("{fixture_name}{seed}");
        let scene = scene(seed, instance_flags, procedural);
        let rays = rays(seed * 7919, procedural);
        let mut rows: Vec<[u32; 4]> = Vec::new();
        rows.extend(
            rays.iter()
                .map(|t| [t.0[0], t.0[1], t.0[2], 0.0].map(f32::to_bits)),
        );
        rows.extend(
            rays.iter()
                .map(|t| [t.1[0], t.1[1], t.1[2], 0.0].map(f32::to_bits)),
        );
        rows.extend(rays.iter().map(|t| [t.2, t.3, 0, 0]));
        if procedural {
            rows.extend(spheres().iter().map(|s| s.map(f32::to_bits)));
        }

        text.push_str(&format!(
            "case {name}\napi d3d12rt\ncs {}\nresults {}\ngroups 1 1 1\ncb 0 {}\n",
            shader.display(),
            64 * words,
            rows.len()
        ));
        for row in &rows {
            text.push_str(&format!(
                "row {:08x} {:08x} {:08x} {:08x}\n",
                row[0], row[1], row[2], row[3]
            ));
        }
        for (b, geometries) in scene.bottoms.iter().enumerate() {
            for g in geometries {
                if !g.boxes.is_empty() {
                    text.push_str(&format!(
                        "procedural {b} {} {}
",
                        u8::from(g.opaque),
                        g.boxes.len()
                    ));
                    for corners in &g.boxes {
                        text.push_str(&format!(
                            "box {}
",
                            corners
                                .iter()
                                .flatten()
                                .map(|x| bits(*x))
                                .collect::<Vec<_>>()
                                .join(" ")
                        ));
                    }
                    continue;
                }
                text.push_str(&format!(
                    "geometry {b} {} {}\n",
                    u8::from(g.opaque),
                    g.positions.len() / 3
                ));
                for tri in g.positions.chunks(3) {
                    text.push_str(&format!(
                        "tri {}\n",
                        tri.iter()
                            .flatten()
                            .map(|x| bits(*x))
                            .collect::<Vec<_>>()
                            .join(" ")
                    ));
                }
            }
        }
        for i in &scene.instances {
            let flags = u32::from(i.flags.cull_disable)
                | u32::from(i.flags.front_counter_clockwise) << 1
                | u32::from(i.flags.force_opaque) << 2
                | u32::from(i.flags.force_non_opaque) << 3;
            text.push_str(&format!(
                "instance {} {} {} {} {flags} {}\n",
                i.bottom,
                i.id,
                i.mask,
                i.contribution,
                i.transform
                    .iter()
                    .flatten()
                    .map(|x| bits(*x))
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
        }
        text.push_str("end\n");

        let mut resources = Resources::default();
        resources.constants.insert(constants, rows);
        resources.scenes.insert(structure, scene);
        let bound = exe.bind_resources(&resources).unwrap();
        let mut buffers = BTreeMap::from([(results, vec![0u32; 64 * words])]);
        exe.dispatch(&bound, &mut NoBackend, &mut buffers, [1, 1, 1], 32)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        ours.insert(name.clone(), buffers.remove(&results).unwrap());
        cases.push((name, rays));
    }
    let cases_file = scratch.join(format!("{label}.cases.txt"));
    std::fs::write(&cases_file, text).unwrap();
    let _oracle = inputs::warp_exclusive();
    let run_warp = |attempt| {
        let out_file = scratch.join(format!("{label}.warp-{attempt}.jsonl"));
        let run = warp
            .command()
            .path_arg(&cases_file)
            .path_arg(&out_file)
            .output_within(inputs::WARP_LIMIT)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(
            run.status.success(),
            "warp run {attempt}: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let mut answers = BTreeMap::new();
        for line in std::fs::read_to_string(&out_file).unwrap().lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            let name = v["case"].as_str().unwrap().to_owned();
            assert!(v["error"].is_null(), "{name}: WARP error {}", v["error"]);
            let mut answer: Vec<u32> = v["words"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| u32::from_str_radix(w.as_str().unwrap(), 16).unwrap())
                .collect();
            assert_eq!(answer.len(), 64 * words, "{name}: WARP word count");
            for hit in answer.chunks_exact_mut(words) {
                if words == 10 && hit[0] == 0 {
                    hit[1..].fill(0);
                }
            }
            assert!(answers.insert(name.clone(), answer).is_none(), "duplicate {name}");
        }
        assert!(answers.keys().eq(ours.keys()), "WARP case names differ");
        answers
    };
    let mut theirs = run_warp(1);
    let second = run_warp(2);
    let third = (theirs != second).then(|| run_warp(3));
    let mut unstable = 0;
    for (name, answer) in &mut theirs {
        for (ray, hit) in answer.chunks_exact_mut(words).enumerate() {
            let range = ray * words..(ray + 1) * words;
            let repeat = &second[name][range.clone()];
            if hit != repeat {
                unstable += 1;
                let last = &third.as_ref().unwrap()[name][range];
                println!("WARP SELF-DISAGREEMENT {name} ray {ray}: {hit:08x?}, {repeat:08x?}, {last:08x?}");
                assert!(
                    last == hit || last == repeat,
                    "{name} ray {ray}: no two WARP runs agree"
                );
                hit.copy_from_slice(last);
            }
        }
    }
    println!("{label}: {unstable} WARP self-disagreements");
    Some(Run {
        words,
        ours,
        theirs,
        cases,
    })
}
