#![allow(clippy::unwrap_used, clippy::indexing_slicing)]
//! Programs that take their descriptors from the heaps (Shader Model 6.6), lifted onto the IR ([`iridia_dxil::ir`]) and run
//! on the executor against what Direct3D 12 WARP gave the same program, recorded beside each fixture (`<name>.warp.json`,
//! written by `<name>.warp.cpp`): every word must agree bit for bit.
//!
//! - `heap.dxil` (default and -Od builds), a compute program that reads a texture, a sampler, a source buffer and a target
//!   buffer from the heaps by indices of a constant buffer, run as it is and with its sampler heap made one sampler
//!   ([`iridia_passes::flatten_sampler_heaps`]), the oracle's two samplers alike;
//! - `sampler_heap.dxil`, a pixel program that samples a bound texture with a sampler from the sampler heap, at the first,
//!   a middle and the last of the oracle's 48 samplers, as it is and flattened.
use iridia_exec::{Backend, Element, Executable, ImageSample, Resources, Runner};
use iridia_ir::{
    Program,
    interface::{BufferKind, ResourceId, ResourceKind},
};
use std::collections::BTreeMap;
use std::path::Path;

fn fixtures() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn lifted(name: &str, flatten: bool) -> Program {
    let bytes = std::fs::read(fixtures().join(name)).unwrap();
    let mut program = iridia_dxil::ir::lift(&bytes).unwrap();
    iridia_passes::prepare(&mut program).unwrap();
    if flatten {
        assert_eq!(iridia_passes::flatten_sampler_heaps(&mut program), 1, "{name}");
    }
    program
}

/// The resource of the program that `pick` says is the one.
fn resource(program: &Program, what: &str, pick: impl Fn(&ResourceKind) -> bool) -> ResourceId {
    let mut found = program.interface.resources.iter().filter(|r| pick(&r.kind));
    let id = found.next().unwrap_or_else(|| panic!("no {what}")).id;
    assert!(found.next().is_none(), "two of {what}");
    id
}

fn constant_buffer(program: &Program) -> ResourceId {
    resource(program, "constant buffer", |k| {
        matches!(k, ResourceKind::ConstantBuffer { .. })
    })
}

fn words(json: &serde_json::Value) -> Vec<u32> {
    json.as_array()
        .unwrap()
        .iter()
        .map(|w| u32::try_from(w.as_u64().unwrap()).unwrap())
        .collect()
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// `heap.warp.cpp`'s texture, texel (x, y) = (x + 1, 2y + 1, 0.5, x - y) of 8 by 8, sampled at a point with clamp.
struct Texels {
    /// The element of the texture array the constant buffer names.
    texture: u32,
    /// The element of the sampler array that a sample names: the index the constant buffer gives, or 0 where the heap is
    /// one sampler.
    sampler: u32,
}
impl Backend for Texels {
    fn sample(&mut self, r: ImageSample<Element>) -> Result<[f32; 4], String> {
        assert_eq!(r.texture.index, self.texture);
        assert_eq!(r.sampler.index, self.sampler);
        let texel = |c: f32| (c * 8.0).floor().clamp(0.0, 7.0);
        let (x, y) = (texel(r.coord[0]), texel(r.coord[1]));
        Ok([x + 1.0, 2.0 * y + 1.0, 0.5, x - y])
    }
}

#[test]
fn a_compute_program_reading_the_heaps_agrees_with_warp() {
    let warp: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("heap.warp.json")).unwrap()).unwrap();
    let expected = words(&warp["target"]);
    assert_eq!(expected.len(), 256);
    for name in ["heap.dxil", "heap.od.dxil"] {
        for flatten in [false, true] {
            let program = lifted(name, flatten);
            let source = resource(&program, "source buffer", |k| {
                matches!(k, ResourceKind::Buffer(BufferKind::Structured { .. }))
            });
            let target = resource(&program, "target buffer", |k| {
                matches!(k, ResourceKind::Storage { .. })
            });
            // Indices { texture_index, sampler_index, source_index, target_index }
            let mut resources = Resources::default();
            resources
                .constants
                .insert(constant_buffer(&program), vec![[3, 1, 5, 6]]);
            resources
                .elements
                .insert((source, 5), (0..8).map(|k| f(1.5 + k as f32)).collect());
            let exe = Executable::from_ir(&program).unwrap();
            let bound = exe.bind_resources(&resources).unwrap();
            let mut elements = BTreeMap::from([((target, 6), vec![0u32; 256])]);
            let mut backend = Texels {
                texture: 3,
                sampler: if flatten { 0 } else { 1 },
            };
            exe.dispatch_elements(
                &bound,
                &mut backend,
                &mut BTreeMap::new(),
                &mut elements,
                [1, 1, 1],
                32,
            )
            .unwrap();
            let ours = &elements[&(target, 6)];
            let differ: Vec<String> = ours
                .iter()
                .zip(&expected)
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(k, (a, b))| format!("word {k}: {a:08x}, WARP {b:08x}"))
                .collect();
            assert!(
                differ.is_empty(),
                "{name} flattened {flatten}: {} words differ:\n{}",
                differ.len(),
                differ.join("\n")
            );
        }
    }
}

/// `sampler_heap.warp.cpp`'s texture: every texel is (0.25, -2, 8, 0.5).
struct One;
impl Backend for One {
    fn sample(&mut self, _: ImageSample<Element>) -> Result<[f32; 4], String> {
        Ok([0.25, -2.0, 8.0, 0.5])
    }
}

#[test]
fn a_pixel_program_sampling_with_the_sampler_heap_agrees_with_warp() {
    let recorded = std::fs::read_to_string(fixtures().join("sampler_heap.warp.json")).unwrap();
    // case name -> (sampler index, TEXCOORD0)
    let cases: BTreeMap<&str, (u32, [f32; 2])> = BTreeMap::from([
        ("heap_first", (0, [0.5, 0.5])),
        ("heap_middle", (17, [0.1, 0.9])),
        ("heap_last", (47, [0.75, 0.25])),
    ]);
    let mut compared = 0;
    for flatten in [false, true] {
        let program = lifted("sampler_heap.dxil", flatten);
        let exe = Executable::from_ir(&program).unwrap();
        for line in recorded.lines() {
            let warp: serde_json::Value = serde_json::from_str(line).unwrap();
            let name = warp["case"].as_str().unwrap();
            let (index, uv) = cases[name];
            let mut resources = Resources::default();
            resources
                .constants
                .insert(constant_buffer(&program), vec![[index, 0, 0, 0]]);
            let bound = exe.bind_resources(&resources).unwrap();
            let mut backend = One;
            let mut runner = Runner::<1>::new(&exe, &bound, &mut backend, None);
            // main(float4 position : SV_Position, float2 uv : TEXCOORD0)
            let [out] = runner
                .run([&[[f(0.5), f(0.5), f(0.5), f(1.0)], [f(uv[0]), f(uv[1]), 0, 0]][..]])
                .unwrap();
            let theirs = words(&warp["target"]);
            assert_eq!(out.values[0].to_vec(), theirs, "{name}, flattened {flatten}");
            compared += 1;
        }
    }
    assert_eq!(compared, 6);
}
