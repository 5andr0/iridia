//! The same HLSL compiled to SM5.1, SM6.0, and SM5.0 where expressible, evaluated on identical CPU inputs: each build
//! runs as lifted onto the IR, prepared, and lowered to WGSL (`iridia_naga`).
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]
use iridia::{Interface, ResourceId, ResourceKind, Stage, StorageTarget};
use iridia_exec::{
    Backend, Element, Executable, ImageLoad, ImageQuery, ImageSample, ImageStore, QueryKind, Resources,
};
use iridia_ir::Program;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

mod lowered;
use lowered::{Unlowered, lower_c};

const NAMES: &[&str] = &[
    "spaces",
    "textures",
    "constants",
    "images",
    "structured",
    "unbounded",
    "nonuniform",
    "unbounded_buffers",
    "unbounded_constants",
    "samplers",
];
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sm51")
}
fn words(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}
fn rows(v: &Value) -> Vec<[u32; 4]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| words(v).try_into().unwrap())
        .collect()
}

struct Host<'a> {
    names: BTreeMap<ResourceId, String>,
    data: &'a Value,
    stores: BTreeMap<String, Vec<u32>>,
}
impl Host<'_> {
    fn image(&self, element: Element, x: i32, y: i32) -> Result<[u32; 4], String> {
        let name = &self.names[&element.id];
        let images = &self.data["images"][name];
        let image = images
            .get(element.index as usize)
            .ok_or_else(|| format!("{name}: missing image {}", element.index))?;
        if !(0..2).contains(&x) || !(0..2).contains(&y) {
            return Ok([0; 4]);
        }
        Ok(words(&image[(y * 2 + x) as usize]).try_into().unwrap())
    }
}
impl Backend for Host<'_> {
    fn load(&mut self, r: ImageLoad<Element>) -> Result<[u32; 4], String> {
        self.image(r.texture, r.texel[0], r.texel[1])
    }
    fn sample(&mut self, r: ImageSample<Element>) -> Result<[f32; 4], String> {
        let name = &self.names[&r.sampler.id];
        let address = self.data["samplers"][name][r.sampler.index as usize]
            .as_str()
            .ok_or("missing sampler")?;
        let mut x = (r.coord[0] * 2.0).floor() as i32;
        let mut y = (r.coord[1] * 2.0).floor() as i32;
        match address {
            "wrap" => {
                x = x.rem_euclid(2);
                y = y.rem_euclid(2);
            }
            "clamp" => {
                x = x.clamp(0, 1);
                y = y.clamp(0, 1);
            }
            "border" => {}
            _ => return Err("unknown sampler".into()),
        }
        let value = self.image(r.texture, x, y)?.map(f32::from_bits);
        Ok(if let Some(reference) = r.depth_ref {
            [if reference <= value[0] { 1.0 } else { 0.0 }; 4]
        } else {
            value
        })
    }
    /// Every image is 2x2, of one level, layer and sample (as the inputs describe them): what the lowered program's
    /// bounds checks ask.
    fn query(&mut self, r: ImageQuery<Element>) -> Result<[u32; 4], String> {
        Ok(match r.kind {
            QueryKind::Size => [2, 2, 1, 0],
            QueryKind::Levels | QueryKind::Layers | QueryKind::Samples => [1, 0, 0, 0],
        })
    }
    fn store_image(&mut self, r: ImageStore<ResourceId>) -> Result<(), String> {
        let key = format!("{}:{}:{:?}:{:?}", self.names[&r.view], r.index, r.texel, r.layer);
        self.stores.insert(key, r.value[..usize::from(r.lanes)].to_vec());
        Ok(())
    }
}

fn execute(program: &Program, input: &Value) -> Value {
    execute_with(&Executable::from_ir(program).unwrap(), &program.interface, input)
}

/// The IR lift of a container of Shader Model `model` (6.0 is DXIL, the others DXBC): as lifted (a graph structured as
/// the executor takes it), or after `iridia_passes::prepare`.
fn lifted(bytes: &[u8], model: u32, prepared: bool) -> Result<Program, String> {
    let mut p = if model == 60 {
        iridia_dxil::ir::lift(bytes)
    } else {
        iridia_dxbc::ir::lift(bytes)
    }
    .map_err(|e| e.to_string())?;
    if prepared {
        iridia_passes::prepare(&mut p)?;
    } else if matches!(p.entry.body, iridia_ir::Body::Cfg(_)) {
        iridia_passes::structurize(&mut p.entry)?;
        iridia_passes::break_flags(&mut p.entry)?;
    }
    Ok(p)
}

/// The runs of `input` on `exe`, bound using its program's interface.
fn execute_with(exe: &Executable, interface: &Interface, input: &Value) -> Value {
    let mut all = Vec::new();
    for invocation in input["runs"].as_array().unwrap() {
        let mut bound_data = Resources::default();
        let mut names = BTreeMap::new();
        let mut writes = BTreeMap::new();
        let mut elements = BTreeMap::new();
        for resource in &interface.resources {
            let name = &resource.provenance.name;
            names.insert(resource.id, name.clone());
            match &resource.kind {
                ResourceKind::ConstantBuffer { .. } => {
                    let data = &invocation["constants"][name];
                    if resource.is_array() {
                        for (k, value) in data.as_array().unwrap().iter().enumerate() {
                            bound_data
                                .constant_elements
                                .insert((resource.id, k as u32), rows(value));
                        }
                    } else {
                        bound_data.constants.insert(resource.id, rows(data));
                    }
                }
                ResourceKind::Buffer(_)
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(_),
                    ..
                } => {
                    let data = &input["buffers"][name];
                    if resource.is_array() {
                        for (k, value) in data.as_array().unwrap().iter().enumerate() {
                            let key = (resource.id, k as u32);
                            let data = words(value);
                            bound_data.elements.insert(key, data.clone());
                            elements.insert(key, data);
                        }
                    } else {
                        let data = words(data);
                        bound_data.buffers.insert(resource.id, data.clone());
                        writes.insert(resource.id, data);
                    }
                }
                _ => {}
            }
        }
        let bound = exe.bind_resources(&bound_data).unwrap();
        let mut host = Host {
            names,
            data: input,
            stores: BTreeMap::new(),
        };
        let output = if interface.stage == Stage::Compute {
            let groups: [u32; 3] = words(&invocation["groups"]).try_into().unwrap();
            exe.dispatch_elements(&bound, &mut host, &mut writes, &mut elements, groups, 4)
                .unwrap();
            let buffers: BTreeMap<_, _> = writes
                .iter()
                .filter(|(id, _)| {
                    interface
                        .resources
                        .iter()
                        .any(|r| r.id == **id && matches!(r.kind, ResourceKind::Storage { .. }))
                })
                .map(|(id, v)| (format!("{}:0", host.names[id]), v))
                .chain(
                    elements
                        .iter()
                        .filter(|((id, _), _)| {
                            interface
                                .resources
                                .iter()
                                .any(|r| r.id == *id && matches!(r.kind, ResourceKind::Storage { .. }))
                        })
                        .map(|((id, k), v)| (format!("{}:{k}", host.names[id]), v)),
                )
                .collect();
            json!({"buffers":buffers,"images":host.stores})
        } else {
            let args = rows(&invocation["arguments"]);
            let mut runner = iridia_exec::Runner::<1>::new(exe, &bound, &mut host, None);
            let [out] = runner.run([&args]).unwrap();
            json!({"values":out.values})
        };
        all.push(output);
    }
    json!(all)
}
fn lanes(value: &Value) -> usize {
    match value {
        Value::Number(_) => 1,
        Value::Array(v) => v.iter().map(lanes).sum(),
        Value::Object(v) => v.values().map(lanes).sum(),
        _ => 0,
    }
}

#[test]
fn sm51_matches_dxil_and_sm50_bit_for_bit() {
    for name in NAMES {
        let input: Value =
            serde_json::from_slice(&std::fs::read(root().join(format!("{name}.inputs.json"))).unwrap())
                .unwrap();
        let mut baseline: Option<Value> = None;
        for (model, ext) in [(51, "dxbc"), (60, "dxil"), (50, "dxbc")] {
            let path = root().join(format!("{name}.{model}.{ext}"));
            if !path.exists() {
                assert_eq!(model, 50);
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            let p = lifted(&bytes, model, true).unwrap_or_else(|e| panic!("{name}.{model}: {e}"));
            iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{name}.{model}: {e:?}"));
            println!("running {name}.{model}");
            let output = execute(&p, &input);
            // the program as lifted, before the passes, gives the same words
            let ir = lifted(&bytes, model, false).unwrap_or_else(|e| panic!("{name}.{model} as lifted: {e}"));
            let exe = Executable::from_ir(&ir).unwrap_or_else(|e| panic!("{name}.{model} as lifted: {e}"));
            assert_eq!(
                execute_with(&exe, &ir.interface, &input),
                output,
                "{name}.{model}: as lifted"
            );
            println!("{name}.{model}: as lifted, {} lanes identical", lanes(&output));
            // C still uses the older module only to restore names and entry bindings after WGSL is read back.
            // Its executable is entirely the prepared IR's lowering, never the older lift's code.
            let old = if model == 60 {
                iridia_dxil::lift(&bytes)
            } else {
                iridia_dxbc::lift(&bytes)
            }
            .unwrap_or_else(|e| panic!("{name}.{model} entry metadata: {e}"));
            let c = lower_c(&old, &p).unwrap_or_else(|e| match e {
                Unlowered::Refused(e) | Unlowered::Unbuilt(e) => panic!("{name}.{model} lowered: {e}"),
            });
            let exe = Executable::new(&c).unwrap_or_else(|e| panic!("{name}.{model} lowered: {e}"));
            assert_eq!(
                execute_with(&exe, &c.interface, &input),
                output,
                "{name}.{model}: the IR lift lowered to WGSL"
            );
            println!(
                "{name}.{model}: the IR lift lowered to WGSL, {} lanes identical",
                lanes(&output)
            );
            if let Some(expected) = &baseline {
                assert_eq!(&output, expected, "{name}.{model}");
                println!("{name}: 5.1 vs {model}, {} lanes identical", lanes(&output));
            } else {
                baseline = Some(output);
            }
        }
        let expected = root().join(format!("{name}.expected.json"));
        let output = baseline.unwrap();
        if std::env::var_os("SM51_WRITE_EXPECTED").is_some() {
            std::fs::write(&expected, serde_json::to_string_pretty(&output).unwrap() + "\n").unwrap();
        }
        let saved: Value = serde_json::from_slice(&std::fs::read(expected).unwrap()).unwrap();
        assert_eq!(saved, output, "{name}: saved expected outputs");
    }
}

#[test]
fn unbounded_constant_bindings_are_sparse_and_missing_descriptors_error() {
    for (model, ext) in [(51, "dxbc"), (60, "dxil")] {
        let bytes = std::fs::read(root().join(format!("unbounded_constants.{model}.{ext}"))).unwrap();
        let p = lifted(&bytes, model, true).unwrap();
        let id = p.interface.resources[0].id;
        let exe = Executable::from_ir(&p).unwrap();
        let mut data = Resources::default();
        let index = 1_000_000_000;
        data.constant_elements.insert((id, index), vec![[17, 23, 31, 47]]);
        let bound = exe.bind_resources(&data).unwrap();
        let input = json!({});
        let mut host = Host {
            names: BTreeMap::new(),
            data: &input,
            stores: BTreeMap::new(),
        };
        let mut runner = iridia_exec::Runner::<1>::new(&exe, &bound, &mut host, None);
        let args = [[index, 0, 0, 0]];
        let [out] = runner.run([&args]).unwrap();
        assert_eq!(out.values, [[17, 23, 31, 47]]);
        let missing = [[index + 1, 0, 0, 0]];
        assert!(
            runner
                .run([&missing])
                .unwrap_err()
                .contains("no constant buffer given for descriptor")
        );
    }
}

#[test]
fn dxil_comparison_fixtures_match_the_lifter_version() {
    let mut outcomes = Vec::new();
    for name in NAMES {
        let file = format!("{name}.60.dxil");
        let bytes = std::fs::read(root().join(&file)).unwrap();
        let output = iridia_dxil::lift(&bytes)
            .map_err(|e| e.to_string())
            .and_then(|p| p.text().map_err(|e| e.to_string()));
        outcomes.push((file, output));
    }
    outcomes.sort_by(|a, b| a.0.cmp(&b.0));
    let current = iridia::gate::table(
        "iridia-dxil-sm51-comparison",
        iridia_dxil::LIFTER_VERSION,
        &outcomes,
    );
    let committed = std::fs::read_to_string(root().join("dxil.hashes")).unwrap_or_default();
    if let Err(why) = iridia::gate::check(&committed, &current) {
        let path = std::env::temp_dir().join("sm51-dxil.hashes");
        std::fs::write(&path, current).unwrap();
        panic!("{why}; current table: {}", path.display());
    }
}
