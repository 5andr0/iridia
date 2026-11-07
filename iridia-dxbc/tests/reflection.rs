#![allow(clippy::unwrap_used)]
mod inputs;

use iridia::{BufferKind, ConstantType, D3dClass, ResourceKind, ScalarType, Value};
mod common;
use std::collections::BTreeMap;

fn set(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
fn fixture(rdef: &[u8]) -> Vec<u8> {
    let signature = [0u32.to_le_bytes(), 8u32.to_le_bytes()].concat();
    let mut bytes = vec![0; 44];
    bytes[..4].copy_from_slice(b"DXBC");
    set(&mut bytes, 20, 1);
    set(&mut bytes, 28, 3);
    for (i, (kind, data)) in [
        (b"RDEF", rdef),
        (b"ISGN", signature.as_slice()),
        (b"OSGN", signature.as_slice()),
    ]
    .into_iter()
    .enumerate()
    {
        let at = bytes.len();
        set(&mut bytes, 32 + 4 * i, at as u32);
        bytes.extend(kind);
        bytes.extend((data.len() as u32).to_le_bytes());
        bytes.extend(data);
    }
    let len = bytes.len();
    set(&mut bytes, 24, len as u32);
    bytes
}
fn rdef() -> Vec<u8> {
    let mut b = vec![0; 208];
    for (at, v) in [
        (0, 1),
        (4, 60),
        // one resource record, at 176: the constant buffer B at b0
        (8, 1),
        (12, 176),
        (176, 172),
        (200, 1),
        (16, 0xffff0500),
        (28, u32::from_le_bytes(*b"RD11")),
        (32, 60),
        (36, 24),
        (40, 32),
        (44, 40),
        (48, 36),
        (52, 12),
        (60, 172),
        (64, 1),
        (68, 84),
        (72, 16),
        (84, 174),
        (92, 16),
        (100, 124),
        (124, 1 | 3 << 16),
        (128, 1 | 4 << 16),
    ] {
        set(&mut b, at, v);
    }
    b[172..176].copy_from_slice(b"B\0v\0");
    b
}
#[test]
fn checked_reflection_ranges_and_cycles() {
    let valid = rdef();
    let r = iridia_dxbc::reflection(&fixture(&valid)).unwrap();
    let r_of_valid = iridia_dxbc::reflection(&fixture(&valid)).unwrap();
    let ResourceKind::ConstantBuffer { layout } = &r.resources[0].kind else {
        panic!("a constant buffer")
    };
    assert_eq!(
        (
            r.resources[0].provenance.name.as_str(),
            r.resources[0].provenance.d3d.unwrap().register,
            layout.constants[0].ty.clone()
        ),
        (
            "B",
            0,
            ConstantType::Vector(Value {
                scalar: ScalarType::Float,
                width: 4
            })
        )
    );
    // what the mapping does not know is refused by name: a resource type with a counter, flags and counts
    for (at, value, named) in [
        (180, 9, "D3D_SIT_UAV_APPEND_STRUCTURED"),
        (180, 99, "an unknown resource type"),
        (204, 0x40, "flags 0x40"),
        (200, 0, "bind count 0"),
    ] {
        let mut bad = valid.clone();
        set(&mut bad, at, value);
        let message = iridia_dxbc::reflection(&fixture(&bad)).unwrap_err().to_string();
        assert!(message.contains(named), "{named}: {message}");
    }
    // a tbuffer (a layout of type 1 and its binding of type 1: constants read through a shader-resource register) is a
    // typed buffer of four-lane rows that keeps its layout, its members as the reflection states them
    let mut tbuffer = valid.clone();
    set(&mut tbuffer, 80, 1);
    set(&mut tbuffer, 180, 1);
    let r = iridia_dxbc::reflection(&fixture(&tbuffer)).unwrap();
    assert_eq!(
        r.resources[0].kind,
        ResourceKind::Buffer(BufferKind::Typed(Value {
            scalar: ScalarType::Float,
            width: 4
        }))
    );
    let binding = r.resources[0].provenance.d3d.unwrap();
    assert_eq!((binding.class, binding.register), (D3dClass::ShaderResource, 0));
    let kept = r.resources[0].provenance.layout.as_ref().unwrap();
    assert_eq!(
        (kept.size, kept.constants.len(), kept.constants[0].name.as_str()),
        (16, 1, "v")
    );
    // a constant buffer keeps none beside its kind
    assert!(r_of_valid.resources[0].provenance.layout.is_none());
    // a tbuffer binding whose layout is a constant buffer's has no layout of its own
    let mut bad = valid.clone();
    set(&mut bad, 180, 1);
    let message = iridia_dxbc::reflection(&fixture(&bad)).unwrap_err().to_string();
    assert!(message.contains("tbuffer without its layout"), "{message}");
    // a tbuffer layout whose binding is a constant buffer has no layout to bind
    let mut bad = valid.clone();
    set(&mut bad, 80, 1);
    let message = iridia_dxbc::reflection(&fixture(&bad)).unwrap_err().to_string();
    assert!(message.contains("without its layout"), "{message}");
    let mut bad = valid.clone();
    set(&mut bad, 92, 17);
    assert!(
        iridia_dxbc::reflection(&fixture(&bad))
            .unwrap_err()
            .to_string()
            .contains("lies outside its 16 byte constant buffer")
    );
    let mut bad = valid.clone();
    set(&mut bad, 124, 5);
    set(&mut bad, 132, 1 << 16);
    set(&mut bad, 136, 160);
    set(&mut bad, 160, 174);
    set(&mut bad, 164, 124);
    assert!(
        iridia_dxbc::reflection(&fixture(&bad))
            .unwrap_err()
            .to_string()
            .contains("part of a cycle")
    );
    let mut bad = valid.clone();
    set(&mut bad, 100, 10000);
    assert!(iridia_dxbc::reflection(&fixture(&bad)).is_err());
    let mut bad = valid.clone();
    set(&mut bad, 40, 31);
    assert!(iridia_dxbc::reflection(&fixture(&bad)).is_err());
    for n in 0..valid.len() {
        assert!(iridia_dxbc::reflection(&fixture(&valid[..n])).is_err());
    }
}

fn leaves<'a>(
    name: &str,
    offset: u32,
    ty: &'a ConstantType,
    out: &mut BTreeMap<String, (u32, &'a ConstantType)>,
) {
    if ty.members().is_empty() {
        assert!(out.insert(name.to_owned(), (offset, ty)).is_none());
    }
    for m in ty.members() {
        leaves(&m.name, offset + m.offset, &m.ty, out);
    }
}

#[test]
fn reflected_bindings_match_independent_disassembly() -> Result<(), Box<dyn std::error::Error>> {
    let Some(root) = inputs::oracle(inputs::OracleSet::Corpus, inputs::Game::General, "dxbc") else {
        return Ok(());
    };
    let (mut shaders, mut fields, mut bindings, mut inputs) = (0, 0, 0, 0);
    let mut refused: BTreeMap<String, usize> = BTreeMap::new();
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.extension().is_none_or(|x| x != "ppo") {
            continue;
        }
        let oracle = path.with_extension("ppo.reflection.json");
        if !oracle.exists() {
            continue;
        }
        let j: serde_json::Value = serde_json::from_slice(&std::fs::read(oracle)?)?;
        let r = match iridia_dxbc::reflection(&std::fs::read(&path)?) {
            Ok(r) => r,
            Err(e) => {
                *refused.entry(common::reason(&e.to_string())).or_default() += 1;
                continue;
            }
        };
        for (group, textures) in [("textures", true), ("buffers", false)] {
            for (name, value) in j[group].as_object().unwrap() {
                let resource = r
                    .resources
                    .iter()
                    .find(|b| {
                        matches!(b.kind, ResourceKind::Texture { .. }) == textures
                            && matches!(
                                b.kind,
                                ResourceKind::Texture { .. } | ResourceKind::ConstantBuffer { .. }
                            )
                            && b.provenance.name == *name
                    })
                    .unwrap();
                assert_eq!(
                    resource.provenance.d3d.unwrap().register as u64,
                    value.as_u64().unwrap()
                );
                bindings += 1;
            }
        }
        for (name, expected) in j["inputs"].as_object().unwrap() {
            assert!(
                r.inputs.iter().any(
                    |v| format!("{}{}", v.label.name.to_uppercase(), v.label.index) == *name
                        && v.provenance.d3d.unwrap().register as u64 == expected.as_u64().unwrap()
                ),
                "{name}"
            );
            inputs += 1;
        }
        for (buffer, expected) in j["layout"].as_object().unwrap() {
            let mut actual = BTreeMap::new();
            let resource = r.resources.iter().find(|b| b.provenance.name == *buffer).unwrap();
            let ResourceKind::ConstantBuffer { layout } = &resource.kind else {
                panic!("{buffer} is not a constant buffer")
            };
            for c in &layout.constants {
                leaves(&c.name, c.offset, &c.ty, &mut actual);
            }
            for (name, expected) in expected.as_object().unwrap() {
                let (offset, ty) = actual
                    .get(name)
                    .unwrap_or_else(|| panic!("{} {buffer}.{name}", path.display()));
                assert_eq!(
                    *offset as u64,
                    expected["offset"].as_u64().unwrap(),
                    "{buffer}.{name}"
                );
                assert_eq!(
                    match ty {
                        ConstantType::Array { count, .. } => *count as u64,
                        _ => 1,
                    },
                    expected["count"].as_u64().unwrap(),
                    "{buffer}.{name} count"
                );
                fields += 1;
            }
        }
        shaders += 1;
    }
    assert!(shaders > 0, "no reflection fixtures");
    for (why, n) in &refused {
        println!("  refused by the reader ({n}): {why}");
    }
    // measured on oracles.corpus.general/dxbc (Sekiro's 390 pixel programs): the reader reflects every one
    common::hold_to_pins(&refused, &[]);
    println!(
        "reflection exact: {shaders} shaders, {bindings} resource bindings, {inputs} input registers, {fields} member offsets/counts"
    );
    Ok(())
}

/// Reflection where the container has one this reader reads: none without an RDEF chunk or with a target it doesn't
/// read (another model's, an unknown stage's), the resources of a Shader Model 5 program of every stage, the full
/// reflection of a pixel, vertex or compute program only, and an error, never none, for one that doesn't parse.
#[test]
fn a_reflection_is_read_where_it_is_one_this_reader_reads() {
    let good = fixture(&rdef());
    assert!(iridia_dxbc::resource_reflection(&good).unwrap().is_some());
    let mut compute = rdef();
    set(&mut compute, 16, 0x4353_0500);
    let reflected = iridia_dxbc::reflection(&fixture(&compute)).unwrap();
    assert_eq!(reflected.stage, iridia::Stage::Compute);
    assert!(reflected.inputs.is_empty() && reflected.outputs.is_empty());
    for stage in [0xffff, 0xfffe, 0x4753, 0x4853, 0x4453, 0x4353] {
        let mut other = rdef();
        set(&mut other, 16, stage << 16 | 0x0500);
        let resources = iridia_dxbc::resource_reflection(&fixture(&other)).unwrap();
        assert_eq!(resources.map(|r| r.len()), Some(1), "{stage:#x}");
        let full = iridia_dxbc::reflection(&fixture(&other));
        assert_eq!(
            full.is_ok(),
            matches!(stage, 0xffff | 0xfffe | 0x4353),
            "{stage:#x}"
        );
    }
    for target in [0xffff_0300, 0xffff_0600, 0x1234_0500] {
        let mut other = rdef();
        set(&mut other, 16, target);
        assert!(
            iridia_dxbc::resource_reflection(&fixture(&other))
                .unwrap()
                .is_none(),
            "{target:#x}"
        );
    }
    // the resource records past the chunk's end
    let mut broken = rdef();
    set(&mut broken, 12, 4096);
    assert!(iridia_dxbc::resource_reflection(&fixture(&broken)).is_err());
    // the RDEF chunk under another tag: none
    let mut without = good.clone();
    let at = without.windows(4).position(|w| w == b"RDEF").unwrap();
    without[at..at + 4].copy_from_slice(b"STAT");
    assert!(iridia_dxbc::resource_reflection(&without).unwrap().is_none());
}

/// One row of the Resource Bindings table of Microsoft's listing: name, type, format, dimension, bind point, count.
struct Row {
    name: String,
    kind: String,
    format: String,
    dim: String,
    class: D3dClass,
    register: u32,
    count: u32,
}

/// The rows of a listing's Resource Bindings table (after its dashed rule, up to the blank comment line that ends it).
fn binding_rows(listing: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    let (mut inside, mut ruled) = (false, false);
    for line in listing.lines() {
        let text = line.trim().trim_start_matches("//").trim();
        if text == "Resource Bindings:" {
            inside = true;
            continue;
        }
        if !inside {
            continue;
        }
        if text.starts_with("---") {
            ruled = true;
            continue;
        }
        if !ruled {
            continue;
        }
        if text.is_empty() {
            break;
        }
        let f: Vec<&str> = text.split_whitespace().collect();
        let [name, kind, format, dim, bind, count] = f[..] else {
            panic!("a binding row {text:?} of other columns");
        };
        let (class, digits) = if let Some(n) = bind.strip_prefix("cb") {
            (D3dClass::ConstantBuffer, n)
        } else if let Some(n) = bind.strip_prefix('t') {
            (D3dClass::ShaderResource, n)
        } else if let Some(n) = bind.strip_prefix('s') {
            (D3dClass::Sampler, n)
        } else if let Some(n) = bind.strip_prefix('u') {
            (D3dClass::UnorderedAccess, n)
        } else {
            panic!("a bind point {bind:?}");
        };
        rows.push(Row {
            name: name.to_owned(),
            kind: kind.to_owned(),
            format: format.to_owned(),
            dim: dim.to_owned(),
            class,
            register: digits.parse().unwrap(),
            count: count.parse().unwrap(),
        });
    }
    rows
}

/// Buffer blocks by name: each leaf's byte offset and array count, by name.
type Layouts = BTreeMap<String, BTreeMap<String, (u32, u32)>>;

/// Every buffer block of a listing's Buffer Definitions (`cbuffer X` and `Resource bind info for X`): its leaves, by
/// name, with their byte offsets and array counts, and how many of its structs list their members outside them. The
/// listing states a struct member's offset within the buffer (the struct's start added to the record's), so a struct
/// only nests it and its own line is not a leaf; where a record's member offsets include the struct's start already
/// (the listing then puts them past the struct's end), that start is taken off again, as the reader takes it.
fn listed_layouts(listing: &str) -> (Layouts, usize) {
    let mut out = Layouts::new();
    let mut outside = 0;
    let mut buffer: Option<String> = None;
    // the members of each open struct: name, listed offset, count
    let mut stack: Vec<Vec<(String, u32, u32)>> = Vec::new();
    for line in listing.lines() {
        let text = line.trim().trim_start_matches("//").trim();
        if let Some(name) = text
            .strip_prefix("cbuffer ")
            .or_else(|| text.strip_prefix("Resource bind info for "))
        {
            buffer = Some(name.trim().to_owned());
            out.entry(name.trim().to_owned()).or_default();
            stack.clear();
            continue;
        }
        let Some(b) = &buffer else { continue };
        if text.starts_with("struct") {
            stack.push(Vec::new());
            continue;
        }
        if text == "}" && stack.is_empty() {
            buffer = None;
            continue;
        }
        let Some((declared, rest)) = text.split_once("// Offset:") else {
            continue;
        };
        let mut numbers = rest.split_whitespace().filter_map(|w| w.parse::<u32>().ok());
        let offset = numbers.next().unwrap();
        let declared = declared.trim().trim_end_matches(';');
        let leaves = if declared.starts_with('}') {
            let mut members = stack.pop().unwrap_or_default();
            let size = numbers.next().unwrap_or(0);
            if members
                .iter()
                .any(|m| m.1 < offset || m.1 >= offset.saturating_add(size))
            {
                outside += 1;
                for m in &mut members {
                    m.1 = m.1.checked_sub(offset).unwrap();
                }
            }
            members
        } else {
            let last = declared.split_whitespace().last().unwrap();
            let (name, count) = match last.split_once('[') {
                Some((n, c)) => (n, c.trim_end_matches(']').parse().unwrap()),
                None => (last, 1),
            };
            vec![(name.to_owned(), offset, count)]
        };
        match stack.last_mut() {
            Some(open) => open.extend(leaves),
            None => {
                for (name, offset, count) in leaves {
                    assert!(
                        out.get_mut(b)
                            .unwrap()
                            .insert(name.clone(), (offset, count))
                            .is_none(),
                        "{b}.{name} listed twice"
                    );
                }
            }
        }
    }
    (out, outside)
}

/// The texture shape a listing's dimension names: dimensionality, array, multisampled.
fn listed_shape(dim: &str) -> Option<(iridia::TextureDim, bool, bool)> {
    use iridia::TextureDim::{Cube, D1, D2, D3};
    Some(match dim {
        "1d" => (D1, false, false),
        "1darray" => (D1, true, false),
        "2d" => (D2, false, false),
        "2darray" => (D2, true, false),
        "2dMS" => (D2, false, true),
        "2dMSarray" => (D2, true, true),
        "3d" => (D3, false, false),
        "cube" => (Cube, false, false),
        "cubearray" => (Cube, true, false),
        _ => return None,
    })
}

/// The reflection of the programs whose RDEF this reader read only from Shader Model 5 pixel, vertex and compute
/// programs before: Sekiro's hull, domain and geometry programs (`corpus.sekiro/d3d11-shaders`) and the Shader Model 4
/// pixel programs of the Unity corpus (`corpus.sledding/shaders`). Each lifts onto the IR, and its interface's
/// resources are those of Microsoft's listing of it (`fxc /dumpbin`): each binding's name, class, register and count,
/// a texture's shape and value type, and every leaf of each constant buffer and structured element at its offset with
/// its count; and no resource the listing does not name.
#[test]
fn every_stage_and_model_reflects_as_fxc_lists_it() {
    let Some(fxc) = inputs::tool(inputs::Tool::Fxc) else {
        return;
    };
    let sekiro = inputs::oracle(inputs::OracleSet::Corpus, inputs::Game::Sekiro, "d3d11-shaders");
    let unity = inputs::oracle(inputs::OracleSet::Corpus, inputs::Game::Sledding, "shaders");
    let both = sekiro.is_some() && unity.is_some();
    let scratch = inputs::own_scratch("fxc-reflection");
    let mut by_target: BTreeMap<String, usize> = BTreeMap::new();
    let (mut bindings, mut leaves_checked, mut structs_rebased) = (0, 0, 0);
    for root in [sekiro, unity].into_iter().flatten() {
        let mut paths = Vec::new();
        common::containers(&root, &mut paths);
        for path in paths {
            let bytes = std::fs::read(&path).unwrap();
            let Some(target) = dxbc_reader::rdef_target(&bytes).unwrap() else {
                continue;
            };
            let other_stage = matches!(target >> 16, 0x4753 | 0x4853 | 0x4453);
            if !(other_stage || target & 0xffff < 0x0500) {
                continue;
            }
            let at = path.display();
            let program = iridia_dxbc::ir::lift(&bytes).unwrap_or_else(|e| panic!("{at}: {e}"));
            let listing_path = scratch.join("listing.asm");
            let out = std::process::Command::new(&fxc)
                .args(["/nologo", "/dumpbin", "/Fc"])
                .arg(&listing_path)
                .arg(&path)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "fxc {at}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let listing = std::fs::read_to_string(&listing_path).unwrap();
            let resources = &program.interface.resources;
            let rows = binding_rows(&listing);
            for row in &rows {
                let r = resources
                    .iter()
                    .find(|r| {
                        r.provenance.name == row.name
                            && r.provenance.d3d.is_some_and(|d| d.class == row.class)
                    })
                    .unwrap_or_else(|| panic!("{at}: {} {:?} is not in the interface", row.name, row.class));
                let d = r.provenance.d3d.unwrap();
                assert_eq!(
                    (d.register, r.count),
                    (row.register, row.count),
                    "{at}: {}",
                    row.name
                );
                if row.kind == "texture"
                    && let Some(shape) = listed_shape(&row.dim)
                {
                    let ResourceKind::Texture {
                        dim,
                        arrayed,
                        multisampled,
                        sample,
                    } = &r.kind
                    else {
                        panic!("{at}: {} is listed as a texture", row.name);
                    };
                    assert_eq!((*dim, *arrayed, *multisampled), shape, "{at}: {}", row.name);
                    let scalar = match sample {
                        iridia::SampleType::Sint => "int",
                        iridia::SampleType::Uint => "uint",
                        _ => "float",
                    };
                    assert_eq!(
                        row.format.trim_end_matches(char::is_numeric),
                        scalar,
                        "{at}: {}",
                        row.name
                    );
                }
                bindings += 1;
            }
            for r in resources {
                assert!(
                    rows.iter().any(|row| row.name == r.provenance.name),
                    "{at}: {} is not in the listing",
                    r.provenance.name
                );
            }
            let (layouts, outside) = listed_layouts(&listing);
            structs_rebased += outside;
            for (buffer, expected) in layouts {
                let r = resources
                    .iter()
                    .find(|r| r.provenance.name == buffer)
                    .unwrap_or_else(|| panic!("{at}: buffer {buffer} is not in the interface"));
                let layout = match (&r.kind, &r.provenance.layout) {
                    (ResourceKind::ConstantBuffer { layout }, _) | (_, Some(layout)) => layout,
                    _ => panic!("{at}: {buffer} has no layout"),
                };
                let mut actual = BTreeMap::new();
                for c in &layout.constants {
                    leaves(&c.name, c.offset, &c.ty, &mut actual);
                }
                for (name, (offset, count)) in expected {
                    let (o, ty) = actual
                        .get(name.as_str())
                        .unwrap_or_else(|| panic!("{at}: {buffer}.{name} is not in the layout"));
                    let n = match ty {
                        ConstantType::Array { count, .. } => *count,
                        _ => 1,
                    };
                    assert_eq!((*o, n), (offset, count), "{at}: {buffer}.{name}");
                    leaves_checked += 1;
                }
            }
            *by_target.entry(format!("{target:#010x}")).or_default() += 1;
        }
    }
    println!(
        "reflection as fxc lists it: {by_target:?}, {bindings} bindings, {leaves_checked} leaves, {structs_rebased} \
         structs whose record's member offsets include their start"
    );
    if both {
        // measured: Sekiro's 46 domain, 22 geometry and 19 hull programs, four of the hull programs with such a struct
        // (SAT_MtdParam); the Unity corpus's 16 ps_4_0 programs
        assert_eq!(
            structs_rebased, 4,
            "structs whose record's member offsets include their start"
        );
        let counts: Vec<(&str, usize)> = by_target.iter().map(|(t, &n)| (t.as_str(), n)).collect();
        assert_eq!(
            counts,
            [
                ("0x44530500", 46),
                ("0x47530500", 22),
                ("0x48530500", 19),
                ("0xffff0400", 16)
            ]
        );
    }
}
