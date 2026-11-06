//! The committed programs the tests lift: the assembled one-feature fixtures (`tests/assembled`, each with the lift
//! options its comments state) and the programs of vkd3d's shader tests (`tests/vkd3d/cases.json`). Shared by their
//! tests and the version gate, which hashes what the lift makes of every one.
#![allow(dead_code, reason = "each test that includes the module uses part of it")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use iridia_ir::interface::TextureDim;
use serde_json::Value;

/// The folder of the assembled fixtures.
pub fn assembled_folder() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/assembled"))
}

/// The assembled fixtures' names, sorted.
pub fn assembled() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(assembled_folder())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "asm"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// An assembled fixture's source and tokens (`None` where it is not assembled).
pub fn assembled_fixture(name: &str) -> (String, Option<Vec<u8>>) {
    let folder = assembled_folder();
    let source = std::fs::read_to_string(folder.join(format!("{name}.asm"))).unwrap();
    (source, std::fs::read(folder.join(format!("{name}.bin"))).ok())
}

/// A texture dimension as a fixture names it.
pub fn dim(text: &str) -> TextureDim {
    match text {
        "1d" => TextureDim::D1,
        "2d" => TextureDim::D2,
        "3d" => TextureDim::D3,
        "cube" => TextureDim::Cube,
        other => panic!("not a texture dimension: {other}"),
    }
}

/// The lift options of an assembled fixture's `; lift:` comments.
pub fn lift_options(source: &str) -> iridia_d3d9_ir::Options {
    let mut options = iridia_d3d9_ir::Options::default();
    for line in source.lines().map(str::trim) {
        let Some(rest) = line.strip_prefix("; lift:") else {
            continue;
        };
        for option in rest.split_whitespace() {
            let (key, value) = option.split_once('=').unwrap();
            match key {
                "mul" if value == "ieee" => options.multiply = iridia_ir::MulZero::Ieee,
                "mul" if value == "zero_wins" => options.multiply = iridia_ir::MulZero::ZeroWins,
                "register_max" => options.ps1x.register_max = Some(value.parse().unwrap()),
                "texture_max" => options.ps1x.texture_max = Some(value.parse().unwrap()),
                "quantize" => {
                    let (bits, rounding) = value.split_once(',').unwrap();
                    let rounding = match rounding {
                        "floor" => iridia_ir::Rounding::Floor,
                        "ceil" => iridia_ir::Rounding::Ceil,
                        "trunc" => iridia_ir::Rounding::Trunc,
                        "nearest_even" => iridia_ir::Rounding::NearestEven,
                        "nearest_away" => iridia_ir::Rounding::NearestAway,
                        other => panic!("not a rounding: {other}"),
                    };
                    options.ps1x.quantize = Some((bits.parse().unwrap(), rounding));
                }
                k if k.starts_with("texdim") => {
                    options
                        .texture_dims
                        .insert(k["texdim".len()..].parse().unwrap(), dim(value));
                }
                other => panic!("not a lift option: {other}"),
            }
        }
    }
    options
}

/// vkd3d's cases file.
pub fn vkd3d() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vkd3d/cases.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The programs of vkd3d's cases file, by name.
pub fn vkd3d_programs(file: &Value) -> BTreeMap<String, Vec<u8>> {
    file["programs"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, hex)| {
            let hex = hex.as_str().unwrap();
            let bytes = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            (name.clone(), bytes)
        })
        .collect()
}
