//! The version gate (`iridia::gate`): what the lifter makes of its committed fixtures is hashed under
//! `LIFTER_VERSION` and `PROGRAM_VERSION` in `tests/lifted.hashes`. A change to the output without a bump of the version that
//! names it fails here, and so does a bump without the table of that version. The current table is written to the scratch root
//! (`dxil-lifted.hashes`, the path is in the failure) when they differ; it replaces the committed one in the commit that bumps the
//! version.
mod inputs;

use std::path::Path;

use iridia::gate;
use iridia_dxil::{LIFTER_VERSION, lift};

#[test]
fn what_the_lifter_makes_of_its_fixtures_is_the_versions() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut names: Vec<String> = std::fs::read_dir(dir.join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".dxil"))
        .collect();
    names.sort();
    let outcomes: Vec<_> = names
        .into_iter()
        .map(|name| {
            let bytes = std::fs::read(dir.join("fixtures").join(&name)).unwrap();
            let outcome = lift(&bytes)
                .map_err(|e| e.to_string())
                .and_then(|p| p.text().map_err(|e| format!("not written: {e}")));
            (name, outcome)
        })
        .collect();
    let current = gate::table("iridia-dxil", LIFTER_VERSION, &outcomes);
    let committed = std::fs::read_to_string(dir.join("lifted.hashes")).unwrap_or_default();
    if let Err(what) = gate::check(&committed, &current) {
        let new = inputs::scratch().join("dxil-lifted.hashes");
        std::fs::write(&new, &current).unwrap();
        panic!("{what}; the current table is {}", new.display());
    }
}

/// The IR lift's version gate (`iridia_ir::gate`): what `ir::lift` makes of the same fixtures is hashed under the IR
/// lifter's `LIFTER_VERSION` and `iridia_ir::VERSION` in `tests/ir-lifted.hashes`, so a change to what the IR lift makes
/// of a bytecode (a program it now lifts, one it now refuses, a different program) needs a bump that a cache keyed by
/// the bytecode sees. The current table is written to the scratch root (`dxil-ir-lifted.hashes`) when they differ.
#[test]
fn what_the_ir_lift_makes_of_its_fixtures_is_its_versions() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut names: Vec<String> = std::fs::read_dir(dir.join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".dxil"))
        .collect();
    names.sort();
    let outcomes: Vec<_> = names
        .into_iter()
        .map(|name| {
            let bytes = std::fs::read(dir.join("fixtures").join(&name)).unwrap();
            let outcome = iridia_dxil::ir::lift(&bytes)
                .map(|p| iridia_ir::gate::lifted(&p))
                .map_err(|e| e.to_string());
            (name, outcome)
        })
        .collect();
    let current = iridia_ir::gate::table("iridia-dxil-ir", iridia_dxil::ir::LIFTER_VERSION, &outcomes);
    let committed = std::fs::read_to_string(dir.join("ir-lifted.hashes")).unwrap_or_default();
    if let Err(what) = iridia_ir::gate::check(&committed, &current) {
        let new = inputs::scratch().join("dxil-ir-lifted.hashes");
        std::fs::write(&new, &current).unwrap();
        panic!("{what}; the current table is {}", new.display());
    }
}
