//! The IR lift's version gate (`iridia_ir::gate`): what the lift makes of the committed fixtures is hashed under its
//! versions in `tests/ir-lifted.hashes`.
mod inputs;

use std::path::Path;

/// The IR lift's version gate (`iridia_ir::gate`): what `ir::lift` makes of the same fixtures is hashed under the IR
/// lifter's `LIFTER_VERSION` and `iridia_ir::VERSION` in `tests/ir-lifted.hashes`, so a change to what the IR lift makes
/// of a bytecode (a program it now lifts, one it now refuses, a different program) needs a bump that a cache keyed by
/// the bytecode sees. The current table is written to the scratch root (`dxbc-ir-lifted.hashes`) when they differ.
#[test]
fn what_the_ir_lift_makes_of_its_fixtures_is_its_versions() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut names: Vec<String> = std::fs::read_dir(dir.join("fixtures"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".dxbc"))
        .collect();
    names.extend(
        std::fs::read_dir(dir.join("fixtures/sm51"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".dxbc"))
            .map(|name| format!("sm51/{name}")),
    );
    names.sort();
    let outcomes: Vec<_> = names
        .into_iter()
        .map(|name| {
            let bytes = std::fs::read(dir.join("fixtures").join(&name)).unwrap();
            let outcome = iridia_dxbc::ir::lift(&bytes)
                .map(|p| iridia_ir::gate::lifted(&p))
                .map_err(|e| e.to_string());
            (name, outcome)
        })
        .collect();
    let current = iridia_ir::gate::table("iridia-dxbc-ir", iridia_dxbc::ir::LIFTER_VERSION, &outcomes);
    let committed = std::fs::read_to_string(dir.join("ir-lifted.hashes")).unwrap_or_default();
    if let Err(what) = iridia_ir::gate::check(&committed, &current) {
        let new = inputs::scratch().join("dxbc-ir-lifted.hashes");
        std::fs::write(&new, &current).unwrap();
        panic!("{what}; the current table is {}", new.display());
    }
}
