#![allow(clippy::unwrap_used)]
//! Lifting real programs: every DXBC container in a folder of extracted programs (the user's own install) is lifted,
//! the lifted program validated and written as WGSL. A program the lifter refuses is counted by reason (the refusals
//! are by name, never silent); a program it lifts must validate and write, or the lifter is wrong. Needs the corpus
//! oracle set's `sekiro/shaders`, `riders/shaders` or `sledding/shaders` (each a folder, searched recursively, or an
//! archive, of containers as the game ships them).
mod inputs;

use std::collections::BTreeMap;

use inputs::Game;

mod common;

#[derive(Default)]
struct Census {
    programs: usize,
    lifted: usize,
    validated: usize,
    written: usize,
    /// Refusals by reason: how many, and the first program.
    refused: BTreeMap<String, (usize, String)>,
    invalid: Vec<String>,
    /// Validated programs a writer cannot write, by reason, with the first program of each.
    unwritten: BTreeMap<String, (usize, String)>,
}

/// The text as wgpu's naga reads it: parsed and validated with every capability.
fn read_back(text: &str) -> Result<(), String> {
    let module = naga_wgsl::front::wgsl::parse_str(text).map_err(|e| e.emit_to_string(text))?;
    naga_wgsl::valid::Validator::new(
        naga_wgsl::valid::ValidationFlags::all(),
        naga_wgsl::valid::Capabilities::all(),
    )
    .validate(&module)
    .map(|_| ())
    .map_err(|e| e.to_string())
}

fn census(game: Game) {
    let Some(corpus) = common::corpus(game) else {
        return;
    };
    let name = &corpus.name;
    let mut c = Census::default();
    for file in &corpus.names {
        let bytes = corpus.read(file);
        let path = corpus.at(file);
        c.programs += 1;
        let mut program = match iridia_dxbc::ir::lift(&bytes) {
            Ok(p) => p,
            Err(e) => {
                c.refused.entry(e.group()).or_insert((0, path.clone())).0 += 1;
                continue;
            }
        };
        c.lifted += 1;
        match iridia_ir::validate(&program) {
            Ok(()) => c.validated += 1,
            Err(problems) => {
                c.invalid.push(format!("{}: {problems:?}", path));
                continue;
            }
        }
        // the form the writers take, then the text of iridia-wgsl for wgpu's naga; a program a pass or the writer
        // refuses by name is counted, not wrong, and a text naga does not read back is wrong
        let written = iridia_passes::prepare(&mut program).and_then(|()| {
            iridia_wgsl::wgsl(&program, &iridia_wgsl::Profile::NAGA).map_err(|e| e.to_string())
        });
        match written {
            Ok(text) => match read_back(&text) {
                Ok(()) => c.written += 1,
                Err(e) => c
                    .invalid
                    .push(format!("{}: the text does not read back: {e}", path)),
            },
            Err(detail) => c.unwritten.entry(detail).or_insert((0, path.clone())).0 += 1,
        }
    }
    println!(
        "{name}: {} programs, {} lifted, {} validated, {} written as WGSL, {} refused",
        c.programs,
        c.lifted,
        c.validated,
        c.written,
        c.refused.values().map(|(n, _)| n).sum::<usize>()
    );
    for (reason, (n, example)) in &c.refused {
        println!("  refused {n:5}  {reason}  (e.g. {example})");
    }
    for (reason, (n, example)) in &c.unwritten {
        println!("  not written {n:5}  {reason}  (e.g. {example})");
    }
    for line in c.invalid.iter().take(20) {
        println!("  INVALID {line}");
    }
    assert!(
        c.invalid.is_empty(),
        "{} lifted programs failed validation or writing",
        c.invalid.len()
    );
}

#[test]
fn sekiro_programs() {
    census(Game::Sekiro);
}
#[test]
fn riders_republic_programs() {
    census(Game::Riders);
}
#[test]
fn unity_programs() {
    census(Game::Sledding);
}
