#![allow(clippy::unwrap_used)]
//! Lifting real programs: every DXIL container in a folder of extracted programs (the user's own install) is lifted,
//! the lifted program validated and written as WGSL (as its functions where its stage has no WGSL stage). A program the
//! lifter refuses is counted by reason, with the first program of each (the refusals are by name, never silent); a
//! program it lifts must validate and write, or the lifter is wrong. Needs the corpus oracle set's `skate/shaders` or
//! `elden/shaders` (each a folder, searched recursively, or an archive, of containers as the game ships them).
mod inputs;

mod common;

use std::collections::BTreeMap;

use inputs::Game;

#[derive(Default)]
struct Census {
    programs: usize,
    lifted: usize,
    validated: usize,
    written: usize,
    /// Refusals by reason: how many, and the first program.
    refused: BTreeMap<String, (usize, String)>,
    /// Lifted programs by stage.
    stages: BTreeMap<String, usize>,
    invalid: Vec<String>,
    /// Validated programs a writer cannot write, by reason, with the first program of each.
    unwritten: BTreeMap<String, (usize, String)>,
}

/// The text as wgpu's naga reads it: parsed and validated with every capability.
fn read_back(text: &str) -> Result<(), String> {
    let module = naga_hlsl::front::wgsl::parse_str(text).map_err(|e| e.emit_to_string(text))?;
    naga_hlsl::valid::Validator::new(
        naga_hlsl::valid::ValidationFlags::all(),
        naga_hlsl::valid::Capabilities::all(),
    )
    .validate(&module)
    .map(|_| ())
    .map_err(|e| e.to_string())
}

fn census(game: Game) {
    let Some(corpus) = common::corpus(game) else {
        return;
    };
    let mut c = Census::default();
    for file in &corpus.names {
        let bytes = corpus.read(file);
        let path = corpus.at(file);
        c.programs += 1;
        let mut program = match iridia_dxil::ir::lift(&bytes) {
            Ok(p) => p,
            Err(e) => {
                let entry = c.refused.entry(e.group()).or_insert((0, path.clone()));
                entry.0 += 1;
                continue;
            }
        };
        c.lifted += 1;
        *c.stages
            .entry(format!("{:?}", program.interface.stage))
            .or_default() += 1;
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
        "{}: {} programs, {} lifted {:?}, {} validated, {} written as WGSL, {} refused",
        corpus.name,
        c.programs,
        c.lifted,
        c.stages,
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
    for line in c.invalid.iter().take(40) {
        println!("  INVALID {line}");
    }
    assert!(
        c.invalid.is_empty(),
        "{} lifted programs failed validation or writing",
        c.invalid.len()
    );
}

#[test]
fn skate_programs() {
    census(Game::Skate);
}
#[test]
fn elden_ring_programs() {
    census(Game::Elden);
}
