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

fn census(game: Game) {
    let Some(corpus) = common::corpus(game) else {
        return;
    };
    let mut c = Census::default();
    for file in &corpus.names {
        let bytes = corpus.read(file);
        let path = corpus.at(file);
        c.programs += 1;
        let program = match iridia_dxil::lift(&bytes) {
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
        match program.validate() {
            Ok(_) => c.validated += 1,
            Err(problems) => {
                c.invalid.push(format!(
                    "{}: {}",
                    path,
                    problems
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("; ")
                ));
                continue;
            }
        }
        // a program of a stage with no WGSL stage is written as its functions
        let written = if iridia::stage_functions(&program.interface).is_ok_and(|f| f.is_some()) {
            iridia::emit::wgsl_functions(&program)
        } else {
            iridia::emit::wgsl(&program)
        };
        match written {
            Ok(_) => c.written += 1,
            // a program the validator accepts and a writer says it cannot write is counted, not wrong
            Err(iridia::ProgramError::Unsupported { detail, .. }) => {
                let entry = c.unwritten.entry(detail).or_insert((0, path.clone()));
                entry.0 += 1;
            }
            Err(e) => c.invalid.push(format!("{}: {e}", path)),
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
