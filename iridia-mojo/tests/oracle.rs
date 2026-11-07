#![allow(
    clippy::unwrap_used,
    reason = "a real-data oracle: a broken input should stop it"
)]
//! MonoGame's GLSL of real Direct3D 9 programs (the oracle sets `mojoshader/*`, made by `tests/oracles/mojoshader.py`)
//! lifted onto the IR and run beside the Direct3D 9 programs they were translated from (`tests/pairing`): every pair
//! both lift must agree, in every form. A set that is not configured prints SKIP.
//!
//! What a pair gives is counted by class; the classes other than agreement are the dialect's limits and the D3D9
//! lift's, each counted in `tests/oracle.tsv` (set, class, count): a set whose counts change fails, and its current
//! counts are written to the scratch root's `mojoshader-oracle.tsv`.
mod inputs;
mod pairing;

use std::io::Write;
use std::sync::Mutex;

use inputs::{Game, OracleSet, oracle_tree};
use pairing::Tally;
use primase_testdata::Tree;
use sha2::{Digest, Sha256};

/// Seeded runs per pair (a pixel program's run is a quad).
const RUNS: u64 = 8;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every pair of `glsl`'s `programs.jsonl` whose bytes `source` holds, by class.
fn tally(glsl: &Tree, source: &Tree) -> Tally {
    let lines = pairing::lines(&String::from_utf8(glsl.read("programs.jsonl").unwrap()).unwrap());
    assert!(!lines.is_empty(), "programs.jsonl lists no programs");
    let tally = Mutex::new(Tally::default());
    // with MOJO_ORACLE_DUMP set, every pair that does not agree is appended to that file: its class, example and GLSL
    let dump = std::env::var_os("MOJO_ORACLE_DUMP").map(|path| {
        Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap(),
        )
    });
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..inputs::threads().get() {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(line) = lines.get(i) else { break };
                    let bytes = source.read(&line.name).unwrap();
                    assert_eq!(
                        hex(&Sha256::digest(&bytes)),
                        line.sha256,
                        "{} is not the program the GLSL was made from",
                        line.name
                    );
                    let (class, example) = pairing::pair(&line.name, &bytes, line, RUNS);
                    if let Some(dump) = &dump
                        && !class.starts_with("agree")
                    {
                        let glsl = line.glsl.as_deref().unwrap_or_default();
                        let mut file = dump.lock().unwrap();
                        writeln!(file, "=== {class}\n{example}\n{glsl}").unwrap();
                    }
                    tally.lock().unwrap().note(class, example);
                }
            });
        }
    });
    tally.into_inner().unwrap()
}

/// The committed counts of `set`: (class, count).
fn committed(set: &str) -> Vec<(String, usize)> {
    include_str!("oracle.tsv")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .filter_map(|l| {
            let mut f = l.split('\t');
            match (f.next(), f.next(), f.next()) {
                (Some(s), Some(c), Some(n)) if s == set => Some((c.to_owned(), n.parse().unwrap())),
                _ => None,
            }
        })
        .collect()
}

/// Check `set`'s tally: no pair differs or fails to run, and the counts are the committed ones.
fn check(set: &str, tally: &Tally) {
    println!("{set}:");
    tally.print();
    let current: Vec<(String, usize)> = tally.classes().map(|(c, n)| (c.clone(), n)).collect();
    // a pair agrees (`AGREE`, `AGREE_NAN`, `KILLED`, `PRINTED`, any with the older lift's part), differs where MojoShader's
    // GLSL departs from its program (`DEPARTS`), has an output the GLSL does not write, or is refused by a lift;
    // anything else is broken
    let fine = |c: &str| {
        [
            "agree",
            pairing::DEPARTS,
            "an output the GLSL does not write",
            "MojoShader refused",
            "D3D9 lift refused",
            "GLSL lift refused",
        ]
        .iter()
        .any(|p| c.starts_with(p))
    };
    let broken: Vec<&(String, usize)> = current.iter().filter(|(c, _)| !fine(c)).collect();
    let table: String = current
        .iter()
        .map(|(c, n)| format!("{set}\t{c}\t{n}\n"))
        .collect();
    if !broken.is_empty() || current != committed(set) {
        let path = inputs::scratch().join("mojoshader-oracle.tsv");
        let mut all = std::fs::read_to_string(&path).unwrap_or_default();
        all.push_str(&table);
        std::fs::write(&path, all).unwrap();
        panic!(
            "{set}: {} pairs disagree or did not run, or the counts changed; the current table is in {}",
            broken.iter().map(|(_, n)| n).sum::<usize>(),
            path.display()
        );
    }
    let agreed = current
        .iter()
        .filter(|(c, _)| c.starts_with("agree"))
        .map(|(_, n)| n)
        .sum::<usize>();
    println!("{set}: {agreed} pairs agree");
}

#[test]
fn the_glsl_of_microsofts_sm3_programs_runs_as_they_do() {
    let (Some(glsl), Some(source)) = (
        oracle_tree(OracleSet::MojoShader, Game::General, "sm3"),
        oracle_tree(OracleSet::D3dDisassembler, Game::General, "sm3"),
    ) else {
        return;
    };
    check("general/sm3", &tally(&glsl, &source));
}

#[test]
fn the_glsl_of_the_slim_sets_d3d9_programs_runs_as_they_do() {
    let (Some(glsl), Some(source)) = (
        oracle_tree(OracleSet::MojoShader, Game::General, "slim-d3d9"),
        oracle_tree(OracleSet::Corpus, Game::General, "slim"),
    ) else {
        return;
    };
    check("general/slim-d3d9", &tally(&glsl, &source));
}

#[test]
fn the_glsl_of_shaun_whites_programs_runs_as_they_do() {
    let (Some(glsl), Some(source)) = (
        oracle_tree(OracleSet::MojoShader, Game::ShaunWhite, "sdb2"),
        oracle_tree(OracleSet::D3dDisassembler, Game::ShaunWhite, "sdb2"),
    ) else {
        return;
    };
    check("shaun-white/sdb2", &tally(&glsl, &source));
}
