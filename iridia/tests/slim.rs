#![allow(clippy::unwrap_used)]
//! The pre-push gate over the slim shader set (`oracles.corpus.general/slim`): 8,287 DXBC, DXIL and Direct3D 9
//! programs that together reach every region and branch side of the lifters the full pool of 108,394 reaches. Each
//! program goes through the IR lift of its family:
//!
//! - `IR`, the IR lift (`iridia_dxbc::ir::lift`, `iridia_dxil::ir::lift`, `iridia_d3d9_ir::lift`): after
//!   `iridia_passes::prepare` (`structurize`, `promote`, `simplify` and `break_flags`), validated, each
//!   function's text printed, parsed back and printed again the same;
//! - `IR WGSL`, that program lowered to stock naga (`iridia_naga`, naga's own dialect) and written as WGSL, read back
//!   and validated by naga.
//!
//! What a column gives a program is `ok` or the step it stopped at and why (each number as N). tests/slim/<family>.tsv
//! lists every program and column that is not `ok`; a program that goes further or less far than the table says
//! fails the gate, and so does a panic. The current table is then written to the scratch root (the path is in the
//! failure): a lift that widens replaces the committed table with it in the same commit.
//!
//! Before pushing: `GAME_DATA=<the gate configuration> cargo test --release --locked -p iridia --test slim`.
mod inputs;

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use inputs::{Game, OracleSet, Query, expect_found, oracle_tree};
use primase_testdata::Tree;

/// What one column of one program gives: `ok`, or the step it stopped at and why.
type Outcome = String;

const OK: &str = "ok";

/// A reason with each run of digits as `N`, as the lifters' refusals count them (`group`).
fn grouped(reason: &str) -> String {
    let mut out = String::with_capacity(reason.len());
    let mut digits = false;
    for c in reason.chars() {
        if c.is_ascii_digit() {
            if !digits {
                out.push('N');
            }
            digits = true;
        } else {
            // a reason is one field of a line of the table
            out.push(if c == '\t' || c == '\n' { ' ' } else { c });
            digits = false;
        }
    }
    out
}

fn stop(step: &str, reason: impl std::fmt::Display) -> Outcome {
    grouped(&format!("{step}: {reason}"))
}

/// naga's reading of WGSL text: parsed and validated with every capability (what the writer may use is the profile's,
/// not this check's).
fn read_back(text: &str) -> Result<(), Outcome> {
    let module =
        naga::front::wgsl::parse_str(text).map_err(|e| stop("WGSL read back", e.emit_to_string(text)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| stop("WGSL read back, validation", e.emit_to_string(text)))?;
    Ok(())
}

/// Columns `IR` and `IR WGSL`: an IR lift's program after the cut-over's passes, its text round trip, and its WGSL.
fn ir(lifted: Result<iridia_ir::Program, iridia_ir::Refusal>) -> [Outcome; 2] {
    let not_reached = || "not reached".to_owned();
    let mut program = match lifted {
        Ok(p) => p,
        Err(e) => return [stop("lift", e.group()), not_reached()],
    };
    if let Err(e) = iridia_passes::prepare(&mut program) {
        return [stop("passes", e), not_reached()];
    }
    if let Err(problems) = iridia_ir::validate(&program) {
        let first = problems.first().map(|p| format!("{p:?}")).unwrap_or_default();
        return [stop("validate", first), not_reached()];
    }
    for f in std::iter::once(&program.entry).chain(program.patch.as_ref()) {
        let text = iridia_ir::text::print(f);
        let again = iridia_ir::text::parse(&text).map(|g| iridia_ir::text::print(&g));
        match again {
            Ok(again) if again == text => {}
            Ok(_) => return [stop("text", "printed again, the text differs"), not_reached()],
            Err(e) => return [stop("text parsed back", e), not_reached()],
        }
    }
    let wgsl = match iridia_naga::wgsl(&program, &iridia_naga::Profile::NAGA) {
        Ok(text) => read_back(&text).err().unwrap_or_else(|| OK.into()),
        Err(e) => stop("lower", e),
    };
    [OK.into(), wgsl]
}

/// A panic's message.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".into())
}

/// The slim set (a folder or an archive) and the names in it of the family's programs (`programs/<family>/<sha256>
/// .<family>`), sorted; `None` where it is not configured (the gate configuration requires it).
fn programs(family: &str) -> Option<(Tree, Vec<String>)> {
    let tree = oracle_tree(OracleSet::Corpus, Game::General, "slim")?;
    let folder = format!("programs/{family}/");
    let extension = format!(".{family}");
    let names: Vec<String> = tree
        .names()
        .into_iter()
        .filter(|n| {
            n.strip_prefix(&folder)
                .is_some_and(|f| !f.contains('/') && f.ends_with(&extension))
        })
        .map(str::to_owned)
        .collect();
    expect_found(
        Query::Oracle(OracleSet::Corpus, Game::General, "slim"),
        &format!("{family} programs"),
        names.len(),
    );
    Some((tree, names))
}

/// Every program of `family` through `columns` (each column's outcome, in the order `names` gives them), held to
/// tests/slim/`family`.tsv.
fn gate(family: &str, names: &[&str], columns: impl Fn(&[u8]) -> Vec<Outcome> + Sync) {
    let Some((tree, files)) = programs(family) else {
        return;
    };
    // a folder's files are read one at a time: files of the NAS's SMB share opened at once on several threads fail at
    // random with Windows error 1450 (ERROR_NO_SYSTEM_RESOURCES); an archive is read at positions of one open file
    let folder_reads = tree.as_folder().map(|_| Mutex::new(()));
    let started = Instant::now();
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<Vec<Outcome>>> = files.iter().map(|_| Mutex::default()).collect();
    std::thread::scope(|s| {
        for _ in 0..inputs::threads().get() {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    let Some(file) = files.get(i) else { break };
                    let bytes = {
                        let _one_at_a_time = folder_reads.as_ref().map(|m| m.lock().unwrap());
                        tree.read(file).unwrap()
                    };
                    let outcomes = catch_unwind(AssertUnwindSafe(|| columns(&bytes)))
                        .unwrap_or_else(|p| vec![format!("panic: {}", panic_message(&*p)); names.len()]);
                    *results[i].lock().unwrap() = outcomes;
                }
            });
        }
    });
    let took = started.elapsed();
    // a program's name: its sha256, the file's name without the folder and the extension
    let name = |file: &str| {
        let stem = file.rsplit('/').next().unwrap_or(file);
        stem.rsplit_once('.').map_or(stem, |(s, _)| s).to_owned()
    };
    let outcomes: Vec<(String, Vec<Outcome>)> = files
        .iter()
        .zip(results)
        .map(|(file, r)| (name(file), r.into_inner().unwrap()))
        .collect();

    // the report: each column's count of ok, and of each step and reason it stopped at
    println!(
        "slim {family}: {} programs, {} thread(s), {:.1} s ({})",
        outcomes.len(),
        inputs::threads(),
        took.as_secs_f64(),
        tree.path().display()
    );
    for (c, column) in names.iter().enumerate() {
        let mut by: BTreeMap<&str, (usize, &str)> = BTreeMap::new();
        for (program, o) in &outcomes {
            by.entry(o[c].as_str()).or_insert((0, program)).0 += 1;
        }
        let ok = by.remove(OK).map_or(0, |(n, _)| n);
        println!("  {column}: {ok} ok, {} not", outcomes.len() - ok);
        let mut rest: Vec<_> = by.into_iter().collect();
        rest.sort_by(|x, y| y.1.0.cmp(&x.1.0).then(x.0.cmp(y.0)));
        for (outcome, (n, example)) in rest {
            println!("    {n:5}  {outcome}  (e.g. {example})");
        }
    }

    let panics: Vec<&str> = outcomes
        .iter()
        .filter(|(_, o)| o.iter().any(|x| x.starts_with("panic: ")))
        .map(|(p, _)| p.as_str())
        .collect();
    let table = |outcomes: &[(String, Vec<Outcome>)]| {
        let mut out = format!(
            "# What the slim set's {family} programs give that is not ok (tests/slim.rs): program, column, where it \
             stopped.\n"
        );
        for (program, o) in outcomes {
            for (column, outcome) in names.iter().zip(o) {
                if outcome != OK {
                    out.push_str(&format!("{program}\t{column}\t{outcome}\n"));
                }
            }
        }
        out
    };
    let current = table(&outcomes);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/slim/{family}.tsv"));
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .replace("\r\n", "\n");
    let differ = differences(&committed, &current);
    if panics.is_empty() && differ.is_empty() {
        return;
    }
    let written = inputs::scratch().join(format!("slim-{family}.tsv"));
    std::fs::write(&written, &current).unwrap();
    let mut why = Vec::new();
    if !panics.is_empty() {
        why.push(format!("{} programs panic: {}", panics.len(), panics.join(", ")));
    }
    if !differ.is_empty() {
        why.push(format!(
            "{} outcomes are not the table's ({}):\n{}",
            differ.len(),
            path.display(),
            differ.iter().take(30).cloned().collect::<Vec<_>>().join("\n")
        ));
    }
    panic!(
        "{}\nthe current table is {} (it replaces the committed one where a lift widens)",
        why.join("\n"),
        written.display()
    );
}

/// The lines in which two tables differ, each as `program column: the table's -> now`.
fn differences(committed: &str, current: &str) -> Vec<String> {
    let parse = |t: &str| -> BTreeMap<(String, String), String> {
        t.lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .filter_map(|l| {
                let mut f = l.splitn(3, '\t');
                Some(((f.next()?.into(), f.next()?.into()), f.next()?.into()))
            })
            .collect()
    };
    let (was, now) = (parse(committed), parse(current));
    let mut keys: Vec<_> = was.keys().chain(now.keys()).collect();
    keys.sort();
    keys.dedup();
    keys.into_iter()
        .filter_map(|k| {
            let (a, b) = (
                was.get(k).map_or(OK, String::as_str),
                now.get(k).map_or(OK, String::as_str),
            );
            (a != b).then(|| format!("  {} {}: {a} -> {b}", k.0, k.1))
        })
        .collect()
}

#[test]
fn dxbc() {
    gate("dxbc", &["IR", "IR WGSL"], |bytes| {
        let [b, c] = ir(iridia_dxbc::ir::lift(bytes));
        vec![b, c]
    });
}

#[test]
fn dxil() {
    gate("dxil", &["IR", "IR WGSL"], |bytes| {
        let [b, c] = ir(iridia_dxil::ir::lift(bytes));
        vec![b, c]
    });
}

#[test]
fn d3d9() {
    gate("d3d9", &["IR", "IR WGSL"], |bytes| {
        let [b, c] = ir(iridia_d3d9_ir::lift(bytes));
        vec![b, c]
    });
}
