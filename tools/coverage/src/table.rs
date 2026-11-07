//! The coverage table: per construct (an opcode or a stage of a bytecode), how many of the suites' programs that use
//! it dx decodes, dx lists exactly as Microsoft's tools do, iridia lifts onto its IR and through its passes, writes as
//! WGSL that naga reads back, and runs on the IR to the result the suite expects; and per test, what it came to, on the
//! IR and on the lift it replaces (the reference). Everything here is measured from the case files and the runs:
//! nothing is written by hand.
//!
//! A program counts as executed `pass` for a construct when some run (vkd3d) or test (offload) using it passes, and
//! `fail` when every run using it that got as far as running failed.
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::front::{self, Bytecode, Ready, Stop};
use crate::inflate::read_text;
use crate::offload;
use crate::shard::{self, Shard};
use crate::vkd3d;

/// The table's schema version.
pub const VERSION: u32 = 2;

/// What the table grades, and its reference, as the JSON's `graded` names them.
const GRADED: [(&str, &str); 5] = [
    (
        "lift",
        "iridia's IR, by each bytecode's IR lift: iridia-dxbc's (DXBC), iridia-dxil's (DXIL), iridia-d3d9-ir \
         (Direct3D 9)",
    ),
    (
        "passes",
        "iridia_passes::prepare (structurize where the lift makes a graph, promote, simplify, break_flags), then the \
         IR validated",
    ),
    ("executor", "iridia-exec on the IR"),
    (
        "wgsl",
        "iridia-naga's lowering for browsers, read back and validated by naga: information, which decides no run",
    ),
    (
        "reference",
        "reference_status: the lift the IR replaces (DXBC and DXIL lifted into naga modules; a Direct3D 9 program's \
         WGSL read back by naga, its IR drawn again and held to it), run by iridia-exec, until it is removed",
    ),
];

/// The groups a status falls in, in the order the page shows them, and what each means.
const GROUPS: [(&str, &str); 6] = [
    ("pass", "it ran and gave what the test expects"),
    (
        "gap",
        "iridia's to close: it ran and gave something else, or iridia refused or could not run one of its programs",
    ),
    (
        "runner-limit",
        "the runner does not model something the test needs: pipeline state, an offloader step, a feature a \
         Direct3D 12 device can have and iridia's does not",
    ),
    (
        "not-applicable",
        "nothing to run on Direct3D: the test is for another API or compiler, Microsoft's compiler refuses its \
         shader, or vkd3d marks that Windows itself reads otherwise",
    ),
    (
        "expected",
        "a line fails where the test expects Direct3D to fail (XFAIL)",
    ),
    (
        "unexpected-pass",
        "every line passes where the test expects Direct3D to fail (XPASS, which lit counts as a failure)",
    ),
];

/// Each status a row can have: its group ([`GROUPS`]) and what it means.
const STATUSES: [(&str, &str, &str); 10] = [
    ("pass", "pass", "every probe or check holds"),
    ("fail", "gap", "it ran and a probe or check does not hold"),
    (
        "error",
        "gap",
        "iridia took the programs and could not run them: a pass, validation or the executor failed",
    ),
    (
        "refused",
        "gap",
        "iridia's lift or executor refused one of its programs, by name",
    ),
    (
        "not-modelled",
        "runner-limit",
        "the runner does not model something the test needs",
    ),
    (
        "not-applicable",
        "not-applicable",
        "lit runs it on no Direct3D 12 device in this configuration: it is for another API or compiler",
    ),
    (
        "compiler-refused",
        "not-applicable",
        "Microsoft's compiler refuses a shader the test needs",
    ),
    (
        "winbug",
        "not-applicable",
        "every probe deciding it is one vkd3d marks winbug where Windows itself reads otherwise",
    ),
    (
        "xfail",
        "expected",
        "a line fails where the test expects a failure (XFAIL)",
    ),
    (
        "xpass",
        "unexpected-pass",
        "every line passes where the test expects a failure (XPASS)",
    ),
];

/// A status's group.
fn group_of(status: &str) -> Result<&'static str, String> {
    STATUSES
        .iter()
        .find(|s| s.0 == status)
        .map(|s| s.1)
        .ok_or_else(|| format!("the status {status}, which the table does not name"))
}

/// vkd3d's runner's classes in the table's statuses.
fn vkd3d_status(class: &str) -> &str {
    match class {
        "failed" => "error",
        "not modelled" => "not-modelled",
        "compiler refused" => "compiler-refused",
        other => other,
    }
}

/// A message with its program names (16 hexadecimal digits) and its numbers taken out, so one kind of outcome is one
/// class.
pub fn class_of(e: &str) -> String {
    let mut s = String::new();
    let mut rest = e;
    while !rest.is_empty() {
        let word = rest.find(|c: char| !c.is_ascii_hexdigit()).unwrap_or(rest.len());
        if word == 16 {
            s.push_str("<program>");
            rest = &rest[word..];
            continue;
        }
        let end = word.max(1);
        let end = rest.char_indices().nth(end).map_or(rest.len(), |(i, _)| i);
        let piece = &rest[..end];
        s.extend(piece.chars().map(|c| if c.is_ascii_digit() { 'N' } else { c }));
        rest = &rest[end..];
    }
    s.chars().take(160).collect()
}

/// A status and a class of reason ([`class_of`]).
type Reason = (String, String);

/// How many runs or tests a reason has, and the first's example.
type Count = (usize, String);

/// What is left of a suite by reason: per status and class of reason, the count and a first example.
#[derive(Default)]
struct Reasons(BTreeMap<Reason, Count>);

impl Reasons {
    fn add(&mut self, status: &str, reason: &str, example: impl FnOnce() -> String) {
        self.0
            .entry((status.to_string(), class_of(reason)))
            .or_insert_with(|| (0, example()))
            .0 += 1;
    }

    /// Most first, then by status and reason.
    fn sorted(&self) -> Vec<(&Reason, &Count)> {
        let mut by: Vec<_> = self.0.iter().collect();
        by.sort_by(|a, b| b.1.0.cmp(&a.1.0).then_with(|| a.0.cmp(b.0)));
        by
    }

    fn json(&self) -> Result<Vec<Value>, String> {
        self.sorted()
            .into_iter()
            .map(|((status, reason), (count, example))| {
                Ok(json!({
                    "status": status,
                    "group": group_of(status)?,
                    "reason": reason,
                    "count": count,
                    "example": example,
                }))
            })
            .collect()
    }

    /// The Markdown list of what is left, by group then most first.
    fn markdown(&self, out: &mut String, unit: &str) -> Result<(), String> {
        for (group, _) in GROUPS.iter().filter(|g| !matches!(g.0, "pass" | "expected")) {
            let mut lines = Vec::new();
            let mut total = 0;
            for ((status, reason), (count, _)) in self.sorted() {
                if group_of(status)? == *group {
                    total += count;
                    lines.push(format!("- {count} {status}: {}", reason.replace('|', "\\|")));
                }
            }
            if !lines.is_empty() {
                let _ = writeln!(out, "\n{group}, {total} {unit}:\n\n{}", lines.join("\n"));
            }
        }
        Ok(())
    }
}

/// A row of counts by status: the status, its group and a count per column.
type StatusRow = (&'static str, &'static str, Vec<usize>);

/// Counts by status, in [`STATUSES`]' order and every status present, for a Markdown table.
fn status_rows(columns: &[&BTreeMap<String, usize>]) -> Result<Vec<StatusRow>, String> {
    let present: BTreeSet<&String> = columns.iter().flat_map(|c| c.keys()).collect();
    for s in &present {
        group_of(s)?;
    }
    Ok(STATUSES
        .iter()
        .filter(|s| present.iter().any(|p| p.as_str() == s.0))
        .map(|s| {
            (
                s.0,
                s.1,
                columns.iter().map(|c| c.get(s.0).copied().unwrap_or(0)).collect(),
            )
        })
        .collect())
}
/// A program's constructs as dx decodes it: its stage (`ps_5_0`) and its opcodes, or why it does not decode.
fn constructs(bytes: &[u8]) -> (Bytecode, Result<(String, BTreeSet<String>), String>) {
    let bytecode = Bytecode::of(bytes);
    let found = match bytecode {
        Bytecode::D3d9 => d3d9_reader::read(bytes).map_err(|e| e.to_string()).map(|p| {
            let stage = format!(
                "{}_{}_{}",
                if p.version.stage == d3d9_reader::Stage::Pixel {
                    "ps"
                } else {
                    "vs"
                },
                p.version.major,
                p.version.minor
            );
            let mut ops = BTreeSet::new();
            for item in &p.items {
                match item {
                    d3d9_reader::Item::Instruction(i) => {
                        ops.insert(i.opcode.name().to_string());
                    }
                    d3d9_reader::Item::Declaration(_) => {
                        ops.insert("dcl".to_string());
                    }
                    d3d9_reader::Item::Definition(_) => {
                        ops.insert("def".to_string());
                    }
                    d3d9_reader::Item::Comment(_) => {}
                }
            }
            (stage, ops)
        }),
        Bytecode::Dxbc => dxbc_reader::code(bytes)
            .and_then(|words| dxbc_reader::decode(&words))
            .map_err(|e| e.to_string())
            .map(|d| {
                let stage = format!(
                    "{}_{}_{}",
                    match d.kind {
                        dxbc_reader::ProgramType::Pixel => "ps",
                        dxbc_reader::ProgramType::Vertex => "vs",
                        dxbc_reader::ProgramType::Geometry => "gs",
                        dxbc_reader::ProgramType::Hull => "hs",
                        dxbc_reader::ProgramType::Domain => "ds",
                        _ => "cs",
                    },
                    d.major,
                    d.minor
                );
                let mut ops: BTreeSet<String> = d
                    .instructions
                    .iter()
                    .map(|i| dxbc_reader::opcode_name(i.opcode).to_string())
                    .collect();
                ops.extend(
                    d.declarations
                        .iter()
                        .map(|i| dxbc_reader::opcode_name(i.opcode).to_string()),
                );
                (stage, ops)
            }),
        Bytecode::Dxil => dxil_reader::operations(bytes)
            .map_err(|e| e.to_string())
            .map(|(kind, ops)| {
                let stage = match kind {
                    0 => "ps",
                    1 => "vs",
                    2 => "gs",
                    3 => "hs",
                    4 => "ds",
                    5 => "cs",
                    6 => "lib",
                    13 => "ms",
                    14 => "as",
                    _ => "other",
                };
                (format!("{stage}_6"), ops.into_keys().collect())
            }),
    };
    (bytecode, found)
}

/// dx's listing of a program.
fn listing(bytes: &[u8]) -> Result<String, String> {
    match Bytecode::of(bytes) {
        Bytecode::D3d9 => d3d9_reader::disassemble(bytes).map_err(|e| e.to_string()),
        Bytecode::Dxbc => dxbc_reader::disassemble(bytes).map_err(|e| e.to_string()),
        Bytecode::Dxil => dxil_reader::disasm::disassemble(bytes).map_err(|e| e.to_string()),
    }
}

/// How one program fared, column by column.
#[derive(Clone, Debug, Default)]
struct Fate {
    bytecode: Option<Bytecode>,
    stage: Option<String>,
    ops: BTreeSet<String>,
    decoded: bool,
    /// `Some(true)` when dx's listing is Microsoft's, `None` when there is no Microsoft listing to compare.
    listed: Option<bool>,
    lifted: Option<bool>,
    wgsl: Option<bool>,
    /// Some run passed (`true`), or every run that ran it failed (`false`).
    executed: Option<bool>,
    suites: BTreeSet<&'static str>,
    passing_suites: BTreeSet<&'static str>,
}

/// Microsoft's listing next to dx's: the same text, line ends and trailing space aside.
fn same_listing(ours: &str, theirs: &str) -> bool {
    let norm = |s: &str| -> Vec<String> {
        let lines: Vec<String> = s.lines().map(|l| l.trim_end().to_string()).collect();
        let end = lines.iter().rposition(|l| !l.is_empty()).map_or(0, |i| i + 1);
        lines[..end].to_vec()
    };
    norm(ours) == norm(theirs)
}

fn note_lift(fate: &mut Fate, lifted: Option<&LiftFate>) {
    if let Some(l) = lifted {
        fate.lifted = Some(l.lifted);
        if let Some(w) = l.wgsl {
            fate.wgsl = Some(w);
        }
    }
}

/// The inputs the table is made of.
#[derive(Debug)]
pub struct Inputs<'a> {
    /// The vkd3d case files (d3d9, dxbc, dxil), each with its listings file beside it.
    pub vkd3d: Vec<&'a Path>,
    /// The offload case file.
    pub offload: Option<&'a Path>,
    /// iridia's commit.
    pub commit: String,
}

/// How far a program got on its path: lifted (and through the passes) or not, and whether WGSL was written for it and
/// read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiftFate {
    /// It lifted and went through the passes.
    pub lifted: bool,
    /// Where it lifted: its WGSL was written and read back.
    pub wgsl: Option<bool>,
}

impl LiftFate {
    fn of(lift: &Result<Arc<Ready>, Stop>) -> Self {
        LiftFate {
            lifted: lift.is_ok(),
            wgsl: lift.as_ref().ok().map(|r| r.lifted.wgsl.is_ok()),
        }
    }
}

/// A suite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Suite {
    /// vkd3d's HLSL shader tests.
    Vkd3d,
    /// LLVM's offload-test-suite.
    Offload,
}

/// A suite's results.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Runs {
    /// vkd3d's runs.
    Vkd3d(Vec<vkd3d::run::RunResult>),
    /// offload's tests.
    Offload(Vec<offload::run::TestResult>),
}

/// One case file run through one path, or one shard of it: the table's unit of work, which `iridia-coverage part` runs
/// alone (CI runs the parts side by side) and [`make`] puts together.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Part {
    /// The case file's name (`d3d9.json.gz`).
    pub cases: String,
    /// The path its programs went through.
    pub path: front::Path,
    /// Which cases of the file the part ran: all of them where the file is one part.
    #[serde(default)]
    pub shard: Shard,
    /// What its runs or tests came to.
    pub runs: Runs,
    /// How far each program got, by name.
    pub lifts: BTreeMap<String, LiftFate>,
}

/// A case file's name, as a [`Part`] names it.
fn cases_name(cases: &Path) -> String {
    cases
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

impl Part {
    /// Run the cases of `shard` of the `suite` case file `cases` through `path`, `threads` at a time; offload's image
    /// tests compare with the checkout of offload-golden-images `golden`.
    pub fn run(
        suite: Suite,
        cases: &Path,
        path: front::Path,
        shard: Shard,
        threads: usize,
        golden: Option<&Path>,
    ) -> Result<Part, String> {
        let text = read_text(cases)?;
        let (runs, lifts) = match suite {
            Suite::Vkd3d => {
                let file = vkd3d::cases::read(&text)?;
                let (results, lifts) = vkd3d::run::run_shard(&file, threads, path, shard);
                (Runs::Vkd3d(results), lifts)
            }
            Suite::Offload => {
                let file = offload::cases::read(&text)?;
                let programs = front::Programs::new(path, |name: &str| {
                    let text = file
                        .programs
                        .get(name)
                        .ok_or_else(|| format!("no program {name}"))?;
                    offload::pipeline::base64(text)
                });
                let results = offload::run::run_shard_with(&file, threads, golden, &programs, shard);
                (Runs::Offload(results), programs.lifted())
            }
        };
        Ok(Part {
            cases: cases_name(cases),
            path,
            shard,
            runs,
            lifts: lifts.iter().map(|(n, l)| (n.clone(), LiftFate::of(l))).collect(),
        })
    }
}

/// Run every part of the table: each case file of `inputs` through the IR and through the reference path.
pub fn run_parts(inputs: &Inputs<'_>, threads: usize, golden: Option<&Path>) -> Result<Vec<Part>, String> {
    let files = inputs
        .vkd3d
        .iter()
        .map(|p| (Suite::Vkd3d, *p))
        .chain(inputs.offload.map(|p| (Suite::Offload, p)));
    let mut parts = Vec::new();
    for (suite, cases) in files {
        for path in [front::Path::Ir, front::Path::Reference] {
            parts.push(Part::run(suite, cases, path, Shard::WHOLE, threads, golden)?);
        }
    }
    Ok(parts)
}

/// The part that ran case file `cases` through `path`: the one part of `parts` that did, or the shards of it put
/// together in the order of the file's cases.
fn part_of<'p>(parts: &'p [Part], cases: &Path, path: front::Path) -> Result<Cow<'p, Part>, String> {
    let name = cases_name(cases);
    let mut mine: Vec<&Part> = parts
        .iter()
        .filter(|p| p.cases == name && p.path == path)
        .collect();
    let Some(first) = mine.first().copied() else {
        return Err(format!("no part ran {name} through the {} path", path.name()));
    };
    if let [only] = mine.as_slice()
        && only.shard == Shard::WHOLE
    {
        return Ok(Cow::Borrowed(only));
    }
    let count = first.shard.count;
    mine.sort_by_key(|p| p.shard.index);
    let indexes: Vec<usize> = mine.iter().map(|p| p.shard.index).collect();
    if mine.iter().any(|p| p.shard.count != count) || indexes != (0..count).collect::<Vec<_>>() {
        return Err(format!(
            "{name} through the {} path: the parts are shards {indexes:?} of {:?}, which are not all of 0..{count}",
            path.name(),
            mine.iter().map(|p| p.shard.count).collect::<Vec<_>>()
        ));
    }
    let mut lifts = BTreeMap::new();
    for p in &mine {
        lifts.extend(p.lifts.iter().map(|(n, l)| (n.clone(), *l)));
    }
    let runs = match first.runs {
        Runs::Vkd3d(_) => {
            let mut shards = Vec::new();
            for p in &mine {
                let Runs::Vkd3d(r) = &p.runs else {
                    return Err(format!("{name}: shards of different suites"));
                };
                shards.push(r.clone());
            }
            Runs::Vkd3d(shard::interleave(shards).map_err(|e| format!("{name}: {e}"))?)
        }
        Runs::Offload(_) => {
            let mut shards = Vec::new();
            for p in &mine {
                let Runs::Offload(r) = &p.runs else {
                    return Err(format!("{name}: shards of different suites"));
                };
                shards.push(r.clone());
            }
            Runs::Offload(shard::interleave(shards).map_err(|e| format!("{name}: {e}"))?)
        }
    };
    Ok(Cow::Owned(Part {
        cases: name,
        path,
        shard: Shard::WHOLE,
        runs,
        lifts,
    }))
}

fn listings_of(path: &Path) -> BTreeMap<String, String> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let stem = name.trim_end_matches(".gz").trim_end_matches(".json");
    let listing = path.with_file_name(format!("{stem}.listings.json.gz"));
    read_text(&listing)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Make the table of the suites' runs ([`run_parts`]): the JSON the demo site reads and a Markdown summary. The table
/// grades each suite's runs on the IR ([`front::Path::Ir`]); its runs on [`front::Path::Reference`] are each row's
/// `reference_status`.
pub fn make(inputs: &Inputs<'_>, parts: &[Part]) -> Result<(Value, String), String> {
    let mut fates: BTreeMap<String, Fate> = BTreeMap::new();
    let mut suites = Vec::new();
    let mut summary = String::new();
    let graded: serde_json::Map<String, Value> =
        GRADED.iter().map(|(k, v)| ((*k).to_string(), json!(v))).collect();
    let _ = writeln!(
        summary,
        "The table grades iridia's IR path: {}; run by {}. The WGSL column: {}. Each row's {}.\n",
        GRADED[0].1, GRADED[2].1, GRADED[3].1, GRADED[4].1
    );
    let _ = writeln!(summary, "| Group | Meaning |\n|---|---|");
    for (group, meaning) in GROUPS {
        let _ = writeln!(summary, "| {group} | {meaning} |");
    }
    summary.push('\n');

    // vkd3d
    let mut rows = Vec::new();
    // the runs vkd3d's runner runs and their selected probes; and apart, the runs it runs none of and the extended
    // probes of the others
    let mut totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut probe_totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut extended_totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut extended_probe_totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut reference_totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut reasons = Reasons::default();
    // the probes vkd3d marks `winbug`: those apart by what they found, and the driver bugs' count
    let mut winbug_outcomes: BTreeMap<String, usize> = BTreeMap::new();
    let mut driver_bugs = 0;
    let mut source = String::new();
    for path in &inputs.vkd3d {
        let file = vkd3d::cases::read(&read_text(path)?)?;
        source.push_str(&file.source);
        source.push_str("; ");
        let listings = listings_of(path);
        let ir = part_of(parts, path, front::Path::Ir)?;
        let reference = part_of(parts, path, front::Path::Reference)?;
        let (Runs::Vkd3d(results), Runs::Vkd3d(reference)) = (&ir.runs, &reference.runs) else {
            return Err(format!("{}: parts of another suite", path.display()));
        };
        let lifted = &ir.lifts;
        let row_id = |r: &vkd3d::run::RunResult| format!("{}:{}:{}", r.file, r.runner, r.model);
        let reference: BTreeMap<String, String> = reference
            .iter()
            .map(|r| (row_id(r), vkd3d_status(r.class()).to_string()))
            .collect();
        for name in file.programs.keys() {
            let bytes = vkd3d::cases::program(&file, name)?;
            let fate = fates.entry(name.clone()).or_default();
            fate.suites.insert("vkd3d");
            fate_of(fate, &bytes, listings.get(name));
            note_lift(fate, lifted.get(name));
        }
        for r in results {
            let id = row_id(r);
            let class = vkd3d_status(r.class());
            group_of(class)?;
            let reference_status = reference
                .get(&id)
                .ok_or_else(|| format!("{id} has no run on the reference path"))?;
            let selected = r.selected();
            let runs = if selected {
                *reference_totals.entry(reference_status.clone()).or_default() += 1;
                &mut totals
            } else {
                &mut extended_totals
            };
            *runs.entry(class.to_string()).or_default() += 1;
            for p in &r.probes {
                let probes = if p.selected {
                    &mut probe_totals
                } else {
                    &mut extended_probe_totals
                };
                *probes.entry(vkd3d_status(p.class()).to_string()).or_default() += 1;
                if p.apart() {
                    *winbug_outcomes
                        .entry(vkd3d_status(p.outcome.class()).to_string())
                        .or_default() += 1;
                } else if p.winbug.is_some() {
                    driver_bugs += 1;
                }
            }
            if selected && let Some(p) = r.deciding() {
                reasons.add(class, &p.outcome.reason(&r.file), || {
                    format!("{}:{}: {}", r.file, p.line, p.outcome.detail())
                });
            }
            // the bytecode of the run's programs; for a run with none (the compiler refused every shader), its runner's
            let format = r
                .programs
                .keys()
                .next()
                .and_then(|n| fates.get(n))
                .and_then(|f| f.bytecode)
                .map_or_else(
                    || match r.runner.as_str() {
                        "d3d9" => "d3d9",
                        "d3d12-dxc" => "dxil",
                        _ => "dxbc",
                    },
                    Bytecode::name,
                );
            let detail = r
                .deciding()
                .map(|p| {
                    format!(
                        "line {}: {}",
                        p.line,
                        p.outcome.detail().lines().next().unwrap_or("")
                    )
                })
                .unwrap_or_default();
            rows.push(json!({
                "id": id,
                "case": r.file,
                "runner": r.runner,
                "format": format,
                "model": r.model,
                "status": class,
                "group": group_of(class)?,
                "detail": detail,
                "probes": r.probes.len(),
                "selected": selected,
                "selected_probes": r.probes.iter().filter(|p| p.selected).count(),
                "extended_status": r.extended_class().map(vkd3d_status),
                "reference_status": reference_status,
            }));
            for (name, ran) in &r.programs {
                let Some(f) = fates.get_mut(name) else { continue };
                if ran.is_some() {
                    continue;
                }
                match class {
                    "pass" => {
                        f.executed = Some(true);
                        f.passing_suites.insert("vkd3d");
                    }
                    "fail" | "error" if f.executed.is_none() => f.executed = Some(false),
                    _ => {}
                }
            }
        }
    }
    let _ = writeln!(
        summary,
        "## vkd3d HLSL shader tests\n\n{}\n",
        source.trim_end_matches("; ")
    );
    let _ = writeln!(
        summary,
        "Runs vkd3d's runner runs, by their selected probes; and apart, the extended ones: the runs it runs none of, \
         and the probes of the others it leaves out (the same expectations at a model vkd3d's own runs leave out). \
         The reference column is the runs vkd3d's runner runs on the reference path.\n"
    );
    let _ = writeln!(
        summary,
        "| Status | Group | Runs | Probes | Extended runs | Extended probes | Reference runs |\n\
         |---|---|---:|---:|---:|---:|---:|"
    );
    for (s, group, n) in status_rows(&[
        &totals,
        &probe_totals,
        &extended_totals,
        &extended_probe_totals,
        &reference_totals,
    ])? {
        let _ = writeln!(
            summary,
            "| {s} | {group} | {} | {} | {} | {} | {} |",
            n[0], n[1], n[2], n[3], n[4]
        );
    }
    let found: Vec<String> = winbug_outcomes.iter().map(|(s, n)| format!("{s} {n}")).collect();
    let _ = writeln!(
        summary,
        "\nThe winbug probes are those vkd3d's tests mark `winbug` where Windows itself reads otherwise (a plain \
         `winbug`, or one of models or of devices it does not hold on): they decide no run, and found {}. The \
         {driver_bugs} marked `winbug` for a driver's bug (`amd`, `nvidia`, `warp=1.0.16`) are probes like any other.",
        if found.is_empty() {
            "nothing".to_string()
        } else {
            found.join(", ")
        }
    );
    let _ = write!(
        summary,
        "\nWhat is left of the runs vkd3d's runner runs, by reason:\n"
    );
    reasons.markdown(&mut summary, "runs")?;
    summary.push('\n');
    suites.push(json!({
        "id": "vkd3d",
        "name": "vkd3d HLSL shader tests",
        "source": "https://gitlab.winehq.org/wine/vkd3d/-/tree/master/tests/hlsl",
        "commit": source.trim_end_matches("; "),
        "licence": "LGPL-2.1-or-later",
        "description": "Each test file run at each shader model on each of vkd3d's Windows runners (d3d9, d3d11, d3d12 \
            with fxc, d3d12 with dxc), every step in order, every probe compared as vkd3d compares it. A row is one run: \
            pass when every probe holds that vkd3d's runner runs at that model (selected); a run it runs none of is \
            extended (selected false), its status that of all its probes, and the probes vkd3d leaves out of the \
            others are their extended_status: the same expectations at a model vkd3d's own runs leave out, which need \
            not hold on any device. A probe vkd3d's test marks winbug where Windows itself reads otherwise is counted \
            as winbug and decides no run (winbug_outcomes: what those found); one marked winbug for a driver's bug \
            (driver_bug_probes) is a probe like any other. reference_status is the run's status on the reference \
            path, reference_totals those of the selected runs.",
        "totals": totals,
        "probe_totals": probe_totals,
        "extended_totals": extended_totals,
        "extended_probe_totals": extended_probe_totals,
        "reference_totals": reference_totals,
        "winbug_outcomes": winbug_outcomes,
        "driver_bug_probes": driver_bugs,
        "reasons": reasons.json()?,
        "rows": rows,
    }));

    // offload
    if let Some(path) = inputs.offload {
        let file = offload::cases::read(&read_text(path)?)?;
        let ir = part_of(parts, path, front::Path::Ir)?;
        let reference = part_of(parts, path, front::Path::Reference)?;
        let (Runs::Offload(results), Runs::Offload(reference)) = (&ir.runs, &reference.runs) else {
            return Err(format!("{}: parts of another suite", path.display()));
        };
        let lifted = &ir.lifts;
        let reference: BTreeMap<&String, &'static str> =
            reference.iter().map(|r| (&r.test, r.outcome.class())).collect();
        for (name, text) in &file.programs {
            let bytes = offload::pipeline::base64(text)?;
            let fate = fates.entry(name.clone()).or_default();
            fate.suites.insert("offload");
            fate_of(fate, &bytes, None);
            note_lift(fate, lifted.get(name));
        }
        let mut rows = Vec::new();
        let mut totals: BTreeMap<String, usize> = BTreeMap::new();
        let mut reference_totals: BTreeMap<String, usize> = BTreeMap::new();
        let mut reasons = Reasons::default();
        for r in results {
            let class = r.outcome.class();
            let reference_status = *reference
                .get(&r.test)
                .ok_or_else(|| format!("{} has no run on the reference path", r.test))?;
            *totals.entry(class.to_string()).or_default() += 1;
            *reference_totals.entry(reference_status.to_string()).or_default() += 1;
            if !matches!(class, "pass" | "xfail" | "xpass") {
                reasons.add(class, &r.outcome.detail(), || {
                    format!("{}: {}", r.test, r.outcome.detail())
                });
            }
            // the suite is read for lit's DirectX configuration with DXC: a test that made no program (not applicable,
            // a runner limit, or one of the offloader's own) is DXIL's too
            let format = r
                .programs
                .first()
                .and_then(|n| fates.get(n))
                .and_then(|f| f.bytecode)
                .map_or("dxil", Bytecode::name);
            let model = r
                .programs
                .first()
                .and_then(|n| fates.get(n))
                .and_then(|f| f.stage.clone())
                .unwrap_or_default();
            rows.push(json!({
                "id": r.test,
                "case": r.test,
                "format": format,
                "model": model,
                "status": class,
                "group": group_of(class)?,
                "detail": r.outcome.detail(),
                "reference_status": reference_status,
            }));
            for name in &r.programs {
                let Some(f) = fates.get_mut(name) else { continue };
                match class {
                    "pass" | "xfail" => {
                        f.executed = Some(true);
                        f.passing_suites.insert("offload");
                    }
                    "fail" | "xpass" if f.executed.is_none() => f.executed = Some(false),
                    _ => {}
                }
            }
        }
        let _ = writeln!(summary, "## LLVM offload-test-suite\n\n{}\n", file.source);
        let _ = writeln!(
            summary,
            "| Status | Group | Tests | Reference tests |\n|---|---|---:|---:|"
        );
        for (s, group, n) in status_rows(&[&totals, &reference_totals])? {
            let _ = writeln!(summary, "| {s} | {group} | {} | {} |", n[0], n[1]);
        }
        let _ = write!(summary, "\nWhat is left, by reason:\n");
        reasons.markdown(&mut summary, "tests")?;
        summary.push('\n');
        suites.push(json!({
            "id": "offload",
            "name": "LLVM offload-test-suite",
            "source": "https://github.com/llvm/offload-test-suite",
            "commit": file.source,
            "licence": "Apache-2.0 WITH LLVM-exception",
            "description": "Each test run as lit runs it for DirectX with DXC: its RUN lines in order, the offloader's \
                pipeline on iridia's executor, its Results rules, FileCheck lines and golden images checked as the \
                suite's own tools check them. A test lit runs on no Direct3D 12 device in that configuration (a \
                REQUIRES no device's features meet, an UNSUPPORTED the configuration meets) is not-applicable; one a \
                Direct3D 12 device can run and iridia's device lacks a feature for is not-modelled; XFAIL against \
                iridia's device's features. reference_status is the test's status on the reference path.",
            "totals": totals,
            "reference_totals": reference_totals,
            "reasons": reasons.json()?,
            "rows": rows,
        }));
    }
    // the features
    #[derive(Default)]
    struct Row {
        programs: usize,
        decoded: [usize; 2],
        listed: [usize; 3],
        lifted: [usize; 2],
        wgsl: [usize; 2],
        executed: [usize; 2],
        proved: BTreeSet<&'static str>,
    }
    let mut table: BTreeMap<(&'static str, &'static str, String), Row> = BTreeMap::new();
    for fate in fates.values() {
        let Some(bytecode) = fate.bytecode else { continue };
        let mut keys: Vec<(&'static str, String)> = fate.ops.iter().map(|o| ("opcode", o.clone())).collect();
        if let Some(s) = &fate.stage {
            keys.push(("stage", s.clone()));
        }
        if !fate.decoded {
            keys.push(("construct", "(programs dx does not decode)".into()));
        }
        for (kind, name) in keys {
            let row = table.entry((bytecode.name(), kind, name)).or_default();
            row.programs += 1;
            row.decoded[usize::from(!fate.decoded)] += 1;
            row.listed[match fate.listed {
                Some(true) => 0,
                Some(false) => 1,
                None => 2,
            }] += 1;
            if let Some(l) = fate.lifted {
                row.lifted[usize::from(!l)] += 1;
            }
            if let Some(w) = fate.wgsl {
                row.wgsl[usize::from(!w)] += 1;
            }
            if let Some(e) = fate.executed {
                row.executed[usize::from(!e)] += 1;
            }
            row.proved.extend(fate.passing_suites.iter().copied());
        }
    }
    let features: Vec<Value> = table
        .iter()
        .map(|((format, kind, name), r)| {
            json!({
                "format": format,
                "kind": kind,
                "name": name,
                "programs": r.programs,
                "decoded": if r.decoded[1] == 0 { "yes" } else { "no" },
                "decoded_counts": {"ok": r.decoded[0], "failed": r.decoded[1]},
                "listed": if r.listed[0] + r.listed[1] == 0 { "unchecked" } else if r.listed[1] == 0 { "exact" } else { "differs" },
                "listed_counts": {"exact": r.listed[0], "differs": r.listed[1], "unchecked": r.listed[2]},
                "lifted": {"ok": r.lifted[0], "refused": r.lifted[1]},
                "wgsl": {"ok": r.wgsl[0], "failed": r.wgsl[1]},
                "executed": {"pass": r.executed[0], "fail": r.executed[1]},
                "proved_by": r.proved.iter().collect::<Vec<_>>(),
            })
        })
        .collect();
    let _ = writeln!(
        summary,
        "\n## Constructs\n\nPrograms using each construct: decoded by dx, listed exactly as Microsoft's tools list them, \
         lifted onto iridia's IR and through its passes, written as WGSL that naga reads back, and run on the IR to the          result their tests expect.\n"
    );
    let _ = writeln!(
        summary,
        "| Format | Kind | Construct | Programs | Decoded | Listed exactly | Lifted | WGSL | Executed pass / fail | Proved by |\n\
         |---|---|---|---:|---:|---:|---:|---:|---:|---|"
    );
    for ((format, kind, name), r) in &table {
        let _ = writeln!(
            summary,
            "| {format} | {kind} | `{name}` | {} | {} | {} | {} | {} | {} / {} | {} |",
            r.programs,
            r.decoded[0],
            if r.listed[0] + r.listed[1] == 0 {
                "-".to_string()
            } else {
                format!("{} of {}", r.listed[0], r.listed[0] + r.listed[1])
            },
            r.lifted[0],
            r.wgsl[0],
            r.executed[0],
            r.executed[1],
            r.proved.iter().copied().collect::<Vec<_>>().join(", ")
        );
    }
    let doc = json!({
        "version": VERSION,
        "iridia": inputs.commit,
        "dx": env!("DX_READERS"),
        "graded": graded,
        "groups": GROUPS.iter().map(|(g, m)| json!({"group": g, "meaning": m})).collect::<Vec<_>>(),
        "statuses": STATUSES
            .iter()
            .map(|(s, g, m)| json!({"status": s, "group": g, "meaning": m}))
            .collect::<Vec<_>>(),
        "features": features,
        "suites": suites,
    });
    let md = format!(
        "# iridia and dx coverage\n\nGenerated by `iridia-coverage table` from the suites in `tools/coverage/suites` \
         at iridia {} with dx's {}; do not edit by hand.\n\n{summary}",
        inputs.commit,
        env!("DX_READERS")
    );
    Ok((doc, md))
}

/// Fill a program's decode and listing columns.
fn fate_of(fate: &mut Fate, bytes: &[u8], microsoft: Option<&String>) {
    if fate.bytecode.is_some() {
        if let (Some(theirs), None) = (microsoft, fate.listed) {
            fate.listed = Some(listing(bytes).is_ok_and(|ours| same_listing(&ours, theirs)));
        }
        return;
    }
    let (bytecode, found) = constructs(bytes);
    fate.bytecode = Some(bytecode);
    match found {
        Ok((stage, ops)) => {
            fate.decoded = true;
            fate.stage = Some(stage);
            fate.ops = ops;
        }
        Err(_) => fate.decoded = false,
    }
    if let Some(theirs) = microsoft {
        fate.listed = Some(listing(bytes).is_ok_and(|ours| same_listing(&ours, theirs)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offload::run::Outcome as Offload;
    use crate::vkd3d::run::Outcome as Vkd3d;

    fn part(shard: Shard, files: &[&str], path: front::Path) -> Part {
        let run = |file: &&str| vkd3d::run::RunResult {
            file: (*file).into(),
            runner: "d3d11".into(),
            model: "4.0".into(),
            probes: Vec::new(),
            programs: BTreeMap::new(),
            took: std::time::Duration::ZERO,
        };
        Part {
            cases: "t.json.gz".into(),
            path,
            shard,
            runs: Runs::Vkd3d(files.iter().map(run).collect()),
            lifts: files
                .iter()
                .map(|f| {
                    (
                        (*f).into(),
                        LiftFate {
                            lifted: true,
                            wgsl: None,
                        },
                    )
                })
                .collect(),
        }
    }

    fn files_of(p: &Part) -> Vec<String> {
        let Runs::Vkd3d(r) = &p.runs else {
            panic!("not vkd3d")
        };
        r.iter().map(|r| r.file.clone()).collect()
    }

    /// The shards of a part are put together in the order of the file's cases, a missing shard is refused, and a part of
    /// the whole file is used as it is.
    #[test]
    fn shards_of_a_part_come_back_in_the_order_of_the_file() {
        let name = Path::new("t.json.gz");
        let ir = front::Path::Ir;
        let shard = |index| Shard { index, count: 3 };
        // seven cases a..g: shard 0 has a, d, g; shard 1 b, e; shard 2 c, f
        let parts = vec![
            part(shard(2), &["c", "f"], ir),
            part(shard(0), &["a", "d", "g"], ir),
            part(shard(1), &["b", "e"], ir),
            part(Shard::WHOLE, &["x"], front::Path::Reference),
        ];
        let whole = part_of(&parts, name, ir).unwrap();
        assert_eq!(files_of(&whole), ["a", "b", "c", "d", "e", "f", "g"]);
        assert_eq!(whole.lifts.len(), 7);
        assert!(matches!(
            part_of(&parts, name, front::Path::Reference),
            Ok(Cow::Borrowed(_))
        ));
        // a shard missing, one twice, or one of the wrong length
        assert!(part_of(&parts[..2], name, ir).is_err());
        let twice = vec![parts[0].clone(), parts[0].clone(), parts[1].clone()];
        assert!(part_of(&twice, name, ir).is_err());
        let short = vec![
            part(shard(0), &["a", "d", "g"], ir),
            part(shard(1), &["b"], ir),
            part(shard(2), &["c", "f"], ir),
        ];
        assert!(part_of(&short, name, ir).is_err());
        assert!(part_of(&parts, Path::new("other.json.gz"), ir).is_err());
    }

    /// Every class either runner gives is a status of the table, every status one a runner gives, and every status's
    /// group one the table names.
    #[test]
    fn every_outcome_is_a_status_in_a_group() {
        let m = String::new;
        let vkd3d = [
            Vkd3d::Pass,
            Vkd3d::PassFlushed,
            Vkd3d::Fail(m()),
            Vkd3d::Refused(m(), m()),
            Vkd3d::Failed(m(), m()),
            Vkd3d::Error(m(), m()),
            Vkd3d::NotModelled(m()),
            Vkd3d::CompilerRefused(m()),
        ];
        let offload = [
            Offload::Pass,
            Offload::XFail(m()),
            Offload::Fail(m()),
            Offload::XPass,
            Offload::NotApplicable(m()),
            Offload::Refused(m()),
            Offload::NotModelled(m()),
            Offload::CompilerRefused(m()),
        ];
        let mut given: BTreeSet<&str> = vkd3d.iter().map(|o| vkd3d_status(o.class())).collect();
        given.insert("winbug");
        given.extend(offload.iter().map(Offload::class));
        for status in &given {
            let group = group_of(status).expect("a status of the table");
            assert!(GROUPS.iter().any(|g| g.0 == group), "{status}'s group {group}");
        }
        let named: BTreeSet<&str> = STATUSES.iter().map(|s| s.0).collect();
        assert_eq!(given, named);
    }
}
