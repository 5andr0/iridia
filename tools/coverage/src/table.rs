//! The coverage table: per construct (an opcode or a stage of a bytecode), how many of the suites' programs that use
//! it dx decodes, dx lists exactly as Microsoft's tools do, iridia lifts, writes as WGSL that naga reads back, and runs
//! to the result the suite expects; and per test, what it came to. Everything here is measured from the case files and
//! the runs: nothing is written by hand.
//!
//! A program counts as executed `pass` for a construct when some run (vkd3d) or test (offload) using it passes, and
//! `fail` when every run using it that got as far as running failed.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::front::{Bytecode, Ready, Stop};
use crate::inflate::read_text;
use crate::offload;
use crate::vkd3d;

/// The table's schema version.
pub const VERSION: u32 = 1;

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

fn note_lift(fate: &mut Fate, lifted: Option<&Result<Arc<Ready>, Stop>>) {
    if let Some(l) = lifted {
        fate.lifted = Some(l.is_ok());
        if let Ok(r) = l {
            fate.wgsl = Some(r.lifted.wgsl.is_ok());
        }
    }
}

/// A suite's status vocabulary for the table.
fn vkd3d_status(class: &str) -> &'static str {
    match class {
        "pass" => "pass",
        "fail" | "failed" => "fail",
        "refused" => "refused",
        "error" => "error",
        "not modelled" => "not-modelled",
        "compiler refused" => "compiler-refused",
        "winbug" => "winbug",
        _ => "other",
    }
}

/// The inputs the table is made of.
#[derive(Debug)]
pub struct Inputs<'a> {
    /// The vkd3d case files (d3d9, dxbc, dxil), each with its listings file beside it.
    pub vkd3d: Vec<&'a Path>,
    /// The offload case file.
    pub offload: Option<&'a Path>,
    /// A checkout of offload-golden-images.
    pub golden: Option<&'a Path>,
    /// iridia's commit.
    pub commit: String,
    /// Threads to run on.
    pub threads: usize,
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

/// Run the suites and make the table: the JSON the demo site reads and a Markdown summary.
pub fn make(inputs: &Inputs<'_>) -> Result<(Value, String), String> {
    let mut fates: BTreeMap<String, Fate> = BTreeMap::new();
    let mut suites = Vec::new();
    let mut summary = String::new();

    // vkd3d
    let mut rows = Vec::new();
    // the runs vkd3d's runner runs and their selected probes; and apart, the runs it runs none of and the extended
    // probes of the others
    let mut totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut probe_totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut extended_totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut extended_probe_totals: BTreeMap<String, usize> = BTreeMap::new();
    // the probes vkd3d marks `winbug`: those apart by what they found, and the driver bugs' count
    let mut winbug_outcomes: BTreeMap<String, usize> = BTreeMap::new();
    let mut driver_bugs = 0;
    let mut source = String::new();
    for path in &inputs.vkd3d {
        let file = vkd3d::cases::read(&read_text(path)?)?;
        source.push_str(&file.source);
        source.push_str("; ");
        let listings = listings_of(path);
        let (results, lifted) = vkd3d::run::run_all(&file, inputs.threads);
        for name in file.programs.keys() {
            let bytes = vkd3d::cases::program(&file, name)?;
            let fate = fates.entry(name.clone()).or_default();
            fate.suites.insert("vkd3d");
            fate_of(fate, &bytes, listings.get(name));
            note_lift(fate, lifted.get(name));
        }
        for r in &results {
            let class = vkd3d_status(r.class());
            let selected = r.selected();
            let runs = if selected {
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
            let format = r
                .programs
                .keys()
                .next()
                .and_then(|n| fates.get(n))
                .and_then(|f| f.bytecode)
                .map_or("", Bytecode::name);
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
                "id": format!("{}:{}:{}", r.file, r.runner, r.model),
                "case": r.file,
                "runner": r.runner,
                "format": format,
                "model": r.model,
                "status": class,
                "detail": detail,
                "probes": r.probes.len(),
                "selected": selected,
                "selected_probes": r.probes.iter().filter(|p| p.selected).count(),
                "extended_status": r.extended_class().map(vkd3d_status),
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
         and the probes of the others it leaves out (the same expectations at a model vkd3d's own runs leave out).\n"
    );
    let _ = writeln!(
        summary,
        "| Status | Runs | Probes | Extended runs | Extended probes |\n|---|---:|---:|---:|---:|"
    );
    let statuses: BTreeSet<&String> = totals
        .keys()
        .chain(probe_totals.keys())
        .chain(extended_totals.keys())
        .chain(extended_probe_totals.keys())
        .collect();
    let n = |m: &BTreeMap<String, usize>, s: &String| m.get(s).copied().unwrap_or(0);
    for s in &statuses {
        let _ = writeln!(
            summary,
            "| {s} | {} | {} | {} | {} |",
            n(&totals, s),
            n(&probe_totals, s),
            n(&extended_totals, s),
            n(&extended_probe_totals, s)
        );
    }
    let found: Vec<String> = winbug_outcomes.iter().map(|(s, n)| format!("{s} {n}")).collect();
    let _ = writeln!(
        summary,
        "\nThe winbug probes are those vkd3d's tests mark `winbug` where Windows itself reads otherwise (a plain \
         `winbug`, or one of models or of devices it does not hold on): they decide no run, and found {}. The \
         {driver_bugs} marked `winbug` for a driver's bug (`amd`, `nvidia`, `warp=1.0.16`) are probes like any other.\n",
        if found.is_empty() {
            "nothing".to_string()
        } else {
            found.join(", ")
        }
    );
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
            (driver_bug_probes) is a probe like any other.",
        "totals": totals,
        "probe_totals": probe_totals,
        "extended_totals": extended_totals,
        "extended_probe_totals": extended_probe_totals,
        "winbug_outcomes": winbug_outcomes,
        "driver_bug_probes": driver_bugs,
        "rows": rows,
    }));

    // offload
    if let Some(path) = inputs.offload {
        let file = offload::cases::read(&read_text(path)?)?;
        let programs = crate::front::Programs::new(|name: &str| {
            let text = file
                .programs
                .get(name)
                .ok_or_else(|| format!("no program {name}"))?;
            offload::pipeline::base64(text)
        });
        let results = offload::run::run_all_with(&file, inputs.threads, inputs.golden, &programs);
        let lifted = programs.lifted();
        for name in file.programs.keys() {
            let bytes = programs.bytes(name)?;
            let fate = fates.entry(name.clone()).or_default();
            fate.suites.insert("offload");
            fate_of(fate, &bytes, None);
            note_lift(fate, lifted.get(name));
        }
        let mut rows = Vec::new();
        let mut totals: BTreeMap<String, usize> = BTreeMap::new();
        for r in &results {
            let class = r.outcome.class();
            *totals.entry(class.to_string()).or_default() += 1;
            let format = r
                .programs
                .first()
                .and_then(|n| fates.get(n))
                .and_then(|f| f.bytecode)
                .map_or("", Bytecode::name);
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
                "detail": r.outcome.detail(),
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
        let _ = writeln!(summary, "\n## LLVM offload-test-suite\n\n{}\n", file.source);
        let _ = writeln!(summary, "| Status | Tests |\n|---|---:|");
        for (s, n) in &totals {
            let _ = writeln!(summary, "| {s} | {n} |");
        }
        suites.push(json!({
            "id": "offload",
            "name": "LLVM offload-test-suite",
            "source": "https://github.com/llvm/offload-test-suite",
            "commit": file.source,
            "licence": "Apache-2.0 WITH LLVM-exception",
            "description": "Each test run as lit runs it for DirectX with DXC: its RUN lines in order, the offloader's \
                pipeline on iridia's executor, its Results rules, FileCheck lines and golden images checked as the \
                suite's own tools check them; REQUIRES, UNSUPPORTED and XFAIL against the executor's features.",
            "totals": totals,
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
         lifted by iridia, written as WGSL that naga reads back, and run to the result their tests expect.\n"
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
        "dx": "d3d9-reader, dxbc-reader, dxil-reader 0.5.0",
        "features": features,
        "suites": suites,
    });
    let md = format!(
        "# iridia and dx coverage\n\nGenerated by `iridia-coverage table` from the suites in `tools/coverage/suites` \
         at iridia {}; do not edit by hand.\n\n{summary}",
        inputs.commit
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
