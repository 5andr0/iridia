#![allow(clippy::unwrap_used)]
//! The analyses on the IR, on every program of the fixtures, the slim set and Elden Ring's corpus: what may reach each
//! output lane and the discard ([`Flow`]) and the constant words read ([`WordsRead`]) of each DXBC, DXIL and Direct3D 9
//! program, held to tests/expected/<set>.tsv (each program's answers as a hash, or what refused it, and the judgement),
//! and the fixtures' vertex and pixel programs, each vertex program with each pixel program of its family, held to
//! tests/expected/demand.tsv for their [`VertexDemand`]. Resources are named by their D3D bind point.
//!
//! The IR's answers were judged against the older analyses (the `iridia` crate's) while those ran beside them: where
//! the answers differed, each source the IR's answers leave out of a lane was changed in runs of the IR program in
//! `iridia-exec-ir`, and one that changed its lane would have made them unsound; each source they add was changed too,
//! and one that changed its lane is one the older answers missed. tests/expected/judged.tsv records, for each program the
//! two read otherwise, the kinds of difference and those sources; the test runs them again, so the judgement of
//! each is as it was, and an answer that changes, or a source that now changes its lane, fails.
//!
//! The IR's answers are held to tests/expected/<set>.tsv: an answer or a judgement that changes fails, a new
//! difference among them, and the current table is written to the scratch root (the path is in the failure) to replace
//! the committed one where the change is meant. A program the IR's analyses refuse and the table does not say refused is a
//! failure; a lift that refuses is the lifter's, counted. Each program's value graph is also evaluated
//! ([`iridia_exec_ir::value`]) from the inputs and constant words of a run, and an output lane it gives otherwise than
//! the run fails.
mod inputs;
mod perturb;

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

use inputs::{Game, OracleSet, Query, expect_found, oracle_tree};
use iridia_analysis::flow::{Flow, Values, VertexDemand};
use iridia_analysis::reads::WordsRead;
use iridia_ir::interface::{Interface, ResourceId, Stage};
use primase_testdata::Tree;

/// Whether a container holds a part of this kind (`DXIL`): its part table read as `dxcontainer` lays it out (the
/// count at byte 28, then each part's offset; at each, the part's kind).
fn has_part(bytes: &[u8], kind: &[u8; 4]) -> bool {
    let word = |at: usize| -> Option<usize> {
        let b = bytes.get(at..at.checked_add(4)?)?;
        usize::try_from(u32::from_le_bytes(b.try_into().ok()?)).ok()
    };
    let Some(count) = word(28) else {
        return false;
    };
    (0..count.min(64)).any(|k| {
        word(32 + 4 * k)
            .and_then(|at| bytes.get(at..at.checked_add(4)?))
            .is_some_and(|k| k == kind)
    })
}

fn lift_ir(bytes: &[u8]) -> Result<iridia_ir::Program, String> {
    let text = |e: iridia::Refusal| e.to_string();
    if !bytes.starts_with(b"DXBC") {
        iridia_d3d9_ir::lift(bytes).map_err(text)
    } else if has_part(bytes, b"DXIL") {
        iridia_dxil::ir::lift(bytes).map_err(text)
    } else {
        iridia_dxbc::ir::lift(bytes).map_err(text)
    }
}

/// A resource as the IR lift names it: its kind and D3D bind point (its id where it has none).
fn resource(interface: &Interface, id: ResourceId) -> String {
    match interface.resources.iter().find(|r| r.id == id) {
        Some(r) => {
            let kind = match r.kind {
                iridia_ir::interface::ResourceKind::ConstantBuffer { .. } => "ConstantBuffer",
                iridia_ir::interface::ResourceKind::Texture { .. } => "Texture",
                iridia_ir::interface::ResourceKind::Sampler { .. } => "Sampler",
                iridia_ir::interface::ResourceKind::Buffer(_) => "Buffer",
                iridia_ir::interface::ResourceKind::Storage { .. } => "Storage",
                iridia_ir::interface::ResourceKind::AccelerationStructure => "AccelerationStructure",
                iridia_ir::interface::ResourceKind::PipelineState(_) => "PipelineState",
            };
            match r.provenance.d3d {
                Some(d) => format!("{kind} {d:?}"),
                None => format!("{kind} #{}", id.0),
            }
        }
        None => format!("unknown #{}", id.0),
    }
}

/// What may reach one output lane or the discard, by names the interface gives.
#[derive(Debug, Default, PartialEq, Eq)]
struct Reached {
    inputs: BTreeMap<(usize, u8), String>,
    texels: BTreeMap<(String, u8), String>,
    words: BTreeMap<(String, u32), String>,
}

impl Reached {
    fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.texels.is_empty() && self.words.is_empty()
    }
}

/// A program's answers: each output lane's reach (a lane nothing reaches left out), the discard's, and the constant
/// words read.
#[derive(Debug, Default, PartialEq, Eq)]
struct Answers {
    outputs: BTreeMap<(usize, u8), Reached>,
    discard: Reached,
    words: BTreeMap<String, BTreeSet<u32>>,
    indexed: BTreeSet<String>,
}

/// A reach of [`Flow`] as [`Reached`].
macro_rules! reached {
    ($reach:expr, $interface:expr) => {{
        let (r, interface) = ($reach, $interface);
        Reached {
            inputs: r
                .inputs
                .iter()
                .map(|(l, v)| ((l.input, l.lane), format!("{v:?}")))
                .collect(),
            texels: r
                .texels
                .iter()
                .map(|(t, v)| ((resource(interface, t.resource), t.lane), format!("{v:?}")))
                .collect(),
            words: r
                .words
                .iter()
                .map(|(w, v)| ((resource(interface, w.resource), w.word), format!("{v:?}")))
                .collect(),
        }
    }};
}

/// A flow and constant reads as [`Answers`].
macro_rules! answers {
    ($flow:expr, $reads:expr, $interface:expr) => {{
        let (flow, reads, interface): (_, _, &Interface) = ($flow, $reads, $interface);
        let mut out = Answers {
            discard: reached!(flow.discard_reach(), interface),
            ..Answers::default()
        };
        for (key, reach) in flow.outputs() {
            let r = reached!(reach, interface);
            if !r.is_empty() {
                out.outputs.insert(*key, r);
            }
        }
        for r in &interface.resources {
            let words: BTreeSet<u32> = reads.words(r.id).collect();
            if !words.is_empty() {
                out.words.insert(resource(interface, r.id), words);
            }
            if reads.indexed(r.id) {
                out.indexed.insert(resource(interface, r.id));
            }
        }
        out
    }};
}

fn ir_answers(program: &iridia_ir::Program) -> Result<Answers, String> {
    let started = Instant::now();
    let values = Values::of_program(program)?;
    if std::env::var_os("ORACLE_ONE").is_some() {
        eprintln!("IR Values: {:?}", started.elapsed());
    }
    let flow = Flow::of_values(&values);
    let reads = WordsRead::of_values(&values, &program.interface);
    Ok(answers!(&flow, &reads, &program.interface))
}

/// A panic's message.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".into())
}

/// `f`, a panic in it as an error.
fn caught<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| Err(format!("panic: {}", panic_message(&*p))))
}

/// A stable 64-bit hash of text (FNV-1a), for the recorded answers.
fn fnv(text: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// How one program's analyses are judged.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Verdict {
    /// The older analyses read it alike (the table records no other judgement).
    Agree,
    /// A lift refused it: not compared.
    NotLifted(String),
    /// Neither the IR's analyses nor the older ones read it.
    BothRefuse,
    /// The older analyses answered otherwise than by reading it, as the table says (a refusal of theirs, or of their lift).
    Recorded(String),
    /// Both read it and the answers differed: how (each kind of difference, by place) and what the executor's runs showed
    /// of the sources the IR's leave out and add. Recorded with the answers, not a failure itself.
    Differ { kinds: String, runs: String },
    /// The IR's analyses leave out a source the executor's runs show reaching its lane: a failure.
    Unsound(String),
    /// The IR's analyses panic: a failure.
    Panic(String),
    /// The value graph's evaluator gives an output lane otherwise than a run of the program: a failure.
    EvaluatorDiffers(String),
    /// The IR's analyses refuse a program the table does not record as refused: a failure.
    IrRefuses(String),
    /// Execution evidence stopped at the measured corpus budget; never counted as an agreement.
    BudgetExhausted(u64, String),
}

impl Verdict {
    fn fails(&self) -> bool {
        matches!(
            self,
            Verdict::IrRefuses(_) | Verdict::Unsound(_) | Verdict::Panic(_) | Verdict::EvaluatorDiffers(_)
        )
    }

    /// The verdict as a report counts it: its kind and reason.
    fn group(&self) -> String {
        match self {
            Verdict::Agree => "agree".into(),
            Verdict::NotLifted(why) => format!("not lifted: {why}"),
            Verdict::BothRefuse => "both refuse".into(),
            Verdict::Recorded(group) => group.clone(),
            Verdict::IrRefuses(why) => format!("IR refuses: {why}"),
            Verdict::Differ { kinds, runs } => format!("differ: {kinds}; {runs}"),
            Verdict::Unsound(_) => "IR unsound".into(),
            Verdict::Panic(why) => format!("IR panics: {why}"),
            Verdict::EvaluatorDiffers(_) => "the evaluator differs from a run".into(),
            Verdict::BudgetExhausted(budget, analysis) => {
                format!("budget exhausted at {budget}; {analysis}")
            }
        }
    }
}

/// How many pairs of runs the executor takes for each source the IR's analyses leave out.
const RUNS: u64 = 6;

/// What the older analyses said of one program (a line of tests/expected/judged.tsv).
enum Judged {
    /// The answers differed: the kinds of difference, and the (target, source) pairs the IR's answers drop and add.
    Differ {
        kinds: String,
        dropped: Vec<(perturb::Target, perturb::Source)>,
        added: Vec<(perturb::Target, perturb::Source)>,
    },
    /// The older analyses answered otherwise than by reading it: the verdict's report line.
    Other(String),
}

fn parse_target(text: &str) -> perturb::Target {
    if text == "d" {
        return perturb::Target::Discard;
    }
    let (slot, lane) = text.strip_prefix('o').unwrap().split_once('.').unwrap();
    perturb::Target::Output(slot.parse().unwrap(), lane.parse().unwrap())
}

fn parse_source(text: &str) -> perturb::Source {
    let (kind, rest) = text.split_at(1);
    let (a, b) = rest.split_once('.').unwrap();
    match kind {
        "i" => perturb::Source::Input(a.parse().unwrap(), b.parse().unwrap()),
        "w" => perturb::Source::Word(ResourceId(a.parse().unwrap()), b.parse().unwrap()),
        "t" => perturb::Source::Texel(ResourceId(a.parse().unwrap()), b.parse().unwrap()),
        _ => panic!("a source: {text}"),
    }
}

fn parse_pairs(text: &str) -> Vec<(perturb::Target, perturb::Source)> {
    text.split(';')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (target, source) = p.split_once("<-").unwrap();
            (parse_target(target), parse_source(source))
        })
        .collect()
}

/// A program's key in tests/expected/judged.tsv: FNV-1a of its bytes, then its length.
fn key(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h = (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}{:08x}", bytes.len())
}

/// The judgements recorded in tests/expected/judged.tsv, by program key.
fn judged() -> &'static BTreeMap<String, Judged> {
    static JUDGED: std::sync::OnceLock<BTreeMap<String, Judged>> = std::sync::OnceLock::new();
    JUDGED.get_or_init(|| {
        include_str!("expected/judged.tsv")
            .replace("\r\n", "\n")
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .map(|line| {
                let mut fields = line.split('\t');
                let key = fields.next().unwrap().to_owned();
                let what = fields.next().unwrap();
                let judged = match what.split_at(1) {
                    ("~", kinds) => Judged::Differ {
                        kinds: kinds.to_owned(),
                        dropped: parse_pairs(fields.next().unwrap_or_default()),
                        added: parse_pairs(fields.next().unwrap_or_default()),
                    },
                    ("=", group) => Judged::Other(group.to_owned()),
                    _ => panic!("a judgement: {line}"),
                };
                (key, judged)
            })
            .collect()
    })
}

/// The recorded difference judged again: each source the IR's analyses leave out of a lane's reach run in the executor
/// changed ([`perturb::check`]); one that changes its lane makes the IR's answers unsound. The sources they add are run
/// too, to say whether the older answers missed them.
fn judge(
    kinds: &str,
    dropped: &[(perturb::Target, perturb::Source)],
    added: &[(perturb::Target, perturb::Source)],
    program: &iridia_ir::Program,
) -> Verdict {
    let mut kept = 0;
    let mut not_run = BTreeSet::new();
    for (target, source) in dropped {
        match perturb::check(program, *target, *source, RUNS) {
            perturb::Outcome::Kept(n) => kept += n,
            perturb::Outcome::NotRun(why) => {
                not_run.insert(grouped(&why));
            }
            perturb::Outcome::Changed(what) => return Verdict::Unsound(what),
            perturb::Outcome::BudgetExhausted(budget) => {
                return Verdict::BudgetExhausted(budget, "perturbing a dropped source".into());
            }
        }
    }
    let mut runs = match (dropped.len(), not_run.is_empty()) {
        (0, _) => "nothing dropped".to_owned(),
        (_, true) => "each dropped source kept its lane in every run".to_owned(),
        _ => format!(
            "dropped sources not run ({}){}",
            not_run.into_iter().collect::<Vec<_>>().join("; "),
            if kept > 0 { ", the others kept" } else { "" }
        ),
    };
    // a source the IR's answers add that changes its lane in a run is one the older answers miss
    if !added.is_empty() {
        let mut changes = 0;
        let mut completed = 0;
        let mut not_run = BTreeSet::new();
        for (target, source) in added {
            match perturb::check(program, *target, *source, RUNS) {
                perturb::Outcome::Changed(_) => changes += 1,
                perturb::Outcome::Kept(_) => completed += 1,
                perturb::Outcome::NotRun(why) => {
                    not_run.insert(grouped(&why));
                }
                perturb::Outcome::BudgetExhausted(budget) => {
                    return Verdict::BudgetExhausted(budget, "perturbing an added source".into());
                }
            }
        }
        runs.push_str(&match changes {
            0 if completed == 0 => format!(
                " (added sources not run: {})",
                not_run.iter().cloned().collect::<Vec<_>>().join("; ")
            ),
            0 => " (no added source changed its lane in the completed runs)".to_owned(),
            n if n == added.len() => {
                " (every added source changed its lane in a run: the older answers miss them)".to_owned()
            }
            _ => " (some added sources changed their lane in a run: the older answers miss them)".to_owned(),
        });
    }
    Verdict::Differ {
        kinds: kinds.to_owned(),
        runs,
    }
}

/// One program's verdict, and the line its IR answers are recorded by.
fn compare(bytes: &[u8]) -> (Verdict, String) {
    let (mut verdict, mut line) = compare_inner(bytes);
    let counts = perturb::take_counts();
    for reason in perturb::take_exhaustions() {
        line.push_str(&format!("\texecution cause: {reason}"));
    }
    if counts.exhausted > 0 && !verdict.fails() && !matches!(verdict, Verdict::BudgetExhausted(..)) {
        let analysis = if matches!(verdict, Verdict::Agree) {
            "analysis answers identical".into()
        } else {
            verdict.group()
        };
        verdict = Verdict::BudgetExhausted(counts.budget, analysis);
    }
    // Every row names its actual comparison, including legacy refusals other than OOM. A bare answer hash must
    // not make a newly supported atomic or a legacy indexing refusal look like an old/new agreement.
    line.push_str(&format!("\t{}", verdict.group()));
    line.push_str(&format!(
        "\tsteps completed={}, max={}, exhausted={}",
        counts.completed, counts.maximum, counts.exhausted
    ));
    if counts.budget > perturb::RUN_BUDGET {
        line.push_str(&format!(
            "\texecution budget={}, class=nested compute loops, {} runs measured at least {} steps; completed maximum={} (refusals return no exact count)",
            counts.budget, counts.enlarged, perturb::RUN_BUDGET + 1, counts.maximum
        ));
    }
    if counts.input_ambiguous {
        line.push_str("\tinput type ambiguous: conflicting inferred scalar views; original fill retained");
    }
    (verdict, line)
}

fn compare_inner(bytes: &[u8]) -> (Verdict, String) {
    let new = match lift_ir(bytes) {
        Ok(p) => p,
        Err(e) => {
            return (
                Verdict::NotLifted(format!("IR lift: {}", grouped(&e))),
                "not lifted".into(),
            );
        }
    };
    let graph = caught(|| {
        let values = Values::of_program(&new)?;
        let flow = Flow::of_values(&values);
        let reads = WordsRead::of_values(&values, &new.interface);
        Ok((answers!(&flow, &reads, &new.interface), values))
    });
    let mut recorded = match &graph {
        Ok((a, _)) => fnv(&format!("{a:?}")),
        Err(e) => format!("refused: {}", grouped(e)),
    };
    let evaluator = graph
        .as_ref()
        .ok()
        .map(|(_, values)| perturb::evaluations(&new, values));
    let execution_note = evaluator.as_ref().and_then(|e| e.not_run.clone());
    let ir = graph.map(|(answers, _)| answers);
    let verdict = match (judged().get(&key(bytes)), ir) {
        (_, Err(e)) if e.starts_with("panic: ") => Verdict::Panic(e),
        _ if let Some(e) = evaluator.filter(|e| !e.differ.is_empty()) => {
            Verdict::EvaluatorDiffers(e.differ.join("; "))
        }
        (Some(Judged::Other(group)), _) => Verdict::Recorded(group.clone()),
        (
            Some(Judged::Differ {
                kinds,
                dropped,
                added,
            }),
            Ok(_),
        ) => judge(kinds, dropped, added, &new),
        (Some(Judged::Differ { .. }), Err(e)) => Verdict::IrRefuses(grouped(&e)),
        (None, Ok(_)) => Verdict::Agree,
        (None, Err(_)) => Verdict::BothRefuse,
    };
    if let Some(why) = execution_note {
        recorded.push_str(&format!("\texecution not compared: {why}"));
    }
    (verdict, recorded)
}

/// A reason with each run of digits as `N` and on one line, as a report counts reasons.
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
            out.push(if c == '\t' || c == '\n' { ' ' } else { c });
            digits = false;
        }
    }
    out
}

/// The programs of a set to compare: a tree and the names in it, or files.
enum Source {
    /// The names in a tree; with `true`, only those that are DXBC containers are programs.
    Tree(Tree, Vec<String>, bool),
    Files(Vec<(String, std::path::PathBuf)>),
}

impl Source {
    fn names(&self) -> Vec<String> {
        match self {
            Source::Tree(_, names, _) => names.clone(),
            Source::Files(files) => files.iter().map(|(n, _)| n.clone()).collect(),
        }
    }
}

/// Every program of `source` compared on as many threads as the inputs allow, held to tests/expected/`set`.tsv.
fn hold(set: &str, source: Source) {
    let names = source.names();
    // a folder's files are read one at a time: files of the NAS's SMB share opened at once on several threads fail at
    // random with Windows error 1450 (ERROR_NO_SYSTEM_RESOURCES); an archive is read at positions of one open file
    let folder_reads = match &source {
        Source::Tree(tree, _, _) => tree.as_folder().map(|_| Mutex::new(())),
        Source::Files(_) => None,
    };
    let read = |i: usize| -> Vec<u8> {
        match &source {
            Source::Tree(tree, names, _) => {
                let _one_at_a_time = folder_reads.as_ref().map(|m| m.lock().unwrap());
                tree.read(&names[i]).unwrap()
            }
            Source::Files(files) => std::fs::read(&files[i].1).unwrap(),
        }
    };
    let containers_only = matches!(source, Source::Tree(_, _, true));
    let started = Instant::now();
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<(Verdict, String)>>> = names.iter().map(|_| Mutex::default()).collect();
    // each worker's program in hand and since when, for the report of one that takes long
    let workers = inputs::threads().get();
    let in_hand: Vec<Mutex<Option<(usize, Instant)>>> = (0..workers).map(|_| Mutex::default()).collect();
    let done = std::sync::atomic::AtomicBool::new(false);
    let workers_done = AtomicUsize::new(0);
    std::thread::scope(|s| {
        s.spawn(|| {
            let mut told = BTreeSet::new();
            while !done.load(Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(500));
                for slot in &in_hand {
                    if let Some((i, since)) = *slot.lock().unwrap()
                        && since.elapsed().as_secs() >= 60
                        && told.insert(i)
                    {
                        println!("  {set}: {} still in hand after 60 s", names[i]);
                    }
                }
            }
        });
        for slot in &in_hand {
            let (names, next, results, done, workers_done, read) =
                (&names, &next, &results, &done, &workers_done, &read);
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= names.len() {
                        if workers_done.fetch_add(1, Relaxed) + 1 == workers {
                            done.store(true, Relaxed);
                        }
                        break;
                    }
                    *slot.lock().unwrap() = Some((i, Instant::now()));
                    let bytes = read(i);
                    if containers_only && !bytes.starts_with(b"DXBC") {
                        continue;
                    }
                    let outcome = catch_unwind(AssertUnwindSafe(|| compare(&bytes))).unwrap_or_else(|p| {
                        let why = format!("panic: {}", panic_message(&*p));
                        (Verdict::Panic(why.clone()), why)
                    });
                    *results[i].lock().unwrap() = Some(outcome);
                    *slot.lock().unwrap() = None;
                }
            });
        }
    });
    let outcomes: Vec<(&String, Verdict, String)> = names
        .iter()
        .zip(results)
        .filter_map(|(n, r)| {
            let (v, line) = r.into_inner().unwrap()?;
            Some((n, v, line))
        })
        .collect();

    // the report: each verdict's count with an example
    println!(
        "{set}: {} programs, {} thread(s), {:.1} s",
        outcomes.len(),
        inputs::threads(),
        started.elapsed().as_secs_f64()
    );
    let mut by: BTreeMap<String, (usize, &str)> = BTreeMap::new();
    for (name, verdict, _) in &outcomes {
        by.entry(verdict.group()).or_insert((0, name)).0 += 1;
    }
    let mut counts: Vec<_> = by.into_iter().collect();
    counts.sort_by(|x, y| y.1.0.cmp(&x.1.0).then(x.0.cmp(&y.0)));
    for (group, (n, example)) in &counts {
        println!("  {n:5}  {group}  (e.g. {example})");
    }

    let failures: Vec<String> = outcomes
        .iter()
        .filter(|(_, v, _)| v.fails())
        .map(|(name, v, _)| match v {
            Verdict::Unsound(at) => format!("  {name}: IR unsound: {at}"),
            Verdict::EvaluatorDiffers(at) => format!("  {name}: the evaluator differs from a run: {at}"),
            other => format!("  {name}: {}", other.group()),
        })
        .collect();

    let mut current = format!(
        "# Each {set} program's answers of the analyses on the IR (tests/oracle.rs): program, then a hash of what reaches \
         each output lane and the discard and the constant words read, or why it has none.\n"
    );
    for (name, _, line) in &outcomes {
        current.push_str(&format!("{name}\t{line}\n"));
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/expected/{set}.tsv"));
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .replace("\r\n", "\n");
    let changed = differences(&committed, &current);
    if failures.is_empty() && changed.is_empty() {
        return;
    }
    let written = inputs::scratch().join(format!("analyses-{set}.tsv"));
    std::fs::write(&written, &current).unwrap();
    let mut why = Vec::new();
    if !failures.is_empty() {
        why.push(format!(
            "{} programs' analyses on the IR fail:\n{}",
            failures.len(),
            failures.iter().take(30).cloned().collect::<Vec<_>>().join("\n")
        ));
    }
    if !changed.is_empty() {
        why.push(format!(
            "{} programs' answers are not the recorded ones ({}):\n{}",
            changed.len(),
            path.display(),
            changed.iter().take(30).cloned().collect::<Vec<_>>().join("\n")
        ));
    }
    panic!(
        "{}\nthe current table is {} (it replaces the committed one where a change is meant)",
        why.join("\n"),
        written.display()
    );
}

/// The lines in which two tables differ, each as `program: the table's -> now`.
fn differences(committed: &str, current: &str) -> Vec<String> {
    let parse = |t: &str| -> BTreeMap<String, String> {
        t.lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .filter_map(|l| l.split_once('\t'))
            .map(|(a, b)| (a.to_owned(), b.to_owned()))
            .collect()
    };
    let (was, now) = (parse(committed), parse(current));
    let keys: BTreeSet<&String> = was.keys().chain(now.keys()).collect();
    keys.into_iter()
        .filter_map(|k| {
            let (a, b) = (
                was.get(k).map_or("absent", String::as_str),
                now.get(k).map_or("absent", String::as_str),
            );
            (a != b).then(|| format!("  {k}: {a} -> {b}"))
        })
        .collect()
}

/// The containers of the fixtures of the `iridia`, `iridia-dxbc` and `iridia-dxil` crates, named by crate and file.
fn fixture_files() -> Vec<(String, std::path::PathBuf)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    for krate in ["iridia", "iridia-dxbc", "iridia-dxil"] {
        let folder = root.join(krate).join("tests/fixtures");
        let mut found: Vec<_> = std::fs::read_dir(&folder)
            .unwrap_or_else(|e| panic!("{}: {e}", folder.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_file() && std::fs::read(p).is_ok_and(|b| b.starts_with(b"DXBC")))
            .collect();
        found.sort();
        for p in found {
            let name = format!("{krate}/{}", p.file_name().unwrap().to_string_lossy());
            files.push((name, p));
        }
    }
    files
}

#[test]
fn fixtures() {
    let files = fixture_files();
    assert!(files.len() >= 150, "{} fixture containers", files.len());
    hold("fixtures", Source::Files(files));
}

/// The slim set's programs of `family` (`programs/<family>/<sha256>.<family>`); `None` where it is not configured.
fn slim(family: &str) -> Option<Source> {
    let tree = oracle_tree(OracleSet::Corpus, Game::General, "slim")?;
    let folder = format!("programs/{family}/");
    let extension = format!(".{family}");
    let mut names: Vec<String> = tree
        .names()
        .into_iter()
        .filter(|n| {
            n.strip_prefix(&folder)
                .is_some_and(|f| !f.contains('/') && f.ends_with(&extension))
        })
        .map(str::to_owned)
        .collect();
    names.sort();
    expect_found(
        Query::Oracle(OracleSet::Corpus, Game::General, "slim"),
        &format!("{family} programs"),
        names.len(),
    );
    Some(Source::Tree(tree, names, false))
}

#[test]
fn slim_dxbc() {
    if let Some(source) = slim("dxbc") {
        hold("slim-dxbc", source);
    }
}

#[test]
fn slim_dxil() {
    if let Some(source) = slim("dxil") {
        hold("slim-dxil", source);
    }
}

#[test]
fn slim_d3d9() {
    if let Some(source) = slim("d3d9") {
        hold("slim-d3d9", source);
    }
}

/// Elden Ring's DXIL containers (`oracles.corpus.elden/shaders`).
#[test]
fn elden() {
    let Some(tree) = oracle_tree(OracleSet::Corpus, Game::Elden, "shaders") else {
        return;
    };
    let mut names: Vec<String> = tree.names().into_iter().map(str::to_owned).collect();
    names.sort_by(|a, b| a.split('/').cmp(b.split('/')));
    expect_found(
        Query::Oracle(OracleSet::Corpus, Game::Elden, "shaders"),
        "files",
        names.len(),
    );
    hold("elden", Source::Tree(tree, names, true));
}

/// The real program behind the exponential loop-walk failure of the older analyses: the IR's complete.
#[test]
fn scene_bokeh_completes() {
    let Some(tree) = oracle_tree(OracleSet::Corpus, Game::Elden, "shaders") else {
        return;
    };
    let bytes = tree.read("gxposteffect/SceneBokeh_SimpleBlurFadePS.ppo").unwrap();
    let program = lift_ir(&bytes).unwrap();
    assert!(ir_answers(&program).is_ok());
}

/// Tint's deliberate two-word countdown has a second input that exits immediately and can be compared exactly.
#[test]
fn tint_countdown_early_exit() {
    let Some(tree) = oracle_tree(OracleSet::Corpus, Game::General, "slim") else {
        return;
    };
    let bytes = tree
        .read("programs/dxbc/f61b7e97b8b31c2f98e733d19b03fbd305b103d7988a833513714c295cc262cd.dxbc")
        .unwrap();
    let program = lift_ir(&bytes).unwrap();
    let values = Values::of_program(&program).unwrap();
    let result = perturb::evaluations_with_input(&program, &values, 0, 0, 0);
    assert!(result.not_run.is_none(), "{result:?}");
    assert!(result.differ.is_empty(), "{result:?}");
    assert!(result.agree > 0, "{result:?}");
    let counts = perturb::take_counts();
    assert_eq!(counts.exhausted, 0);
    eprintln!(
        "Tint input.x=0: {} lanes compared, {} steps",
        result.agree, counts.maximum
    );
}

/// Each vertex fixture with each pixel fixture of its family: the vertex input lanes the pair needs
/// (or why it has none), held to tests/expected/demand.tsv.
#[test]
fn vertex_demand() {
    let mut lifted: Vec<(String, bool, iridia_ir::Program)> = Vec::new();
    for (name, path) in fixture_files() {
        let bytes = std::fs::read(&path).unwrap();
        if let Ok(new) = lift_ir(&bytes) {
            lifted.push((name, has_part(&bytes, b"DXIL"), new));
        }
    }
    let of = |stage| lifted.iter().filter(move |(_, _, p)| p.stage == stage);
    let mut pairs = 0;
    let mut current =
        "# Each vertex fixture with each pixel fixture of its family: the vertex input lanes the pair needs on the IR \
         (tests/oracle.rs), or why it has none.\n"
            .to_owned();
    for (vn, vdxil, vnew) in of(Stage::Vertex) {
        for (pn, _, pnew) in of(Stage::Pixel).filter(|p| p.1 == *vdxil) {
            pairs += 1;
            let new = caught(|| VertexDemand::of_programs(vnew, pnew).map_err(|e| e.to_string()))
                .map(|d| (d.vertex, d.pixel));
            current.push_str(&format!("{vn} {pn}\t{new:?}\n"));
        }
    }
    println!("vertex demand: {pairs} pairs");
    assert!(pairs >= 50, "{pairs} pairs");
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/expected/demand.tsv");
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .replace("\r\n", "\n");
    let changed = differences(&committed, &current);
    if changed.is_empty() {
        return;
    }
    let written = inputs::scratch().join("analyses-demand.tsv");
    std::fs::write(&written, &current).unwrap();
    panic!(
        "{} answers are not the recorded ones ({}):\n{}\nthe current table is {}",
        changed.len(),
        path.display(),
        changed.iter().take(30).cloned().collect::<Vec<_>>().join("\n"),
        written.display()
    );
}

/// One program's lift and answers, for a difference the oracle prints: `ORACLE_ONE=<file> cargo test -p
/// iridia-analysis --test oracle one -- --ignored --nocapture`. The IR lift as text (structured where it is a control
/// flow graph), its answers and value graph, and the row the oracle makes of it.
#[test]
#[ignore = "a diagnostic: names its program in ORACLE_ONE"]
fn one() {
    let path = std::env::var("ORACLE_ONE").expect("ORACLE_ONE names a program");
    let bytes = std::fs::read(&path).unwrap();
    let new = lift_ir(&bytes);
    if let Ok(which) = std::env::var("ORACLE_ANALYSIS") {
        let started = Instant::now();
        let mut new = new.unwrap();
        iridia_passes::structurize(&mut new.entry).unwrap();
        eprintln!("IR structured: {:?}", started.elapsed());
        fn depth(region: &iridia_ir::Region) -> usize {
            region
                .nodes
                .iter()
                .map(|n| {
                    let inside = n.regions().into_iter().map(depth).max().unwrap_or(0);
                    inside + usize::from(matches!(n, iridia_ir::Node::Loop { .. }))
                })
                .max()
                .unwrap_or(0)
        }
        if let iridia_ir::Body::Regions(r) = &new.entry.body {
            println!("nested loop depth: {}", depth(r));
        }
        println!("IR: {:#?}", ir_answers(&new));
        println!("{which} elapsed: {:?}", started.elapsed());
        return;
    }
    let mut new = new.unwrap();
    for f in std::iter::once(&mut new.entry).chain(new.patch.as_mut()) {
        if matches!(f.body, iridia_ir::Body::Cfg(_)) {
            iridia_passes::structurize(f).unwrap();
        }
        println!("IR:\n{}", iridia_ir::text::print(f));
    }
    println!("IR: {:#?}", ir_answers(&new));
    let values = Values::of_program(&new).unwrap();
    println!("value graph: {values:#?}");
    let (verdict, line) = compare(&bytes);
    println!("comparison: {} {line}", verdict.group());
    if let Verdict::EvaluatorDiffers(why) | Verdict::Unsound(why) = verdict {
        panic!("{why}");
    }
}

/// Run a selected input once with a larger budget when diagnosing an exhausted corpus row.
#[test]
#[ignore = "a diagnostic: ORACLE_ONE, ORACLE_PROBE_BUDGET, optional ORACLE_PROBE_SEED / ORACLE_PROBE_SOURCE"]
fn budget_probe() {
    let bytes = std::fs::read(std::env::var("ORACLE_ONE").unwrap()).unwrap();
    let program = lift_ir(&bytes).unwrap();
    let seed = std::env::var("ORACLE_PROBE_SEED").map_or(0, |s| s.parse().unwrap());
    let budget = std::env::var("ORACLE_PROBE_BUDGET").unwrap().parse().unwrap();
    let source = std::env::var("ORACLE_PROBE_SOURCE").ok().map(|s| {
        let parts: Vec<_> = s.split(':').collect();
        match parts.as_slice() {
            ["word", r, w] => perturb::Source::Word(ResourceId(r.parse().unwrap()), w.parse().unwrap()),
            ["input", i, l] => perturb::Source::Input(i.parse().unwrap(), l.parse().unwrap()),
            ["texel", r, l] => perturb::Source::Texel(ResourceId(r.parse().unwrap()), l.parse().unwrap()),
            _ => panic!("probe source: word:resource:word, input:slot:lane, or texel:resource:lane"),
        }
    });
    println!("IR:\n{}", iridia_ir::text::print(&program.entry));
    perturb::budget_probe(&program, seed, source, budget);
}

#[test]
#[ignore = "a diagnostic: ORACLE_ONE names a program with conflicting unreflected word views"]
fn view_probe() {
    let bytes = std::fs::read(std::env::var("ORACLE_ONE").unwrap()).unwrap();
    let program = lift_ir(&bytes).unwrap();
    println!("{}", perturb::view_report(&program));
}
