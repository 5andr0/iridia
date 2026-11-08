#![allow(clippy::unwrap_used)]
//! The blend recogniser on the IR ([`iridia_analysis::blend`]) on every pixel program of the fixtures, the slim set and
//! Elden Ring's corpus, held to the older one's (the `iridia` crate's, run beside it while it exists): for each output
//! lane of a program both lift, the lane as one [`Scalar`] and as its [`Over`] chain (the base, and each layer's value and
//! weight as scalars), each written with its leaves named by what they are (an input lane, a texel lane or a word, by
//! the interface's own names); an operation a term does not spell is named `op`, since the two vocabularies name their
//! operations differently. Each program's answers are held to tests/expected/blend-<set>.tsv: a hash of the IR's, or why
//! it has none, and whether the older recogniser's were the same, with the first difference's text.
mod inputs;

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

use inputs::{Game, OracleSet, Query, expect_found, oracle_tree};
use iridia_analysis::blend as new_blend;
use iridia_analysis::flow as new_flow;
use primase_testdata::Tree;

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

fn lift_new(bytes: &[u8]) -> Result<iridia_ir::Program, String> {
    let text = |e: iridia_ir::Refusal| e.to_string();
    if !bytes.starts_with(b"DXBC") {
        iridia_d3d9_ir::lift(bytes).map_err(text)
    } else if has_part(bytes, b"DXIL") {
        iridia_dxil::ir::lift(bytes).map_err(text)
    } else {
        iridia_dxbc::ir::lift(bytes).map_err(text)
    }
}

fn lift_old(bytes: &[u8]) -> Result<iridia::Program, String> {
    if !bytes.starts_with(b"DXBC") {
        iridia_d3d9::lift(bytes).map_err(|e| e.to_string())
    } else if has_part(bytes, b"DXIL") {
        iridia_dxil::lift(bytes).map_err(|e| e.to_string())
    } else {
        iridia_dxbc::lift(bytes).map_err(|e| e.to_string())
    }
}

/// What a resource is, by the interface's names: its bind point.
fn register(interface: &iridia_ir::interface::Interface, id: iridia_ir::interface::ResourceId) -> String {
    match interface.resources.iter().find(|r| r.id == id) {
        Some(r) => match &r.provenance.d3d {
            Some(d) => format!("{d:?}"),
            None => format!("#{}", id.0),
        },
        None => format!("unknown #{}", id.0),
    }
}

/// The two vocabularies' answers for one program, as text: each output lane's scalar and chain.
type Lanes = BTreeMap<(usize, u8), Lane>;

/// One output lane's answer: its text and, where every leaf has a name (a leaf the text calls `op` is a node the two
/// graphs do not share), its value under four assignments of numbers to the named leaves (the same name, the same
/// number in both vocabularies; a literal is its value) for the scalar and for each part of its chain.
#[derive(Clone, Debug)]
struct Lane {
    text: String,
    values: Option<Vec<f32>>,
    /// The leaves the lane names (an input lane, a texel lane, a word, a literal), without the `op` ones.
    names: BTreeSet<String>,
}

/// A number for a leaf's name under assignment `k`.
fn assigned(name: &str, k: u64) -> f32 {
    let h = u64::from_str_radix(&fnv(&format!("{k}{name}")), 16).unwrap_or(1);
    0.1 + (h % 2000) as f32 / 1000.0
}

macro_rules! describe {
    ($blend:ident, $flow:ident, $program:expr, $values:expr, $leaf:expr) => {{
        use $blend::{Over, Scalar, Term};
        let (program, values) = ($program, $values);
        let leaf = $leaf;
        let scalar = |n: $flow::NodeId| -> String {
            // a term is spelled once (terms are shared, so a spelling that expanded them would grow with the number of paths);
            // a spelling longer than a line is its hash
            fn at(s: &Scalar, i: usize, name: &dyn Fn(usize) -> String, memo: &mut Vec<Option<String>>) -> String {
                if let Some(Some(done)) = memo.get(i) {
                    return done.clone();
                }
                let mut t = |k: &usize| at(s, *k, name, memo);
                let text = match s.terms.get(i) {
                    Some(Term::Leaf(_)) => name(i),
                    Some(Term::Constant(c)) => format!("{c:?}"),
                    Some(Term::Add([a, b])) => format!("({} + {})", t(a), t(b)),
                    Some(Term::Subtract([a, b])) => format!("({} - {})", t(a), t(b)),
                    Some(Term::Multiply([a, b])) => format!("({} * {})", t(a), t(b)),
                    Some(Term::Min([a, b])) => format!("min({}, {})", t(a), t(b)),
                    Some(Term::Max([a, b])) => format!("max({}, {})", t(a), t(b)),
                    Some(Term::Saturate(a)) => format!("saturate({})", t(a)),
                    None => "?".into(),
                };
                let text = if text.len() > 160 { format!("#{}", fnv(&text)) } else { text };
                if let Some(slot) = memo.get_mut(i) {
                    *slot = Some(text.clone());
                }
                text
            }
            let s = Scalar::of(values, n, &BTreeSet::new());
            let name = |i: usize| match s.terms.get(i) {
                Some(Term::Leaf(n)) => leaf(program, values, *n),
                _ => "?".into(),
            };
            let mut memo = vec![None; s.terms.len()];
            at(&s, s.terms.len().saturating_sub(1), &name, &mut memo)
        };
        let numbers = |n: $flow::NodeId| -> Option<Vec<f32>> {
            let s = Scalar::of(values, n, &BTreeSet::new());
            let mut out = Vec::new();
            for k in 0..4u64 {
                let unnamed = std::cell::Cell::new(false);
                let v = s.evaluate(|leaf_node| {
                    let name = leaf(program, values, leaf_node);
                    if name == "op" {
                        unnamed.set(true);
                        None
                    } else if let Some(value) = name.strip_prefix("literal ") {
                        value.parse::<f32>().ok()
                    } else {
                        Some(assigned(&name, k))
                    }
                });
                if unnamed.get() {
                    return None;
                }
                out.push(v);
            }
            Some(out)
        };
        let named = |n: $flow::NodeId, into: &mut BTreeSet<String>| {
            let s = Scalar::of(values, n, &BTreeSet::new());
            for t in &s.terms {
                if let Term::Leaf(l) = t {
                    let name = leaf(program, values, *l);
                    if name != "op" {
                        into.insert(name);
                    }
                }
            }
        };
        let mut out = Lanes::new();
        for (&(output, lane), &node) in values.outputs() {
            let mut names = BTreeSet::new();
            named(node, &mut names);
            let mut text = scalar(node);
            let mut numeric = numbers(node);
            if let Some(over) = Over::under(values, node) {
                text.push_str(&format!(" over base {}", scalar(over.base)));
                named(over.base, &mut names);
                numeric = numeric.zip(numbers(over.base)).map(|(mut a, b)| {
                    a.extend(b);
                    a
                });
                for layer in &over.layers {
                    text.push_str(&format!(
                        "; layer {} by {}",
                        scalar(layer.value),
                        scalar(layer.weight)
                    ));
                    named(layer.value, &mut names);
                    named(layer.weight, &mut names);
                    for part in [layer.value, layer.weight] {
                        numeric = numeric.zip(numbers(part)).map(|(mut a, b)| {
                            a.extend(b);
                            a
                        });
                    }
                }
            }
            out.insert((output, lane), Lane { text, values: numeric, names });
        }
        out
    }};
}

fn new_lanes(program: &iridia_ir::Program, values: &new_flow::Values) -> Lanes {
    describe!(
        new_blend,
        new_flow,
        program,
        values,
        |program: &iridia_ir::Program, values: &new_flow::Values, n: new_flow::NodeId| {
            match values.node(n) {
                new_flow::Node::Input { input, lane } => {
                    let label = &program.interface.inputs[*input].label;
                    format!("in {}{}.{lane}", label.name, label.index)
                }
                new_flow::Node::Texel { resource, lane, .. } => {
                    format!("texel {}.{lane}", register(&program.interface, *resource))
                }
                new_flow::Node::Global {
                    resource,
                    at: new_flow::At::Component(c),
                } => format!("word {}[{c}]", register(&program.interface, *resource)),
                new_flow::Node::Literal(l) => format!("literal {:?}", l.0),
                _ => "op".into(),
            }
        }
    )
}

fn old_lanes(program: &iridia::Program, values: &iridia::flow::Values) -> Lanes {
    use iridia::flow as old_flow;
    describe!(
        blend_old,
        old_flow,
        program,
        values,
        |program: &iridia::Program, values: &old_flow::Values, n: old_flow::NodeId| {
            let old_register = |r: &Option<iridia::ResourceId>| match r {
                Some(id) => register(&program.interface, *id),
                None => "?".into(),
            };
            match values.node(n) {
                old_flow::Node::Input { input, lane } => {
                    let label = &program.interface.inputs[*input].label;
                    format!("in {}{}.{lane}", label.name, label.index)
                }
                old_flow::Node::Texel { resource, lane, .. } => {
                    format!("texel {}.{lane}", old_register(resource))
                }
                old_flow::Node::Global {
                    resource,
                    at: old_flow::At::Component(c),
                } => format!("word {}[{c}]", old_register(resource)),
                old_flow::Node::Literal(l) => format!("literal {:?}", l.0),
                _ => "op".into(),
            }
        }
    )
}

mod blend_old {
    pub use iridia::blend::*;
}

/// BLEND_ORACLE_TRACE=1: where a program is, on stderr.
fn trace(what: &str) {
    if std::env::var_os("BLEND_ORACLE_TRACE").is_some() {
        eprintln!("[blend_oracle] {what}");
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(ToString::to_string)
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic".into())
}

fn fnv(text: &str) -> String {
    let h = text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{h:016x}")
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Verdict {
    /// The IR's and the older answers are the same text.
    Same,
    /// Every lane's texts differ only where they are the same value of the same named leaves (regrouped or reordered); the
    /// first, as `lane: older -> IR`.
    Equivalent(String),
    /// A lane whose texts differ has a leaf one side or the other does not name (a node the graphs do not share: an
    /// older bitcast, an operation a vocabulary does not spell), so no value is compared; the IR names every leaf the
    /// older one does (it recognises more); the first.
    Gain(String),
    /// As [`Verdict::Gain`], but the older recogniser names a leaf the IR's lane does not; the first.
    Partial(String),
    /// A lane's values differ under the same named leaves; the first, as `lane: older -> IR`.
    Differ(String),
    /// Not a pixel program, or one side did not lift or build its graph, as the group says.
    Skipped(String),
}

impl Verdict {
    fn group(&self) -> String {
        match self {
            Verdict::Same => "same".into(),
            Verdict::Equivalent(_) => "equivalent".into(),
            Verdict::Gain(_) => "gain".into(),
            Verdict::Partial(_) => "partial".into(),
            Verdict::Differ(_) => "differ".into(),
            Verdict::Skipped(why) => format!("skipped: {why}"),
        }
    }
}

fn reason_group(reason: &str) -> String {
    let mut out = String::new();
    let mut digits = false;
    for c in reason.lines().next().unwrap_or("").chars() {
        if c.is_ascii_digit() {
            if !digits {
                out.push('N');
            }
            digits = true;
        } else {
            digits = false;
            out.push(c);
        }
    }
    out.chars().take(120).collect()
}

/// (verdict, the table's line for the program).
fn compare(bytes: &[u8], older_refused: Option<&str>) -> (Verdict, String) {
    let started = std::time::Instant::now();
    let new = match lift_new(bytes) {
        Ok(p) => p,
        Err(e) => {
            return (
                Verdict::Skipped(format!("IR lift: {}", reason_group(&e))),
                "not lifted".into(),
            );
        }
    };
    if new.stage != iridia_ir::interface::Stage::Pixel {
        return (
            Verdict::Skipped("not a pixel program".into()),
            "not a pixel program".into(),
        );
    }
    let new_values = match new_flow::Values::of_program(&new) {
        Ok(v) => v,
        Err(e) => {
            return (
                Verdict::Skipped(format!("IR graph: {}", reason_group(&e))),
                format!("no graph: {}", reason_group(&e)),
            );
        }
    };
    trace("IR graph built");
    let ir = new_lanes(&new, &new_values);
    trace("IR lanes described");
    if std::env::var_os("BLEND_ORACLE_SIDE").is_some_and(|s| s == "new") {
        return (Verdict::Skipped("one side".into()), "one side".into());
    }
    if let Some(why) = older_refused {
        assert!(
            started.elapsed() < IR_BOUND,
            "the IR side took {:?} on a program the older side is refused (bound {IR_BOUND:?})",
            started.elapsed()
        );
        return (
            Verdict::Skipped(format!("older side not run: {why}")),
            format!(
                "{}\tolder side not run: {why}",
                fnv(&format!(
                    "{:?}",
                    ir.iter().map(|(k, l)| (k, &l.text)).collect::<Vec<_>>()
                ))
            ),
        );
    }
    let line = fnv(&format!(
        "{:?}",
        ir.iter().map(|(k, l)| (k, &l.text)).collect::<Vec<_>>()
    ));
    let old = match lift_old(bytes) {
        Ok(p) => p,
        Err(e) => {
            return (
                Verdict::Skipped(format!("older lift: {}", reason_group(&e))),
                line,
            );
        }
    };
    let old_values = match iridia::flow::Values::of_program(&old) {
        Ok(v) => v,
        Err(e) => {
            return (
                Verdict::Skipped(format!("older graph: {}", reason_group(&e))),
                line,
            );
        }
    };
    trace("older graph built");
    let older = old_lanes(&old, &old_values);
    trace("older lanes described");
    // lanes whose text differs are held to their values where both sides name every leaf
    let (mut equivalent, mut gain, mut partial, mut worse): (
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    ) = (None, None, None, None);
    for k in ir.keys().chain(older.keys()).collect::<BTreeSet<_>>() {
        let (new_lane, old_lane) = (ir.get(k), older.get(k));
        if new_lane.map(|l| &l.text) == old_lane.map(|l| &l.text) {
            continue;
        }
        let text = format!(
            "lane {k:?}: {} -> {}",
            old_lane.map_or("none", |l| l.text.as_str()),
            new_lane.map_or("none", |l| l.text.as_str())
        );
        match (
            new_lane.and_then(|l| l.values.as_ref()),
            old_lane.and_then(|l| l.values.as_ref()),
        ) {
            (Some(n), Some(o)) if n.len() == o.len() => {
                let close = n.iter().zip(o).all(|(a, b)| {
                    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1e-2)
                });
                if close {
                    equivalent.get_or_insert(text);
                } else {
                    worse.get_or_insert(text);
                }
            }
            _ => {
                // the IR names every leaf the older lane did (and at least as many "op"s are the older one's own)
                let gained = match (new_lane, old_lane) {
                    (Some(n), Some(o)) => o.names.is_subset(&n.names),
                    _ => false,
                };
                if gained {
                    gain.get_or_insert(text);
                } else {
                    partial.get_or_insert(text);
                }
            }
        }
    }
    match (worse, partial, gain, equivalent) {
        (Some(text), _, _, _) => (Verdict::Differ(text.clone()), format!("{line}\tdiffer\t{text}")),
        (None, Some(text), _, _) => (Verdict::Partial(text.clone()), format!("{line}\tpartial\t{text}")),
        (None, None, Some(text), _) => (Verdict::Gain(text.clone()), format!("{line}\tgain\t{text}")),
        (None, None, None, Some(text)) => (
            Verdict::Equivalent(text.clone()),
            format!("{line}\tequivalent\t{text}"),
        ),
        (None, None, None, None) => (Verdict::Same, format!("{line}\tsame")),
    }
}

/// Programs the older value graph cannot build in bounded memory, named rather than run on the older side (the IR side
/// runs on them, and must stay within [`IR_BOUND`]): `gxposteffect/SceneBokeh_SimpleBlurFadePS.ppo` has 24 nested loops,
/// and the older graph keeps exponentially many trial graphs (a 4-worker Elden pass of this oracle reached 54.8 GB on
/// it). The IR graph walks each loop once (header phis before the body).
const KNOWN_OLDER_BLOWUPS: [(&str, &str); 1] = [(
    "gxposteffect/SceneBokeh_SimpleBlurFadePS.ppo",
    "24 nested loops: the older value graph is exponential in memory",
)];

/// What the IR side may take on a program the older side is refused: its lift, graph and blend answers.
const IR_BOUND: std::time::Duration = std::time::Duration::from_secs(1);

enum Source {
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

fn hold(set: &str, source: Source) {
    let mut names = source.names();
    // BLEND_ORACLE_ONLY=<text>: only the programs whose name holds it (measuring one program)
    if let Ok(only) = std::env::var("BLEND_ORACLE_ONLY") {
        names.retain(|n| n.contains(&only));
    }
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
    let next = AtomicUsize::new(0);
    // BLEND_ORACLE_PROGRESS=<file> logs each program as a worker starts and finishes it (to find the one a run is stuck in)
    let progress =
        std::env::var_os("BLEND_ORACLE_PROGRESS").map(|p| Mutex::new(std::fs::File::create(p).unwrap()));
    let results: Vec<Mutex<Option<(Verdict, String)>>> = names.iter().map(|_| Mutex::default()).collect();
    std::thread::scope(|s| {
        for _ in 0..inputs::threads().get() {
            let (names, next, results, read, progress) = (&names, &next, &results, &read, &progress);
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= names.len() {
                        break;
                    }
                    let bytes = read(i);
                    if containers_only && !bytes.starts_with(b"DXBC") {
                        continue;
                    }
                    if let Some(log) = &progress {
                        let _ = log.lock().unwrap().write_all(
                            format!(
                                "start {i} {} {}
",
                                names[i],
                                bytes.len()
                            )
                            .as_bytes(),
                        );
                    }
                    let older_refused = KNOWN_OLDER_BLOWUPS
                        .iter()
                        .find(|(n, _)| *n == names[i] && std::env::var_os("BLEND_ORACLE_ONLY").is_none())
                        .map(|(_, why)| *why);
                    let outcome = catch_unwind(AssertUnwindSafe(|| compare(&bytes, older_refused)))
                        .unwrap_or_else(|p| {
                            let why = format!("panic: {}", panic_message(&*p));
                            (Verdict::Skipped(why.clone()), why)
                        });
                    *results[i].lock().unwrap() = Some(outcome);
                    if let Some(log) = &progress {
                        let _ = log.lock().unwrap().write_all(
                            format!(
                                "done {i}
"
                            )
                            .as_bytes(),
                        );
                    }
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
    println!("{set}: {} programs", outcomes.len());
    let mut by: BTreeMap<String, (usize, &str)> = BTreeMap::new();
    for (name, verdict, _) in &outcomes {
        by.entry(verdict.group()).or_insert((0, name)).0 += 1;
    }
    for (group, (n, example)) in &by {
        println!("  {n:5}  {group}  (e.g. {example})");
    }
    let panics: Vec<String> = outcomes
        .iter()
        .filter(|(_, v, _)| matches!(v, Verdict::Skipped(w) if w.starts_with("panic")))
        .map(|(n, v, _)| format!("  {n}: {}", v.group()))
        .collect();
    let mut current = format!(
        "# Each {set} pixel program's blend answers on the IR (tests/blend_oracle.rs): program, a hash of each output lane's \
         scalar and layers, and whether the older recogniser's were the same.\n"
    );
    for (name, _, line) in &outcomes {
        if !line.starts_with("not a pixel program") && !line.starts_with("not lifted") {
            current.push_str(&format!("{name}\t{line}\n"));
        }
    }
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/expected/blend-{set}.tsv"));
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .replace("\r\n", "\n");
    if panics.is_empty() && committed == current {
        return;
    }
    let written = inputs::scratch().join(format!("blend-{set}.tsv"));
    std::fs::write(&written, &current).unwrap();
    panic!(
        "{} panics:\n{}\nthe current table is {} (it replaces {} where a change is meant)",
        panics.len(),
        panics.join("\n"),
        written.display(),
        path.display()
    );
}

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
