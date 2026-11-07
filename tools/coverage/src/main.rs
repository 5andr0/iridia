//! `iridia-coverage`: run the suites and write the coverage table.
use std::collections::BTreeMap;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("vkd3d") => {
            let path = args.get(1).ok_or("usage: iridia-coverage vkd3d <cases.json>")?;
            let text = iridia_coverage::inflate::read_text(std::path::Path::new(path))?;
            let file = iridia_coverage::vkd3d::cases::read(&text)?;
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let (results, programs) = iridia_coverage::vkd3d::run::run_all(&file, threads);
            let mut runs: BTreeMap<String, usize> = BTreeMap::new();
            let mut probes: BTreeMap<String, usize> = BTreeMap::new();
            let mut why: BTreeMap<String, (usize, String)> = BTreeMap::new();
            for r in &results {
                let extended = if r.selected() { "" } else { " (extended)" };
                *runs.entry(format!("{} {}{extended}", r.runner, r.class())).or_default() += 1;
                for p in &r.probes {
                    let extended = if p.selected { "" } else { " (extended)" };
                    *probes.entry(format!("{}{extended}", p.class())).or_default() += 1;
                }
                if let Some(p) = r.deciding() {
                    let extended = if r.selected() { "" } else { " (extended)" };
                    let key = format!(
                        "{} {}{extended} {}",
                        r.runner,
                        p.outcome.class(),
                        class_of(&p.outcome.reason(&r.file))
                    );
                    let e = why.entry(key).or_insert((0, format!("{}:{}: {}", r.file, p.line, p.outcome.detail())));
                    e.0 += 1;
                }
            }
            let lifted = programs.values().filter(|p| p.is_ok()).count();
            println!("{} programs tried, {lifted} lifted", programs.len());
            for (k, v) in &runs {
                println!("{v:6} runs {k}");
            }
            for (k, v) in &probes {
                println!("{v:6} probes {k}");
            }
            println!("runs that do not pass, by why:");
            print_classes(why);
            Ok(())
        }
        Some("offload") => {
            let usage = "usage: iridia-coverage offload <cases.json> [<offload-golden-images checkout>]";
            let path = args.get(1).ok_or(usage)?;
            let text = iridia_coverage::inflate::read_text(std::path::Path::new(path))?;
            let file = iridia_coverage::offload::cases::read(&text)?;
            let golden = args.get(2).map(std::path::PathBuf::from);
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let results = iridia_coverage::offload::run::run_all(&file, threads, golden.as_deref());
            let mut classes: BTreeMap<String, (usize, String)> = BTreeMap::new();
            let mut totals: BTreeMap<&str, usize> = BTreeMap::new();
            for r in &results {
                *totals.entry(r.outcome.class()).or_default() += 1;
                if r.outcome.class() != "pass" {
                    let key = format!("{} {}", r.outcome.class(), class_of(&r.outcome.detail()));
                    let e = classes
                        .entry(key)
                        .or_insert((0, format!("{}: {}", r.test, r.outcome.detail())));
                    e.0 += 1;
                }
            }
            for (k, v) in &totals {
                println!("{v:6} tests {k}");
            }
            println!("tests that do not pass, by why:");
            print_classes(classes);
            Ok(())
        }
        Some("table") => {
            // iridia-coverage table [<out dir>] [<offload-golden-images checkout>]
            let suites = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("suites");
            let out = args
                .get(1)
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/coverage"));
            let golden = args.get(2).map(std::path::PathBuf::from);
            let vkd3d: Vec<std::path::PathBuf> = ["d3d9", "dxbc", "dxil"]
                .iter()
                .map(|s| suites.join("vkd3d").join(format!("{s}.json.gz")))
                .collect();
            let offload = suites.join("offload/offload.json.gz");
            let inputs = iridia_coverage::table::Inputs {
                vkd3d: vkd3d.iter().map(std::path::PathBuf::as_path).collect(),
                offload: Some(offload.as_path()),
                golden: golden.as_deref(),
                commit: env!("CARGO_PKG_VERSION").to_string(),
                threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            };
            let (doc, md) = iridia_coverage::table::make(&inputs)?;
            std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
            let json = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())? + "\n";
            std::fs::write(out.join("coverage.json"), json).map_err(|e| e.to_string())?;
            std::fs::write(out.join("coverage.md"), md).map_err(|e| e.to_string())?;
            println!("wrote {}", out.display());
            Ok(())
        }
        _ => Err("usage: iridia-coverage vkd3d <cases.json> | offload <cases.json> [<golden images>] | table [<out dir>] [<golden images>]".into()),
    }
}

/// Classes and their counts, most first, each with an example.
fn print_classes(classes: BTreeMap<String, (usize, String)>) {
    let mut by: Vec<_> = classes.into_iter().collect();
    by.sort_by_key(|a| std::cmp::Reverse(a.1.0));
    for (k, (n, e)) in by {
        println!(
            "{n:6} {k}
         e.g. {e}"
        );
    }
}

/// A message with its program names (16 hexadecimal digits) and its numbers taken out, so one kind of outcome is one
/// class.
fn class_of(e: &str) -> String {
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
