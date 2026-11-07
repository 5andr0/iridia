//! `iridia-coverage`: run the suites and write the coverage table.
use std::collections::BTreeMap;

fn main() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--reference`: the suites' programs through the lift the IR replaces, not the IR
    let through = match args.iter().position(|a| a == "--reference") {
        Some(i) => {
            args.remove(i);
            iridia_coverage::front::Path::Reference
        }
        None => iridia_coverage::front::Path::Ir,
    };
    // `--shard <index>/<count>`: the part runs that share of the case file's cases
    let shard = match args.iter().position(|a| a == "--shard") {
        Some(i) if i + 1 < args.len() => {
            let text = args.remove(i + 1);
            args.remove(i);
            iridia_coverage::shard::Shard::parse(&text)?
        }
        Some(_) => return Err("--shard needs index/count".into()),
        None => iridia_coverage::shard::Shard::WHOLE,
    };
    // `--parts <dir>`: the table of the parts `part` wrote there, not run again
    let parts_dir = match args.iter().position(|a| a == "--parts") {
        Some(i) if i + 1 < args.len() => {
            let dir = args.remove(i + 1);
            args.remove(i);
            Some(std::path::PathBuf::from(dir))
        }
        Some(_) => return Err("--parts needs a folder".into()),
        None => None,
    };
    match args.first().map(String::as_str) {
        Some("vkd3d") => {
            let path = args.get(1).ok_or("usage: iridia-coverage vkd3d <cases.json>")?;
            let text = iridia_coverage::inflate::read_text(std::path::Path::new(path))?;
            let file = iridia_coverage::vkd3d::cases::read(&text)?;
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let (results, programs) = iridia_coverage::vkd3d::run::run_all(&file, threads, through);
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
                        iridia_coverage::table::class_of(&p.outcome.reason(&r.file))
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
            let mut slowest: Vec<_> = results.iter().collect();
            slowest.sort_by_key(|r| std::cmp::Reverse(r.took));
            println!("slowest runs:");
            for r in slowest.iter().take(10) {
                println!("{:8.1}s {} {} {}", r.took.as_secs_f64(), r.file, r.runner, r.model);
            }
            Ok(())
        }
        Some("offload") => {
            let usage = "usage: iridia-coverage offload <cases.json> [<offload-golden-images checkout>]";
            let path = args.get(1).ok_or(usage)?;
            let text = iridia_coverage::inflate::read_text(std::path::Path::new(path))?;
            let file = iridia_coverage::offload::cases::read(&text)?;
            let golden = args.get(2).map(std::path::PathBuf::from);
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let results = iridia_coverage::offload::run::run_all(&file, threads, golden.as_deref(), through);
            let mut classes: BTreeMap<String, (usize, String)> = BTreeMap::new();
            let mut totals: BTreeMap<&str, usize> = BTreeMap::new();
            for r in &results {
                *totals.entry(r.outcome.class()).or_default() += 1;
                if r.outcome.class() != "pass" {
                    let key = format!("{} {}", r.outcome.class(), iridia_coverage::table::class_of(&r.outcome.detail()));
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
            let mut slowest: Vec<_> = results.iter().collect();
            slowest.sort_by_key(|r| std::cmp::Reverse(r.took));
            println!("slowest tests:");
            for r in slowest.iter().take(10) {
                println!("{:8.1}s {}", r.took.as_secs_f64(), r.test);
            }
            Ok(())
        }
        Some("part") => {
            // iridia-coverage part vkd3d|offload <cases.json> <out.json> [<offload-golden-images checkout>] [--reference]
            //     [--shard <index>/<count>]
            let usage = "usage: iridia-coverage part vkd3d|offload <cases.json> <out.json> [<golden images>] [--reference] [--shard <index>/<count>]";
            let suite = match args.get(1).map(String::as_str) {
                Some("vkd3d") => iridia_coverage::table::Suite::Vkd3d,
                Some("offload") => iridia_coverage::table::Suite::Offload,
                _ => return Err(usage.into()),
            };
            let cases = std::path::PathBuf::from(args.get(2).ok_or(usage)?);
            let out = std::path::PathBuf::from(args.get(3).ok_or(usage)?);
            let golden = args.get(4).map(std::path::PathBuf::from);
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let part = iridia_coverage::table::Part::run(suite, &cases, through, shard, threads, golden.as_deref())?;
            let json = serde_json::to_string(&part).map_err(|e| e.to_string())?;
            std::fs::write(&out, json).map_err(|e| format!("{}: {e}", out.display()))?;
            println!("wrote {}", out.display());
            Ok(())
        }
        Some("table") => {
            // iridia-coverage table [<out dir>] [<offload-golden-images checkout>] [--parts <dir>]
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
                commit: env!("CARGO_PKG_VERSION").to_string(),
            };
            let parts = match &parts_dir {
                Some(dir) => read_parts(dir)?,
                None => {
                    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
                    iridia_coverage::table::run_parts(&inputs, threads, golden.as_deref())?
                }
            };
            let (doc, md) = iridia_coverage::table::make(&inputs, &parts)?;
            std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
            let json = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())? + "
";
            std::fs::write(out.join("coverage.json"), json).map_err(|e| e.to_string())?;
            std::fs::write(out.join("coverage.md"), md).map_err(|e| e.to_string())?;
            println!("wrote {}", out.display());
            Ok(())
        }
        _ => Err("usage: iridia-coverage vkd3d <cases.json> [--reference] | offload <cases.json> [<golden images>] [--reference] | part vkd3d|offload <cases.json> <out.json> [<golden images>] [--reference] [--shard <index>/<count>] | table [<out dir>] [<golden images>] [--parts <dir>]".into()),
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

/// The parts `iridia-coverage part` wrote in `dir`: every `.json` file there.
fn read_parts(dir: &std::path::Path) -> Result<Vec<iridia_coverage::table::Part>, String> {
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|p| {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))
        })
        .collect()
}
