//! The dx readers' versions as this build takes them, from the workspace's lock file, for the table to name
//! (`DX_READERS`: `d3d9-reader, dxbc-reader, dxil-reader 0.11.0`, or each with its own version where they differ).
use std::path::Path;

const READERS: [&str; 3] = ["d3d9-reader", "dxbc-reader", "dxil-reader"];

fn main() {
    let lock = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let text = std::fs::read_to_string(&lock).unwrap_or_else(|e| panic!("{}: {e}", lock.display()));
    let versions: Vec<String> = READERS
        .iter()
        .map(|reader| version_of(&text, reader).unwrap_or_else(|| panic!("{reader} is not in Cargo.lock")))
        .collect();
    let named = if versions.iter().all(|v| *v == versions[0]) {
        format!("{} {}", READERS.join(", "), versions[0])
    } else {
        READERS
            .iter()
            .zip(&versions)
            .map(|(r, v)| format!("{r} {v}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("cargo:rustc-env=DX_READERS={named}");
}

/// The version of the package `name` in a lock file: the `version` line of its `[[package]]` entry.
fn version_of(lock: &str, name: &str) -> Option<String> {
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() == format!("name = \"{name}\"") {
            let version = lines
                .next()?
                .trim()
                .strip_prefix("version = \"")?
                .strip_suffix('"')?;
            return Some(version.to_owned());
        }
    }
    None
}
