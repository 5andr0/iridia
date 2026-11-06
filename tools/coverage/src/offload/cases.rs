//! The offload-test-suite case file `suites/offload/offload_cases.py` writes: each test's RUN lines as steps, the
//! programs dxc made of its shaders (base64 by digest), the golden images it compares with (by path and SHA-256), and
//! its lit directives and FileCheck lines.
use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

/// The case file's format number.
pub const FORMAT: u32 = 1;

/// A case file.
#[derive(Clone, Debug, Deserialize)]
pub struct File {
    /// Its format number.
    pub format: u32,
    /// Where its tests come from.
    pub source: String,
    /// The programs, base64, by digest.
    pub programs: BTreeMap<String, String>,
    /// The golden images, SHA-256 by path in llvm/offload-golden-images.
    pub images: BTreeMap<String, String>,
    /// The tests.
    pub cases: Vec<Case>,
}

/// A FileCheck line of a test.
#[derive(Clone, Debug, Deserialize)]
pub struct CheckLine {
    /// Its prefix.
    pub prefix: String,
    /// `CHECK`, `NEXT`, `NOT`, ...
    pub kind: String,
    /// Its pattern.
    pub pattern: String,
}

/// A test.
#[derive(Clone, Debug, Deserialize)]
pub struct Case {
    /// Its path under test/.
    pub test: String,
    /// Its `REQUIRES:` lines.
    pub requires: Vec<String>,
    /// Its `UNSUPPORTED:` lines.
    pub unsupported: Vec<String>,
    /// Its `XFAIL:` lines.
    pub xfail: Vec<String>,
    /// Its RUN lines, as steps (JSON objects keyed by what they run: `compile`, `offload`, `imgdiff`, `unmodelled`).
    pub steps: Vec<Value>,
    /// Its FileCheck lines, every prefix.
    pub checks: Vec<CheckLine>,
}

/// Read a case file.
pub fn read(text: &str) -> Result<File, String> {
    let file: File = serde_json::from_str(text).map_err(|e| e.to_string())?;
    if file.format != FORMAT {
        return Err(format!(
            "case file format {}, this runner reads {FORMAT}",
            file.format
        ));
    }
    Ok(file)
}
