//! lit's feature expressions (llvm/utils/lit/lit/BooleanExpression.py) for `REQUIRES`, `UNSUPPORTED` and `XFAIL`:
//! identifiers, `!`, `&&`, `||`, parentheses and `true`, against the features a device has; the features of iridia's
//! device ([`features`]), and which tests no Direct3D 12 device runs ([`not_for_direct3d`]).
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Ident(String),
    Not,
    And,
    Or,
    Open,
    Close,
}

fn tokens(s: &str) -> Result<Vec<Token>, String> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    let ident = |c: u8| c.is_ascii_alphanumeric() || b"-+=._".contains(&c);
    while i < b.len() {
        match b[i] {
            c if c.is_ascii_whitespace() => i += 1,
            b'(' => {
                out.push(Token::Open);
                i += 1;
            }
            b')' => {
                out.push(Token::Close);
                i += 1;
            }
            b'!' => {
                out.push(Token::Not);
                i += 1;
            }
            b'&' if b.get(i + 1) == Some(&b'&') => {
                out.push(Token::And);
                i += 2;
            }
            b'|' if b.get(i + 1) == Some(&b'|') => {
                out.push(Token::Or);
                i += 2;
            }
            c if ident(c) => {
                let start = i;
                while i < b.len() && ident(b[i]) {
                    i += 1;
                }
                out.push(Token::Ident(s[start..i].to_string()));
            }
            _ => {
                return Err(format!(
                    "'{s}': token at {i} lit does not read (regex features are not modelled)"
                ));
            }
        }
    }
    Ok(out)
}

struct Eval<'a> {
    t: Vec<Token>,
    i: usize,
    features: &'a BTreeSet<String>,
}

impl Eval<'_> {
    fn or(&mut self) -> Result<bool, String> {
        let mut v = self.and()?;
        while self.t.get(self.i) == Some(&Token::Or) {
            self.i += 1;
            let r = self.and()?;
            v = v || r;
        }
        Ok(v)
    }

    fn and(&mut self) -> Result<bool, String> {
        let mut v = self.not()?;
        while self.t.get(self.i) == Some(&Token::And) {
            self.i += 1;
            let r = self.not()?;
            v = v && r;
        }
        Ok(v)
    }

    fn not(&mut self) -> Result<bool, String> {
        if self.t.get(self.i) == Some(&Token::Not) {
            self.i += 1;
            return Ok(!self.not()?);
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<bool, String> {
        match self.t.get(self.i).cloned() {
            Some(Token::Open) => {
                self.i += 1;
                let v = self.or()?;
                if self.t.get(self.i) != Some(&Token::Close) {
                    return Err("expected ')'".into());
                }
                self.i += 1;
                Ok(v)
            }
            Some(Token::Ident(name)) => {
                self.i += 1;
                Ok(name == "true" || self.features.contains(&name))
            }
            other => Err(format!("expected a feature, found {other:?}")),
        }
    }
}

/// Whether `expr` holds for `features`.
pub fn holds(expr: &str, features: &BTreeSet<String>) -> Result<bool, String> {
    let mut e = Eval {
        t: tokens(expr)?,
        i: 0,
        features,
    };
    let v = e.or()?;
    if e.i != e.t.len() {
        return Err(format!("'{expr}': trailing tokens"));
    }
    Ok(v)
}

/// A `REQUIRES:`, `UNSUPPORTED:` or `XFAIL:` line's comma-separated expressions.
pub fn split(line: &str) -> impl Iterator<Item = &str> {
    line.split(',').map(str::trim).filter(|s| !s.is_empty())
}

/// What lit makes of a test's directives on a device: the first `REQUIRES` expression that fails, or the first
/// `UNSUPPORTED` that holds, and whether an `XFAIL` holds.
pub fn applies(
    requires: &[String],
    unsupported: &[String],
    xfail: &[String],
    features: &BTreeSet<String>,
) -> Result<(Option<String>, bool), String> {
    for line in requires {
        for e in split(line) {
            if !holds(e, features)? {
                return Ok((Some(format!("REQUIRES: {e}")), false));
            }
        }
    }
    for line in unsupported {
        for e in split(line) {
            if e == "*" || holds(e, features)? {
                return Ok((Some(format!("UNSUPPORTED: {e}")), false));
            }
        }
    }
    let mut x = false;
    for line in xfail {
        for e in split(line) {
            if e == "*" || holds(e, features)? {
                x = true;
            }
        }
    }
    Ok((None, x))
}

/// The features of lit's configuration the suite is read for, DirectX with DXC on Windows, whatever the device.
const CONFIGURATION: [&str; 3] = ["DirectX", "DXC", "Windows"];

/// The features a Direct3D 12 device can report that iridia's does not ([`features`]), of those the suite asks for:
/// the wave widths but 32, barycentrics, stencil reference from the pixel program, shading rate, several viewports,
/// and acceleration structures and ray-tracing pipelines.
const OTHER_DEVICES: [&str; 8] = [
    "WaveSize_64",
    "WaveSize_128",
    "Barycentrics",
    "ShaderStencilRef",
    "FragmentShadingRate",
    "multi-viewport",
    "acceleration-structure",
    "raytracing-pipeline",
];

/// The features the device has, as lit.cfg.py would set them from api-query for it: the configuration
/// (`CONFIGURATION`), every shader model 6 the lifter reads, what the executor runs (16- and 64-bit types, 64-bit and
/// float atomics, derivatives in compute, a 32-lane wave), unbounded descriptor tables, and the golden images; not
/// those of `OTHER_DEVICES`.
pub fn features() -> BTreeSet<String> {
    let mut f: BTreeSet<String> = CONFIGURATION.map(String::from).into();
    f.extend(
        [
            "Int16",
            "Half",
            "Double",
            "Int64",
            "Int64GroupSharedAtomics",
            "Int64TypedResourceAtomics",
            "Float32GroupSharedAtomics",
            "DerivativesInCompute",
            "ResourceBindingTier3",
            "WaveSize_32",
            "goldenimage",
            "MeshShader",
        ]
        .map(String::from),
    );
    for minor in 0..=9 {
        f.insert(format!("SM_6_{minor}"));
    }
    f
}

/// Why lit would run a test on no Direct3D 12 device in this configuration, if it would not: a `REQUIRES` that fails
/// with every feature some device has ([`features`] and `OTHER_DEVICES`: another API, another compiler), or an
/// `UNSUPPORTED` that holds with the configuration's alone (`CONFIGURATION`).
pub fn not_for_direct3d(requires: &[String], unsupported: &[String]) -> Result<Option<String>, String> {
    let mut any_device = features();
    any_device.extend(OTHER_DEVICES.map(String::from));
    let configuration: BTreeSet<String> = CONFIGURATION.map(String::from).into();
    let (on_any, _) = applies(requires, &[], &[], &any_device)?;
    let (on_none, _) = applies(&[], unsupported, &[], &configuration)?;
    Ok(on_any.or(on_none))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expressions() {
        let f: BTreeSet<String> = ["DirectX", "Int64", "SM_6_6"].map(String::from).into();
        assert!(holds("DirectX && Int64", &f).expect("reads"));
        assert!(!holds("Vulkan || Clang", &f).expect("reads"));
        assert!(holds("!Vulkan", &f).expect("reads"));
        assert!(holds("(!Vulkan || VulkanInt64BufferAtomics) && SM_6_6", &f).expect("reads"));
        let (skip, x) =
            applies(&["Int64, Half".into()], &[], &["DirectX && Int64".into()], &f).expect("reads");
        assert_eq!(skip.as_deref(), Some("REQUIRES: Half"));
        assert!(!x);
    }

    /// A test for another API or compiler is for no Direct3D 12 device; one needing a feature some Direct3D 12 device
    /// has is for Direct3D, though iridia's device lacks it, and lit skips it there.
    #[test]
    fn not_for_direct3d_is_what_no_device_runs() {
        let none = |requires: &str, unsupported: &str| {
            let list = |s: &str| if s.is_empty() { vec![] } else { vec![s.to_string()] };
            not_for_direct3d(&list(requires), &list(unsupported)).expect("reads")
        };
        assert_eq!(none("Vulkan", "").as_deref(), Some("REQUIRES: Vulkan"));
        assert_eq!(none("Clang", "").as_deref(), Some("REQUIRES: Clang"));
        assert_eq!(
            none("", "Metal || DirectX").as_deref(),
            Some("UNSUPPORTED: Metal || DirectX")
        );
        assert_eq!(none("", "!Vulkan").as_deref(), Some("UNSUPPORTED: !Vulkan"));
        assert_eq!(none("", "DXC").as_deref(), Some("UNSUPPORTED: DXC"));
        for device in [
            "acceleration-structure",
            "raytracing-pipeline",
            "WaveSize_64",
            "Barycentrics",
        ] {
            assert_eq!(none(device, ""), None, "{device}");
            let (skip, _) = applies(&[device.to_string()], &[], &[], &features()).expect("reads");
            assert_eq!(
                skip,
                Some(format!("REQUIRES: {device}")),
                "iridia's device lacks {device}"
            );
        }
        // a device's own UNSUPPORTED (WARP) holds on no device of the configuration alone, nor on iridia's
        assert_eq!(none("", "WARP"), None);
        assert_eq!(
            applies(&[], &["WARP".into()], &[], &features()).expect("reads").0,
            None
        );
        assert_eq!(none("Int64, SM_6_6", ""), None);
    }
}
