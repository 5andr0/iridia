//! lit's feature expressions (llvm/utils/lit/lit/BooleanExpression.py) for `REQUIRES`, `UNSUPPORTED` and `XFAIL`:
//! identifiers, `!`, `&&`, `||`, parentheses and `true`, against the features a device has.
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
}
