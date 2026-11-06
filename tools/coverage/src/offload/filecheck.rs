//! LLVM's FileCheck (llvm/lib/FileCheck) as the offload-test-suite uses it: `CHECK`, `-NEXT`, `-SAME`, `-NOT`,
//! `-DAG`, `-LABEL` and `-EMPTY` directives under the prefixes a RUN line names, whitespace canonicalised in the input
//! and the patterns (FileCheck's default without `--strict-whitespace`), and `{{regex}}` in a pattern, read by a small
//! backtracking engine (the suite's regexes are `.*`, literals and an alternation). Pattern variables (`[[VAR]]`) are
//! refused, since no test uses them.

/// A directive kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `CHECK`.
    Plain,
    /// `CHECK-NEXT`.
    Next,
    /// `CHECK-SAME`.
    Same,
    /// `CHECK-NOT`.
    Not,
    /// `CHECK-DAG`.
    Dag,
    /// `CHECK-LABEL`.
    Label,
    /// `CHECK-EMPTY`.
    Empty,
}

impl Kind {
    /// The kind a directive suffix names (`""`, `"NEXT"`, ...).
    pub fn named(s: &str) -> Option<Kind> {
        Some(match s {
            "" | "CHECK" => Kind::Plain,
            "NEXT" => Kind::Next,
            "SAME" => Kind::Same,
            "NOT" => Kind::Not,
            "DAG" => Kind::Dag,
            "LABEL" => Kind::Label,
            "EMPTY" => Kind::Empty,
            _ => return None,
        })
    }
}

/// One directive: its kind and pattern.
#[derive(Clone, Debug)]
pub struct Directive {
    /// Its kind.
    pub kind: Kind,
    /// Its pattern, as written.
    pub pattern: String,
}

// A small regex engine: the syntax POSIX ERE has, as FileCheck's `{{...}}` reads it (llvm::Regex is an ERE).

#[derive(Clone, Debug)]
enum Node {
    Char(u8),
    Any,
    Class(Vec<(u8, u8)>, bool),
    Start,
    End,
    Group(Box<Node>),
    Alt(Vec<Node>),
    Seq(Vec<Node>),
    Repeat(Box<Node>, usize, Option<usize>),
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn alt(&mut self) -> Result<Node, String> {
        let mut branches = vec![self.seq()?];
        while self.i < self.s.len() && self.s[self.i] == b'|' {
            self.i += 1;
            branches.push(self.seq()?);
        }
        Ok(if branches.len() == 1 {
            branches.remove(0)
        } else {
            Node::Alt(branches)
        })
    }

    fn seq(&mut self) -> Result<Node, String> {
        let mut items = Vec::new();
        while self.i < self.s.len() && self.s[self.i] != b'|' && self.s[self.i] != b')' {
            let atom = self.atom()?;
            items.push(self.repeat(atom)?);
        }
        Ok(Node::Seq(items))
    }

    fn repeat(&mut self, atom: Node) -> Result<Node, String> {
        let mut node = atom;
        while self.i < self.s.len() {
            let (lo, hi) = match self.s[self.i] {
                b'*' => (0, None),
                b'+' => (1, None),
                b'?' => (0, Some(1)),
                b'{' => {
                    let close = self.s[self.i..]
                        .iter()
                        .position(|&c| c == b'}')
                        .ok_or("regex: unclosed {")?;
                    let body = std::str::from_utf8(&self.s[self.i + 1..self.i + close])
                        .map_err(|e| e.to_string())?;
                    self.i += close;
                    let (a, b) = match body.split_once(',') {
                        Some((a, "")) => (a.parse().map_err(|_| "regex: bad bound")?, None),
                        Some((a, b)) => (
                            a.parse().map_err(|_| "regex: bad bound")?,
                            Some(b.parse().map_err(|_| "regex: bad bound")?),
                        ),
                        None => {
                            let n = body.parse().map_err(|_| "regex: bad bound")?;
                            (n, Some(n))
                        }
                    };
                    (a, b)
                }
                _ => break,
            };
            self.i += 1;
            node = Node::Repeat(Box::new(node), lo, hi);
        }
        Ok(node)
    }

    fn atom(&mut self) -> Result<Node, String> {
        let c = self.s[self.i];
        self.i += 1;
        Ok(match c {
            b'.' => Node::Any,
            b'^' => Node::Start,
            b'$' => Node::End,
            b'(' => {
                let inner = self.alt()?;
                if self.i >= self.s.len() || self.s[self.i] != b')' {
                    return Err("regex: unclosed (".into());
                }
                self.i += 1;
                Node::Group(Box::new(inner))
            }
            b'[' => self.class()?,
            b'\\' => {
                let e = *self.s.get(self.i).ok_or("regex: trailing \\")?;
                self.i += 1;
                match e {
                    b'd' => Node::Class(vec![(b'0', b'9')], false),
                    b's' => Node::Class(vec![(b' ', b' '), (b'\t', b'\r')], false),
                    b'w' => Node::Class(
                        vec![(b'a', b'z'), (b'A', b'Z'), (b'0', b'9'), (b'_', b'_')],
                        false,
                    ),
                    b'n' => Node::Char(b'\n'),
                    b't' => Node::Char(b'\t'),
                    other => Node::Char(other),
                }
            }
            other => Node::Char(other),
        })
    }

    fn class(&mut self) -> Result<Node, String> {
        let mut negate = false;
        if self.s.get(self.i) == Some(&b'^') {
            negate = true;
            self.i += 1;
        }
        let mut ranges = Vec::new();
        let mut first = true;
        loop {
            let c = *self.s.get(self.i).ok_or("regex: unclosed [")?;
            self.i += 1;
            if c == b']' && !first {
                break;
            }
            first = false;
            if c == b'[' && self.s.get(self.i) == Some(&b':') {
                let end = self.s[self.i..]
                    .windows(2)
                    .position(|w| w == b":]")
                    .ok_or("regex: unclosed [:")?;
                let name = &self.s[self.i + 1..self.i + end];
                self.i += end + 2;
                match name {
                    b"digit" => ranges.push((b'0', b'9')),
                    b"alpha" => ranges.extend([(b'a', b'z'), (b'A', b'Z')]),
                    b"alnum" => ranges.extend([(b'a', b'z'), (b'A', b'Z'), (b'0', b'9')]),
                    b"space" => ranges.extend([(b' ', b' '), (b'\t', b'\r')]),
                    b"xdigit" => ranges.extend([(b'0', b'9'), (b'a', b'f'), (b'A', b'F')]),
                    b"upper" => ranges.push((b'A', b'Z')),
                    b"lower" => ranges.push((b'a', b'z')),
                    _ => return Err(format!("regex: class [:{}:]", String::from_utf8_lossy(name))),
                }
                continue;
            }
            if self.s.get(self.i) == Some(&b'-') && self.s.get(self.i + 1).is_some_and(|&n| n != b']') {
                let hi = self.s[self.i + 1];
                self.i += 2;
                ranges.push((c, hi));
            } else {
                ranges.push((c, c));
            }
        }
        Ok(Node::Class(ranges, negate))
    }
}

/// The ends of every match of `node` starting at `at`, each passed to `k` until it accepts one.
fn walk(node: &Node, s: &[u8], at: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    match node {
        Node::Char(c) => at < s.len() && s[at] == *c && k(at + 1),
        Node::Any => at < s.len() && s[at] != b'\n' && k(at + 1),
        Node::Class(ranges, negate) => {
            at < s.len() && {
                let inside = ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&s[at]));
                inside != *negate && k(at + 1)
            }
        }
        Node::Start => (at == 0 || s[at - 1] == b'\n') && k(at),
        Node::End => (at == s.len() || s[at] == b'\n') && k(at),
        Node::Group(inner) => walk(inner, s, at, k),
        Node::Alt(branches) => branches.iter().any(|b| walk(b, s, at, k)),
        Node::Seq(items) => seq(items, s, at, k),
        Node::Repeat(inner, lo, hi) => repeat(inner, *lo, *hi, s, at, 0, k),
    }
}

fn seq(items: &[Node], s: &[u8], at: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    match items.split_first() {
        None => k(at),
        Some((first, rest)) => walk(first, s, at, &mut |next| seq(rest, s, next, k)),
    }
}

fn repeat(
    inner: &Node,
    lo: usize,
    hi: Option<usize>,
    s: &[u8],
    at: usize,
    count: usize,
    k: &mut dyn FnMut(usize) -> bool,
) -> bool {
    // greedy: one more first, then stop here
    if hi.is_none_or(|h| count < h)
        && walk(inner, s, at, &mut |next| {
            next != at && repeat(inner, lo, hi, s, next, count + 1, k)
        })
    {
        return true;
    }
    count >= lo && k(at)
}

/// A compiled pattern.
#[derive(Clone, Debug)]
pub struct Pattern {
    node: Node,
    literal: Option<Vec<u8>>,
}

/// Canonicalise horizontal whitespace: every run of spaces and tabs becomes one space.
pub fn canonical(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        if c == ' ' || c == '\t' {
            if !space {
                out.push(' ');
            }
            space = true;
        } else {
            out.push(c);
            space = false;
        }
    }
    out
}

impl Pattern {
    /// A FileCheck pattern: literal text with `{{regex}}` parts; its whitespace canonicalised and trimmed.
    pub fn new(text: &str) -> Result<Pattern, String> {
        let text = canonical(text.trim());
        if text.contains("[[") {
            return Err(format!("pattern variables are not modelled: {text}"));
        }
        let mut items = Vec::new();
        let mut literal = Some(Vec::new());
        let mut rest = text.as_str();
        while !rest.is_empty() {
            if let Some(after) = rest.strip_prefix("{{") {
                let end = after
                    .find("}}")
                    .ok_or_else(|| format!("unclosed {{{{ in {text}"))?;
                let mut p = Parser {
                    s: &after.as_bytes()[..end],
                    i: 0,
                };
                let node = p.alt()?;
                if p.i != p.s.len() {
                    return Err(format!("regex: unbalanced ) in {}", &after[..end]));
                }
                items.push(Node::Group(Box::new(node)));
                literal = None;
                rest = &after[end + 2..];
            } else {
                let next = rest.find("{{").unwrap_or(rest.len());
                for &b in &rest.as_bytes()[..next] {
                    items.push(Node::Char(b));
                    if let Some(l) = literal.as_mut() {
                        l.push(b);
                    }
                }
                rest = &rest[next..];
            }
        }
        Ok(Pattern {
            node: Node::Seq(items),
            literal,
        })
    }

    /// The first match in `s` at or after `from`: its start and end.
    pub fn find(&self, s: &[u8], from: usize) -> Option<(usize, usize)> {
        if let Some(lit) = &self.literal {
            if lit.is_empty() {
                return Some((from, from));
            }
            return s
                .get(from..)?
                .windows(lit.len())
                .position(|w| w == lit.as_slice())
                .map(|p| (from + p, from + p + lit.len()));
        }
        for start in from..=s.len() {
            let mut end = None;
            if walk(&self.node, s, start, &mut |e| {
                end = Some(e);
                true
            }) {
                return end.map(|e| (start, e));
            }
        }
        None
    }
}

fn newlines(s: &[u8]) -> usize {
    s.iter().filter(|&&c| c == b'\n').count()
}

/// Check `input` against `directives` (in file order); the first failure, as a message.
pub fn check(input: &str, directives: &[Directive]) -> Result<(), String> {
    if directives.is_empty() {
        return Err("no check strings found with the prefixes given".into());
    }
    let text = canonical(input);
    let s = text.as_bytes();
    let compiled = directives
        .iter()
        .map(|d| Pattern::new(&d.pattern).map(|p| (d, p)))
        .collect::<Result<Vec<_>, _>>()?;
    // the labels divide the input first
    let mut bounds = Vec::new();
    let mut at = 0;
    for (d, p) in &compiled {
        if d.kind == Kind::Label {
            let (start, end) = p
                .find(s, at)
                .ok_or_else(|| format!("CHECK-LABEL: expected string not found in input: {}", d.pattern))?;
            bounds.push((start, end));
            at = end;
        }
    }
    let mut label = 0;
    let mut block_end = bounds.first().map_or(s.len(), |b| b.0);
    let mut last = 0; // end of the last positive match
    let mut last_line_end = 0; // where the last match was, for -NEXT and -SAME
    let mut nots: Vec<&Pattern> = Vec::new();
    let mut dags: Vec<(&Directive, &Pattern)> = Vec::new();
    let mut dag_used: Vec<(usize, usize)> = Vec::new();

    let flush_dags = |dags: &mut Vec<(&Directive, &Pattern)>,
                      used: &mut Vec<(usize, usize)>,
                      from: usize,
                      to: usize|
     -> Result<usize, String> {
        let mut furthest = from;
        for (d, p) in dags.drain(..) {
            let mut start = from;
            loop {
                let Some((a, b)) = p.find(&s[..to], start) else {
                    return Err(format!(
                        "CHECK-DAG: expected string not found in input: {}",
                        d.pattern
                    ));
                };
                if used.iter().any(|&(x, y)| a < y && x < b || (a == x && b == y)) {
                    start = a + 1;
                    continue;
                }
                used.push((a, b));
                furthest = furthest.max(b);
                break;
            }
        }
        used.clear();
        Ok(furthest)
    };
    let check_nots = |nots: &mut Vec<&Pattern>, from: usize, to: usize| -> Result<(), String> {
        for p in nots.drain(..) {
            if let Some((a, b)) = p.find(&s[..to], from) {
                return Err(format!(
                    "CHECK-NOT: excluded string found in input: {}",
                    String::from_utf8_lossy(&s[a..b])
                ));
            }
        }
        Ok(())
    };

    for (d, p) in &compiled {
        match d.kind {
            Kind::Not => nots.push(p),
            Kind::Dag => {
                // NOTs before a DAG group apply between the last match and the group
                dags.push((d, p));
            }
            Kind::Label => {
                if !dags.is_empty() {
                    last = flush_dags(&mut dags, &mut dag_used, last, block_end)?;
                }
                check_nots(&mut nots, last, block_end)?;
                let (start, end) = bounds[label];
                let _ = start;
                last = end;
                last_line_end = end;
                label += 1;
                block_end = bounds.get(label).map_or(s.len(), |b| b.0);
            }
            Kind::Plain | Kind::Next | Kind::Same | Kind::Empty => {
                if !dags.is_empty() {
                    last = flush_dags(&mut dags, &mut dag_used, last, block_end)?;
                }
                let found = if d.kind == Kind::Empty {
                    // the line after the last match's is empty
                    let eol = s[last..block_end]
                        .iter()
                        .position(|&c| c == b'\n')
                        .map(|p| last + p);
                    match eol {
                        Some(e) if s.get(e + 1) == Some(&b'\n') => Some((e + 1, e + 1)),
                        _ => None,
                    }
                } else {
                    p.find(&s[..block_end], last)
                };
                let (a, b) = found.ok_or_else(|| {
                    format!(
                        "CHECK{}: expected string not found in input: {}",
                        match d.kind {
                            Kind::Next => "-NEXT",
                            Kind::Same => "-SAME",
                            Kind::Empty => "-EMPTY",
                            _ => "",
                        },
                        d.pattern
                    )
                })?;
                match d.kind {
                    Kind::Next if newlines(&s[last_line_end..a]) != 1 => {
                        return Err(format!(
                            "CHECK-NEXT: is not on the line after the previous match: {}",
                            d.pattern
                        ));
                    }
                    Kind::Same if newlines(&s[last_line_end..a]) != 0 => {
                        return Err(format!(
                            "CHECK-SAME: is not on the same line as the previous match: {}",
                            d.pattern
                        ));
                    }
                    _ => {}
                }
                check_nots(&mut nots, last, a)?;
                last = b;
                last_line_end = b;
            }
        }
    }
    if !dags.is_empty() {
        last = flush_dags(&mut dags, &mut dag_used, last, block_end)?;
    }
    check_nots(&mut nots, last, block_end)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(kind: Kind, p: &str) -> Directive {
        Directive {
            kind,
            pattern: p.into(),
        }
    }

    #[test]
    fn whitespace_and_order() {
        let input = "- Name:            Out\n  Data:            [ 4, 16, 36, 64 ]\n";
        assert!(
            check(
                input,
                &[
                    d(Kind::Plain, "Name: Out"),
                    d(Kind::Plain, "Data: [ 4, 16, 36, 64 ]")
                ]
            )
            .is_ok()
        );
        assert!(
            check(
                input,
                &[
                    d(Kind::Plain, "Name: Out"),
                    d(Kind::Next, "Data: [ 4, 16, 36, 64 ]")
                ]
            )
            .is_ok()
        );
        assert!(check(input, &[d(Kind::Plain, "Data:"), d(Kind::Plain, "Name: Out")]).is_err());
        assert!(check(input, &[d(Kind::Plain, "Name"), d(Kind::Same, "Out")]).is_ok());
        assert!(
            check(
                input,
                &[d(Kind::Plain, "Name"), d(Kind::Not, "Data"), d(Kind::Plain, "36")]
            )
            .is_err()
        );
        assert!(check(input, &[d(Kind::Not, "0x7FC"), d(Kind::Plain, "Data")]).is_ok());
    }

    #[test]
    fn regexes() {
        assert!(check("pi 3.14159\n", &[d(Kind::Plain, "pi {{3.14.*}}")]).is_ok());
        assert!(check("arch aarch64\n", &[d(Kind::Plain, "arch {{(arm64|aarch64)}}")]).is_ok());
        assert!(check("arch x86\n", &[d(Kind::Plain, "arch {{(arm64|aarch64)}}")]).is_err());
        let p = Pattern::new("{{[0-9]+}}x").expect("compiles");
        assert_eq!(p.find(b"ab 123x", 0), Some((3, 7)));
    }

    #[test]
    fn labels_and_dags() {
        let input = "Name: A\nData: [ 1 ]\nName: B\nData: [ 2 ]\n";
        assert!(
            check(
                input,
                &[
                    d(Kind::Label, "Name: A"),
                    d(Kind::Plain, "[ 1 ]"),
                    d(Kind::Label, "Name: B"),
                    d(Kind::Plain, "[ 2 ]")
                ]
            )
            .is_ok()
        );
        assert!(
            check(
                input,
                &[
                    d(Kind::Label, "Name: A"),
                    d(Kind::Plain, "[ 2 ]"),
                    d(Kind::Label, "Name: B")
                ]
            )
            .is_err()
        );
        assert!(check(input, &[d(Kind::Dag, "[ 2 ]"), d(Kind::Dag, "[ 1 ]")]).is_ok());
    }
}
