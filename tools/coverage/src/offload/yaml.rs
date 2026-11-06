//! LLVM's YAML writer (`llvm::yaml::Output`, llvm/lib/Support/YAMLTraits.cpp) as the offloader prints a pipeline with
//! it: the same state machine, so a test's FileCheck lines read the same text. The calls mirror `Output`'s: a mapping
//! is `begin_mapping`, a `key` per field (which says whether the field is written) and `end_mapping`; a sequence is
//! `begin_sequence` or `begin_flow_sequence`, an element at a time.

/// Where the writer is, as `Output::InState`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    SeqFirst,
    SeqOther,
    FlowSeqFirst,
    FlowSeqOther,
    MapFirst,
    MapOther,
}

fn in_seq(s: State) -> bool {
    matches!(s, State::SeqFirst | State::SeqOther)
}

fn in_flow_seq(s: State) -> bool {
    matches!(s, State::FlowSeqFirst | State::FlowSeqOther)
}

/// The column past which a flow sequence wraps (`Output`'s default `WrapColumn`).
const WRAP: usize = 70;

/// A YAML document being written.
#[derive(Debug, Default)]
pub struct Output {
    out: String,
    column: usize,
    stack: Vec<State>,
    padding: String,
    padding_before_container: String,
    need_flow_comma: bool,
    column_at_flow_start: usize,
}

/// How a scalar is quoted (`QuotingType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Quote {
    /// Plain.
    None,
    /// In single quotes.
    Single,
    /// In double quotes.
    Double,
}

/// `yaml::isNumeric`.
fn is_numeric(s: &str) -> bool {
    let b = s.as_bytes();
    if b.is_empty() {
        return false;
    }
    let (sign, rest) = match b[0] {
        b'+' | b'-' => (true, &s[1..]),
        _ => (false, s),
    };
    if matches!(rest, ".inf" | ".Inf" | ".INF") {
        return true;
    }
    if !sign && matches!(s, ".nan" | ".NaN" | ".NAN") {
        return true;
    }
    let digits = |t: &str, ok: fn(u8) -> bool| t.len() > 2 && t.bytes().skip(2).all(ok);
    if !sign && rest.len() > 2 {
        match &rest[..2] {
            "0o" => return digits(rest, |c| (b'0'..=b'7').contains(&c)),
            "0x" => return digits(rest, |c| c.is_ascii_hexdigit()),
            _ => {}
        }
    }
    // [-+]? (\. [0-9]+ | [0-9]+ (\. [0-9]* )?) ( [eE] [-+]? [0-9]+ )?
    let r = rest.as_bytes();
    let mut i = 0;
    let int_start = i;
    while i < r.len() && r[i].is_ascii_digit() {
        i += 1;
    }
    let int_digits = i - int_start;
    if i < r.len() && r[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < r.len() && r[i].is_ascii_digit() {
            i += 1;
        }
        if int_digits == 0 && i == frac_start {
            return false;
        }
    } else if int_digits == 0 {
        return false;
    }
    if i == r.len() {
        return true;
    }
    if r[i] == b'e' || r[i] == b'E' {
        i += 1;
        if i < r.len() && (r[i] == b'+' || r[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < r.len() && r[i].is_ascii_digit() {
            i += 1;
        }
        return i == r.len() && i > exp_start;
    }
    false
}

/// `yaml::needsQuotes` for a string scalar (`ForcePreserveAsString`), or for a key when `as_string` is false.
pub fn needs_quotes(s: &str, as_string: bool) -> Quote {
    if s.is_empty() {
        return Quote::Single;
    }
    let mut q = Quote::None;
    let b = s.as_bytes();
    let space = |c: u8| matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c);
    if space(b[0]) || space(b[b.len() - 1]) {
        q = Quote::Single;
    }
    if as_string {
        let null = matches!(s, "null" | "Null" | "NULL" | "~");
        let boolean = matches!(s, "true" | "True" | "TRUE" | "false" | "False" | "FALSE");
        if null || boolean || is_numeric(s) {
            q = Quote::Single;
        }
    }
    if br#"-?:\,[]{}#&*!|>'"%@`"#.contains(&b[0]) {
        q = Quote::Single;
    }
    for &c in b {
        if c.is_ascii_alphanumeric() {
            continue;
        }
        match c {
            b'_' | b'-' | b'^' | b'.' | b',' | b' ' | 0x09 => {}
            0x0a | 0x0d | 0x7f => return Quote::Double,
            _ => {
                if c <= 0x1f || c & 0x80 != 0 {
                    return Quote::Double;
                }
                q = Quote::Single;
            }
        }
    }
    q
}

impl Output {
    /// A new document writer.
    pub fn new() -> Output {
        Output::default()
    }

    /// The text written.
    pub fn finish(self) -> String {
        self.out
    }

    fn output(&mut self, s: &str) {
        self.column += s.len();
        self.out.push_str(s);
    }

    fn output_quoted(&mut self, s: &str, q: Quote) {
        match q {
            Quote::None => self.output(s),
            Quote::Single => {
                self.output("'");
                self.output(&s.replace('\'', "''"));
                self.output("'");
            }
            Quote::Double => {
                self.output("\"");
                let mut e = String::new();
                for c in s.chars() {
                    match c {
                        '\\' => e.push_str("\\\\"),
                        '"' => e.push_str("\\\""),
                        '\n' => e.push_str("\\n"),
                        '\r' => e.push_str("\\r"),
                        '\t' => e.push_str("\\t"),
                        '\0' => e.push_str("\\0"),
                        c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                            e.push_str(&format!("\\x{:02X}", c as u32))
                        }
                        c => e.push(c),
                    }
                }
                self.output(&e);
                self.output("\"");
            }
        }
    }

    fn output_up_to_end_of_line(&mut self, s: &str) {
        self.output(s);
        if self.stack.last().is_none_or(|&t| !in_flow_seq(t)) {
            self.padding = "\n".into();
        }
    }

    fn output_new_line(&mut self) {
        self.out.push('\n');
        self.column = 0;
    }

    fn new_line_check(&mut self, empty_sequence: bool) {
        if self.padding != "\n" {
            let p = std::mem::take(&mut self.padding);
            self.output(&p);
            return;
        }
        self.output_new_line();
        self.padding.clear();
        if self.stack.is_empty() || empty_sequence {
            return;
        }
        let mut indent = self.stack.len() - 1;
        let mut nested = false;
        let mut it = self.stack.iter().rev().peekable();
        let top = **it.peek().expect("non-empty");
        if in_seq(top) {
            nested = true;
            indent += 1;
        } else if top == State::MapFirst || in_flow_seq(top) {
            nested = true;
            it.next();
        }
        let mut dashes = 0;
        if nested {
            for &s in it {
                if !in_seq(s) {
                    break;
                }
                dashes += 1;
                if s != State::SeqFirst {
                    break;
                }
            }
        }
        for _ in dashes..indent {
            self.output("  ");
        }
        for _ in 0..dashes {
            self.output("- ");
        }
    }

    fn padded_key(&mut self, key: &str) {
        self.output_quoted(key, needs_quotes(key, false));
        self.output(":");
        const SPACES: usize = 16;
        self.padding = if key.len() < SPACES {
            " ".repeat(SPACES - key.len())
        } else {
            " ".into()
        };
    }

    /// `beginDocuments`.
    pub fn begin_documents(&mut self) {
        self.output_up_to_end_of_line("---");
    }

    /// `endDocuments`.
    pub fn end_documents(&mut self) {
        self.output("\n...\n");
    }

    /// `beginMapping`.
    pub fn begin_mapping(&mut self) {
        self.stack.push(State::MapFirst);
        self.padding_before_container = std::mem::replace(&mut self.padding, "\n".into());
    }

    /// `endMapping`.
    pub fn end_mapping(&mut self) {
        if self.stack.last() == Some(&State::MapFirst) {
            self.padding = self.padding_before_container.clone();
            self.new_line_check(false);
            self.output("{}");
            self.padding = "\n".into();
        }
        self.stack.pop();
    }

    /// `preflightKey`: whether the field is written (it is when required or not its default).
    pub fn key(&mut self, key: &str, required: bool, same_as_default: bool) -> bool {
        if required || !same_as_default {
            self.new_line_check(false);
            self.padded_key(key);
            return true;
        }
        false
    }

    /// `postflightKey`.
    pub fn post_key(&mut self) {
        if self.stack.last() == Some(&State::MapFirst) {
            self.stack.pop();
            self.stack.push(State::MapOther);
        }
    }

    /// `canElideEmptySequence`.
    pub fn can_elide_empty_sequence(&self) -> bool {
        let n = self.stack.len();
        if n < 2 || self.stack[n - 1] != State::MapFirst {
            return true;
        }
        !in_seq(self.stack[n - 2])
    }

    /// `beginSequence`.
    pub fn begin_sequence(&mut self) {
        self.stack.push(State::SeqFirst);
        self.padding_before_container = std::mem::replace(&mut self.padding, "\n".into());
    }

    /// `endSequence`.
    pub fn end_sequence(&mut self) {
        if self.stack.last() == Some(&State::SeqFirst) {
            self.padding = self.padding_before_container.clone();
            self.new_line_check(true);
            self.output("[]");
            self.padding = "\n".into();
        }
        self.stack.pop();
    }

    /// `postflightElement`.
    pub fn post_element(&mut self) {
        match self.stack.last() {
            Some(State::SeqFirst) => {
                self.stack.pop();
                self.stack.push(State::SeqOther);
            }
            Some(State::FlowSeqFirst) => {
                self.stack.pop();
                self.stack.push(State::FlowSeqOther);
            }
            _ => {}
        }
    }

    /// `beginFlowSequence`.
    pub fn begin_flow_sequence(&mut self) {
        self.stack.push(State::FlowSeqFirst);
        self.new_line_check(false);
        self.column_at_flow_start = self.column;
        self.output("[ ");
        self.need_flow_comma = false;
    }

    /// `endFlowSequence`.
    pub fn end_flow_sequence(&mut self) {
        self.stack.pop();
        self.output_up_to_end_of_line(" ]");
    }

    /// `preflightFlowElement`.
    pub fn pre_flow_element(&mut self) {
        if self.need_flow_comma {
            self.output(", ");
        }
        if self.column > WRAP {
            self.output("\n");
            for _ in 0..self.column_at_flow_start {
                self.output(" ");
            }
            self.column = self.column_at_flow_start;
            self.output("  ");
        }
    }

    /// `postflightFlowElement`.
    pub fn post_flow_element(&mut self) {
        self.need_flow_comma = true;
    }

    /// `scalarString`.
    pub fn scalar(&mut self, s: &str, q: Quote) {
        self.new_line_check(false);
        if s.is_empty() {
            self.output_up_to_end_of_line("''");
            return;
        }
        self.output_quoted(s, q);
        self.output_up_to_end_of_line("");
    }

    /// An enumeration's name (`matchEnumScalar`).
    pub fn enum_scalar(&mut self, s: &str) {
        self.new_line_check(false);
        self.output_up_to_end_of_line(s);
    }

    // The forms the offloader's mappings use.

    /// A required plain scalar field.
    pub fn field(&mut self, key: &str, value: &str) {
        if self.key(key, true, false) {
            self.scalar(value, Quote::None);
            self.post_key();
        }
    }

    /// A string field (`std::string`), quoted as `needsQuotes` says; `optional` fields are written anyway (an
    /// `mapOptional` of a string without a default is never elided).
    pub fn string(&mut self, key: &str, value: &str) {
        if self.key(key, true, false) {
            self.scalar(value, needs_quotes(value, true));
            self.post_key();
        }
    }

    /// An optional plain scalar field with a default: written when it is not the default.
    pub fn defaulted(&mut self, key: &str, value: &str, is_default: bool) {
        if self.key(key, false, is_default) {
            self.scalar(value, Quote::None);
            self.post_key();
        }
    }

    /// An enumeration field.
    pub fn enumeration(&mut self, key: &str, value: &str, required: bool, is_default: bool) {
        if self.key(key, required, is_default) {
            self.enum_scalar(value);
            self.post_key();
        }
    }

    /// A flow sequence of scalars as a field's value.
    pub fn flow_values(&mut self, values: &[String]) {
        self.begin_flow_sequence();
        for v in values {
            self.pre_flow_element();
            self.scalar(v, Quote::None);
            self.post_flow_element();
            self.post_element();
        }
        self.end_flow_sequence();
    }
}

/// C's `printf("%g", v)` (what `ScalarTraits<float>` and `<double>` write).
pub fn format_g(v: f64) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan" } else { "nan" }.into();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf" } else { "inf" }.into();
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0" } else { "0" }.into();
    }
    const P: i32 = 6;
    // the exponent of the value as %e would print it with P - 1 digits after the point
    let e_form = format!("{:.*e}", (P - 1) as usize, v);
    let (mantissa, exp) = e_form.split_once('e').expect("exponent");
    let x: i32 = exp.parse().expect("exponent digits");
    if !(-4..P).contains(&x) {
        let m = trim_zeros(mantissa);
        let sign = if x < 0 { '-' } else { '+' };
        format!("{m}e{sign}{:02}", x.abs())
    } else {
        let digits = (P - 1 - x).max(0) as usize;
        trim_zeros(&format!("{v:.digits$}")).to_string()
    }
}

fn trim_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_matches_printf() {
        for (v, s) in [
            (4.0, "4"),
            (0.5, "0.5"),
            (1e-5, "1e-05"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (f64::from(f32::MAX), "3.40282e+38"),
            (0.0001, "0.0001"),
            (-2.5, "-2.5"),
            (1.0 / 3.0, "0.333333"),
            (100.0, "100"),
        ] {
            assert_eq!(format_g(v), s, "{v}");
        }
    }

    #[test]
    fn a_mapping_with_a_flow_sequence() {
        let mut o = Output::new();
        o.begin_documents();
        o.begin_mapping();
        if o.key("Buffers", true, false) {
            o.begin_sequence();
            o.begin_mapping();
            o.string("Name", "Out");
            o.enumeration("Format", "Float32", true, false);
            if o.key("Data", true, false) {
                o.flow_values(&["4".into(), "16".into()]);
                o.post_key();
            }
            o.end_mapping();
            o.post_element();
            o.end_sequence();
            o.post_key();
        }
        if o.key("Bindings", false, false) {
            o.begin_mapping();
            o.string("VertexBuffer", "");
            o.end_mapping();
            o.post_key();
        }
        if o.key("Settings", false, false) {
            o.begin_mapping();
            o.end_mapping();
            o.post_key();
        }
        o.end_mapping();
        o.end_documents();
        assert_eq!(
            o.finish(),
            "---\nBuffers:\n  - Name:            Out\n    Format:          Float32\n    Data:            [ 4, 16 ]\n\
             Bindings:\n  VertexBuffer:    ''\nSettings:        {}\n...\n"
        );
    }

    #[test]
    fn quoting() {
        assert_eq!(needs_quotes("Out", true), Quote::None);
        assert_eq!(needs_quotes("1", true), Quote::Single);
        assert_eq!(needs_quotes("true", true), Quote::Single);
        assert_eq!(needs_quotes("a/b", true), Quote::Single);
        assert_eq!(needs_quotes("main", true), Quote::None);
    }
}
