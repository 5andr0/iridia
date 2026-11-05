//! The text form read back: [`parse`] is the inverse of [`print`](super::print) for everything the text states.
//! What it does not state comes back empty: origins, and values the body never places (an id the text skips is an
//! unplaced `undef`). Printing what `parse` gives gives the same text.

use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt;

use super::{SCHEMA, binary, unary};
use crate::{
    Binary, Block, Body, Cmp, End, FloatToInt, Function, LabelId, Level, MinMaxNan, MulZero, Node, OnZero,
    Op, OutOfBounds, Precision, Region, Rounding, SaturateNan, Scalar, Target, Terminator, Type, Unary,
    ValueDef, ValueId, Var, VarId, interface::ResourceId,
};

/// Why a text cannot be read: the line (from 1) and what is wrong there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextError {
    /// The line, counting from 1; 0 for the text as a whole.
    pub line: usize,
    /// What is wrong.
    pub message: String,
}

impl fmt::Display for TextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl core::error::Error for TextError {}

type Result<T> = core::result::Result<T, TextError>;

const ROUNDINGS: [Rounding; 5] = [
    Rounding::Floor,
    Rounding::Ceil,
    Rounding::Trunc,
    Rounding::NearestEven,
    Rounding::NearestAway,
];
const CMPS: [Cmp; 6] = [Cmp::Eq, Cmp::Ne, Cmp::Lt, Cmp::Le, Cmp::Gt, Cmp::Ge];
const NANS: [MinMaxNan; 3] = [MinMaxNan::MinNum, MinMaxNan::Propagate, MinMaxNan::Any];
const ON_ZERO: [OnZero; 3] = [OnZero::AllOnes, OnZero::Dividend, OnZero::Undef];

/// Every unary operation, for matching a name against the name the printer gives it.
fn unaries() -> Vec<Unary> {
    let mut all = alloc::vec![
        Unary::FNeg,
        Unary::FAbs,
        Unary::FFract,
        Unary::FSqrt,
        Unary::FInverseSqrt,
        Unary::FExp2,
        Unary::FLog2,
        Unary::FSin,
        Unary::FCos,
        Unary::Saturate(SaturateNan::Zero),
        Unary::Saturate(SaturateNan::Any),
        Unary::Not,
        Unary::INeg,
        Unary::Any,
        Unary::All,
        Unary::IsNan,
    ];
    all.extend(ROUNDINGS.map(Unary::FRound));
    all
}

/// Every binary operation, likewise.
fn binaries() -> Vec<Binary> {
    let mut all = alloc::vec![
        Binary::FAdd,
        Binary::FSub,
        Binary::FMul(MulZero::Ieee),
        Binary::FMul(MulZero::ZeroWins),
        Binary::FDiv,
        Binary::FPow,
        Binary::IAdd,
        Binary::ISub,
        Binary::IMul,
        Binary::SMin,
        Binary::SMax,
        Binary::And,
        Binary::Or,
        Binary::Xor,
    ];
    all.extend(NANS.map(Binary::FMin));
    all.extend(NANS.map(Binary::FMax));
    all.extend(CMPS.map(Binary::FCmp));
    all.extend(CMPS.map(Binary::SCmp));
    all.extend(CMPS.map(Binary::UCmp));
    all.extend(ON_ZERO.map(Binary::UDiv));
    all.extend(ON_ZERO.map(Binary::URem));
    all
}

fn lane_of(c: char) -> Option<u8> {
    match c {
        'x' => Some(0),
        'y' => Some(1),
        'z' => Some(2),
        'w' => Some(3),
        _ => None,
    }
}

struct Parser<'a> {
    lines: Vec<(usize, &'a str)>,
    at: usize,
    values: Vec<Option<ValueDef>>,
    vars: Vec<Var>,
    var_names: BTreeMap<String, VarId>,
}

impl<'a> Parser<'a> {
    fn line_no(&self) -> usize {
        self.lines.get(self.at.saturating_sub(1)).map_or(0, |l| l.0)
    }
    fn fail<T>(&self, message: impl Into<String>) -> Result<T> {
        Err(TextError {
            line: self.line_no(),
            message: message.into(),
        })
    }
    fn next(&mut self) -> Result<&'a str> {
        let line = self.lines.get(self.at).map(|l| l.1);
        self.at = self.at.saturating_add(1);
        match line {
            Some(l) => Ok(l),
            None => self.fail("the text ends inside the function"),
        }
    }
    fn peek(&self) -> Option<&'a str> {
        self.lines.get(self.at).map(|l| l.1)
    }

    fn ty(&self, s: &str) -> Result<Type> {
        let s = s.trim();
        let t = match s.strip_prefix("vec").and_then(|r| r.split_once('<')) {
            Some((n, rest)) => {
                let lanes = n.parse::<u8>().ok();
                let scalar = rest.strip_suffix('>').and_then(Scalar::parse);
                match (lanes, scalar) {
                    (Some(l), Some(sc)) => Type::vector(sc, l),
                    _ => return self.fail(format!("not a type: {s}")),
                }
            }
            None => match Scalar::parse(s) {
                Some(sc) => Type::scalar(sc),
                None => return self.fail(format!("not a type: {s}")),
            },
        };
        if t.is_valid() {
            Ok(t)
        } else {
            self.fail(format!("not a type: {s}"))
        }
    }
    fn value_id(&self, s: &str) -> Result<ValueId> {
        match s.trim().strip_prefix('%').and_then(|n| n.parse::<u32>().ok()) {
            Some(n) => Ok(ValueId(n)),
            None => self.fail(format!("not a value: {s}")),
        }
    }
    fn list(&self, s: &str) -> Result<Vec<ValueId>> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Vec::new());
        }
        s.split(',').map(|v| self.value_id(v)).collect()
    }
    fn resource(&self, s: &str) -> Result<ResourceId> {
        match s.trim().strip_prefix('r').and_then(|n| n.parse::<u32>().ok()) {
            Some(n) => Ok(ResourceId(n)),
            None => self.fail(format!("not a resource: {s}")),
        }
    }
    fn label(&self, s: &str) -> Result<LabelId> {
        match s.trim().strip_prefix('\'').and_then(|n| n.parse::<u32>().ok()) {
            Some(n) => Ok(LabelId(n)),
            None => self.fail(format!("not a label: {s}")),
        }
    }
    fn mask(&self, s: &str) -> Result<u8> {
        let mut mask = 0u8;
        let mut last = None;
        for c in s.chars() {
            let Some(l) = lane_of(c) else {
                return self.fail(format!("not lanes: {s}"));
            };
            if last.is_some_and(|p| p >= l) {
                return self.fail(format!("lanes out of order: {s}"));
            }
            last = Some(l);
            mask |= 1 << l;
        }
        if mask == 0 {
            self.fail("no lanes")
        } else {
            Ok(mask)
        }
    }
    fn define(&mut self, v: ValueId, def: ValueDef) -> Result<()> {
        let i = v.0 as usize;
        if self.values.len() <= i {
            self.values.resize(i.saturating_add(1), None);
        }
        match self.values.get_mut(i) {
            Some(slot @ None) => {
                *slot = Some(def);
                Ok(())
            }
            _ => self.fail(format!("%{} is defined twice", v.0)),
        }
    }
    /// `%n: T, %m: U`: values a construct defines.
    fn params(&mut self, s: &str) -> Result<Vec<ValueId>> {
        let s = s.trim();
        if s.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for p in s.split(", ") {
            let Some((v, t)) = p.split_once(':') else {
                return self.fail(format!("not a parameter: {p}"));
            };
            let (v, ty) = (self.value_id(v)?, self.ty(t)?);
            self.define(
                v,
                ValueDef {
                    ty,
                    op: Op::Param,
                    origin: None,
                    precision: Precision::Full,
                },
            )?;
            out.push(v);
        }
        Ok(out)
    }
    fn var(&self, name: &str) -> Result<VarId> {
        match self.var_names.get(name.trim()) {
            Some(v) => Ok(*v),
            None => self.fail(format!("no variable {}", name.trim())),
        }
    }

    fn literal(&self, t: Type, s: &str) -> Result<Op> {
        let Some(inner) = s.strip_prefix(&format!("{t}(")).and_then(|r| r.strip_suffix(')')) else {
            return self.fail(format!("a literal of another type than {t}: {s}"));
        };
        let parts: Vec<&str> = inner.split(", ").collect();
        if parts.len() != usize::from(t.lanes) {
            return self.fail(format!("{} lanes for {t}", parts.len()));
        }
        let mut bits = [0u64; 4];
        for (b, p) in bits.iter_mut().zip(&parts) {
            let hex = p.strip_prefix("0x").and_then(|h| u64::from_str_radix(h, 16).ok());
            *b = match t.scalar {
                Scalar::F32 => match hex {
                    Some(h) => h,
                    None => match p.parse::<f32>() {
                        Ok(x) => u64::from(x.to_bits()),
                        Err(_) => return self.fail(format!("not an f32: {p}")),
                    },
                },
                Scalar::F64 => match hex {
                    Some(h) => h,
                    None => match p.parse::<f64>() {
                        Ok(x) => x.to_bits(),
                        Err(_) => return self.fail(format!("not an f64: {p}")),
                    },
                },
                Scalar::Bool => match *p {
                    "true" => 1,
                    "false" => 0,
                    _ => return self.fail(format!("not a bool: {p}")),
                },
                sc if sc.is_signed() => match p.parse::<i64>() {
                    // kept in the low bits of the word, as the builder keeps them
                    Ok(x) => x.cast_unsigned() & (u64::MAX >> 64u32.saturating_sub(sc.bits())),
                    Err(_) => return self.fail(format!("not an integer: {p}")),
                },
                _ => match p.parse::<u64>() {
                    Ok(x) => x,
                    Err(_) => return self.fail(format!("not an integer: {p}")),
                },
            };
        }
        Ok(Op::Literal(bits))
    }

    fn op(&self, t: Type, s: &str) -> Result<Op> {
        let s = s.trim();
        if s.starts_with(&format!("{t}(")) {
            return self.literal(t, s);
        }
        match s {
            "undef" => return Ok(Op::Undef),
            "param" => return Ok(Op::Param),
            _ => {}
        }
        if let Some(rest) = s.strip_prefix('%') {
            // a swizzle: `%n.lanes`, in the result's lane count
            let Some((v, names)) = rest.split_once('.') else {
                return self.fail(format!("not an operation: {s}"));
            };
            let v = self.value_id(&format!("%{v}"))?;
            let mut lanes = [0u8; 4];
            if names.len() != usize::from(t.lanes) {
                return self.fail(format!("a swizzle of {} lanes to {t}", names.len()));
            }
            for (l, c) in lanes.iter_mut().zip(names.chars()) {
                *l = lane_of(c).map_or_else(|| self.fail(format!("not a lane: {c}")), Ok)?;
            }
            return Ok(Op::Swizzle(v, lanes));
        }
        let (word, rest) = s.split_once(' ').unwrap_or((s, ""));
        Ok(match word {
            "select" => match self.list(rest)?.as_slice() {
                [c, a, b] => Op::Select {
                    cond: *c,
                    accept: *a,
                    reject: *b,
                },
                _ => return self.fail("select takes three values"),
            },
            "compose" => Op::Compose(self.list(rest)?),
            "splat" => Op::Splat(self.value_id(rest)?),
            "dot" => match self.list(rest)?.as_slice() {
                [a, b] => Op::Dot(*a, *b),
                _ => return self.fail("dot takes two values"),
            },
            "convert.saturate" => Op::Convert(self.value_id(rest)?, FloatToInt::Saturate),
            "convert.undef" => Op::Convert(self.value_id(rest)?, FloatToInt::Undef),
            "bitcast" => Op::Bitcast(self.value_id(rest)?),
            "input" => {
                let Some((slot, lanes)) = rest.split_once('.') else {
                    return self.fail("input takes slot.lanes");
                };
                let slot = slot
                    .parse::<u32>()
                    .map_or_else(|_| self.fail(format!("not a slot: {slot}")), Ok)?;
                Op::LoadInput(slot, self.mask(lanes)?)
            }
            "load" => Op::LoadVar(self.var(rest)?),
            "state" => Op::LoadState(self.resource(rest)?),
            "buffer" => self.buffer(rest)?,
            "sample" => self.sample(rest)?,
            name => {
                if let Some(u) = unaries().into_iter().find(|u| unary(u) == name) {
                    Op::Unary(u, self.value_id(rest)?)
                } else if let Some(b) = binaries().into_iter().find(|b| binary(b) == name) {
                    match self.list(rest)?.as_slice() {
                        [a, c] => Op::Binary(b, *a, *c),
                        _ => return self.fail(format!("{name} takes two values")),
                    }
                } else {
                    return self.fail(format!("no operation {name}"));
                }
            }
        })
    }
    /// `rN[%i * stride] + offset oob.rule`, the index part optional.
    fn buffer(&self, s: &str) -> Result<Op> {
        let Some((head, oob)) = s.rsplit_once(" oob.") else {
            return self.fail("a buffer load without oob");
        };
        let oob = match oob {
            "zero" => OutOfBounds::Zero,
            "undef" => OutOfBounds::Undef,
            _ => return self.fail(format!("no out-of-bounds rule {oob}")),
        };
        let Some((head, offset)) = head.rsplit_once(" + ") else {
            return self.fail("a buffer load without offset");
        };
        let offset = offset
            .parse::<u32>()
            .map_or_else(|_| self.fail(format!("not an offset: {offset}")), Ok)?;
        let (resource, index, stride) = match head.split_once('[') {
            Some((r, idx)) => {
                let Some((i, stride)) = idx.strip_suffix(']').and_then(|x| x.split_once(" * ")) else {
                    return self.fail("a buffer index without a stride");
                };
                let stride = stride
                    .parse::<u32>()
                    .map_or_else(|_| self.fail(format!("not a stride: {stride}")), Ok)?;
                (self.resource(r)?, Some(self.value_id(i)?), stride)
            }
            // a fixed address states no stride: the printer writes none, so it is zero
            None => (self.resource(head)?, None, 0),
        };
        Ok(Op::LoadBuffer {
            resource,
            index,
            stride,
            offset,
            oob,
        })
    }
    /// `rT, rS, %c` and an optional `level 0`, `bias %x` or `level %x`.
    fn sample(&self, s: &str) -> Result<Op> {
        let (head, level) = match s.split_once(" level ").map(|(h, l)| (h, Some(("level", l)))) {
            Some(x) => x,
            None => match s.split_once(" bias ") {
                Some((h, b)) => (h, Some(("bias", b))),
                None => (s, None),
            },
        };
        let parts: Vec<&str> = head.split(", ").collect();
        let [texture, sampler, coord] = parts.as_slice() else {
            return self.fail("sample takes texture, sampler, coordinate");
        };
        let level = match level {
            None => Level::Auto,
            Some(("level", "0")) => Level::Zero,
            Some(("level", x)) => Level::Exact(self.value_id(x)?),
            Some((_, x)) => Level::Bias(self.value_id(x)?),
        };
        Ok(Op::Sample {
            texture: self.resource(texture)?,
            sampler: self.resource(sampler)?,
            coord: self.value_id(coord)?,
            level,
        })
    }

    /// A node line (not a construct's), or `None` for a line that is not one.
    fn plain(&mut self, line: &str) -> Result<Option<Node>> {
        if let Some(rest) = line.strip_prefix('%') {
            let Some((v, rest)) = rest.split_once(": ") else {
                return self.fail("a value without a type");
            };
            let v = self.value_id(&format!("%{v}"))?;
            let Some((ty, op)) = rest.split_once(" = ") else {
                return self.fail("a value without an operation");
            };
            let (ty, precision) = match ty.strip_suffix(" pp") {
                Some(t) => (t, Precision::AtLeast16),
                None => (ty, Precision::Full),
            };
            let ty = self.ty(ty)?;
            let op = self.op(ty, op)?;
            self.define(
                v,
                ValueDef {
                    ty,
                    op,
                    origin: None,
                    precision,
                },
            )?;
            return Ok(Some(Node::Value(v)));
        }
        if let Some(rest) = line.strip_prefix("output ") {
            let Some((at, v)) = rest.split_once(" = ") else {
                return self.fail("an output store without a value");
            };
            let Some((slot, lanes)) = at.split_once('.') else {
                return self.fail("an output store without lanes");
            };
            let slot = slot
                .parse::<u32>()
                .map_or_else(|_| self.fail(format!("not a slot: {slot}")), Ok)?;
            return Ok(Some(Node::StoreOutput(
                slot,
                self.mask(lanes)?,
                self.value_id(v)?,
            )));
        }
        if let Some((name, v)) = line.split_once(" = ") {
            return Ok(Some(Node::StoreVar(self.var(name)?, self.value_id(v)?)));
        }
        Ok(None)
    }

    /// A region's end line, or `None` for a line that is not one.
    fn end(&self, line: &str) -> Result<Option<End>> {
        let (word, rest) = line.split_once(' ').unwrap_or((line, ""));
        Ok(Some(match word {
            "yield" => End::Yield(self.list(rest)?),
            "br" | "continue" => {
                let (l, vs) = rest.split_once(' ').unwrap_or((rest, ""));
                let (l, vs) = (self.label(l)?, self.list(vs)?);
                if word == "br" {
                    End::Br(l, vs)
                } else {
                    End::Continue(l, vs)
                }
            }
            "return" if rest.is_empty() => End::Return,
            "kill" if rest.is_empty() => End::Kill,
            "unreachable" if rest.is_empty() => End::Unreachable,
            _ => return Ok(None),
        }))
    }

    /// A construct's closing line: `}` or `} -> %n: T, ...`.
    fn close(&mut self) -> Result<Vec<ValueId>> {
        let line = self.next()?;
        match line.strip_prefix('}') {
            Some("") => Ok(Vec::new()),
            Some(rest) => match rest.strip_prefix(" -> ") {
                Some(ps) => self.params(ps),
                None => self.fail(format!("not a closing line: {line}")),
            },
            None => self.fail(format!("not a closing line: {line}")),
        }
    }

    fn region(&mut self) -> Result<Region> {
        let mut r = Region::default();
        loop {
            let line = self.next()?;
            if let Some(end) = self.end(line)? {
                r.end = Some(end);
                return Ok(r);
            }
            if let Some(c) = line.strip_prefix("if ").and_then(|x| x.strip_suffix(" {")) {
                let cond = self.value_id(c)?;
                let accept = self.region()?;
                if self.next()? != "} else {" {
                    return self.fail("an if without its else");
                }
                let reject = self.region()?;
                let results = self.close()?;
                r.nodes.push(Node::If {
                    cond,
                    accept,
                    reject,
                    results,
                });
            } else if let Some(l) = line.strip_prefix("block ").and_then(|x| x.strip_suffix(" {")) {
                let label = self.label(l)?;
                let body = self.region()?;
                let results = self.close()?;
                r.nodes.push(Node::Block { label, body, results });
            } else if let Some(rest) = line.strip_prefix("loop ").and_then(|x| x.strip_suffix(") {")) {
                let Some((l, rest)) = rest.split_once(" (") else {
                    return self.fail("a loop without parameters");
                };
                let Some((ps, inits)) = rest.split_once(") = (") else {
                    return self.fail("a loop without initial values");
                };
                let label = self.label(l)?;
                let params = self.params(ps)?;
                let inits = self.list(inits)?;
                let body = self.region()?;
                let results = self.close()?;
                r.nodes.push(Node::Loop {
                    label,
                    params,
                    inits,
                    body,
                    results,
                });
            } else if let Some(n) = self.plain(line)? {
                r.nodes.push(n);
            } else {
                return self.fail(format!("not a node: {line}"));
            }
        }
    }

    fn target(&self, s: &str) -> Result<Target> {
        let Some((b, args)) = s
            .trim()
            .strip_prefix('b')
            .and_then(|x| x.strip_suffix(')'))
            .and_then(|x| x.split_once('('))
        else {
            return self.fail(format!("not a target: {s}"));
        };
        let block = b
            .parse::<u32>()
            .map_or_else(|_| self.fail(format!("not a block: {b}")), Ok)?;
        Ok(Target {
            block,
            args: self.list(args)?,
        })
    }

    fn graph(&mut self) -> Result<Vec<Block>> {
        let mut blocks = Vec::new();
        while let Some(header) = self.peek() {
            if header == "}" {
                break;
            }
            self.next()?;
            let want = format!("b{}(", blocks.len());
            let Some(ps) = header.strip_prefix(&want).and_then(|x| x.strip_suffix("):")) else {
                return self.fail(format!("not block {}'s header: {header}", blocks.len()));
            };
            let params = self.params(ps)?;
            let mut nodes = Vec::new();
            let term = loop {
                let line = self.next()?;
                if let Some(t) = line.strip_prefix("jump ") {
                    break Terminator::Jump(self.target(t)?);
                }
                if let Some(rest) = line.strip_prefix("branch ") {
                    let Some((c, rest)) = rest.split_once(", ") else {
                        return self.fail("a branch without targets");
                    };
                    let Some((a, b)) = rest.split_once("), ") else {
                        return self.fail("a branch without two targets");
                    };
                    break Terminator::Branch {
                        cond: self.value_id(c)?,
                        accept: self.target(&format!("{a})"))?,
                        reject: self.target(b)?,
                    };
                }
                match line {
                    "return" => break Terminator::Return,
                    "kill" => break Terminator::Kill,
                    "unreachable" => break Terminator::Unreachable,
                    _ => {}
                }
                match self.plain(line)? {
                    Some(n) => nodes.push(n),
                    None => return self.fail(format!("not a node: {line}")),
                }
            };
            blocks.push(Block { params, nodes, term });
        }
        Ok(blocks)
    }
}

/// The function a text form states. Refused, with its line: a text without the schema line this version writes,
/// or with anything [`print`](super::print) does not write.
pub fn parse(text: &str) -> core::result::Result<Function, TextError> {
    let lines: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .map(|(i, l)| (i.saturating_add(1), l.trim()))
        .filter(|(_, l)| !l.is_empty())
        .collect();
    let mut p = Parser {
        lines,
        at: 0,
        values: Vec::new(),
        vars: Vec::new(),
        var_names: BTreeMap::new(),
    };
    if p.next()? != SCHEMA {
        return p.fail(format!("the first line is not {SCHEMA:?}"));
    }
    let header = p.next()?;
    let Some(name) = header.strip_prefix("fn ").and_then(|x| x.strip_suffix(" {")) else {
        return p.fail("not a function header");
    };
    while let Some(rest) = p.peek().and_then(|l| l.strip_prefix("var ")) {
        p.next()?;
        let Some((n, t)) = rest.split_once(": ") else {
            return p.fail("a variable without a type");
        };
        let ty = p.ty(t)?;
        let id = VarId(u32::try_from(p.vars.len()).map_or_else(|_| p.fail("more than 2^32 variables"), Ok)?);
        if p.var_names.insert(n.to_string(), id).is_some() {
            return p.fail(format!("variable {n} is declared twice"));
        }
        p.vars.push(Var {
            name: n.to_string(),
            ty,
        });
    }
    let graph = p
        .peek()
        .is_some_and(|l| l.starts_with("b0(") && l.ends_with("):"));
    let body = if graph {
        Body::Cfg(p.graph()?)
    } else {
        Body::Regions(p.region()?)
    };
    if p.next()? != "}" {
        return p.fail("the function does not end after its body");
    }
    if let Some(extra) = p.peek() {
        p.at = p.at.saturating_add(1);
        return p.fail(format!("text after the function: {extra}"));
    }
    let undef = ValueDef {
        ty: Type::BOOL,
        op: Op::Undef,
        origin: None,
        precision: Precision::Full,
    };
    let values = p
        .values
        .into_iter()
        .map(|v| v.unwrap_or_else(|| undef.clone()))
        .collect();
    Ok(Function {
        name: name.to_string(),
        values,
        vars: p.vars,
        body,
    })
}
