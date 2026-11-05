//! The text form of a function, for fixtures, diffs and notes: one line per node, indented by construct, values as
//! `%n`, variables by name, labels as `'n`. The first line names the schema and its version, bumped when the form
//! changes.

use alloc::string::String;
use core::fmt::Write;

use crate::{
    Binary, Body, End, Function, Level, Node, Op, Precision, Region, Scalar, Target, Terminator, Type, Unary,
    ValueId,
};

/// The schema line every text form starts with.
pub const SCHEMA: &str = "iridia-ir 1";

fn list(out: &mut String, vs: &[ValueId]) {
    for (i, v) in vs.iter().enumerate() {
        let sep = if i == 0 { "" } else { ", " };
        let _ = write!(out, "{sep}%{}", v.0);
    }
}

/// A literal's lanes in the type's own notation: floats as their exact bits where the decimal form would not give
/// them back (`f32:0x7fc00000`), otherwise as decimals.
fn literal(out: &mut String, ty: Type, bits: &[u64; 4]) {
    let lanes = usize::from(ty.lanes);
    let _ = write!(out, "{ty}(");
    for (i, b) in bits.iter().take(lanes).enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = match ty.scalar {
            Scalar::F32 => {
                let x = f32::from_bits(u32::try_from(*b).unwrap_or(0));
                if x.is_finite() { write!(out, "{x:?}") } else { write!(out, "0x{b:08x}") }
            }
            Scalar::F64 => {
                let x = f64::from_bits(*b);
                if x.is_finite() { write!(out, "{x:?}") } else { write!(out, "0x{b:016x}") }
            }
            Scalar::Bool => write!(out, "{}", *b != 0),
            s if s.is_signed() => {
                // sign-extend from the type's width
                let shift = 64u32.saturating_sub(s.bits());
                write!(out, "{}", (b.cast_signed() << shift) >> shift)
            }
            _ => write!(out, "{b}"),
        };
    }
    out.push(')');
}

fn unary(u: &Unary) -> String {
    match u {
        Unary::FRound(r) => alloc::format!("fround.{r:?}"),
        Unary::Saturate(n) => alloc::format!("saturate.{n:?}"),
        other => alloc::format!("{other:?}"),
    }
    .to_lowercase()
}

fn binary(b: &Binary) -> String {
    match b {
        Binary::FMul(z) => alloc::format!("fmul.{z:?}"),
        Binary::FMin(n) => alloc::format!("fmin.{n:?}"),
        Binary::FMax(n) => alloc::format!("fmax.{n:?}"),
        Binary::FCmp(c) => alloc::format!("fcmp.{c:?}"),
        Binary::SCmp(c) => alloc::format!("scmp.{c:?}"),
        Binary::UCmp(c) => alloc::format!("ucmp.{c:?}"),
        Binary::UDiv(z) => alloc::format!("udiv.{z:?}"),
        Binary::URem(z) => alloc::format!("urem.{z:?}"),
        other => alloc::format!("{other:?}"),
    }
    .to_lowercase()
}

fn op(out: &mut String, f: &Function, ty: Type, o: &Op) {
    let _ = match o {
        Op::Literal(bits) => {
            literal(out, ty, bits);
            Ok(())
        }
        Op::Undef => write!(out, "undef"),
        Op::Param => write!(out, "param"),
        Op::Unary(u, a) => write!(out, "{} %{}", unary(u), a.0),
        Op::Binary(b, x, y) => write!(out, "{} %{}, %{}", binary(b), x.0, y.0),
        Op::Select { cond, accept, reject } => write!(out, "select %{}, %{}, %{}", cond.0, accept.0, reject.0),
        Op::Compose(parts) => {
            out.push_str("compose ");
            list(out, parts);
            Ok(())
        }
        Op::Swizzle(a, lanes) => {
            let names = lanes.iter().take(usize::from(ty.lanes)).map(|&l| match l {
                0 => 'x',
                1 => 'y',
                2 => 'z',
                3 => 'w',
                _ => '?',
            });
            write!(out, "%{}.{}", a.0, names.collect::<String>())
        }
        Op::Splat(a) => write!(out, "splat %{}", a.0),
        Op::Dot(x, y) => write!(out, "dot %{}, %{}", x.0, y.0),
        Op::Convert(a, rule) => write!(out, "convert.{} %{}", alloc::format!("{rule:?}").to_lowercase(), a.0),
        Op::Bitcast(a) => write!(out, "bitcast %{}", a.0),
        Op::LoadInput(slot) => write!(out, "input {slot}"),
        Op::LoadVar(v) => match f.vars.get(v.0 as usize) {
            Some(var) => write!(out, "load {}", var.name),
            None => write!(out, "load ?{}", v.0),
        },
        Op::LoadBuffer { resource, index, stride, offset, oob } => {
            let _ = write!(out, "buffer r{}", resource.0);
            if let Some(i) = index {
                let _ = write!(out, "[%{} * {stride}]", i.0);
            }
            write!(out, " + {offset} oob.{}", alloc::format!("{oob:?}").to_lowercase())
        }
        Op::Sample { texture, sampler, coord, level } => {
            let _ = write!(out, "sample r{}, r{}, %{}", texture.0, sampler.0, coord.0);
            match level {
                Level::Auto => Ok(()),
                Level::Zero => write!(out, " level 0"),
                Level::Bias(x) => write!(out, " bias %{}", x.0),
                Level::Exact(x) => write!(out, " level %{}", x.0),
            }
        }
    };
}

struct Printer<'a> {
    f: &'a Function,
    out: String,
    depth: usize,
}

impl Printer<'_> {
    fn indent(&mut self) {
        for _ in 0..self.depth {
            self.out.push_str("  ");
        }
    }
    fn value(&mut self, v: ValueId) {
        self.indent();
        match self.f.value(v) {
            Some(d) => {
                let pp = if d.precision == Precision::AtLeast16 { " pp" } else { "" };
                let _ = write!(self.out, "%{}: {}{pp} = ", v.0, d.ty);
                op(&mut self.out, self.f, d.ty, &d.op);
            }
            None => {
                let _ = write!(self.out, "%{} = ?", v.0);
            }
        }
        self.out.push('\n');
    }
    fn params(&mut self, vs: &[ValueId]) {
        for (i, v) in vs.iter().enumerate() {
            let sep = if i == 0 { "" } else { ", " };
            let ty = self.f.ty(*v).map(|t| alloc::format!("{t}")).unwrap_or_default();
            let _ = write!(self.out, "{sep}%{}: {ty}", v.0);
        }
    }
    fn region(&mut self, r: &Region) {
        self.depth = self.depth.saturating_add(1);
        for n in &r.nodes {
            self.node(n);
        }
        self.indent();
        let o = &mut self.out;
        match &r.end {
            None => o.push_str("<open>"),
            Some(End::Yield(vs)) => {
                o.push_str("yield ");
                list(o, vs);
            }
            Some(End::Br(l, vs)) => {
                let _ = write!(o, "br '{} ", l.0);
                list(o, vs);
            }
            Some(End::Continue(l, vs)) => {
                let _ = write!(o, "continue '{} ", l.0);
                list(o, vs);
            }
            Some(End::Return) => o.push_str("return"),
            Some(End::Kill) => o.push_str("kill"),
            Some(End::Unreachable) => o.push_str("unreachable"),
        }
        let trimmed = o.trim_end().len();
        o.truncate(trimmed);
        o.push('\n');
        self.depth = self.depth.saturating_sub(1);
    }
    fn close(&mut self, results: &[ValueId]) {
        self.indent();
        self.out.push('}');
        if !results.is_empty() {
            self.out.push_str(" -> ");
            self.params(results);
        }
        self.out.push('\n');
    }
    fn node(&mut self, n: &Node) {
        match n {
            Node::Value(v) => self.value(*v),
            Node::StoreVar(var, v) => {
                self.indent();
                let name = self.f.vars.get(var.0 as usize).map_or("?", |d| d.name.as_str());
                let _ = writeln!(self.out, "{name} = %{}", v.0);
            }
            Node::StoreOutput(slot, v) => {
                self.indent();
                let _ = writeln!(self.out, "output {slot} = %{}", v.0);
            }
            Node::If { cond, accept, reject, results } => {
                self.indent();
                let _ = writeln!(self.out, "if %{} {{", cond.0);
                self.region(accept);
                self.indent();
                self.out.push_str("} else {\n");
                self.region(reject);
                self.close(results);
            }
            Node::Block { label, body, results } => {
                self.indent();
                let _ = writeln!(self.out, "block '{} {{", label.0);
                self.region(body);
                self.close(results);
            }
            Node::Loop { label, params, inits, body, results } => {
                self.indent();
                let _ = write!(self.out, "loop '{} (", label.0);
                self.params(params);
                self.out.push_str(") = (");
                list(&mut self.out, inits);
                self.out.push_str(") {\n");
                self.region(body);
                self.close(results);
            }
        }
    }
    fn target(&mut self, t: &Target) {
        let _ = write!(self.out, "b{}(", t.block);
        list(&mut self.out, &t.args);
        self.out.push(')');
    }
}

/// The text form of a function.
pub fn print(f: &Function) -> String {
    let mut p = Printer { f, out: String::new(), depth: 0 };
    let _ = writeln!(p.out, "{SCHEMA}\nfn {} {{", f.name);
    for var in &f.vars {
        let _ = writeln!(p.out, "  var {}: {}", var.name, var.ty);
    }
    match &f.body {
        Body::Regions(r) => p.region(r),
        Body::Cfg(blocks) => {
            for (i, b) in blocks.iter().enumerate() {
                let _ = write!(p.out, "  b{i}(");
                p.params(&b.params);
                p.out.push_str("):\n");
                p.depth = 2;
                for n in &b.nodes {
                    p.node(n);
                }
                p.indent();
                match &b.term {
                    Terminator::Jump(t) => {
                        p.out.push_str("jump ");
                        p.target(t);
                    }
                    Terminator::Branch { cond, accept, reject } => {
                        let _ = write!(p.out, "branch %{}, ", cond.0);
                        p.target(accept);
                        p.out.push_str(", ");
                        p.target(reject);
                    }
                    Terminator::Return => p.out.push_str("return"),
                    Terminator::Kill => p.out.push_str("kill"),
                    Terminator::Unreachable => p.out.push_str("unreachable"),
                }
                p.out.push('\n');
                p.depth = 0;
            }
        }
    }
    p.out.push_str("}\n");
    p.out
}
