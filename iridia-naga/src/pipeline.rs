//! A program's WGSL as one pipeline binds it ([`wgsl_for`]): each resource where the pipeline's [`BindingPlan`] puts
//! it, and a vertex program's outputs where the program that reads them takes them. The lowered module is changed, never
//! the text, and the program keeps its own interface.
//!
//! Two programs of one pipeline share its bind groups, so a vertex and a pixel program that both read constant buffer
//! `b0` need plans that put them apart; and WebGPU links a vertex output to a pixel input by location and type, where
//! Direct3D links them by semantic and register lanes. [`Pipeline::feeds`] gives the consumer, and each output it
//! reads takes the consumer's location, interpolation and type (as `iridia::emit::wgsl_for` does for naga-path
//! programs).

use alloc::{format, string::String, vec, vec::Vec};

use iridia_ir::interface::{Interface, Label, ScalarType, Stage, Value};
use naga::{
    Binding, Block, Expression, Handle, Span, Statement, StructMember, SwizzleComponent, Type, TypeInner,
    VectorSize,
};

use crate::io::{EntryIo, entry_io};
use crate::{BindingPlan, Error, Profile};

/// The pipeline a program's WGSL is written for.
#[derive(Clone, Debug)]
pub struct Pipeline<'a> {
    /// Where each resource of the program is bound ([`BindingPlan::d3d`] is the program's own plan).
    pub plan: BindingPlan,
    /// The program a vertex program's outputs feed. Each output the consumer reads, matched as Direct3D links stages
    /// (by semantic: the label's name without case, and its index), takes the consumer's location and interpolation
    /// and is read as the consumer's type: fewer lanes are the output's first ones, more lanes are the output's
    /// followed by zeros (Direct3D leaves them undefined; noted), another 32-bit scalar kind is the same bits. The
    /// outputs the consumer does not read take the locations after the consumer's. A consumer input no output
    /// carries is an error.
    pub feeds: Option<&'a Interface>,
}

/// A program's WGSL for a [`Pipeline`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Written {
    /// The WGSL source.
    pub text: String,
    /// The first location after every one the text's outputs (a vertex program) or inputs (a pixel program) use.
    pub next_location: u32,
    /// What the text approximates of the program, each once.
    pub notes: Vec<String>,
}

fn same_label(a: &Label, b: &Label) -> bool {
    a.index == b.index && a.name.eq_ignore_ascii_case(&b.name)
}

fn next_after<'b>(bindings: impl Iterator<Item = &'b Binding>) -> u32 {
    bindings
        .filter_map(|b| match b {
            Binding::Location { location, .. } => Some(location.saturating_add(1)),
            Binding::BuiltIn(_) => None,
        })
        .max()
        .unwrap_or(0)
}

/// The program as WGSL for `pipeline`, for `profile`: preflighted, lowered with the pipeline's plan, linked to its
/// consumer, validated and written. A link the pipeline cannot make (a consumer input no output carries, a boolean or
/// 64-bit output read as another type, `feeds` on a program that is not a vertex program) is an error.
pub fn wgsl_for(
    program: &iridia_ir::Program,
    profile: &Profile,
    pipeline: &Pipeline,
) -> Result<Written, Error> {
    crate::preflight(program, profile)?;
    let mut module = crate::module(program, &pipeline.plan)?;
    let io = entry_io(&program.interface).map_err(Error)?;
    let mut notes = Vec::new();
    let next_location = match (program.stage, pipeline.feeds) {
        (Stage::Vertex, Some(consumer)) => outputs(program, &io, consumer, &mut module, &mut notes)?,
        (Stage::Vertex, None) => next_after(io.results.iter().map(|r| &r.binding)),
        (stage, Some(_)) => {
            return Err(Error(format!("a {stage:?} program feeds no consumer")));
        }
        (_, None) => next_after(io.arguments.iter().map(|a| &a.binding)),
    };
    let info = crate::validate(&module, profile).map_err(|e| {
        Error(format!(
            "the linked module does not validate (a pipeline bug): {}",
            e.0
        ))
    })?;
    let text = crate::write(program, &module, &info, profile)?;
    Ok(Written {
        text,
        next_location,
        notes,
    })
}

/// How an output becomes the value its consumer reads.
#[derive(Clone, Copy, Debug)]
struct Conversion {
    from: Value,
    to: Value,
    /// The consumer's type.
    ty: Handle<Type>,
    /// A vector of the output's scalar at the consumer's width, where that is wider.
    padded: Option<Handle<Type>>,
    /// The zero of the output's scalar, where it is padded.
    zero: Option<Handle<Expression>>,
}

impl Conversion {
    /// `part` (the output's value) as the consumer's: its first lanes, or padded with zeros, then the same bits as the
    /// consumer's scalar.
    fn apply(
        &self,
        part: Handle<Expression>,
        expressions: &mut naga::Arena<Expression>,
    ) -> Handle<Expression> {
        let mut add = |e: Expression| expressions.append(e, Span::UNDEFINED);
        let (from, to) = (self.from.width, self.to.width);
        let mut value = part;
        if to < from {
            value = match vector_size(to) {
                Some(size) => add(Expression::Swizzle {
                    size,
                    vector: value,
                    pattern: [
                        SwizzleComponent::X,
                        SwizzleComponent::Y,
                        SwizzleComponent::Z,
                        SwizzleComponent::W,
                    ],
                }),
                None => add(Expression::AccessIndex {
                    base: value,
                    index: 0,
                }),
            };
        } else if to > from
            && let (Some(ty), Some(zero)) = (self.padded, self.zero)
        {
            let mut components: Vec<Handle<Expression>> = if from == 1 {
                vec![value]
            } else {
                (0..u32::from(from))
                    .map(|i| {
                        add(Expression::AccessIndex {
                            base: value,
                            index: i,
                        })
                    })
                    .collect()
            };
            components.resize(usize::from(to), zero);
            value = add(Expression::Compose { ty, components });
        }
        if self.from.scalar != self.to.scalar {
            value = add(Expression::As {
                expr: value,
                kind: kind(self.to.scalar),
                convert: None,
            });
        }
        value
    }
}

fn vector_size(width: u8) -> Option<VectorSize> {
    match width {
        2 => Some(VectorSize::Bi),
        3 => Some(VectorSize::Tri),
        4 => Some(VectorSize::Quad),
        _ => None,
    }
}

fn kind(s: ScalarType) -> naga::ScalarKind {
    match s {
        ScalarType::Uint => naga::ScalarKind::Uint,
        ScalarType::Sint => naga::ScalarKind::Sint,
        ScalarType::Bool => naga::ScalarKind::Bool,
        ScalarType::Float | ScalarType::Double => naga::ScalarKind::Float,
    }
}

/// The naga scalar of a linked value; a boolean or 64-bit one is refused (its bits are not one 32-bit lane).
fn linked_scalar(s: ScalarType) -> Result<naga::Scalar, Error> {
    match s {
        ScalarType::Float => Ok(naga::Scalar::F32),
        ScalarType::Uint => Ok(naga::Scalar::U32),
        ScalarType::Sint => Ok(naga::Scalar::I32),
        ScalarType::Bool | ScalarType::Double => Err(Error(format!(
            "a {s:?} output read as another value: its bits are not a 32-bit lane"
        ))),
    }
}

/// The type of `width` lanes of `scalar`: the scalar itself for one.
fn value_type(module: &mut naga::Module, scalar: naga::Scalar, width: u8) -> Handle<Type> {
    let inner = match vector_size(width) {
        None => TypeInner::Scalar(scalar),
        Some(size) => TypeInner::Vector { size, scalar },
    };
    module.types.insert(Type { name: None, inner }, Span::UNDEFINED)
}

/// The vertex program's result bindings and types for `consumer` (see [`Pipeline::feeds`]): per result member, its
/// binding and, where the consumer reads it as another value, the output's value and the consumer's.
type Links = (Vec<Binding>, Vec<Option<(Value, Value)>>);

fn link(program: &iridia_ir::Program, io: &EntryIo, consumer: &Interface) -> Result<Links, Error> {
    let mut bindings: Vec<Binding> = io.results.iter().map(|r| r.binding.clone()).collect();
    let mut retyped = vec![None; bindings.len()];
    let wanted = entry_io(consumer).map_err(|e| Error(format!("the consumer: {e}")))?;
    let mut consumed = vec![false; bindings.len()];
    let mut next = 0u32;
    for argument in &wanted.arguments {
        let Binding::Location { location, .. } = argument.binding else {
            continue;
        };
        next = next.max(location.saturating_add(1));
        let label = &consumer
            .inputs
            .get(argument.varying)
            .ok_or_else(|| Error(format!("the consumer's input {} names no varying", argument.name)))?
            .label;
        let k = io
            .results
            .iter()
            .position(|r| {
                matches!(r.binding, Binding::Location { .. })
                    && program
                        .interface
                        .outputs
                        .get(r.varying)
                        .is_some_and(|output| same_label(&output.label, label))
            })
            .ok_or_else(|| {
                let written: Vec<String> = io
                    .results
                    .iter()
                    .filter(|r| matches!(r.binding, Binding::Location { .. }))
                    .map(|r| r.name.clone())
                    .collect();
                Error(format!(
                    "the consumer reads {}{}, which the vertex program does not write (it writes {})",
                    label.name,
                    label.index,
                    written.join(", ")
                ))
            })?;
        if let Some(binding) = bindings.get_mut(k) {
            *binding = argument.binding.clone();
        }
        if let Some(taken) = consumed.get_mut(k) {
            *taken = true;
        }
        if let Some(from) = io.results.get(k).map(|r| r.value)
            && from != argument.value
        {
            if from.scalar != argument.value.scalar {
                linked_scalar(from.scalar)?;
                linked_scalar(argument.value.scalar)?;
            }
            if let Some(slot) = retyped.get_mut(k) {
                *slot = Some((from, argument.value));
            }
        }
    }
    for (binding, consumed) in bindings.iter_mut().zip(consumed) {
        if let (Binding::Location { location, .. }, false) = (binding, consumed) {
            *location = next;
            next = next.saturating_add(1);
        }
    }
    Ok((bindings, retyped))
}

/// The notes a link's retypes earn.
fn retype_notes(retyped: &[Option<(Value, Value)>], notes: &mut Vec<String>) {
    let plural = |n: usize| if n == 1 { "" } else { "s" };
    let padded = retyped
        .iter()
        .flatten()
        .filter(|(from, to)| to.width > from.width)
        .count();
    if padded > 0 {
        notes.push(format!(
            "{padded} output{} read by the consumer with more lanes than written: the lanes past them are 0 (Direct3D \
             leaves them undefined)",
            plural(padded)
        ));
    }
    let narrowed = retyped
        .iter()
        .flatten()
        .filter(|(from, to)| to.width < from.width)
        .count();
    if narrowed > 0 {
        notes.push(format!(
            "{narrowed} output{} read by the consumer with fewer lanes than written: only the first lanes are passed on",
            plural(narrowed)
        ));
    }
    let rekinded = retyped
        .iter()
        .flatten()
        .filter(|(from, to)| from.scalar != to.scalar)
        .count();
    if rekinded > 0 {
        notes.push(format!(
            "{rekinded} output{} read by the consumer as another scalar type: the same bits, reinterpreted (Direct3D \
             links bits)",
            plural(rekinded)
        ));
    }
}

/// Link the vertex program's result structure to `consumer`: its members' bindings and types, and every return made
/// of the new structure. Returns the first location after the outputs'.
fn outputs(
    program: &iridia_ir::Program,
    io: &EntryIo,
    consumer: &Interface,
    module: &mut naga::Module,
    notes: &mut Vec<String>,
) -> Result<u32, Error> {
    let (bindings, retyped) = link(program, io, consumer)?;
    retype_notes(&retyped, notes);
    let next = next_after(bindings.iter());
    let unchanged =
        retyped.iter().all(Option::is_none) && bindings.iter().zip(&io.results).all(|(b, r)| *b == r.binding);
    if unchanged {
        return Ok(next);
    }
    let old = module
        .entry_points
        .first()
        .and_then(|e| e.function.result.as_ref())
        .ok_or_else(|| Error("a vertex program with no result".into()))?
        .ty;
    let (members, span) = match module.types.get_handle(old).map(|t| t.inner.clone()) {
        Ok(TypeInner::Struct { members, span }) => (members, span),
        _ => return Err(Error("a vertex program's result is not a structure".into())),
    };
    if members.len() != bindings.len() {
        return Err(Error(format!(
            "the result structure has {} members, the interface {}",
            members.len(),
            bindings.len()
        )));
    }
    let mut conversions: Vec<Option<Conversion>> = Vec::with_capacity(retyped.len());
    for change in &retyped {
        conversions.push(match change {
            None => None,
            Some((from, to)) => {
                let (from_scalar, to_scalar) = (linked_scalar(from.scalar)?, linked_scalar(to.scalar)?);
                let ty = value_type(module, to_scalar, to.width);
                let padded = (to.width > from.width).then(|| value_type(module, from_scalar, to.width));
                Some(Conversion {
                    from: *from,
                    to: *to,
                    ty,
                    padded,
                    zero: None,
                })
            }
        });
    }
    {
        let entry = module
            .entry_points
            .first_mut()
            .ok_or_else(|| Error("no entry point".into()))?;
        // a literal is no emitted expression: made here, outside the returns' emit ranges
        for c in conversions.iter_mut().flatten() {
            if c.padded.is_some() {
                let literal = match c.from.scalar {
                    ScalarType::Uint => naga::Literal::U32(0),
                    ScalarType::Sint => naga::Literal::I32(0),
                    _ => naga::Literal::F32(0.0),
                };
                c.zero = Some(
                    entry
                        .function
                        .expressions
                        .append(Expression::Literal(literal), Span::UNDEFINED),
                );
            }
        }
    }
    let new_members: Vec<StructMember> = members
        .iter()
        .zip(&bindings)
        .zip(&conversions)
        .map(|((member, binding), conversion)| StructMember {
            binding: Some(binding.clone()),
            ty: conversion.as_ref().map_or(member.ty, |c| c.ty),
            ..member.clone()
        })
        .collect();
    let name = module.types.get_handle(old).ok().and_then(|t| t.name.clone());
    let new = module.types.insert(
        Type {
            name: name.clone(),
            inner: TypeInner::Struct {
                members: new_members,
                span,
            },
        },
        Span::UNDEFINED,
    );
    // the old structure stays as the value each return is made of, without its bindings
    let plain = Type {
        name,
        inner: TypeInner::Struct {
            members: members
                .into_iter()
                .map(|m| StructMember { binding: None, ..m })
                .collect(),
            span,
        },
    };
    if module.types.get(&plain).is_some() {
        return Err(Error(
            "the result structure without its bindings is already a type of the module".into(),
        ));
    }
    module.types.replace(old, plain);
    let entry = module
        .entry_points
        .first_mut()
        .ok_or_else(|| Error("no entry point".into()))?;
    if let Some(result) = entry.function.result.as_mut() {
        result.ty = new;
    }
    let function = &mut entry.function;
    let body = core::mem::take(&mut function.body);
    function.body = returns(body, &mut function.expressions, &mut |value, expressions| {
        // a function with a result returns a value (the module was lowered and is validated after)
        let Some(value) = value else {
            return (Vec::new(), None);
        };
        let start = expressions.len();
        let components: Vec<Handle<Expression>> = conversions
            .iter()
            .enumerate()
            .map(|(k, conversion)| {
                let part = expressions.append(
                    Expression::AccessIndex {
                        base: value,
                        index: u32::try_from(k).unwrap_or(u32::MAX),
                    },
                    Span::UNDEFINED,
                );
                match conversion {
                    Some(c) => c.apply(part, expressions),
                    None => part,
                }
            })
            .collect();
        let compose = expressions.append(Expression::Compose { ty: new, components }, Span::UNDEFINED);
        let range = |n: usize| u32::try_from(n).unwrap_or(u32::MAX);
        let emitted = naga::Range::from_index_range(range(start)..range(expressions.len()), expressions);
        (vec![Statement::Emit(emitted)], Some(compose))
    });
    Ok(next)
}

/// `block` with each return (in nested blocks too) preceded by the statements `at` makes and returning the value it
/// gives, from the value it returned.
#[allow(clippy::type_complexity)]
fn returns(
    block: Block,
    expressions: &mut naga::Arena<Expression>,
    at: &mut dyn FnMut(
        Option<Handle<Expression>>,
        &mut naga::Arena<Expression>,
    ) -> (Vec<Statement>, Option<Handle<Expression>>),
) -> Block {
    let mut out = Block::with_capacity(block.len());
    for (statement, span) in block.span_into_iter() {
        match statement {
            Statement::Return { value } => {
                let (before, value) = at(value, expressions);
                for s in before {
                    out.push(s, span);
                }
                out.push(Statement::Return { value }, span);
            }
            Statement::Block(b) => out.push(Statement::Block(returns(b, expressions, at)), span),
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                let accept = returns(accept, expressions, at);
                let reject = returns(reject, expressions, at);
                out.push(
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    span,
                );
            }
            Statement::Switch { selector, cases } => {
                let cases = cases
                    .into_iter()
                    .map(|c| naga::SwitchCase {
                        body: returns(c.body, expressions, at),
                        ..c
                    })
                    .collect();
                out.push(Statement::Switch { selector, cases }, span);
            }
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => {
                let body = returns(body, expressions, at);
                let continuing = returns(continuing, expressions, at);
                out.push(
                    Statement::Loop {
                        body,
                        continuing,
                        break_if,
                    },
                    span,
                );
            }
            other => out.push(other, span),
        }
    }
    out
}
