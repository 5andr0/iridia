//! A program's WGSL as one pipeline binds it ([`wgsl_for`]): each resource where the pipeline's [`BindingPlan`] puts it, and a
//! vertex program's outputs where the program that reads them takes them. The program keeps its own interface.
//!
//! Two programs of one pipeline share its bind groups, so a vertex and a pixel program that both read constant buffer `b0`
//! need plans that put them apart; and WebGPU links a vertex output to a pixel input by location and type, where Direct3D
//! links them by semantic and register lanes. [`Pipeline::feeds`] gives the consumer, and each output it reads takes the
//! consumer's location, interpolation and type.

use alloc::{format, string::String, vec, vec::Vec};

use iridia_ir::Program;
use iridia_ir::interface::{Interface, Label, Stage, Value};
use iridia_ir::target::{Binding, IoItem, IoType, entry_io};

use crate::emit::{R, Writer};
use crate::module::Linked;
use crate::profile::{Profile, preflight};
use crate::text::err;
use crate::{BindingPlan, Error};

/// The pipeline a program's WGSL is written for.
#[derive(Clone, Debug)]
pub struct Pipeline<'a> {
    /// Where each resource of the program is bound ([`BindingPlan::d3d`] is the program's own plan).
    pub plan: BindingPlan,
    /// The program a vertex program's outputs feed. Each output the consumer reads, matched as Direct3D links stages (by
    /// semantic: the label's name without case, and its index), takes the consumer's location and interpolation and is read
    /// as the consumer's type: fewer lanes are the output's first ones, more lanes are the output's followed by zeros
    /// (Direct3D leaves them undefined; noted), another 32-bit scalar kind is the same bits. The outputs the consumer does
    /// not read take the locations after the consumer's. A consumer input no output carries is an error.
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

fn value_of(item: &IoItem) -> R<Value> {
    match item.ty {
        IoType::Value(v) => Ok(v),
        IoType::ClipDistances(n) => Ok(Value {
            scalar: iridia_ir::interface::ScalarType::Float,
            width: u8::try_from(n).map_err(|_| err(format!("{}: {n} clip distances", item.name)))?,
        }),
        ref other => Err(err(format!("{}: {other:?} in an entry point", item.name))),
    }
}

/// The program as WGSL for `pipeline`, for `profile`: preflighted, linked to its consumer and written. A link the pipeline
/// cannot make (a consumer input no output carries, a boolean or 64-bit output read as another type, `feeds` on a program
/// that is not a vertex program) is an error.
pub fn wgsl_for(program: &Program, profile: &Profile, pipeline: &Pipeline) -> Result<Written, Error> {
    preflight(program, profile)?;
    let io = entry_io(&program.interface).map_err(Error)?;
    let mut notes = Vec::new();
    let mut retyped: Vec<Option<(Value, Value)>> = Vec::new();
    let mut bindings: Vec<Binding> = io.results.iter().map(|r| r.binding).collect();
    let next_location = match (program.stage, pipeline.feeds) {
        (Stage::Vertex, Some(consumer)) => {
            (bindings, retyped) = link(program, &io.results, consumer)?;
            retype_notes(&retyped, &mut notes);
            next_after(bindings.iter())
        }
        (Stage::Vertex, None) => next_after(bindings.iter()),
        (stage, Some(_)) => {
            return Err(Error(format!(
                "{} program feeds no consumer",
                stage.with_article()
            )));
        }
        (_, None) => next_after(io.arguments.iter().map(|a| &a.binding)),
    };
    let text = Writer::module(program, &pipeline.plan, profile, |results| {
        Ok(results
            .iter()
            .enumerate()
            .map(|(k, item)| Linked {
                binding: bindings.get(k).copied().unwrap_or(item.binding),
                value: retyped.get(k).copied().flatten().map(|(_, to)| to),
            })
            .collect())
    })?;
    Ok(Written {
        text,
        next_location,
        notes,
    })
}

/// The vertex program's result bindings and types for `consumer` (see [`Pipeline::feeds`]): per result member, its binding
/// and, where the consumer reads it as another value, the output's value and the consumer's.
type Links = (Vec<Binding>, Vec<Option<(Value, Value)>>);

fn link(program: &Program, results: &[IoItem], consumer: &Interface) -> R<Links> {
    let mut bindings: Vec<Binding> = results.iter().map(|r| r.binding).collect();
    let mut retyped = vec![None; bindings.len()];
    let wanted = entry_io(consumer).map_err(|e| err(format!("the consumer: {e}")))?;
    let mut consumed = vec![false; bindings.len()];
    let mut next = 0u32;
    for argument in &wanted.arguments {
        let Binding::Location { location, .. } = argument.binding else {
            continue;
        };
        next = next.max(location.saturating_add(1));
        let varying = *argument
            .varyings
            .first()
            .ok_or_else(|| err(format!("the consumer's input {} names no varying", argument.name)))?;
        let label = &consumer
            .inputs
            .get(varying)
            .ok_or_else(|| err(format!("the consumer's input {} names no varying", argument.name)))?
            .label;
        let k = results
            .iter()
            .position(|r| {
                matches!(r.binding, Binding::Location { .. })
                    && r.varyings
                        .first()
                        .and_then(|v| program.interface.outputs.get(*v))
                        .is_some_and(|output| same_label(&output.label, label))
            })
            .ok_or_else(|| {
                let written: Vec<String> = results
                    .iter()
                    .filter(|r| matches!(r.binding, Binding::Location { .. }))
                    .map(|r| r.name.clone())
                    .collect();
                err(format!(
                    "the consumer reads {}{}, which the vertex program does not write (it writes {})",
                    label.name,
                    label.index,
                    written.join(", ")
                ))
            })?;
        if let Some(binding) = bindings.get_mut(k) {
            *binding = argument.binding;
        }
        if let Some(taken) = consumed.get_mut(k) {
            *taken = true;
        }
        let from = results.get(k).map(value_of).transpose()?;
        let to = value_of(argument)?;
        if let Some(from) = from
            && from != to
            && let Some(slot) = retyped.get_mut(k)
        {
            *slot = Some((from, to));
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
