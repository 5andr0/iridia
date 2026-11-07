//! Control flow: the regions of a function as WGSL statements. A construct's results and a loop's parameters are
//! variables (`p3`) that each exit assigns and the code after the construct reads once, as a value (`let v3 = p3;`).
//!
//! WGSL's `break` and `continue` reach only the innermost loop or switch: run `iridia_passes::break_flags` first.

use alloc::{format, string::String, vec::Vec};

use iridia_ir::{End, LabelId, Node, Region, Scalar, Type, ValueId};

use crate::emit::{R, Writer};
use crate::text::{self, err};

/// A construct being written: the label exits name, and where they put their values.
#[derive(Clone, Debug)]
pub(crate) struct Construct {
    label: LabelId,
    /// The variables a yield or a break stores the construct's results in.
    results: Vec<String>,
    /// The variables a continue stores the loop's parameters in; `None` for a construct that is not a loop.
    params: Option<Vec<String>>,
    kind: Kind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Block,
    Loop,
    Switch,
}

/// What a region's yield does.
#[derive(Clone, Copy)]
pub(crate) enum Yield<'a> {
    /// The function's body: a yield of nothing falls off its end.
    Top,
    /// An `If` arm or a `Switch` case: store the results and fall through.
    Fall(&'a [String]),
    /// A loop body: store the results and leave the loop.
    Break(&'a [String]),
}

impl Writer<'_> {
    fn construct(&self, l: LabelId) -> R<(usize, &Construct)> {
        self.constructs
            .iter()
            .enumerate()
            .rev()
            .find(|(_, c)| c.label == l)
            .ok_or_else(|| {
                err(format!(
                    "an exit to label {}, which no enclosing construct has",
                    l.0
                ))
            })
    }

    /// A break to construct `at` is a WGSL `break` only when it is the innermost loop or switch; a continue, when it is the
    /// innermost loop (a switch between does not stop it).
    fn innermost(&self, at: usize, continues: bool) -> R<()> {
        let last = self
            .constructs
            .iter()
            .rposition(|c| c.kind == Kind::Loop || (!continues && c.kind == Kind::Switch));
        if last == Some(at) {
            Ok(())
        } else {
            Err(err(
                "an exit past an enclosing construct: run iridia_passes::break_flags first",
            ))
        }
    }

    /// A variable for each value, of the value's type, and its name.
    fn storage_for(&mut self, vs: &[ValueId]) -> R<Vec<String>> {
        vs.iter()
            .map(|v| {
                let t = self.vty(*v)?;
                let name = format!("p{}", v.0);
                self.line(&format!("var {name}: {};", text::ty(t)?));
                Ok(name)
            })
            .collect()
    }

    /// The construct's values read the variables they were stored in: `let v3 = p3;`.
    fn load_storage(&mut self, vs: &[ValueId], storage: &[String]) -> R<()> {
        for (v, s) in vs.iter().zip(storage) {
            self.define(*v, s)?;
        }
        Ok(())
    }

    /// Assign the variables the values: `p3 = v9;`. Every value is read before any variable is written, as a loop's
    /// parameters are all read from the values the iteration started with.
    fn store_all(&mut self, storage: &[String], vs: &[ValueId]) -> R<()> {
        if storage.len() != vs.len() {
            return Err(err(
                "an exit with another number of values than its construct takes",
            ));
        }
        for (s, v) in storage.iter().zip(vs) {
            let value = self.get(*v)?;
            self.line(&format!("{s} = {value};"));
        }
        Ok(())
    }

    pub(crate) fn region(&mut self, r: &Region, yields: Yield<'_>) -> R<()> {
        for n in &r.nodes {
            self.node(n)?;
        }
        self.finish(&r.end, yields)
    }

    /// How a region ends.
    fn finish(&mut self, end: &Option<End>, yields: Yield<'_>) -> R<()> {
        match end {
            None => return Err(err("a region with no end")),
            Some(End::Yield(vs)) => match yields {
                Yield::Top if vs.is_empty() => {}
                Yield::Top => return Err(err("a yield of values at the end of the function")),
                Yield::Fall(storage) => self.store_all(storage, vs)?,
                Yield::Break(storage) => {
                    self.store_all(storage, vs)?;
                    self.line("break;");
                }
            },
            Some(End::Br(l, vs)) => {
                let (at, c) = self.construct(*l)?;
                let (storage, kind) = (c.results.clone(), c.kind);
                if kind == Kind::Block {
                    return Err(err(
                        "a break out of a block: run iridia_passes::break_flags first",
                    ));
                }
                self.store_all(&storage, vs)?;
                self.innermost(at, false)?;
                self.line("break;");
            }
            Some(End::Continue(l, vs)) => {
                let (at, c) = self.construct(*l)?;
                let storage = c
                    .params
                    .clone()
                    .ok_or_else(|| err("a continue to a construct that is not a loop"))?;
                self.store_all(&storage, vs)?;
                self.innermost(at, true)?;
                self.line("continue;");
            }
            Some(End::Return) => self.ret()?,
            // WGSL's `discard` demotes the invocation and goes on: the return ends it
            Some(End::Kill) => {
                self.line("discard;");
                self.ret()?;
            }
            Some(End::IgnoreHit | End::AcceptHitAndEndSearch) => {
                return Err(err("an any-hit program's decision is not written yet"));
            }
            // a function that returns a value ends in a return: the function's unreachable end returns its outputs, which
            // no invocation reaches
            Some(End::Unreachable) if matches!(yields, Yield::Top) => self.ret()?,
            Some(End::Unreachable) => {}
        }
        Ok(())
    }

    pub(crate) fn node(&mut self, n: &Node) -> R<()> {
        match n {
            Node::Value(v) => self.value(*v)?,
            // WGSL's `discard` is a demotion: the invocation goes on as a helper
            Node::Demote => self.line("discard;"),
            Node::StoreVar(var, x) => self.store_var(*var, *x)?,
            Node::StoreOutput(slot, mask, x) => self.store_output(*slot, *mask, *x)?,
            Node::StoreElement(var, index, x, oob) => self.store_element_of_var(*var, *index, *x, *oob)?,
            Node::StoreBuffer {
                resource,
                address,
                value,
            } => {
                if resource.index.is_some() {
                    return Err(err(
                        "a write of an element of an array of buffers is not written yet",
                    ));
                }
                if !self.words_buffer(resource.id)? {
                    return Err(err(format!("a buffer write of resource {}", resource.id.0)));
                }
                self.store_words(resource.id, *address, *value)?;
            }
            Node::StoreTexel { image, coord, value } if self.typed_buffer(image.id)? => {
                if image.index.is_some() {
                    return Err(err(
                        "a write of an element of an array of buffers is not written yet",
                    ));
                }
                self.store_typed_element(image.id, *coord, *value)?;
            }
            Node::StoreTexel { image, coord, value } => self.store_texel(*image, *coord, *value)?,
            Node::StorePatch(..) => return Err(err("a patch constant store is not written yet")),
            Node::Barrier(b) => self.barrier(*b),
            Node::Emit(_) | Node::Cut(_) => {
                return Err(err("a geometry program's emit and cut are not written yet"));
            }
            Node::SetMeshCounts { .. } | Node::StoreVertex { .. } | Node::StoreIndices { .. } => {
                return Err(err("a mesh program's meshlet is not written yet"));
            }
            Node::DispatchMesh { .. } => {
                return Err(err("an amplification program's mesh launch is not written yet"));
            }
            Node::TraceRay { .. } | Node::CallShader { .. } => {
                return Err(err("a traced ray or a shader call is not written yet"));
            }
            Node::If {
                cond,
                accept,
                reject,
                results,
            } => {
                let condition = self.get(*cond)?;
                let storage = self.storage_for(results)?;
                let a = self.capture(|w| w.region(accept, Yield::Fall(&storage)))?;
                let b = self.capture(|w| w.region(reject, Yield::Fall(&storage)))?;
                self.line(&format!("if ({condition}) {{"));
                self.out.push_str(&a);
                if b.is_empty() {
                    self.line("}");
                } else {
                    self.line("} else {");
                    self.out.push_str(&b);
                    self.line("}");
                }
                self.load_storage(results, &storage)?;
            }
            Node::Switch {
                label,
                selector,
                cases,
                default,
                results,
            } => {
                let sel = self.get(*selector)?;
                let signed = self.vty(*selector)? == Type::I32;
                let storage = self.storage_for(results)?;
                self.constructs.push(Construct {
                    label: *label,
                    results: storage.clone(),
                    params: None,
                    kind: Kind::Switch,
                });
                let mut bodies = Vec::new();
                // the cases are one level in from the `switch`, their bodies one more
                self.depth = self.depth.saturating_add(1);
                for c in cases {
                    let body = self.capture(|w| w.region(&c.body, Yield::Fall(&storage)))?;
                    let values: Vec<String> = c
                        .values
                        .iter()
                        .map(|v| text::literal(if signed { Scalar::I32 } else { Scalar::U32 }, u64::from(*v)))
                        .collect::<Result<_, _>>()?;
                    bodies.push((values.join(", "), body));
                }
                let default_body = self.capture(|w| w.region(default, Yield::Fall(&storage)));
                self.depth = self.depth.saturating_sub(1);
                let default_body = default_body?;
                self.constructs.pop();
                self.line(&format!("switch ({sel}) {{"));
                self.depth = self.depth.saturating_add(1);
                for (values, body) in bodies {
                    self.line(&format!("case {values}: {{"));
                    self.out.push_str(&body);
                    self.line("}");
                }
                self.line("default: {");
                self.out.push_str(&default_body);
                self.line("}");
                self.depth = self.depth.saturating_sub(1);
                self.line("}");
                self.load_storage(results, &storage)?;
            }
            Node::Block { label, body, results } => {
                // a block is left by falling off its end (an exit to it is refused): its nodes are the enclosing region's,
                // and its results the values it yields
                self.constructs.push(Construct {
                    label: *label,
                    results: Vec::new(),
                    params: None,
                    kind: Kind::Block,
                });
                for n in &body.nodes {
                    self.node(n)?;
                }
                self.constructs.pop();
                match &body.end {
                    Some(End::Yield(vs)) if vs.len() == results.len() => {
                        for (r, v) in results.iter().zip(vs) {
                            let value = self.get(*v)?;
                            self.define(*r, &value)?;
                        }
                    }
                    Some(End::Yield(_)) => {
                        return Err(err("a block yields another number of values than it has results"));
                    }
                    // a block that ends in a return, a kill or an exit of an enclosing construct has no results to give
                    other => self.finish(other, Yield::Fall(&[]))?,
                }
            }
            Node::Loop {
                label,
                params,
                inits,
                body,
                results,
            } => {
                let param_storage = self.storage_for(params)?;
                for (s, v) in param_storage.iter().zip(inits) {
                    let value = self.get(*v)?;
                    self.line(&format!("{s} = {value};"));
                }
                let storage = self.storage_for(results)?;
                self.constructs.push(Construct {
                    label: *label,
                    results: storage.clone(),
                    params: Some(param_storage.clone()),
                    kind: Kind::Loop,
                });
                let inner = self.capture(|w| {
                    w.load_storage(params, &param_storage)?;
                    w.region(body, Yield::Break(&storage))
                })?;
                self.constructs.pop();
                self.line("loop {");
                self.out.push_str(&inner);
                self.line("}");
                self.load_storage(results, &storage)?;
            }
        }
        Ok(())
    }

    /// The text a return is.
    pub(crate) fn ret(&mut self) -> R<()> {
        if self.members.is_empty() {
            self.line("return;");
            return Ok(());
        }
        let parts: Vec<String> = (0..self.members.len())
            .map(|m| self.member_value(m))
            .collect::<Result<_, _>>()?;
        let value = format!("Out({})", parts.join(", "));
        self.line(&format!("return {value};"));
        Ok(())
    }
}
