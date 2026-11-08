//! Zeroing a compute program's workgroup memory in the module, rather than leaving it to wgpu.
//!
//! WGSL starts every `var<workgroup>` zeroed, and wgpu 30.0.1's naga writes that to HLSL as one aggregate assignment in the
//! first invocation (`if (local_invocation_index == 0) { g = (uint[3072])0; }`). DXC or FXC and WARP's compiler take time
//! that grows faster than the array to build that: the same program took 3 s with 1,024 words of group-shared memory, 21 s
//! with 2,048 and 50 to 120 s with 3,072, whatever the work and the number of threads. The original programs have no such
//! assignment (D3D does not zero group-shared memory). wgpu's trunk (PR 10314, merged 2026-11-06, not in a release yet) writes
//! the initialisation as loops distributed over the group's invocations. This does the same in the naga module: at the top of
//! the entry point, every used workgroup variable is zeroed by a loop over `local_invocation_index` that steps by the group's
//! size, a lone value by invocation 0, then a workgroup barrier; the pipeline is made without wgpu's own initialisation.
//! A program's results are the same: nothing reads workgroup memory before the barrier, and every word is zero after it.

use std::collections::BTreeMap;

use naga::{
    AddressSpace, Arena, ArraySize, AtomicFunction, Barrier, BinaryOperator as Bo, Binding, Block, BuiltIn,
    Expression as E, FunctionArgument, Handle, Literal, LocalVariable, Module, Scalar, ScalarKind, Span,
    Statement, Type, TypeInner, valid::ModuleInfo,
};

/// A step from a variable to a leaf: into an array's element (the array's length), or into a structure's member.
#[derive(Clone, Copy)]
enum Step {
    Element(u32),
    Member(u32),
}

/// A part of a variable that is not an array or a structure, and the steps to it.
struct Leaf {
    path: Vec<Step>,
    ty: Handle<Type>,
}

fn leaves(
    module: &Module,
    ty: Handle<Type>,
    path: &mut Vec<Step>,
    out: &mut Vec<Leaf>,
) -> Result<(), String> {
    match &module.types[ty].inner {
        TypeInner::Array {
            base,
            size: ArraySize::Constant(n),
            ..
        } => {
            path.push(Step::Element(n.get()));
            leaves(module, *base, path, out)?;
            path.pop();
        }
        TypeInner::Struct { members, .. } => {
            for (k, m) in members.iter().enumerate() {
                path.push(Step::Member(k as u32));
                leaves(module, m.ty, path, out)?;
                path.pop();
            }
        }
        TypeInner::Scalar(_) | TypeInner::Vector { .. } | TypeInner::Matrix { .. } | TypeInner::Atomic(_) => {
            out.push(Leaf {
                path: path.clone(),
                ty,
            });
        }
        other => return Err(format!("a workgroup variable with a {other:?} in it")),
    }
    Ok(())
}

/// A block being built over a function's expression arena, with the first expression it has not emitted.
struct Scope<'a> {
    expressions: &'a mut Arena<E>,
    block: Block,
    start: Option<Handle<E>>,
}

impl<'a> Scope<'a> {
    fn new(expressions: &'a mut Arena<E>) -> Self {
        Scope {
            expressions,
            block: Block::new(),
            start: None,
        }
    }
    fn flush(&mut self) {
        if let Some(first) = self.start.take()
            && let Some((last, _)) = self.expressions.iter().last()
        {
            self.block.push(
                Statement::Emit(naga::Range::new_from_bounds(first, last)),
                Span::UNDEFINED,
            );
        }
    }
    fn expr(&mut self, e: E) -> Handle<E> {
        let unemitted = e.needs_pre_emit() || matches!(e, E::AtomicResult { .. });
        if unemitted {
            self.flush();
        }
        let h = self.expressions.append(e, Span::UNDEFINED);
        if !unemitted && self.start.is_none() {
            self.start = Some(h);
        }
        h
    }
    fn statement(&mut self, s: Statement) {
        self.flush();
        self.block.push(s, Span::UNDEFINED);
    }
    fn finish(mut self) -> Block {
        self.flush();
        self.block
    }
    fn u32(&mut self, v: u32) -> Handle<E> {
        self.expr(E::Literal(Literal::U32(v)))
    }
    fn binary(&mut self, op: Bo, left: Handle<E>, right: Handle<E>) -> Handle<E> {
        self.expr(E::Binary { op, left, right })
    }
}

/// The literal zero of a 32-bit scalar.
fn zero_literal(s: Scalar) -> Result<Literal, String> {
    match (s.kind, s.width) {
        (ScalarKind::Uint, 4) => Ok(Literal::U32(0)),
        (ScalarKind::Sint, 4) => Ok(Literal::I32(0)),
        (ScalarKind::Float, 4) => Ok(Literal::F32(0.0)),
        (kind, width) => Err(format!("an atomic of {} bits of {kind:?}", width * 8)),
    }
}

/// The lengths of the arrays on the way to a leaf, outermost first.
fn lengths(leaf: &Leaf) -> Vec<u32> {
    leaf.path
        .iter()
        .filter_map(|s| match s {
            Step::Element(n) => Some(*n),
            Step::Member(_) => None,
        })
        .collect()
}

/// The pointer to a leaf of workgroup variable `handle`: of element `flat` of the leaf's arrays taken as one, outermost
/// dimension slowest (`None` for a leaf in no array).
fn pointer_to(
    scope: &mut Scope<'_>,
    handle: Handle<naga::GlobalVariable>,
    leaf: &Leaf,
    flat: Option<Handle<E>>,
) -> Handle<E> {
    let lengths = lengths(leaf);
    let mut pointer = scope.expr(E::GlobalVariable(handle));
    let mut dimension = 0;
    for step in &leaf.path {
        pointer = match *step {
            Step::Member(k) => scope.expr(E::AccessIndex {
                base: pointer,
                index: k,
            }),
            Step::Element(_) => {
                let mut index = flat.expect("a leaf in an array has a flat index");
                let stride: u32 = lengths[dimension + 1..].iter().product();
                if stride > 1 {
                    let stride = scope.u32(stride);
                    index = scope.binary(Bo::Divide, index, stride);
                }
                if dimension > 0 {
                    let length = scope.u32(lengths[dimension]);
                    index = scope.binary(Bo::Modulo, index, length);
                }
                dimension += 1;
                scope.expr(E::Access { base: pointer, index })
            }
        };
    }
    pointer
}

/// Write zero through `pointer` to a leaf of type `ty` (an atomic by exchange, whose result `atomic_result` types).
fn zero_leaf(
    scope: &mut Scope<'_>,
    module: &Module,
    atomic_result: &BTreeMap<Handle<Type>, Handle<Type>>,
    pointer: Handle<E>,
    ty: Handle<Type>,
) -> Result<(), String> {
    if let TypeInner::Atomic(s) = module.types[ty].inner {
        let value = scope.expr(E::Literal(zero_literal(s)?));
        let result = scope.expr(E::AtomicResult {
            ty: atomic_result[&ty],
            comparison: false,
        });
        scope.statement(Statement::Atomic {
            pointer,
            fun: AtomicFunction::Exchange { compare: None },
            value,
            result: Some(result),
        });
    } else {
        let value = scope.expr(E::ZeroValue(ty));
        scope.statement(Statement::Store { pointer, value });
    }
    Ok(())
}

/// Zero the workgroup variables entry point `index` uses, in the module, at the top of the entry point: whether there were
/// any. `Err` says why that cannot be done for this program (the caller leaves the initialisation to wgpu).
pub fn initialize(module: &mut Module, info: &ModuleInfo, index: usize) -> Result<bool, String> {
    let entry = &module.entry_points[index];
    if entry.workgroup_size_overrides.is_some() {
        return Err("a workgroup size set by overrides".into());
    }
    let size = entry
        .workgroup_size
        .iter()
        .try_fold(1u32, |n, &s| n.checked_mul(s))
        .ok_or("a workgroup of more than 2^32 invocations")?;
    let used = info.get_entry_point(index);
    let mut variables = Vec::new();
    for (handle, variable) in module.global_variables.iter() {
        if variable.space == AddressSpace::WorkGroup && !used[handle].is_empty() {
            let mut found = Vec::new();
            leaves(module, variable.ty, &mut Vec::new(), &mut found)?;
            variables.push((handle, found));
        }
    }
    if variables.is_empty() {
        return Ok(false);
    }
    let mut function = entry.function.clone();
    // the invocation's index in its group: the entry point's own argument for it, or one added
    let mut direct = None;
    for (k, a) in function.arguments.iter().enumerate() {
        match (&a.binding, &module.types[a.ty].inner) {
            (Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)), _) => direct = Some(k),
            (None, TypeInner::Struct { members, .. })
                if members
                    .iter()
                    .any(|m| m.binding == Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex))) =>
            {
                return Err("a local invocation index inside a structure argument".into());
            }
            _ => {}
        }
    }
    let u32_type = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::UNDEFINED,
    );
    let argument = direct.unwrap_or_else(|| {
        function.arguments.push(FunctionArgument {
            name: Some("harness_local_index".into()),
            ty: u32_type,
            binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
        });
        function.arguments.len() - 1
    });
    // a type for what each atomic leaf holds
    let mut atomic_result = BTreeMap::new();
    for (_, found) in &variables {
        for leaf in found {
            if let TypeInner::Atomic(s) = module.types[leaf.ty].inner {
                zero_literal(s)?;
                atomic_result.insert(
                    leaf.ty,
                    module.types.insert(
                        Type {
                            name: None,
                            inner: TypeInner::Scalar(s),
                        },
                        Span::UNDEFINED,
                    ),
                );
            }
        }
    }
    let counter = function.local_variables.append(
        LocalVariable {
            name: Some("harness_zero_i".into()),
            ty: u32_type,
            init: None,
        },
        Span::UNDEFINED,
    );

    let mut scope = Scope::new(&mut function.expressions);
    let local = scope.expr(E::FunctionArgument(argument as u32));
    let zero = scope.u32(0);
    let counter_pointer = scope.expr(E::LocalVariable(counter));
    // the leaves that are not in an array: invocation 0, once
    let mut lone = Vec::new();
    let mut arrays = Vec::new();
    for (handle, found) in &variables {
        for leaf in found {
            if leaf.path.iter().any(|s| matches!(s, Step::Element(_))) {
                arrays.push((*handle, leaf));
            } else {
                lone.push((*handle, leaf));
            }
        }
    }
    if !lone.is_empty() {
        let first = scope.binary(Bo::Equal, local, zero);
        scope.flush();
        let mut inner = Scope::new(&mut *scope.expressions);
        for (handle, leaf) in lone {
            let pointer = pointer_to(&mut inner, handle, leaf, None);
            zero_leaf(&mut inner, module, &atomic_result, pointer, leaf.ty)?;
        }
        let accept = inner.finish();
        scope.statement(Statement::If {
            condition: first,
            accept,
            reject: Block::new(),
        });
    }
    for (handle, leaf) in arrays {
        // for (i = local; i < total; i += size) { element i of the leaf = 0 }
        scope.statement(Statement::Store {
            pointer: counter_pointer,
            value: local,
        });
        let mut body = Scope::new(&mut *scope.expressions);
        let i = body.expr(E::Load {
            pointer: counter_pointer,
        });
        let total = body.u32(lengths(leaf).iter().product());
        let done = body.binary(Bo::GreaterEqual, i, total);
        body.statement(Statement::If {
            condition: done,
            accept: Block::from_vec(vec![Statement::Break]),
            reject: Block::new(),
        });
        let pointer = pointer_to(&mut body, handle, leaf, Some(i));
        zero_leaf(&mut body, module, &atomic_result, pointer, leaf.ty)?;
        let body = body.finish();
        let mut continuing = Scope::new(&mut *scope.expressions);
        let now = continuing.expr(E::Load {
            pointer: counter_pointer,
        });
        let step = continuing.u32(size);
        let next = continuing.binary(Bo::Add, now, step);
        continuing.statement(Statement::Store {
            pointer: counter_pointer,
            value: next,
        });
        let continuing = continuing.finish();
        scope.statement(Statement::Loop {
            body,
            continuing,
            break_if: None,
        });
    }
    scope.statement(Statement::ControlBarrier(Barrier::WORK_GROUP));
    let mut prologue = scope.finish();
    prologue.extend_block(std::mem::take(&mut function.body));
    function.body = prologue;
    module.entry_points[index].function = function;
    Ok(true)
}
