//! The program made callable: its entry point becomes a plain function (`harness_program`), and a new entry point
//! (`harness_main`) reads each invocation's arguments from a storage buffer, calls it, and writes its results and a word
//! saying it returned to another. A vertex program is called from a compute entry point, one workgroup an invocation (no
//! rasterizer, so no interpolation touches its inputs); a pixel program from a fragment entry point, one pixel an
//! invocation over 2x2 quads, so its derivatives and its discarded invocations' helpers are the rasterizer's own. A
//! compute program runs as it is.

use crate::job::WorkgroupInit;
use naga::{
    AddressSpace, ArraySize, BinaryOperator as Bo, Binding, Block, BuiltIn, EntryPoint, Expression as E,
    Function, FunctionArgument, FunctionResult, GlobalVariable, Handle, Literal, ResourceBinding, Scalar,
    ScalarKind, ShaderStage, Span, Statement, StorageAccess, Type, TypeInner, VectorSize,
};

/// The bind group of the harness's own buffers, past any group a program binds.
pub const GROUP: u32 = 7;
/// Their bindings: the invocations' arguments, read; their results, written.
pub const INPUTS: u32 = 0;
pub const OUTPUTS: u32 = 1;
/// The name of the entry point the harness adds.
pub const ENTRY: &str = "harness_main";

/// A program ready to run.
pub struct Wrapped {
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
    /// The program's own stage.
    pub stage: ShaderStage,
    /// The entry point to run: [`ENTRY`], or a compute program's own.
    pub entry: String,
    /// The index of that entry point in the module.
    pub entry_index: usize,
    /// The words of each argument of the program's entry point.
    pub arguments: Vec<u32>,
    /// The words of each of its results.
    pub results: Vec<u32>,
    /// Whether the module zeroes its workgroup memory itself ([`crate::workgroup`]), so the pipeline is made without wgpu's.
    pub zeroes_workgroup_memory: bool,
}

impl Wrapped {
    /// The words of one invocation's results, after the word saying it returned.
    pub fn output_words(&self) -> u32 {
        1 + self.results.iter().sum::<u32>()
    }
}

fn scalar_words(s: Scalar) -> Result<u32, String> {
    match s.width {
        1 if s.kind == ScalarKind::Bool => Ok(1),
        4 => Ok(1),
        8 => Ok(2),
        w => Err(format!("a {}-bit {:?} has no words here", w * 8, s.kind)),
    }
}

/// How many words a value of `ty` takes.
fn words(module: &naga::Module, ty: Handle<Type>) -> Result<u32, String> {
    Ok(match &module.types[ty].inner {
        TypeInner::Scalar(s) => scalar_words(*s)?,
        TypeInner::Vector { size, scalar } => *size as u32 * scalar_words(*scalar)?,
        TypeInner::Array {
            base,
            size: ArraySize::Constant(n),
            ..
        } => n.get() * words(module, *base)?,
        TypeInner::Struct { members, .. } => {
            let mut n = 0;
            for m in members {
                n += words(module, m.ty)?;
            }
            n
        }
        other => return Err(format!("a value of {other:?} has no words here")),
    })
}

/// A function being built, with the first expression its innermost block has not emitted.
struct Build<'m> {
    module: &'m mut naga::Module,
    f: Function,
    body: Block,
    start: Option<Handle<E>>,
    inputs: Handle<GlobalVariable>,
    outputs: Handle<GlobalVariable>,
}

impl Build<'_> {
    fn ty(&mut self, inner: TypeInner) -> Handle<Type> {
        self.module
            .types
            .insert(Type { name: None, inner }, Span::UNDEFINED)
    }
    fn flush(&mut self) {
        if let Some(first) = self.start.take()
            && let Some((last, _)) = self.f.expressions.iter().last()
        {
            self.body.push(
                Statement::Emit(naga::Range::new_from_bounds(first, last)),
                Span::UNDEFINED,
            );
        }
    }
    fn expr(&mut self, e: E) -> Handle<E> {
        let unemitted = e.needs_pre_emit() || matches!(e, E::CallResult(_));
        if unemitted {
            self.flush();
        }
        let h = self.f.expressions.append(e, Span::UNDEFINED);
        if !unemitted && self.start.is_none() {
            self.start = Some(h);
        }
        h
    }
    fn statement(&mut self, s: Statement) {
        self.flush();
        self.body.push(s, Span::UNDEFINED);
    }
    fn u32(&mut self, v: u32) -> Handle<E> {
        self.expr(E::Literal(Literal::U32(v)))
    }
    fn binary(&mut self, op: Bo, left: Handle<E>, right: Handle<E>) -> Handle<E> {
        self.expr(E::Binary { op, left, right })
    }
    fn cast(&mut self, expr: Handle<E>, kind: ScalarKind, convert: Option<u8>) -> Handle<E> {
        self.expr(E::As { expr, kind, convert })
    }
    /// The pointer to word `base + k` of a harness buffer.
    fn word(&mut self, buffer: Handle<GlobalVariable>, base: Handle<E>, k: u32) -> Handle<E> {
        let g = self.expr(E::GlobalVariable(buffer));
        let k = self.u32(k);
        let index = self.binary(Bo::Add, base, k);
        self.expr(E::Access { base: g, index })
    }
    /// A scalar read from the input words at `*at`.
    fn read_scalar(&mut self, s: Scalar, base: Handle<E>, at: &mut u32) -> Result<Handle<E>, String> {
        let mut next = |b: &mut Self| {
            let pointer = b.word(b.inputs, base, *at);
            *at += 1;
            b.expr(E::Load { pointer })
        };
        Ok(match (s.kind, s.width) {
            (ScalarKind::Uint, 4) => next(self),
            (ScalarKind::Sint | ScalarKind::Float, 4) => {
                let w = next(self);
                self.cast(w, s.kind, None)
            }
            (ScalarKind::Bool, _) => {
                let w = next(self);
                let zero = self.u32(0);
                self.binary(Bo::NotEqual, w, zero)
            }
            (kind, 8) => {
                let (lo, hi) = (next(self), next(self));
                let lo = self.cast(lo, ScalarKind::Uint, Some(8));
                let hi = self.cast(hi, ScalarKind::Uint, Some(8));
                let thirty_two = self.u32(32);
                let hi = self.binary(Bo::ShiftLeft, hi, thirty_two);
                let both = self.binary(Bo::InclusiveOr, hi, lo);
                if kind == ScalarKind::Uint {
                    both
                } else {
                    self.cast(both, kind, None)
                }
            }
            (kind, width) => return Err(format!("a {}-bit {kind:?} argument", width * 8)),
        })
    }
    /// A value of `ty` read from the input words at `*at`.
    fn read(&mut self, ty: Handle<Type>, base: Handle<E>, at: &mut u32) -> Result<Handle<E>, String> {
        let inner = self.module.types[ty].inner.clone();
        let components = match inner {
            TypeInner::Scalar(s) => return self.read_scalar(s, base, at),
            TypeInner::Vector { size, scalar } => (0..size as u32)
                .map(|_| self.read_scalar(scalar, base, at))
                .collect::<Result<Vec<_>, _>>()?,
            TypeInner::Array {
                base: element,
                size: ArraySize::Constant(n),
                ..
            } => (0..n.get())
                .map(|_| self.read(element, base, at))
                .collect::<Result<Vec<_>, _>>()?,
            TypeInner::Struct { members, .. } => members
                .iter()
                .map(|m| self.read(m.ty, base, at))
                .collect::<Result<Vec<_>, _>>()?,
            other => return Err(format!("an argument of {other:?}")),
        };
        Ok(self.expr(E::Compose { ty, components }))
    }
    /// `value`, a scalar, written to the output words at `*at`.
    fn write_scalar(
        &mut self,
        s: Scalar,
        value: Handle<E>,
        base: Handle<E>,
        at: &mut u32,
    ) -> Result<(), String> {
        let mut put = |b: &mut Self, w: Handle<E>| {
            let pointer = b.word(b.outputs, base, *at);
            *at += 1;
            b.statement(Statement::Store { pointer, value: w });
        };
        match (s.kind, s.width) {
            (ScalarKind::Uint, 4) => put(self, value),
            (ScalarKind::Sint | ScalarKind::Float, 4) => {
                let w = self.cast(value, ScalarKind::Uint, None);
                put(self, w);
            }
            (ScalarKind::Bool, _) => {
                let (one, zero) = (self.u32(1), self.u32(0));
                let w = self.expr(E::Select {
                    condition: value,
                    accept: one,
                    reject: zero,
                });
                put(self, w);
            }
            (kind, 8) => {
                let u = if kind == ScalarKind::Uint {
                    value
                } else {
                    self.cast(value, ScalarKind::Uint, None)
                };
                let lo = self.cast(u, ScalarKind::Uint, Some(4));
                let thirty_two = self.u32(32);
                let high = self.binary(Bo::ShiftRight, u, thirty_two);
                let hi = self.cast(high, ScalarKind::Uint, Some(4));
                put(self, lo);
                put(self, hi);
            }
            (kind, width) => return Err(format!("a {}-bit {kind:?} result", width * 8)),
        }
        Ok(())
    }
    /// `value` of `ty` written to the output words at `*at`.
    fn write(
        &mut self,
        ty: Handle<Type>,
        value: Handle<E>,
        base: Handle<E>,
        at: &mut u32,
    ) -> Result<(), String> {
        let inner = self.module.types[ty].inner.clone();
        match inner {
            TypeInner::Scalar(s) => self.write_scalar(s, value, base, at)?,
            TypeInner::Vector { size, scalar } => {
                for k in 0..size as u32 {
                    let lane = self.expr(E::AccessIndex {
                        base: value,
                        index: k,
                    });
                    self.write_scalar(scalar, lane, base, at)?;
                }
            }
            TypeInner::Array {
                base: element,
                size: ArraySize::Constant(n),
                ..
            } => {
                for k in 0..n.get() {
                    let e = self.expr(E::AccessIndex {
                        base: value,
                        index: k,
                    });
                    self.write(element, e, base, at)?;
                }
            }
            TypeInner::Struct { members, .. } => {
                for (k, m) in members.iter().enumerate() {
                    let e = self.expr(E::AccessIndex {
                        base: value,
                        index: k as u32,
                    });
                    self.write(m.ty, e, base, at)?;
                }
            }
            other => return Err(format!("a result of {other:?}")),
        }
        Ok(())
    }
}

/// The type `ty` with no binding on any member (a structure that carried an entry point's inputs or outputs, which is
/// now a plain function's), replaced where it is.
fn unbind(module: &mut naga::Module, ty: Handle<Type>) -> Result<(), String> {
    let t = module.types[ty].clone();
    let TypeInner::Struct { members, span } = t.inner else {
        return Ok(());
    };
    if members.iter().all(|m| m.binding.is_none()) {
        return Ok(());
    }
    let plain = Type {
        name: t.name,
        inner: TypeInner::Struct {
            members: members
                .into_iter()
                .map(|m| naga::StructMember { binding: None, ..m })
                .collect(),
            span,
        },
    };
    if module.types.get(&plain).is_some() {
        return Err(
            "a structure of the entry point's is also a type of the module without its bindings".into(),
        );
    }
    module.types.replace(ty, plain);
    Ok(())
}

fn harness_buffer(
    module: &mut naga::Module,
    name: &str,
    binding: u32,
    write: bool,
) -> Handle<GlobalVariable> {
    let word = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::UNDEFINED,
    );
    let ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Array {
                base: word,
                size: ArraySize::Dynamic,
                stride: 4,
            },
        },
        Span::UNDEFINED,
    );
    let access = if write {
        StorageAccess::LOAD | StorageAccess::STORE
    } else {
        StorageAccess::LOAD
    };
    module.global_variables.append(
        GlobalVariable {
            name: Some(name.into()),
            space: AddressSpace::Storage { access },
            binding: Some(ResourceBinding {
                group: GROUP,
                binding,
            }),
            ty,
            init: None,
            memory_decorations: Default::default(),
        },
        Span::UNDEFINED,
    )
}

/// naga's validator's error, each link of its chain.
pub fn validation_error(e: &naga::WithSpan<naga::valid::ValidationError>) -> String {
    let mut message = e.as_inner().to_string();
    let mut cause = std::error::Error::source(e.as_inner());
    while let Some(c) = cause {
        message.push_str(&format!(": {c}"));
        cause = c.source();
    }
    message
}

fn validate(module: &naga::Module) -> Result<naga::valid::ModuleInfo, String> {
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(module)
    .map_err(|e| validation_error(&e))
}

/// The program of `wgsl` ready to run its entry point `entry` (the only one where `None`).
pub fn wrap(wgsl: &str, entry: Option<&str>, workgroup_init: WorkgroupInit) -> Result<Wrapped, String> {
    let mut module = naga::front::wgsl::parse_str(wgsl)
        .map_err(|e| format!("naga's WGSL front end: {}", e.emit_to_string(wgsl)))?;
    validate(&module).map_err(|e| format!("naga's validator, on the program: {e}"))?;
    let index = match entry {
        Some(name) => module
            .entry_points
            .iter()
            .position(|e| e.name == name)
            .ok_or_else(|| format!("no entry point {name}"))?,
        None if module.entry_points.len() == 1 => 0,
        None => {
            return Err(format!(
                "{} entry points and none named",
                module.entry_points.len()
            ));
        }
    };
    let stage = module.entry_points[index].stage;
    if stage == ShaderStage::Compute {
        let mut info = validate(&module)?;
        let mut zeroes_workgroup_memory = false;
        if workgroup_init == WorkgroupInit::Loop {
            // a program that cannot be changed this way keeps wgpu's initialisation
            let mut changed = module.clone();
            if let Ok(zeroed) = crate::workgroup::initialize(&mut changed, &info, index) {
                info = validate(&changed).map_err(|e| {
                    format!("the workgroup initialisation does not validate (a harness bug): {e}")
                })?;
                module = changed;
                zeroes_workgroup_memory = zeroed;
            }
        }
        let name = module.entry_points[index].name.clone();
        return Ok(Wrapped {
            module,
            info,
            stage,
            entry: name,
            entry_index: index,
            arguments: Vec::new(),
            results: Vec::new(),
            zeroes_workgroup_memory,
        });
    }
    if !matches!(stage, ShaderStage::Vertex | ShaderStage::Fragment) {
        return Err(format!("a {stage:?} entry point"));
    }
    let ep = module.entry_points.remove(index);
    let mut program = ep.function;
    program.name = Some("harness_program".into());
    let arguments = program
        .arguments
        .iter()
        .map(|a| words(&module, a.ty))
        .collect::<Result<Vec<_>, _>>()?;
    let result_ty = program.result.as_ref().map(|r| r.ty);
    let results = match result_ty.map(|ty| &module.types[ty].inner) {
        None => Vec::new(),
        Some(TypeInner::Struct { members, .. }) => members
            .iter()
            .map(|m| words(&module, m.ty))
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => vec![words(&module, result_ty.unwrap_or_else(|| unreachable!()))?],
    };
    for a in &mut program.arguments {
        a.binding = None;
    }
    if let Some(r) = &mut program.result {
        r.binding = None;
    }
    let types: Vec<Handle<Type>> = program.arguments.iter().map(|a| a.ty).chain(result_ty).collect();
    for ty in types {
        unbind(&mut module, ty)?;
    }
    let argument_types: Vec<Handle<Type>> = program.arguments.iter().map(|a| a.ty).collect();
    let callee = module.functions.append(program, Span::UNDEFINED);
    let inputs = harness_buffer(&mut module, "harness_inputs", INPUTS, false);
    let outputs = harness_buffer(&mut module, "harness_outputs", OUTPUTS, true);
    let in_words: u32 = arguments.iter().sum();
    let out_words: u32 = 1 + results.iter().sum::<u32>();
    let mut b = Build {
        module: &mut module,
        f: Function::default(),
        body: Block::new(),
        start: None,
        inputs,
        outputs,
    };
    let uvec3 = b.ty(TypeInner::Vector {
        size: VectorSize::Tri,
        scalar: Scalar::U32,
    });
    let vec4 = b.ty(TypeInner::Vector {
        size: VectorSize::Quad,
        scalar: Scalar::F32,
    });
    // which invocation this is: the workgroup's number, or the pixel's place among the 2x2 quads of a target two
    // pixels high
    let invocation = if stage == ShaderStage::Vertex {
        b.f.arguments.push(FunctionArgument {
            name: Some("group".into()),
            ty: uvec3,
            binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
        });
        let id = b.expr(E::FunctionArgument(0));
        b.expr(E::AccessIndex { base: id, index: 0 })
    } else {
        b.f.arguments.push(FunctionArgument {
            name: Some("pixel".into()),
            ty: vec4,
            binding: Some(Binding::BuiltIn(BuiltIn::Position { invariant: false })),
        });
        b.f.result = Some(FunctionResult {
            ty: vec4,
            binding: Some(Binding::Location {
                location: 0,
                interpolation: None,
                sampling: None,
                blend_src: None,
                per_primitive: false,
            }),
        });
        let position = b.expr(E::FunctionArgument(0));
        let px = b.expr(E::AccessIndex {
            base: position,
            index: 0,
        });
        let py = b.expr(E::AccessIndex {
            base: position,
            index: 1,
        });
        let x = b.cast(px, ScalarKind::Uint, Some(4));
        let y = b.cast(py, ScalarKind::Uint, Some(4));
        let one = b.u32(1);
        let quad = b.binary(Bo::ShiftRight, x, one);
        let four = b.u32(4);
        let first = b.binary(Bo::Multiply, quad, four);
        let column = b.binary(Bo::And, x, one);
        let row = b.binary(Bo::And, y, one);
        let row = b.binary(Bo::ShiftLeft, row, one);
        let lane = b.binary(Bo::InclusiveOr, column, row);
        b.binary(Bo::Add, first, lane)
    };
    let n = b.u32(in_words);
    let in_base = b.binary(Bo::Multiply, invocation, n);
    let n = b.u32(out_words);
    let out_base = b.binary(Bo::Multiply, invocation, n);
    let mut at = 0;
    let mut values = Vec::new();
    for ty in argument_types {
        values.push(b.read(ty, in_base, &mut at)?);
    }
    let result = result_ty.map(|_| b.expr(E::CallResult(callee)));
    b.statement(Statement::Call {
        function: callee,
        arguments: values,
        result,
    });
    if let (Some(value), Some(ty)) = (result, result_ty) {
        let mut at = 1;
        match b.module.types[ty].inner.clone() {
            TypeInner::Struct { members, .. } => {
                for (k, m) in members.iter().enumerate() {
                    let member = b.expr(E::AccessIndex {
                        base: value,
                        index: k as u32,
                    });
                    b.write(m.ty, member, out_base, &mut at)?;
                }
            }
            _ => b.write(ty, value, out_base, &mut at)?,
        }
    }
    // the word saying it returned, last: a discarded pixel's writes after its discard have no effect
    let returned = b.word(outputs, out_base, 0);
    let one = b.u32(1);
    b.statement(Statement::Store {
        pointer: returned,
        value: one,
    });
    if stage == ShaderStage::Fragment {
        let zero = b.expr(E::ZeroValue(vec4));
        b.statement(Statement::Return { value: Some(zero) });
    }
    b.flush();
    let mut f = std::mem::take(&mut b.f);
    f.body = std::mem::take(&mut b.body);
    f.name = Some(ENTRY.into());
    let wrapper_stage = if stage == ShaderStage::Vertex {
        ShaderStage::Compute
    } else {
        ShaderStage::Fragment
    };
    module.entry_points.push(EntryPoint {
        name: ENTRY.into(),
        stage: wrapper_stage,
        early_depth_test: None,
        workgroup_size: if stage == ShaderStage::Vertex {
            [1, 1, 1]
        } else {
            [0; 3]
        },
        workgroup_size_overrides: None,
        function: f,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });
    let entry_index = module.entry_points.len() - 1;
    let info = validate(&module)
        .map_err(|e| format!("the harness's entry point does not validate (a harness bug): {e}"))?;
    Ok(Wrapped {
        module,
        info,
        stage,
        entry: ENTRY.into(),
        entry_index,
        arguments,
        results,
        zeroes_workgroup_memory: false,
    })
}
