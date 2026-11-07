//! What a compute program reaches besides its registers: buffers by byte address, typed buffers by element, workgroup
//! memory, the atomics of each, the hidden counters of structured views, and barriers.
//!
//! A raw or structured buffer is a runtime-sized array of 32-bit words (`array<u32>`, or of atomic words where an
//! atomic reaches it); a typed buffer is an array of `vec4<u32>`, each element four 32-bit lanes the host wrote in the
//! view's type (where an atomic reaches it, structures of four atomic words: the same bytes). Every read and write is bounds-checked against the array's length, as the IR states: a word past the
//! end reads zero and is not written, and an atomic there changes nothing and gives zero.

use alloc::{collections::BTreeMap, format, vec::Vec};

use iridia_ir::{
    Address, AtomicOp, Barrier, Function, Memory, Op, Res, Scalar, Scope, Type, ValueId, VarId,
    interface::{BufferKind, ResourceId, ResourceKind, StorageTarget},
};
use naga::{AtomicFunction, BinaryOperator as Bo, Block, Expression as E, Handle, ScalarKind, Statement};

use super::{Lower, err};
use crate::Error;

/// The scalar of the atomic words each resource and variable an atomic reaches is made of: `i32` where its only
/// minimum and maximum operations are signed, `u32` otherwise (an operation of the other signedness is a
/// compare-exchange loop).
#[derive(Default)]
pub(super) struct Atomics {
    pub resources: BTreeMap<ResourceId, naga::Scalar>,
    pub vars: BTreeMap<u32, naga::Scalar>,
}

impl Atomics {
    pub fn of(f: &Function) -> Result<Atomics, Error> {
        // per memory: a signed minimum or maximum, an unsigned one
        let mut seen: BTreeMap<(bool, u32), (bool, bool)> = BTreeMap::new();
        for v in &f.values {
            let Op::Atomic { memory, op, .. } = v.op else {
                continue;
            };
            if v.ty.scalar.bits() != 32 {
                return Err(err(format!(
                    "a {}-bit atomic is not lowered yet",
                    v.ty.scalar.bits()
                )));
            }
            let key = match memory {
                Memory::Buffer(r, _) | Memory::Image(r, _) => (false, r.id.0),
                Memory::Var(var, _) => (true, var.0),
            };
            let entry = seen.entry(key).or_default();
            entry.0 |= matches!(op, AtomicOp::SMin | AtomicOp::SMax);
            entry.1 |= matches!(op, AtomicOp::UMin | AtomicOp::UMax);
        }
        let mut atomics = Atomics::default();
        for ((var, id), (signed, unsigned)) in seen {
            let scalar = if signed && !unsigned {
                naga::Scalar::I32
            } else {
                naga::Scalar::U32
            };
            if var {
                atomics.vars.insert(id, scalar);
            } else {
                atomics.resources.insert(ResourceId(id), scalar);
            }
        }
        Ok(atomics)
    }
}

/// The word an atomic reaches: its pointer, its scalar, and whether it is inside its memory (`None`: always).
struct Word {
    pointer: Handle<E>,
    scalar: naga::Scalar,
    inside: Option<Handle<E>>,
}

/// The resources whose hidden counter the function changes.
pub(super) fn counted(f: &Function) -> Vec<ResourceId> {
    let mut out: Vec<ResourceId> = f
        .values
        .iter()
        .filter_map(|v| match v.op {
            Op::Counter(r, _) => Some(r.id),
            _ => None,
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

impl Lower<'_> {
    fn resource_kind(&self, id: ResourceId) -> Result<&ResourceKind, Error> {
        Ok(&self.resource(id)?.kind)
    }

    /// The words of a raw or structured buffer (read or storage), or `None` for another resource.
    pub(super) fn words_buffer(&self, id: ResourceId) -> Result<bool, Error> {
        Ok(matches!(
            self.resource_kind(id)?,
            ResourceKind::Buffer(BufferKind::Raw | BufferKind::Structured { .. })
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Raw | BufferKind::Structured { .. }),
                    ..
                }
        ))
    }

    /// Whether the resource is a typed buffer (read or storage).
    pub(super) fn typed_buffer(&self, id: ResourceId) -> Result<bool, Error> {
        Ok(matches!(
            self.resource_kind(id)?,
            ResourceKind::Buffer(BufferKind::Typed(_))
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Typed(_)),
                    ..
                }
        ))
    }

    /// `arrayLength` of a buffer's runtime-sized array.
    fn length(&mut self, r: Res) -> Result<Handle<E>, Error> {
        let pointer = self.buffer(r)?;
        Ok(self.expr(E::ArrayLength(pointer)))
    }

    /// The word index (`u32`) of a byte address: `(offset + stride * index) >> 2` in wrapping 32-bit arithmetic.
    fn word_index(&mut self, address: Address) -> Result<Handle<E>, Error> {
        let bytes = match address.index {
            None => self.literal_bits(Scalar::U32, u64::from(address.offset))?,
            Some(i) => {
                let index = self.index_u32(i)?;
                let scaled = if address.stride == 1 {
                    index
                } else {
                    let stride = self.literal_bits(Scalar::U32, u64::from(address.stride))?;
                    self.binary(Bo::Multiply, index, stride)
                };
                if address.offset == 0 {
                    scaled
                } else {
                    let offset = self.literal_bits(Scalar::U32, u64::from(address.offset))?;
                    self.binary(Bo::Add, scaled, offset)
                }
            }
        };
        let two = self.literal_bits(Scalar::U32, 2)?;
        Ok(self.binary(Bo::ShiftRight, bytes, two))
    }

    /// The pointer to element `index` (`u32`) of a buffer's array, clamped into it, and whether `index` is inside.
    fn element_of(&mut self, r: Res, index: Handle<E>) -> Result<(Handle<E>, Handle<E>), Error> {
        let len = self.length(r)?;
        let inside = self.binary(Bo::Less, index, len);
        let one = self.literal_bits(Scalar::U32, 1)?;
        let last = self.binary(Bo::Subtract, len, one);
        let clamped = self.math(naga::MathFunction::Min, index, &[last]);
        let base = self.buffer(r)?;
        let pointer = self.expr(E::Access { base, index: clamped });
        Ok((pointer, inside))
    }

    /// The `u32` index `k` past `w`.
    fn plus(&mut self, w: Handle<E>, k: u8) -> Result<Handle<E>, Error> {
        if k == 0 {
            return Ok(w);
        }
        let k = self.literal_bits(Scalar::U32, u64::from(k))?;
        Ok(self.binary(Bo::Add, w, k))
    }

    /// A word as `u32`: an atomic `i32` word's bits.
    fn word_u32(&mut self, id: ResourceId, word: Handle<E>) -> Handle<E> {
        if self.atomics.resources.get(&id) == Some(&naga::Scalar::I32) {
            self.bitcast(word, ScalarKind::Uint)
        } else {
            word
        }
    }

    /// `LoadBuffer` of a raw or structured buffer: the result's lanes as consecutive words from the byte address, each
    /// word past the end zero.
    pub(super) fn load_words(&mut self, r: Res, address: Address, t: Type) -> Result<Handle<E>, Error> {
        if t.scalar.bits() != 32 {
            return Err(err(format!("a buffer read of {t} is not lowered yet")));
        }
        let w = self.word_index(address)?;
        let zero = self.literal_bits(Scalar::U32, 0)?;
        let mut lanes = Vec::new();
        for k in 0..t.lanes {
            let wk = self.plus(w, k)?;
            let (pointer, inside) = self.element_of(r, wk)?;
            let word = self.expr(E::Load { pointer });
            let word = self.word_u32(r.id, word);
            lanes.push(self.select(inside, word, zero));
        }
        let bits = match lanes.as_slice() {
            [one] => *one,
            _ => {
                let ty = self.ty_of(t.with(Scalar::U32))?;
                self.expr(E::Compose {
                    ty,
                    components: lanes,
                })
            }
        };
        let kind = super::naga_scalar(t.scalar)?.kind;
        Ok(if kind == ScalarKind::Uint {
            bits
        } else {
            self.bitcast(bits, kind)
        })
    }

    /// `if inside { *pointer = value }`.
    fn store_if(&mut self, inside: Handle<E>, pointer: Handle<E>, value: Handle<E>) -> Result<(), Error> {
        self.open();
        self.statement(Statement::Store { pointer, value })?;
        let accept = self.close()?;
        self.statement(Statement::If {
            condition: inside,
            accept,
            reject: Block::new(),
        })
    }

    /// `StoreBuffer` of a raw or structured buffer: each word inside the buffer takes its lane.
    pub(super) fn store_words(&mut self, r: Res, address: Address, value: ValueId) -> Result<(), Error> {
        let t = self.vty(value)?;
        if t.scalar.bits() != 32 {
            return Err(err(format!("a buffer write of {t} is not lowered yet")));
        }
        let x = self.get(value)?;
        let word_kind = match self.atomics.resources.get(&r.id) {
            Some(s) => s.kind,
            None => ScalarKind::Uint,
        };
        let bits = if super::naga_scalar(t.scalar)?.kind == word_kind {
            x
        } else {
            self.bitcast(x, word_kind)
        };
        let w = self.word_index(address)?;
        for k in 0..t.lanes {
            let lane = self.lane(bits, t, k)?;
            let wk = self.plus(w, k)?;
            let (pointer, inside) = self.element_of(r, wk)?;
            self.store_if(inside, pointer, lane)?;
        }
        Ok(())
    }

    /// The atomic word of lane `lane` of element `index` of a typed buffer an atomic reaches (its element clamped
    /// into the buffer), and whether the element is inside it.
    fn element_word(&mut self, r: Res, index: Handle<E>, lane: u8) -> Result<(Handle<E>, Handle<E>), Error> {
        let (element, inside) = self.element_of(r, index)?;
        let pointer = self.expr(E::AccessIndex {
            base: element,
            index: u32::from(lane),
        });
        Ok((pointer, inside))
    }

    /// `LoadTexel` of a typed buffer: element `coord`'s four lanes in the view's type, zero past the end.
    pub(super) fn load_element(&mut self, r: Res, coord: ValueId, t: Type) -> Result<Handle<E>, Error> {
        if t.lanes != 4 || t.scalar.bits() != 32 {
            return Err(err(format!("a typed buffer read of {t}")));
        }
        let index = self.index_u32(coord)?;
        let ty = self.ty_of(Type::U32X4)?;
        let zero = self.expr(E::ZeroValue(ty));
        let bits = if self.atomics.resources.contains_key(&r.id) {
            let mut lanes = Vec::new();
            let mut inside = None;
            for k in 0..4 {
                let (pointer, at) = self.element_word(r, index, k)?;
                let word = self.expr(E::Load { pointer });
                lanes.push(self.word_u32(r.id, word));
                inside = Some(at);
            }
            let bits = self.expr(E::Compose {
                ty,
                components: lanes,
            });
            let inside = inside.ok_or_else(|| err("an element of no lanes"))?;
            self.select(inside, bits, zero)
        } else {
            let (pointer, inside) = self.element_of(r, index)?;
            let bits = self.expr(E::Load { pointer });
            self.select(inside, bits, zero)
        };
        let kind = super::naga_scalar(t.scalar)?.kind;
        Ok(if kind == ScalarKind::Uint {
            bits
        } else {
            self.bitcast(bits, kind)
        })
    }

    /// `StoreTexel` of a typed buffer: element `coord` takes the value's four lanes, where it is inside.
    pub(super) fn store_element(&mut self, r: Res, coord: ValueId, value: ValueId) -> Result<(), Error> {
        let t = self.vty(value)?;
        if t.lanes != 4 || t.scalar.bits() != 32 {
            return Err(err(format!("a typed buffer write of {t}")));
        }
        let x = self.get(value)?;
        let bits = if t.scalar == Scalar::U32 {
            x
        } else {
            self.bitcast(x, ScalarKind::Uint)
        };
        let index = self.index_u32(coord)?;
        if let Some(&scalar) = self.atomics.resources.get(&r.id) {
            for k in 0..4 {
                let (pointer, inside) = self.element_word(r, index, k)?;
                let lane = self.lane(bits, Type::U32X4, k)?;
                let lane = if scalar.kind == ScalarKind::Sint {
                    self.bitcast(lane, ScalarKind::Sint)
                } else {
                    lane
                };
                self.store_if(inside, pointer, lane)?;
            }
            return Ok(());
        }
        let (pointer, inside) = self.element_of(r, index)?;
        self.store_if(inside, pointer, bits)
    }

    /// `Query::Elements` of a buffer: its bytes for a raw buffer, its elements for a structured or typed one.
    pub(super) fn elements(&mut self, r: Res) -> Result<Handle<E>, Error> {
        let kind = match self.resource_kind(r.id)? {
            ResourceKind::Buffer(k)
            | ResourceKind::Storage {
                target: StorageTarget::Buffer(k),
                ..
            } => *k,
            other => return Err(err(format!("the elements of a {other:?}"))),
        };
        let len = self.length(r)?;
        Ok(match kind {
            BufferKind::Typed(_) => len,
            BufferKind::Raw => {
                let four = self.literal_bits(Scalar::U32, 4)?;
                self.binary(Bo::Multiply, len, four)
            }
            BufferKind::Structured { stride } => {
                // the words the binding holds, as whole elements
                let four = self.literal_bits(Scalar::U32, 4)?;
                let bytes = self.binary(Bo::Multiply, len, four);
                let stride = self.literal_bits(Scalar::U32, u64::from(stride.max(1)))?;
                self.binary(Bo::Divide, bytes, stride)
            }
        })
    }

    /// `Op::Counter`: the view's hidden counter incremented (the count before) or decremented (the count after); an
    /// element of an array of views has a counter of its own, word n of the resource's packed counter buffer.
    pub(super) fn counter(&mut self, r: Res, op: iridia_ir::CounterOp) -> Result<Handle<E>, Error> {
        let g = self
            .counters
            .iter()
            .find(|(id, _)| *id == r.id)
            .map(|(_, g)| *g)
            .ok_or_else(|| err(format!("resource {} has no counter", r.id.0)))?;
        let base = self.expr(E::GlobalVariable(g));
        let pointer = match r.index {
            Some(i) => {
                let index = self.index_u32(i)?;
                self.expr(E::Access { base, index })
            }
            None => self.expr(E::AccessIndex { base, index: 0 }),
        };
        let one = self.literal_bits(Scalar::U32, 1)?;
        let fun = match op {
            iridia_ir::CounterOp::Increment => AtomicFunction::Add,
            iridia_ir::CounterOp::Decrement => AtomicFunction::Subtract,
        };
        let ty = self.ty_of(Type::U32)?;
        let result = self.expr(E::AtomicResult {
            ty,
            comparison: false,
        });
        self.statement(Statement::Atomic {
            pointer,
            fun,
            value: one,
            result: Some(result),
        })?;
        Ok(match op {
            iridia_ir::CounterOp::Increment => result,
            iridia_ir::CounterOp::Decrement => self.binary(Bo::Subtract, result, one),
        })
    }

    /// The atomic word an atomic reaches.
    fn atomic_word(&mut self, memory: Memory) -> Result<Word, Error> {
        match memory {
            Memory::Buffer(r, address) => {
                if !self.words_buffer(r.id)? {
                    return Err(err(format!("a word atomic of resource {}", r.id.0)));
                }
                let scalar = *self
                    .atomics
                    .resources
                    .get(&r.id)
                    .ok_or_else(|| err(format!("resource {} has no atomic words", r.id.0)))?;
                let w = self.word_index(address)?;
                let (pointer, inside) = self.element_of(r, w)?;
                Ok(Word {
                    pointer,
                    scalar,
                    inside: Some(inside),
                })
            }
            Memory::Image(r, coord) if self.typed_buffer(r.id)? => {
                let scalar = *self
                    .atomics
                    .resources
                    .get(&r.id)
                    .ok_or_else(|| err(format!("resource {} has no atomic words", r.id.0)))?;
                let index = self.index_u32(coord)?;
                let (pointer, inside) = self.element_word(r, index, 0)?;
                Ok(Word {
                    pointer,
                    scalar,
                    inside: Some(inside),
                })
            }
            // the node lowers one whose old value nothing reads ([`Lower::image_atomic`])
            Memory::Image(..) => Err(err(
                "a texel atomic whose old value is read, or a compare-exchange of a texel, has no naga form: naga's \
                 image atomic gives no old value and takes no comparand",
            )),
            Memory::Var(var, index) => {
                let scalar = *self
                    .atomics
                    .vars
                    .get(&var.0)
                    .ok_or_else(|| err(format!("variable {} has no atomic words", var.0)))?;
                let (pointer, inside) = match index {
                    None => (self.var_pointer(var)?, None),
                    Some(i) => self.element(var, i, iridia_ir::OutOfBounds::Zero)?,
                };
                Ok(Word {
                    pointer,
                    scalar,
                    inside,
                })
            }
        }
    }

    /// `Op::Atomic`: the operation on the word, where it is inside its memory; the value the word held before, zero
    /// outside.
    pub(super) fn atomic(
        &mut self,
        memory: Memory,
        op: AtomicOp,
        value: ValueId,
        compare: Option<ValueId>,
        t: Type,
    ) -> Result<Handle<E>, Error> {
        let Word {
            pointer,
            scalar,
            inside,
        } = self.atomic_word(memory)?;
        let kind = scalar.kind;
        let as_word = |l: &mut Self, v: ValueId| -> Result<Handle<E>, Error> {
            let x = l.get(v)?;
            Ok(if super::naga_scalar(l.vty(v)?.scalar)?.kind == kind {
                x
            } else {
                l.bitcast(x, kind)
            })
        };
        let operand = as_word(self, value)?;
        let comparand = match compare {
            Some(c) => Some(as_word(self, c)?),
            None => None,
        };
        let word_t = if kind == ScalarKind::Sint {
            Type::I32
        } else {
            Type::U32
        };
        let result = self.local(word_t)?;
        let zero = self.literal_bits(word_t.scalar, 0)?;
        self.store_local(result, zero)?;
        if let Some(inside) = inside {
            self.open();
            self.atomic_into(pointer, op, operand, comparand, word_t, result)?;
            let accept = self.close()?;
            self.statement(Statement::If {
                condition: inside,
                accept,
                reject: Block::new(),
            })?;
        } else {
            self.atomic_into(pointer, op, operand, comparand, word_t, result)?;
        }
        let p = self.expr(E::LocalVariable(result));
        let old = self.expr(E::Load { pointer: p });
        let want = super::naga_scalar(t.scalar)?.kind;
        Ok(if want == kind {
            old
        } else {
            self.bitcast(old, want)
        })
    }

    /// The atomic itself, its result stored in `result`. naga's compare-exchange is weak (it may fail with the word
    /// equal to the comparand), so the IR's is a loop until it exchanges or the word differs; a minimum or maximum of
    /// the other signedness than the word's is a compare-exchange loop.
    fn atomic_into(
        &mut self,
        pointer: Handle<E>,
        op: AtomicOp,
        operand: Handle<E>,
        comparand: Option<Handle<E>>,
        word_t: Type,
        result: Handle<naga::LocalVariable>,
    ) -> Result<(), Error> {
        let signed_word = word_t == Type::I32;
        let fun = match op {
            AtomicOp::Add => Some(AtomicFunction::Add),
            AtomicOp::Sub => Some(AtomicFunction::Subtract),
            AtomicOp::And => Some(AtomicFunction::And),
            AtomicOp::Or => Some(AtomicFunction::InclusiveOr),
            AtomicOp::Xor => Some(AtomicFunction::ExclusiveOr),
            AtomicOp::SMin if signed_word => Some(AtomicFunction::Min),
            AtomicOp::SMax if signed_word => Some(AtomicFunction::Max),
            AtomicOp::UMin if !signed_word => Some(AtomicFunction::Min),
            AtomicOp::UMax if !signed_word => Some(AtomicFunction::Max),
            AtomicOp::Exchange => Some(AtomicFunction::Exchange { compare: None }),
            _ => None,
        };
        let ty = self.ty_of(word_t)?;
        if let Some(fun) = fun {
            let r = self.expr(E::AtomicResult {
                ty,
                comparison: false,
            });
            self.statement(Statement::Atomic {
                pointer,
                fun,
                value: operand,
                result: Some(r),
            })?;
            return self.store_local(result, r);
        }
        // a loop of compare-exchanges: `expected` the word as last seen, `new` what the operation makes of it
        let expected = match op {
            AtomicOp::CompareExchange => {
                comparand.ok_or_else(|| err("a compare-exchange without its comparand"))?
            }
            _ => self.expr(E::Load { pointer }),
        };
        let seen = self.local(word_t)?;
        self.store_local(seen, expected)?;
        self.open();
        let p = self.expr(E::LocalVariable(seen));
        let current = self.expr(E::Load { pointer: p });
        let new = match op {
            AtomicOp::CompareExchange => operand,
            AtomicOp::SMin | AtomicOp::SMax | AtomicOp::UMin | AtomicOp::UMax => {
                let as_kind = if matches!(op, AtomicOp::SMin | AtomicOp::SMax) {
                    ScalarKind::Sint
                } else {
                    ScalarKind::Uint
                };
                let (a, b) = if signed_word == (as_kind == ScalarKind::Sint) {
                    (current, operand)
                } else {
                    (self.bitcast(current, as_kind), self.bitcast(operand, as_kind))
                };
                let fun = if matches!(op, AtomicOp::SMin | AtomicOp::UMin) {
                    naga::MathFunction::Min
                } else {
                    naga::MathFunction::Max
                };
                let m = self.math(fun, a, &[b]);
                if signed_word == (as_kind == ScalarKind::Sint) {
                    m
                } else {
                    self.bitcast(
                        m,
                        if signed_word {
                            ScalarKind::Sint
                        } else {
                            ScalarKind::Uint
                        },
                    )
                }
            }
            _ => return Err(err(format!("atomic {op:?}"))),
        };
        let result_ty =
            self.module
                .generate_predeclared_type(naga::PredeclaredType::AtomicCompareExchangeWeakResult(
                    naga::Scalar {
                        kind: if signed_word {
                            ScalarKind::Sint
                        } else {
                            ScalarKind::Uint
                        },
                        width: 4,
                    },
                ));
        let r = self.expr(E::AtomicResult {
            ty: result_ty,
            comparison: true,
        });
        self.statement(Statement::Atomic {
            pointer,
            fun: AtomicFunction::Exchange {
                compare: Some(current),
            },
            value: new,
            result: Some(r),
        })?;
        let old = self.expr(E::AccessIndex { base: r, index: 0 });
        let exchanged = self.expr(E::AccessIndex { base: r, index: 1 });
        self.store_local(seen, old)?;
        self.store_local(result, old)?;
        // done when it exchanged, or (a compare-exchange) when the word is not the comparand
        let done = if op == AtomicOp::CompareExchange {
            let differs = self.binary(Bo::NotEqual, old, current);
            self.binary(Bo::LogicalOr, exchanged, differs)
        } else {
            exchanged
        };
        self.open();
        self.statement(Statement::Break)?;
        let leave = self.close()?;
        self.statement(Statement::If {
            condition: done,
            accept: leave,
            reject: Block::new(),
        })?;
        let body = self.close()?;
        self.statement(Statement::Loop {
            body,
            continuing: Block::new(),
            break_if: None,
        })
    }

    /// A barrier of the workgroup. WGSL's barriers are all execution barriers, and none reaches past the workgroup:
    /// a memory barrier without execution is lowered as one with it, and a device-wide one as the workgroup's.
    pub(super) fn barrier(&mut self, b: Barrier) -> Result<(), Error> {
        let mut flags = naga::Barrier::empty();
        if b.workgroup {
            flags |= naga::Barrier::WORK_GROUP;
        }
        if let Some(Scope::Workgroup | Scope::Device) = b.storage {
            flags |= naga::Barrier::STORAGE;
        }
        if flags.is_empty() && !b.execution {
            return Ok(());
        }
        if flags.is_empty() {
            flags = naga::Barrier::WORK_GROUP;
        }
        self.statement(Statement::ControlBarrier(flags))
    }

    /// A value read from a variable as the IR's type `t`: a variable of atomic words of the other signedness gives the
    /// same bits.
    pub(super) fn read_word(&mut self, var: VarId, v: Handle<E>, t: Type) -> Result<Handle<E>, Error> {
        let want = super::naga_scalar(t.scalar)?.kind;
        Ok(match self.atomics.vars.get(&var.0) {
            Some(word) if word.kind != want => self.bitcast(v, want),
            _ => v,
        })
    }

    /// A value of the IR's type `t` as a variable holds it: the same bits as its atomic words' type, for a variable of
    /// atomic words of the other signedness.
    pub(super) fn written_word(&mut self, var: VarId, v: Handle<E>, t: Type) -> Result<Handle<E>, Error> {
        let have = super::naga_scalar(t.scalar)?.kind;
        Ok(match self.atomics.vars.get(&var.0) {
            Some(word) if word.kind != have => self.bitcast(v, word.kind),
            _ => v,
        })
    }

    /// The pointer to a variable (a local, or a workgroup variable's global).
    pub(super) fn var_pointer(&mut self, var: VarId) -> Result<Handle<E>, Error> {
        let place = *self
            .vars
            .get(var.0 as usize)
            .ok_or_else(|| err(format!("variable {} does not exist", var.0)))?;
        Ok(match place {
            super::Place::Local(l) => self.expr(E::LocalVariable(l)),
            super::Place::Global(g) => self.expr(E::GlobalVariable(g)),
        })
    }
}
