//! What a program reaches besides its registers: constant buffers by byte address, raw and structured buffers by word,
//! typed buffers by element, variables (private arrays and workgroup memory), the atomics of each, the hidden counters of
//! structured views, barriers, and the outputs.
//!
//! A constant buffer is rows of `vec4<u32>`; a raw or structured buffer is a runtime-sized array of 32-bit words
//! (`array<u32>`, or of atomic words where an atomic reaches it); a typed buffer is an array of `vec4<u32>`, each element
//! four 32-bit lanes the host wrote in the view's type (where an atomic reaches it, structures of four atomic words: the
//! same bytes). Every read and write is bounds-checked against the array's length, as the IR states: a word past the end
//! reads zero and is not written, and an atomic there changes nothing and gives zero.

use alloc::{
    collections::BTreeMap,
    format,
    string::{String, ToString},
    vec::Vec,
};

use iridia_ir::interface::{BufferKind, ResourceId, ResourceKind, StorageTarget};
use iridia_ir::{
    Address, AtomicOp, Barrier, CounterOp, Function, Memory, Op, OutOfBounds, Res, Scalar, Type, ValueId,
    VarId,
};

use crate::emit::{R, Writer};
use crate::text::{self, err};

/// The scalar of the atomic words each resource and variable an atomic reaches is made of: `i32` where its only minimum and
/// maximum operations are signed, `u32` otherwise (an operation of the other signedness is a compare-exchange loop).
#[derive(Debug, Default)]
pub(crate) struct Atomics {
    pub resources: BTreeMap<ResourceId, Scalar>,
    pub vars: BTreeMap<u32, Scalar>,
}

impl Atomics {
    pub(crate) fn of(f: &Function) -> R<Atomics> {
        // per memory: a signed minimum or maximum, an unsigned one
        let mut seen: BTreeMap<(bool, u32), (bool, bool)> = BTreeMap::new();
        for v in &f.values {
            let Op::Atomic { memory, op, .. } = v.op else {
                continue;
            };
            if v.ty.scalar.bits() != 32 {
                return Err(err(format!(
                    "a {}-bit atomic is not written yet",
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
                Scalar::I32
            } else {
                Scalar::U32
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

/// The word an atomic reaches: its place, its scalar, and whether it is inside its memory (`None`: always).
struct Word {
    place: String,
    scalar: Scalar,
    inside: Option<String>,
}

/// The resources whose hidden counter the function changes.
pub(crate) fn counted(f: &Function) -> Vec<ResourceId> {
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

fn masked(mask: u8) -> impl Iterator<Item = u8> {
    (0..4u8).filter(move |n| mask >> n & 1 != 0)
}

impl Writer<'_> {
    // kinds of buffer

    /// Whether the resource is a raw or structured buffer (read or storage): words.
    pub(crate) fn words_buffer(&self, id: ResourceId) -> R<bool> {
        Ok(matches!(
            self.kind(id)?,
            ResourceKind::Buffer(BufferKind::Raw | BufferKind::Structured { .. })
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Raw | BufferKind::Structured { .. }),
                    ..
                }
        ))
    }

    /// Whether the resource is a typed buffer (read or storage).
    pub(crate) fn typed_buffer(&self, id: ResourceId) -> R<bool> {
        Ok(matches!(
            self.kind(id)?,
            ResourceKind::Buffer(BufferKind::Typed(_))
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Typed(_)),
                    ..
                }
        ))
    }

    // constant buffers

    /// A constant buffer read: rows of `vec4<u32>`, the result's bits from the row (and lane) the address names. A read at
    /// a fixed address may name any 32-bit word (a bank of 32-bit booleans, four to a row); an indexed one whole rows.
    pub(crate) fn load_constants(
        &mut self,
        resource: Res,
        address: Address,
        oob: OutOfBounds,
        t: Type,
    ) -> R<String> {
        let (index, stride, offset) = (address.index, address.stride, address.offset);
        let rows_apart = stride.is_multiple_of(16) || index.is_none();
        if t.scalar.bits() != 32 || t.scalar == Scalar::Bool || !offset.is_multiple_of(4) || !rows_apart {
            return Err(err(format!(
                "a buffer read of {t} at stride {stride}, offset {offset} is not written yet"
            )));
        }
        let lane = (offset % 16) / 4;
        if lane.saturating_add(u32::from(t.lanes)) > 4 {
            return Err(err(format!(
                "a buffer read of {t} across a row at offset {offset}"
            )));
        }
        let rows = match self.kind(resource.id)? {
            ResourceKind::ConstantBuffer { layout } => layout.size.div_ceil(16),
            _ => {
                return Err(err(format!(
                    "resource {} is not a constant buffer",
                    resource.id.0
                )));
            }
        };
        let bank = self.buffer(resource)?;
        let first = offset / 16;
        let (row, inside) = match index {
            None => {
                if first >= rows {
                    return Err(err(format!("a read of row {first} of a buffer of {rows}")));
                }
                (format!("{bank}[{first}u]"), None)
            }
            Some(i) => {
                // the byte address `offset + stride * index` in wrapping 32-bit arithmetic, as the IR states it (a negative
                // index wraps far past the end), and its row; the row read is clamped into the buffer, and where the
                // address is outside it the read gives what the rule says
                let index = self.index_u32(i)?;
                let scaled = format!("({index} * {stride}u)");
                let address = if offset == 0 {
                    scaled
                } else {
                    format!("({scaled} + {offset}u)")
                };
                let row = self.tmp(&format!("({address} >> 4u)"));
                let inside = self.tmp(&format!("({row} < {rows}u)"));
                (
                    format!("{bank}[min({row}, {}u)]", rows.saturating_sub(1)),
                    Some(inside),
                )
            }
        };
        let bits = if t.lanes == 1 {
            format!("{row}.{}", text::lane(u8::try_from(lane).unwrap_or(0))?)
        } else if lane == 0 && t.lanes == 4 {
            row
        } else {
            let first = u8::try_from(lane).unwrap_or(0);
            format!("{row}.{}", text::swizzle(first..first.saturating_add(t.lanes))?)
        };
        let v = self.bitcast(&bits, t.with(Scalar::U32), t.scalar)?;
        Ok(match (inside, oob) {
            (Some(inside), OutOfBounds::Zero) => {
                let zero = self.konst(t, t.scalar, 0)?;
                format!("select({zero}, {v}, {inside})")
            }
            _ => v,
        })
    }

    // raw and structured buffers

    /// The word index (`u32`) of a byte address: `(offset + stride * index) >> 2` in wrapping 32-bit arithmetic.
    fn word_index(&self, address: Address) -> R<String> {
        let bytes = match address.index {
            None => format!("{}u", address.offset),
            Some(i) => {
                let index = self.index_u32(i)?;
                let scaled = if address.stride == 1 {
                    index
                } else {
                    format!("({index} * {}u)", address.stride)
                };
                if address.offset == 0 {
                    scaled
                } else {
                    format!("({scaled} + {}u)", address.offset)
                }
            }
        };
        Ok(format!("({bytes} >> 2u)"))
    }

    /// `arrayLength` of a buffer's runtime-sized array.
    fn length(&mut self, r: Res) -> R<String> {
        let g = self.buffer(r)?;
        Ok(self.tmp(&format!("arrayLength(&{g})")))
    }

    /// The place of element `index` (`u32`) of a resource's array, clamped into it, and whether `index` is inside.
    fn element_of(&mut self, r: Res, index: &str) -> R<(String, String)> {
        let len = self.length(r)?;
        let inside = self.tmp(&format!("({index} < {len})"));
        let g = self.buffer(r)?;
        Ok((format!("{g}[min({index}, ({len} - 1u))]"), inside))
    }

    /// The word `k` past `w`.
    fn plus(&mut self, w: &str, k: u8) -> String {
        if k == 0 {
            w.to_string()
        } else {
            self.tmp(&format!("({w} + {k}u)"))
        }
    }

    /// A word of a buffer read as `u32`: an atomic `i32` word's bits.
    fn word_u32(&self, id: ResourceId, place: &str) -> String {
        match self.atomics.resources.get(&id) {
            Some(Scalar::I32) => format!("bitcast<u32>(atomicLoad(&{place}))"),
            Some(_) => format!("atomicLoad(&{place})"),
            None => place.to_string(),
        }
    }

    /// `LoadBuffer` of a raw or structured buffer: the result's lanes as consecutive words from the byte address, each word
    /// past the end zero.
    pub(crate) fn load_words(&mut self, r: Res, address: Address, t: Type) -> R<String> {
        if t.scalar.bits() != 32 || t.scalar == Scalar::Bool {
            return Err(err(format!("a buffer read of {t} is not written yet")));
        }
        let w = self.word_index(address)?;
        let w = self.tmp(&w);
        let mut lanes = Vec::new();
        for k in 0..t.lanes {
            let wk = self.plus(&w, k);
            let (place, inside) = self.element_of(r, &wk)?;
            let word = self.word_u32(r.id, &place);
            lanes.push(format!("select(0u, {word}, {inside})"));
        }
        let bits = match lanes.as_slice() {
            [one] => one.clone(),
            _ => format!("{}({})", self.ty(t.with(Scalar::U32))?, lanes.join(", ")),
        };
        self.bitcast(&bits, t.with(Scalar::U32), t.scalar)
    }

    /// `if (inside) { place = value; }`, an atomic word stored atomically.
    fn store_if(&mut self, id: Option<ResourceId>, inside: &str, place: &str, value: &str) -> R<()> {
        let atomic = id.is_some_and(|id| self.atomics.resources.contains_key(&id));
        let store = if atomic {
            format!("atomicStore(&{place}, {value});")
        } else {
            format!("{place} = {value};")
        };
        self.line(&format!("if ({inside}) {{ {store} }}"));
        Ok(())
    }

    /// `StoreBuffer` of a raw or structured buffer: each word inside the buffer takes its lane.
    pub(crate) fn store_words(&mut self, r: Res, address: Address, value: ValueId) -> R<()> {
        let t = self.vty(value)?;
        if t.scalar.bits() != 32 || t.scalar == Scalar::Bool {
            return Err(err(format!("a buffer write of {t} is not written yet")));
        }
        let x = self.get(value)?;
        let word = self.atomics.resources.get(&r.id).copied().unwrap_or(Scalar::U32);
        let bits = self.bitcast(&x, t, word)?;
        let bits = self.tmp(&bits);
        let w = self.word_index(address)?;
        let w = self.tmp(&w);
        for k in 0..t.lanes {
            let lane = self.lane(&bits, t.with(word), k)?;
            let wk = self.plus(&w, k);
            let (place, inside) = self.element_of(r, &wk)?;
            self.store_if(Some(r.id), &inside, &place, &lane)?;
        }
        Ok(())
    }

    // typed buffers

    /// The place of lane `lane` of element `index` of a typed buffer an atomic reaches (its element clamped into the
    /// buffer), and whether the element is inside it.
    fn element_word(&mut self, r: Res, index: &str, lane: u8) -> R<(String, String)> {
        let (element, inside) = self.element_of(r, index)?;
        Ok((format!("{element}.w{lane}"), inside))
    }

    /// `LoadTexel` of a typed buffer: element `coord`'s four lanes in the view's type, zero past the end.
    pub(crate) fn load_typed_element(&mut self, r: Res, coord: ValueId, t: Type) -> R<String> {
        if t.lanes != 4 || t.scalar.bits() != 32 || t.scalar == Scalar::Bool {
            return Err(err(format!("a typed buffer read of {t}")));
        }
        let index = self.index_u32(coord)?;
        let index = self.tmp(&index);
        let zero = self.konst(Type::U32X4, Scalar::U32, 0)?;
        let bits = if self.atomics.resources.contains_key(&r.id) {
            let mut lanes = Vec::new();
            let mut inside = None;
            for k in 0..4 {
                let (place, at) = self.element_word(r, &index, k)?;
                lanes.push(self.word_u32(r.id, &place));
                inside = Some(at);
            }
            let inside = inside.ok_or_else(|| err("an element of no lanes"))?;
            format!("select({zero}, vec4<u32>({}), {inside})", lanes.join(", "))
        } else {
            let (place, inside) = self.element_of(r, &index)?;
            format!("select({zero}, {place}, {inside})")
        };
        self.bitcast(&bits, Type::U32X4, t.scalar)
    }

    /// `StoreTexel` of a typed buffer: element `coord` takes the value's four lanes, where it is inside.
    pub(crate) fn store_typed_element(&mut self, r: Res, coord: ValueId, value: ValueId) -> R<()> {
        let t = self.vty(value)?;
        if t.lanes != 4 || t.scalar.bits() != 32 || t.scalar == Scalar::Bool {
            return Err(err(format!("a typed buffer write of {t}")));
        }
        let x = self.get(value)?;
        let bits = self.bitcast(&x, t, Scalar::U32)?;
        let bits = self.tmp(&bits);
        let index = self.index_u32(coord)?;
        let index = self.tmp(&index);
        if let Some(&scalar) = self.atomics.resources.get(&r.id) {
            for k in 0..4 {
                let (place, inside) = self.element_word(r, &index, k)?;
                let lane = self.lane(&bits, Type::U32X4, k)?;
                let lane = if scalar == Scalar::I32 {
                    format!("bitcast<i32>({lane})")
                } else {
                    lane
                };
                self.store_if(Some(r.id), &inside, &place, &lane)?;
            }
            return Ok(());
        }
        let (place, inside) = self.element_of(r, &index)?;
        self.store_if(None, &inside, &place, &bits)
    }

    /// `Query::Elements` of a buffer: its bytes for a raw buffer, its elements for a structured or typed one.
    pub(crate) fn elements(&mut self, r: Res) -> R<String> {
        let kind = match self.kind(r.id)? {
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
            BufferKind::Raw => format!("({len} * 4u)"),
            // the words the binding holds, as whole elements
            BufferKind::Structured { stride } => format!("(({len} * 4u) / {}u)", stride.max(1)),
        })
    }

    // counters

    /// `Op::Counter`: the view's hidden counter incremented (the count before) or decremented (the count after); an
    /// element of an array of views has a counter of its own, word n of the resource's packed counter buffer.
    pub(crate) fn counter(&mut self, r: Res, op: CounterOp) -> R<String> {
        let g = self
            .counters
            .iter()
            .find(|(id, _)| *id == r.id)
            .map(|(_, g)| g.clone())
            .ok_or_else(|| err(format!("resource {} has no counter", r.id.0)))?;
        let at = match r.index {
            Some(i) => self.index_u32(i)?,
            None => "0u".to_string(),
        };
        Ok(match op {
            CounterOp::Increment => format!("atomicAdd(&{g}[{at}], 1u)"),
            CounterOp::Decrement => format!("(atomicSub(&{g}[{at}], 1u) - 1u)"),
        })
    }

    // atomics

    /// The atomic word an atomic reaches.
    fn atomic_word(&mut self, memory: Memory) -> R<Word> {
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
                let w = self.tmp(&w);
                let (place, inside) = self.element_of(r, &w)?;
                Ok(Word {
                    place,
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
                let index = self.tmp(&index);
                let (place, inside) = self.element_word(r, &index, 0)?;
                Ok(Word {
                    place,
                    scalar,
                    inside: Some(inside),
                })
            }
            Memory::Image(..) => Err(err(
                "an atomic of a storage image's texel whose old value is read has no naga form: naga's image atomic gives none",
            )),
            Memory::Var(var, index) => {
                let scalar = *self
                    .atomics
                    .vars
                    .get(&var.0)
                    .ok_or_else(|| err(format!("variable {} has no atomic words", var.0)))?;
                let (place, inside) = match index {
                    None => (self.var_name(var)?.to_string(), None),
                    Some(i) => {
                        let (place, inside) = self.element(var, i, OutOfBounds::Zero)?;
                        (place, inside)
                    }
                };
                Ok(Word {
                    place,
                    scalar,
                    inside,
                })
            }
        }
    }

    /// `Op::Atomic`: the operation on the word, where it is inside its memory; the value the word held before, zero
    /// outside.
    pub(crate) fn atomic(
        &mut self,
        memory: Memory,
        op: AtomicOp,
        value: ValueId,
        compare: Option<ValueId>,
        t: Type,
    ) -> R<String> {
        let Word {
            place,
            scalar,
            inside,
        } = self.atomic_word(memory)?;
        let word_t = Type::scalar(scalar);
        let as_word = |w: &Self, v: ValueId| -> R<String> { w.bitcast(&w.get(v)?, w.vty(v)?, scalar) };
        let operand = as_word(self, value)?;
        let comparand = match compare {
            Some(c) => Some(as_word(self, c)?),
            None => None,
        };
        let zero = text::literal(scalar, 0)?;
        let result = self.var(&self.ty(word_t)?, Some(&zero));
        match inside {
            Some(inside) => {
                let body = self.capture(|w| {
                    w.atomic_into(&place, op, &operand, comparand.as_deref(), word_t, &result)
                })?;
                self.line(&format!("if ({inside}) {{"));
                self.out.push_str(&body);
                self.line("}");
            }
            None => self.atomic_into(&place, op, &operand, comparand.as_deref(), word_t, &result)?,
        }
        self.bitcast(&result, word_t, t.scalar)
    }

    /// The atomic itself, its old value stored in `result`. WGSL's compare-exchange is weak (it may fail with the word equal
    /// to the comparand), so the IR's is a loop until it exchanges or the word differs; a minimum or maximum of the other
    /// signedness than the word's is a compare-exchange loop.
    fn atomic_into(
        &mut self,
        place: &str,
        op: AtomicOp,
        operand: &str,
        comparand: Option<&str>,
        word_t: Type,
        result: &str,
    ) -> R<()> {
        let signed_word = word_t.scalar == Scalar::I32;
        let fun = match op {
            AtomicOp::Add => Some("atomicAdd"),
            AtomicOp::Sub => Some("atomicSub"),
            AtomicOp::And => Some("atomicAnd"),
            AtomicOp::Or => Some("atomicOr"),
            AtomicOp::Xor => Some("atomicXor"),
            AtomicOp::SMin if signed_word => Some("atomicMin"),
            AtomicOp::SMax if signed_word => Some("atomicMax"),
            AtomicOp::UMin if !signed_word => Some("atomicMin"),
            AtomicOp::UMax if !signed_word => Some("atomicMax"),
            AtomicOp::Exchange => Some("atomicExchange"),
            _ => None,
        };
        if let Some(fun) = fun {
            self.line(&format!("{result} = {fun}(&{place}, {operand});"));
            return Ok(());
        }
        // a loop of compare-exchanges: `seen` the word as last seen, `new` what the operation makes of it
        let word = self.ty(word_t)?;
        let expected = match op {
            AtomicOp::CompareExchange => comparand
                .ok_or_else(|| err("a compare-exchange without its comparand"))?
                .to_string(),
            _ => format!("atomicLoad(&{place})"),
        };
        let seen = self.var(&word, Some(&expected));
        let body = self.capture(|w| {
            let current = w.tmp(&seen);
            let new = match op {
                AtomicOp::CompareExchange => operand.to_string(),
                AtomicOp::SMin | AtomicOp::SMax | AtomicOp::UMin | AtomicOp::UMax => {
                    let signed = matches!(op, AtomicOp::SMin | AtomicOp::SMax);
                    let fun = if matches!(op, AtomicOp::SMin | AtomicOp::UMin) {
                        "min"
                    } else {
                        "max"
                    };
                    let as_kind = if signed { Scalar::I32 } else { Scalar::U32 };
                    let (a, b) = if signed_word == signed {
                        (current.clone(), operand.to_string())
                    } else {
                        (
                            w.bitcast(&current, word_t, as_kind)?,
                            w.bitcast(operand, word_t, as_kind)?,
                        )
                    };
                    let m = format!("{fun}({a}, {b})");
                    if signed_word == signed {
                        m
                    } else {
                        format!("bitcast<{word}>({m})")
                    }
                }
                _ => return Err(err(format!("atomic {op:?}"))),
            };
            let r = w.tmp(&format!("atomicCompareExchangeWeak(&{place}, {current}, {new})"));
            w.line(&format!("{seen} = {r}.old_value;"));
            w.line(&format!("{result} = {r}.old_value;"));
            // done when it exchanged, or (a compare-exchange) when the word is not the comparand
            let done = if op == AtomicOp::CompareExchange {
                format!("{r}.exchanged || {r}.old_value != {current}")
            } else {
                format!("{r}.exchanged")
            };
            w.line(&format!("if ({done}) {{"));
            w.line("    break;");
            w.line("}");
            Ok(())
        })?;
        self.line("loop {");
        self.out.push_str(&body);
        self.line("}");
        Ok(())
    }

    /// A barrier of the workgroup. WGSL's barriers are all execution barriers, and none reaches past the workgroup: a
    /// memory barrier without execution is written as one with it, and a device-wide one as the workgroup's.
    pub(crate) fn barrier(&mut self, b: Barrier) {
        let storage = b.storage.is_some();
        if !storage && !b.workgroup && !b.execution {
            return;
        }
        if storage {
            self.line("storageBarrier();");
        }
        if b.workgroup || !storage {
            self.line("workgroupBarrier();");
        }
    }

    // variables

    /// A value read from a variable as the IR's type `t`: a variable of atomic words of the other signedness gives the same
    /// bits.
    fn read_word(&self, var: VarId, e: &str, t: Type) -> R<String> {
        match self.atomics.vars.get(&var.0) {
            Some(word) if *word != t.scalar => self.bitcast(e, Type::scalar(*word), t.scalar),
            _ => Ok(e.to_string()),
        }
    }

    /// A value of the IR's type `t` as a variable holds it: the same bits as its atomic words' type.
    fn written_word(&self, var: VarId, e: &str, t: Type) -> R<String> {
        match self.atomics.vars.get(&var.0) {
            Some(word) if *word != t.scalar => self.bitcast(e, t, *word),
            _ => Ok(e.to_string()),
        }
    }

    pub(crate) fn load_var(&mut self, var: VarId, t: Type) -> R<String> {
        let name = self.var_name(var)?.to_string();
        let v = if self.atomics.vars.contains_key(&var.0) {
            format!("atomicLoad(&{name})")
        } else {
            name
        };
        self.read_word(var, &v, t)
    }

    /// The place of element `index` of array variable `var`, and where an index past its end reads zero and writes nothing
    /// (`OutOfBounds::Zero`), whether the index is inside: the place is then of the index clamped, which WGSL keeps in
    /// bounds. A signed index is taken as unsigned, so a negative one is past the end.
    pub(crate) fn element(
        &mut self,
        var: VarId,
        index: ValueId,
        oob: OutOfBounds,
    ) -> R<(String, Option<String>)> {
        let len = self
            .f
            .vars
            .get(var.0 as usize)
            .and_then(|v| v.len)
            .ok_or_else(|| err(format!("variable {} is not an array", var.0)))?;
        let name = self.var_name(var)?.to_string();
        let i = self.index_u32(index)?;
        match oob {
            OutOfBounds::Undef => Ok((format!("{name}[{i}]"), None)),
            OutOfBounds::Zero => {
                let i = if i.starts_with("bitcast") { self.tmp(&i) } else { i };
                let inside = self.tmp(&format!("({i} < {len}u)"));
                Ok((
                    format!("{name}[min({i}, {}u)]", len.saturating_sub(1)),
                    Some(inside),
                ))
            }
        }
    }

    pub(crate) fn load_element_of_var(
        &mut self,
        var: VarId,
        index: ValueId,
        oob: OutOfBounds,
        t: Type,
    ) -> R<String> {
        let (place, inside) = self.element(var, index, oob)?;
        let v = if self.atomics.vars.contains_key(&var.0) {
            format!("atomicLoad(&{place})")
        } else {
            place
        };
        let v = self.read_word(var, &v, t)?;
        Ok(match inside {
            None => v,
            Some(inside) => format!("select({}(), {v}, {inside})", self.ty(t)?),
        })
    }

    pub(crate) fn store_var(&mut self, var: VarId, x: ValueId) -> R<()> {
        let value = self.get(x)?;
        let value = self.written_word(var, &value, self.vty(x)?)?;
        let name = self.var_name(var)?.to_string();
        if self.atomics.vars.contains_key(&var.0) {
            self.line(&format!("atomicStore(&{name}, {value});"));
        } else {
            self.line(&format!("{name} = {value};"));
        }
        Ok(())
    }

    pub(crate) fn store_element_of_var(
        &mut self,
        var: VarId,
        index: ValueId,
        x: ValueId,
        oob: OutOfBounds,
    ) -> R<()> {
        let value = self.get(x)?;
        let value = self.written_word(var, &value, self.vty(x)?)?;
        let (place, inside) = self.element(var, index, oob)?;
        let store = if self.atomics.vars.contains_key(&var.0) {
            format!("atomicStore(&{place}, {value});")
        } else {
            format!("{place} = {value};")
        };
        match inside {
            None => self.line(&store),
            Some(inside) => self.line(&format!("if ({inside}) {{ {store} }}")),
        }
        Ok(())
    }

    // outputs

    /// Lanes `mask` of output `slot` take the value: the lanes of the slot it does not name keep what they held.
    pub(crate) fn store_output(&mut self, slot: u32, mask: u8, x: ValueId) -> R<()> {
        let value = self.get(x)?;
        let local = self
            .outputs
            .get(slot as usize)
            .and_then(Clone::clone)
            .ok_or_else(|| err(format!("output {slot} is not carried")))?;
        let ot = self.slot_type(slot as usize)?;
        let lanes: Vec<u8> = masked(mask).collect();
        if lanes.len() == usize::from(ot.lanes) {
            self.line(&format!("{local} = {value};"));
            return Ok(());
        }
        let xt = self.vty(x)?;
        for (k, lane) in lanes.iter().enumerate() {
            let part = self.lane(&value, xt, u8::try_from(k).map_err(|_| err("a lane past w"))?)?;
            self.line(&format!("{local}.{} = {part};", text::lane(*lane)?));
        }
        Ok(())
    }

    /// The type an output slot's value has: its varying's own.
    pub(crate) fn slot_type(&self, slot: usize) -> R<Type> {
        self.program
            .interface
            .outputs
            .get(slot)
            .map(|v| iridia_ir::value_type(v.value))
            .ok_or_else(|| err(format!("output {slot} is not in the interface")))
    }

    /// The value a member of the output structure carries.
    pub(crate) fn member_value(&mut self, m: usize) -> R<String> {
        let member = self
            .members
            .get(m)
            .ok_or_else(|| err(format!("member {m} of the outputs")))?
            .clone();
        let carried = if member.clip {
            // every clip distance's lanes in order
            let mut parts = Vec::new();
            for slot in &member.slots {
                let t = self.slot_type(*slot)?;
                let local = self.output_local(*slot)?;
                for k in 0..t.lanes {
                    parts.push(self.lane(&local, t, k)?);
                }
            }
            format!("array<f32, {}>({})", parts.len(), parts.join(", "))
        } else {
            let slot = *member
                .slots
                .first()
                .ok_or_else(|| err(format!("member {m} carries no output")))?;
            let local = self.output_local(slot)?;
            // a pixel target narrower than four lanes is carried as four: the lanes it lacks are zero
            let ot = self.slot_type(slot)?;
            let ct = iridia_ir::value_type(member.carried);
            if ct.lanes == ot.lanes {
                local
            } else {
                let mut parts = Vec::new();
                for n in 0..ct.lanes {
                    parts.push(if n < ot.lanes {
                        self.lane(&local, ot, n)?
                    } else {
                        text::literal(ot.scalar, 0)?
                    });
                }
                format!("{}({})", self.ty(ct)?, parts.join(", "))
            }
        };
        if member.clip || member.carried == member.linked {
            Ok(carried)
        } else {
            let carried = self.tmp(&carried);
            crate::io::relink(&carried, member.carried, member.linked)
        }
    }

    fn output_local(&self, slot: usize) -> R<String> {
        self.outputs
            .get(slot)
            .and_then(Clone::clone)
            .ok_or_else(|| err(format!("output {slot} is not carried")))
    }
}
