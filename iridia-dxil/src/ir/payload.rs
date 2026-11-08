//! A task payload's memory: the bytes an amplification program hands its mesh thread groups (`DispatchMesh`) and a mesh
//! program reads (`GetMeshPayload`). Direct3D passes the payload as bytes, so the two programs may declare different
//! structures over them; the bytes are laid out as the module's datalayout lays out the payload's structure (each field
//! at its natural alignment, a bool four bytes). A variable the payload lives in is laid out so too, in bytes, where
//! the lift's other variables keep each value of 32 bits or fewer in a word of its own.
//!
//! A load or store of 4 or 8 bytes reaches whole words (natural alignment puts it at a multiple of its size); one of
//! 2 bytes (a `half` or an `i16`) reaches half a word: a load takes the half and widens it as the lift carries the
//! type, a store narrows it and writes that half alone, by a read and write of the word in private memory, and by two
//! atomic operations (clear the half, then set its bits) in thread-group memory, which another invocation may write
//! the other half of.

use std::collections::BTreeSet;

use iridia_ir::{AtomicOp, Binary, Memory, Op, OutOfBounds, Type, Unary, ValueId as IrValue};

use super::{Lift, Lv, Pointer};
use crate::Result;
use crate::layout::DataLayout;
use crate::setup::dx_opcode;
use dxil_reader::module::{Op as LlvmOp, Type as LlvmType, TypeId, Value, ValueId};

impl Lift<'_> {
    /// The allocas and module globals an amplification program hands to `DispatchMesh` as its payload, and a ray tracing
    /// program to `TraceRay` (its payload), `ReportHit` (the hit's attributes) and `CallShader` (the parameter): the
    /// variables laid out in bytes.
    pub(crate) fn payload_roots(&self) -> BTreeSet<ValueId> {
        let mut roots = BTreeSet::new();
        for inst in &self.f.instructions {
            let LlvmOp::Call { function, args, .. } = &inst.op else {
                continue;
            };
            let operand = match dx_opcode(self.m, self.f, *function, args) {
                Some(173) => 4,
                Some(157) => 15,
                Some(158) => 3,
                Some(159) => 2,
                _ => continue,
            };
            let Some(mut v) = args.get(operand).copied() else {
                continue;
            };
            // through getelementptrs and casts to the alloca or the global
            for _ in 0..=self.f.instructions.len() {
                if v < self.f.first {
                    if let Some(Value::Constant(dxil_reader::module::Constant::Gep { base, .. })) =
                        self.m.values.get(v)
                    {
                        v = *base;
                        continue;
                    }
                    roots.insert(v);
                    break;
                }
                let Some(def) = self
                    .f
                    .instructions
                    .iter()
                    .find(|i| i.result.is_some_and(|(r, _)| r == v))
                else {
                    break;
                };
                match &def.op {
                    LlvmOp::Gep { base, .. } => v = *base,
                    LlvmOp::Cast { value, .. } => v = *value,
                    LlvmOp::Alloca { .. } => {
                        roots.insert(v);
                        break;
                    }
                    _ => break,
                }
            }
        }
        roots
    }

    /// The module's datalayout, which memory laid out in bytes is laid out by.
    fn datalayout(&self) -> Result<DataLayout> {
        DataLayout::of(self.m).map_err(|why| crate::error(self.at, why))
    }

    /// The ABI alignment and allocation size in bytes of a value of LLVM type `ty`, as the datalayout lays it out.
    pub(crate) fn byte_layout(&self, ty: TypeId) -> Result<(u32, u32)> {
        let dl = self.datalayout()?;
        dl.size_align(self.m, ty)
            .map_err(|why| crate::error(self.at, why))
    }

    /// The byte offset of field `field` of a structure of `elements` (`packed` without padding).
    pub(crate) fn field_offset(&self, elements: &[TypeId], packed: bool, field: usize) -> Result<u32> {
        let dl = self.datalayout()?;
        dl.field_offset(self.m, elements, packed, field)
            .map_err(|why| crate::error(self.at, why))
    }

    /// The byte address (`u32`) of a byte-laid pointer plus `k`.
    fn payload_address(&mut self, p: Pointer, k: u32) -> IrValue {
        let fixed = self.u32_lit(p.at.saturating_add(k));
        match p.index {
            None => fixed,
            Some(i) => self.binary(Type::U32, Binary::IAdd, i, fixed),
        }
    }

    /// The word (`u32`) of a byte-laid pointer plus `k`: its byte address over four.
    pub(crate) fn byte_word(&mut self, p: Pointer, k: u32) -> IrValue {
        match p.index {
            None => self.u32_lit(p.at.saturating_add(k) / 4),
            Some(_) => {
                let address = self.payload_address(p, k);
                let two = self.u32_lit(2);
                self.binary(Type::U32, Binary::ShrU, address, two)
            }
        }
    }

    /// The bit (`u32`) a byte-laid half-word value starts at in its word: 0 or 16.
    fn half_shift(&mut self, p: Pointer) -> IrValue {
        match p.index {
            None => self.u32_lit((p.at & 2) << 3),
            Some(_) => {
                let address = self.payload_address(p, 0);
                let (two, three) = (self.u32_lit(2), self.u32_lit(3));
                let half = self.binary(Type::U32, Binary::And, address, two);
                self.binary(Type::U32, Binary::Shl, half, three)
            }
        }
    }

    /// A load of a value of LLVM type `ty` through a byte-laid pointer.
    pub(crate) fn load_bytes(&mut self, p: Pointer, ty: TypeId) -> Result<Lv> {
        match self.type_at(ty)?.clone() {
            LlvmType::Struct { elements, packed, .. } => {
                let mut items = Vec::with_capacity(elements.len());
                for (k, &e) in elements.iter().enumerate() {
                    let at = self.field_offset(&elements, packed, k)?;
                    items.push(self.load_bytes(
                        Pointer {
                            at: p.at.saturating_add(at),
                            ..p
                        },
                        e,
                    )?);
                }
                return Ok(Lv::Agg(items));
            }
            LlvmType::Array(n, e) => {
                let (_, size) = self.byte_layout(e)?;
                let mut items = Vec::new();
                for k in 0..u32::try_from(n).unwrap_or(0).min(4096) {
                    items.push(self.load_bytes(
                        Pointer {
                            at: p.at.saturating_add(k.saturating_mul(size)),
                            ..p
                        },
                        e,
                    )?);
                }
                return Ok(Lv::Agg(items));
            }
            // a vector's lanes are consecutive elements
            LlvmType::Vector(n, e) => {
                let t = self
                    .carried_type(ty)?
                    .ok_or_else(|| crate::error(self.at, "a load of a vector that is no value"))?;
                let (_, size) = self.byte_layout(e)?;
                let mut lanes = Vec::new();
                for k in 0..u32::try_from(n).unwrap_or(0) {
                    let lane = self.load_bytes(
                        Pointer {
                            at: p.at.saturating_add(k.saturating_mul(size)),
                            ..p
                        },
                        e,
                    )?;
                    let Lv::Value(lane) = lane else {
                        return self.refuse("a vector lane that is no value");
                    };
                    lanes.push(lane);
                }
                return Ok(Lv::Value(self.g.value(t, Op::Compose(lanes))));
            }
            _ => {}
        }
        let t = self
            .carried_type(ty)?
            .ok_or_else(|| crate::error(self.at, "a load of something that is no value"))?;
        let bits = self.stored_bits(ty)?;
        match bits {
            16 => {
                let w = self.byte_word(p, 0);
                let word = self
                    .g
                    .value(Type::U32, Op::LoadElement(p.var, w, OutOfBounds::Zero));
                let shift = self.half_shift(p);
                let moved = self.binary(Type::U32, Binary::ShrU, word, shift);
                let mask = self.u32_lit(0xffff);
                let low = self.binary(Type::U32, Binary::And, moved, mask);
                Ok(Lv::Value(if matches!(self.type_at(ty)?, LlvmType::Half) {
                    self.unary(t, Unary::F16ToF32, low)
                } else {
                    low
                }))
            }
            32 | 64 => {
                let mut lanes = Vec::new();
                for k in 0..bits / 32 {
                    let w = self.byte_word(p, k.saturating_mul(4));
                    lanes.push(
                        self.g
                            .value(Type::U32, Op::LoadElement(p.var, w, OutOfBounds::Zero)),
                    );
                }
                Ok(Lv::Value(self.compose_words(&lanes, t)?))
            }
            other => self.refuse(format!("a task payload value of {other} bits")),
        }
    }

    /// The bits a scalar of LLVM type `ty` takes in a task payload (a bool four bytes, as dxc stores it).
    fn stored_bits(&self, ty: TypeId) -> Result<u32> {
        Ok(match self.type_at(ty)? {
            LlvmType::Integer(1 | 32) | LlvmType::Float => 32,
            LlvmType::Integer(16) | LlvmType::Half => 16,
            LlvmType::Integer(64) | LlvmType::Double => 64,
            other => return self.refuse(format!("a task payload value of {other:?}")),
        })
    }

    /// A store of `lv` (of LLVM type `ty`) through a byte-laid pointer.
    pub(crate) fn store_bytes(&mut self, p: Pointer, lv: Lv, ty: TypeId) -> Result<()> {
        match (lv, self.type_at(ty)?.clone()) {
            (Lv::Agg(items), LlvmType::Struct { elements, packed, .. }) => {
                for (k, (item, &e)) in items.into_iter().zip(&elements).enumerate() {
                    let at = self.field_offset(&elements, packed, k)?;
                    self.store_bytes(
                        Pointer {
                            at: p.at.saturating_add(at),
                            ..p
                        },
                        item,
                        e,
                    )?;
                }
                Ok(())
            }
            (Lv::Agg(items), LlvmType::Array(_, e)) => {
                let (_, size) = self.byte_layout(e)?;
                for (k, item) in (0u32..).zip(items) {
                    self.store_bytes(
                        Pointer {
                            at: p.at.saturating_add(k.saturating_mul(size)),
                            ..p
                        },
                        item,
                        e,
                    )?;
                }
                Ok(())
            }
            (Lv::Value(x), LlvmType::Vector(n, e)) => {
                let (_, size) = self.byte_layout(e)?;
                let lane_type = self
                    .carried_type(e)?
                    .ok_or_else(|| crate::error(self.at, "a vector of lanes that are no values"))?;
                for k in 0..u8::try_from(n).unwrap_or(0) {
                    let lane = self.g.value(lane_type, Op::Swizzle(x, [k, 0, 0, 0]));
                    self.store_bytes(
                        Pointer {
                            at: p.at.saturating_add(u32::from(k).saturating_mul(size)),
                            ..p
                        },
                        Lv::Value(lane),
                        e,
                    )?;
                }
                Ok(())
            }
            (Lv::Value(x), _) => match self.stored_bits(ty)? {
                16 => {
                    let bits = if matches!(self.type_at(ty)?, LlvmType::Half) {
                        self.unary(Type::U32, Unary::F32ToF16, x)
                    } else {
                        self.uint_value(x)
                    };
                    let mask = self.u32_lit(0xffff);
                    let half = self.binary(Type::U32, Binary::And, bits, mask);
                    let shift = self.half_shift(p);
                    let w = self.byte_word(p, 0);
                    if self.g.vars.get(p.var.0 as usize).map(|v| v.space) == Some(iridia_ir::Space::Workgroup)
                    {
                        // clear the half, then set its bits: another invocation may write the other half
                        let field = self.binary(Type::U32, Binary::Shl, mask, shift);
                        let keep = self.unary(Type::U32, Unary::Not, field);
                        let moved = self.binary(Type::U32, Binary::Shl, half, shift);
                        for (op, value) in [(AtomicOp::And, keep), (AtomicOp::Or, moved)] {
                            self.g.value(
                                Type::U32,
                                Op::Atomic {
                                    memory: Memory::Var(p.var, Some(w)),
                                    op,
                                    value,
                                    compare: None,
                                },
                            );
                        }
                    } else {
                        let old = self
                            .g
                            .value(Type::U32, Op::LoadElement(p.var, w, OutOfBounds::Zero));
                        let sixteen = self.u32_lit(16);
                        let new = self.g.value(
                            Type::U32,
                            Op::BitInsert {
                                base: old,
                                insert: half,
                                offset: shift,
                                count: sixteen,
                            },
                        );
                        self.g
                            .node(iridia_ir::Node::StoreElement(p.var, w, new, OutOfBounds::Zero));
                    }
                    Ok(())
                }
                32 | 64 => {
                    let words = self.split_words(x)?;
                    for (k, word) in (0u32..).zip(words) {
                        let w = self.byte_word(p, k.saturating_mul(4));
                        self.g
                            .node(iridia_ir::Node::StoreElement(p.var, w, word, OutOfBounds::Zero));
                    }
                    Ok(())
                }
                other => self.refuse(format!("a task payload value of {other} bits")),
            },
            (other, _) => self.refuse(format!("a store of {other:?} into a task payload")),
        }
    }

    /// The `u32` of a 16-bit integer the lift carries.
    fn uint_value(&mut self, x: IrValue) -> IrValue {
        match self.g.ty(x) {
            Some(t) if t == Type::U32 => x,
            _ => self.bitcast(Type::U32, x),
        }
    }
}
