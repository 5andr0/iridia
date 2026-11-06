//! What reaches memory a program writes: unordered-access views (`u#`: typed texture views, typed, raw and structured
//! buffers) and thread-group shared memory (`g#`), their loads, stores, atomic operations and hidden counters, and
//! `sync`.
//!
//! Direct3D's rules, from the functional specification: a read of a buffer word outside it gives zero and a write
//! there changes nothing; a typed store always writes the four lanes (`dstUAV.xyzw`); a raw or structured store's
//! mask is `.x`, `.xy`, `.xyz` or `.xyzw`, that many consecutive words; an atomic operation outside its memory changes
//! nothing (its old value is zero here, which the specification leaves undefined); shared memory outside its words is
//! undefined, read here as zero and written as nothing, as for a buffer.

use iridia_ir::{
    Address, AtomicOp, Barrier, Binary, CounterOp, Memory, Op, OutOfBounds, Res, Scalar, Scope, Type, ValueId,
};

use super::operands::{OperandAt, of_four};
use super::resources::{Buffer, Shared, Uav};
use super::{Lift, lanes_of};
use crate::Result;
use dxbc_reader::decode::{Instruction, Operand, Register, Selection};
use iridia::D3dClass;

/// The words a raw or structured instruction reaches.
#[derive(Clone, Copy)]
pub(super) enum Words {
    /// A shader-resource or unordered-access buffer (an element of a range of them); a structured one's element stride.
    Buffer(Res, Option<u32>),
    /// Thread-group shared memory.
    Shared(Shared),
}

impl Words {
    fn stride(self) -> Option<u32> {
        match self {
            Words::Buffer(_, s) => s,
            Words::Shared(g) => g.stride,
        }
    }
}

impl Lift<'_> {
    /// The words a resource operand of a raw or structured instruction names: `t#`, `u#` or `g#`.
    pub(super) fn words(&mut self, op: &Operand) -> Result<Words> {
        let slot = self.index_of(op, 0)?.base;
        match op.register {
            Register::Texture => match self.resources.buffers.get(&slot).copied() {
                Some((id, Buffer::Words { stride })) => Ok(Words::Buffer(
                    self.res_of(op, id, D3dClass::ShaderResource)?,
                    stride,
                )),
                _ => self.refuse(format!("t{slot} is not a raw or structured buffer")),
            },
            Register::Uav => match self.resources.uavs.get(&slot).copied() {
                Some((id, Uav::Words { stride })) => Ok(Words::Buffer(
                    self.res_of(op, id, D3dClass::UnorderedAccess)?,
                    stride,
                )),
                Some(_) => self.refuse(format!(
                    "u{slot} is typed, where a raw or structured view is taken"
                )),
                None => self.refuse(format!("u{slot} is used but not declared")),
            },
            Register::GroupSharedMemory => match self.resources.shared.get(&slot) {
                Some(&g) => Ok(Words::Shared(g)),
                None => self.refuse(format!("g{slot} is used but not declared")),
            },
            other => self.refuse(format!("a memory operand of register type {other:?}")),
        }
    }

    /// The byte address of a raw (`[byte]`) or structured (`[element, byte offset]`) instruction.
    fn byte_address(&mut self, words: Words, address: &[&Operand]) -> Result<ValueId> {
        match (words.stride(), address) {
            (None, [byte]) => self.read(byte, 1, Scalar::U32),
            (Some(stride), [element, offset]) => {
                let element = self.read(element, 1, Scalar::U32)?;
                let offset = self.read(offset, 1, Scalar::U32)?;
                let stride = self.b.u32(stride);
                let base = self.binary(Binary::IMul, element, stride)?;
                self.binary(Binary::IAdd, base, offset)
            }
            // an atomic operation's structured address: the element and the byte offset in one operand's x and y
            (Some(stride), [both]) => {
                let pair = self.read_lanes(both, &[0, 1], Scalar::U32)?;
                let element = self.lane(pair, 0)?;
                let offset = self.lane(pair, 1)?;
                let stride = self.b.u32(stride);
                let base = self.binary(Binary::IMul, element, stride)?;
                self.binary(Binary::IAdd, base, offset)
            }
            _ => self.refuse("a memory address of the wrong number of operands for its memory"),
        }
    }

    /// The shared-memory word of a byte address.
    fn shared_word(&mut self, byte: ValueId, k: u32) -> Result<ValueId> {
        let two = self.b.u32(2);
        let word = self.binary(Binary::ShrU, byte, two)?;
        if k == 0 {
            return Ok(word);
        }
        let k = self.b.u32(k);
        self.binary(Binary::IAdd, word, k)
    }

    /// Word `k` after byte `byte` of `words`, as `u32`; zero outside the memory.
    fn word_at(&mut self, words: Words, byte: ValueId, k: u32) -> Result<ValueId> {
        Ok(match words {
            Words::Buffer(resource, _) => self.b.value(
                Type::U32,
                Op::LoadBuffer {
                    resource,
                    address: Address {
                        index: Some(byte),
                        stride: 1,
                        offset: k.saturating_mul(4),
                    },
                    oob: OutOfBounds::Zero,
                },
            ),
            Words::Shared(g) => {
                let index = self.shared_word(byte, k)?;
                self.b
                    .value(Type::U32, Op::LoadElement(g.var, index, OutOfBounds::Zero))
            }
        })
    }

    /// `ld_raw` and `ld_structured`: for each lane of the destination's mask, the word of its component of the resource
    /// operand's swizzle.
    pub(super) fn load_memory(
        &mut self,
        ins: &Instruction,
        address: &[&Operand],
        resource: &Operand,
    ) -> Result<()> {
        let words = self.words(resource)?;
        let mask = self.dest_mask(ins.operand(0)?);
        let byte = self.byte_address(words, address)?;
        let mut lanes = Vec::new();
        for l in lanes_of(mask) {
            let c = match resource.selection {
                Selection::Swizzle(s) => of_four(s, l),
                Selection::Scalar(c) => c,
                _ => 0,
            };
            lanes.push(self.word_at(words, byte, u32::from(c))?);
        }
        let result = if let [one] = lanes[..] {
            one
        } else {
            self.op(Op::Compose(lanes))?
        };
        self.write(ins.operand(0)?, result, false)
    }

    /// `store_raw` and `store_structured`: the first lanes of the source (by its swizzle), as many as the destination's
    /// mask (`.x`, `.xy`, `.xyz` or `.xyzw`), to consecutive words from the address.
    pub(super) fn store_memory(
        &mut self,
        ins: &Instruction,
        address: &[&Operand],
        value: &Operand,
    ) -> Result<()> {
        let destination = ins.operand(0)?;
        let words = self.words(destination)?;
        let mask = self.dest_mask(destination);
        let count = match mask {
            0b0001 => 1u8,
            0b0011 => 2,
            0b0111 => 3,
            0b1111 => 4,
            m => {
                return self.refuse(format!(
                    "a store with the write mask {m:#06b}: Direct3D takes only .x, .xy, .xyz and .xyzw"
                ));
            }
        };
        let byte = self.byte_address(words, address)?;
        let lanes: Vec<u8> = (0..count).collect();
        let data = self.read_lanes(value, &lanes, Scalar::U32)?;
        match words {
            Words::Buffer(resource, _) => {
                self.b.store_buffer(resource, Address::bytes(byte), data);
            }
            Words::Shared(g) => {
                for k in 0..count {
                    let index = self.shared_word(byte, u32::from(k))?;
                    let datum = if count == 1 { data } else { self.lane(data, k)? };
                    self.b.store_element(g.var, index, datum, OutOfBounds::Zero);
                }
            }
        }
        Ok(())
    }

    /// A typed view's address lanes: the element of a buffer, or a texture's coordinates and its layer.
    fn typed_address(&mut self, op: &Operand, uav: Uav) -> Result<ValueId> {
        let n = match uav {
            Uav::Image { dims, arrayed, .. } => dims.saturating_add(u8::from(arrayed)),
            Uav::Typed(_) => 1,
            Uav::Words { .. } => return self.refuse("a typed access of a raw or structured view"),
        };
        let lanes: Vec<u8> = (0..n).collect();
        self.read_lanes(op, &lanes, Scalar::U32)
    }

    /// The typed view a `u#` operand names (an element of a range of them), its kind and its lanes' scalar.
    fn typed_view(&mut self, op: &Operand) -> Result<(Res, Uav, Scalar)> {
        if op.register != Register::Uav {
            return self.refuse(format!(
                "{:?}, where a typed unordered-access view is taken",
                op.register
            ));
        }
        let slot = self.index_of(op, 0)?.base;
        match self.resources.uavs.get(&slot).copied() {
            Some((id, uav @ (Uav::Image { scalar, .. } | Uav::Typed(scalar)))) => {
                Ok((self.res_of(op, id, D3dClass::UnorderedAccess)?, uav, scalar))
            }
            Some(_) => self.refuse(format!(
                "u{slot} is raw or structured, where a typed view is taken"
            )),
            None => self.refuse(format!("u{slot} is used but not declared")),
        }
    }

    /// `ld_uav_typed dest, address, u#`: the texel's or element's four lanes, zero outside the view.
    pub(super) fn load_typed(&mut self, ins: &Instruction) -> Result<()> {
        let resource = ins.operand(2)?;
        let (texture, uav, scalar) = self.typed_view(resource)?;
        let mask = self.dest_mask(ins.operand(0)?);
        let coord = self.typed_address(ins.operand(1)?, uav)?;
        let texel = self.b.value(
            Type::vector(scalar, 4),
            Op::LoadTexel {
                texture,
                coord,
                level: None,
                sample: None,
                offset: [0; 3],
            },
        );
        let bits = self.bitcast(texel, Scalar::U32)?;
        let picked = self.pick(bits, resource, mask)?;
        self.write(ins.operand(0)?, picked, false)
    }

    /// `store_uav_typed u#.xyzw, address, value`: the source's four lanes to the texel or element; nothing outside the
    /// view.
    pub(super) fn store_typed(&mut self, ins: &Instruction) -> Result<()> {
        let destination = ins.operand(0)?;
        let (image, uav, scalar) = self.typed_view(destination)?;
        if self.dest_mask(destination) != 15 {
            return self.refuse("a typed store whose mask is not .xyzw, which Direct3D requires");
        }
        let coord = self.typed_address(ins.operand(1)?, uav)?;
        let value = self.read_lanes(ins.operand(2)?, &[0, 1, 2, 3], scalar)?;
        self.b.store_texel(image, coord, value);
        Ok(())
    }

    /// The atomic operation of an `atomic_*` or `imm_atomic_*` instruction, and whether it takes a comparand.
    fn atomic_op(&self, name: &str) -> Result<AtomicOp> {
        Ok(
            match name.trim_start_matches("imm_").trim_start_matches("atomic_") {
                "iadd" => AtomicOp::Add,
                "and" => AtomicOp::And,
                "or" => AtomicOp::Or,
                "xor" => AtomicOp::Xor,
                "imax" => AtomicOp::SMax,
                "imin" => AtomicOp::SMin,
                "umax" => AtomicOp::UMax,
                "umin" => AtomicOp::UMin,
                "exch" => AtomicOp::Exchange,
                "cmp_exch" | "cmp_store" => AtomicOp::CompareExchange,
                other => return self.refuse(format!("atomic operation {other}")),
            },
        )
    }

    /// `atomic_* dst, address, src0[, src1]` and `imm_atomic_* old, dst, address, src0[, src1]`: one 32-bit word of a
    /// buffer, a typed view (an `r32` texel or element) or shared memory, changed atomically; the immediate forms
    /// write the word as it was before. A compare-exchange's comparand is `src0` and its new value `src1`.
    pub(super) fn atomic(&mut self, ins: &Instruction) -> Result<()> {
        let name = ins.name();
        let immediate = name.starts_with("imm_");
        let at = usize::from(immediate);
        let op = self.atomic_op(name)?;
        let target = ins.operand(at)?;
        let address = ins.operand(at.saturating_add(1))?;
        let first = ins.operand(at.saturating_add(2))?;
        let (memory, scalar) = match target.register {
            Register::Uav
                if matches!(
                    self.resources.uavs.get(&self.index_of(target, 0)?.base),
                    Some((_, Uav::Image { .. } | Uav::Typed(_)))
                ) =>
            {
                let (image, uav, scalar) = self.typed_view(target)?;
                if scalar == Scalar::F32 {
                    return self
                        .refuse("an atomic operation on a float view: Direct3D takes R32_UINT and R32_SINT");
                }
                let coord = self.typed_address(address, uav)?;
                (Memory::Image(image, coord), scalar)
            }
            _ => {
                let words = self.words(target)?;
                let byte = self.byte_address(words, &[address])?;
                let memory = match words {
                    Words::Buffer(resource, _) => Memory::Buffer(resource, Address::bytes(byte)),
                    Words::Shared(g) => {
                        let index = self.shared_word(byte, 0)?;
                        Memory::Var(g.var, Some(index))
                    }
                };
                (memory, Scalar::U32)
            }
        };
        let (value, compare) = if op == AtomicOp::CompareExchange {
            let comparand = self.read(first, 1, scalar)?;
            let new = self.read(ins.operand(at.saturating_add(3))?, 1, scalar)?;
            (new, Some(comparand))
        } else {
            (self.read(first, 1, scalar)?, None)
        };
        let old = self.b.value(
            Type::vector(scalar, 1),
            Op::Atomic {
                memory,
                op,
                value,
                compare,
            },
        );
        if immediate {
            let bits = self.bitcast(old, Scalar::U32)?;
            self.write(ins.operand(0)?, bits, false)?;
        }
        Ok(())
    }

    /// `imm_atomic_alloc dest, u#` (the count before an increment) and `imm_atomic_consume dest, u#` (the count after a
    /// decrement) of a structured view's hidden counter.
    pub(super) fn counter(&mut self, ins: &Instruction) -> Result<()> {
        let view = ins.operand(1)?;
        if view.register != Register::Uav {
            return self.refuse(format!("a counter of {:?}, where a u# is taken", view.register));
        }
        let slot = self.index_of(view, 0)?.base;
        let Some(&(id, Uav::Words { stride: Some(_) })) = self.resources.uavs.get(&slot) else {
            return self.refuse(format!(
                "u{slot} is not a structured view, which alone has a counter"
            ));
        };
        let view = self.res_of(view, id, D3dClass::UnorderedAccess)?;
        let op = if ins.name() == "imm_atomic_alloc" {
            CounterOp::Increment
        } else {
            CounterOp::Decrement
        };
        let count = self.b.value(Type::U32, Op::Counter(view, op));
        self.write(ins.operand(0)?, count, false)
    }

    /// `sync` with its flags: 1 the thread group meets, 2 shared memory is fenced, 4 views within the group, 8 views
    /// across the device (within the group for a view not declared globally coherent, which device scope covers).
    pub(super) fn sync(&mut self, ins: &Instruction) -> Result<()> {
        let c = ins.control;
        let barrier = Barrier {
            execution: c & 1 != 0,
            workgroup: c & 2 != 0,
            storage: if c & 8 != 0 {
                Some(Scope::Device)
            } else if c & 4 != 0 {
                Some(Scope::Workgroup)
            } else {
                None
            },
        };
        if barrier == Barrier::default() {
            return self.refuse("a sync with no flags");
        }
        self.b.barrier(barrier);
        Ok(())
    }
}
