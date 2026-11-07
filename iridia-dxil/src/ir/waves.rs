//! The wave and quad operations (`dx.op` 110 to 123, 135 and 136), each as the naga-path lift ([`crate::lift`]) reads
//! it: the device's subgroup is the wave, a lane read at a number is a shuffle, a prefix is an exclusive scan, and the
//! bit counts count a ballot's bits.

use iridia_ir::{
    Binary, Cmp, Op, QuadSwap, Scalar, Type, Unary, ValueId as IrValue, Wave, WaveOp, WaveScan,
    interface::SystemValue,
};

use super::{Lift, Lv};
use crate::Result;
use crate::lift::compute::{WAVE_MAX, WAVE_MIN, WAVE_PRODUCT, WAVE_SUM};
use dxil_reader::module::{Instruction, ValueId};

/// The integer scalar of `bits` bits, signed or not; `None` for a width there is none of.
fn integer(bits: u32, signed: bool) -> Option<Scalar> {
    Some(match (bits, signed) {
        (8, false) => Scalar::U8,
        (8, true) => Scalar::I8,
        (16, false) => Scalar::U16,
        (16, true) => Scalar::I16,
        (32, false) => Scalar::U32,
        (32, true) => Scalar::I32,
        (64, false) => Scalar::U64,
        (64, true) => Scalar::I64,
        _ => return None,
    })
}

impl Lift<'_> {
    /// The lane's number: the input the pipeline gives, or in a thread group laid out in more than one dimension the
    /// invocation's rank among its wave's invocations when the program starts (see [`Lift::lane_at_start`]).
    fn lane_index(&mut self) -> Result<IrValue> {
        match self.lane {
            Some(lane) => Ok(lane),
            None => self.system_input(SystemValue::LaneIndex, None),
        }
    }

    /// In a thread group laid out in more than one dimension, which WebGPU gives no lane number, the lane's number as
    /// the naga-path lift makes it: the exclusive count of the wave's active invocations where the program starts,
    /// when every invocation is active. Placed at the start of the entry block, which every block is after.
    pub(crate) fn lane_at_start(&mut self) {
        if !self.setup.lane_by_rank {
            return;
        }
        self.notes.insert(
            "the lane number of a thread group laid out in more than one dimension (WebGPU gives such a group no \
             lane-number built-in) is the invocation's rank among its subgroup's invocations when the program starts: \
             every invocation that exists is active then, so the exclusive count of active invocations is the index in \
             the subgroup, which is as valid a numbering as the hardware's own that D3D leaves open; a subgroup the \
             thread group fills only in part is assumed to have its invocations at the lowest lanes"
                .into(),
        );
        self.g.current = 0;
        let one = self.u32_lit(1);
        let rank = self.g.value(
            Type::U32,
            Op::Wave(Wave::Collective {
                op: WaveOp::Add,
                scan: WaveScan::Exclusive,
                value: one,
            }),
        );
        self.lane = Some(rank);
    }

    /// A condition argument as a `bool`: an integer is one where it is not zero.
    fn condition(&mut self, v: ValueId) -> Result<IrValue> {
        let x = self.value(v)?;
        let t = self
            .g
            .ty(x)
            .ok_or_else(|| crate::error(self.at, "a condition of no type"))?;
        if t == Type::BOOL {
            return Ok(x);
        }
        let x = self.retype(x, Type::U32)?;
        let zero = self.u32_lit(0);
        Ok(self.binary(Type::BOOL, Binary::UCmp(Cmp::Ne), x, zero))
    }

    /// A wave operation's value as its lanes' bits, unsigned integers of their width: a flag as 1 or 0.
    fn wave_bits(&mut self, v: ValueId) -> Result<IrValue> {
        let x = self.value(v)?;
        let t = self
            .g
            .ty(x)
            .ok_or_else(|| crate::error(self.at, "a wave operand of no type"))?;
        if t.scalar == Scalar::Bool {
            let (on, off) = (self.u32_lit(1), self.u32_lit(0));
            return Ok(self.select(Type::U32, x, on, off));
        }
        let Some(bits) = integer(t.scalar.bits(), false) else {
            return self.refuse(format!("a wave operation of {t}"));
        };
        self.retype(x, t.with(bits))
    }

    /// The operand of a wave operation as the type its signedness argument says (0 signed): a float stays one.
    fn wave_operand(&mut self, v: ValueId, unsigned: Option<i64>) -> Result<IrValue> {
        let x = self.value(v)?;
        let t = self
            .g
            .ty(x)
            .ok_or_else(|| crate::error(self.at, "a wave operand of no type"))?;
        if t.scalar.is_float() {
            return Ok(x);
        }
        let Some(scalar) = integer(t.scalar.bits(), unsigned == Some(0)) else {
            return self.refuse(format!("a wave operation of {t}"));
        };
        self.retype(x, t.with(scalar))
    }

    /// A read of another invocation's value, `read` of the value as the wave reads it: a flag, which the wave's
    /// reads do not take, as 1 or 0, and a flag again.
    fn read_across(&mut self, v: ValueId, read: impl FnOnce(IrValue) -> Wave) -> Result<IrValue> {
        let x = self.value(v)?;
        let t = self
            .g
            .ty(x)
            .ok_or_else(|| crate::error(self.at, "a wave operand of no type"))?;
        if t.scalar != Scalar::Bool {
            return self.wave_op(read(x));
        }
        let u = t.with(Scalar::U32);
        let (one, zero) = (self.u32_lit(1), self.u32_lit(0));
        let (one, zero) = if t.lanes > 1 {
            (self.g.value(u, Op::Splat(one)), self.g.value(u, Op::Splat(zero)))
        } else {
            (one, zero)
        };
        let bits = self.select(u, x, one, zero);
        let got = self.wave_op(read(bits))?;
        Ok(self.binary(t, Binary::UCmp(Cmp::Ne), got, zero))
    }

    fn wave_op(&mut self, w: Wave) -> Result<IrValue> {
        let t = iridia_ir::derived_type(&Op::Wave(w), |v| self.g.ty(v))
            .map_err(|e| crate::error(self.at, e))?
            .ok_or_else(|| crate::error(self.at, "a wave operation of no type"))?;
        Ok(self.g.value(t, Op::Wave(w)))
    }

    /// The wave and quad operations.
    pub(crate) fn wave(&mut self, inst: &Instruction, code: u64, args: &[ValueId]) -> Result<()> {
        self.notes.insert(
            "wave operations act on the device's subgroup, as D3D's act on the hardware's wave: the lane count is \
             the device's, not a fixed 32 or 64"
                .into(),
        );
        let result = match code {
            // IsFirstLane, LaneIndex, LaneCount
            110 => {
                let lane = self.lane_index()?;
                let first = self.wave_op(Wave::First(lane))?;
                self.binary(Type::BOOL, Binary::UCmp(Cmp::Eq), lane, first)
            }
            111 => self.lane_index()?,
            112 => self.system_input(SystemValue::LaneCount, None)?,
            // AnyTrue, AllTrue
            113 | 114 => {
                let c = self.condition(self.arg(args, 1)?)?;
                let op = if code == 113 { WaveOp::Any } else { WaveOp::All };
                self.wave_op(Wave::Collective {
                    op,
                    scan: WaveScan::Reduce,
                    value: c,
                })?
            }
            // ActiveAllEqual: every lane's bits are the first lane's
            115 => {
                let bits = self.wave_bits(self.arg(args, 1)?)?;
                let first = self.wave_op(Wave::First(bits))?;
                let t = self
                    .g
                    .ty(bits)
                    .ok_or_else(|| crate::error(self.at, "a wave operand of no type"))?;
                let same = self.binary(t.with(Scalar::Bool), Binary::UCmp(Cmp::Eq), bits, first);
                let same = if t.lanes > 1 {
                    self.unary(Type::BOOL, Unary::All, same)
                } else {
                    same
                };
                self.wave_op(Wave::Collective {
                    op: WaveOp::All,
                    scan: WaveScan::Reduce,
                    value: same,
                })?
            }
            // ActiveBallot: the ballot's four words
            116 => {
                let c = self.condition(self.arg(args, 1)?)?;
                let ballot = self.wave_op(Wave::Ballot(c))?;
                let mut words = Vec::new();
                for k in 0..4u8 {
                    words.push(Lv::Value(
                        self.g.value(Type::U32, Op::Swizzle(ballot, [k, 0, 0, 0])),
                    ));
                }
                return self.define_result(inst, Lv::Agg(words));
            }
            // ReadLaneAt, ReadLaneFirst
            117 => {
                let lane = self.uint(self.arg(args, 2)?)?;
                self.read_across(self.arg(args, 1)?, |v| Wave::Lane(v, lane))?
            }
            118 => self.read_across(self.arg(args, 1)?, Wave::First)?,
            // ActiveOp, PrefixOp: (value, operation, signedness)
            119 | 121 => {
                let value = self.wave_operand(self.arg(args, 1)?, self.const_arg(args, 3))?;
                let op = match self.const_arg(args, 2) {
                    Some(WAVE_SUM) => WaveOp::Add,
                    Some(WAVE_PRODUCT) => WaveOp::Mul,
                    Some(WAVE_MIN) if code == 119 => WaveOp::Min,
                    Some(WAVE_MAX) if code == 119 => WaveOp::Max,
                    _ => return self.refuse("a wave operation of a kind that is not lifted"),
                };
                let scan = if code == 119 {
                    WaveScan::Reduce
                } else {
                    WaveScan::Exclusive
                };
                self.wave_op(Wave::Collective { op, scan, value })?
            }
            // ActiveBit: and, or, xor
            120 => {
                let value = self.wave_bits(self.arg(args, 1)?)?;
                let op = match self.const_arg(args, 2) {
                    Some(0) => WaveOp::And,
                    Some(1) => WaveOp::Or,
                    Some(2) => WaveOp::Xor,
                    _ => return self.refuse("a wave bit operation of a kind that is not lifted"),
                };
                self.wave_op(Wave::Collective {
                    op,
                    scan: WaveScan::Reduce,
                    value,
                })?
            }
            // QuadReadLaneAt, QuadOp (across x, y, the diagonal)
            122 => {
                let lane = self.uint(self.arg(args, 2)?)?;
                self.read_across(self.arg(args, 1)?, |v| Wave::QuadLane(v, lane))?
            }
            123 => {
                let swap = match self.const_arg(args, 2) {
                    Some(0) => QuadSwap::X,
                    Some(1) => QuadSwap::Y,
                    Some(2) => QuadSwap::Diagonal,
                    _ => return self.refuse("a quad operation of a kind that is not lifted"),
                };
                self.read_across(self.arg(args, 1)?, |v| Wave::QuadSwap(v, swap))?
            }
            // AllBitCount, PrefixBitCount: the lanes whose condition holds, all of them or those before this one
            135 | 136 => {
                let c = self.condition(self.arg(args, 1)?)?;
                let ballot = self.wave_op(Wave::Ballot(c))?;
                let lane = if code == 136 {
                    Some(self.lane_index()?)
                } else {
                    None
                };
                let u = Type::U32;
                let mut total = self.u32_lit(0);
                for k in 0..4u8 {
                    let word = self.g.value(u, Op::Swizzle(ballot, [k, 0, 0, 0]));
                    let word = match lane {
                        // the bits of this word below the lane: all, none, or the low bits up to it
                        Some(lane) => {
                            let base = self.u32_lit(u32::from(k).saturating_mul(32));
                            let relative = self.binary(u, Binary::ISub, lane, base);
                            let thirty_two = self.u32_lit(32);
                            let in_word =
                                self.binary(Type::BOOL, Binary::UCmp(Cmp::Lt), relative, thirty_two);
                            let above = self.binary(Type::BOOL, Binary::UCmp(Cmp::Ge), lane, base);
                            let one = self.u32_lit(1);
                            let shifted = self.binary(u, Binary::Shl, one, relative);
                            let low = self.binary(u, Binary::ISub, shifted, one);
                            let all = self.u32_lit(u32::MAX);
                            let none = self.u32_lit(0);
                            // lane in this word: the low bits; past it: every bit; before it: none
                            let before = self.select(u, in_word, low, all);
                            let mask = self.select(u, above, before, none);
                            self.binary(u, Binary::And, word, mask)
                        }
                        None => word,
                    };
                    let ones = self.unary(u, Unary::CountOnes, word);
                    total = self.binary(u, Binary::IAdd, total, ones);
                }
                total
            }
            other => return self.refuse(format!("wave operation {other}")),
        };
        // the result is the type the program says, whatever the operand's bits were carried as
        let want = inst
            .result
            .map(|(_, ty)| self.carried_type(ty))
            .transpose()?
            .flatten()
            .ok_or_else(|| crate::error(self.at, "a wave operation of no result type"))?;
        let r = self.retype(result, want)?;
        self.define_result(inst, Lv::Value(r))
    }
}
