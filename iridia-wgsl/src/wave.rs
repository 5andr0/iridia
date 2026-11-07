//! `Op::Wave`: WGSL's subgroup operations (`enable subgroups;` in the standard's directives).

use alloc::format;
use alloc::string::String;

use iridia_ir::{Op, QuadSwap, Scalar, Type, Wave, WaveOp, WaveScan};

use crate::emit::{R, Writer};
use crate::text::err;

impl Writer<'_> {
    pub(crate) fn wave(&mut self, w: Wave, _t: Type) -> R<String> {
        Ok(match w {
            Wave::Ballot(c) => format!("subgroupBallot({})", self.get(c)?),
            Wave::Collective { op, scan, value } => {
                let name = match (scan, op) {
                    (WaveScan::Reduce, WaveOp::Add) => "subgroupAdd",
                    (WaveScan::Reduce, WaveOp::Mul) => "subgroupMul",
                    (WaveScan::Reduce, WaveOp::Min) => "subgroupMin",
                    (WaveScan::Reduce, WaveOp::Max) => "subgroupMax",
                    (WaveScan::Reduce, WaveOp::And) => "subgroupAnd",
                    (WaveScan::Reduce, WaveOp::Or) => "subgroupOr",
                    (WaveScan::Reduce, WaveOp::Xor) => "subgroupXor",
                    (WaveScan::Reduce, WaveOp::All) => "subgroupAll",
                    (WaveScan::Reduce, WaveOp::Any) => "subgroupAny",
                    (WaveScan::Inclusive, WaveOp::Add) => "subgroupInclusiveAdd",
                    (WaveScan::Inclusive, WaveOp::Mul) => "subgroupInclusiveMul",
                    (WaveScan::Exclusive, WaveOp::Add) => "subgroupExclusiveAdd",
                    (WaveScan::Exclusive, WaveOp::Mul) => "subgroupExclusiveMul",
                    (_, op) => return Err(err(format!("a wave scan of {op:?} has no WGSL form"))),
                };
                format!("{name}({})", self.get(value)?)
            }
            Wave::First(a) => format!("subgroupBroadcastFirst({})", self.get(a)?),
            Wave::Lane(a, l) => format!("subgroupShuffle({}, {})", self.get(a)?, self.lane_number(l)?),
            Wave::QuadLane(a, l) => {
                // the lane must be a constant expression in WGSL
                let Some(Op::Literal(bits)) = self.f.value(l).map(|d| d.op.clone()) else {
                    return Err(err("a quad read of a computed lane is not written yet"));
                };
                format!("quadBroadcast({}, {}u)", self.get(a)?, bits[0] & 3)
            }
            Wave::QuadSwap(a, s) => {
                let name = match s {
                    QuadSwap::X => "quadSwapX",
                    QuadSwap::Y => "quadSwapY",
                    QuadSwap::Diagonal => "quadSwapDiagonal",
                };
                format!("{name}({})", self.get(a)?)
            }
        })
    }

    /// A lane number as the `u32` WGSL's subgroup reads take.
    fn lane_number(&self, l: iridia_ir::ValueId) -> R<String> {
        let x = self.get(l)?;
        self.bitcast(&x, self.vty(l)?, Scalar::U32)
    }
}
