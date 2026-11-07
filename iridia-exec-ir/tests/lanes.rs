//! `Prepared::run_lanes` over a register file gives each active lane what `Prepared::run` gives that lane alone, and
//! leaves the other lanes and the operands as they were, for one operation of every kind and any register contents,
//! lane mask and flushing rule.

#![allow(clippy::unwrap_used, reason = "a failing case is the report")]

use iridia_exec_ir::{Lanes, Prepared};
use iridia_ir::{
    Binary, Cmp, FloatToInt, MinMaxNan, MulZero, OnZero, Op, Rounding, SaturateNan, Scalar, Ternary, Type,
    Unary, ValueId,
};
use proptest::prelude::*;

const F: fn(u8) -> Type = |n| Type::vector(Scalar::F32, n);
const I: fn(u8) -> Type = |n| Type::vector(Scalar::I32, n);
const U: fn(u8) -> Type = |n| Type::vector(Scalar::U32, n);
const B: fn(u8) -> Type = |n| Type::vector(Scalar::Bool, n);
const D: fn(u8) -> Type = |n| Type::vector(Scalar::F64, n);

/// One operation: its operands' types in order (values 0, 1, ...), its result type, and the operation.
struct Case {
    operands: Vec<Type>,
    result: Type,
    op: Op,
}

fn v(i: u32) -> ValueId {
    ValueId(i)
}

fn case(operands: &[Type], result: Type, op: Op) -> Case {
    Case {
        operands: operands.to_vec(),
        result,
        op,
    }
}

/// An operation of every kind the lane loops take, on more than one shape.
fn cases() -> Vec<Case> {
    let mut all = Vec::new();
    for n in 1..=4u8 {
        for u in [
            Unary::FNeg,
            Unary::FAbs,
            Unary::FFract,
            Unary::FRound(Rounding::Floor),
            Unary::FRound(Rounding::NearestEven),
            Unary::FSqrt,
            Unary::FSign,
            Unary::Saturate(SaturateNan::Zero),
            Unary::ClampRange {
                lo: (-1.0f32).to_bits(),
                hi: 1.0f32.to_bits(),
                nan: SaturateNan::Zero,
            },
            Unary::Quantize {
                frac_bits: 8,
                rounding: Rounding::NearestEven,
            },
        ] {
            all.push(case(&[F(n)], F(n), Op::Unary(u, v(0))));
        }
        for u in [
            Unary::INeg,
            Unary::Not,
            Unary::CountOnes,
            Unary::BitReverse,
            Unary::FirstBitHigh,
        ] {
            all.push(case(&[U(n)], U(n), Op::Unary(u, v(0))));
        }
        all.push(case(&[B(n)], B(n), Op::Unary(Unary::Not, v(0))));
        all.push(case(&[B(n)], B(1), Op::Unary(Unary::Any, v(0))));
        all.push(case(&[B(n)], B(1), Op::Unary(Unary::All, v(0))));
        all.push(case(&[F(n)], B(n), Op::Unary(Unary::IsNan, v(0))));
        all.push(case(&[F(n)], U(n), Op::Unary(Unary::F32ToF16, v(0))));
        for b in [
            Binary::FAdd,
            Binary::FSub,
            Binary::FMul(MulZero::Ieee),
            Binary::FMul(MulZero::ZeroWins),
            Binary::FDiv,
            Binary::FMin(MinMaxNan::Propagate),
            Binary::FMax(MinMaxNan::MinNum),
            Binary::FPow,
        ] {
            all.push(case(&[F(n), F(n)], F(n), Op::Binary(b, v(0), v(1))));
        }
        for c in [Cmp::Eq, Cmp::Ne, Cmp::Lt, Cmp::Ge] {
            all.push(case(&[F(n), F(n)], B(n), Op::Binary(Binary::FCmp(c), v(0), v(1))));
            all.push(case(&[I(n), I(n)], B(n), Op::Binary(Binary::SCmp(c), v(0), v(1))));
            all.push(case(&[U(n), U(n)], B(n), Op::Binary(Binary::UCmp(c), v(0), v(1))));
        }
        for b in [
            Binary::IAdd,
            Binary::IMul,
            Binary::UDiv(OnZero::AllOnes),
            Binary::URem(OnZero::ByOne),
            Binary::Shl,
            Binary::ShrU,
            Binary::And,
            Binary::Xor,
            Binary::UMax,
            Binary::UMulHigh,
        ] {
            all.push(case(&[U(n), U(n)], U(n), Op::Binary(b, v(0), v(1))));
        }
        for b in [Binary::SDiv(OnZero::AllOnes), Binary::ShrS, Binary::SMin] {
            all.push(case(&[I(n), I(n)], I(n), Op::Binary(b, v(0), v(1))));
        }
        all.push(case(
            &[F(n), F(n), F(n)],
            F(n),
            Op::Ternary(Ternary::Fma, v(0), v(1), v(2)),
        ));
        all.push(case(
            &[U(n), U(n), U(n)],
            U(n),
            Op::Ternary(Ternary::BitExtract { signed: true }, v(0), v(1), v(2)),
        ));
        all.push(case(
            &[U(n), U(n), U(n), U(n)],
            U(n),
            Op::BitInsert {
                base: v(0),
                insert: v(1),
                offset: v(2),
                count: v(3),
            },
        ));
        // a select on one condition for all lanes, and on a condition a lane
        all.push(case(
            &[B(1), F(n), F(n)],
            F(n),
            Op::Select {
                cond: v(0),
                accept: v(1),
                reject: v(2),
            },
        ));
        all.push(case(
            &[B(n), U(n), U(n)],
            U(n),
            Op::Select {
                cond: v(0),
                accept: v(1),
                reject: v(2),
            },
        ));
        all.push(case(&[F(n)], F(n), Op::Swizzle(v(0), [3, 2, 1, 0])));
        all.push(case(&[F(1)], F(n), Op::Splat(v(0))));
        all.push(case(&[F(n), F(n)], F(1), Op::Dot(v(0), v(1))));
        all.push(case(&[F(n)], I(n), Op::Convert(v(0), FloatToInt::Saturate)));
        all.push(case(&[F(n)], U(n), Op::Convert(v(0), FloatToInt::Saturate)));
        all.push(case(&[I(n)], F(n), Op::Convert(v(0), FloatToInt::Saturate)));
        all.push(case(&[U(n)], F(n), Op::Convert(v(0), FloatToInt::Saturate)));
        all.push(case(&[U(n)], I(n), Op::Convert(v(0), FloatToInt::Saturate)));
        all.push(case(&[F(n)], U(n), Op::Bitcast(v(0))));
        // a double's lanes are two words each
        if n <= 2 {
            all.push(case(&[D(n), D(n)], D(n), Op::Binary(Binary::FAdd, v(0), v(1))));
        }
    }
    // a compose of parts of other widths, up to four lanes
    all.push(case(
        &[F(1), F(1), F(1), F(1)],
        F(4),
        Op::Compose(vec![v(0), v(1), v(2), v(3)]),
    ));
    all.push(case(
        &[F(1), F(2), F(1)],
        F(4),
        Op::Compose(vec![v(0), v(1), v(2)]),
    ));
    all.push(case(&[F(2), F(2)], F(4), Op::Compose(vec![v(0), v(1)])));
    all.push(case(&[F(3), F(1)], F(4), Op::Compose(vec![v(0), v(1)])));
    all.push(case(&[F(1), F(1)], F(2), Op::Compose(vec![v(0), v(1)])));
    // an operation an executor places itself
    all.push(case(
        &[F(2)],
        F(2),
        Op::Unary(
            Unary::Derivative(iridia_ir::Axis::X, iridia_ir::Fineness::Coarse),
            v(0),
        ),
    ));
    all
}

/// `lanes` of a value of type `t` as a comparison sees them: IEEE 754 leaves the payload and sign of the NaN an
/// operation on NaNs gives open (which operand's it keeps depends on how the compiler orders the operands of a
/// multiplication), so every NaN of an `f32` result is one NaN.
fn canonical(t: Type, lanes: Lanes) -> Lanes {
    let mut out = lanes;
    if t.scalar == Scalar::F32 {
        for w in out.iter_mut().take(usize::from(t.lanes)) {
            if *w & 0x7fff_ffff > 0x7f80_0000 {
                *w = 0x7fc0_0000;
            }
        }
    }
    out
}

/// A word of a register: an arbitrary one, or one that matters (zeros, denormals, infinities, NaNs, ones).
fn word() -> impl Strategy<Value = u32> {
    prop_oneof![
        4 => any::<u32>(),
        1 => prop::sample::select(vec![
            0u32,
            0x8000_0000,
            1,
            0x0000_0001,
            0x8000_0001,
            0x007f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0x7f80_0001,
            1.0f32.to_bits(),
            (-1.0f32).to_bits(),
            0.5f32.to_bits(),
            3.0f32.to_bits(),
            u32::MAX,
            31,
            32,
        ]),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn the_lane_loop_gives_each_active_lane_what_a_lane_alone_gets(
        stride in prop::sample::select(vec![1usize, 2, 4, 16]),
        mask in any::<u32>(),
        flush in any::<bool>(),
        words in prop::collection::vec(word(), 5 * 16 * 4),
    ) {
        let active = |lane: usize| mask >> lane & 1 != 0;
        for c in cases() {
            let operands = c.operands.len();
            let types = c.operands.clone();
            let prepared = Prepared::new(&c.op, c.result, |value| {
                types
                    .get(value.0 as usize)
                    .copied()
                    .ok_or_else(|| format!("%{} has no type", value.0))
            });
            // the registers: every operand's lanes, then the result's, which hold something to be kept or overwritten
            let mut at = 0usize;
            let mut take = |t: Type, bools: bool| -> Lanes {
                let mut lanes = [0u32; 4];
                for (k, w) in lanes.iter_mut().enumerate() {
                    let raw = words[at % words.len()];
                    at += 1;
                    // a bool is 0 or 1
                    *w = if bools && k < usize::from(t.lanes) { raw & 1 } else { raw };
                }
                lanes
            };
            let mut regs: Vec<Lanes> = Vec::new();
            for t in &types {
                for _ in 0..stride {
                    regs.push(take(*t, t.scalar == Scalar::Bool));
                }
            }
            for _ in 0..stride {
                regs.push(take(c.result, false));
            }
            let before = regs.clone();
            let dst = operands * stride;
            let ran = prepared.run_lanes(&mut regs, stride, dst, active, flush);
            let alone = |lane: usize| {
                prepared.run(
                    |value| Ok(before[value.0 as usize * stride + lane]),
                    flush,
                )
            };
            match ran {
                Ok(true) => {
                    for lane in 0..stride {
                        let want = if active(lane) {
                            alone(lane).unwrap().unwrap()
                        } else {
                            before[dst + lane]
                        };
                        prop_assert_eq!(
                            canonical(c.result, regs[dst + lane]),
                            canonical(c.result, want),
                            "{:?} lane {} of {}",
                            c.op,
                            lane,
                            stride
                        );
                    }
                    prop_assert_eq!(&regs[..dst], &before[..dst], "{:?} wrote an operand", c.op);
                }
                // an operation that is not pure writes nothing
                Ok(false) => {
                    prop_assert!(!prepared.is_pure(), "{:?}", c.op);
                    prop_assert_eq!(&regs, &before, "{:?} wrote", c.op);
                }
                Err(e) => {
                    // the same failure of any lane alone
                    let lane = (0..stride).find(|&l| active(l));
                    if let Some(lane) = lane {
                        prop_assert_eq!(alone(lane), Err(e), "{:?}", c.op);
                    }
                }
            }
        }
    }
}
