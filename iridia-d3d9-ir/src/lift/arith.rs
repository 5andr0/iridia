//! Arithmetic: every instruction that computes from its sources and writes its destination, each by its definition
//! in Microsoft's instruction reference (or, where a version's reference leaves a case open, as DXVK computes it).

use iridia_ir::{Binary, Cmp as Compare, End, MinMaxNan, Op, Precision, Rounding, Type, Unary, ValueId};

use super::{Lift, splat4};
use crate::{Instruction, Opcode, PARTIAL_PRECISION, Register, Result, error};

/// `lit`'s bound on its exponent, `127.9961` in the reference.
const LIT_POWER: f32 = 127.9961;

impl Lift<'_> {
    pub(super) fn instruction(&mut self, i: &Instruction) -> Result<()> {
        use Opcode::*;
        // a `phase` marks where pixel 1.4's second phase starts; the pads give their rows to the instruction after
        if matches!(i.opcode, Nop | Phase) {
            return Ok(());
        }
        let d = i
            .destination
            .ok_or_else(|| error(self.at, "missing destination"))?;
        if matches!(i.opcode, Rcp | Rsq | Exp | Log | Pow | SinCos | ExpP | LogP)
            && i.sources
                .iter()
                .enumerate()
                .filter(|&(n, _)| i.source_mask(n) != 0)
                .any(|(_, s)| !s.swizzle.iter().all(|&n| n == s.swizzle[0]))
        {
            return Err(error(
                self.at,
                format!("{} requires replicate scalar swizzles", i.opcode.name()),
            ));
        }
        if d.partial_precision {
            self.notes.push(PARTIAL_PRECISION.into());
        }
        self.b
            .precision(if d.partial_precision || matches!(i.opcode, ExpP | LogP) {
                Precision::AtLeast16
            } else {
                Precision::Full
            });
        if i.opcode == Mova || i.opcode == Mov && d.register == Register::Address {
            // mova to the nearest integer, a half going away from zero (Microsoft's reference rasterizer: -2.5 -> -3,
            // 2.5 -> 3); vs 1.x's mov down, as DXVK found hardware does where the reference says to the nearest
            let a = self.source(i.source(0)?, i.source_mask(0))?;
            let rounding = if i.opcode == Mova {
                Rounding::NearestAway
            } else {
                Rounding::Floor
            };
            let int = self.address(a, rounding)?;
            return self.store(Register::Address, d.mask, int, None);
        }
        if i.opcode == SinCos && d.mask & !3 != 0 {
            return Err(error(self.at, "SINCOS writes only x, y or xy"));
        }
        if i.opcode == Nrm
            && (!matches!(d.register, Register::Temporary(_)) || i.source(0)?.register == d.register)
        {
            return Err(error(
                self.at,
                "NRM requires a temporary destination distinct from its source",
            ));
        }
        if i.opcode == TexKill {
            return self.kill(d);
        }
        if self.texture_instruction(i, d)? {
            return Ok(());
        }
        if let Some(rows) = i.matrix_rows() {
            let v = self.matrix(i, d.mask, rows)?;
            return self.write(i, d, v);
        }
        let mut args = Vec::new();
        for (n, &s) in i.sources.iter().enumerate() {
            let mask = i.source_mask(n);
            // `sincos`'s 2.0 constants and `sgn`'s scratch registers are not read
            if mask == 0 {
                continue;
            }
            args.push(self.source(s, mask)?);
        }
        let arg = |n: usize| {
            args.get(n).copied().ok_or_else(|| {
                error(
                    i.word,
                    format!(
                        "{} reads source {n}, where it has {}",
                        i.opcode.name(),
                        args.len()
                    ),
                )
            })
        };
        let a = arg(0)?;
        let v = match i.opcode {
            Mov => a,
            Add => self.binary(Binary::FAdd, a, arg(1)?)?,
            Sub => self.binary(Binary::FSub, a, arg(1)?)?,
            Mul => self.mul(a, arg(1)?)?,
            Mad => {
                let product = self.mul(a, arg(1)?)?;
                self.binary(Binary::FAdd, product, arg(2)?)?
            }
            Abs => self.unary(Unary::FAbs, a)?,
            Min => self.binary(Binary::FMin(MinMaxNan::MinNum), a, arg(1)?)?,
            Max => self.binary(Binary::FMax(MinMaxNan::MinNum), a, arg(1)?)?,
            Frc => self.unary(Unary::FFract, a)?,
            Slt | Sge => {
                let cmp = if i.opcode == Slt { Compare::Lt } else { Compare::Ge };
                let holds = self.binary(Binary::FCmp(cmp), a, arg(1)?)?;
                let one = splat4(&mut self.b, 1.0);
                let zero = splat4(&mut self.b, 0.0);
                self.select(holds, one, zero)?
            }
            SinCos => {
                let x = self.lane(a, 0)?;
                let cosine = self.unary(Unary::FCos, x)?;
                let sine = self.unary(Unary::FSin, x)?;
                let zero = self.b.f32(0.0);
                self.op(Op::Compose(vec![cosine, sine, zero, zero]))?
            }
            Cmp => {
                let zero = splat4(&mut self.b, 0.0);
                let cond = self.binary(Binary::FCmp(Compare::Ge), a, zero)?;
                self.select(cond, arg(1)?, arg(2)?)?
            }
            // 1.1 to 1.3 compare r0.a (a replicated source 0), 1.4 each lane
            Cnd => {
                let half = splat4(&mut self.b, 0.5);
                let cond = self.binary(Binary::FCmp(Compare::Gt), a, half)?;
                self.select(cond, arg(1)?, arg(2)?)?
            }
            Lrp => {
                let one = splat4(&mut self.b, 1.0);
                let inverse = self.binary(Binary::FSub, one, a)?;
                let left = self.mul(a, arg(1)?)?;
                let right = self.mul(inverse, arg(2)?)?;
                self.binary(Binary::FAdd, left, right)?
            }
            Dp3 | Dp4 | Dp2Add => {
                let lanes: &[u8] = match i.opcode {
                    Dp3 => &[0, 1, 2],
                    Dp4 => &[0, 1, 2, 3],
                    _ => &[0, 1],
                };
                let mut dot = self.dot(a, arg(1)?, lanes)?;
                if i.opcode == Dp2Add {
                    let z = self.lane(arg(2)?, 0)?;
                    dot = self.binary(Binary::FAdd, dot, z)?;
                }
                self.splat(dot, 4)?
            }
            Nrm => {
                let xyz = self.swizzle(a, &[0, 1, 2])?;
                let squared = self.op(Op::Dot(xyz, xyz))?;
                let zero = self.b.f32(0.0);
                let empty = self.binary(Binary::FCmp(Compare::Eq), squared, zero)?;
                let reciprocal = self.unary(Unary::FInverseSqrt, squared)?;
                let largest = self.b.f32(f32::MAX);
                let scale = self.select(empty, largest, reciprocal)?;
                let scale = self.splat(scale, 4)?;
                self.mul(a, scale)?
            }
            Rcp | Rsq | Exp | Log | Pow | LogP => {
                let mut x = self.lane(a, 0)?;
                if matches!(i.opcode, Rsq | Log | Pow | LogP) {
                    x = self.unary(Unary::FAbs, x)?;
                }
                let scalar = match i.opcode {
                    Rcp => {
                        let one = self.b.f32(1.0);
                        self.binary(Binary::FDiv, one, x)?
                    }
                    Rsq => self.unary(Unary::FInverseSqrt, x)?,
                    Exp => self.unary(Unary::FExp2, x)?,
                    // logp is log at partial precision in every lane, as DXVK computes it for every version
                    Log | LogP => self.unary(Unary::FLog2, x)?,
                    _ => {
                        let y = self.lane(arg(1)?, 0)?;
                        self.binary(Binary::FPow, x, y)?
                    }
                };
                self.splat(scalar, 4)?
            }
            ExpP => self.expp(a)?,
            Lit => self.lit(a)?,
            Dst => {
                let b = arg(1)?;
                let one = self.b.f32(1.0);
                let ay = self.lane(a, 1)?;
                let by = self.lane(b, 1)?;
                let y = self.mul(ay, by)?;
                let z = self.lane(a, 2)?;
                let w = self.lane(b, 3)?;
                self.op(Op::Compose(vec![one, y, z, w]))?
            }
            Crs => {
                let b = arg(1)?;
                let mut lanes = Vec::new();
                for (p, q) in [(1, 2), (2, 0), (0, 1)] {
                    let ap = self.lane(a, p)?;
                    let bq = self.lane(b, q)?;
                    let aq = self.lane(a, q)?;
                    let bp = self.lane(b, p)?;
                    let left = self.mul(ap, bq)?;
                    let right = self.mul(aq, bp)?;
                    lanes.push(self.binary(Binary::FSub, left, right)?);
                }
                let zero = self.b.f32(0.0);
                lanes.push(zero);
                self.op(Op::Compose(lanes))?
            }
            Sgn => {
                // -1 below zero, 0 at zero, 1 otherwise (NaN included), as the reference's pseudocode compares
                let zero = splat4(&mut self.b, 0.0);
                let one = splat4(&mut self.b, 1.0);
                let minus = splat4(&mut self.b, -1.0);
                let below = self.binary(Binary::FCmp(Compare::Lt), a, zero)?;
                let at_zero = self.binary(Binary::FCmp(Compare::Eq), a, zero)?;
                let rest = self.select(at_zero, zero, one)?;
                self.select(below, minus, rest)?
            }
            Bem => self.bem(d.register, a, arg(1)?)?,
            Dsx | Dsy => {
                let axis = if i.opcode == Dsx {
                    iridia_ir::Axis::X
                } else {
                    iridia_ir::Axis::Y
                };
                self.unary(Unary::Derivative(axis, iridia_ir::Fineness::Any), a)?
            }
            _ => {
                return Err(error(
                    self.at,
                    format!("operation {} has no lowering", i.opcode.name()),
                ));
            }
        };
        self.write(i, d, v)
    }

    /// The dot product of the `lanes` of two values.
    pub(super) fn dot(&mut self, a: ValueId, b: ValueId, lanes: &[u8]) -> Result<ValueId> {
        let (x, y) = if lanes.len() == 4 {
            (a, b)
        } else {
            (self.swizzle(a, lanes)?, self.swizzle(b, lanes)?)
        };
        self.op(Op::Dot(x, y))
    }

    /// `m4x4`, `m4x3`, `m3x4`, `m3x3`, `m3x2`: lane k of the result is the dot product of source 0 with the register
    /// k after source 1's (its modifiers and relative index kept), over four or three lanes.
    fn matrix(&mut self, i: &Instruction, mask: u8, rows: u32) -> Result<ValueId> {
        let lanes: &[u8] = if matches!(i.opcode, Opcode::M4x4 | Opcode::M4x3) {
            &[0, 1, 2, 3]
        } else {
            &[0, 1, 2]
        };
        let width = if lanes.len() == 4 { 15 } else { 7 };
        let a = self.source(i.source(0)?, width)?;
        let row = i.source(1)?;
        let zero = self.b.f32(0.0);
        let mut out = vec![zero; 4];
        for (k, o) in (0..rows).zip(out.iter_mut()) {
            if mask & (1 << k) == 0 {
                continue;
            }
            let mut s = row;
            s.register = match row.register {
                Register::Constant(n) => Register::Constant(n.saturating_add(k)),
                Register::Temporary(n) => Register::Temporary(n.saturating_add(k)),
                Register::Input(n) => Register::Input(n.saturating_add(k)),
                other => {
                    return Err(error(
                        self.at,
                        format!("{} reads its matrix from {}", i.opcode.name(), other.name()),
                    ));
                }
            };
            let line = self.source(s, width)?;
            *o = self.dot(a, line, lanes)?;
        }
        self.op(Op::Compose(out))
    }

    /// `expp`: from 2.0, `2^x` at partial precision in every lane; in vertex 1.x the reference's four lanes,
    /// `2^floor(x)`, `x - floor(x)`, `2^x` (at partial precision) and 1.
    fn expp(&mut self, a: ValueId) -> Result<ValueId> {
        let x = self.lane(a, 0)?;
        if self.source.version.major >= 2 {
            let e = self.unary(Unary::FExp2, x)?;
            return self.splat(e, 4);
        }
        self.b.precision(Precision::Full);
        let floor = self.unary(Unary::FRound(Rounding::Floor), x)?;
        let whole = self.unary(Unary::FExp2, floor)?;
        let fraction = self.unary(Unary::FFract, x)?;
        self.b.precision(Precision::AtLeast16);
        let partial = self.unary(Unary::FExp2, x)?;
        self.b.precision(Precision::Full);
        let one = self.b.f32(1.0);
        self.op(Op::Compose(vec![whole, fraction, partial, one]))
    }

    /// `lit`, by the reference: x 1; y `src.x` where it is above zero, else 0; z `src.y ^ power` where `src.x` and
    /// `src.y` are both above zero, else 0, with the power `src.w` bounded to ±127.9961; w 1.
    fn lit(&mut self, a: ValueId) -> Result<ValueId> {
        let x = self.lane(a, 0)?;
        let y = self.lane(a, 1)?;
        let w = self.lane(a, 3)?;
        let zero = self.b.f32(0.0);
        let one = self.b.f32(1.0);
        let max = self.b.f32(LIT_POWER);
        let min = self.b.f32(-LIT_POWER);
        // the reference's two comparisons, which leave NaN as it is
        let below = self.binary(Binary::FCmp(Compare::Lt), w, min)?;
        let above = self.binary(Binary::FCmp(Compare::Gt), w, max)?;
        let upper = self.select(above, max, w)?;
        let power = self.select(below, min, upper)?;
        let x_positive = self.binary(Binary::FCmp(Compare::Gt), x, zero)?;
        let y_positive = self.binary(Binary::FCmp(Compare::Gt), y, zero)?;
        let ly = self.select(x_positive, x, zero)?;
        let raised = self.binary(Binary::FPow, y, power)?;
        let lz = self.select(y_positive, raised, zero)?;
        let lz = self.select(x_positive, lz, zero)?;
        self.op(Op::Compose(vec![one, ly, lz, one]))
    }

    /// `bem` (pixel 1.4): x and y of source 0 plus the bump matrix of the stage the destination register names times
    /// x and y of source 1: `x + m00 * s.x + m10 * s.y`, `y + m01 * s.x + m11 * s.y`.
    pub(super) fn bem(&mut self, destination: Register, a: ValueId, b: ValueId) -> Result<ValueId> {
        let Register::Temporary(stage) = destination else {
            return Err(error(self.at, "bem writes a register that is not a temporary"));
        };
        let xy = self.bump(stage, a, b)?;
        let x = self.lane(xy, 0)?;
        let y = self.lane(xy, 1)?;
        let zero = self.b.f32(0.0);
        self.op(Op::Compose(vec![x, y, zero, zero]))
    }

    /// The bumped coordinate of `texbem` and `bem`: `base.xy` plus stage `stage`'s bump matrix times `offset.xy`.
    pub(super) fn bump(&mut self, stage: u32, base: ValueId, offset: ValueId) -> Result<ValueId> {
        let state = self
            .signature
            .state(iridia_ir::interface::StateParam::BumpEnvMat(stage))?;
        let m = self.b.value(Type::F32X4, Op::LoadState(state));
        let ox = self.lane(offset, 0)?;
        let oy = self.lane(offset, 1)?;
        let mut lanes = Vec::new();
        // the matrix is 00, 01, 10, 11: row 0 times x and row 1 times y
        for (lane, first, second) in [(0, 0, 2), (1, 1, 3)] {
            let base_lane = self.lane(base, lane)?;
            let m0 = self.lane(m, first)?;
            let m1 = self.lane(m, second)?;
            let p0 = self.mul(m0, ox)?;
            let p1 = self.mul(m1, oy)?;
            let sum = self.binary(Binary::FAdd, base_lane, p0)?;
            lanes.push(self.binary(Binary::FAdd, sum, p1)?);
        }
        self.op(Op::Compose(lanes))
    }

    /// `texkill`: discard the pixel where a lane of the register is below zero: x, y and z in 1.x (before 1.4, of the
    /// texture coordinate the register names), all four from 2.0.
    fn kill(&mut self, d: crate::Destination) -> Result<()> {
        let v = if self.source.pixel_1x() && self.source.version.minor < 4 {
            let Register::Texture(_) = d.register else {
                return Err(error(
                    self.at,
                    "texkill of a register that is not a texture register",
                ));
            };
            self.input(d.register, 7)?
        } else if self.source.pixel_1x() {
            match d.register {
                Register::Texture(_) => self.input(d.register, 7)?,
                r => self.register(r, 7)?,
            }
        } else {
            self.register(d.register, 15)?
        };
        let v = if self.source.pixel_1x() {
            self.swizzle(v, &[0, 1, 2])?
        } else {
            v
        };
        let lanes = self.ty(v)?.lanes;
        let zero = self.b.f32x4([0.0; 4]);
        let zero = if lanes == 4 {
            zero
        } else {
            self.swizzle(zero, &[0, 1, 2])?
        };
        let negative = self.binary(Binary::FCmp(Compare::Lt), v, zero)?;
        let any = self.unary(Unary::Any, negative)?;
        self.b.open_if(any);
        self.b.close(End::Kill, &[]).map_err(|e| error(self.at, e))?;
        Ok(())
    }
}
