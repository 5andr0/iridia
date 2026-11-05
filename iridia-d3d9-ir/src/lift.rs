//! The lift: each register a variable, each instruction its operations in order, the outputs stored at the end.

use std::collections::BTreeMap;

use iridia_ir::{
    Binary, Builder, Cmp, Conventions, End, FloatEnv, FloatToInt, Level, MinMaxNan, MulZero, Op, Origin,
    OutOfBounds, Precision, Program, Provenance, Rounding, SaturateNan, Scalar, Type, Unary, Unit, ValueId, VarId,
    interface::{ShaderFormat, Stage, TextureDim},
    source_hash,
};

use crate::{
    Destination, Instruction, Opcode, PARTIAL_PRECISION, Register, Result, Shader, Signature, Source, error,
    signature::{outputs, read_input, signature},
};

/// The lifter's name in a program's provenance.
pub const LIFTER: &str = "iridia-d3d9-ir";

/// Increment when the same tokens lift differently.
pub const LIFTER_VERSION: u32 = 1;

/// The compatibility policies this lifter chooses, as provenance states them.
const POLICIES: [&str; 3] = ["d3d9.mul=ieee", "d3d9.min_max_nan=min_num", "d3d9.saturate_nan=zero"];

struct Lift<'a> {
    source: &'a Shader,
    signature: Signature,
    b: Builder,
    /// The variable of each register written so far, with the lanes written.
    registers: BTreeMap<Register, (VarId, u8)>,
    /// Each declared input register as four lanes, the undeclared ones zero.
    inputs: BTreeMap<u32, ValueId>,
    /// Each `def` constant's value.
    definitions: BTreeMap<u32, ValueId>,
    notes: Vec<String>,
    at: usize,
}

/// The f32 vector of every lane `x`.
fn splat4(b: &mut Builder, x: f32) -> ValueId {
    b.f32x4([x; 4])
}

impl Lift<'_> {
    fn op(&mut self, op: Op) -> Result<ValueId> {
        self.b.op(op).map_err(|e| error(self.at, e))
    }
    fn unary(&mut self, u: Unary, a: ValueId) -> Result<ValueId> {
        self.op(Op::Unary(u, a))
    }
    fn binary(&mut self, op: Binary, a: ValueId, b: ValueId) -> Result<ValueId> {
        self.op(Op::Binary(op, a, b))
    }
    fn select(&mut self, cond: ValueId, accept: ValueId, reject: ValueId) -> Result<ValueId> {
        self.op(Op::Select { cond, accept, reject })
    }
    fn ty(&self, v: ValueId) -> Result<Type> {
        self.b.ty(v).ok_or_else(|| error(self.at, format!("value %{} has no type", v.0)))
    }
    /// Lanes of a value, by index: a scalar for one.
    fn swizzle(&mut self, v: ValueId, lanes: &[u8]) -> Result<ValueId> {
        let ty = self.ty(v)?;
        let n = u8::try_from(lanes.len()).map_err(|_| error(self.at, "a swizzle of more than four lanes"))?;
        let mut all = [0u8; 4];
        for (to, from) in all.iter_mut().zip(lanes) {
            *to = *from;
        }
        Ok(self.b.value(Type::vector(ty.scalar, n), Op::Swizzle(v, all)))
    }
    fn lane(&mut self, v: ValueId, n: u8) -> Result<ValueId> {
        self.swizzle(v, &[n])
    }
    fn splat(&mut self, v: ValueId, lanes: u8) -> Result<ValueId> {
        let ty = self.ty(v)?;
        Ok(self.b.value(Type::vector(ty.scalar, lanes), Op::Splat(v)))
    }
    fn var(&mut self, r: Register) -> VarId {
        let ty = if r == Register::Address { Type::vector(Scalar::I32, 4) } else { Type::F32X4 };
        match self.registers.get(&r) {
            Some(&(v, _)) => v,
            None => {
                let name = match r {
                    Register::Temporary(n) => format!("r{n}"),
                    Register::Target(n) => format!("oC{n}"),
                    Register::Output(n) => format!("o{n}"),
                    Register::Address => "a0".into(),
                    other => format!("{other:?}"),
                };
                let v = self.b.var(&name, ty);
                self.registers.insert(r, (v, 0));
                v
            }
        }
    }
    /// Store `v`'s lanes of `mask` into a register's variable, its other lanes kept.
    fn store(&mut self, r: Register, mask: u8, v: ValueId) -> Result<()> {
        let var = self.var(r);
        let ty = self.ty(v)?;
        let written = self.registers.get(&r).map_or(0, |&(_, m)| m);
        let whole = if mask == 15 {
            v
        } else {
            let old = self.b.value(ty, Op::LoadVar(var));
            let mut parts = Vec::new();
            for n in 0u8..4 {
                let from = if mask & (1 << n) != 0 { v } else { old };
                parts.push(self.lane(from, n)?);
            }
            self.op(Op::Compose(parts))?
        };
        self.b.store_var(var, whole);
        self.registers.insert(r, (var, written | mask));
        Ok(())
    }
    fn prepare(&mut self) -> Result<()> {
        for i in &self.source.inputs {
            let &(slot, mask) = self
                .signature
                .inputs
                .get(&i.register)
                .ok_or_else(|| error(self.at, format!("input v{} has no slot", i.register)))?;
            let width = self.signature.interface.inputs.get(slot as usize).map_or(0, |v| v.value.width);
            let arg = self.b.value(Type::vector(Scalar::F32, width), Op::LoadInput(slot));
            let zero = self.b.f32(0.0);
            let mut parts = [zero; 4];
            let mut lane = 0;
            for (n, p) in (0u8..).zip(parts.iter_mut()) {
                if mask & (1 << n) != 0 {
                    *p = if width == 1 { arg } else { self.lane(arg, lane)? };
                    lane = lane.saturating_add(1);
                }
            }
            let expanded = self.op(Op::Compose(parts.to_vec()))?;
            self.inputs.insert(i.register, expanded);
        }
        for (&n, words) in &self.source.definitions {
            let v = self.b.literal(Type::F32X4, words.map(u64::from));
            self.definitions.insert(n, v);
        }
        Ok(())
    }
    fn bank(&self) -> Result<crate::Bank> {
        self.signature.bank.ok_or_else(|| error(self.at, "constant bank not allocated"))
    }
    fn register(&mut self, r: Register, required: u8) -> Result<ValueId> {
        match r {
            Register::Input(n) => {
                read_input(&mut self.signature, n, required, self.at)?;
                self.inputs.get(&n).copied().ok_or_else(|| error(self.at, format!("undeclared input v{n}")))
            }
            Register::Constant(n) => {
                if let Some(&v) = self.definitions.get(&n) {
                    return Ok(v);
                }
                let bank = self.bank()?;
                let load = Op::LoadBuffer {
                    resource: bank.resource,
                    index: None,
                    stride: 16,
                    offset: n.saturating_mul(16),
                    oob: OutOfBounds::Zero,
                };
                Ok(self.b.value(Type::F32X4, load))
            }
            Register::Temporary(_) => {
                let &(var, mask) =
                    self.registers.get(&r).ok_or_else(|| error(self.at, format!("uninitialized {r:?}")))?;
                if required & !mask != 0 {
                    return Err(error(self.at, format!("{r:?}: uninitialized lanes {:#x}", required & !mask)));
                }
                Ok(self.b.value(Type::F32X4, Op::LoadVar(var)))
            }
            _ => Err(error(self.at, format!("not a readable float register: {r:?}"))),
        }
    }
    fn source(&mut self, s: Source, needed: u8) -> Result<ValueId> {
        let required = s.read_mask(needed);
        let mut v = match (s.relative, s.register) {
            (Some(lane), Register::Constant(base)) => self.relative_constant(base, lane)?,
            _ => self.register(s.register, required)?,
        };
        if s.swizzle != [0, 1, 2, 3] {
            v = self.swizzle(v, &s.swizzle)?;
        }
        if s.absolute {
            v = self.unary(Unary::FAbs, v)?;
        }
        if s.negate {
            v = self.unary(Unary::FNeg, v)?;
        }
        Ok(v)
    }
    /// The float constant `c[a0.lane + base]`: the row of the bank (or the `def` that sits at that register) at the
    /// address register's lane plus `base`, and zero outside the bank.
    fn relative_constant(&mut self, base: u32, lane: u8) -> Result<ValueId> {
        let &(var, written) = self
            .registers
            .get(&Register::Address)
            .ok_or_else(|| error(self.at, "a relative constant before any mova wrote a0"))?;
        if written & (1 << lane) == 0 {
            return Err(error(self.at, format!("a0 lane {lane} is read before mova writes it")));
        }
        let bank = self.bank()?;
        let a0 = self.b.value(Type::vector(Scalar::I32, 4), Op::LoadVar(var));
        let a = self.lane(a0, lane)?;
        let offset = self.b.i32(i32::try_from(base).map_err(|_| error(self.at, "constant register"))?);
        let index = self.binary(Binary::IAdd, a, offset)?;
        let load = Op::LoadBuffer {
            resource: bank.resource,
            index: Some(index),
            stride: 16,
            offset: 0,
            oob: OutOfBounds::Zero,
        };
        let mut v = self.b.value(Type::F32X4, load);
        // a `def` constant is not in the bank: the register it names reads as its value
        let definitions: Vec<_> = self.definitions.iter().map(|(&n, &value)| (n, value)).collect();
        for (n, value) in definitions {
            let n = self.b.i32(i32::try_from(n).map_err(|_| error(self.at, "constant register"))?);
            let hit = self.binary(Binary::SCmp(Cmp::Eq), index, n)?;
            v = self.select(hit, value, v)?;
        }
        Ok(v)
    }
    fn write(&mut self, d: Destination, mut v: ValueId) -> Result<()> {
        if d.saturate {
            v = self.unary(Unary::Saturate(SaturateNan::Zero), v)?;
        }
        self.store(d.register, d.mask, v)
    }
    fn sample(&mut self, i: &Instruction) -> Result<ValueId> {
        let s = i.source(1)?;
        let Register::Sampler(slot) = s.register else {
            return Err(error(self.at, "TEX source 1 is not a sampler"));
        };
        if s.absolute || s.negate || s.swizzle != [0, 1, 2, 3] {
            return Err(error(self.at, "modified/swizzled TEX sampler"));
        }
        let sampler = *self
            .signature
            .samplers
            .get(&slot)
            .ok_or_else(|| error(self.at, format!("undeclared sampler s{slot}")))?;
        let lanes: &[u8] = if sampler.dim == TextureDim::D2 { &[0, 1] } else { &[0, 1, 2] };
        let mut required = if sampler.dim == TextureDim::D2 { 3 } else { 7 };
        if i.control != 0 {
            required |= 8;
        }
        let coordinates = self.source(i.source(0)?, required)?;
        let mut coord = self.swizzle(coordinates, lanes)?;
        let width = u8::try_from(lanes.len()).unwrap_or(3);
        let level = match i.control {
            0 => Level::Auto,
            1 => {
                let w = self.lane(coordinates, 3)?;
                let w = self.splat(w, width)?;
                coord = self.binary(Binary::FDiv, coord, w)?;
                Level::Auto
            }
            2 => Level::Bias(self.lane(coordinates, 3)?),
            _ => return Err(error(self.at, "TEX control 3 (projection and bias together)")),
        };
        let op = Op::Sample { texture: sampler.texture, sampler: sampler.sampler, coord, level };
        Ok(self.b.value(Type::F32X4, op))
    }
    fn instruction(&mut self, i: &Instruction) -> Result<()> {
        use Opcode::*;
        if i.opcode == Nop {
            return Ok(());
        }
        let d = i.destination.ok_or_else(|| error(self.at, "missing destination"))?;
        if matches!(i.opcode, Rcp | Rsq | Exp | Log | Pow | SinCos)
            && i.sources.iter().any(|s| !s.swizzle.iter().all(|&n| n == s.swizzle[0]))
        {
            return Err(error(self.at, format!("{:?} requires replicate scalar swizzles", i.opcode)));
        }
        if d.partial_precision {
            self.notes.push(PARTIAL_PRECISION.into());
        }
        self.b.precision(if d.partial_precision { Precision::AtLeast16 } else { Precision::Full });
        if i.opcode == Mova {
            // to the nearest integer, a half going away from zero (Microsoft's reference rasterizer: -2.5 -> -3, 2.5 -> 3)
            let a = self.source(i.source(0)?, i.source_mask(0))?;
            let rounded = self.unary(Unary::FRound(Rounding::NearestAway), a)?;
            let ty = self.ty(rounded)?;
            let int = self.b.value(ty.with(Scalar::I32), Op::Convert(rounded, FloatToInt::Saturate));
            return self.store(Register::Address, d.mask, int);
        }
        if i.opcode == SinCos && d.mask & !3 != 0 {
            return Err(error(self.at, "SINCOS writes only x, y or xy"));
        }
        if i.opcode == Nrm && (!matches!(d.register, Register::Temporary(_)) || i.source(0)?.register == d.register) {
            return Err(error(self.at, "NRM requires a temporary destination distinct from its source"));
        }
        if i.opcode == TexKill {
            // SM2/3 use all four lanes (the SDK's early texkill page describes SM1's xyz only).
            let v = self.register(d.register, 15)?;
            let zero = splat4(&mut self.b, 0.0);
            let negative = self.binary(Binary::FCmp(iridia_ir::Cmp::Lt), v, zero)?;
            let any = self.unary(Unary::Any, negative)?;
            self.b.open_if(any);
            self.b.close(End::Kill, &[]).map_err(|e| error(self.at, e))?;
            return Ok(());
        }
        if i.opcode == Tex {
            let v = self.sample(i)?;
            return self.write(d, v);
        }
        let mut args = Vec::new();
        for (n, &s) in i.sources.iter().enumerate() {
            args.push(self.source(s, i.source_mask(n))?);
        }
        let arg = |n: usize| {
            args.get(n).copied().ok_or_else(|| {
                error(i.word, format!("{:?} reads source {n}, where it has {}", i.opcode, args.len()))
            })
        };
        let a = arg(0)?;
        let mul = Binary::FMul(MulZero::Ieee);
        let v = match i.opcode {
            Mov => a,
            Add => self.binary(Binary::FAdd, a, arg(1)?)?,
            Sub => self.binary(Binary::FSub, a, arg(1)?)?,
            Mul => self.binary(mul, a, arg(1)?)?,
            Mad => {
                let product = self.binary(mul, a, arg(1)?)?;
                self.binary(Binary::FAdd, product, arg(2)?)?
            }
            Abs => self.unary(Unary::FAbs, a)?,
            Min => self.binary(Binary::FMin(MinMaxNan::MinNum), a, arg(1)?)?,
            Max => self.binary(Binary::FMax(MinMaxNan::MinNum), a, arg(1)?)?,
            Frc => self.unary(Unary::FFract, a)?,
            Slt | Sge => {
                let cmp = if i.opcode == Slt { iridia_ir::Cmp::Lt } else { iridia_ir::Cmp::Ge };
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
                let cond = self.binary(Binary::FCmp(iridia_ir::Cmp::Ge), a, zero)?;
                self.select(cond, arg(1)?, arg(2)?)?
            }
            Lrp => {
                let one = splat4(&mut self.b, 1.0);
                let inverse = self.binary(Binary::FSub, one, a)?;
                let left = self.binary(mul, a, arg(1)?)?;
                let right = self.binary(mul, inverse, arg(2)?)?;
                self.binary(Binary::FAdd, left, right)?
            }
            Dp3 | Dp4 | Dp2Add => {
                let lanes: &[u8] = match i.opcode {
                    Dp3 => &[0, 1, 2],
                    Dp4 => &[0, 1, 2, 3],
                    _ => &[0, 1],
                };
                let x = if lanes.len() == 4 { a } else { self.swizzle(a, lanes)? };
                let y = arg(1)?;
                let y = if lanes.len() == 4 { y } else { self.swizzle(y, lanes)? };
                let mut dot = self.op(Op::Dot(x, y))?;
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
                let empty = self.binary(Binary::FCmp(iridia_ir::Cmp::Eq), squared, zero)?;
                let reciprocal = self.unary(Unary::FInverseSqrt, squared)?;
                let largest = self.b.f32(f32::MAX);
                let scale = self.select(empty, largest, reciprocal)?;
                let scale = self.splat(scale, 4)?;
                self.binary(mul, a, scale)?
            }
            Rcp | Rsq | Exp | Log | Pow => {
                let mut x = self.lane(a, 0)?;
                if matches!(i.opcode, Rsq | Log | Pow) {
                    x = self.unary(Unary::FAbs, x)?;
                }
                let scalar = match i.opcode {
                    Rcp => {
                        let one = self.b.f32(1.0);
                        self.binary(Binary::FDiv, one, x)?
                    }
                    Rsq => self.unary(Unary::FInverseSqrt, x)?,
                    Exp => self.unary(Unary::FExp2, x)?,
                    Log => self.unary(Unary::FLog2, x)?,
                    _ => {
                        let y = self.lane(arg(1)?, 0)?;
                        self.binary(Binary::FPow, x, y)?
                    }
                };
                self.splat(scalar, 4)?
            }
            _ => return Err(error(self.at, format!("operation {:?} has no lowering", i.opcode))),
        };
        self.write(d, v)
    }
    fn finish(mut self, bytes: &[u8]) -> Result<Program> {
        self.b.precision(Precision::Full);
        self.b.at(None);
        let written: BTreeMap<Register, u8> = self.registers.iter().map(|(&r, &(_, m))| (r, m)).collect();
        let outs = outputs(self.source, &written, self.at)?;
        for (slot, (r, v)) in (0u32..).zip(outs) {
            let mask = v.provenance.d3d.as_ref().map_or(0, |d| d.mask);
            let &(var, _) =
                self.registers.get(&r).ok_or_else(|| error(self.at, format!("uninitialized output {r:?}")))?;
            self.signature.interface.outputs.push(v);
            let value = self.b.value(Type::F32X4, Op::LoadVar(var));
            let lanes: Vec<u8> = (0u8..4).filter(|n| mask & (1 << n) != 0).collect();
            let value = if lanes.len() == 4 { value } else { self.swizzle(value, &lanes)? };
            self.b.store_output(slot, value);
        }
        let entry = self.b.finish(End::Return).map_err(|e| error(self.at, e))?;
        let mut notes = self.notes;
        notes.extend(self.signature.notes);
        notes.sort();
        notes.dedup();
        let stage = self.source.stage;
        let program = Program {
            stage,
            entry,
            interface: self.signature.interface,
            conventions: Conventions::D3D,
            float: FloatEnv::default(),
            provenance: Provenance {
                format: ShaderFormat::D3d9,
                target: if stage == Stage::Vertex { "vs_3_0" } else { "ps_3_0" }.into(),
                lifter: LIFTER.into(),
                lifter_version: LIFTER_VERSION,
                source_hash: Some(source_hash(bytes)),
                instruction_count: None,
                policies: POLICIES.iter().map(|p| (*p).into()).collect(),
            },
            notes,
        };
        iridia_ir::validate(&program).map_err(|problems| {
            error(0, problems.iter().map(ToString::to_string).collect::<Vec<_>>().join("; "))
        })?;
        Ok(program)
    }
}

/// Lift one complete `ps_3_0` or `vs_3_0` program; trailing bytes and unknown features are refused.
pub fn lift(bytes: &[u8]) -> Result<Program> {
    let source = crate::read(bytes)?;
    let signature = signature(&source)?;
    let mut l = Lift {
        source: &source,
        signature,
        b: Builder::new("main"),
        registers: BTreeMap::new(),
        inputs: BTreeMap::new(),
        definitions: BTreeMap::new(),
        notes: Vec::new(),
        at: 0,
    };
    l.prepare()?;
    for i in &source.instructions {
        l.at = i.word;
        l.b.at(Some(Origin { unit: Unit::Word, offset: u32::try_from(i.word).unwrap_or(u32::MAX) }));
        l.instruction(i)?;
    }
    l.finish(bytes)
}
