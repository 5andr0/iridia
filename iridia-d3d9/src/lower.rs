use std::{collections::BTreeMap, num::NonZeroU32};

use iridia::{
    Conventions, D3dModes, Interface, Program, Provenance, ResourceId, ShaderFormat, Stage, StageFacts,
    TextureDim,
    build::{Builder, Kind, Val, Width},
    entry_io,
};
use naga::{
    AddressSpace, ArraySize, BinaryOperator as Bo, Expression as E, GlobalVariable, Handle, ImageClass,
    ImageDimension, LocalVariable, MathFunction as Mf, RelationalFunction, ResourceBinding, SampleLevel,
    ScalarKind, Statement, TypeInner, UnaryOperator,
};

use crate::{Destination, Instruction, LIFTER_VERSION, Opcode, Register, Result, Shader, Source, error};

struct Lower<'a> {
    source: &'a Shader,
    b: Builder,
    interface: Interface,
    bindings: Vec<(ResourceId, Handle<GlobalVariable>)>,
    constants: Option<Handle<GlobalVariable>>,
    samplers: BTreeMap<u32, (Handle<GlobalVariable>, Handle<GlobalVariable>, TextureDim)>,
    locals: BTreeMap<Register, (Handle<LocalVariable>, u8)>,
    inputs: BTreeMap<u32, (Val, u8, usize)>,
    definitions: BTreeMap<u32, Val>,
    /// The rows of the float constant bank `cb0`.
    rows: u32,
    notes: Vec<String>,
    at: usize,
}

/// The index of an entry point argument.
fn argument(index: usize, at: usize) -> Result<u32> {
    u32::try_from(index).map_err(|_| error(at, "more than 4294967295 inputs"))
}

pub(crate) fn lift(source: &Shader, bytes: &[u8]) -> Result<Program> {
    lower(source, bytes, None, false)
}

pub(crate) fn project(source: &Shader, bytes: &[u8], cut: crate::Projection, tail: bool) -> Result<Program> {
    if source.stage != Stage::Pixel {
        return Err(error(cut.after_word, "register RGB projections are pixel-only"));
    }
    if cut.lanes.iter().any(|&n| n > 3)
        || cut.lanes[0] == cut.lanes[1]
        || cut.lanes[0] == cut.lanes[2]
        || cut.lanes[1] == cut.lanes[2]
        || !source.instructions.iter().any(|i| i.word == cut.after_word)
    {
        return Err(error(
            cut.after_word,
            "invalid projection lanes or instruction boundary",
        ));
    }
    if tail && source.inputs.iter().any(|i| i.register == 31) {
        return Err(error(
            cut.after_word,
            "projection input register v31 already occupied",
        ));
    }
    let mut program = lower(source, bytes, Some((cut, tail)), false)?;
    // A derived value and its original program share the source hash, but are distinct cache artifacts.
    program.provenance.lifter = format!(
        "iridia-d3d9/{}:word={}:r{}:{:?}",
        if tail { "reencode" } else { "projection" },
        cut.after_word,
        cut.temporary,
        cut.lanes
    );
    Ok(program)
}

/// A sliced vertex output runs as an invocation program. Unselected RGBA lanes are placeholders (0,0,0,1).
pub(crate) fn vertex_output(source: &Shader, bytes: &[u8]) -> Result<Program> {
    lower(source, bytes, None, true)
}

fn lower(
    source: &Shader,
    bytes: &[u8],
    projection: Option<(crate::Projection, bool)>,
    pad_output: bool,
) -> Result<Program> {
    let mut l = Lower {
        source,
        b: Builder::new(),
        interface: Interface {
            stage: source.stage,
            inputs: Vec::new(),
            outputs: Vec::new(),
            resources: Vec::new(),
            facts: StageFacts::None,
        },
        bindings: Vec::new(),
        constants: None,
        samplers: BTreeMap::new(),
        locals: BTreeMap::new(),
        inputs: BTreeMap::new(),
        definitions: BTreeMap::new(),
        rows: 0,
        notes: Vec::new(),
        at: 0,
    };
    l.prepare()?;
    for i in &source.instructions {
        l.at = i.word;
        l.instruction(i)?;
        if let Some((cut, tail)) = projection
            && i.word == cut.after_word
        {
            let mask = cut.lanes.iter().fold(0, |m, n| m | (1 << n));
            let value = l.register(Register::Temporary(cut.temporary), mask)?;
            l.notes.push(format!(
                "derived register projection after word {}: r{} lanes {:?}; {}",
                cut.after_word,
                cut.temporary,
                cut.lanes,
                if tail {
                    "original tail with replacement input"
                } else {
                    "prefix only; alpha one"
                }
            ));
            if tail {
                let index = l.interface.inputs.len();
                let mut input = crate::signature::varying("DERIVED_RGB", 0, 7, 31, None);
                input.used = true;
                l.interface.inputs.push(input);
                let arg = Val {
                    expr: l.b.app(E::FunctionArgument(argument(index, cut.after_word)?)),
                    kind: Kind::F,
                    width: Width::Three,
                };
                let mut parts = (0..4).map(|n| l.b.lane(value, n)).collect::<Vec<_>>();
                for (n, &lane) in (0u8..).zip(&cut.lanes) {
                    let part = parts.get_mut(usize::from(lane)).ok_or_else(|| {
                        error(
                            cut.after_word,
                            format!("projection lane {lane}, where a register has lanes 0 to 3"),
                        )
                    })?;
                    *part = l.b.lane(arg, n);
                }
                let replaced = l.b.compose(&parts);
                l.write(
                    Destination {
                        register: Register::Temporary(cut.temporary),
                        mask,
                        saturate: false,
                        partial_precision: false,
                    },
                    replaced,
                );
            } else {
                let mut parts = cut.lanes.map(|n| l.b.lane(value, n)).to_vec();
                parts.push(l.b.lit_f(1.));
                let rgb = l.b.compose(&parts);
                l.locals.retain(|r, _| !matches!(r, Register::Target(_)));
                l.write(
                    Destination {
                        register: Register::Target(0),
                        mask: 15,
                        saturate: false,
                        partial_precision: false,
                    },
                    rgb,
                );
                break;
            }
        }
        l.b.check().map_err(|e| error(i.word, e))?;
    }
    if pad_output {
        let &(_, mask) = l
            .locals
            .get(&Register::Target(0))
            .ok_or_else(|| error(0, "projected output has no writes"))?;
        if mask != 15 {
            let zero = l.b.lit_f(0.);
            let one = l.b.lit_f(1.);
            let padding = l.b.compose(&[zero, zero, zero, one]);
            l.write(
                Destination {
                    register: Register::Target(0),
                    mask: 15 & !mask,
                    saturate: false,
                    partial_precision: false,
                },
                padding,
            );
        }
    }
    l.finish(bytes)
}

impl Lower<'_> {
    fn global(
        &mut self,
        name: String,
        ty: Handle<naga::Type>,
        group: u32,
        binding: u32,
        space: AddressSpace,
    ) -> Handle<GlobalVariable> {
        self.b.global(GlobalVariable {
            name: Some(name),
            space,
            binding: Some(ResourceBinding { group, binding }),
            ty,
            init: None,
            memory_decorations: Default::default(),
        })
    }
    fn prepare(&mut self) -> Result<()> {
        let signature = crate::signature::signature(self.source)?;
        self.interface = signature.interface;
        self.notes.extend(signature.notes);
        for i in &self.source.inputs {
            let &(slot, mask) = signature
                .inputs
                .get(&i.register)
                .ok_or_else(|| error(self.at, format!("input v{} has no slot", i.register)))?;
            let index = slot as usize;
            let width = self.interface.inputs.get(index).map_or(0, |v| v.value.width);
            let arg = Val {
                expr: self.b.app(E::FunctionArgument(slot)),
                kind: Kind::F,
                width: Width::try_from(width).map_err(|e| error(self.at, format!("an input of {e}")))?,
            };
            let zero = self.b.lit_f(0.0);
            let mut parts = [zero; 4];
            let mut lane = 0;
            for (n, p) in parts.iter_mut().enumerate() {
                if mask & (1 << n) != 0 {
                    *p = self.b.lane(arg, lane);
                    lane = lane.saturating_add(1);
                }
            }
            let expanded = self.b.compose(&parts);
            self.inputs.insert(i.register, (expanded, mask, index));
        }
        for (&n, words) in &self.source.definitions {
            let parts: Vec<_> = words.iter().map(|w| self.b.lit_f(f32::from_bits(*w))).collect();
            self.definitions.insert(n, self.b.compose(&parts));
        }
        if let Some(bank) = signature.bank {
            let rows = NonZeroU32::new(bank.rows).ok_or_else(|| error(self.at, "an empty constant bank"))?;
            let u4 = self.b.ty_of(Kind::U, Width::Four);
            let ty = self.b.ty(TypeInner::Array {
                base: u4,
                size: ArraySize::Constant(rows),
                stride: 16,
            });
            let g = self.global("cb0".into(), ty, 0, 0, AddressSpace::Uniform);
            self.constants = Some(g);
            self.rows = bank.rows;
            self.bindings.push((bank.resource, g));
        }
        for (&slot, s) in &signature.samplers {
            let naga_dim = match s.dim {
                TextureDim::D2 => ImageDimension::D2,
                TextureDim::D3 => ImageDimension::D3,
                TextureDim::Cube => ImageDimension::Cube,
                TextureDim::D1 => return Err(error(0, "1D sampler")),
            };
            let ty = self.b.ty(TypeInner::Image {
                dim: naga_dim,
                arrayed: false,
                class: ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi: false,
                },
            });
            let texture = self.global(format!("t{slot}"), ty, 1, slot, AddressSpace::Handle);
            let ty = self.b.ty(TypeInner::Sampler { comparison: false });
            let sampler = self.global(format!("s{slot}"), ty, 2, slot, AddressSpace::Handle);
            self.bindings.push((s.texture, texture));
            self.bindings.push((s.sampler, sampler));
            self.samplers.insert(slot, (texture, sampler, s.dim));
        }
        Ok(())
    }
    fn register(&mut self, r: Register, required: u8) -> Result<Val> {
        match r {
            Register::Input(n) => {
                let &(v, mask, index) = self
                    .inputs
                    .get(&n)
                    .ok_or_else(|| error(self.at, format!("undeclared input v{n}")))?;
                if required & !mask != 0 {
                    return Err(error(
                        self.at,
                        format!("v{n}: read of undeclared lanes {required:#x}"),
                    ));
                }
                let input = self.interface.inputs.get_mut(index).ok_or_else(|| {
                    error(
                        self.at,
                        format!("input v{n} is input {index} of a list that is shorter"),
                    )
                })?;
                input.used = true;
                if let Some(d) = &mut input.provenance.d3d {
                    d.read_write_mask |= required;
                }
                Ok(v)
            }
            Register::Constant(n) => {
                if let Some(&v) = self.definitions.get(&n) {
                    return Ok(v);
                }
                let g = self
                    .constants
                    .ok_or_else(|| error(self.at, "constant bank not allocated"))?;
                let base = self.b.global_ptr(g);
                let p = self.b.app(E::AccessIndex { base, index: n });
                let v = self.b.load(p, Kind::U, Width::Four);
                Ok(self.b.bitcast(v, Kind::F))
            }
            Register::Temporary(_) => {
                let &(local, mask) = self
                    .locals
                    .get(&r)
                    .ok_or_else(|| error(self.at, format!("uninitialized {r:?}")))?;
                if required & !mask != 0 {
                    return Err(error(
                        self.at,
                        format!("{r:?}: uninitialized lanes {:#x}", required & !mask),
                    ));
                }
                let p = self.b.local_ptr(local);
                Ok(self.b.load(p, Kind::F, Width::Four))
            }
            _ => Err(error(self.at, format!("not a readable float register: {r:?}"))),
        }
    }
    fn source(&mut self, s: Source, needed: u8) -> Result<Val> {
        let required = s.read_mask(needed);
        let mut v = match (s.relative, s.register) {
            (Some(lane), Register::Constant(base)) => self.relative_constant(base, lane)?,
            _ => self.register(s.register, required)?,
        };
        v = self.b.swizzle(v, &s.swizzle);
        if s.absolute {
            v = self.b.math(Mf::Abs, &[v]);
        }
        if s.negate {
            v = self.b.unary(UnaryOperator::Negate, v);
        }
        Ok(v)
    }
    /// The float constant `c[a0.lane + base]`: the row of the bank (or the `def` that sits at that register) at the
    /// address register's lane plus `base`, and zero where that is outside the 256 rows of the vertex constant file.
    fn relative_constant(&mut self, base: u32, lane: u8) -> Result<Val> {
        let &(local, written) = self
            .locals
            .get(&Register::Address)
            .ok_or_else(|| error(self.at, "a relative constant before any mova wrote a0"))?;
        if written & (1 << lane) == 0 {
            return Err(error(
                self.at,
                format!("a0 lane {lane} is read before mova writes it"),
            ));
        }
        let g = self
            .constants
            .ok_or_else(|| error(self.at, "constant bank not allocated"))?;
        let rows = i32::try_from(self.rows).map_err(|_| error(self.at, "constant bank rows"))?;
        let p = self.b.local_ptr(local);
        let a0 = self.b.load(p, Kind::I, Width::Four);
        let a = self.b.lane(a0, lane);
        let offset = self
            .b
            .lit_i(i32::try_from(base).map_err(|_| error(self.at, "constant register"))?);
        let index = self.b.binary(Bo::Add, a, offset);
        let (first, last) = (self.b.lit_i(0), self.b.lit_i(rows.saturating_sub(1)));
        let above = self.b.binary(Bo::GreaterEqual, index, first);
        let below = self.b.binary(Bo::LessEqual, index, last);
        let inside = self.b.binary(Bo::LogicalAnd, above, below);
        let raised = self.b.math(Mf::Max, &[index, first]);
        let clamped = self.b.math(Mf::Min, &[raised, last]);
        let bank = self.b.global_ptr(g);
        let row = self.b.app(E::Access {
            base: bank,
            index: clamped.expr,
        });
        let raw = self.b.load(row, Kind::U, Width::Four);
        let mut v = self.b.bitcast(raw, Kind::F);
        // a `def` constant is not in the bank: the register it names reads as its value
        let definitions: Vec<_> = self.definitions.iter().map(|(&n, &value)| (n, value)).collect();
        for (n, value) in definitions {
            let n = self
                .b
                .lit_i(i32::try_from(n).map_err(|_| error(self.at, "constant register"))?);
            let hit = self.b.binary(Bo::Equal, index, n);
            v = self.b.select(hit, value, v);
        }
        let zero = self.b.lit_f(0.0);
        Ok(self.b.select(inside, v, zero))
    }
    /// `mova`: the address register's written lanes take the operand rounded to the nearest integer (halves away from zero).
    fn write_address(&mut self, mask: u8, rounded: Val) {
        let (local, written) = *self.locals.entry(Register::Address).or_insert_with(|| {
            let ty = self.b.ty_of(Kind::I, Width::Four);
            (self.b.local("a0", ty), 0)
        });
        let p = self.b.local_ptr(local);
        for n in 0u8..4 {
            if mask & (1 << n) != 0 {
                let ptr = self.b.app(E::AccessIndex {
                    base: p,
                    index: u32::from(n),
                });
                let lane = self.b.lane(rounded, n);
                self.b.store(ptr, lane);
            }
        }
        self.locals.insert(Register::Address, (local, written | mask));
    }
    fn write(&mut self, d: Destination, mut v: Val) {
        if d.partial_precision {
            self.notes
                .push("partial-precision requests computed at f32".into());
        }
        if d.saturate {
            v = self.b.d3d_saturate(v);
        }
        let (local, mask) = *self.locals.entry(d.register).or_insert_with(|| {
            let ty = self.b.ty_of(Kind::F, Width::Four);
            (self.b.local(&format!("{:?}", d.register), ty), 0)
        });
        let p = self.b.local_ptr(local);
        if d.mask == 15 {
            self.b.store(p, v);
        } else {
            for n in 0u8..4 {
                if d.mask & (1 << n) != 0 {
                    let ptr = self.b.app(E::AccessIndex {
                        base: p,
                        index: u32::from(n),
                    });
                    let lane = self.b.lane(v, n);
                    self.b.store(ptr, lane);
                }
            }
        }
        self.locals.insert(d.register, (local, mask | d.mask));
    }
    fn sample(&mut self, i: &Instruction) -> Result<Val> {
        let s = i.source(1)?;
        let Register::Sampler(slot) = s.register else {
            return Err(error(self.at, "TEX source 1 is not a sampler"));
        };
        if s.absolute || s.negate || s.swizzle != [0, 1, 2, 3] {
            return Err(error(self.at, "modified/swizzled TEX sampler"));
        }
        let &(texture, sampler, dim) = self
            .samplers
            .get(&slot)
            .ok_or_else(|| error(self.at, format!("undeclared sampler s{slot}")))?;
        let lanes: &[u8] = if dim == TextureDim::D2 {
            &[0, 1]
        } else {
            &[0, 1, 2]
        };
        let mut required = if dim == TextureDim::D2 { 3 } else { 7 };
        if i.control != 0 {
            required |= 8;
        }
        let coordinates = self.source(i.source(0)?, required)?;
        let mut coord = self.b.swizzle(coordinates, lanes);
        let level = match i.control {
            0 => SampleLevel::Auto,
            1 => {
                let w = self.b.lane(coordinates, 3);
                coord = self.b.binary(Bo::Divide, coord, w);
                SampleLevel::Auto
            }
            2 => SampleLevel::Bias(self.b.lane(coordinates, 3).expr),
            _ => return Err(error(self.at, "TEX control 3 (projection and bias together)")),
        };
        let image = self.b.global_ptr(texture);
        let sampler = self.b.global_ptr(sampler);
        Ok(Val {
            expr: self.b.app(E::ImageSample {
                image,
                sampler,
                gather: None,
                coordinate: coord.expr,
                array_index: None,
                offset: None,
                level,
                depth_ref: None,
                clamp_to_edge: false,
            }),
            kind: Kind::F,
            width: Width::Four,
        })
    }
    fn instruction(&mut self, i: &Instruction) -> Result<()> {
        use Opcode::*;
        if i.opcode == Nop {
            return Ok(());
        }
        let d = i
            .destination
            .ok_or_else(|| error(self.at, "missing destination"))?;
        if matches!(i.opcode, Rcp | Rsq | Exp | Log | Pow | SinCos)
            && i.sources
                .iter()
                .any(|s| !s.swizzle.iter().all(|&n| n == s.swizzle[0]))
        {
            return Err(error(
                self.at,
                format!("{:?} requires replicate scalar swizzles", i.opcode),
            ));
        }
        if i.opcode == Mova {
            // to the nearest integer, a half going away from zero (Microsoft's reference rasterizer: -2.5 -> -3, 2.5 -> 3)
            let a = self.source(i.source(0)?, i.source_mask(0))?;
            let magnitude = self.b.math(Mf::Abs, &[a]);
            let half = self.b.lit_f(0.5);
            let shifted = self.b.binary(Bo::Add, magnitude, half);
            let floor = self.b.math(Mf::Floor, &[shifted]);
            let sign = self.b.math(Mf::Sign, &[a]);
            let away = self.b.binary(Bo::Multiply, sign, floor);
            let rounded = self.b.d3d_float_to_int(away);
            self.write_address(d.mask, rounded);
            return Ok(());
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
            // SM2/3 use all four lanes (the SDK's early texkill page describes SM1's xyz only).
            let v = self.register(d.register, 15)?;
            let zero = self.b.lit_f(0.0);
            let negative = self.b.binary(Bo::Less, v, zero);
            let any = Val {
                expr: self.b.app(E::Relational {
                    fun: RelationalFunction::Any,
                    argument: negative.expr,
                }),
                kind: Kind::B,
                width: Width::One,
            };
            self.b.when(any, |b| b.push(Statement::Kill));
            return Ok(());
        }
        if i.opcode == Tex {
            let v = self.sample(i)?;
            self.write(d, v);
            return Ok(());
        }
        let mut args = Vec::new();
        for (n, &s) in i.sources.iter().enumerate() {
            args.push(self.source(s, i.source_mask(n))?);
        }
        let arg = |n: usize| {
            args.get(n).copied().ok_or_else(|| {
                error(
                    i.word,
                    format!("{:?} reads source {n}, where it has {}", i.opcode, args.len()),
                )
            })
        };
        let a = arg(0)?;
        let v = match i.opcode {
            Mov => a,
            Add | Sub | Mul => self.b.binary(
                match i.opcode {
                    Add => Bo::Add,
                    Sub => Bo::Subtract,
                    _ => Bo::Multiply,
                },
                a,
                arg(1)?,
            ),
            Mad => {
                let product = self.b.binary(Bo::Multiply, a, arg(1)?);
                self.b.binary(Bo::Add, product, arg(2)?)
            }
            Abs => self.b.math(Mf::Abs, &[a]),
            Min | Max => self
                .b
                .d3d_min_max(if i.opcode == Min { Mf::Min } else { Mf::Max }, a, arg(1)?),
            Frc => self.b.math(Mf::Fract, &[a]),
            Slt | Sge => {
                let b = arg(1)?;
                let holds = self.b.binary(
                    if i.opcode == Slt {
                        Bo::Less
                    } else {
                        Bo::GreaterEqual
                    },
                    a,
                    b,
                );
                let (one, zero) = (self.b.lit_f(1.0), self.b.lit_f(0.0));
                self.b.select(holds, one, zero)
            }
            SinCos => {
                let x = self.b.lane(a, 0);
                let cosine = self.b.math(Mf::Cos, &[x]);
                let sine = self.b.math(Mf::Sin, &[x]);
                let zero = self.b.lit_f(0.0);
                self.b.compose(&[cosine, sine, zero, zero])
            }
            Cmp => {
                let zero = self.b.lit_f(0.0);
                let cond = self.b.binary(Bo::GreaterEqual, a, zero);
                self.b.select(cond, arg(1)?, arg(2)?)
            }
            Lrp => {
                let one = self.b.lit_f(1.0);
                let inverse = self.b.binary(Bo::Subtract, one, a);
                let left = self.b.binary(Bo::Multiply, a, arg(1)?);
                let right = self.b.binary(Bo::Multiply, inverse, arg(2)?);
                self.b.binary(Bo::Add, left, right)
            }
            Dp3 | Dp4 | Dp2Add => {
                let lanes: &[u8] = if i.opcode == Dp3 {
                    &[0, 1, 2]
                } else if i.opcode == Dp4 {
                    &[0, 1, 2, 3]
                } else {
                    &[0, 1]
                };
                let x = self.b.swizzle(a, lanes);
                let y = self.b.swizzle(arg(1)?, lanes);
                let mut dot = self.b.math(Mf::Dot, &[x, y]);
                if i.opcode == Dp2Add {
                    let z = self.b.lane(arg(2)?, 0);
                    dot = self.b.binary(Bo::Add, dot, z);
                }
                self.b.splat(dot, Width::Four)
            }
            Nrm => {
                let xyz = self.b.swizzle(a, &[0, 1, 2]);
                let squared = self.b.math(Mf::Dot, &[xyz, xyz]);
                let zero = self.b.lit_f(0.0);
                let empty = self.b.binary(Bo::Equal, squared, zero);
                let reciprocal = self.b.math(Mf::InverseSqrt, &[squared]);
                let largest = self.b.lit_f(f32::MAX);
                let scale = self.b.select(empty, largest, reciprocal);
                self.b.binary(Bo::Multiply, a, scale)
            }
            Rcp | Rsq | Exp | Log | Pow => {
                let mut x = self.b.lane(a, 0);
                if matches!(i.opcode, Rsq | Log | Pow) {
                    x = self.b.math(Mf::Abs, &[x]);
                }
                let scalar = match i.opcode {
                    Rcp => {
                        let one = self.b.lit_f(1.0);
                        self.b.binary(Bo::Divide, one, x)
                    }
                    Rsq => self.b.math(Mf::InverseSqrt, &[x]),
                    Exp => self.b.math(Mf::Exp2, &[x]),
                    Log => {
                        // a zero source gives -FLT_MAX, as the instruction reference's code fragment states
                        let log = self.b.math(Mf::Log2, &[x]);
                        let zero = self.b.lit_f(0.0);
                        let empty = self.b.binary(Bo::Equal, x, zero);
                        let least = self.b.lit_f(-f32::MAX);
                        self.b.select(empty, least, log)
                    }
                    _ => {
                        let y = self.b.lane(arg(1)?, 0);
                        self.b.math(Mf::Pow, &[x, y])
                    }
                };
                self.b.splat(scalar, Width::Four)
            }
            _ => {
                return Err(error(
                    self.at,
                    format!("operation {:?} has no lowering", i.opcode),
                ));
            }
        };
        self.write(d, v);
        Ok(())
    }
    fn finish(mut self, bytes: &[u8]) -> Result<Program> {
        let mut outputs = Vec::new();
        let written: BTreeMap<Register, u8> = self.locals.iter().map(|(&r, &(_, mask))| (r, mask)).collect();
        for (r, v) in crate::signature::outputs(self.source, &written, self.at)? {
            let mask = v.provenance.d3d.as_ref().map_or(0, |d| d.mask);
            let &(local, _) = self
                .locals
                .get(&r)
                .ok_or_else(|| error(self.at, format!("uninitialized output {r:?}")))?;
            self.interface.outputs.push(v);
            let ptr = self.b.local_ptr(local);
            let v = self.b.load(ptr, Kind::F, Width::Four);
            let lanes: Vec<_> = (0..4).filter(|n| mask & (1 << n) != 0).collect();
            outputs.push(self.b.swizzle(v, &lanes).expr);
        }
        let io = entry_io(&self.interface).map_err(|e| error(0, e))?;
        let decl = self.b.declare_entry(&io).map_err(|e| error(0, e))?;
        let result = decl.result.map(|ty| {
            self.b.app(E::Compose {
                ty,
                components: outputs,
            })
        });
        self.b.push(Statement::Return { value: result });
        self.b.check().map_err(|e| error(self.at, e))?;
        let at = self.at;
        let module = self
            .b
            .finish(
                "main",
                if self.source.stage == Stage::Vertex {
                    naga::ShaderStage::Vertex
                } else {
                    naga::ShaderStage::Fragment
                },
                decl.arguments,
                decl.result,
                false,
            )
            .map_err(|e| error(at, e))?;
        self.notes.extend(iridia::interface_notes(&self.interface));
        self.notes.sort();
        self.notes.dedup();
        let program = Program {
            module,
            stage: self.source.stage,
            entry: "main".into(),
            interface: self.interface,
            bindings: self.bindings,
            conventions: Conventions::D3D,
            modes: D3dModes {
                min_max_nan: true,
                saturate_nan: true,
                ..Default::default()
            },
            provenance: Provenance {
                format: ShaderFormat::D3d9,
                target: if self.source.stage == Stage::Vertex {
                    "vs_3_0"
                } else {
                    "ps_3_0"
                }
                .into(),
                lifter: "iridia-d3d9".into(),
                lifter_version: LIFTER_VERSION,
                source_hash: Some(iridia::source_hash(bytes)),
                instruction_count: None,
            },
            notes: self.notes,
        };
        program.validate().map_err(|problems| {
            error(
                0,
                problems
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })?;
        Ok(program)
    }
}
