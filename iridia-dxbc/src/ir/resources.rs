//! The declared resources, and the instructions that read them: samples, gathers, loads, sizes and buffer words.

use std::collections::BTreeMap;

use iridia_ir::{
    Binary, Cmp, FloatToInt, GatherOffset, Level, Op, Query, Res, Scalar, Space, Type, ValueId, VarId,
    interface::{BufferKind, ResourceId, ResourceKind, ScalarType},
};

use super::operands::{OperandAt, of_four};
use super::{Lift, lanes_of};
use crate::Result;
use crate::setup::{Declaration, Texture, UavKind};
use dxbc_reader::decode::{Instruction, Operand, Register, Selection};
use iridia::D3dClass;

/// How a shader-resource buffer `t#` is addressed.
#[derive(Clone, Copy, Debug)]
pub(super) enum Buffer {
    /// Four lanes per element, of this scalar.
    Typed(Scalar),
    /// Bytes; a structured element takes `stride` of them, a raw buffer has none.
    Words { stride: Option<u32> },
}

/// An unordered-access view `u#`.
#[derive(Clone, Copy, Debug)]
pub(super) enum Uav {
    /// A typed texture view: its coordinate lanes (without a layer), whether it is arrayed, and its lanes' scalar.
    Image { dims: u8, arrayed: bool, scalar: Scalar },
    /// A typed buffer: four lanes of this scalar per element.
    Typed(Scalar),
    /// Bytes; a structured element takes `stride` of them, a raw buffer has none.
    Words { stride: Option<u32> },
}

/// Thread-group shared memory `g#`: the array variable of its words, and a structured one's element stride in bytes.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shared {
    pub var: VarId,
    pub stride: Option<u32>,
}

#[derive(Default)]
pub(super) struct Resources {
    pub constant_buffers: BTreeMap<u32, ResourceId>,
    pub textures: BTreeMap<u32, (ResourceId, Texture)>,
    pub samplers: BTreeMap<u32, ResourceId>,
    pub buffers: BTreeMap<u32, (ResourceId, Buffer)>,
    pub uavs: BTreeMap<u32, (ResourceId, Uav)>,
    pub shared: BTreeMap<u32, Shared>,
    /// The immediate constant buffer, as an array variable, and its rows.
    pub immediate: Option<(VarId, u32)>,
}

/// The IR scalar of a lifted kind: a texel's or element's lanes.
pub(super) fn scalar_of_kind(k: iridia::build::Kind) -> Scalar {
    match k {
        iridia::build::Kind::I => Scalar::I32,
        iridia::build::Kind::U => Scalar::U32,
        _ => Scalar::F32,
    }
}

/// The coordinate lanes of a texture: 1 (1D), 2 (2D, cube faces' size), 3 (3D, cube direction).
fn dims(t: &Texture) -> u8 {
    match t.dim {
        naga::ImageDimension::D1 => 1,
        naga::ImageDimension::D2 => 2,
        naga::ImageDimension::D3 | naga::ImageDimension::Cube => 3,
    }
}

/// The scalar a texture's loads return.
fn texel_scalar(t: &Texture) -> Scalar {
    scalar_of_kind(t.kind)
}

impl Lift<'_> {
    pub(super) fn declare_resources(&mut self) -> Result<()> {
        for (i, d) in self.setup.declarations.clone().iter().enumerate() {
            let Ok(id) = u32::try_from(i).map(ResourceId) else {
                return self.refuse("more than 4294967295 resources");
            };
            match d {
                Declaration::ConstantBuffer { slot, .. } => {
                    self.resources.constant_buffers.insert(*slot, id);
                }
                Declaration::Texture(t) => {
                    self.resources.textures.insert(t.slot, (id, *t));
                }
                Declaration::Sampler { slot, .. } => {
                    self.resources.samplers.insert(*slot, id);
                }
                Declaration::TypedBuffer { slot } => {
                    let scalar = match self
                        .setup
                        .interface
                        .resources
                        .iter()
                        .find(|r| r.id == id)
                        .map(|r| &r.kind)
                    {
                        Some(ResourceKind::Buffer(BufferKind::Typed(v))) => match v.scalar {
                            ScalarType::Sint => Scalar::I32,
                            ScalarType::Uint => Scalar::U32,
                            _ => Scalar::F32,
                        },
                        _ => {
                            return self
                                .refuse(format!("t{slot}: a typed buffer the interface does not state"));
                        }
                    };
                    self.resources.buffers.insert(*slot, (id, Buffer::Typed(scalar)));
                }
                Declaration::RawBuffer { slot } => {
                    self.resources
                        .buffers
                        .insert(*slot, (id, Buffer::Words { stride: None }));
                }
                Declaration::StructuredBuffer { slot, stride } => {
                    self.resources.buffers.insert(
                        *slot,
                        (
                            id,
                            Buffer::Words {
                                stride: Some((*stride).max(4)),
                            },
                        ),
                    );
                }
                Declaration::View(v) => {
                    let dims = match v.dim {
                        naga::ImageDimension::D1 => 1,
                        naga::ImageDimension::D2 => 2,
                        naga::ImageDimension::D3 | naga::ImageDimension::Cube => 3,
                    };
                    let uav = Uav::Image {
                        dims,
                        arrayed: v.arrayed,
                        scalar: scalar_of_kind(v.kind),
                    };
                    self.resources.uavs.insert(v.slot, (id, uav));
                }
                Declaration::UavBuffer(u) => {
                    let uav = match u.kind {
                        UavKind::Raw => Uav::Words { stride: None },
                        UavKind::Structured { stride } => Uav::Words { stride: Some(stride) },
                        UavKind::Typed(k) => Uav::Typed(scalar_of_kind(k)),
                    };
                    self.resources.uavs.insert(u.slot, (id, uav));
                }
            }
        }
        for g in self.setup.shared.clone() {
            let var = self.b.array(
                &format!("g{}", g.register),
                Type::U32,
                g.words,
                Space::Workgroup,
                Vec::new(),
            );
            self.resources.shared.insert(
                g.register,
                Shared {
                    var,
                    stride: g.stride,
                },
            );
        }
        if !self.setup.immediates.is_empty() {
            let Ok(rows) = u32::try_from(self.setup.immediates.len()) else {
                return self.refuse("more than 4294967295 immediate rows");
            };
            let init = self
                .setup
                .immediates
                .iter()
                .map(|row| row.map(u64::from))
                .collect();
            let var = self.b.array("icb", Type::U32X4, rows, Space::Private, init);
            self.resources.immediate = Some((var, rows));
        }
        Ok(())
    }

    /// The resource a register operand names: its declaration's, and in Shader Model 5.1 the element of the declared
    /// range the operand's register index picks (counted from the range's first register; a computed index may differ
    /// between invocations where the operand says so).
    pub(super) fn res_of(&mut self, op: &Operand, id: ResourceId, class: D3dClass) -> Result<Res> {
        if !self.setup.sm51 {
            return Ok(Res::of(id));
        }
        let declaration = self.index_of(op, 0)?.clone();
        if declaration.relative.is_some() {
            return self.refuse(
                "a declaration ID addressed by a computed index, which Shader Model 5.1 does not have",
            );
        }
        let Some(binding) = self.setup.bindings.get(&(class, declaration.base)).copied() else {
            return self.refuse(format!("{class:?} {} has no declared range", declaration.base));
        };
        let expected = if op.register == Register::Constant { 3 } else { 2 };
        if op.indices.len() != expected {
            return self.refuse(format!(
                "a Shader Model 5.1 {:?} operand of {} indices, where it has {expected}",
                op.register,
                op.indices.len()
            ));
        }
        let index = self.index_of(op, 1)?.clone();
        let range = binding.range;
        if index.relative.is_none() && !(range.lower..=range.upper).contains(&index.base) {
            return self.refuse(format!(
                "register {} outside its declared range {}..={}",
                index.base, range.lower, range.upper
            ));
        }
        if binding.count == 1 {
            return Ok(Res::of(id));
        }
        let absolute = self.index_value(&index)?;
        let lower = self.b.u32(range.lower);
        let element = self.binary(Binary::ISub, absolute, lower)?;
        Ok(Res {
            id,
            index: Some(element),
            nonuniform: op.non_uniform,
        })
    }

    fn texture(&self, slot: u32) -> Result<(ResourceId, Texture)> {
        match self.resources.textures.get(&slot) {
            Some(t) => Ok(*t),
            None if self.resources.buffers.contains_key(&slot) => {
                self.refuse(format!("t{slot} is a buffer, used as a texture"))
            }
            None => self.refuse(format!("t{slot} is used but not declared")),
        }
    }

    /// The slot operand `n` of `ins` names, which must be a register of one of `kinds`.
    fn slot_of(&self, ins: &Instruction, n: usize, kinds: &[Register]) -> Result<u32> {
        let operand = ins.operand(n)?;
        match operand.indices.first() {
            Some(index) if kinds.contains(&operand.register) => Ok(index.base),
            _ => self.refuse(format!(
                "operand {n} is {:?}, where {kinds:?} is taken",
                operand.register
            )),
        }
    }

    /// The result lanes of `mask` of a resource operand's swizzle applied to a four-lane result.
    pub(super) fn pick(&mut self, result: ValueId, op: &Operand, mask: u8) -> Result<ValueId> {
        let lanes: Vec<u8> = lanes_of(mask)
            .iter()
            .map(|l| match op.selection {
                Selection::Swizzle(s) => of_four(s, *l),
                Selection::Scalar(c) => c,
                _ => 0,
            })
            .collect();
        self.swizzle(result, &lanes)
    }

    /// A depth texture's one lane as D3D's float view returns it: the depth in x, then 0, 0, 1.
    fn depth_view(&mut self, d: ValueId) -> Result<ValueId> {
        let zero = self.b.f32(0.0);
        let one = self.b.f32(1.0);
        self.op(Op::Compose(vec![d, zero, zero, one]))
    }

    pub(super) fn sample(&mut self, ins: &Instruction) -> Result<()> {
        let name = ins.name();
        let slot = self.slot_of(ins, 2, &[Register::Texture])?;
        let sampler = self.slot_of(ins, 3, &[Register::Sampler])?;
        let (texture, t) = self.texture(slot)?;
        let Some(&sampler) = self.resources.samplers.get(&sampler) else {
            return self.refuse(format!("s{sampler} is used but not declared"));
        };
        if t.multi {
            return self.refuse("a sample of a multisampled texture");
        }
        let n = dims(&t);
        let mask = self.dest_mask(ins.operand(0)?);
        let lanes: Vec<u8> = (0..n.saturating_add(u8::from(t.arrayed))).collect();
        let coord = self.read_lanes(ins.operand(1)?, &lanes, Scalar::F32)?;
        let mut offset = [0i8; 3];
        if let Some(o) = ins.sample_offset() {
            for (to, from) in offset.iter_mut().zip(o).take(usize::from(n)) {
                *to = i8::try_from(from).unwrap_or(0);
            }
        }
        let component = match ins.operand(3)?.selection {
            Selection::Scalar(c) => c,
            Selection::Swizzle(s) => s[0],
            _ => 0,
        };
        let reference = |l: &mut Self| l.read(ins.operand(4)?, 1, Scalar::F32);
        let gradient = |l: &mut Self, k: usize| {
            let g: Vec<u8> = (0..n).collect();
            l.read_lanes(ins.operand(k)?, &g, Scalar::F32)
        };
        let texture = self.res_of(ins.operand(2)?, texture, D3dClass::ShaderResource)?;
        let sampler = self.res_of(ins.operand(3)?, sampler, D3dClass::Sampler)?;
        let (op, gather, compare) = match name {
            "gather4" | "gather4_c" => {
                let compare = if name == "gather4_c" {
                    Some(reference(self)?)
                } else {
                    None
                };
                (
                    Op::Gather {
                        texture,
                        sampler,
                        coord,
                        component,
                        compare,
                        offset: GatherOffset::Fixed(offset),
                    },
                    true,
                    compare.is_some(),
                )
            }
            _ => {
                let (level, compare) = match name {
                    "sample" => (Level::Auto, None),
                    "sample_l" => (Level::Exact(self.read(ins.operand(4)?, 1, Scalar::F32)?), None),
                    "sample_b" => (Level::Bias(self.read(ins.operand(4)?, 1, Scalar::F32)?), None),
                    "sample_d" => (Level::Grad(gradient(self, 4)?, gradient(self, 5)?), None),
                    "sample_c" => (Level::Auto, Some(reference(self)?)),
                    "sample_c_lz" => (Level::Zero, Some(reference(self)?)),
                    other => return self.refuse(format!("{other} is not lifted to the IR")),
                };
                (
                    Op::Sample {
                        texture,
                        sampler,
                        coord,
                        level,
                        compare,
                        offset,
                        clamp: None,
                    },
                    false,
                    compare.is_some(),
                )
            }
        };
        let one = t.depth && !gather;
        // a sample or gather gives the texture's sample type; a depth texture's, and a comparison's, floats
        let scalar = if t.depth || compare {
            Scalar::F32
        } else {
            texel_scalar(&t)
        };
        let mut result = self
            .b
            .value(if one { Type::F32 } else { Type::vector(scalar, 4) }, op);
        if one {
            // D3D's float view of a depth texture: the depth, then 0, 0, 1; a comparison's result in every lane
            result = if compare {
                self.b.value(Type::F32X4, Op::Splat(result))
            } else {
                self.depth_view(result)?
            };
        }
        let bits = self.bitcast(result, Scalar::U32)?;
        let picked = self.pick(bits, ins.operand(2)?, mask)?;
        self.write(ins.operand(0)?, picked, ins.saturate)
    }

    pub(super) fn load(&mut self, ins: &Instruction) -> Result<()> {
        let slot = self.slot_of(ins, 2, &[Register::Texture])?;
        let mask = self.dest_mask(ins.operand(0)?);
        if let Some((id, buffer)) = self.resources.buffers.get(&slot).copied() {
            let Buffer::Typed(scalar) = buffer else {
                return self.refuse("ld on a raw or structured buffer");
            };
            let element = self.read(ins.operand(1)?, 1, Scalar::I32)?;
            let texture = self.res_of(ins.operand(2)?, id, D3dClass::ShaderResource)?;
            let texel = self.b.value(
                Type::vector(scalar, 4),
                Op::LoadTexel {
                    texture,
                    coord: element,
                    level: None,
                    sample: None,
                    offset: [0; 3],
                },
            );
            let row = self.bitcast(texel, Scalar::U32)?;
            let picked = self.pick(row, ins.operand(2)?, mask)?;
            return self.write(ins.operand(0)?, picked, false);
        }
        let (id, t) = self.texture(slot)?;
        let n = dims(&t);
        let all = self.read_lanes(ins.operand(1)?, &[0, 1, 2, 3], Scalar::I32)?;
        let lanes: Vec<u8> = (0..n.saturating_add(u8::from(t.arrayed))).collect();
        let coord = self.swizzle(all, &lanes)?;
        let level = if t.multi { None } else { Some(self.lane(all, 3)?) };
        let sample = if ins.name() == "ld_ms" {
            Some(self.read(ins.operand(3)?, 1, Scalar::I32)?)
        } else {
            None
        };
        let mut offset = [0i8; 3];
        if let Some(o) = ins.sample_offset() {
            for (to, from) in offset.iter_mut().zip(o).take(usize::from(n)) {
                *to = i8::try_from(from).unwrap_or(0);
            }
        }
        let op = Op::LoadTexel {
            texture: self.res_of(ins.operand(2)?, id, D3dClass::ShaderResource)?,
            coord,
            level,
            sample,
            offset,
        };
        let result = if t.depth {
            let d = self.b.value(Type::F32, op);
            self.depth_view(d)?
        } else {
            self.b.value(Type::vector(texel_scalar(&t), 4), op)
        };
        let bits = self.bitcast(result, Scalar::U32)?;
        let picked = self.pick(bits, ins.operand(2)?, mask)?;
        self.write(ins.operand(0)?, picked, false)
    }

    /// `sample_info dest, t#`: the sample count of a multisampled texture in x (0 for one that is not), and 0 in the
    /// other lanes, as a uint or a float by the instruction's control.
    pub(super) fn sample_info(&mut self, ins: &Instruction) -> Result<()> {
        if ins.operand(1)?.register != Register::Texture {
            return self.refuse("the sample count of the rasterizer is the pipeline's, not the program's");
        }
        let slot = self.index_of(ins.operand(1)?, 0)?.base;
        let (id, t) = self.texture(slot)?;
        let mask = self.dest_mask(ins.operand(0)?);
        let count = if t.multi {
            let texture = self.res_of(ins.operand(1)?, id, D3dClass::ShaderResource)?;
            self.b.value(Type::U32, Op::Query(texture, Query::Samples))
        } else {
            self.b.u32(0)
        };
        let zero = self.b.u32(0);
        let all = self.op(Op::Compose(vec![count, zero, zero, zero]))?;
        let result = if ins.control & 1 != 0 {
            all
        } else {
            self.b.value(Type::F32X4, Op::Convert(all, FloatToInt::Saturate))
        };
        let bits = self.bitcast(result, Scalar::U32)?;
        let picked = self.pick(bits, ins.operand(1)?, mask)?;
        self.write(ins.operand(0)?, picked, false)
    }

    /// `resinfo dest, level, t#` (or a typed texture view `u#`): the size at the level, the layers of an array, then
    /// the level count (one for a multisampled texture and for a view); at a level the resource does not have, zero for
    /// all but the level count, as for one not bound. As floats, uints, or floats with the size's lanes as their
    /// reciprocals (`_rcpFloat`: the layers, the level count and the lanes a smaller texture fills with zero are not),
    /// by the instruction's control.
    pub(super) fn resource_info(&mut self, ins: &Instruction) -> Result<()> {
        let slot = self.slot_of(ins, 2, &[Register::Texture, Register::Uav])?;
        // the resource, its dimension, whether it is an array, and whether its size is a level's
        let (texture, dim, arrayed, mipped) = if ins.operand(2)?.register == Register::Uav {
            let Some(&(id, uav)) = self.resources.uavs.get(&slot) else {
                return self.refuse(format!("u{slot} is used but not declared"));
            };
            let Uav::Image { dims, arrayed, .. } = uav else {
                return self.refuse(format!(
                    "u{slot} is a buffer view, whose size resinfo does not give"
                ));
            };
            let dim = match dims {
                1 => naga::ImageDimension::D1,
                2 => naga::ImageDimension::D2,
                _ => naga::ImageDimension::D3,
            };
            (
                self.res_of(ins.operand(2)?, id, D3dClass::UnorderedAccess)?,
                dim,
                arrayed,
                false,
            )
        } else {
            let (id, t) = self.texture(slot)?;
            (
                self.res_of(ins.operand(2)?, id, D3dClass::ShaderResource)?,
                t.dim,
                t.arrayed,
                !t.multi,
            )
        };
        let mask = self.dest_mask(ins.operand(0)?);
        let level = self.read(ins.operand(1)?, 1, Scalar::U32)?;
        let size_lanes = match dim {
            naga::ImageDimension::D1 => 1,
            naga::ImageDimension::D2 | naga::ImageDimension::Cube => 2,
            naga::ImageDimension::D3 => 3,
        };
        let size = self.b.value(
            Type::vector(Scalar::U32, size_lanes),
            Op::Query(texture, Query::Size(mipped.then_some(level))),
        );
        let levels = if mipped {
            self.b.value(Type::U32, Op::Query(texture, Query::Levels))
        } else {
            self.b.u32(1)
        };
        let zero = self.b.u32(0);
        let mut parts = Vec::new();
        for k in 0..size_lanes {
            parts.push(self.lane(size, k)?);
        }
        if arrayed {
            parts.push(self.b.value(Type::U32, Op::Query(texture, Query::Layers)));
        }
        while parts.len() < 3 {
            parts.push(zero);
        }
        parts.truncate(3);
        let xyz = self.op(Op::Compose(parts))?;
        let none = self.b.literal(Type::vector(Scalar::U32, 3), [0; 4]);
        let inside = self.binary(Binary::UCmp(Cmp::Lt), level, levels)?;
        let xyz = self.op(Op::Select {
            cond: inside,
            accept: xyz,
            reject: none,
        })?;
        let all = self.op(Op::Compose(vec![xyz, levels]))?;
        let result = match ins.control & 3 {
            2 => all,
            0 => self.b.value(Type::F32X4, Op::Convert(all, FloatToInt::Saturate)),
            _ => {
                let f = self.b.value(Type::F32X4, Op::Convert(all, FloatToInt::Saturate));
                let one = self.b.f32x4([1.0; 4]);
                let reciprocal = self.binary(Binary::FDiv, one, f)?;
                let sized = std::array::from_fn(|k| k < usize::from(size_lanes));
                let cond = self
                    .b
                    .literal(Type::vector(Scalar::Bool, 4), sized.map(u64::from));
                self.op(Op::Select {
                    cond,
                    accept: reciprocal,
                    reject: f,
                })?
            }
        };
        let bits = self.bitcast(result, Scalar::U32)?;
        let picked = self.pick(bits, ins.operand(2)?, mask)?;
        self.write(ins.operand(0)?, picked, false)
    }
}
