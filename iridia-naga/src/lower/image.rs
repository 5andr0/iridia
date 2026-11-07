//! Textures and storage images: samples, gathers, texel loads and stores, and queries.
//!
//! A texel load or store outside the image (its coordinate past the size of its level, a level, layer or sample it does
//! not have) reads zero and writes nothing, as the IR states; WGSL leaves such an access to the implementation, so each
//! is checked against the image's own size and clamped into it. A sample's array layer is the coordinate's last lane
//! rounded to nearest even and taken at least 0, as Direct3D does (WGSL clamps it to the layers there are).

use alloc::{format, vec::Vec};

use iridia_ir::{
    GatherOffset, Level, Res, Scalar, Type, ValueId,
    interface::{ResourceKind, SampleType, StorageTarget, TextureDim},
};
use naga::{
    BinaryOperator as Bo, Block, Expression as E, Handle, ImageQuery, MathFunction as Mf, RelationalFunction,
    SampleLevel, ScalarKind, Statement, SwizzleComponent,
};

use super::{Lower, component, err, size};
use crate::Error;

/// What a program reads from an image resource: its dimensions, whether it is arrayed or multisampled, whether it is
/// a storage image, and what a read gives.
#[derive(Clone, Copy)]
struct Shape {
    dim: TextureDim,
    arrayed: bool,
    multisampled: bool,
    storage: bool,
    sample: SampleType,
}

impl Shape {
    /// The lanes of a texel address (without the layer).
    fn lanes(self) -> Result<u8, Error> {
        Ok(match self.dim {
            TextureDim::D1 => 1,
            TextureDim::D2 => 2,
            TextureDim::D3 => 3,
            TextureDim::Cube => return Err(err("a texel address of a cube texture")),
        })
    }
    /// The lanes of a sample's coordinate (without the layer).
    fn coordinate_lanes(self) -> u8 {
        match self.dim {
            TextureDim::D1 => 1,
            TextureDim::D2 => 2,
            TextureDim::D3 | TextureDim::Cube => 3,
        }
    }
    /// Whether it has mip levels: a texture that is not multisampled.
    fn levels(self) -> bool {
        !self.storage && !self.multisampled
    }
}

impl Lower<'_> {
    fn shape(&self, r: Res) -> Result<Shape, Error> {
        Ok(match self.resource(r.id)?.kind {
            ResourceKind::Texture {
                dim,
                arrayed,
                multisampled,
                sample,
            } => Shape {
                dim,
                arrayed,
                multisampled,
                storage: false,
                sample,
            },
            ResourceKind::Storage {
                target: StorageTarget::Image { dim, arrayed, sample },
                ..
            } => Shape {
                dim,
                arrayed,
                multisampled: false,
                storage: true,
                sample,
            },
            ref other => return Err(err(format!("an image operation on a {other:?}"))),
        })
    }

    fn image(&mut self, r: Res) -> Result<Handle<E>, Error> {
        self.handle(r)
    }

    fn query(&mut self, image: Handle<E>, query: ImageQuery) -> Handle<E> {
        self.expr(E::ImageQuery { image, query })
    }

    /// Lanes `from..from + n` of a vector (one lane as a scalar, all of them as the vector itself).
    fn lanes_of(&mut self, x: Handle<E>, t: Type, from: u8, n: u8) -> Result<Handle<E>, Error> {
        if from == 0 && n == t.lanes {
            return Ok(x);
        }
        if n == 1 {
            return Ok(self.expr(E::AccessIndex {
                base: x,
                index: u32::from(from),
            }));
        }
        let mut pattern = [SwizzleComponent::X; 4];
        for (k, p) in (0..n).zip(pattern.iter_mut()) {
            *p = component(from.saturating_add(k))?;
        }
        Ok(self.expr(E::Swizzle {
            size: size(n)?,
            vector: x,
            pattern,
        }))
    }

    /// An integer value (scalar or vector, `i32` or `u32`) as `u32` lanes of the same bits.
    fn as_u32(&mut self, v: ValueId) -> Result<(Handle<E>, Type), Error> {
        let t = self.vty(v)?;
        let x = self.get(v)?;
        match t.scalar {
            Scalar::U32 => Ok((x, t)),
            Scalar::I32 => Ok((self.bitcast(x, ScalarKind::Uint), t.with(Scalar::U32))),
            _ => Err(err(format!("an integer address of {t}"))),
        }
    }

    /// `x < bound` lane by lane, and over every lane.
    fn below(&mut self, x: Handle<E>, bound: Handle<E>, lanes: u8) -> Handle<E> {
        let less = self.binary(Bo::Less, x, bound);
        if lanes == 1 {
            less
        } else {
            self.expr(E::Relational {
                fun: RelationalFunction::All,
                argument: less,
            })
        }
    }

    /// `min(x, bound - 1)`, lane by lane: an address clamped into a size (`bound`) the caller checked is not zero where
    /// it uses it.
    fn clamp_below(&mut self, x: Handle<E>, bound: Handle<E>, lanes: u8) -> Result<Handle<E>, Error> {
        let one = self.literal_bits(Scalar::U32, 1)?;
        let one = self.splat_to(one, lanes)?;
        let last = self.binary(Bo::Subtract, bound, one);
        Ok(self.math(Mf::Min, x, &[last]))
    }

    /// A fixed texel offset as an `i32` constant of `n` lanes, made where naga takes it (unemitted).
    fn offset_constant(&mut self, offset: [i8; 3], n: u8) -> Result<Handle<E>, Error> {
        let parts: Vec<Handle<E>> = offset
            .iter()
            .take(usize::from(n))
            .map(|&o| self.literal_bits(Scalar::I32, u64::from(i32::from(o).cast_unsigned())))
            .collect::<Result<_, _>>()?;
        Ok(match parts.as_slice() {
            [one] => *one,
            _ => {
                let ty = self.ty_of(Type::vector(Scalar::I32, n))?;
                self.unemitted(E::Compose {
                    ty,
                    components: parts,
                })
            }
        })
    }

    /// A sample's coordinate and array index: the coordinate lanes, and the layer rounded to nearest even and taken at
    /// least 0 (a NaN layer as 0), as `u32`.
    fn sample_address(
        &mut self,
        shape: Shape,
        coord: ValueId,
    ) -> Result<(Handle<E>, Option<Handle<E>>), Error> {
        let t = self.vty(coord)?;
        let x = self.get(coord)?;
        let n = shape.coordinate_lanes();
        if t.lanes != n.saturating_add(u8::from(shape.arrayed)) {
            return Err(err(format!(
                "a sample coordinate of {t} for a {:?} texture",
                shape.dim
            )));
        }
        let coordinate = self.lanes_of(x, t, 0, n)?;
        let layer = if shape.arrayed {
            let layer = self.lanes_of(x, t, n, 1)?;
            let rounded = self.math(Mf::Round, layer, &[]);
            let zero = self.f32s(Type::F32, 0.0)?;
            let raised = self.min_max(Mf::Max, rounded, zero, Type::F32, iridia_ir::MinMaxNan::MinNum)?;
            Some(self.expr(E::As {
                expr: raised,
                kind: ScalarKind::Uint,
                convert: Some(4),
            }))
        } else {
            None
        };
        Ok((coordinate, layer))
    }

    /// `Op::Sample`: filtered through the sampler, compared with a depth reference where it has one; a fixed texel
    /// offset as naga takes it. A level clamp has no WGSL form.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sample(
        &mut self,
        texture: Res,
        sampler: Res,
        coord: ValueId,
        level: Level,
        compare: Option<ValueId>,
        offset: [i8; 3],
        clamp: Option<ValueId>,
    ) -> Result<Handle<E>, Error> {
        if clamp.is_some() {
            return Err(err("a sample with a level clamp has no WGSL form"));
        }
        let shape = self.shape(texture)?;
        let (coordinate, array_index) = self.sample_address(shape, coord)?;
        let offset = if offset == [0; 3] {
            None
        } else if shape.dim == TextureDim::Cube {
            return Err(err("a texel offset of a cube sample"));
        } else {
            Some(self.offset_constant(offset, shape.coordinate_lanes())?)
        };
        let depth_ref = match compare {
            Some(c) => Some(self.get(c)?),
            None => None,
        };
        // WGSL samples a 1D texture at the derivatives' level only, and a depth texture at none it biases or at a
        // whole level only
        let depth = shape.sample == SampleType::Depth;
        match (shape.dim, level) {
            (TextureDim::D1, Level::Auto) => {}
            (TextureDim::D1, other) => {
                let at = match other {
                    Level::Zero | Level::Exact(_) => "an explicit level",
                    Level::Bias(_) => "a biased level",
                    Level::Grad(..) | Level::Auto => "explicit derivatives",
                };
                return Err(err(format!("a sample of a 1D texture at {at} has no WGSL form")));
            }
            (_, Level::Bias(_)) if depth => {
                return Err(err("a biased sample of a depth texture has no WGSL form"));
            }
            (_, Level::Grad(..)) if depth && depth_ref.is_some() => {
                return Err(err(
                    "a comparison sample at explicit derivatives has no WGSL form",
                ));
            }
            _ => {}
        }
        let level = match (level, depth_ref.is_some()) {
            (Level::Auto, _) => SampleLevel::Auto,
            (Level::Zero, _) => SampleLevel::Zero,
            (Level::Exact(l), false) if depth => SampleLevel::Exact(self.whole_level(l)?),
            (Level::Bias(b), false) => SampleLevel::Bias(self.get(b)?),
            (Level::Exact(l), false) => SampleLevel::Exact(self.get(l)?),
            (Level::Grad(x, y), false) => SampleLevel::Gradient {
                x: self.get(x)?,
                y: self.get(y)?,
            },
            (other, true) => {
                return Err(err(format!(
                    "a comparison sample at level {other:?} has no WGSL form"
                )));
            }
        };
        let image = self.image(texture)?;
        let sampler = self.handle(sampler)?;
        Ok(self.expr(E::ImageSample {
            image,
            sampler,
            gather: None,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref,
            clamp_to_edge: false,
        }))
    }

    /// A depth texture's explicit level, which WGSL takes as an integer: a level the program states as a whole number;
    /// a computed or fractional one (filtered between levels) has no WGSL form.
    fn whole_level(&mut self, l: ValueId) -> Result<Handle<E>, Error> {
        let stated = self.f.value(l).and_then(|d| match d.op {
            iridia_ir::Op::Literal(bits) if d.ty == Type::F32 => {
                u32::try_from(bits[0]).ok().map(f32::from_bits)
            }
            _ => None,
        });
        // a whole level of 0 to 16 (no texture has more)
        match stated.and_then(|level| (0u8..=16).find(|&n| f32::from(n) == level)) {
            Some(n) => self.literal_bits(Scalar::I32, u64::from(n)),
            None => Err(err(
                "a depth sample at a computed or fractional level has no WGSL form",
            )),
        }
    }

    /// `Op::Gather`: lane `component` of the four texels a bilinear sample reads (each one's comparison result for a
    /// depth comparison), in Direct3D's order, which is WGSL's. A computed offset has no WGSL form.
    pub(super) fn gather(
        &mut self,
        texture: Res,
        sampler: Res,
        coord: ValueId,
        lane: u8,
        compare: Option<ValueId>,
        offset: GatherOffset,
    ) -> Result<Handle<E>, Error> {
        let shape = self.shape(texture)?;
        let offset = match offset {
            GatherOffset::Fixed([0, 0, 0]) => None,
            GatherOffset::Fixed(_) if shape.dim == TextureDim::Cube => {
                return Err(err("a texel offset of a cube gather"));
            }
            GatherOffset::Fixed(o) => Some(self.offset_constant(o, shape.coordinate_lanes())?),
            GatherOffset::Dynamic(_) => return Err(err("a gather at a computed offset has no WGSL form")),
        };
        // WGSL gathers a depth texture's one lane
        let depth = shape.sample == SampleType::Depth;
        if depth && lane != 0 {
            return Err(err(format!("a gather of lane {lane} of a depth texture")));
        }
        let (coordinate, array_index) = self.sample_address(shape, coord)?;
        let depth_ref = match compare {
            Some(c) => Some(self.get(c)?),
            None => None,
        };
        let image = self.image(texture)?;
        let sampler = self.handle(sampler)?;
        Ok(self.expr(E::ImageSample {
            image,
            sampler,
            gather: Some(component(lane)?),
            coordinate,
            array_index,
            offset,
            level: SampleLevel::Zero,
            depth_ref,
            clamp_to_edge: false,
        }))
    }

    /// The texel an integer address names, checked against the image: `(coordinate, array index, sample, level)`
    /// clamped into it, and whether every part is inside.
    fn texel_address(
        &mut self,
        texture: Res,
        shape: Shape,
        coord: ValueId,
        level: Option<ValueId>,
        sample: Option<ValueId>,
        offset: [i8; 3],
    ) -> Result<TexelAddress, Error> {
        let n = shape.lanes()?;
        let image = self.image(texture)?;
        let (mut c, ct) = self.as_u32(coord)?;
        if ct.lanes != n.saturating_add(u8::from(shape.arrayed)) {
            return Err(err(format!(
                "a texel address of {ct} for a {:?} image",
                shape.dim
            )));
        }
        if offset != [0; 3] {
            // the offset moves the texel before it is checked: added in wrapping arithmetic, as `u32` lanes
            let parts: Vec<Handle<E>> = offset
                .iter()
                .take(usize::from(n))
                .chain(core::iter::repeat_n(&0, usize::from(u8::from(shape.arrayed))))
                .map(|&o| self.literal_bits(Scalar::U32, u64::from(i32::from(o).cast_unsigned())))
                .collect::<Result<_, _>>()?;
            let moved = match parts.as_slice() {
                [one] => *one,
                _ => {
                    let ty = self.ty_of(ct)?;
                    self.expr(E::Compose {
                        ty,
                        components: parts,
                    })
                }
            };
            c = self.binary(Bo::Add, c, moved);
        }
        let texel = self.lanes_of(c, ct, 0, n)?;
        let mut inside = Vec::new();
        let level = match (shape.levels(), level) {
            (true, Some(l)) => {
                let (l, _) = self.as_u32(l)?;
                let levels = self.query(image, ImageQuery::NumLevels);
                inside.push(self.below(l, levels, 1));
                Some(self.clamp_below(l, levels, 1)?)
            }
            (true, None) => Some(self.literal_bits(Scalar::U32, 0)?),
            (false, _) => None,
        };
        let dims = self.query(image, ImageQuery::Size { level });
        inside.push(self.below(texel, dims, n));
        let coordinate = self.clamp_below(texel, dims, n)?;
        let array_index = if shape.arrayed {
            let layer = self.lanes_of(c, ct, n, 1)?;
            let layers = self.query(image, ImageQuery::NumLayers);
            inside.push(self.below(layer, layers, 1));
            Some(self.clamp_below(layer, layers, 1)?)
        } else {
            None
        };
        let sample = match (shape.multisampled, sample) {
            (true, Some(s)) => {
                let (s, _) = self.as_u32(s)?;
                let samples = self.query(image, ImageQuery::NumSamples);
                inside.push(self.below(s, samples, 1));
                Some(self.clamp_below(s, samples, 1)?)
            }
            (true, None) => return Err(err("a load of a multisampled texture without its sample")),
            (false, _) => None,
        };
        let mut all = inside
            .first()
            .copied()
            .ok_or_else(|| err("a texel address with no parts"))?;
        for &part in inside.iter().skip(1) {
            all = self.binary(Bo::LogicalAnd, all, part);
        }
        Ok(TexelAddress {
            image,
            coordinate,
            array_index,
            sample,
            level,
            inside: all,
        })
    }

    /// `Op::LoadTexel` of a texture or storage image: the texel, zero outside the image.
    pub(super) fn load_texel(
        &mut self,
        texture: Res,
        coord: ValueId,
        level: Option<ValueId>,
        sample: Option<ValueId>,
        offset: [i8; 3],
        t: Type,
    ) -> Result<Handle<E>, Error> {
        let shape = self.shape(texture)?;
        let a = self.texel_address(texture, shape, coord, level, sample, offset)?;
        let texel = self.expr(E::ImageLoad {
            image: a.image,
            coordinate: a.coordinate,
            array_index: a.array_index,
            sample: a.sample,
            level: a.level,
        });
        let ty = self.ty_of(t)?;
        let zero = self.expr(E::ZeroValue(ty));
        Ok(self.select(a.inside, texel, zero))
    }

    /// `Node::StoreTexel` of a storage image: the texel takes the value where it is inside the image.
    pub(super) fn store_texel(&mut self, image: Res, coord: ValueId, value: ValueId) -> Result<(), Error> {
        let shape = self.shape(image)?;
        if !shape.storage {
            return Err(err("a texel store of a texture that is not a storage image"));
        }
        let a = self.texel_address(image, shape, coord, None, None, [0; 3])?;
        let value = self.get(value)?;
        self.open();
        self.statement(Statement::ImageStore {
            image: a.image,
            coordinate: a.coordinate,
            array_index: a.array_index,
            value,
        })?;
        let accept = self.close()?;
        self.statement(Statement::If {
            condition: a.inside,
            accept,
            reject: Block::new(),
        })
    }

    /// `Op::Query` of a texture or storage image: its size at a level (zero in every lane at a level it does not
    /// have), its layers, levels or samples.
    pub(super) fn image_query(&mut self, r: Res, q: iridia_ir::Query, t: Type) -> Result<Handle<E>, Error> {
        let shape = self.shape(r)?;
        let image = self.image(r)?;
        Ok(match q {
            iridia_ir::Query::Size(level) => {
                let lanes = match shape.dim {
                    TextureDim::D1 => 1,
                    TextureDim::D2 | TextureDim::Cube => 2,
                    TextureDim::D3 => 3,
                };
                if t != Type::vector(Scalar::U32, lanes) && !(lanes == 1 && t == Type::U32) {
                    return Err(err(format!("a size query of {t} for a {:?} image", shape.dim)));
                }
                match (shape.levels(), level) {
                    (true, Some(l)) => {
                        let (l, _) = self.as_u32(l)?;
                        let levels = self.query(image, ImageQuery::NumLevels);
                        let inside = self.below(l, levels, 1);
                        let clamped = self.clamp_below(l, levels, 1)?;
                        let dims = self.query(image, ImageQuery::Size { level: Some(clamped) });
                        let ty = self.ty_of(t)?;
                        let zero = self.expr(E::ZeroValue(ty));
                        self.select(inside, dims, zero)
                    }
                    _ => self.query(image, ImageQuery::Size { level: None }),
                }
            }
            iridia_ir::Query::Layers if shape.arrayed => self.query(image, ImageQuery::NumLayers),
            iridia_ir::Query::Levels if shape.levels() => self.query(image, ImageQuery::NumLevels),
            iridia_ir::Query::Samples if shape.multisampled => self.query(image, ImageQuery::NumSamples),
            other => return Err(err(format!("query {other:?} of a {:?} image", shape.dim))),
        })
    }
}

/// [`Lower::texel_address`]'s parts, clamped into the image, and whether the address is inside it.
struct TexelAddress {
    image: Handle<E>,
    coordinate: Handle<E>,
    array_index: Option<Handle<E>>,
    sample: Option<Handle<E>>,
    level: Option<Handle<E>>,
    inside: Handle<E>,
}
