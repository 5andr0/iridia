//! Textures and storage images: samples, gathers, texel loads and stores, and queries.
//!
//! A texel load or store outside the image (its coordinate past the size of its level, a level, layer or sample it does not
//! have) reads zero and writes nothing, as the IR states; WGSL leaves such an access to the implementation, so each is
//! checked against the image's own size and clamped into it. A sample's array layer is the coordinate's last lane rounded to
//! nearest even and taken at least 0, as Direct3D does (WGSL clamps it to the layers there are).

use alloc::{format, string::String, vec::Vec};

use iridia_ir::interface::{ResourceKind, SampleType, StorageTarget, TextureDim};
use iridia_ir::{AtomicOp, GatherOffset, Level, Op, Res, Scalar, Type, ValueId};

use crate::emit::{R, Writer, all_of};
use crate::text::{self, err};

/// What a program reads from an image resource: its dimensions, whether it is arrayed or multisampled, whether it is a
/// storage image, and what a read gives.
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
    fn lanes(self) -> R<u8> {
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

/// A texel address checked against the image: its parts clamped into it, and whether every part is inside.
struct TexelAddress {
    image: String,
    coordinate: String,
    array_index: Option<String>,
    sample: Option<String>,
    level: Option<String>,
    inside: String,
}

impl Writer<'_> {
    fn shape(&self, r: Res) -> R<Shape> {
        Ok(match *self.kind(r.id)? {
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

    fn image(&self, r: Res) -> R<String> {
        self.pick(r)
    }

    /// Lanes `from..from + n` of a vector (one lane as a scalar, all of them as the vector itself).
    fn lanes_of(&self, x: &str, t: Type, from: u8, n: u8) -> R<String> {
        if from == 0 && n == t.lanes {
            return Ok(x.into());
        }
        Ok(format!("{x}.{}", text::swizzle(from..from.saturating_add(n))?))
    }

    /// An integer value (scalar or vector, `i32` or `u32`) as `u32` lanes of the same bits.
    fn as_u32(&self, v: ValueId) -> R<(String, Type)> {
        let t = self.vty(v)?;
        let x = self.get(v)?;
        match t.scalar {
            Scalar::U32 => Ok((x, t)),
            Scalar::I32 => Ok((
                format!("bitcast<{}>({x})", text::ty(t.with(Scalar::U32))?),
                t.with(Scalar::U32),
            )),
            _ => Err(err(format!("an integer address of {t}"))),
        }
    }

    /// `x < bound` lane by lane, over every lane.
    fn below(&self, x: &str, bound: &str, lanes: u8) -> String {
        all_of(&format!("({x} < {bound})"), lanes)
    }

    /// `min(x, bound - 1)`, lane by lane: an address clamped into a size the caller checked is not zero where it uses it.
    fn clamp_below(&self, x: &str, bound: &str, lanes: u8) -> R<String> {
        let one = self.u32s(Type::vector(Scalar::U32, lanes), 1)?;
        Ok(format!("min({x}, ({bound} - {one}))"))
    }

    /// A fixed texel offset as an `i32` constant of `n` lanes.
    fn offset_constant(&self, offset: [i8; 3], n: u8) -> R<String> {
        let parts: Vec<String> = offset
            .iter()
            .take(usize::from(n))
            .map(|&o| text::literal(Scalar::I32, u64::from(i32::from(o).cast_unsigned())))
            .collect::<Result<_, _>>()?;
        Ok(match parts.as_slice() {
            [one] => one.clone(),
            _ => format!("vec{n}<i32>({})", parts.join(", ")),
        })
    }

    /// A sample's coordinate and array index: the coordinate lanes, and the layer rounded to nearest even and taken at least
    /// 0 (a NaN layer as 0), as `u32`.
    fn sample_address(&mut self, shape: Shape, coord: ValueId) -> R<(String, Option<String>)> {
        let t = self.vty(coord)?;
        let x = self.get(coord)?;
        let n = shape.coordinate_lanes();
        if t.lanes != n.saturating_add(u8::from(shape.arrayed)) {
            return Err(err(format!(
                "a sample coordinate of {t} for a {:?} texture",
                shape.dim
            )));
        }
        let coordinate = self.lanes_of(&x, t, 0, n)?;
        let layer = if shape.arrayed {
            let layer = self.lanes_of(&x, t, n, 1)?;
            let rounded = self.tmp(&format!("round({layer})"));
            let nan = self.is_nan(&rounded, Type::F32)?;
            Some(format!("u32(select(max({rounded}, 0.0f), 0.0f, {nan}))"))
        } else {
            None
        };
        Ok((coordinate, layer))
    }

    /// The arguments of a texture call after the sampler: the coordinate, the array layer.
    fn call(
        fun: &str,
        leading: &[String],
        coordinate: &str,
        array_index: &Option<String>,
        rest: &[String],
    ) -> String {
        let mut args: Vec<String> = leading.to_vec();
        args.push(coordinate.into());
        args.extend(array_index.clone());
        args.extend(rest.iter().cloned());
        format!("{fun}({})", args.join(", "))
    }

    /// `Op::Sample`: filtered through the sampler, compared with a depth reference where it has one; a fixed texel offset as
    /// WGSL takes it. A level clamp has no WGSL form.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample(
        &mut self,
        texture: Res,
        sampler: Res,
        coord: ValueId,
        level: Level,
        compare: Option<ValueId>,
        offset: [i8; 3],
        clamp: Option<ValueId>,
    ) -> R<String> {
        if clamp.is_some() {
            return Err(err("a sample with a level clamp has no WGSL form"));
        }
        let shape = self.shape(texture)?;
        if matches!(shape.sample, SampleType::Sint | SampleType::Uint) {
            return Err(err("a filtered sample of an integer texture has no WGSL form"));
        }
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
        // WGSL samples a 1D texture at the derivatives' level only, and a depth texture at none it biases or at a whole
        // level only
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
        let image = self.image(texture)?;
        let sampler_name = self.pick(sampler)?;
        let leading = [image, sampler_name];
        let mut extra: Vec<String> = Vec::new();
        let fun = match (level, depth_ref) {
            (Level::Auto, None) => "textureSample",
            (Level::Auto, Some(r)) => {
                extra.push(r);
                "textureSampleCompare"
            }
            (Level::Zero, Some(r)) => {
                extra.push(r);
                "textureSampleCompareLevel"
            }
            (Level::Zero, None) => {
                extra.push(if depth { "0i".into() } else { "0.0f".into() });
                "textureSampleLevel"
            }
            (Level::Exact(l), None) if depth => {
                extra.push(self.whole_level(l)?);
                "textureSampleLevel"
            }
            (Level::Bias(b), None) => {
                extra.push(self.get(b)?);
                "textureSampleBias"
            }
            (Level::Exact(l), None) => {
                extra.push(self.get(l)?);
                "textureSampleLevel"
            }
            (Level::Grad(x, y), None) => {
                extra.push(self.get(x)?);
                extra.push(self.get(y)?);
                "textureSampleGrad"
            }
            (other, Some(_)) => {
                return Err(err(format!(
                    "a comparison sample at level {other:?} has no WGSL form"
                )));
            }
        };
        extra.extend(offset);
        Ok(Self::call(fun, &leading, &coordinate, &array_index, &extra))
    }

    /// A depth texture's explicit level, which WGSL takes as an integer: a level the program states as a whole number; a
    /// computed or fractional one (filtered between levels) has no WGSL form.
    fn whole_level(&self, l: ValueId) -> R<String> {
        let stated = self.f.value(l).and_then(|d| match d.op {
            Op::Literal(bits) if d.ty == Type::F32 => u32::try_from(bits[0]).ok().map(f32::from_bits),
            _ => None,
        });
        // a whole level of 0 to 16 (no texture has more)
        match stated.and_then(|level| (0u8..=16).find(|&n| f32::from(n) == level)) {
            Some(n) => text::literal(Scalar::I32, u64::from(n)),
            None => Err(err(
                "a depth sample at a computed or fractional level has no WGSL form",
            )),
        }
    }

    /// `Op::Gather`: lane `component` of the four texels a bilinear sample reads (each one's comparison result for a depth
    /// comparison), in Direct3D's order, which is WGSL's. A computed offset has no WGSL form.
    pub(crate) fn gather(
        &mut self,
        texture: Res,
        sampler: Res,
        coord: ValueId,
        lane: u8,
        compare: Option<ValueId>,
        offset: GatherOffset,
    ) -> R<String> {
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
        let image = self.image(texture)?;
        let sampler_name = self.pick(sampler)?;
        let mut extra: Vec<String> = Vec::new();
        let (fun, leading) = match (depth, compare) {
            (true, Some(c)) => {
                extra.push(self.get(c)?);
                ("textureGatherCompare", alloc::vec![image, sampler_name])
            }
            (true, None) => ("textureGather", alloc::vec![image, sampler_name]),
            (false, None) => (
                "textureGather",
                alloc::vec![format!("{lane}i"), image, sampler_name],
            ),
            (false, Some(_)) => {
                return Err(err(
                    "a comparison gather of a texture that is not a depth texture",
                ));
            }
        };
        extra.extend(offset);
        Ok(Self::call(fun, &leading, &coordinate, &array_index, &extra))
    }

    /// The texel an integer address names, checked against the image: its coordinate, array index, sample and level clamped
    /// into it, and whether every part is inside.
    fn texel_address(
        &mut self,
        texture: Res,
        shape: Shape,
        coord: ValueId,
        level: Option<ValueId>,
        sample: Option<ValueId>,
        offset: [i8; 3],
    ) -> R<TexelAddress> {
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
            let parts: Vec<String> = offset
                .iter()
                .take(usize::from(n))
                .chain(core::iter::repeat_n(&0, usize::from(u8::from(shape.arrayed))))
                .map(|&o| text::literal(Scalar::U32, u64::from(i32::from(o).cast_unsigned())))
                .collect::<Result<_, _>>()?;
            let moved = match parts.as_slice() {
                [one] => one.clone(),
                _ => format!("{}({})", text::ty(ct)?, parts.join(", ")),
            };
            c = self.tmp(&format!("({c} + {moved})"));
        }
        let texel = self.lanes_of(&c, ct, 0, n)?;
        let mut inside = Vec::new();
        let level = match (shape.levels(), level) {
            (true, Some(l)) => {
                let (l, _) = self.as_u32(l)?;
                let levels = self.tmp(&format!("textureNumLevels({image})"));
                inside.push(self.below(&l, &levels, 1));
                Some(self.tmp(&self.clamp_below(&l, &levels, 1)?))
            }
            (true, None) => Some("0u".into()),
            (false, _) => None,
        };
        let dims = match &level {
            Some(l) => self.tmp(&format!("textureDimensions({image}, {l})")),
            None => self.tmp(&format!("textureDimensions({image})")),
        };
        inside.push(self.below(&texel, &dims, n));
        let coordinate = self.tmp(&self.clamp_below(&texel, &dims, n)?);
        let array_index = if shape.arrayed {
            let layer = self.lanes_of(&c, ct, n, 1)?;
            let layers = self.tmp(&format!("textureNumLayers({image})"));
            inside.push(self.below(&layer, &layers, 1));
            Some(self.tmp(&self.clamp_below(&layer, &layers, 1)?))
        } else {
            None
        };
        let sample = match (shape.multisampled, sample) {
            (true, Some(s)) => {
                let (s, _) = self.as_u32(s)?;
                let samples = self.tmp(&format!("textureNumSamples({image})"));
                inside.push(self.below(&s, &samples, 1));
                Some(self.tmp(&self.clamp_below(&s, &samples, 1)?))
            }
            (true, None) => return Err(err("a load of a multisampled texture without its sample")),
            (false, _) => None,
        };
        let all = inside.join(" && ");
        let inside = self.tmp(&all);
        Ok(TexelAddress {
            image,
            coordinate,
            array_index,
            sample,
            level,
            inside,
        })
    }

    /// `Op::LoadTexel` of a texture or storage image: the texel, zero outside the image.
    pub(crate) fn load_texel(
        &mut self,
        texture: Res,
        coord: ValueId,
        level: Option<ValueId>,
        sample: Option<ValueId>,
        offset: [i8; 3],
        t: Type,
    ) -> R<String> {
        let shape = self.shape(texture)?;
        let a = self.texel_address(texture, shape, coord, level, sample, offset)?;
        // sampled: the level (or, multisampled, the sample); storage: neither
        let mut args = alloc::vec![a.image.clone(), a.coordinate.clone()];
        args.extend(a.array_index.clone());
        args.extend(a.sample.clone());
        if !shape.multisampled {
            args.extend(a.level.clone());
        }
        let texel = self.tmp(&format!("textureLoad({})", args.join(", ")));
        Ok(format!("select({}(), {texel}, {})", self.ty(t)?, a.inside))
    }

    /// `Node::StoreTexel` of a storage image: the texel takes the value where it is inside the image.
    pub(crate) fn store_texel(&mut self, image: Res, coord: ValueId, value: ValueId) -> R<()> {
        let shape = self.shape(image)?;
        if !shape.storage {
            return Err(err("a texel store of a texture that is not a storage image"));
        }
        let a = self.texel_address(image, shape, coord, None, None, [0; 3])?;
        let value = self.get(value)?;
        let mut args = alloc::vec![a.image.clone(), a.coordinate.clone()];
        args.extend(a.array_index.clone());
        args.push(value);
        self.line(&format!(
            "if ({}) {{ textureStore({}); }}",
            a.inside,
            args.join(", ")
        ));
        Ok(())
    }

    /// The position of sample `index` of a multisampled texture in Direct3D's standard pattern for its sample count, from
    /// [`Writer::sample_positions`]; (0, 0) for an index past its samples or a count with no standard pattern (one other than
    /// 1, 2, 4, 8 or 16).
    fn sample_position(&mut self, image: &str, index: ValueId) -> R<String> {
        let table = self.sample_positions()?;
        let (index, _) = self.as_u32(index)?;
        let count = self.tmp(&format!("textureNumSamples({image})"));
        // a standard count is a power of two of at most 16 (a count of 0 has no sample an index is below)
        let last = self.tmp(&format!("({count} - 1u)"));
        let low = self.tmp(&format!("({count} & {last})"));
        let inside = self.tmp(&format!(
            "(({low} == 0u) && ({count} <= 16u) && ({index} < {count}))"
        ));
        // the count's rows start at row `count - 1`; clamped into the table where it is not inside
        let row = self.tmp(&format!(
            "min(({last} + {index}), {}u)",
            crate::module::STANDARD_ROWS - 1
        ));
        Ok(format!("select(vec2<f32>(), {table}[{row}], {inside})"))
    }

    /// `Op::Atomic` of a storage image's texel whose old value nothing reads: naga's dialect's `textureAtomic*`, where the texel
    /// is inside the image (outside it the IR's changes nothing). It gives no old value and takes no exchange: one is refused
    /// by name. A subtraction is an addition of the negation (the same bits); a minimum or maximum is of the image's texels'
    /// signedness, so one of the other signedness is refused by name.
    pub(crate) fn image_atomic(&mut self, r: Res, coord: ValueId, op: AtomicOp, value: ValueId) -> R<()> {
        let shape = self.shape(r)?;
        if !shape.storage {
            return Err(err("an atomic of a texture that is not a storage image"));
        }
        let signed = shape.sample == SampleType::Sint;
        let fun = match op {
            AtomicOp::Add | AtomicOp::Sub => "textureAtomicAdd",
            AtomicOp::And => "textureAtomicAnd",
            AtomicOp::Or => "textureAtomicOr",
            AtomicOp::Xor => "textureAtomicXor",
            AtomicOp::SMin if signed => "textureAtomicMin",
            AtomicOp::SMax if signed => "textureAtomicMax",
            AtomicOp::UMin if !signed => "textureAtomicMin",
            AtomicOp::UMax if !signed => "textureAtomicMax",
            AtomicOp::Exchange | AtomicOp::CompareExchange => {
                return Err(err(
                    "an exchange of a texel has no naga form: naga's image atomic takes no exchange",
                ));
            }
            other => {
                return Err(err(format!(
                    "a {other:?} of a storage image of {:?} texels has no naga form: an image's minimum and maximum are of \
                     its texels' signedness",
                    shape.sample
                )));
            }
        };
        let a = self.texel_address(r, shape, coord, None, None, [0; 3])?;
        let word = if signed { Scalar::I32 } else { Scalar::U32 };
        let x = self.bitcast(&self.get(value)?, self.vty(value)?, word)?;
        let x = if op == AtomicOp::Sub {
            format!("({} - {x})", text::literal(word, 0)?)
        } else {
            x
        };
        let mut args = alloc::vec![a.image.clone(), a.coordinate.clone()];
        args.extend(a.array_index.clone());
        args.push(x);
        self.line(&format!("if ({}) {{ {fun}({}); }}", a.inside, args.join(", ")));
        Ok(())
    }

    /// `Op::Query` of a texture or storage image: its size at a level (zero in every lane at a level it does not have), its
    /// layers, levels or samples.
    pub(crate) fn image_query(&mut self, r: Res, q: iridia_ir::Query, t: Type) -> R<String> {
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
                        let levels = self.tmp(&format!("textureNumLevels({image})"));
                        let inside = self.below(&l, &levels, 1);
                        let clamped = self.clamp_below(&l, &levels, 1)?;
                        format!(
                            "select({}(), textureDimensions({image}, {clamped}), {inside})",
                            self.ty(t)?
                        )
                    }
                    _ => format!("textureDimensions({image})"),
                }
            }
            iridia_ir::Query::Layers if shape.arrayed => format!("textureNumLayers({image})"),
            iridia_ir::Query::Levels if shape.levels() => format!("textureNumLevels({image})"),
            iridia_ir::Query::Samples if shape.multisampled => format!("textureNumSamples({image})"),
            iridia_ir::Query::SamplePosition(index) if shape.multisampled => {
                self.sample_position(&image, index)?
            }
            other => return Err(err(format!("query {other:?} of a {:?} image", shape.dim))),
        })
    }
}
