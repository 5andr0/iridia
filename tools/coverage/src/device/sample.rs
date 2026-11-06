//! Texture sampling as Direct3D 11 and 12 define it (the D3D11.3 functional specification, section 7.18), over a
//! [`Resource`]: 1D, 2D, 3D and cube textures and arrays of them, every address mode, point and linear minification,
//! magnification and mip filters, comparison filtering, the minimum and maximum reductions, gathers, texel offsets, the
//! sampler's and the program's level-of-detail bias and the view's least level.
//!
//! - A coordinate is scaled by the level's size: point filtering reads the texel it falls in (`floor(u * W)`), linear
//!   filtering the texels about `u * W - 0.5`, weighted by its fraction truncated to 8 bits (7.18.16.1:
//!   `D3D11_SUBTEXEL_FRACTIONAL_BIT_COUNT`; Microsoft's WARP truncates, as `primase_foundations::sampling::Subtexel`
//!   measured).
//! - The level of detail is the base-2 logarithm of the longer of the footprint's two axes in texels (7.18.11), plus
//!   the program's bias (clamped to -16..15.99) and the sampler's, clamped to the sampler's range and to the view's
//!   least level. At or below 0 the magnification filter applies, above it the minification filter. A point mip
//!   filter takes level `floor(lod + 0.5)`, a linear one blends the two levels about it.
//! - A cube is sampled on the face of the coordinate's major axis (7.18.6, Table 7.18.6.1). A linear tap past a face's
//!   edge reads the texel of the face its direction falls on, as a seamless cube does (7.18.6.4); a tap past a corner
//!   reads the mean of the corner's two other texels.
//! - A comparison compares each texel's first lane with the reference (a texel that passes is 1, one that fails 0)
//!   before filtering (7.18.13); a minimum or maximum reduction takes the least or greatest of the texels a linear
//!   filter weighs (not 0).
use super::resource::{Dim, Resource};
use crate::vkd3d::cases::SamplerDesc;

/// An axis's address mode (7.18.9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Address {
    /// Repeat.
    Wrap,
    /// Repeat, every other copy mirrored.
    Mirror,
    /// The edge texel.
    Clamp,
    /// The border colour.
    Border,
    /// Mirrored once about the origin, then the edge texel.
    MirrorOnce,
}

/// How a sampler reduces the texels it filters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    /// A weighted mean.
    Standard,
    /// Each texel compared with a reference first.
    Comparison(Compare),
    /// The least texel the filter weighs.
    Minimum,
    /// The greatest.
    Maximum,
}

/// A comparison function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    /// Never passes.
    Never,
    /// `ref < texel`.
    Less,
    /// `ref == texel`.
    Equal,
    /// `ref <= texel`.
    LessEqual,
    /// `ref > texel`.
    Greater,
    /// `ref != texel`.
    NotEqual,
    /// `ref >= texel`.
    GreaterEqual,
    /// Always passes.
    Always,
}

impl Compare {
    /// The function vkd3d names `name`.
    pub fn named(name: &str) -> Result<Compare, String> {
        Ok(match name {
            "never" => Compare::Never,
            "less" => Compare::Less,
            "equal" => Compare::Equal,
            "less equal" => Compare::LessEqual,
            "greater" => Compare::Greater,
            "not equal" => Compare::NotEqual,
            "greater equal" => Compare::GreaterEqual,
            "always" => Compare::Always,
            _ => return Err(format!("the comparison {name}")),
        })
    }

    /// Whether `a` compared with `b` passes.
    pub fn passes(self, a: f32, b: f32) -> bool {
        match self {
            Compare::Never => false,
            Compare::Less => a < b,
            Compare::Equal => a == b,
            Compare::LessEqual => a <= b,
            Compare::Greater => a > b,
            Compare::NotEqual => a != b,
            Compare::GreaterEqual => a >= b,
            Compare::Always => true,
        }
    }
}

/// A sampler's state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampler {
    /// Minification filter is linear.
    pub min_linear: bool,
    /// Magnification filter is linear.
    pub mag_linear: bool,
    /// Mip filter is linear.
    pub mip_linear: bool,
    /// The reduction.
    pub reduction: Reduction,
    /// The address modes along u, v and w.
    pub address: [Address; 3],
    /// The border colour (vkd3d's runner leaves it transparent black).
    pub border: [f32; 4],
    /// The sampler's level-of-detail bias.
    pub bias: f32,
    /// The least and greatest level of detail.
    pub lod_range: [f32; 2],
}

impl Sampler {
    /// The state vkd3d's runner gives a sampler it declares (`D3D12_STATIC_SAMPLER_DESC` with the runner's fields; the
    /// rest as `shader_runner_d3d12.c` leaves them: no bias, levels 0 to `D3D12_FLOAT32_MAX`, a transparent black
    /// border).
    pub fn of(desc: &SamplerDesc) -> Result<Sampler, String> {
        let words: Vec<&str> = desc.filter.split_whitespace().collect();
        let [min, mag, mip] = words.as_slice() else {
            return Err(format!("the filter {}", desc.filter));
        };
        let linear = |w: &str| match w {
            "linear" => Ok(true),
            "point" => Ok(false),
            _ => Err(format!("the filter {w}")),
        };
        let address = |a: &str| {
            Ok(match a {
                "wrap" => Address::Wrap,
                "mirror" => Address::Mirror,
                "clamp" => Address::Clamp,
                "border" => Address::Border,
                "mirror_once" => Address::MirrorOnce,
                _ => return Err(format!("the address mode {a}")),
            })
        };
        let reduction = match (desc.reduction.as_deref(), &desc.comparison) {
            (_, Some(c)) => Reduction::Comparison(Compare::named(c)?),
            (Some("minimum"), None) => Reduction::Minimum,
            (Some("maximum"), None) => Reduction::Maximum,
            (Some("standard") | None, None) => Reduction::Standard,
            (Some(r), None) => return Err(format!("the reduction {r}")),
        };
        Ok(Sampler {
            min_linear: linear(min)?,
            mag_linear: linear(mag)?,
            mip_linear: linear(mip)?,
            reduction,
            address: [
                address(&desc.address[0])?,
                address(&desc.address[1])?,
                address(&desc.address[2])?,
            ],
            border: [0.0; 4],
            bias: 0.0,
            lod_range: [0.0, f32::MAX],
        })
    }
}

/// How a sample picks its level of detail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lod {
    /// From the coordinate's derivatives along x and y (the axes the dimension has), plus the program's bias.
    Gradients([[f32; 3]; 2], f32),
    /// The level the program states.
    Level(f32),
    /// No footprint is known (one invocation alone, a compute program): the program's bias on level 0 (D3D gives a
    /// compute program's implicit-level sample no footprint; vkd3d's tests read level 0 there).
    None(f32),
}

/// A sample request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Request {
    /// The coordinate (1D: u; 2D: u, v; 3D and cube: u, v, w).
    pub coord: [f32; 3],
    /// An array's layer, as the program gives it.
    pub layer: Option<f32>,
    /// The level of detail.
    pub lod: Lod,
    /// The texel offset.
    pub offset: [i32; 3],
    /// A gather of this component.
    pub gather: Option<u8>,
    /// A comparison's reference value.
    pub reference: Option<f32>,
}

/// The texel index `i` reads along an axis of `n` texels, or `None` at a border.
fn address(mode: Address, i: i64, n: u32) -> Option<i64> {
    let n = i64::from(n.max(1));
    let last = n - 1;
    Some(match mode {
        Address::Wrap => i.rem_euclid(n),
        Address::Clamp => i.clamp(0, last),
        Address::Mirror => {
            let m = i.rem_euclid(2 * n);
            if m < n { m } else { 2 * n - 1 - m }
        }
        Address::MirrorOnce => (if i < 0 { -1 - i } else { i }).clamp(0, last),
        Address::Border => {
            if (0..n).contains(&i) {
                i
            } else {
                return None;
            }
        }
    })
}

/// A coordinate's fraction truncated to 8 bits.
fn subtexel(x: f64) -> f64 {
    (x * 256.0).floor() / 256.0
}

/// The axes a texture's coordinate has: 1, 2 or 3.
fn axes(r: &Resource, one_d: bool) -> usize {
    match r.dim {
        Dim::D3 | Dim::Cube => 3,
        _ if one_d => 1,
        _ => 2,
    }
}

/// The face of a cube a direction falls on, and the face's coordinate in 0..1 (7.18.6, Table 7.18.6.1).
fn cube_face(d: [f32; 3]) -> (u32, [f32; 2]) {
    let [x, y, z] = d;
    let (ax, ay, az) = (x.abs(), y.abs(), z.abs());
    // the major axis: x where |x| is the greatest, then y, then z (ties to z, then y, as hardware picks)
    let (face, sc, tc, ma) = if ax >= ay && ax >= az {
        if x >= 0.0 { (0, -z, -y, ax) } else { (1, z, -y, ax) }
    } else if ay >= az {
        if y >= 0.0 { (2, x, z, ay) } else { (3, x, -z, ay) }
    } else if z >= 0.0 {
        (4, x, -y, az)
    } else {
        (5, -x, -y, az)
    };
    let ma = if ma == 0.0 { 1.0 } else { ma };
    (face, [(sc / ma + 1.0) * 0.5, (tc / ma + 1.0) * 0.5])
}

/// The direction of the centre of texel (i, j) of `face` of a cube level `n` texels wide (a texel past the face's edge
/// has a direction past it).
fn cube_direction(face: u32, i: i64, j: i64, n: u32) -> [f32; 3] {
    let n = n as f64;
    let s = ((i as f64 + 0.5) / n * 2.0 - 1.0) as f32;
    let t = ((j as f64 + 0.5) / n * 2.0 - 1.0) as f32;
    match face {
        0 => [1.0, -t, -s],
        1 => [-1.0, -t, s],
        2 => [s, 1.0, t],
        3 => [s, -1.0, -t],
        4 => [s, -t, 1.0],
        _ => [-s, -t, -1.0],
    }
}

/// What a texture read gives: a texel's four lanes as floats, or none (the border).
type Texel = Option<[f32; 4]>;

/// One level of a texture as a sampler reads it.
struct Level<'a> {
    r: &'a Resource,
    level: u32,
    layer: u32,
    s: &'a Sampler,
    reference: Option<f32>,
}

impl Level<'_> {
    fn dims(&self) -> [u32; 3] {
        self.r.level_dims(self.level)
    }

    /// The texel at a (wrapped) index, compared where the sample compares; `None` at a border.
    fn texel(&self, at: [i64; 3]) -> Texel {
        let dims = self.dims();
        let mut idx = [0i64; 3];
        let n = if self.r.dim == Dim::Cube { 2 } else { 3 };
        for k in 0..n {
            idx[k] = address(self.s.address[k], at[k], dims[k])?;
        }
        let lanes = self.r.mapped(self.r.texel(self.level, self.layer, idx)?);
        Some(self.compared(lanes.map(f32::from_bits)))
    }

    fn compared(&self, t: [f32; 4]) -> [f32; 4] {
        match (self.reference, self.s.reduction) {
            (Some(reference), Reduction::Comparison(c)) => {
                let pass = if c.passes(reference, t[0]) { 1.0 } else { 0.0 };
                [pass, pass, pass, pass]
            }
            _ => t,
        }
    }

    /// The texel of a cube's face at a (possibly past its edge) index: across an edge, the texel the direction of
    /// its centre falls on; across a corner, the mean of the corner's other two texels.
    fn cube_texel(&self, face: u32, i: i64, j: i64) -> Texel {
        let [n, _, _] = self.dims();
        let n64 = i64::from(n);
        let out_i = !(0..n64).contains(&i);
        let out_j = !(0..n64).contains(&j);
        let read = |face: u32, i: i64, j: i64| -> Texel {
            let base = self.layer - self.layer % 6;
            let lanes = self.r.mapped(self.r.texel(self.level, base + face, [i, j, 0])?);
            Some(self.compared(lanes.map(f32::from_bits)))
        };
        if !out_i && !out_j {
            return read(face, i, j);
        }
        let across = |i: i64, j: i64| -> Texel {
            let (f, uv) = cube_face(cube_direction(face, i, j, n));
            let x = ((f64::from(uv[0]) * f64::from(n)).floor() as i64).clamp(0, n64 - 1);
            let y = ((f64::from(uv[1]) * f64::from(n)).floor() as i64).clamp(0, n64 - 1);
            read(f, x, y)
        };
        if out_i && out_j {
            // a corner: the two texels next to it on the faces that meet there
            let a = across(i, j.clamp(0, n64 - 1))?;
            let b = across(i.clamp(0, n64 - 1), j)?;
            return Some([0, 1, 2, 3].map(|k| (a[k] + b[k]) * 0.5));
        }
        across(i, j)
    }

    /// A filtered read at texel-space coordinate `c` (scaled by the level's size, not yet offset by half a texel).
    fn filtered(
        &self,
        c: [f64; 3],
        offset: [i32; 3],
        linear: bool,
        axes: usize,
        cube_face: Option<u32>,
    ) -> [f32; 4] {
        let border = self.compared(self.s.border);
        let fetch = |at: [i64; 3]| -> [f32; 4] {
            let t = match cube_face {
                Some(face) => self.cube_texel(face, at[0], at[1]),
                None => self.texel(at),
            };
            t.unwrap_or(border)
        };
        if !linear {
            let mut at = [0i64; 3];
            for k in 0..axes {
                at[k] = c[k].floor() as i64 + i64::from(offset[k]);
            }
            return fetch(at);
        }
        let mut base = [0i64; 3];
        let mut frac = [0f32; 3];
        for k in 0..axes {
            let x = c[k] - 0.5;
            let fl = x.floor();
            base[k] = fl as i64 + i64::from(offset[k]);
            frac[k] = subtexel(x - fl) as f32;
        }
        let corners = 1usize << axes;
        let mut taps = Vec::with_capacity(corners);
        for m in 0..corners {
            let mut at = base;
            let mut w = 1f32;
            for k in 0..axes {
                if m >> k & 1 == 1 {
                    at[k] += 1;
                    w *= frac[k];
                } else {
                    w *= 1.0 - frac[k];
                }
            }
            taps.push((fetch(at), w));
        }
        match self.s.reduction {
            Reduction::Minimum | Reduction::Maximum => {
                let max = self.s.reduction == Reduction::Maximum;
                let mut out: Option<[f32; 4]> = None;
                for (t, w) in &taps {
                    if *w == 0.0 {
                        continue;
                    }
                    out = Some(match out {
                        None => *t,
                        Some(o) => [0, 1, 2, 3].map(|k| if max { o[k].max(t[k]) } else { o[k].min(t[k]) }),
                    });
                }
                out.unwrap_or(taps[0].0)
            }
            _ => {
                // tap i has bit k set for the far texel on axis k, so neighbouring pairs differ on the lowest axis
                // left: interpolate along u, then v, then w
                let mut layer = taps.iter().map(|(t, _)| *t).collect::<Vec<_>>();
                for &f in &frac[..axes] {
                    layer = layer
                        .chunks(2)
                        .map(|p| [0, 1, 2, 3].map(|l| p[0][l] + (p[1][l] - p[0][l]) * f))
                        .collect();
                }
                layer[0]
            }
        }
    }

    /// The four texels a gather reads, component `c`: (u0, v1), (u1, v1), (u1, v0), (u0, v0) (7.18.10).
    fn gather(&self, c: [f64; 3], offset: [i32; 3], component: u8, cube_face: Option<u32>) -> [f32; 4] {
        let border = self.compared(self.s.border);
        let fetch = |at: [i64; 3]| -> [f32; 4] {
            let t = match cube_face {
                Some(face) => self.cube_texel(face, at[0], at[1]),
                None => self.texel(at),
            };
            t.unwrap_or(border)
        };
        let x = (c[0] - 0.5).floor() as i64 + i64::from(offset[0]);
        let y = (c[1] - 0.5).floor() as i64 + i64::from(offset[1]);
        let z = c[2].floor() as i64;
        let k = usize::from(component.min(3));
        let comp = if self.reference.is_some() { 0 } else { k };
        [
            fetch([x, y + 1, z])[comp],
            fetch([x + 1, y + 1, z])[comp],
            fetch([x + 1, y, z])[comp],
            fetch([x, y, z])[comp],
        ]
    }
}

/// The level of detail of a request on `r` (7.18.11), before clamping: `None` where the program states none and no
/// footprint is known.
fn level_of_detail(r: &Resource, s: &Sampler, lod: Lod, axes: usize) -> f32 {
    let [w, h, d] = r.level_dims(0);
    let size = [w as f32, h as f32, d as f32];
    match lod {
        Lod::Level(l) => l + s.bias,
        Lod::None(b) => 0.0 + b.clamp(-16.0, 15.99) + s.bias,
        Lod::Gradients(g, b) => {
            let n = if r.dim == Dim::Cube { 3 } else { axes };
            let scale = if r.dim == Dim::Cube {
                // a cube's derivatives are of its direction: the face's texels per unit of the face coordinate is
                // half its width (7.18.11.3)
                [w as f32 * 0.5; 3]
            } else {
                size
            };
            let length = |v: [f32; 3]| -> f32 {
                (0..n)
                    .map(|k| f64::from(v[k] * scale[k]).powi(2))
                    .sum::<f64>()
                    .sqrt() as f32
            };
            let rho = length(g[0]).max(length(g[1]));
            rho.log2() + b.clamp(-16.0, 15.99) + s.bias
        }
    }
}

/// A sample of `r` through `s` (7.18): four lanes as floats.
pub fn sample(r: &Resource, s: &Sampler, req: &Request, one_d: bool) -> [f32; 4] {
    let axes = axes(r, one_d);
    let layers = if r.dim == Dim::Cube {
        r.layers / 6
    } else {
        r.layers
    };
    let layer = req.layer.map_or(0, |l| {
        (l.round_ties_even().max(0.0) as u32).min(layers.saturating_sub(1))
    });
    let layer = if r.dim == Dim::Cube { layer * 6 } else { layer };
    let lod = level_of_detail(r, s, req.lod, axes);
    let lo = s.lod_range[0].max(r.min_lod_clamp);
    let lod = if lod.is_nan() {
        lo
    } else {
        lod.clamp(lo, s.lod_range[1])
    };
    let last = (r.levels - 1) as f32;
    let magnified = lod <= 0.0;
    let linear = if magnified { s.mag_linear } else { s.min_linear };
    let level_at = |l: u32| Level {
        r,
        level: l,
        layer,
        s,
        reference: req.reference,
    };
    let at_level = |l: u32| -> [f32; 4] {
        let lv = level_at(l);
        let [w, h, d] = lv.dims();
        let (coord, face) = if r.dim == Dim::Cube {
            let (face, uv) = cube_face(req.coord);
            ([uv[0], uv[1], 0.0], Some(face))
        } else {
            (req.coord, None)
        };
        let c = [
            f64::from(coord[0]) * f64::from(w),
            f64::from(coord[1]) * f64::from(h),
            f64::from(coord[2]) * f64::from(d),
        ];
        let n = if face.is_some() { 2 } else { axes };
        match req.gather {
            Some(component) => lv.gather(c, req.offset, component, face),
            None => lv.filtered(c, req.offset, linear, n, face),
        }
    };
    if req.gather.is_some() {
        return at_level(0);
    }
    let lod = lod.max(0.0).min(last);
    if s.mip_linear && !magnified {
        let lo = lod.floor();
        let f = subtexel(f64::from(lod - lo)) as f32;
        let a = at_level(lo as u32);
        if f == 0.0 || lo as u32 + 1 >= r.levels {
            return a;
        }
        let b = at_level(lo as u32 + 1);
        return [0, 1, 2, 3].map(|k| a[k] + (b[k] - a[k]) * f);
    }
    at_level((lod + 0.5).floor().min(last) as u32)
}

/// A texel load (`ld`): the lanes as the view returns them (f32 bits or integers), 0 outside the resource (D3D11.3
/// 22.4.6).
pub fn load(r: &Resource, texel: [i32; 3], layer: Option<i32>, level: i32, offset: [i32; 3]) -> [u32; 4] {
    if level < 0 {
        return [0; 4];
    }
    let at = [
        i64::from(texel[0]) + i64::from(offset[0]),
        i64::from(texel[1]) + i64::from(offset[1]),
        i64::from(texel[2]) + i64::from(offset[2]),
    ];
    let layer = match layer {
        Some(l) if l < 0 => return [0; 4],
        Some(l) => l as u32,
        None => 0,
    };
    r.texel(level as u32, layer, at).map_or([0; 4], |t| r.mapped(t))
}
