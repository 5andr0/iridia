//! A draw on the CPU, as Direct3D draws points, lines and triangles: the vertex program for each vertex, each primitive
//! clipped, mapped through the viewport and rasterized, the pixel program for each pixel it covers, the depth test,
//! and the render targets written. It is for re-running a draw recorded from a running game on the programs the record
//! names, so its pixels can be set against the ones the game wrote.
//!
//! The rules are the Direct3D 11 functional specification's (which Direct3D 12 keeps), held to Microsoft's WARP
//! rasterizer by the `exec_warp_draw` tests:
//!
//! - **Assembly.** [`Primitive::assemble`] makes a draw's indices into points, lines and triangles by the topology, as
//!   [`Topology::primitives`] does for a geometry program (a triangle strip's odd triangles swap their last two
//!   vertices); the first vertex of each is the provoking one. Each primitive's number, which a pixel program reads as
//!   `SV_PrimitiveID`, counts the draw's primitives from 0. An adjacency topology needs a geometry program, and a
//!   strip restart is refused, as [`crate::geometry::GeometryShader::draw_indexed`] refuses it.
//! - **Linking.** A pixel program input reads the vertex program output of the same semantic and index (names compared
//!   without case), component by component; `SV_Position` is the pixel's position, `SV_IsFrontFace` the triangle's
//!   facing (a point and a line face the front). `SV_PrimitiveID`, `SV_RenderTargetArrayIndex` and
//!   `SV_ViewportArrayIndex` are the stage before's output of each where it writes one (the value as written, 15.8.1
//!   and 15.15), else the primitive's number and 0. A draw has one sample a pixel: `SV_SampleIndex` is 0 and
//!   `SV_Coverage` the pixel's one bit ANDed with the sample mask, 0 on a helper (16.3.2). An input no output carries
//!   is an error.
//! - **Clipping.** A triangle is clipped to `0 <= z <= w` (depth clipping) and to `-w <= x <= w`, `-w <= y <= w`: the
//!   viewport, section 15.4 of the D3D11.3 specification's clip volume. 15.4 allows a guard band ("or arbitrarily wider
//!   range if implementation uses a guard band"); WARP uses none for a triangle (one cut 16.25 pixels past the viewport
//!   instead of at it puts the pixels it draws up to 1e-4 off WARP's), so a vertex clipping makes on a side of the
//!   viewport is snapped like any other. A line is clipped to `0 <= z <= w` and in x and y only to a guard band that
//!   keeps it in the fixed-point range, as a GPU draws it: 3.4.3.1 lets a line clipped to the viewport differ in the
//!   viewport's edge pixels, and WARP clips it there. A vertex near the camera plane, whose screen position is far
//!   outside, is snapped only after clipping brings it in. Those planes leave every vertex `w >= 0`, `z >= 0`, so a
//!   division by `w` is finite but at the clip-space origin, which a clipped polygon reaches only when its triangle's
//!   vertices are linearly dependent: its plane holds the camera and it has no area on screen, so it draws nothing (a
//!   line through the origin likewise). A vertex clipping makes holds its weights on the primitive's vertices in clip
//!   space, and its outputs are the primitive's by the interpolation of the input they feed: perspective-correct ones
//!   by those weights, the screen-linear ones (`noperspective`) by the weights of its screen position (a clip-space
//!   weight times the vertex's w, normalised), and flat ones the provoking vertex's. A point is drawn or not as a
//!   whole: kept when its vertex is inside the clip volume. A primitive with a NaN position after clipping draws
//!   nothing (15.4); one with an infinity becomes NaN where clipping cuts it, which 15.4 allows ("may or may not result
//!   in a discarded primitive").
//! - **Clip and cull distances.** The stage before the rasterizer's `SV_ClipDistance` and `SV_CullDistance` outputs
//!   (one distance a lane): a primitive whose vertices are all below 0 on one cull distance is culled whole, and each
//!   clip distance is one more plane a primitive is clipped to, keeping where it is at least 0 (15.4: the distances
//!   are interpolated across the primitive; the distance at a point clipping makes is its weights' blend of the
//!   vertices', as its outputs are). A point is drawn when every clip distance of its vertex is at least 0.
//!   Direct3D 9's user clip planes ([`Rasterizer::clip_planes`], in clip space as a programmable vertex stage has them)
//!   are clip distances too: a vertex's distance from a plane is the plane's dot product with its clip position.
//! - **Viewport.** `x = X + (x/w + 1) / 2 * W`, `y = Y + (1 - y/w) / 2 * H` (y down), `z = MinDepth + z/w * (MaxDepth -
//!   MinDepth)`, by the viewport the primitive's leading vertex picks (`SV_ViewportArrayIndex`; an index of 16 or
//!   more is 0, 15.8.1; one the state sets no viewport for draws nothing, as an unset viewport has no area). It
//!   writes the target slice the leading vertex picks (`SV_RenderTargetArrayIndex`; one past the surface's slices
//!   is 0, 15.15).
//! - **Rasterization.** Positions are snapped to 1/256 of a pixel (16.8 fixed point), as D3D's 8 bits of sub-pixel
//!   precision (15.16), to the nearest with a tie to even and saturating at the format's range (3.2.4.1).
//!   - A triangle draws a pixel when its centre is inside it, or on an edge that is a top or a left edge. The facing is
//!     the sign of the snapped area: with y down a clockwise triangle is front facing unless the state says
//!     counter-clockwise; culling drops the facing it names; a zero-area triangle draws nothing.
//!   - A point is the 1 x 1 square about its position, two triangles under the top-left rule: it draws the pixel whose
//!     centre is in `[x - 1/2, x + 1/2) x [y - 1/2, y + 1/2)`. Direct3D 9's point size ([`Rasterizer::point_size`])
//!     makes it the `s x s` square, `[x - s/2, x + s/2) x [y - s/2, y + s/2)`, `s` the vertex program's `PSIZE`
//!     output where it writes one, else the state's size, held to the state's range; with point sprites on, each
//!     texture coordinate a pixel reads (`TEXCOORD`) is its centre's place on the square, (0, 0) at the top left to
//!     (1, 1) at the bottom right, in its first two lanes.
//!   - An aliased line draws by the diamond exit rule (3.4.3's figure): "a line covers a pixel if the line exits the
//!     pixel's diamond test area, when traveling along the line from the start towards the end". The test area is
//!     `|dx| + |dy| < 1/2` about the pixel's centre with its lower-left and lower-right edges and its bottom corner,
//!     and for a line more vertical than horizontal its right corner too (a line at 45 degrees is x-major). So a pixel
//!     is drawn when the line meets its test area and does not end inside it. A line of no snapped length draws
//!     nothing. A GPU (an NVIDIA RTX 5070 Ti) draws exactly that on every pixel of 10,640 recorded lines: 3,000
//!     random ones, 2,400 at 45 degrees, 4,640 ending at every 1/16-pixel point of a diamond in all eight octants and
//!     600 sliding along diamond edges (`exec_warp_draw`'s `points_and_lines_draw_what_a_gpu_drew`). WARP does not
//!     (`warp_draws_otherwise_only_as_documented`, on the same lines): it draws 1,999 pixels that lines end inside,
//!     where a line entering the test area, extended past its end, crosses the area's diagonal along the major axis
//!     between the corner it entered by and the centre; it draws the pixel on the other side for 856 pixels along
//!     which a 45-degree line lies on a diamond's edge (the choice changes with each 1/256 step of an end) and for 36
//!     where a line passes exactly through a diamond's corner on its minor axis; and it clips a line to the viewport,
//!     which changed 1 pixel on the viewport's border. Points it draws as the GPU does.
//! - **Interpolation.** At the pixel centre, from the snapped positions: perspective-correct (the default), linear in
//!   screen space (`noperspective`) or the provoking vertex's value (`nointerpolation`, integers; the primitive's first
//!   vertex). Direct3D 9's flat shade mode ([`Rasterizer::flat_colours`]) gives the colour inputs (`COLOR`) the
//!   provoking vertex's value and leaves the rest as they are. On a line the weight is the centre's place along the
//!   line's major axis; on a point every value is the vertex's. Centroid and sample interpolation are the centre's (one
//!   sample a pixel). Depth is linear in screen space; the depth test and write clamp it to the viewport's depth range,
//!   but `SV_Position.z` is the value before the clamp (a line's pixel whose centre is past an end the near plane cut
//!   reads a depth below 0 on WARP and the GPU).
//! - **Pixels.** In 2 x 2 quads aligned to even pixels, uncovered lanes run as helpers (their inputs extrapolated) so
//!   derivatives and implicit levels of detail are the quad's; a helper writes nothing. A discarded pixel writes
//!   nothing, and neither does one that fails Direct3D 9's alpha test ([`Rasterizer::alpha_test`]), which compares
//!   the alpha of the program's first target with the reference, or one whose sample the sample mask leaves out. The
//!   depth test runs after the pixel program (its result is the same: a pixel that fails writes nothing); a pixel
//!   that passes writes every target lane the program writes and, when the state writes depth, its depth. Primitives are drawn in order, so a later one overwrites an earlier one where both pass.
use crate::geometry::Topology;
use crate::{Error, Execution, PixelShader, ProgramShader, Sampler};
use iridia::{InterpMode, SystemValue, Varying};

fn error(detail: impl Into<String>) -> Error {
    Error {
        bytecode: "draw",
        at: 0,
        detail: detail.into(),
    }
}

/// The viewport: where clip space lands in the targets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    /// Left edge of the viewport, in pixels.
    pub x: f32,
    /// Top edge of the viewport, in pixels (y runs down).
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
    /// The depth that clip-space z/w = 0 maps to.
    pub min_depth: f32,
    /// The depth that clip-space z/w = 1 maps to.
    pub max_depth: f32,
}

/// Which facing culling drops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cull {
    /// No triangle is culled.
    None,
    /// Front-facing triangles are culled.
    Front,
    /// Back-facing triangles are culled.
    Back,
}

/// A comparison of a pixel's value (left) with another (right): its depth with the depth buffer's, its alpha with the
/// alpha test's reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    /// Never passes.
    Never,
    /// Passes where the pixel's depth is less.
    Less,
    /// Passes where equal.
    Equal,
    /// Passes where less or equal.
    LessEqual,
    /// Passes where greater.
    Greater,
    /// Passes where not equal.
    NotEqual,
    /// Passes where greater or equal.
    GreaterEqual,
    /// Always passes.
    Always,
}

impl Compare {
    fn passes(self, pixel: f32, stored: f32) -> bool {
        match self {
            Compare::Never => false,
            Compare::Less => pixel < stored,
            Compare::Equal => pixel == stored,
            Compare::LessEqual => pixel <= stored,
            Compare::Greater => pixel > stored,
            Compare::NotEqual => pixel != stored,
            Compare::GreaterEqual => pixel >= stored,
            Compare::Always => true,
        }
    }
}

/// The depth test and whether a pixel that passes writes its depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthState {
    /// The comparison of the pixel's depth with the depth buffer's.
    pub compare: Compare,
    /// Whether a pixel that passes writes its depth.
    pub write: bool,
}

/// The fixed-function state of a draw.
#[derive(Clone, Debug, PartialEq)]
pub struct Rasterizer {
    /// Where clip space lands in the targets: the viewports by index, which a primitive picks
    /// (`SV_ViewportArrayIndex`), the first by default. At least one.
    pub viewports: Vec<Viewport>,
    /// The sample mask: a draw's one sample is written where bit 0 is set, and `SV_Coverage` is ANDed with it.
    pub sample_mask: u32,
    /// Which facing is dropped.
    pub cull: Cull,
    /// Whether a counter-clockwise triangle (y down) is front facing.
    pub front_counter_clockwise: bool,
    /// The depth test; `None`: no depth buffer.
    pub depth: Option<DepthState>,
    /// Direct3D 9's alpha test (`D3DRS_ALPHATESTENABLE`); `None`: none.
    pub alpha_test: Option<AlphaTest>,
    /// Direct3D 9's user clip planes (`SetClipPlane`, `D3DRS_CLIPPLANEENABLE`), each `None` where not enabled: in clip
    /// space, a primitive kept where a plane's dot product with the position is at least 0.
    pub clip_planes: [Option<[f32; 4]>; 6],
    /// Direct3D 9's point size; `None`: a point is one pixel, as from Direct3D 10 on.
    pub point_size: Option<PointSize>,
    /// Direct3D 9's point sprites (`D3DRS_POINTSPRITEENABLE`): a point's texture coordinates are each pixel's place on
    /// its square.
    pub point_sprite: bool,
    /// Direct3D 9's flat shade mode (`D3DSHADE_FLAT`): the pixel program's colour inputs (`COLOR`) are the provoking
    /// vertex's, whatever their interpolation; its other inputs are interpolated as they would be.
    pub flat_colours: bool,
}

/// Direct3D 9's point size: `D3DRS_POINTSIZE`, which a vertex program's `PSIZE` output replaces, held to
/// `[D3DRS_POINTSIZE_MIN, D3DRS_POINTSIZE_MAX]`, in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointSize {
    /// The size where the vertex program writes none.
    pub size: f32,
    /// The least size.
    pub min: f32,
    /// The greatest size.
    pub max: f32,
}

impl PointSize {
    /// The size of a point whose vertex program wrote `written` (`None`: no `PSIZE` output), held to the range (a NaN
    /// is the least).
    fn of(&self, written: Option<f32>) -> f32 {
        written.unwrap_or(self.size).max(self.min).min(self.max)
    }
}

/// Direct3D 9's alpha test: a pixel is written only where the alpha of its first target (output register 0, lane 3)
/// passes the comparison with the reference (`D3DRS_ALPHAFUNC`; the alpha is the left operand). The reference is
/// `D3DRS_ALPHAREF`'s byte as a fraction of 255.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaTest {
    /// The comparison.
    pub compare: Compare,
    /// The value compared with.
    pub reference: f32,
}

/// What a draw writes to: render targets by output register (`width * height * layers` texels each, slice by slice
/// and row by row, as the pixel program's output words) and the depth buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct Surface {
    /// Width of every target in pixels.
    pub width: u32,
    /// Height of every target in pixels.
    pub height: u32,
    /// Slices of every target (an array target's), which a primitive picks (`SV_RenderTargetArrayIndex`). At least
    /// one.
    pub layers: u32,
    /// By output register; `None` for a register the draw does not write.
    pub targets: Vec<Option<Vec<[u32; 4]>>>,
    /// `width * height * layers` depths, when the state tests depth.
    pub depth: Option<Vec<f32>>,
}

/// What a draw did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawStats {
    /// Primitives drawn (points, lines and triangles).
    pub primitives: usize,
    /// Triangles dropped by culling, and primitives by a cull distance.
    pub culled: usize,
    /// Primitives clipped away entirely.
    pub clipped_away: usize,
    /// Primitives with a NaN position after clipping, which draw nothing.
    pub not_a_number: usize,
    /// Pixels the program ran for (helpers not counted).
    pub shaded: usize,
    /// Pixels the pixel program discarded.
    pub discarded: usize,
    /// Pixels that failed the alpha test.
    pub alpha_failed: usize,
    /// Pixels that failed the depth test.
    pub depth_failed: usize,
    /// Pixels the sample mask left out.
    pub masked: usize,
    /// Pixels written to the targets.
    pub written: usize,
}

/// The vertices a primitive joins, as indices into a draw's vertices; the first is the provoking one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corners {
    /// A point.
    Point(usize),
    /// A line from the first vertex to the second.
    Line([usize; 2]),
    /// A triangle.
    Triangle([usize; 3]),
}

/// A primitive of a draw: its vertices and its number, which a pixel program reads as `SV_PrimitiveID`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Primitive {
    /// Its vertices.
    pub corners: Corners,
    /// Its number in the draw.
    pub id: u32,
}

impl Primitive {
    /// The primitives `topology` makes of a draw's `indices` (each an index into its vertices), numbered from 0 in
    /// order. When the draw has a restart index (`restart`, the value that cuts a strip: 0xFFFF or 0xFFFFFFFF in D3D),
    /// an index list that holds it is refused, as [`crate::geometry::GeometryShader::draw_indexed`] refuses it; an
    /// adjacency topology is refused, as its adjacent vertices are a geometry program's to read.
    pub fn assemble(topology: Topology, indices: &[u32], restart: Option<u32>) -> Result<Vec<Self>, Error> {
        if let Some(cut) = restart
            && let Some(at) = indices.iter().position(|&i| i == cut)
        {
            return Err(error(format!(
                "index {at} is the restart index {cut:#x}: a draw with a strip restart is not assembled (the \
                 primitives around a cut are not measured on the D3D reference)"
            )));
        }
        if matches!(
            topology,
            Topology::LineListAdjacency
                | Topology::LineStripAdjacency
                | Topology::TriangleListAdjacency
                | Topology::TriangleStripAdjacency
        ) {
            return Err(error(format!("a {topology:?} draw without a geometry program")));
        }
        let index = |at: usize| -> Result<usize, Error> {
            indices
                .get(at)
                .map(|&i| i as usize)
                .ok_or_else(|| error("a primitive past the indices"))
        };
        topology
            .primitives(indices.len())
            .into_iter()
            .enumerate()
            .map(|(n, group)| {
                let corners = match group.as_slice() {
                    [a] => Corners::Point(index(*a)?),
                    [a, b] => Corners::Line([index(*a)?, index(*b)?]),
                    [a, b, c] => Corners::Triangle([index(*a)?, index(*b)?, index(*c)?]),
                    _ => return Err(error("a primitive of an adjacency topology")),
                };
                let id = u32::try_from(n).map_err(|_| error("more than 2^32 primitives"))?;
                Ok(Self { corners, id })
            })
            .collect()
    }

    /// The indices of its vertices.
    fn indices(&self) -> &[usize] {
        match &self.corners {
            Corners::Point(i) => std::slice::from_ref(i),
            Corners::Line(l) => l,
            Corners::Triangle(t) => t,
        }
    }
}

/// Where a pixel program input comes from.
#[derive(Clone, Debug)]
enum Source {
    /// A vertex program output: its register and lanes, the pixel input's register and lanes, the interpolation,
    /// whether it is a texture coordinate (`TEXCOORD`), which a point sprite replaces, and whether it is a colour
    /// (`COLOR`), which Direct3D 9's flat shade mode makes flat.
    Varying {
        from_register: usize,
        from_lanes: Vec<u8>,
        to_register: usize,
        to_lanes: Vec<u8>,
        mode: InterpMode,
        texcoord: bool,
        colour: bool,
    },
    Position {
        register: usize,
        lanes: Vec<u8>,
    },
    FrontFace {
        register: usize,
        lane: u8,
    },
    /// The primitive's number.
    Primitive {
        register: usize,
        lane: u8,
    },
    /// A word every pixel reads alike (`SV_SampleIndex`'s 0, a target or viewport index no stage writes).
    Word {
        register: usize,
        lane: u8,
        word: u32,
    },
    /// The pixel's coverage (`SV_Coverage`).
    Coverage {
        register: usize,
        lane: u8,
    },
}

fn lanes_of(mask: u8) -> Vec<u8> {
    (0..4).filter(|l| mask >> l & 1 != 0).collect()
}

fn same_label(a: &Varying, b: &Varying) -> bool {
    a.label.index == b.label.index && a.label.name.eq_ignore_ascii_case(&b.label.name)
}

/// A vertex and a pixel program linked for drawing.
#[derive(Debug)]
pub struct DrawShader<'a> {
    vertex: &'a ProgramShader,
    raster: Raster<'a>,
    /// Where the vertex program's clip and cull distances and point size are in its output registers.
    values: RasterValues,
}

/// A pixel program linked to the outputs of the last stage before it, by varying (a geometry, domain or mesh
/// program's, whose vertices are one register per output varying in the order of its interface, each varying's lanes
/// from the first): draws the vertices that stage made.
#[derive(Debug)]
pub struct ShadedDraw<'a> {
    raster: Raster<'a>,
    /// The output varying that is the clip-space position.
    position: usize,
    /// Where the clip and cull distances and point size are in a vertex's registers.
    values: RasterValues,
}

/// Where the rasterizer's values of a vertex a stage makes, other than its position, are in its registers: its clip
/// and cull distances (in the order of the stage's outputs), its target slice and viewport indices, and Direct3D 9's
/// point size (the `PSIZE` output), a register and a lane each.
#[derive(Clone, Debug, Default)]
struct RasterValues {
    clip: Vec<(usize, u8)>,
    cull: Vec<(usize, u8)>,
    point_size: Option<(usize, u8)>,
    target_index: Option<(usize, u8)>,
    viewport_index: Option<(usize, u8)>,
}

impl RasterValues {
    /// The values of `outputs`, an output varying's lanes in the registers `from` gives.
    fn of(
        outputs: &[Varying],
        from: impl Fn(usize, &Varying) -> Result<(usize, Vec<u8>), Error>,
    ) -> Result<Self, Error> {
        let mut d = RasterValues::default();
        for (at, out) in outputs.iter().enumerate() {
            let one = match out.system {
                Some(SystemValue::ClipDistance) => None,
                Some(SystemValue::CullDistance) => None,
                Some(SystemValue::RenderTargetArrayIndex) => Some(&mut d.target_index),
                Some(SystemValue::ViewportArrayIndex) => Some(&mut d.viewport_index),
                None if out.label.name.eq_ignore_ascii_case("PSIZE") && out.label.index == 0 => {
                    Some(&mut d.point_size)
                }
                _ => continue,
            };
            if let Some(one) = one {
                let (register, lanes) = from(at, out)?;
                *one = lanes.first().map(|&lane| (register, lane));
                continue;
            }
            let into = if out.system == Some(SystemValue::ClipDistance) {
                &mut d.clip
            } else {
                &mut d.cull
            };
            let (register, lanes) = from(at, out)?;
            into.extend(lanes.into_iter().map(|lane| (register, lane)));
        }
        Ok(d)
    }

    /// A vertex whose clip-space position is `position` and whose output registers are `outputs`, with its values
    /// (0 for a lane the registers do not hold).
    fn shaded(&self, position: [f32; 4], outputs: Vec<[u32; 4]>) -> Shaded {
        let bits = |&(register, lane): &(usize, u8)| {
            outputs
                .get(register)
                .and_then(|r| r.get(usize::from(lane)))
                .copied()
                .unwrap_or(0)
        };
        let word = |at: &(usize, u8)| f32::from_bits(bits(at));
        Shaded {
            position,
            clip: self.clip.iter().map(word).collect(),
            cull: self.cull.iter().map(word).collect(),
            point_size: self.point_size.as_ref().map(word),
            target_index: self.target_index.as_ref().map_or(0, bits),
            viewport_index: self.viewport_index.as_ref().map_or(0, bits),
            outputs,
        }
    }
}

/// A pixel program, where each of its inputs comes from, and the rasterizer that runs it over primitives.
#[derive(Debug)]
struct Raster<'a> {
    pixel: &'a ProgramShader,
    sources: Vec<Source>,
    /// How many input registers a pixel's inputs take.
    inputs: usize,
}

/// A vertex after the vertex program: its clip-space position, its output registers, its clip and cull distances,
/// the point size it writes, and the target slice and viewport it picks (0 where it writes none).
#[derive(Clone, Debug)]
struct Shaded {
    position: [f32; 4],
    outputs: Vec<[u32; 4]>,
    clip: Vec<f32>,
    cull: Vec<f32>,
    point_size: Option<f32>,
    target_index: u32,
    viewport_index: u32,
}

/// A vertex in the rasterizer: its clip-space position and its weights on the drawn primitive's vertices in clip space
/// (a vertex of the primitive weighs itself one; one clipping makes, the blend of its edge's ends; a line's third
/// weight and a point's second and third are 0).
#[derive(Clone, Debug)]
struct ClipVertex {
    clip: [f32; 4],
    weights: [f64; 3],
}

/// The weights of the screen position of a point whose clip-space weights on a primitive are `clip` and whose
/// vertices' w are `w`: each clip-space weight times its vertex's w, normalised (a screen position is the clip-space
/// one over its w, so it is the blend of the vertices' screen positions by these).
fn screen_weights(clip: [f64; 3], w: [f64; 3]) -> [f64; 3] {
    let scaled = [clip[0] * w[0], clip[1] * w[1], clip[2] * w[2]];
    let sum = scaled[0] + scaled[1] + scaled[2];
    scaled.map(|x| x / sum)
}

/// A pixel of a primitive, for the pixel program: whether the primitive covers it, its depth, and the weights its
/// inputs blend the primitive's vertices by.
#[derive(Clone, Copy, Debug, Default)]
struct Fragment {
    covered: bool,
    depth: f32,
    /// The weights of a perspective-correct input on the primitive's vertices.
    perspective: [f64; 3],
    /// The weights of a screen-linear input.
    linear: [f64; 3],
    /// The interpolated w (the fourth lane of `SV_Position`).
    w: f32,
    /// On a point sprite, the pixel's place on the point's square, which its texture coordinates read.
    sprite: Option<[f32; 2]>,
}

/// What a primitive's pixels share: its vertices (the first the provoking one), its facing, its number, its viewport
/// and the target slice it writes.
#[derive(Clone, Copy, Debug)]
struct Drawn<'v> {
    original: &'v [&'v Shaded],
    front: bool,
    id: u32,
    viewport: Viewport,
    layer: u32,
}

impl<'a> Raster<'a> {
    /// Link `pixel`'s inputs to the outputs of the stage before it by semantic: `outputs` are that stage's output
    /// varyings, `from` where an output's lanes are in the registers of a vertex it made.
    fn new(
        outputs: &'a [Varying],
        pixel: &'a ProgramShader,
        from: impl Fn(usize, &Varying) -> Result<(usize, Vec<u8>), Error>,
    ) -> Result<Self, Error> {
        let mut sources = Vec::new();
        let mut inputs = pixel.input_count();
        for v in pixel.interface().inputs.iter().filter(|v| v.used) {
            let name = || format!("{}{}", v.label.name, v.label.index);
            // the inputs a draw supplies: the stage before's output of each where it writes one (a geometry program's
            // primitive number, a target or viewport index), else the draw's own: the primitive's number, index 0,
            // the one sample (0) and its coverage
            if let Some(
                system @ (SystemValue::PrimitiveIndex
                | SystemValue::RenderTargetArrayIndex
                | SystemValue::ViewportArrayIndex
                | SystemValue::SampleIndex
                | SystemValue::Coverage),
            ) = v.system
            {
                let (register, lane) = match v.provenance.d3d {
                    Some(sig) => (
                        sig.register as usize,
                        lanes_of(sig.mask).first().copied().unwrap_or(0),
                    ),
                    None => (
                        pixel.system_input(system).ok_or_else(|| {
                            error(format!(
                                "pixel input {} has no register in the program's view",
                                name()
                            ))
                        })?,
                        0,
                    ),
                };
                inputs = inputs.max(register.saturating_add(1));
                let written = !matches!(system, SystemValue::SampleIndex | SystemValue::Coverage);
                match outputs
                    .iter()
                    .enumerate()
                    .find(|(_, o)| written && o.system == Some(system))
                {
                    Some((at, out)) => {
                        let (from_register, from_lanes) = from(at, out)?;
                        sources.push(Source::Varying {
                            from_register,
                            from_lanes,
                            to_register: register,
                            to_lanes: vec![lane],
                            mode: InterpMode::Flat,
                            texcoord: false,
                            colour: false,
                        });
                    }
                    None => sources.push(match system {
                        SystemValue::PrimitiveIndex => Source::Primitive { register, lane },
                        SystemValue::Coverage => Source::Coverage { register, lane },
                        _ => Source::Word {
                            register,
                            lane,
                            word: 0,
                        },
                    }),
                }
                continue;
            }
            let Some(sig) = v.provenance.d3d else {
                return Err(error(format!("pixel input {} has no register", name())));
            };
            let to_register = sig.register as usize;
            let to_lanes = lanes_of(sig.mask);
            match v.system {
                Some(SystemValue::FragCoord) => {
                    sources.push(Source::Position {
                        register: to_register,
                        lanes: to_lanes,
                    });
                    continue;
                }
                Some(SystemValue::FrontFacing) => {
                    // the signature names the lane the facing is read in: a mask of none is a malformed signature
                    let Some(&lane) = to_lanes.first() else {
                        return Err(error(format!(
                            "pixel input {} is the facing, and its signature mask names no lane",
                            name()
                        )));
                    };
                    sources.push(Source::FrontFace {
                        register: to_register,
                        lane,
                    });
                    continue;
                }
                Some(other) => {
                    return Err(error(format!(
                        "pixel input {} is the system value {other:?}, which a draw does not supply",
                        name()
                    )));
                }
                None => {}
            }
            let (at, out) = outputs
                .iter()
                .enumerate()
                .find(|(_, o)| o.system.is_none() && same_label(o, v))
                .ok_or_else(|| {
                    error(format!(
                        "pixel input {} has no output of its semantic in the stage before it",
                        name()
                    ))
                })?;
            let (from_register, from_lanes) = from(at, out)?;
            let mode = match v.interp {
                Some(i) => i.mode,
                None if v.value.scalar == iridia::ScalarType::Float => InterpMode::Perspective,
                None => InterpMode::Flat,
            };
            sources.push(Source::Varying {
                from_register,
                from_lanes,
                to_register,
                to_lanes,
                mode,
                texcoord: v.label.name.eq_ignore_ascii_case("TEXCOORD"),
                colour: v.label.name.eq_ignore_ascii_case("COLOR"),
            });
        }
        Ok(Self {
            pixel,
            sources,
            inputs,
        })
    }

    /// Check `surface` against `state` before a draw.
    fn check(state: &Rasterizer, surface: &Surface) -> Result<(), Error> {
        let n = (surface.width as usize)
            .saturating_mul(surface.height as usize)
            .saturating_mul(surface.layers as usize);
        if surface.layers == 0 {
            return Err(error("a surface of no slices"));
        }
        if surface.targets.iter().flatten().any(|t| t.len() != n)
            || surface.depth.as_ref().is_some_and(|d| d.len() != n)
        {
            return Err(error(
                "a target or the depth buffer is not width x height x layers",
            ));
        }
        if state.viewports.is_empty() {
            return Err(error("a state of no viewports"));
        }
        if state.depth.is_some() && surface.depth.is_none() {
            return Err(error("the state tests depth and the surface has no depth buffer"));
        }
        Ok(())
    }

    /// Draw a primitive of shaded vertices (the first the provoking one).
    #[allow(clippy::too_many_arguments)]
    fn primitive(
        &self,
        original: &[&Shaded],
        id: u32,
        buffers: &[(u32, &[[u32; 4]])],
        sampler: &mut dyn Sampler,
        state: &Rasterizer,
        surface: &mut Surface,
        stats: &mut DrawStats,
    ) -> Result<(), Error> {
        stats.primitives = stats.primitives.saturating_add(1);
        // the leading vertex picks the viewport (an index of 16 or more is 0, 15.8.1; an unset one has no area) and
        // the target slice (one past the surface's is 0, 15.15)
        let lead = original.first();
        let index = lead.map_or(0, |v| v.viewport_index);
        let index = if index < 16 { index } else { 0 };
        let Some(&vp) = state.viewports.get(index as usize) else {
            stats.clipped_away = stats.clipped_away.saturating_add(1);
            return Ok(());
        };
        let layer = lead.map_or(0, |v| v.target_index);
        let layer = if layer < surface.layers { layer } else { 0 };
        // a cull distance below 0 at every vertex culls the primitive
        let culls = original.iter().map(|v| v.cull.len()).max().unwrap_or(0);
        if (0..culls).any(|k| original.iter().all(|v| v.cull.get(k).is_some_and(|d| *d < 0.))) {
            stats.culled = stats.culled.saturating_add(1);
            return Ok(());
        }
        // each clip distance at the primitive's vertices: one more plane to clip to
        let clips = original.iter().map(|v| v.clip.len()).max().unwrap_or(0);
        let user: Vec<[f64; 3]> = (0..clips)
            .map(|k| {
                [0, 1, 2].map(|i| {
                    original
                        .get(i)
                        .and_then(|v| v.clip.get(k))
                        .map_or(0., |d| f64::from(*d))
                })
            })
            .chain(state.clip_planes.iter().flatten().map(|plane| {
                [0, 1, 2].map(|i| {
                    original.get(i).map_or(0., |v| {
                        plane
                            .iter()
                            .zip(&v.position)
                            .map(|(&p, &x)| f64::from(p) * f64::from(x))
                            .sum()
                    })
                })
            }))
            .collect();
        let corners = original
            .iter()
            .enumerate()
            .map(|(k, vertex)| {
                let mut weights = [0.; 3];
                if let Some(weight) = weights.get_mut(k) {
                    *weight = 1.;
                }
                ClipVertex {
                    clip: vertex.position,
                    weights,
                }
            })
            .collect();
        let mut drawn = Drawn {
            original,
            front: true,
            id,
            viewport: vp,
            layer,
        };
        let mut run = |cover: &Cover<'_>, drawn: Drawn<'_>, stats: &mut DrawStats| {
            self.quads(cover, drawn, buffers, sampler, state, surface, stats)
        };
        match original.len() {
            3 => {
                let polygon = clip(corners, &user);
                if polygon.len() < 3 {
                    stats.clipped_away = stats.clipped_away.saturating_add(1);
                    return Ok(());
                }
                if polygon.iter().any(|v| v.clip.iter().any(|c| c.is_nan())) {
                    stats.not_a_number = stats.not_a_number.saturating_add(1);
                    return Ok(());
                }
                if let Some((fan, rest)) = polygon.split_first() {
                    for pair in rest.windows(2) {
                        if let [second, third] = pair {
                            let Some(setup) = Setup::new([fan, second, third], &vp) else {
                                continue;
                            };
                            drawn.front = setup.clockwise != state.front_counter_clockwise;
                            if (state.cull == Cull::Front && drawn.front)
                                || (state.cull == Cull::Back && !drawn.front)
                            {
                                stats.culled = stats.culled.saturating_add(1);
                                continue;
                            }
                            let corner_w = original.map_w();
                            run(
                                &Cover::Triangle(&setup, [fan, second, third], corner_w),
                                drawn,
                                stats,
                            )?;
                        }
                    }
                }
                Ok(())
            }
            2 => {
                let Some(ends) = clip_line(corners, &vp, &user) else {
                    stats.clipped_away = stats.clipped_away.saturating_add(1);
                    return Ok(());
                };
                if ends.iter().any(|v| v.clip.iter().any(|c| c.is_nan())) {
                    stats.not_a_number = stats.not_a_number.saturating_add(1);
                    return Ok(());
                }
                let Some(line) = Line::new(&ends, &vp) else {
                    return Ok(());
                };
                run(&Cover::Line(&line, &ends, original.map_w()), drawn, stats)
            }
            _ => {
                let Some(vertex) = corners.into_iter().next() else {
                    return Ok(());
                };
                if vertex.clip.iter().any(|c| c.is_nan()) {
                    stats.not_a_number = stats.not_a_number.saturating_add(1);
                    return Ok(());
                }
                let size = state
                    .point_size
                    .map_or(1., |s| s.of(original.first().and_then(|v| v.point_size)));
                let Some(point) = Point::new(&vertex, &vp, size, state.point_sprite) else {
                    stats.clipped_away = stats.clipped_away.saturating_add(1);
                    return Ok(());
                };
                if user.iter().any(|d| d[0].is_nan() || d[0] < 0.) {
                    stats.clipped_away = stats.clipped_away.saturating_add(1);
                    return Ok(());
                }
                run(&Cover::Point(&point), drawn, stats)
            }
        }
    }

    /// Run the pixel program over the pixels `cover` covers, in 2 x 2 quads aligned to even pixels, and write the
    /// pixels that pass.
    #[allow(clippy::too_many_arguments)]
    fn quads(
        &self,
        cover: &Cover<'_>,
        drawn: Drawn<'_>,
        buffers: &[(u32, &[[u32; 4]])],
        sampler: &mut dyn Sampler,
        state: &Rasterizer,
        surface: &mut Surface,
        stats: &mut DrawStats,
    ) -> Result<(), Error> {
        let vp = drawn.viewport;
        let (w, h) = (i64::from(surface.width), i64::from(surface.height));
        let slice = i64::from(drawn.layer).saturating_mul(w).saturating_mul(h);
        let Some(([min_x, max_x], [min_y, max_y])) = cover.bounds(w, h) else {
            return Ok(());
        };
        let depth_range = (vp.min_depth.min(vp.max_depth), vp.max_depth.max(vp.min_depth));
        let qx0 = min_x & !1;
        let qy0 = min_y & !1;
        let mut lane_inputs: [Vec<[u32; 4]>; 4] = Default::default();
        for qy in (qy0..max_y).step_by(2) {
            for qx in (qx0..max_x).step_by(2) {
                let (qx1, qy1) = (qx.saturating_add(1), qy.saturating_add(1));
                let lanes = [(qx, qy), (qx1, qy), (qx, qy1), (qx1, qy1)];
                let mut covered = [false; 4];
                let mut depths = [0f32; 4];
                for (((&(px, py), covered), depth), lane_input) in lanes
                    .iter()
                    .zip(covered.iter_mut())
                    .zip(depths.iter_mut())
                    .zip(lane_inputs.iter_mut())
                {
                    // the pixel program reads the depth as interpolated; the depth test and write take it clamped
                    let f = cover.fragment(px, py);
                    *covered = f.covered && px < w && py < h;
                    *depth = f.depth.clamp(depth_range.0, depth_range.1);
                    let centre = [px as f32 + 0.5, py as f32 + 0.5];
                    self.inputs_at(drawn, &f, centre, *covered, state, lane_input);
                }
                if !covered.iter().any(|c| *c) {
                    continue;
                }
                let [input0, input1, input2, input3] = &lane_inputs;
                let inputs = [
                    input0.as_slice(),
                    input1.as_slice(),
                    input2.as_slice(),
                    input3.as_slice(),
                ];
                let quad: [Execution; 4] = self.pixel.execute_quad(inputs, buffers, sampler)?;
                for (((e, &covered), &(px, py)), &depth) in quad.iter().zip(&covered).zip(&lanes).zip(&depths)
                {
                    if !covered {
                        continue;
                    }
                    stats.shaded = stats.shaded.saturating_add(1);
                    if e.discarded {
                        stats.discarded = stats.discarded.saturating_add(1);
                        continue;
                    }
                    if let Some(test) = state.alpha_test {
                        let alpha = e.outputs.first().map_or(0., |o| f32::from_bits(o[3]));
                        if !test.compare.passes(alpha, test.reference) {
                            stats.alpha_failed = stats.alpha_failed.saturating_add(1);
                            continue;
                        }
                    }
                    if state.sample_mask & 1 == 0 {
                        stats.masked = stats.masked.saturating_add(1);
                        continue;
                    }
                    // the pixel is inside the surface: the bounds are clamped to its width and height
                    let at = usize::try_from(slice.saturating_add(py.saturating_mul(w)).saturating_add(px))
                        .map_err(|_| error(format!("pixel ({px}, {py}) is outside the surface")))?;
                    if let (Some(test), Some(buffer)) = (state.depth, surface.depth.as_mut()) {
                        let stored = buffer.get_mut(at).ok_or_else(|| {
                            error(format!("pixel ({px}, {py}) is outside the depth buffer"))
                        })?;
                        if !test.compare.passes(depth, *stored) {
                            stats.depth_failed = stats.depth_failed.saturating_add(1);
                            continue;
                        }
                        if test.write {
                            *stored = depth;
                        }
                    }
                    for (register, target) in surface.targets.iter_mut().enumerate() {
                        let Some(target) = target else { continue };
                        let mask = e.output_masks.get(register).copied().unwrap_or(0);
                        let value = e.outputs.get(register).copied().unwrap_or([0; 4]);
                        let texel = target.get_mut(at).ok_or_else(|| {
                            error(format!("pixel ({px}, {py}) is outside target {register}"))
                        })?;
                        for (lane, (stored, &written)) in texel.iter_mut().zip(&value).enumerate() {
                            if mask >> lane & 1 != 0 {
                                *stored = written;
                            }
                        }
                    }
                    stats.written = stats.written.saturating_add(1);
                }
            }
        }
        Ok(())
    }

    /// The pixel program's input registers at a pixel `f` of the primitive `drawn`, whose centre is `centre`, which
    /// it covers where `covered` (not a helper), drawn with the sample mask `sample_mask`.
    fn inputs_at(
        &self,
        drawn: Drawn<'_>,
        f: &Fragment,
        centre: [f32; 2],
        covered: bool,
        state: &Rasterizer,
        into: &mut Vec<[u32; 4]>,
    ) {
        into.clear();
        into.resize(self.inputs, [0; 4]);
        for s in &self.sources {
            match s {
                Source::Position { register, lanes } => {
                    let [x, y] = centre;
                    let value = [x, y, f.depth, f.w];
                    for (lane, value) in lanes.iter().zip(value) {
                        put(into, *register, *lane as usize, value.to_bits());
                    }
                }
                Source::FrontFace { register, lane } => {
                    put(
                        into,
                        *register,
                        *lane as usize,
                        if drawn.front { u32::MAX } else { 0 },
                    );
                }
                Source::Primitive { register, lane } => put(into, *register, *lane as usize, drawn.id),
                Source::Word { register, lane, word } => put(into, *register, *lane as usize, *word),
                Source::Coverage { register, lane } => {
                    put(
                        into,
                        *register,
                        *lane as usize,
                        u32::from(covered) & state.sample_mask,
                    );
                }
                Source::Varying {
                    from_register,
                    from_lanes,
                    to_register,
                    to_lanes,
                    mode,
                    texcoord,
                    colour,
                } => {
                    let mode = if *colour && state.flat_colours {
                        InterpMode::Flat
                    } else {
                        *mode
                    };
                    for (k, to) in to_lanes.iter().enumerate() {
                        // a point sprite's texture coordinate: the pixel's place on its square
                        if let Some(place) = f.sprite.filter(|_| *texcoord)
                            && let Some(&lane) = place.get(k)
                        {
                            put(into, *to_register, *to as usize, lane.to_bits());
                            continue;
                        }
                        let Some(from) = from_lanes.get(k) else { continue };
                        let word = |v: &Shaded| {
                            v.outputs
                                .get(*from_register)
                                .and_then(|r| r.get(*from as usize))
                                .copied()
                                .unwrap_or(0)
                        };
                        // the primitive's own vertices only: a line's third weight and a point's others weigh nothing
                        let blend = |weights: [f64; 3]| {
                            let v: f64 = weights
                                .iter()
                                .zip(drawn.original)
                                .map(|(weight, vertex)| weight * f64::from(f32::from_bits(word(vertex))))
                                .sum();
                            crate::narrow(v).to_bits()
                        };
                        let word = match (mode, drawn.original.first()) {
                            (_, None) => 0,
                            (InterpMode::Flat, Some(provoking)) => word(provoking),
                            // a point's values are its vertex's, whatever the interpolation
                            (_, Some(only)) if drawn.original.len() == 1 => word(only),
                            (InterpMode::Perspective, _) => blend(f.perspective),
                            (InterpMode::Linear, _) => blend(f.linear),
                        };
                        put(into, *to_register, *to as usize, word);
                    }
                }
            }
        }
    }
}

/// The w of each of a primitive's vertices, for its screen weights (1 past its last, which weighs nothing).
trait MapW {
    fn map_w(&self) -> [f64; 3];
}

impl MapW for [&Shaded] {
    fn map_w(&self) -> [f64; 3] {
        [0, 1, 2].map(|k| self.get(k).map_or(1., |v| f64::from(v.position[3])))
    }
}

impl<'a> DrawShader<'a> {
    /// Link `vertex`'s outputs to `pixel`'s inputs by semantic.
    pub fn new(vertex: &'a ProgramShader, pixel: &'a ProgramShader) -> Result<Self, Error> {
        let from = |_, out: &Varying| {
            let from = out.provenance.d3d.ok_or_else(|| {
                error(format!(
                    "vertex output {}{} has no register",
                    out.label.name, out.label.index
                ))
            })?;
            Ok((from.register as usize, lanes_of(from.mask)))
        };
        let outputs = &vertex.interface().outputs;
        let raster = Raster::new(outputs, pixel, from)?;
        let values = RasterValues::of(outputs, from)?;
        Ok(Self {
            vertex,
            raster,
            values,
        })
    }

    /// Draw `primitives` (from [`Primitive::assemble`]) of `vertices` (each the vertex program's input registers) into
    /// `surface`, in order.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        vertices: &[Vec<[u32; 4]>],
        primitives: &[Primitive],
        vertex_buffers: &[(u32, &[[u32; 4]])],
        pixel_buffers: &[(u32, &[[u32; 4]])],
        sampler: &mut dyn Sampler,
        state: &Rasterizer,
        surface: &mut Surface,
    ) -> Result<DrawStats, Error> {
        Raster::check(state, surface)?;
        // the vertex program once per vertex a primitive names
        let mut shaded: Vec<Option<Shaded>> = vec![None; vertices.len()];
        let mut stats = DrawStats::default();
        for p in primitives {
            for &i in p.indices() {
                let input = vertices.get(i).ok_or_else(|| {
                    error(format!(
                        "index {i} names a vertex past the {} given",
                        vertices.len()
                    ))
                })?;
                let entry = shaded
                    .get_mut(i)
                    .ok_or_else(|| error(format!("vertex {i} has no room for its shading")))?;
                if entry.is_none() {
                    let (e, position) = self.vertex.execute_vertex(input, vertex_buffers, sampler)?;
                    *entry = Some(self.values.shaded(position.map(f32::from_bits), e.outputs));
                }
            }
            let original = p
                .indices()
                .iter()
                .map(|&i| shaded.get(i).and_then(Option::as_ref))
                .collect::<Option<Vec<&Shaded>>>()
                .ok_or_else(|| error("a vertex of a primitive was not shaded"))?;
            self.raster.primitive(
                &original,
                p.id,
                pixel_buffers,
                sampler,
                state,
                surface,
                &mut stats,
            )?;
        }
        Ok(stats)
    }
}

impl<'a> ShadedDraw<'a> {
    /// Link `pixel`'s inputs by semantic to `outputs`, the output varyings of the stage before it (in the order of
    /// that program's interface: the registers of the vertices it makes). One of them must be the clip-space
    /// position.
    pub fn new(outputs: &'a [Varying], pixel: &'a ProgramShader) -> Result<Self, Error> {
        let position = outputs
            .iter()
            .position(|o| o.system == Some(SystemValue::ClipPosition))
            .ok_or_else(|| error("the stage before the pixel program writes no position"))?;
        let from = |at, out: &Varying| Ok((at, (0..out.value.width.min(4)).collect()));
        let raster = Raster::new(outputs, pixel, from)?;
        let values = RasterValues::of(outputs, from)?;
        Ok(Self {
            raster,
            position,
            values,
        })
    }

    /// Draw `primitives` (their corners indices into `vertices`, each one register per output varying) into
    /// `surface`, in order.
    pub fn draw(
        &self,
        vertices: &[Vec<[u32; 4]>],
        primitives: &[Primitive],
        pixel_buffers: &[(u32, &[[u32; 4]])],
        sampler: &mut dyn Sampler,
        state: &Rasterizer,
        surface: &mut Surface,
    ) -> Result<DrawStats, Error> {
        Raster::check(state, surface)?;
        let shaded: Vec<Shaded> = vertices
            .iter()
            .map(|v| {
                let position = v
                    .get(self.position)
                    .copied()
                    .unwrap_or([0; 4])
                    .map(f32::from_bits);
                self.values.shaded(position, v.clone())
            })
            .collect();
        let mut stats = DrawStats::default();
        for p in primitives {
            let original = p
                .indices()
                .iter()
                .map(|&i| {
                    shaded.get(i).ok_or_else(|| {
                        error(format!(
                            "a primitive names vertex {i}, past the {} given",
                            shaded.len()
                        ))
                    })
                })
                .collect::<Result<Vec<&Shaded>, Error>>()?;
            self.raster.primitive(
                &original,
                p.id,
                pixel_buffers,
                sampler,
                state,
                surface,
                &mut stats,
            )?;
        }
        Ok(stats)
    }
}

/// What covers a primitive's pixels: a triangle of its clipped polygon (set up, its corners, and the w of the
/// primitive's vertices), its clipped line, or its point.
enum Cover<'c> {
    Triangle(&'c Setup, [&'c ClipVertex; 3], [f64; 3]),
    Line(&'c Line, &'c [ClipVertex; 2], [f64; 3]),
    Point(&'c Point),
}

impl Cover<'_> {
    /// The pixels of a `w` x `h` target that can be covered, as x and y ranges; `None` when there are none.
    fn bounds(&self, w: i64, h: i64) -> Option<([i64; 2], [i64; 2])> {
        let fixed: &[[i64; 2]] = match self {
            Cover::Triangle(setup, ..) => &setup.fixed,
            Cover::Line(line, ..) => &line.fixed,
            Cover::Point(point) => std::slice::from_ref(&point.fixed),
        };
        // a line's pixels reach half a pixel past its positions, and a point's half its side
        let reach = match self {
            Cover::Triangle(..) => 0,
            Cover::Line(..) => 128,
            Cover::Point(point) => point.half,
        };
        bounds_of(fixed, reach, w, h)
    }

    /// The pixel `(px, py)`: whether it is covered, its depth (before the clamp to the viewport's range) and the
    /// weights of its inputs on the primitive's vertices (extrapolated for a pixel the primitive does not cover).
    fn fragment(&self, px: i64, py: i64) -> Fragment {
        match self {
            Cover::Triangle(setup, tri, corner_w) => {
                let (b, covered) = setup.weights(px, py);
                let z = b[0] * setup.screen[0][2] + b[1] * setup.screen[1][2] + b[2] * setup.screen[2][2];
                on_primitive(covered, crate::narrow(z), b, setup.inv_w, *tri, *corner_w)
            }
            Cover::Line(line, ends, corner_w) => {
                let (t, covered) = line.at(px, py);
                let b = [1. - t, t, 0.];
                let z = b[0] * line.screen[0][2] + b[1] * line.screen[1][2];
                let inv_w = [line.inv_w[0], line.inv_w[1], 0.];
                let [first, second] = ends;
                // a third corner that weighs nothing
                on_primitive(
                    covered,
                    crate::narrow(z),
                    b,
                    inv_w,
                    [first, second, second],
                    *corner_w,
                )
            }
            Cover::Point(point) => Fragment {
                covered: point.covers(px, py),
                depth: point.depth,
                perspective: [1., 0., 0.],
                linear: [1., 0., 0.],
                w: point.w,
                sprite: point.sprite.then(|| point.place(px, py)),
            },
        }
    }
}

/// A pixel of a clipped primitive whose screen weights on its corners `corners` are `b` (each corner's 1/w in
/// `inv_w`): its inputs' weights on the drawn primitive's vertices, perspective-correct through each corner's
/// clip-space weights, screen-linear through the weights of its screen position (`corner_w`, the vertices' w).
fn on_primitive(
    covered: bool,
    depth: f32,
    b: [f64; 3],
    inv_w: [f64; 3],
    corners: [&ClipVertex; 3],
    corner_w: [f64; 3],
) -> Fragment {
    // perspective-correct weights: b / w, normalised; and the interpolated w (1 / sum of b / w)
    let pw: [f64; 3] = [b[0] * inv_w[0], b[1] * inv_w[1], b[2] * inv_w[2]];
    let sum = pw[0] + pw[1] + pw[2];
    let persp = pw.map(|x| x / sum);
    let (mut perspective, mut linear) = ([0f64; 3], [0f64; 3]);
    for ((v, &persp_k), &b_k) in corners.iter().zip(&persp).zip(&b) {
        let screen = screen_weights(v.weights, corner_w);
        for (((persp_sum, linear_sum), &weight), &on_screen) in perspective
            .iter_mut()
            .zip(linear.iter_mut())
            .zip(&v.weights)
            .zip(&screen)
        {
            *persp_sum += persp_k * weight;
            *linear_sum += b_k * on_screen;
        }
    }
    Fragment {
        covered,
        depth,
        perspective,
        linear,
        w: crate::narrow(1. / sum),
        sprite: None,
    }
}

/// Set word `lane` of register `register` of a pixel program's inputs, where the program has them.
fn put(into: &mut [[u32; 4]], register: usize, lane: usize, word: u32) {
    if let Some(slot) = into.get_mut(register).and_then(|r| r.get_mut(lane)) {
        *slot = word;
    }
}

/// The 16.8 fixed point's range in 1/256 pixels: 16 signed integer bits, 8 fractional (3.2.4.1).
const FIXED: [i64; 2] = [-(1 << 23), (1 << 23) - 1];

/// A screen coordinate in 16.8 fixed point: to the nearest, a tie to even, saturating at the format's range (3.2.4.1's
/// reference conversion; WARP takes the tie to even too, on exec_warp_draw's tie cases).
#[allow(clippy::cast_possible_truncation)] // the clamp to the 16.8 range follows, and a NaN casts to 0
fn fixed(x: f64) -> i64 {
    ((x * 256.).round_ties_even() as i64).clamp(FIXED[0], FIXED[1])
}

/// Twice the signed area of `a`, `b`, `p` in fixed point: positive when `p` is right of `a -> b` (y down). With every
/// position in the 16.8 range a difference is under 2^24 and a product under 2^48.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)] // fixed-size arrays, and the range above
fn edge(a: [i64; 2], b: [i64; 2], p: [i64; 2]) -> i64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// A clip-space position through the viewport: its screen position and depth, its 1/w, and its screen position
/// snapped.
fn to_screen(clip: [f32; 4], vp: &Viewport) -> ([f64; 3], f64, [i64; 2]) {
    let [cx, cy, cz, w] = clip.map(f64::from);
    let (nx, ny, nz) = (cx / w, cy / w, cz / w);
    let x = f64::from(vp.x) + (nx + 1.) * 0.5 * f64::from(vp.width);
    let y = f64::from(vp.y) + (1. - ny) * 0.5 * f64::from(vp.height);
    let z = f64::from(vp.min_depth) + nz * f64::from(vp.max_depth - vp.min_depth);
    ([x, y, z], 1. / w, [fixed(x), fixed(y)])
}

/// A clipped triangle set up for rasterization: its positions through the viewport, snapped, and its edges.
#[derive(Debug)]
struct Setup {
    /// Screen positions (unsnapped) and depth.
    screen: [[f64; 3]; 3],
    /// The reciprocal of each vertex's w.
    inv_w: [f64; 3],
    /// The snapped positions.
    fixed: [[i64; 2]; 3],
    /// Whether the snapped triangle is clockwise on screen (y down).
    clockwise: bool,
    /// Twice the snapped triangle's area, in fixed point (positive).
    area: i64,
    /// The edges in clockwise order, as indices into the triangle.
    edges: [(usize, usize); 3],
    /// Whether each edge is a top or a left edge.
    top_left: [bool; 3],
}

impl Setup {
    /// `None` for a triangle of no snapped area, which draws nothing.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)] // fixed-size arrays indexed by 0..3, and the 16.8 range
    fn new(tri: [&ClipVertex; 3], vp: &Viewport) -> Option<Self> {
        let mut screen = [[0f64; 3]; 3];
        let mut inv_w = [0f64; 3];
        let mut fixed_at = [[0i64; 2]; 3];
        for (k, v) in tri.iter().enumerate() {
            (screen[k], inv_w[k], fixed_at[k]) = to_screen(v.clip, vp);
        }
        let area = edge(fixed_at[0], fixed_at[1], fixed_at[2]);
        if area == 0 {
            return None;
        }
        // y down: a positive area is clockwise on screen
        let clockwise = area > 0;
        let order: [usize; 3] = if clockwise { [0, 1, 2] } else { [0, 2, 1] };
        let edges: [(usize, usize); 3] = [(order[0], order[1]), (order[1], order[2]), (order[2], order[0])];
        // top-left rule for a clockwise triangle with y down: a top edge is horizontal and runs right (dx > 0); a
        // left edge runs up (dy < 0)
        let top_left: [bool; 3] = edges.map(|(a, b)| {
            let dx = fixed_at[b][0] - fixed_at[a][0];
            let dy = fixed_at[b][1] - fixed_at[a][1];
            (dy == 0 && dx > 0) || dy < 0
        });
        Some(Self {
            screen,
            inv_w,
            fixed: fixed_at,
            clockwise,
            area: area.abs(),
            edges,
            top_left,
        })
    }

    /// The pixels of a `w` x `h` target the triangle's bounding box holds, as x and y ranges; `None` when it holds
    /// none.
    #[cfg(test)]
    fn bounds(&self, w: i64, h: i64) -> Option<([i64; 2], [i64; 2])> {
        bounds_of(&self.fixed, 0, w, h)
    }

    /// The barycentrics of a pixel centre from the snapped positions (the weight of each vertex of the triangle), and
    /// whether the triangle covers it.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)] // the edges' indices are 0..3; pixel and edge values stay within the 16.8 range
    fn weights(&self, px: i64, py: i64) -> ([f64; 3], bool) {
        let p = [px * 256 + 128, py * 256 + 128];
        let mut inside = true;
        let mut b = [0f64; 3];
        for (e, &(a, c)) in self.edges.iter().enumerate() {
            let value = edge(self.fixed[a], self.fixed[c], p);
            if value < 0 || (value == 0 && !self.top_left[e]) {
                inside = false;
            }
            // the edge a -> c weighs the vertex opposite it
            let opposite = 3 - a - c;
            b[opposite] = value as f64 / self.area as f64;
        }
        (b, inside)
    }
}

/// The pixels of a `w` x `h` target within `reach` (1/256 pixels) of the bounding box of snapped positions `fixed`, as
/// x and y ranges; `None` when there are none.
#[allow(clippy::indexing_slicing)] // an axis of 0 or 1
fn bounds_of(fixed: &[[i64; 2]], reach: i64, w: i64, h: i64) -> Option<([i64; 2], [i64; 2])> {
    let along = |axis: usize, size: i64| {
        let low = fixed
            .iter()
            .map(|p| p[axis])
            .min()
            .unwrap_or(0)
            .saturating_sub(reach)
            >> 8;
        let high = (fixed
            .iter()
            .map(|p| p[axis])
            .max()
            .unwrap_or(0)
            .saturating_add(reach)
            >> 8)
            .saturating_add(1);
        [low.clamp(0, size), high.clamp(0, size)]
    };
    let (x, y) = (along(0, w), along(1, h));
    (x[0] < x[1] && y[0] < y[1]).then_some((x, y))
}

/// A clipped line set up for rasterization: its ends through the viewport, snapped.
#[derive(Debug)]
struct Line {
    /// Screen positions (unsnapped) and depth of the ends.
    screen: [[f64; 3]; 2],
    /// The reciprocal of each end's w.
    inv_w: [f64; 2],
    /// The snapped ends.
    fixed: [[i64; 2]; 2],
    /// The snapped direction, end less start.
    d: [i64; 2],
    /// The major axis: 1 (y) when the line is more vertical than horizontal, else 0 (x).
    major: usize,
}

impl Line {
    /// `None` for a line of no snapped length, which draws nothing.
    #[allow(clippy::arithmetic_side_effects)] // differences of 16.8 values
    fn new(ends: &[ClipVertex; 2], vp: &Viewport) -> Option<Self> {
        let [(s0, w0, f0), (s1, w1, f1)] = [to_screen(ends[0].clip, vp), to_screen(ends[1].clip, vp)];
        let d = [f1[0] - f0[0], f1[1] - f0[1]];
        if d == [0, 0] {
            return None;
        }
        Some(Self {
            screen: [s0, s1],
            inv_w: [w0, w1],
            fixed: [f0, f1],
            d,
            major: usize::from(d[1].abs() > d[0].abs()),
        })
    }

    /// Where pixel `(px, py)`'s centre is along the line's major axis (0 at its start, 1 at its end), and whether the
    /// line draws the pixel.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)] // axes 0 and 1; pixel values in the 16.8 range
    fn at(&self, px: i64, py: i64) -> (f64, bool) {
        let c = [px * 256 + 128, py * 256 + 128];
        let m = self.major;
        let t = (c[m] - self.fixed[0][m]) as f64 / self.d[m] as f64;
        (t, self.draws(c))
    }

    /// Whether the line draws the pixel of centre `c` (1/256 pixels): it meets the pixel's diamond's test area and
    /// does not end inside it.
    #[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)] // axes 0 and 1; i128 products of 16.8 values
    fn draws(&self, c: [i64; 2]) -> bool {
        let [p0, p1] = self.fixed;
        let wide = |v: i64| i128::from(v);
        let u0 = [wide(p0[0] - c[0]), wide(p0[1] - c[1])];
        let u1 = [wide(p1[0] - c[0]), wide(p1[1] - c[1])];
        let d = self.d.map(wide);
        let y_major = self.major == 1;
        meets(u0, d, y_major) && !in_test_area(u1, 1, y_major)
    }
}

/// Whether the point `n / q` (1/256 pixels from a pixel centre, `q > 0`) is in the diamond's test area: inside
/// `|x| + |y| < 128`, or on its lower-left or lower-right edge or bottom corner (y > 0, y down), or for a y-major line
/// its right corner.
#[allow(clippy::arithmetic_side_effects)] // magnitudes under 2^100
fn in_test_area(n: [i128; 2], q: i128, y_major: bool) -> bool {
    let (l1, r) = (n[0].abs() + n[1].abs(), 128 * q);
    l1 < r || (l1 == r && (n[1] > 0 || (y_major && n[1] == 0 && n[0] > 0)))
}

/// Whether the segment from `u0` along `d` (t from 0 to 1, 1/256 pixels from a pixel centre) meets the diamond's test
/// area. The segment's points in the closed diamond are the t where all four `a x + b y <= 128` (a, b = +-1) hold, an
/// interval: its middle is inside the diamond unless the interval is a point or runs along an edge, which its ends and
/// middle then decide.
#[allow(clippy::arithmetic_side_effects)] // u0 and d under 2^26, so every product here is under 2^110
fn meets(u0: [i128; 2], d: [i128; 2], y_major: bool) -> bool {
    // the interval's ends as fractions (numerator, positive denominator)
    let (mut lo, mut hi) = ((0i128, 1i128), (1i128, 1i128));
    for (a, b) in [(1, 1), (-1, 1), (1, -1), (-1, -1)] {
        let g = a * u0[0] + b * u0[1];
        let h = a * d[0] + b * d[1];
        if h == 0 {
            if g > 128 {
                return false;
            }
            continue;
        }
        // g + t h <= 128
        if h > 0 {
            let t = (128 - g, h);
            if t.0 * hi.1 < hi.0 * t.1 {
                hi = t;
            }
        } else {
            let t = (g - 128, -h);
            if t.0 * lo.1 > lo.0 * t.1 {
                lo = t;
            }
        }
    }
    if lo.0 * hi.1 > hi.0 * lo.1 {
        return false;
    }
    let point = |(n, q): (i128, i128)| [u0[0] * q + n * d[0], u0[1] * q + n * d[1]];
    let middle = (lo.0 * hi.1 + hi.0 * lo.1, 2 * lo.1 * hi.1);
    [lo, hi, middle]
        .into_iter()
        .any(|t| in_test_area(point(t), t.1, y_major))
}

/// A point set up for rasterization: its position through the viewport, snapped, its depth and its w.
#[derive(Debug)]
struct Point {
    fixed: [i64; 2],
    /// Half the square's side, in 1/256 of a pixel.
    half: i64,
    depth: f32,
    w: f32,
    /// Whether it is a point sprite.
    sprite: bool,
}

impl Point {
    /// A point whose square's side is `size` pixels, a point sprite where `sprite`; `None` for a point outside the
    /// clip volume, which draws nothing.
    fn new(v: &ClipVertex, vp: &Viewport, size: f32, sprite: bool) -> Option<Self> {
        if (0..6).any(|k| plane(k, &v.clip) < 0.) || v.clip[3] == 0. {
            return None;
        }
        let (screen, _, at) = to_screen(v.clip, vp);
        Some(Self {
            fixed: at,
            // half the side in 16.8 fixed point, as a position is
            half: fixed(f64::from(size) / 2.).max(0),
            depth: crate::narrow(screen[2]),
            w: v.clip[3],
            sprite,
        })
    }

    /// Whether the point's square, `[x - s/2, x + s/2) x [y - s/2, y + s/2)` about its snapped position, holds the
    /// centre of pixel `(px, py)`.
    #[allow(clippy::arithmetic_side_effects)] // pixel values in the 16.8 range
    fn covers(&self, px: i64, py: i64) -> bool {
        let c = [px * 256 + 128, py * 256 + 128];
        (0..2).all(|k| {
            let (at, p) = (
                c.get(k).copied().unwrap_or(0),
                self.fixed.get(k).copied().unwrap_or(0),
            );
            p - self.half <= at && at < p + self.half
        })
    }

    /// The centre of pixel `(px, py)`'s place on the point's square: (0, 0) at its top left, (1, 1) at its bottom
    /// right.
    #[allow(clippy::arithmetic_side_effects)] // pixel values in the 16.8 range
    fn place(&self, px: i64, py: i64) -> [f32; 2] {
        let side = (2 * self.half).max(1) as f64;
        let c = [px * 256 + 128, py * 256 + 128];
        [0, 1].map(|k| {
            let (at, p) = (
                c.get(k).copied().unwrap_or(0),
                self.fixed.get(k).copied().unwrap_or(0),
            );
            crate::narrow((at - (p - self.half)) as f64 / side)
        })
    }
}

/// The signed distance of a clip-space position from plane `k` of the clip volume (`0 <= z <= w`, `-w <= x <= w`,
/// `-w <= y <= w`): kept where >= 0.
#[allow(clippy::indexing_slicing)] // the four components of a clip position
fn plane(k: usize, c: &[f32; 4]) -> f64 {
    let (x, y, z, w) = (f64::from(c[0]), f64::from(c[1]), f64::from(c[2]), f64::from(c[3]));
    match k {
        0 => z,
        1 => w - z,
        2 => w + x,
        3 => w - x,
        4 => w + y,
        _ => w - y,
    }
}

/// The distance of a point with clip-space weights `weights` from the plane of a clip distance whose values at the
/// primitive's vertices are `at`: kept where >= 0.
fn user_plane(at: &[f64; 3], weights: &[f64; 3]) -> f64 {
    at[0] * weights[0] + at[1] * weights[1] + at[2] * weights[2]
}

/// Clip a triangle to `0 <= z <= w`, the viewport (`-w <= x, y <= w`) and the planes of its clip distances (`user`,
/// each one's values at the triangle's vertices): the polygon left, as clip-space vertices with their weights on the
/// triangle's (an empty list when nothing is left). A polygon with a vertex at the clip-space
/// origin (w = 0, so x, y and z are 0 too: the planes keep `w >= |x|, |y|` and `w >= z >= 0`) is empty: a point of a
/// triangle is the zero vector only when its vertices are linearly dependent, so the triangle's plane holds the camera
/// and it has no area on screen; it has no screen position to divide out there.
#[allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)] // the corners are indexed by a loop over the polygon
fn clip(triangle: Vec<ClipVertex>, user: &[[f64; 3]]) -> Vec<ClipVertex> {
    let mut polygon = triangle;
    let distance = |k: usize, v: &ClipVertex| match k.checked_sub(6) {
        None => plane(k, &v.clip),
        Some(u) => user_plane(&user[u], &v.weights),
    };
    for k in 0..6 + user.len() {
        if polygon.iter().all(|v| distance(k, v) >= 0.) {
            continue;
        }
        let mut out = Vec::new();
        for i in 0..polygon.len() {
            let a = &polygon[i];
            let b = &polygon[(i + 1) % polygon.len()];
            let (da, db) = (distance(k, a), distance(k, b));
            if da >= 0. {
                out.push(a.clone());
            }
            if (da >= 0.) != (db >= 0.) {
                let t = da / (da - db);
                out.push(blend(a, b, t));
            }
        }
        polygon = out;
        if polygon.len() < 3 {
            return polygon;
        }
    }
    if polygon.iter().any(|v| v.clip[3] == 0.) {
        return Vec::new();
    }
    polygon
}

/// Clip a line to `0 <= z <= w` and, in x and y, to a guard band about the viewport that keeps its positions on screen
/// within 2^14 pixels (well inside the 16.8 range), not to the viewport: 3.4.3.1 allows either, and the GPU
/// `exec_warp_draw` sets the executor against draws a line unclipped. A line inside the band keeps its own ends, and an
/// end the band cuts is off every pixel of the viewport by more than a diamond's reach; and to the planes of its clip
/// distances (`user`, each one's values at the line's ends). Its ends, as clip-space vertices with their weights on the
/// line's; `None` when nothing is left, or when an end is at the clip-space origin (the line runs through the camera, a
/// point on screen).
#[allow(clippy::indexing_slicing)] // the four components of a clip position, and the clip distances by plane
fn clip_line(line: Vec<ClipVertex>, vp: &Viewport, user: &[[f64; 3]]) -> Option<[ClipVertex; 2]> {
    let [a, b]: [ClipVertex; 2] = line.try_into().ok()?;
    // |x| <= g w keeps x on screen within |X| + (g + 1) / 2 W
    let band = |origin: f32, size: f32| {
        let size = f64::from(size).abs().max(1.);
        (2. * (16384. - f64::from(origin).abs()) / size - 1.).max(1.)
    };
    let g = band(vp.x, vp.width).min(band(vp.y, vp.height));
    let guarded = |k: usize, v: &ClipVertex| {
        let c = &v.clip;
        let (x, y, w) = (f64::from(c[0]), f64::from(c[1]), f64::from(c[3]));
        match k {
            0 | 1 => plane(k, c),
            2 => g * w + x,
            3 => g * w - x,
            4 => g * w + y,
            _ => g * w - y,
        }
    };
    let (mut t0, mut t1) = (0f64, 1f64);
    let distances = (0..6).map(|k| (guarded(k, &a), guarded(k, &b))).chain(
        user.iter()
            .map(|u| (user_plane(u, &a.weights), user_plane(u, &b.weights))),
    );
    for (da, db) in distances {
        if da < 0. && db < 0. {
            return None;
        }
        if da < 0. {
            t0 = t0.max(da / (da - db));
        } else if db < 0. {
            t1 = t1.min(da / (da - db));
        }
    }
    if t0 > t1 {
        return None;
    }
    let start = if t0 > 0. { blend(&a, &b, t0) } else { a.clone() };
    let end = if t1 < 1. { blend(&a, &b, t1) } else { b };
    (start.clip[3] != 0. && end.clip[3] != 0.).then_some([start, end])
}

/// The point at `t` from `a` to `b` in clip space, its weights blended alike.
#[allow(clippy::indexing_slicing, clippy::cast_possible_truncation)] // fixed-size arrays indexed by 0..4, and a clip position is single precision
fn blend(a: &ClipVertex, b: &ClipVertex, t: f64) -> ClipVertex {
    let clip = [0, 1, 2, 3]
        .map(|k| (f64::from(a.clip[k]) + (f64::from(b.clip[k]) - f64::from(a.clip[k])) * t) as f32);
    let weights = [0, 1, 2].map(|i| a.weights[i] + (b.weights[i] - a.weights[i]) * t);
    ClipVertex { clip, weights }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: i64 = 16;
    const VIEWPORT: Viewport = Viewport {
        x: 0.,
        y: 0.,
        width: SIZE as f32,
        height: SIZE as f32,
        min_depth: 0.,
        max_depth: 1.,
    };

    /// A clip-space position at pixel coordinates (x right, y down), depth z, w.
    fn at(x: f32, y: f32, z: f32, w: f32) -> [f32; 4] {
        let s = SIZE as f32;
        [(x / s * 2. - 1.) * w, (1. - y / s * 2.) * w, z * w, w]
    }

    fn clipped(corners: [[f32; 4]; 3]) -> Vec<ClipVertex> {
        let mut k = 0;
        let triangle = corners.map(|clip| {
            let mut weights = [0.; 3];
            weights[k] = 1.;
            k += 1;
            ClipVertex { clip, weights }
        });
        clip(triangle.to_vec(), &[])
    }

    /// A clip-space position on screen, in pixels of the target.
    fn on_screen(c: [f64; 4]) -> [f64; 2] {
        [
            (c[0] / c[3] + 1.) * 0.5 * SIZE as f64,
            (1. - c[1] / c[3]) * 0.5 * SIZE as f64,
        ]
    }

    /// How many times each pixel of the target is covered by the triangles of the clipped polygon.
    fn coverage(polygon: &[ClipVertex]) -> Vec<u32> {
        let mut count = vec![0; (SIZE * SIZE) as usize];
        for k in 1..polygon.len().saturating_sub(1) {
            let Some(setup) = Setup::new([&polygon[0], &polygon[k], &polygon[k + 1]], &VIEWPORT) else {
                continue;
            };
            let Some(([x0, x1], [y0, y1])) = setup.bounds(SIZE, SIZE) else {
                continue;
            };
            for py in y0..y1 {
                for px in x0..x1 {
                    if setup.weights(px, py).1 {
                        count[(py * SIZE + px) as usize] += 1;
                    }
                }
            }
        }
        count
    }

    #[test]
    fn a_vertex_near_the_camera_plane_is_clipped_to_the_viewport() {
        // c, at w = 1e-7, is about 8e9 pixels right of the target (and a fifth of that up), past what 16.8 fixed point
        // holds and what the edge function's products hold in i64; the viewport's sides cut its edges inside the
        // target. The triangle seen from the target is a, b and the point at infinity along (5, -1): the centres with
        // 42.25 < x + 5 y < 78.25 right of a -> b (x - y > -5.75). No pixel centre is within 0.04 pixels of its
        // edges, and snapping the cut vertices moves an edge by under 1/256 of a pixel.
        let a = at(2.25, 8., 0.5, 1.);
        let b = at(8.25, 14., 0.5, 1.);
        let c = [1e3, 2e2, 0., 1e-7];
        let polygon = clipped([a, b, c]);
        let expected: Vec<u32> = (0..SIZE * SIZE)
            .map(|i| {
                let (x, y) = ((i % SIZE) as f64 + 0.5, (i / SIZE) as f64 + 0.5);
                u32::from(x + 5. * y > 42.25 && x + 5. * y < 78.25 && x - y > -5.75)
            })
            .collect();
        assert_eq!(coverage(&polygon), expected);
        for v in &polygon {
            for screen in on_screen(v.clip.map(f64::from)) {
                assert!(
                    (-1e-3..=SIZE as f64 + 1e-3).contains(&screen),
                    "{screen} past the viewport"
                );
            }
        }
    }

    /// A vertex clipping makes is the blend of the triangle's vertices by its weights in clip space, and of their
    /// screen positions by its screen weights ([`screen_weights`]): what a perspective-correct and a screen-linear
    /// output of it are blended by. Where the ends of a cut edge have different w the two differ: the viewport's right
    /// side cuts both edges to the far vertex (w = 1e-7) about 1e-3 of the way along in clip space, and its weight on
    /// screen is 1e-7 times that (the ratio of the ends' w).
    #[test]
    fn a_cut_vertex_holds_its_weights_in_clip_space_and_on_screen() {
        let corners = [
            at(2.25, 8., 0.5, 1.),
            at(8.25, 14., 0.5, 1.),
            [1e3, 2e2, 0., 1e-7],
        ];
        let polygon = clipped(corners);
        let corner_w = corners.map(|c| f64::from(c[3]));
        let mut cut = 0;
        for v in &polygon {
            let in_clip: [f64; 4] =
                std::array::from_fn(|k| (0..3).map(|i| v.weights[i] * f64::from(corners[i][k])).sum());
            for (k, (blended, kept)) in in_clip.iter().zip(v.clip).enumerate() {
                let kept = f64::from(kept);
                assert!(
                    (blended - kept).abs() <= 1e-6 * kept.abs().max(1.),
                    "clip lane {k}: {blended} from the weights, {kept} kept"
                );
            }
            let screen = screen_weights(v.weights, corner_w);
            let far = on_screen(corners[2].map(f64::from));
            let near = [0, 1].map(|i| on_screen(corners[i].map(f64::from)));
            let at = on_screen(v.clip.map(f64::from));
            for axis in 0..2 {
                let blended = screen[0] * near[0][axis] + screen[1] * near[1][axis] + screen[2] * far[axis];
                assert!(
                    (blended - at[axis]).abs() <= 1e-3,
                    "screen axis {axis}: {blended} from the weights, {} kept",
                    at[axis]
                );
            }
            if v.weights[2] > 0. && v.weights[2] < 1. {
                cut += 1;
                assert!(
                    v.weights[2] < 1e-2 && screen[2] < 1e-6 * v.weights[2],
                    "{:?} {screen:?}",
                    v.weights
                );
            }
        }
        assert_eq!(cut, 2, "{polygon:?}");
    }

    /// A vertex inside the viewport nearer than any fixed w is kept, as Direct3D's clip volume keeps it: its screen
    /// position is a division by its w, which the viewport's planes keep within the target.
    #[test]
    fn a_vertex_at_a_tiny_w_inside_the_viewport_is_kept() {
        let near = [1e-8, -1e-8, 0., 1e-7];
        let polygon = clipped([at(2., 2., 0.5, 1.), at(14., 3., 0.5, 1.), near]);
        let kept: Vec<[u32; 4]> = polygon.iter().map(|v| v.clip.map(f32::to_bits)).collect();
        assert!(kept.contains(&near.map(f32::to_bits)), "{polygon:?}");
        assert!(coverage(&polygon).iter().any(|&n| n > 0));
    }

    /// A triangle through the clip-space origin (its vertices linearly dependent: `c = -(a + b) / 2` here) is a line on
    /// screen; clipping leaves the origin a vertex of it, and it draws nothing.
    #[test]
    fn a_triangle_through_the_clip_space_origin_draws_nothing() {
        let a = at(4., 4., 0.5, 1.);
        let b = at(12., 6., 0.5, 1.);
        let c = [0, 1, 2, 3].map(|k| -(a[k] + b[k]) / 2.);
        assert!(clipped([a, b, c]).is_empty());
    }

    #[test]
    fn a_triangle_inside_the_viewport_is_not_clipped() {
        let corners = [
            at(1., 14., 0.5, 1.),
            at(8., 1.5, 0.5, 3.),
            at(15., 13., 0.25, 0.5),
        ];
        let polygon = clipped(corners);
        let kept: Vec<[u32; 4]> = polygon.iter().map(|v| v.clip.map(f32::to_bits)).collect();
        assert_eq!(kept, corners.map(|c| c.map(f32::to_bits)));
    }

    #[test]
    fn an_infinite_position_is_clipped_away_or_not_a_number() {
        let polygon = clipped([
            at(2., 2., 0.5, 1.),
            [f32::INFINITY, 0., 0.5, 1.],
            at(2., 12., 0.5, 1.),
        ]);
        assert!(polygon.len() < 3 || polygon.iter().any(|v| v.clip.iter().any(|c| c.is_nan())));
    }

    #[test]
    fn snapping_rounds_to_the_nearest_a_tie_to_even_and_saturates() {
        assert_eq!(fixed(100.25), 25664);
        assert_eq!(fixed(1. / 1024.), 0);
        assert_eq!(fixed(3. / 1024.), 1);
        // a tie goes to the even step, either way from zero
        assert_eq!(fixed(1152.5 / 256.), 1152);
        assert_eq!(fixed(1153.5 / 256.), 1154);
        assert_eq!(fixed(-1152.5 / 256.), -1152);
        assert_eq!(fixed(1. / 512.), 0);
        assert_eq!(fixed(1e30), FIXED[1]);
        assert_eq!(fixed(-1e30), FIXED[0]);
        assert_eq!(fixed(f64::INFINITY), FIXED[1]);
        assert_eq!(fixed(f64::NAN), 0);
    }
}
