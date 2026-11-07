#![allow(clippy::unwrap_used)]
//! The draw stage ([`iridia_exec::draw`]) against Direct3D 12: each case draws the same points, lines or triangles
//! through the same vertex and pixel programs (tests/fixtures/draw.hlsl, lifted from their DXIL) into 16 x 16 targets
//! on both, and every pixel is compared: which pixels each draws (exactly), what it writes (to a stated tolerance: the
//! interpolation's arithmetic is each implementation's own) and the depth. Triangles draw on Microsoft's WARP
//! rasterizer; points and lines on the machine's GPU (the oracle's `adapter hardware`), as WARP draws lines otherwise
//! than the specification and the GPU do (`warp_draws_otherwise_only_as_documented`).
//!
//! Needs the WARP oracle program with the `d3d12draw` case form (the configured `Tool::Warp`; the form is in
//! `oracles/warp-draw`; points and lines need its `topology` and `adapter` lines, primase-oracles fa44129) and the
//! scratch folder. Without them the live test does nothing. Which pixels points and lines draw is also held to the
//! GPU's and WARP's recorded answers for 11,240 of them (tests/fixtures/warp_raster_hardware.txt and
//! warp_raster_warp.txt, made by tests/oracles/warp_raster.py), which need neither.
mod inputs;

use inputs::{Tool, own_scratch, program};
use iridia_exec::draw::{
    Compare, Cull, DepthState, DrawShader, Primitive, Rasterizer, ShadedDraw, Surface, Viewport,
};
use iridia_exec::geometry::Topology;
use iridia_exec::{PixelShader, ProgramShader, Sample};
use std::path::Path;

const SIZE: u32 = 16;
/// What every target is cleared to: no pixel program here writes it.
const CLEAR: f32 = -7.0;
/// The interpolated values' tolerance: relative to the larger magnitude, with a floor for values near 0.
const TOLERANCE: f32 = 2e-5;

/// One vertex: position (x, y, z, w in clip space), colour, uv, id (the four inputs of `vs_main`, 14 floats).
type Vertex = [f32; 14];

fn v(pos: [f32; 4], col: [f32; 4], uv: [f32; 2], id: f32) -> Vertex {
    [
        pos[0],
        pos[1],
        pos[2],
        pos[3],
        col[0],
        col[1],
        col[2],
        col[3],
        uv[0],
        uv[1],
        id,
        id + 0.5,
        0.0,
        1.0,
    ]
}

/// A clip-space position at pixel coordinates (x right, y down, over the 16 x 16 targets), depth z, w.
fn at(x: f32, y: f32, z: f32, w: f32) -> [f32; 4] {
    let s = SIZE as f32;
    [(x / s * 2.0 - 1.0) * w, (1.0 - y / s * 2.0) * w, z * w, w]
}

struct DrawCase {
    name: &'static str,
    /// The vertex program, `draw_<vs>.dxil`.
    vs: &'static str,
    ps: &'static str,
    vertices: Vec<Vertex>,
    topology: Topology,
    indices: Vec<u32>,
    cull: Cull,
    ccw: bool,
    depth: Option<(Compare, bool, f32)>,
    targets: Vec<u32>,
}

fn compare_name(c: Compare) -> &'static str {
    match c {
        Compare::Never => "never",
        Compare::Less => "less",
        Compare::Equal => "equal",
        Compare::LessEqual => "lessequal",
        Compare::Greater => "greater",
        Compare::NotEqual => "notequal",
        Compare::GreaterEqual => "greaterequal",
        Compare::Always => "always",
    }
}

fn cases() -> Vec<DrawCase> {
    let red = [1.0, 0.0, 0.0, 1.0];
    let green = [0.0, 1.0, 0.0, 1.0];
    let blue = [0.0, 0.0, 1.0, 1.0];
    // a triangle whose vertices have different w: perspective-correct and linear interpolation differ
    let perspective = vec![
        v(at(1.0, 14.0, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v(at(8.0, 1.5, 0.5, 3.0), green, [0.5, 1.0], 2.0),
        v(at(15.0, 13.0, 0.25, 0.5), blue, [1.0, 0.0], 3.0),
    ];
    // a square of two triangles sharing a diagonal, edges on pixel centres and between them, and a triangle with a
    // horizontal top edge on a row of centres: the top-left rule decides every shared and on-centre pixel
    let edges = vec![
        v(at(2.0, 2.0, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v(at(10.0, 2.0, 0.5, 1.0), green, [1.0, 0.0], 1.0),
        v(at(2.0, 10.0, 0.5, 1.0), blue, [0.0, 1.0], 1.0),
        v(at(10.0, 10.0, 0.5, 1.0), red, [1.0, 1.0], 2.0),
        v(at(10.5, 3.5, 0.5, 1.0), green, [0.0, 0.0], 3.0),
        v(at(15.5, 3.5, 0.5, 1.0), blue, [1.0, 0.0], 3.0),
        v(at(13.0, 13.5, 0.5, 1.0), red, [0.5, 1.0], 3.0),
    ];
    // two clockwise triangles and one counter-clockwise, at three depths
    let layered = vec![
        v(at(1.0, 1.0, 0.7, 1.0), red, [0.0, 0.0], 1.0),
        v(at(15.0, 1.0, 0.7, 1.0), red, [1.0, 0.0], 1.0),
        v(at(1.0, 15.0, 0.7, 1.0), red, [0.0, 1.0], 1.0),
        v(at(3.0, 3.0, 0.3, 2.0), green, [0.0, 0.0], 2.0),
        v(at(15.0, 8.0, 0.3, 2.0), green, [1.0, 0.0], 2.0),
        v(at(5.0, 15.0, 0.3, 2.0), green, [0.0, 1.0], 2.0),
        v(at(0.0, 8.0, 0.9, 1.0), blue, [0.0, 0.0], 3.0),
        v(at(16.0, 16.0, 0.9, 1.0), blue, [1.0, 0.0], 3.0),
        v(at(16.0, 0.0, 0.9, 1.0), blue, [0.0, 1.0], 3.0),
    ];
    // a triangle with one vertex behind the near plane (z < 0): clipped to a quadrilateral
    let near = vec![
        v(at(2.0, 14.0, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v([0.2, 0.9, -0.5, 1.0], green, [0.5, 1.0], 1.0),
        v(at(14.0, 12.0, 0.5, 1.0), blue, [1.0, 0.0], 1.0),
    ];
    // a triangle with a vertex near the camera plane (w = 1e-7): cut by the w plane, it is about 8e9 pixels right of
    // the targets, so x and y are clipped to the guard band before the positions are snapped
    let near_w = vec![
        v(at(2.25, 8.0, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v(at(8.25, 14.0, 0.5, 1.0), green, [1.0, 0.0], 1.0),
        v([1e3, 2e2, 0.0, 1e-7], blue, [0.5, 1.0], 1.0),
    ];
    // a vertical left edge and a horizontal top edge exactly halfway between two 16.8 fixed-point steps (x * 256 =
    // 1152.5, y * 256 = 896.5; the values are dyadic, so the clip coordinates are exact in f32): rounded half away from
    // zero the edge is 1/256 past the centres of column 4 (row 3) and leaves them out; rounded to even it passes
    // through them, and the top-left rule draws them. WARP draws them: the snap takes ties to even (3.2.4.1's
    // reference conversion; it allows 0.6 ULP)
    let tie = 4.5 + 1.0 / 512.0;
    let tie_x = vec![
        v(at(tie, 2.25, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v(at(12.25, 8.25, 0.5, 1.0), green, [1.0, 0.0], 1.0),
        v(at(tie, 13.75, 0.5, 1.0), blue, [0.0, 1.0], 1.0),
    ];
    let tie_y = vec![
        v(at(2.25, tie - 1.0, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v(at(13.75, tie - 1.0, 0.5, 1.0), green, [1.0, 0.0], 1.0),
        v(at(8.25, 13.25, 0.5, 1.0), blue, [0.0, 1.0], 1.0),
    ];
    // a strip of four triangles, culled by facing: unless its odd triangles swap their last two vertices, they face
    // the other way and are culled
    let strip = vec![
        v(at(1.0, 12.0, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v(at(3.0, 2.0, 0.5, 2.0), green, [0.2, 1.0], 2.0),
        v(at(6.0, 13.0, 0.4, 1.0), blue, [0.4, 0.0], 3.0),
        v(at(9.0, 3.0, 0.3, 0.5), red, [0.6, 1.0], 4.0),
        v(at(12.0, 14.0, 0.5, 1.0), green, [0.8, 0.0], 5.0),
        v(at(15.0, 1.5, 0.5, 1.0), blue, [1.0, 1.0], 6.0),
    ];
    // lines: on pixel centres and between them, both ways along an axis, 45 degrees, steep and shallow, off-centre
    // ends, w differing along it (perspective-correct and linear values differ), and clipped by the viewport and the
    // near plane
    let lines = vec![
        v(at(2.5, 1.5, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v(at(8.5, 1.5, 0.5, 1.0), green, [1.0, 1.0], 2.0),
        v(at(8.5, 3.0, 0.5, 1.0), red, [0.0, 0.0], 3.0),
        v(at(2.5, 3.0, 0.5, 1.0), blue, [1.0, 1.0], 4.0),
        v(at(10.0, 1.0, 0.5, 1.0), red, [0.0, 0.0], 5.0),
        v(at(15.0, 6.0, 0.5, 1.0), green, [1.0, 1.0], 6.0),
        v(at(1.2, 5.3, 0.5, 1.0), red, [0.0, 0.0], 7.0),
        v(at(3.7, 15.8, 0.25, 3.0), blue, [1.0, 1.0], 8.0),
        v(at(2.3, 6.7, 0.75, 0.5), green, [0.0, 0.0], 9.0),
        v(at(13.8, 9.2, 0.25, 2.0), blue, [1.0, 1.0], 10.0),
        v(at(-5.0, 12.5, 0.5, 1.0), red, [0.0, 0.0], 11.0),
        v(at(20.0, 14.5, 0.5, 1.0), green, [1.0, 1.0], 12.0),
        v([0.0, 0.5, -0.5, 1.0], red, [0.0, 0.0], 13.0),
        v(at(14.5, 11.5, 0.5, 1.0), blue, [1.0, 1.0], 14.0),
    ];
    let line_strip = vec![
        v(at(2.5, 2.5, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v(at(12.25, 4.75, 0.5, 2.0), green, [1.0, 0.0], 2.0),
        v(at(9.5, 13.5, 0.25, 1.0), blue, [1.0, 1.0], 3.0),
        v(at(3.0, 9.0, 0.75, 0.5), red, [0.0, 1.0], 4.0),
    ];
    // points: on a pixel centre, on a corner and on edges between pixels (the top-left rule), at a quarter, with w
    // not 1, half off the targets and outside the clip volume (not drawn)
    let points = vec![
        v(at(4.5, 4.5, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v(at(8.0, 2.0, 0.5, 1.0), green, [0.1, 0.2], 2.0),
        v(at(11.0, 8.5, 0.5, 1.0), blue, [0.3, 0.4], 3.0),
        v(at(2.5, 12.0, 0.5, 1.0), red, [0.5, 0.6], 4.0),
        v(at(13.25, 13.75, 0.3, 2.0), green, [0.7, 0.8], 5.0),
        v(at(-0.25, 9.5, 0.5, 1.0), blue, [0.9, 1.0], 6.0),
        v(at(16.25, 5.5, 0.5, 1.0), red, [1.0, 0.0], 7.0),
        v(at(6.5, 14.5, 1.5, 1.0), green, [0.0, 1.0], 8.0),
    ];
    // clip and cull distances (`vs_distances`): a triangle whose vertices are all below 0 on the cull distance
    // (culled), one whose vertices are not all below (drawn), and the clip distance, from the coordinate, cutting the
    // drawn ones (their vertices' w differing, so clip-space and screen-space interpolation of a distance differ)
    let distances = vec![
        v(at(1.0, 1.0, 0.5, 1.0), red, [0.0, 0.0], 1.0),
        v(at(7.0, 1.0, 0.5, 1.0), red, [1.0, 0.0], 1.0),
        v(at(1.0, 7.0, 0.5, 1.0), red, [0.0, 1.0], 2.0),
        v(at(9.0, 1.0, 0.5, 1.0), green, [0.0, 0.0], 1.0),
        v(at(15.0, 1.0, 0.5, 2.0), green, [1.0, 0.0], 3.0),
        v(at(9.0, 7.0, 0.5, 1.0), green, [0.0, 1.0], 3.0),
        v(at(1.0, 9.0, 0.5, 1.0), blue, [0.0, 0.0], 3.0),
        v(at(15.0, 9.0, 0.5, 0.5), blue, [1.0, 0.0], 3.0),
        v(at(1.0, 15.0, 0.5, 1.0), blue, [0.0, 1.0], 3.0),
    ];
    let one = |name, ps, vertices: &Vec<Vertex>, targets: Vec<u32>| DrawCase {
        name,
        vs: "vs",
        ps,
        topology: Topology::TriangleList,
        indices: (0..vertices.len() as u32).collect(),
        vertices: vertices.clone(),
        cull: Cull::None,
        ccw: false,
        depth: None,
        targets,
    };
    vec![
        one("perspective", "colour", &perspective, vec![0, 1]),
        DrawCase {
            indices: vec![0, 1, 2, 2, 1, 3, 4, 5, 6],
            ..one("edges", "colour", &edges, vec![0, 1])
        },
        DrawCase {
            depth: Some((Compare::LessEqual, true, 1.0)),
            ..one("depth", "colour", &layered, vec![0, 1])
        },
        DrawCase {
            cull: Cull::Back,
            ..one("cull", "position", &layered, vec![0, 1])
        },
        DrawCase {
            ccw: true,
            ..one("facing", "position", &layered, vec![0, 1])
        },
        one("discard", "discard", &perspective, vec![0]),
        one("near", "colour", &near, vec![0, 1]),
        one("near_w", "colour", &near_w, vec![0, 1]),
        one("tie_x", "colour", &tie_x, vec![0, 1]),
        one("tie_y", "colour", &tie_y, vec![0, 1]),
        one("derivatives", "derivatives", &perspective, vec![0, 1]),
        one("texture", "texture", &perspective, vec![0]),
        DrawCase {
            topology: Topology::TriangleStrip,
            cull: Cull::Back,
            ..one("strip", "primitive", &strip, vec![0, 1])
        },
        DrawCase {
            topology: Topology::LineList,
            ..one("lines", "colour", &lines, vec![0, 1])
        },
        DrawCase {
            topology: Topology::LineList,
            ..one("lines_position", "position", &lines, vec![0, 1])
        },
        DrawCase {
            topology: Topology::LineList,
            depth: Some((Compare::Always, true, 1.0)),
            ..one("lines_depth", "position", &lines, vec![0])
        },
        DrawCase {
            topology: Topology::LineStrip,
            ..one("line_strip", "primitive", &line_strip, vec![0, 1])
        },
        DrawCase {
            topology: Topology::PointList,
            depth: Some((Compare::Less, true, 1.0)),
            ..one("points", "colour", &points, vec![0, 1])
        },
        DrawCase {
            topology: Topology::PointList,
            ..one("points_position", "position", &points, vec![0, 1])
        },
        DrawCase {
            indices: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            ..one("primitive", "primitive", &layered, vec![0, 1])
        },
        // the pixel's input coverage and sample index: one sample a pixel
        one("coverage", "coverage", &layered, vec![0, 1]),
        DrawCase {
            vs: "vs_distances",
            ..one("distances", "colour", &distances, vec![0, 1])
        },
        DrawCase {
            vs: "vs_distances",
            topology: Topology::LineList,
            ..one("lines_distances", "colour", &lines, vec![0, 1])
        },
        DrawCase {
            vs: "vs_distances",
            topology: Topology::PointList,
            ..one("points_distances", "colour", &points, vec![0, 1])
        },
    ]
}

fn hex(x: f32) -> String {
    format!("{:08x}", x.to_bits())
}

fn case_text(c: &DrawCase, dir: &Path) -> String {
    let mut t = format!("case {}\napi d3d12draw\n", c.name);
    t += &format!("vs {}\n", dir.join(format!("draw_{}.dxil", c.vs)).display());
    t += &format!("ps {}\n", dir.join(format!("draw_{}.dxil", c.ps)).display());
    t += &format!("size {SIZE} {SIZE}\n");
    t += &format!(
        "cull {}\nfront {}\n",
        match c.cull {
            Cull::None => "none",
            Cull::Front => "front",
            Cull::Back => "back",
        },
        u8::from(!c.ccw)
    );
    t += match c.topology {
        Topology::TriangleList => "",
        Topology::PointList => "topology pointlist\n",
        Topology::LineList => "topology linelist\n",
        Topology::LineStrip => "topology linestrip\n",
        Topology::TriangleStrip => "topology trianglestrip\n",
        other => panic!("{}: the oracle draws no {other:?}", c.name),
    };
    // points and lines draw on the GPU, which draws them by the specification (WARP's lines differ)
    if matches!(
        c.topology,
        Topology::PointList | Topology::LineList | Topology::LineStrip
    ) {
        t += "adapter hardware\n";
    }
    if let Some((compare, write, clear)) = c.depth {
        t += &format!(
            "depth {} {} {}\n",
            compare_name(compare),
            u8::from(write),
            hex(clear)
        );
    }
    t += "layout POSITION 0 f32x4 0\nlayout COLOR 0 f32x4 16\nlayout TEXCOORD 0 f32x2 32\nlayout TEXCOORD 1 f32x4 40\nstride 56\n";
    for v in &c.vertices {
        t += "vertex";
        for x in v {
            t += &format!(" {}", hex(*x));
        }
        t += "\n";
    }
    t += "index";
    for i in &c.indices {
        t += &format!(" {i:x}");
    }
    t += "\n";
    t += &format!(
        "vcb 0 2\nrow {} {} {} {}\nrow {} {} {} {}\n",
        hex(1.0),
        hex(1.0),
        hex(1.0),
        hex(1.0),
        hex(0.0),
        hex(0.0),
        hex(0.0),
        hex(0.0)
    );
    if c.ps == "texture" {
        t += &format!(
            "texture 0 {} {} {} {} ramp\n",
            hex(0.0),
            hex(0.0),
            hex(0.0),
            hex(0.0)
        );
    }
    for r in &c.targets {
        t += &format!(
            "target {r} float\nclear {r} {} {} {} {}\n",
            hex(CLEAR),
            hex(CLEAR),
            hex(CLEAR),
            hex(CLEAR)
        );
    }
    t + "end\n"
}

/// The oracle's ramp texture (4 x 4, texel (x, y) is (x, y, 0, 1)) through a point, clamp sampler.
fn ramp(s: Sample) -> Result<[f32; 4], String> {
    let texel = |c: f32| (c * 4.0).floor().clamp(0.0, 3.0);
    Ok([texel(s.uv[0]), texel(s.uv[1]), 0.0, 1.0])
}

fn lift(dir: &Path, name: &str) -> ProgramShader {
    ProgramShader::new(iridia_dxil::lift(&std::fs::read(dir.join(name)).unwrap()).unwrap()).unwrap()
}

/// Each vertex's input registers, by the vertex program's signature: POSITION0, COLOR0, TEXCOORD0, TEXCOORD1.
fn inputs_of(c: &DrawCase, vs: &ProgramShader) -> Vec<Vec<[u32; 4]>> {
    c.vertices
        .iter()
        .map(|vertex| {
            let mut regs = vec![[0u32; 4]; vs.input_count()];
            for input in &vs.interface().inputs {
                let (start, width) = match (input.label.name.to_ascii_uppercase().as_str(), input.label.index)
                {
                    ("POSITION", 0) => (0, 4),
                    ("COLOR", 0) => (4, 4),
                    ("TEXCOORD", 0) => (8, 2),
                    ("TEXCOORD", 1) => (10, 4),
                    _ => continue,
                };
                let sig = input.provenance.d3d.unwrap();
                let lanes: Vec<usize> = (0..4).filter(|l| sig.mask >> l & 1 != 0).collect();
                for (k, lane) in lanes.iter().take(width).enumerate() {
                    regs[sig.register as usize][*lane] = vertex[start + k].to_bits();
                }
            }
            regs
        })
        .collect()
}

/// The targets a case draws to, cleared, and its depth buffer.
fn surface_of(c: &DrawCase) -> Surface {
    let n = (SIZE * SIZE) as usize;
    let highest = c.targets.iter().max().map_or(0, |r| r + 1) as usize;
    Surface {
        width: SIZE,
        height: SIZE,
        layers: 1,
        targets: (0..highest)
            .map(|r| {
                c.targets
                    .contains(&(r as u32))
                    .then(|| vec![[CLEAR.to_bits(); 4]; n])
            })
            .collect(),
        depth: c.depth.map(|(_, _, clear)| vec![clear; n]),
    }
}

fn state_of(c: &DrawCase) -> Rasterizer {
    Rasterizer {
        viewports: vec![Viewport {
            x: 0.0,
            y: 0.0,
            width: SIZE as f32,
            height: SIZE as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }],
        sample_mask: u32::MAX,
        cull: c.cull,
        front_counter_clockwise: c.ccw,
        depth: c.depth.map(|(compare, write, _)| DepthState { compare, write }),
        alpha_test: None,
        clip_planes: [None; 6],
        point_size: None,
        point_sprite: false,
        flat_colours: false,
        clamp_colours: false,
        fog: None,
    }
}

/// The vertex program's constant buffer: a scale of 1.
const VERTEX_CB: [[u32; 4]; 2] = [[0x3f80_0000; 4], [0; 4]];

fn ours(c: &DrawCase, dir: &Path) -> Surface {
    let vs = lift(dir, &format!("draw_{}.dxil", c.vs));
    let ps = lift(dir, &format!("draw_{}.dxil", c.ps));
    let draw = DrawShader::new(&vs, &ps).unwrap();
    let mut surface = surface_of(c);
    let mut sampler = ramp;
    draw.draw(
        &inputs_of(c, &vs),
        &Primitive::assemble(c.topology, &c.indices, None).unwrap(),
        &[(0, &VERTEX_CB[..])],
        &[],
        &mut sampler,
        &state_of(c),
        &mut surface,
    )
    .unwrap();
    surface
}

/// A case drawn from vertices a stage already made: the vertex program's outputs, one register per output varying
/// (its lanes from the first), as a geometry, domain or mesh program's vertices are.
fn shaded(c: &DrawCase, dir: &Path) -> Surface {
    let vs = lift(dir, &format!("draw_{}.dxil", c.vs));
    let ps = lift(dir, &format!("draw_{}.dxil", c.ps));
    let outputs = &vs.interface().outputs;
    let mut sampler = ramp;
    let vertices: Vec<Vec<[u32; 4]>> = inputs_of(c, &vs)
        .iter()
        .map(|input| {
            let (e, _) = vs
                .execute_vertex(input, &[(0, &VERTEX_CB[..])], &mut sampler)
                .unwrap();
            outputs
                .iter()
                .map(|o| {
                    let sig = o.provenance.d3d.unwrap();
                    let mut value = [0u32; 4];
                    let lanes = (0..4).filter(|l| sig.mask >> l & 1 != 0);
                    for (k, lane) in lanes.enumerate() {
                        value[k] = e.outputs[sig.register as usize][lane];
                    }
                    value
                })
                .collect()
        })
        .collect();
    let mut surface = surface_of(c);
    ShadedDraw::new(outputs, &ps)
        .unwrap()
        .draw(
            &vertices,
            &Primitive::assemble(c.topology, &c.indices, None).unwrap(),
            &[],
            &mut sampler,
            &state_of(c),
            &mut surface,
        )
        .unwrap();
    surface
}

/// The rasterizer is one: vertices a stage before the pixel program made, linked by varying, draw what the vertex
/// path draws, bit for bit, on every WARP case.
#[test]
fn a_shaded_draw_draws_what_the_vertex_path_draws() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for c in cases() {
        let (a, b) = (ours(&c, &dir), shaded(&c, &dir));
        assert!(
            a.targets
                .iter()
                .flatten()
                .flatten()
                .any(|t| *t != [CLEAR.to_bits(); 4]),
            "{}: draws nothing",
            c.name
        );
        for (r, (x, y)) in a.targets.iter().zip(&b.targets).enumerate() {
            let (Some(x), Some(y)) = (x, y) else {
                assert_eq!(
                    x.is_some(),
                    y.is_some(),
                    "{} o{r}: written by one path only",
                    c.name
                );
                continue;
            };
            if let Some(p) = (0..x.len()).find(|&p| x[p] != y[p]) {
                panic!(
                    "{} o{r} at ({}, {}): vertex path {:?}, shaded {:?}",
                    c.name,
                    p as u32 % a.width,
                    p as u32 / a.width,
                    x[p].map(f32::from_bits),
                    y[p].map(f32::from_bits)
                );
            }
        }
        assert_eq!(a.depth, b.depth, "{}: depth", c.name);
    }
}

fn close(a: u32, b: u32) -> bool {
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    a == b || (x - y).abs() <= TOLERANCE * x.abs().max(y.abs()).max(1.0)
}

#[test]
fn draws_agree_with_direct3d() {
    let Some(warp) = program(Tool::Warp) else {
        return;
    };
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let own = own_scratch("warp");
    let work = own.join("draw");
    std::fs::create_dir_all(&work).unwrap();
    let cases = cases();
    let text: String = cases.iter().map(|c| case_text(c, &dir)).collect();
    let (input, output) = (work.join("cases.txt"), work.join("results.jsonl"));
    std::fs::write(&input, text).unwrap();
    let _oracle = inputs::warp_exclusive();
    let run = warp
        .command()
        .path_arg(&input)
        .path_arg(&output)
        .output_within(inputs::WARP_LIMIT)
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(
        run.status.success(),
        "the oracle failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    let results: Vec<serde_json::Value> = std::fs::read_to_string(&output)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let mut failures = Vec::new();
    let mut compared = 0usize;
    for (c, r) in cases.iter().zip(&results) {
        assert_eq!(r["case"], c.name);
        if let Some(e) = r["error"].as_str() {
            failures.push(format!("{}: oracle error {e}", c.name));
            continue;
        }
        let words = |v: &serde_json::Value| -> Vec<u32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|w| u32::from_str_radix(w.as_str().unwrap(), 16).unwrap())
                .collect()
        };
        let surface = ours(c, &dir);
        for (k, reg) in c.targets.iter().enumerate() {
            let warp_words = words(&r["targets"][k]);
            // a case must draw something, or agreement says nothing
            let drawn = warp_words
                .chunks(4)
                .filter(|t| t.iter().any(|w| *w != CLEAR.to_bits()))
                .count();
            println!("{} o{reg}: the oracle drew {drawn} pixels", c.name);
            if drawn == 0 {
                failures.push(format!("{} o{reg}: the oracle drew nothing", c.name));
            }
            let mine = surface.targets[*reg as usize].as_ref().unwrap();
            for p in 0..(SIZE * SIZE) as usize {
                let (x, y) = (p % SIZE as usize, p / SIZE as usize);
                let theirs = &warp_words[p * 4..p * 4 + 4];
                let covered_theirs = theirs.iter().any(|w| *w != CLEAR.to_bits());
                let covered_ours = mine[p].iter().any(|w| *w != CLEAR.to_bits());
                if covered_theirs != covered_ours {
                    failures.push(format!(
                        "{} o{reg} ({x}, {y}): drawn by {}",
                        c.name,
                        if covered_theirs {
                            "the oracle only"
                        } else {
                            "the executor only"
                        }
                    ));
                    continue;
                }
                for lane in 0..4 {
                    compared += 1;
                    if !close(mine[p][lane], theirs[lane]) {
                        failures.push(format!(
                            "{} o{reg}.{lane} ({x}, {y}): executor {}, oracle {}",
                            c.name,
                            f32::from_bits(mine[p][lane]),
                            f32::from_bits(theirs[lane])
                        ));
                    }
                }
            }
        }
        if c.depth.is_some() {
            let warp_depth = words(&r["depth"]);
            let mine = surface.depth.as_ref().unwrap();
            for p in 0..(SIZE * SIZE) as usize {
                compared += 1;
                if !close(mine[p].to_bits(), warp_depth[p]) {
                    failures.push(format!(
                        "{} depth ({}, {}): executor {}, oracle {}",
                        c.name,
                        p % SIZE as usize,
                        p / SIZE as usize,
                        mine[p],
                        f32::from_bits(warp_depth[p])
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {compared} values differ:\n{}",
        failures.len(),
        failures.iter().take(60).cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(compared > 0);
}

/// A vertex program's register view carries its position at the position's register, as the stage after it reads
/// it: the register view alone is a vertex a geometry, hull or pixel stage can take.
#[test]
fn the_vertex_register_view_carries_the_position() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let vs = lift(&dir, "draw_vs.dxil");
    let sig = vs
        .interface()
        .outputs
        .iter()
        .find(|o| o.system == Some(iridia::SystemValue::ClipPosition))
        .and_then(|o| o.provenance.d3d)
        .unwrap();
    let mut sampler = ramp;
    for c in cases() {
        for input in inputs_of(&c, &vs) {
            let (e, position) = vs
                .execute_vertex(&input, &[(0, &VERTEX_CB[..])], &mut sampler)
                .unwrap();
            assert_eq!(e.outputs[sig.register as usize], position, "{}", c.name);
            assert_eq!(e.output_masks[sig.register as usize], sig.mask, "{}", c.name);
        }
    }
}

/// One primitive of a recorded raster file: whether it is a line, its positions (1/256 pixels; a point's depth third),
/// and the pixels drawn.
struct Recorded {
    line: bool,
    coords: Vec<i64>,
    drawn: Vec<(u32, u32)>,
}

/// A recorded raster file (tests/fixtures/warp_raster_<adapter>.txt, made by tests/oracles/warp_raster.py): its header
/// and primitives.
fn recorded(adapter: &str) -> (String, Vec<Recorded>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let text = std::fs::read_to_string(dir.join(format!("warp_raster_{adapter}.txt"))).unwrap();
    let mut lines = text.lines();
    let header = lines.next().unwrap().to_string();
    let prims = lines
        .map(|l| {
            let words: Vec<&str> = l.split_whitespace().collect();
            let line = words[0] == "line";
            let n = if line { 4 } else { 3 };
            Recorded {
                line,
                coords: words[1..=n].iter().map(|w| w.parse().unwrap()).collect(),
                drawn: words[n + 1..]
                    .iter()
                    .map(|p| {
                        let p = u8::from_str_radix(p, 16).unwrap();
                        (u32::from(p >> 4), u32::from(p & 15))
                    })
                    .collect(),
            }
        })
        .collect();
    (header, prims)
}

/// The pixels the executor draws for a recorded primitive, in the recording's order (column, then row).
fn executor_draws(draw: &DrawShader, vs: &ProgramShader, r: &Recorded) -> Vec<(u32, u32)> {
    let c: Vec<f32> = r.coords.iter().map(|&v| v as f32 / 256.0).collect();
    let (topology, positions) = if r.line {
        (
            Topology::LineList,
            vec![at(c[0], c[1], 0.5, 1.0), at(c[2], c[3], 0.5, 1.0)],
        )
    } else {
        (Topology::PointList, vec![at(c[0], c[1], c[2], 1.0)])
    };
    let case = DrawCase {
        name: "recorded",
        vs: "vs",
        ps: "colour",
        indices: (0..positions.len() as u32).collect(),
        vertices: positions
            .into_iter()
            .map(|p| v(p, [1.0, 0.0, 0.0, 1.0], [0.0, 0.0], 1.0))
            .collect(),
        topology,
        cull: Cull::None,
        ccw: false,
        depth: None,
        targets: vec![0],
    };
    let mut surface = surface_of(&case);
    let mut sampler = ramp;
    draw.draw(
        &inputs_of(&case, vs),
        &Primitive::assemble(case.topology, &case.indices, None).unwrap(),
        &[(0, &VERTEX_CB[..])],
        &[],
        &mut sampler,
        &state_of(&case),
        &mut surface,
    )
    .unwrap();
    let target = surface.targets[0].as_ref().unwrap();
    let mut drawn: Vec<(u32, u32)> = (0..SIZE * SIZE)
        .filter(|&p| target[p as usize] != [CLEAR.to_bits(); 4])
        .map(|p| (p % SIZE, p / SIZE))
        .collect();
    drawn.sort_unstable();
    drawn
}

/// The GPU and WARP files: the same primitives, the GPU's header naming hardware, WARP's naming WARP, and their
/// answers not all the same (a run the oracle sent to WARP cannot pass for the GPU's).
fn both_recorded() -> (Vec<Recorded>, Vec<Recorded>) {
    let (gpu_header, gpu) = recorded("hardware");
    let (warp_header, warp) = recorded("warp");
    assert!(gpu_header.starts_with("# adapter hardware: "), "{gpu_header}");
    assert_eq!(warp_header, "# adapter warp");
    assert_eq!(gpu.len(), warp.len());
    for (a, b) in gpu.iter().zip(&warp) {
        assert_eq!((a.line, &a.coords), (b.line, &b.coords));
    }
    assert!(gpu.iter().zip(&warp).any(|(a, b)| a.drawn != b.drawn));
    (gpu, warp)
}

/// Which pixels points and lines draw, against a GPU's answers (an NVIDIA RTX 5070 Ti, recorded in
/// tests/fixtures/warp_raster_hardware.txt): 600 points and 10,640 lines, among them 3,000 at 45 degrees and line
/// ends at every 1/16-pixel point of a diamond in all eight octants. Every pixel matches: the GPU draws the D3D11.3
/// diamond exit rule exactly, and so does the executor.
#[test]
fn points_and_lines_draw_what_a_gpu_drew() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let vs = lift(&dir, "draw_vs.dxil");
    let ps = lift(&dir, "draw_colour.dxil");
    let draw = DrawShader::new(&vs, &ps).unwrap();
    let (gpu, _) = both_recorded();
    let mut failures = Vec::new();
    for r in &gpu {
        let ours = executor_draws(&draw, &vs, r);
        if ours != r.drawn {
            failures.push(format!(
                "{} {:?}: the GPU drew {:?}, the executor {ours:?}",
                if r.line { "line" } else { "point" },
                r.coords,
                r.drawn
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} primitives differ:\n{}",
        failures.len(),
        gpu.len(),
        failures.iter().take(30).cloned().collect::<Vec<_>>().join("\n")
    );
}

/// Whether a line from `p0` to `p1` (1/256 pixels) ends inside the test area of pixel `(px, py)`'s diamond.
fn ends_inside(p0: [i64; 2], p1: [i64; 2], (px, py): (u32, u32)) -> bool {
    let u = [
        p1[0] - (i64::from(px) * 256 + 128),
        p1[1] - (i64::from(py) * 256 + 128),
    ];
    let y_major = (p1[1] - p0[1]).abs() > (p1[0] - p0[0]).abs();
    let l1 = u[0].abs() + u[1].abs();
    l1 < 128 || (l1 == 128 && (u[1] > 0 || (y_major && u[1] == 0 && u[0] > 0)))
}

/// Whether a 45-degree line from `p0` to `p1` lies along an edge of pixel `(px, py)`'s diamond for more than a point.
fn along_an_edge(p0: [i64; 2], p1: [i64; 2], (px, py): (u32, u32)) -> bool {
    let d = [p1[0] - p0[0], p1[1] - p0[1]];
    if d[0].abs() != d[1].abs() || d[0] == 0 {
        return false;
    }
    let u = [
        p0[0] - (i64::from(px) * 256 + 128),
        p0[1] - (i64::from(py) * 256 + 128),
    ];
    [(1, 1), (-1, 1), (1, -1), (-1, -1)].iter().any(|&(a, b)| {
        if a * d[0] + b * d[1] != 0 || a * u[0] + b * u[1] != 128 {
            return false;
        }
        // along the line, in steps of its x: the edge runs from x = 0 to x = 128 a about the centre, the line from
        // 0 to |d.x|
        let (t0, t1) = (-u[0] * d[0].signum(), (128 * a - u[0]) * d[0].signum());
        t0.min(t1).max(0) < t0.max(t1).min(d[0].abs())
    })
}

/// Whether the line from `p0` to `p1` passes exactly through a corner of pixel `(px, py)`'s diamond on its minor axis
/// (the top or bottom corner for a line more horizontal than vertical, left or right otherwise), which the pixel shares
/// with the next pixel along that axis.
fn through_a_minor_corner(p0: [i64; 2], p1: [i64; 2], (px, py): (u32, u32)) -> bool {
    let d = [p1[0] - p0[0], p1[1] - p0[1]];
    let (major, minor) = if d[1].abs() > d[0].abs() { (1, 0) } else { (0, 1) };
    let c = [i64::from(px) * 256 + 128, i64::from(py) * 256 + 128];
    // where the line crosses the centre's major coordinate: at t = (c - p0) / d along the major axis, in [0, 1]
    let along = (c[major] - p0[major]) * d[major].signum();
    if d[major] == 0 || along < 0 || along > d[major].abs() {
        return false;
    }
    // the minor offset there, times d[major], is +-128 d[major]
    let offset = (p0[minor] - c[minor]) * d[major] + (c[major] - p0[major]) * d[minor];
    offset.abs() == 128 * d[major].abs()
}

/// Whether a line from `p0` to `p1` runs past the viewport (the 16 x 16 target) and `(px, py)` is on its border.
fn past_the_viewport(p0: [i64; 2], p1: [i64; 2], (px, py): (u32, u32)) -> bool {
    let outside = |p: [i64; 2]| p.iter().any(|&c| !(0..=i64::from(SIZE) * 256).contains(&c));
    (outside(p0) || outside(p1)) && (px == 0 || py == 0 || px == SIZE - 1 || py == SIZE - 1)
}

/// WARP draws lines otherwise than the GPU and the executor in four ways only, which iridia_exec::draw documents: it
/// also draws some pixels a line ends inside (the spec's diamond exit rule leaves every one out); a 45-degree line
/// along a diamond's edge can draw the pixel on its other side; a line through a diamond's corner on its minor axis can
/// draw the pixel on the corner's other side; and it clips a line to the viewport, which can change the viewport's
/// border pixels. Every pixel WARP draws differently is one of those, and it draws points as the GPU does.
#[test]
fn warp_draws_otherwise_only_as_documented() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let vs = lift(&dir, "draw_vs.dxil");
    let ps = lift(&dir, "draw_colour.dxil");
    let draw = DrawShader::new(&vs, &ps).unwrap();
    let (_, warp) = both_recorded();
    let (mut ends, mut edges, mut corners, mut clipped, mut failures) =
        (0usize, 0usize, 0usize, 0usize, Vec::new());
    for r in &warp {
        let ours = executor_draws(&draw, &vs, r);
        let warp_only = r.drawn.iter().filter(|p| !ours.contains(p)).map(|p| (*p, true));
        let ours_only = ours.iter().filter(|p| !r.drawn.contains(p)).map(|p| (*p, false));
        for (p, by_warp) in warp_only.chain(ours_only) {
            if r.line {
                let (p0, p1) = ([r.coords[0], r.coords[1]], [r.coords[2], r.coords[3]]);
                if along_an_edge(p0, p1, p) {
                    edges += 1;
                    continue;
                }
                if by_warp && ends_inside(p0, p1, p) {
                    ends += 1;
                    continue;
                }
                if through_a_minor_corner(p0, p1, p) {
                    corners += 1;
                    continue;
                }
                if past_the_viewport(p0, p1, p) {
                    clipped += 1;
                    continue;
                }
            }
            failures.push(format!(
                "{} {:?} at {p:?}: drawn by {}",
                if r.line { "line" } else { "point" },
                r.coords,
                if by_warp { "WARP only" } else { "the executor only" }
            ));
        }
    }
    println!(
        "WARP differs on {ends} pixels lines end inside, {edges} along 45-degree edges, {corners} at minor-axis corners and {clipped} on the viewport's border"
    );
    assert!(
        failures.is_empty(),
        "{} pixels differ otherwise:\n{}",
        failures.len(),
        failures.iter().take(30).cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(
        ends > 0 && edges > 0 && corners > 0 && clipped > 0,
        "the documented differences are in the recording"
    );
}
