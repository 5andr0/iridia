//! A draw through every stage a suite's pipeline names: the vertex program for each vertex, the hull and domain
//! programs for each patch (`iridia_exec::tessellation`), the geometry program for each primitive
//! (`iridia_exec::geometry`), and the pixel program over the points, lines and triangles the last of them made
//! (`iridia_exec::draw::ShadedDraw`). A draw of a vertex and a pixel program alone is `iridia_exec::draw::DrawShader`'s,
//! which these runners call directly.
//!
//! Stages link by semantic, as the D3D runtime matches one stage's output signature to the next one's input: an input
//! reads the output of the same name (without case) and index, `SV_Position` the clip-space position. An input no
//! output carries reads 0. The vertices between stages are one register per varying, in the order of the program's
//! interface, each varying's lanes from the first (the form `iridia_exec`'s stage drivers take and give).
use iridia::{Interface, SystemValue, TessTopology, Varying};
use iridia_exec::amplification::AmplificationShader;
use iridia_exec::draw::{Corners, DrawStats, Primitive, Rasterizer, ShadedDraw, Surface};
use iridia_exec::geometry::{GeometryShader, Topology, Vertex};
use iridia_exec::mesh::MeshShader;
use iridia_exec::tessellation::TessellationShader;
use iridia_exec::{Constants, PixelShader, ProgramShader, Sampler};

/// Why a draw did not run to its end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stopped {
    /// The draw needs what the CPU draw does not do.
    NotModelled(String),
    /// A stage stopped: the program's role (`vs`, `hs`, ...) and the message.
    Failed(&'static str, String),
}

/// How the draw's vertices make primitives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Assembly {
    /// A list or strip of points, lines or triangles.
    Primitives(Topology),
    /// Patches of this many control points.
    Patches(u32),
}

/// A constant buffer bound to a `ProgramShader` bind slot: the slot and its rows.
pub type Bank<'a> = (u32, &'a [[u32; 4]]);

/// The programs of a draw past the vertex program.
#[derive(Debug)]
pub struct Stages<'a> {
    /// The vertex program.
    pub vs: &'a ProgramShader,
    /// The hull and domain programs.
    pub tessellation: Option<&'a TessellationShader>,
    /// The geometry program.
    pub gs: Option<&'a GeometryShader>,
    /// The pixel program.
    pub ps: &'a ProgramShader,
}

/// The constant buffers a stage's program, of `interface`, reads: `rows(register, space, rows)` for each.
pub fn constants_of(interface: &Interface, rows: &dyn Fn(u32, u32, usize) -> Vec<[u32; 4]>) -> Constants {
    let mut out = Constants::new();
    for r in &interface.resources {
        if let (iridia::ResourceKind::ConstantBuffer { layout }, Some(d)) = (&r.kind, r.provenance.d3d) {
            let n = (layout.size as usize).div_ceil(16).max(1);
            out.insert((d.register, d.space), rows(d.register, d.space, n));
        }
    }
    out
}

/// For each of `to`, the one of `from` it reads: the same semantic, or both the clip-space position.
fn link(from: &[Varying], to: &[Varying]) -> Vec<Option<usize>> {
    to.iter()
        .map(|t| {
            from.iter().position(|f| {
                (f.label.index == t.label.index && f.label.name.eq_ignore_ascii_case(&t.label.name))
                    || (f.system == Some(SystemValue::ClipPosition)
                        && t.system == Some(SystemValue::ClipPosition))
            })
        })
        .collect()
}

/// `vertices` of a stage with outputs `from` as the inputs `to` of the next.
fn relink(vertices: &[Vertex], from: &[Varying], to: &[Varying]) -> Vec<Vertex> {
    let map = link(from, to);
    vertices
        .iter()
        .map(|v| {
            map.iter()
                .map(|m| m.and_then(|k| v.get(k).copied()).unwrap_or([0; 4]))
                .collect()
        })
        .collect()
}

/// The kind of primitive a list of index groups makes: what a geometry program after the tessellator reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Points,
    Lines,
    Triangles,
}

impl Stages<'_> {
    /// Draw `inputs` (each vertex's input registers of the vertex program) as `assembly` into `surface`. `rows` gives
    /// a constant buffer's rows by register and space; the vertex and pixel programs read theirs from `banks`
    /// (`ProgramShader`'s bind-slot banks: vertex, pixel).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        inputs: &[Vec<[u32; 4]>],
        assembly: Assembly,
        banks: [&[Bank<'_>]; 2],
        rows: &dyn Fn(u32, u32, usize) -> Vec<[u32; 4]>,
        sampler: &mut dyn Sampler,
        state: &Rasterizer,
        surface: &mut Surface,
    ) -> Result<DrawStats, Stopped> {
        let failed = |stage: &'static str| move |e: String| Stopped::Failed(stage, e);
        // the vertex program, into one register per output varying
        let vs_outputs = &self.vs.interface().outputs;
        let mut vertices = Vec::with_capacity(inputs.len());
        for input in inputs {
            let (e, _) = self
                .vs
                .execute_vertex(input, banks[0], sampler)
                .map_err(|e| Stopped::Failed("vs", e.to_string()))?;
            let mut vertex = Vec::with_capacity(vs_outputs.len());
            for o in vs_outputs {
                let sig = o.provenance.d3d.ok_or_else(|| {
                    Stopped::Failed(
                        "vs",
                        format!("output {}{} has no register", o.label.name, o.label.index),
                    )
                })?;
                let register = e.outputs.get(sig.register as usize).copied().unwrap_or([0; 4]);
                let mut value = [0u32; 4];
                for (k, lane) in (0..4).filter(|l| sig.mask >> l & 1 != 0).enumerate() {
                    value[k] = register[lane];
                }
                vertex.push(value);
            }
            vertices.push(vertex);
        }
        let mut outputs: &[Varying] = vs_outputs;
        // the primitives, as groups of indices into `vertices` and the number a pixel program reads as
        // `SV_PrimitiveID`: the draw's count, the patch's under tessellation, the primitive a geometry program read
        let (kind, mut groups): (Kind, Vec<(Vec<usize>, u32)>) = match assembly {
            Assembly::Primitives(t) => (kind_of(t), numbered(t.primitives(vertices.len()))?),
            Assembly::Patches(n) => {
                let Some(tess) = self.tessellation else {
                    return Err(Stopped::NotModelled(
                        "a patch list drawn without a hull and a domain program".into(),
                    ));
                };
                let hull = tess.hull();
                if n != tess.hull_facts().input_control_points {
                    return Err(Stopped::NotModelled(format!(
                        "patches of {n} control points to a hull program that reads {}",
                        tess.hull_facts().input_control_points
                    )));
                }
                let (hc, dc) = (constants_of(hull, rows), constants_of(tess.domain(), rows));
                let points = relink(&vertices, outputs, &hull.inputs);
                let (mut made, mut groups, mut kind) = (Vec::new(), Vec::new(), Kind::Triangles);
                for (p, patch) in points.chunks_exact(n as usize).enumerate() {
                    let patch = tess
                        .tessellate(&hc, &dc, sampler, patch, p as u32)
                        .map_err(failed("hs"))?;
                    let base = made.len();
                    let per = match patch.topology {
                        TessTopology::Point => 1,
                        TessTopology::Line => 2,
                        TessTopology::TriangleCw | TessTopology::TriangleCcw => 3,
                    };
                    kind = [Kind::Points, Kind::Lines, Kind::Triangles][per - 1];
                    let id = u32::try_from(p).map_err(|e| Stopped::NotModelled(e.to_string()))?;
                    groups.extend(
                        patch
                            .indices
                            .chunks_exact(per)
                            .map(|g| (g.iter().map(|i| base + *i as usize).collect(), id)),
                    );
                    made.extend(patch.vertices);
                }
                vertices = made;
                outputs = &tess.domain().outputs;
                (kind, groups)
            }
        };
        if let Some(gs) = self.gs {
            let interface = gs.interface();
            let constants = constants_of(interface, rows);
            // the draw's vertices as it assembles them (an adjacency topology, a strip), or the primitives the
            // tessellator made as a list
            let (listed, input) = match (assembly, self.tessellation) {
                (Assembly::Primitives(t), None) => (vertices.clone(), t),
                _ => (
                    groups
                        .iter()
                        .flat_map(|(g, _)| g)
                        .filter_map(|i| vertices.get(*i).cloned())
                        .collect(),
                    match kind {
                        Kind::Points => Topology::PointList,
                        Kind::Lines => Topology::LineList,
                        Kind::Triangles => Topology::TriangleList,
                    },
                ),
            };
            let strips = gs
                .draw(
                    &constants,
                    sampler,
                    &relink(&listed, outputs, &interface.inputs),
                    input,
                )
                .map_err(failed("gs"))?;
            let output = gs.facts().output;
            (vertices, groups) = (Vec::new(), Vec::new());
            for strip in strips {
                let base = vertices.len();
                groups.extend(
                    strip
                        .primitives(output)
                        .into_iter()
                        .map(|g| (g.into_iter().map(|i| base + i).collect(), strip.primitive)),
                );
                vertices.extend(strip.vertices);
            }
            outputs = &interface.outputs;
        }
        let primitives = groups
            .iter()
            .map(|(g, id)| {
                let corners = match g.as_slice() {
                    [a] => Corners::Point(*a),
                    [a, b] => Corners::Line([*a, *b]),
                    [a, b, c] => Corners::Triangle([*a, *b, *c]),
                    _ => {
                        return Err(Stopped::NotModelled(
                            "an adjacency topology drawn without a geometry program".into(),
                        ));
                    }
                };
                Ok(Primitive { corners, id: *id })
            })
            .collect::<Result<Vec<_>, _>>()?;
        ShadedDraw::new(outputs, self.ps)
            .map_err(|e| Stopped::Failed("ps", e.to_string()))?
            .draw(&vertices, &primitives, banks[1], sampler, state, surface)
            .map_err(|e| Stopped::Failed("ps", e.to_string()))
    }
}

/// A mesh dispatch (`DispatchMesh`): `ms` runs `groups` thread groups, each writing a meshlet, or with an amplification
/// program `amplification` runs `groups` thread groups that launch `ms`'s, and `ps` runs over the meshlets' primitives
/// in the order of the dispatch. A primitive's number (`SV_PrimitiveID`, which a mesh program writes per primitive
/// where its pixel program reads it) is its place in its meshlet here.
#[allow(clippy::too_many_arguments)]
pub fn mesh_draw(
    amplification: Option<&AmplificationShader>,
    ms: &MeshShader,
    ps: &ProgramShader,
    groups: [u32; 3],
    wave: u32,
    ps_banks: &[Bank<'_>],
    rows: &dyn Fn(u32, u32, usize) -> Vec<[u32; 4]>,
    sampler: &mut dyn Sampler,
    state: &Rasterizer,
    surface: &mut Surface,
) -> Result<DrawStats, Stopped> {
    let mut constants = constants_of(ms.interface(), rows);
    let meshlets = match amplification {
        Some(a) => {
            constants.extend(constants_of(a.interface(), rows));
            a.draw(ms, &constants, sampler, groups, wave)
                .map_err(|e| Stopped::Failed("as", e.to_string()))?
        }
        None => ms
            .draw(&constants, sampler, groups, wave)
            .map_err(|e| Stopped::Failed("ms", e.to_string()))?,
    };
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut primitives = Vec::new();
    for meshlet in meshlets {
        let base = vertices.len();
        for (n, indices) in meshlet.primitives.iter().enumerate() {
            let at: Vec<usize> = indices.iter().map(|&i| base.saturating_add(i as usize)).collect();
            let corners = match at.as_slice() {
                [a, b, c] => Corners::Triangle([*a, *b, *c]),
                [a, b] => Corners::Line([*a, *b]),
                _ => {
                    return Err(Stopped::Failed(
                        "ms",
                        format!("a primitive of {} vertices", at.len()),
                    ));
                }
            };
            let id = u32::try_from(n).map_err(|e| Stopped::NotModelled(e.to_string()))?;
            primitives.push(Primitive { corners, id });
        }
        vertices.extend(meshlet.vertices);
    }
    ShadedDraw::new(&ms.interface().outputs, ps)
        .map_err(|e| Stopped::Failed("ps", e.to_string()))?
        .draw(&vertices, &primitives, ps_banks, sampler, state, surface)
        .map_err(|e| Stopped::Failed("ps", e.to_string()))
}

/// Groups of indices numbered in order from 0.
fn numbered(groups: Vec<Vec<usize>>) -> Result<Vec<(Vec<usize>, u32)>, Stopped> {
    groups
        .into_iter()
        .enumerate()
        .map(|(n, g)| {
            Ok((
                g,
                u32::try_from(n).map_err(|e| Stopped::NotModelled(e.to_string()))?,
            ))
        })
        .collect()
}

fn kind_of(t: Topology) -> Kind {
    match t {
        Topology::PointList => Kind::Points,
        Topology::LineList
        | Topology::LineStrip
        | Topology::LineListAdjacency
        | Topology::LineStripAdjacency => Kind::Lines,
        Topology::TriangleList
        | Topology::TriangleStrip
        | Topology::TriangleListAdjacency
        | Topology::TriangleStripAdjacency => Kind::Triangles,
    }
}
