//! Ray queries (DXR 1.1's inline ray tracing) as the WGSL of wgpu's `wgpu_ray_query` extension, which stock naga reads and
//! its writers (HLSL, SPIR-V, Metal) write; naga's own WGSL writer does not, so this one does.
//!
//! Each query of the function is a local `var q: ray_query;` and, where something reads the ray the query was started with
//! (naga's query does not keep it), four locals beside it that hold the ray's flags, least distance, origin and direction.
//! A start is `rayQueryInitialize(&q, scene, RayDesc(...))` with the flags the query is declared with joined to the ray's
//! own; a step is `rayQueryProceed(&q)`; a commit is `rayQueryConfirmIntersection` or `rayQueryGenerateIntersection`; an
//! abort is `rayQueryTerminate`. What the query holds is a field of what `rayQueryGetCommittedIntersection(&q)` or
//! `rayQueryGetCandidateIntersection(&q)` returns, called where the program reads it (the search changes the answer at
//! every step), the extension's predeclared `RayIntersection`: `kind` (naga's: 0 none, 1 triangle, 2 a generated
//! procedural hit for a committed intersection; 1 a triangle, 3 a procedural primitive for a candidate), `t`,
//! `instance_custom_data` (the instance id), `instance_index`, `sbt_record_offset` (the contribution to the hit group
//! index), `geometry_index`, `primitive_index`, `barycentrics`, `front_face`, and the instance's `object_to_world` and
//! `world_to_object` as 4 by 3 matrices (four columns of three rows). The ray in the hit's object space is the hit's
//! `world_to_object` applied to the kept ray, as Direct3D defines it.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    format,
    string::{String, ToString},
};

use iridia_ir::{End, Function, HitField, LabelId, Node, Op, QueryField, QueryId, Region};

use crate::emit::{R, Writer};
use crate::text::err;

/// A query of the function as the text declares it.
#[derive(Clone, Debug)]
pub(crate) struct QueryVars {
    /// The `ray_query` local.
    pub query: String,
    /// The ray the query was started with, kept for the reads of it: flags, least distance, origin and direction. `None`
    /// where nothing reads it.
    pub ray: Option<[String; 4]>,
    /// The flags the query is declared with, joined to every ray it traces.
    pub flags: u32,
}

/// The query a program starts again: a query with a second start, or one whose start sits in a loop that can run again.
///
/// Such a query has no WGSL form that naga carries to Direct3D 12, where wgpu runs the text, so the program is refused by
/// name rather than written to give a wrong answer. Both ways of writing a restart fail in naga 30.0.1 (stock, as wgpu 30
/// uses it); neither is a rule of WGSL:
///
/// - One function-level `var q: ray_query;` that `rayQueryInitialize` starts again (in a loop, or at a second site): the
///   text parses and validates, but naga's HLSL writer keeps the query's "finished" bit across the new start, so the
///   second search never steps. A loop that starts one query twice over one triangle leaves the committed status 1 for the
///   first search and 0 for the second, where Direct3D leaves 1 for both (naga's HLSL, compiled by dxc as `cs_6_5`, run on
///   WARP).
/// - A fresh `var q: ray_query;` declared inside the loop body, the way naga's own front end reads a restart: it
///   validates, and naga's HLSL writer panics (`unreachable!()` in `back/hlsl/conv.rs`, `hlsl_type_id`).
///
/// A loop whose body never continues runs once however it is written (the structurizer's blocks that leave early are
/// loops with no back edge), and does not count.
pub(crate) fn restarted(body: &Region) -> Option<QueryId> {
    fn continues(region: &Region, label: LabelId) -> bool {
        matches!(&region.end, Some(End::Continue(l, _)) if *l == label)
            || region
                .nodes
                .iter()
                .any(|n| n.regions().into_iter().any(|r| continues(r, label)))
    }
    fn walk(region: &Region, again: bool, starts: &mut BTreeMap<u32, u32>, bad: &mut Option<QueryId>) {
        for node in &region.nodes {
            if let Node::RayQueryInit { query, .. } = node {
                let n = starts.entry(query.0).or_default();
                *n = n.saturating_add(1);
                if again || *n > 1 {
                    bad.get_or_insert(*query);
                }
            }
            for r in node.regions() {
                let loops = match node {
                    Node::Loop { label, .. } => continues(r, *label),
                    _ => false,
                };
                walk(r, again || loops, starts, bad);
            }
        }
    }
    let mut bad = None;
    walk(body, false, &mut BTreeMap::new(), &mut bad);
    bad
}

impl Writer<'_> {
    /// The queries read for the ray they were started with: a field of the ray, or of a hit in its object space.
    fn queries_read_for_their_ray(f: &Function) -> BTreeSet<u32> {
        f.values
            .iter()
            .filter_map(|v| match v.op {
                Op::RayQueryState(
                    q,
                    QueryField::RayFlags
                    | QueryField::RayTMin
                    | QueryField::WorldRayOrigin(_)
                    | QueryField::WorldRayDirection(_)
                    | QueryField::Committed(HitField::ObjectRayOrigin(_) | HitField::ObjectRayDirection(_))
                    | QueryField::Candidate(HitField::ObjectRayOrigin(_) | HitField::ObjectRayDirection(_)),
                ) => Some(q.0),
                _ => None,
            })
            .collect()
    }

    /// The locals of the function's queries, at the top of the entry point.
    pub(crate) fn declare_queries(&mut self) -> R<String> {
        let mut out = String::new();
        let read = Self::queries_read_for_their_ray(self.f);
        for (n, decl) in self.f.queries.iter().enumerate() {
            let query = self.names.unique("rquery");
            out.push_str(&format!("    var {query}: ray_query;\n"));
            let keeps = u32::try_from(n).is_ok_and(|n| read.contains(&n));
            let ray = if keeps {
                let mut part = |suffix: &str, ty: &str| {
                    let name = self.names.unique(&format!("{query}_{suffix}"));
                    out.push_str(&format!("    var {name}: {ty};\n"));
                    name
                };
                Some([
                    part("flags", "u32"),
                    part("tmin", "f32"),
                    part("origin", "vec3<f32>"),
                    part("dir", "vec3<f32>"),
                ])
            } else {
                None
            };
            self.queries.push(QueryVars {
                query,
                ray,
                flags: decl.flags,
            });
        }
        Ok(out)
    }

    fn query(&self, q: QueryId) -> R<&QueryVars> {
        self.queries
            .get(q.0 as usize)
            .ok_or_else(|| err(format!("ray query q{}, which the function does not declare", q.0)))
    }

    /// A ray query's start, abort or commit.
    pub(crate) fn ray_query_node(&mut self, n: &Node) -> R<()> {
        match n {
            Node::RayQueryInit { query, scene, ray } => {
                if scene.index.is_some() {
                    return Err(err(
                        "a ray query in an element of an array of acceleration structures has no WGSL form",
                    ));
                }
                let q = self.query(*query)?.clone();
                let scene = self.global(scene.id)?.to_string();
                // the flags the query is declared with join the ray's
                let flags = self.get(ray.flags)?;
                let flags = if q.flags == 0 {
                    flags
                } else {
                    format!("({flags} | {}u)", q.flags)
                };
                let (mask, t_min, t_max) = (self.get(ray.mask)?, self.get(ray.t_min)?, self.get(ray.t_max)?);
                let (origin, direction) = (self.get(ray.origin)?, self.get(ray.direction)?);
                if let Some([kept_flags, kept_t_min, kept_origin, kept_direction]) = &q.ray {
                    self.line(&format!("{kept_flags} = {flags};"));
                    self.line(&format!("{kept_t_min} = {t_min};"));
                    self.line(&format!("{kept_origin} = {origin};"));
                    self.line(&format!("{kept_direction} = {direction};"));
                }
                self.line(&format!(
                    "rayQueryInitialize(&{}, {scene}, RayDesc({flags}, {mask}, {t_min}, {t_max}, {origin}, {direction}));",
                    q.query
                ));
            }
            Node::RayQueryAbort(query) => {
                let line = format!("rayQueryTerminate(&{});", self.query(*query)?.query);
                self.line(&line);
            }
            Node::RayQueryCommitTriangle(query) => {
                let line = format!("rayQueryConfirmIntersection(&{});", self.query(*query)?.query);
                self.line(&line);
            }
            Node::RayQueryCommitProcedural { query, t } => {
                let line = format!(
                    "rayQueryGenerateIntersection(&{}, {});",
                    self.query(*query)?.query,
                    self.get(*t)?
                );
                self.line(&line);
            }
            _ => return Err(err("not a ray query operation")),
        }
        Ok(())
    }

    /// `RayQueryProceed`: whether the search has another candidate.
    pub(crate) fn ray_query_proceed(&self, q: QueryId) -> R<String> {
        Ok(format!("rayQueryProceed(&{})", self.query(q)?.query))
    }

    /// `RayQueryState`: a field of the kept ray, or of the candidate or committed intersection.
    pub(crate) fn ray_query_state(&self, query: QueryId, field: QueryField) -> R<String> {
        let q = self.query(query)?;
        let kept = |part: usize| -> R<&str> {
            q.ray
                .as_ref()
                .and_then(|r| r.get(part))
                .map(String::as_str)
                .ok_or_else(|| {
                    err(format!(
                        "ray query q{} is read for its ray and keeps none",
                        query.0
                    ))
                })
        };
        let lane = |n: u8| ["x", "y", "z", "w"].get(usize::from(n)).copied().unwrap_or("x");
        let (committed, hit) = match field {
            QueryField::RayFlags => return Ok(kept(0)?.to_string()),
            QueryField::RayTMin => return Ok(kept(1)?.to_string()),
            QueryField::WorldRayOrigin(n) => return Ok(format!("{}.{}", kept(2)?, lane(n))),
            QueryField::WorldRayDirection(n) => return Ok(format!("{}.{}", kept(3)?, lane(n))),
            QueryField::CandidateNonOpaque => {
                return Err(err(
                    "whether a ray query's candidate procedural primitive is not opaque has no WGSL form: the intersection does not say",
                ));
            }
            QueryField::CommittedStatus => (true, None),
            QueryField::CandidateType => (false, None),
            QueryField::Committed(h) => (true, Some(h)),
            QueryField::Candidate(h) => (false, Some(h)),
        };
        let call = format!(
            "rayQueryGet{}Intersection(&{})",
            if committed { "Committed" } else { "Candidate" },
            q.query
        );
        let Some(hit) = hit else {
            // a committed intersection's kind is the status as it is (0 none, 1 triangle, 2 procedural); a candidate's is
            // 1 for a triangle and 3 for a procedural primitive, where the IR says 0 and 1
            return Ok(if committed {
                format!("{call}.kind")
            } else {
                format!("select(1u, 0u, ({call}.kind == 1u))")
            });
        };
        Ok(match hit {
            HitField::T => format!("{call}.t"),
            HitField::InstanceIndex => format!("{call}.instance_index"),
            HitField::InstanceId => format!("{call}.instance_custom_data"),
            HitField::GeometryIndex => format!("{call}.geometry_index"),
            HitField::PrimitiveIndex => format!("{call}.primitive_index"),
            HitField::ContributionToHitGroup => format!("{call}.sbt_record_offset"),
            HitField::FrontFace => format!("{call}.front_face"),
            HitField::Barycentrics(n) => format!("{call}.barycentrics.{}", lane(n)),
            // a 3 by 4 matrix's row and column: the intersection's is four columns of three rows
            HitField::ObjectToWorld { row, column } => {
                format!("{call}.object_to_world[{}][{}]", column.min(3), row.min(2))
            }
            HitField::WorldToObject { row, column } => {
                format!("{call}.world_to_object[{}][{}]", column.min(3), row.min(2))
            }
            // the kept ray in the hit's object space: the hit's world-to-object transform of the ray's origin (a point,
            // w 1) or direction (w 0)
            HitField::ObjectRayOrigin(n) => {
                format!(
                    "({call}.world_to_object * vec4<f32>({}, 1f)).{}",
                    kept(2)?,
                    lane(n)
                )
            }
            HitField::ObjectRayDirection(n) => {
                format!(
                    "({call}.world_to_object * vec4<f32>({}, 0f)).{}",
                    kept(3)?,
                    lane(n)
                )
            }
        })
    }
}
