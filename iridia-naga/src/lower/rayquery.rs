//! Ray queries (DXR 1.1's inline ray tracing) as naga's: each query of the function a local `ray_query` with a
//! `RayDesc` local beside it that keeps the ray it was started with (naga's query keeps the ray's own values to
//! itself), its operations naga's `RayQuery` statements, and a hit's fields those of naga's `RayIntersection`. naga's
//! `RayDesc` and `RayIntersection` are its special types, laid out here as its own front ends lay them out
//! (`naga::front::type_gen`, which a build without them lacks). Needs `Capabilities::RAY_QUERY`.

use alloc::{format, string::ToString, vec};

use iridia_ir::{HitField, Node, QueryField, QueryId, Type};
use naga::{
    BinaryOperator as Bo, Expression as E, Handle, LocalVariable, RayQueryFunction, Span, Statement,
    StructMember, TypeInner, VectorSize,
};

use super::{Lower, err};
use crate::Error;

/// A field of naga's `RayDesc`, by member index (the composed one's order: flags, cull mask, t min, t max, origin,
/// direction).
const RAY_FLAGS: u32 = 0;
const RAY_T_MIN: u32 = 2;
const RAY_ORIGIN: u32 = 4;
const RAY_DIRECTION: u32 = 5;

/// A field of naga's `RayIntersection`, by member index.
const KIND: u32 = 0;
const T: u32 = 1;
const INSTANCE_CUSTOM_DATA: u32 = 2;
const INSTANCE_INDEX: u32 = 3;
const SBT_RECORD_OFFSET: u32 = 4;
const GEOMETRY_INDEX: u32 = 5;
const PRIMITIVE_INDEX: u32 = 6;
const BARYCENTRICS: u32 = 7;
const FRONT_FACE: u32 = 8;
const OBJECT_TO_WORLD: u32 = 9;
const WORLD_TO_OBJECT: u32 = 10;

/// naga's intersection kind of a triangle (`RayQueryIntersection::Triangle`).
const KIND_TRIANGLE: u64 = 1;

/// A query of the function as naga holds it: its `ray_query` local, the `RayDesc` local of the ray it was started with,
/// and the flags it is declared with.
#[derive(Clone, Copy)]
pub(super) struct Query {
    query: Handle<LocalVariable>,
    ray: Handle<LocalVariable>,
    flags: u32,
}

/// A structure member of naga's special types.
fn member(name: &str, ty: Handle<naga::Type>, offset: u32) -> StructMember {
    StructMember {
        name: Some(name.to_string()),
        ty,
        binding: None,
        offset,
    }
}

impl Lower<'_> {
    /// naga's `RayDesc` (`SpecialTypes::ray_desc`).
    fn ray_desc_type(&mut self) -> Result<Handle<naga::Type>, Error> {
        if let Some(t) = self.module.special_types.ray_desc {
            return Ok(t);
        }
        let (u, f, v3) = (
            self.ty_of(Type::U32)?,
            self.ty_of(Type::F32)?,
            self.ty_of(Type::vector(iridia_ir::Scalar::F32, 3))?,
        );
        let members = vec![
            member("flags", u, 0),
            member("cull_mask", u, 4),
            member("tmin", f, 8),
            member("tmax", f, 12),
            member("origin", v3, 16),
            member("dir", v3, 32),
        ];
        let t = self.module.types.insert(
            naga::Type {
                name: Some("RayDesc".to_string()),
                inner: TypeInner::Struct { members, span: 48 },
            },
            Span::UNDEFINED,
        );
        self.module.special_types.ray_desc = Some(t);
        Ok(t)
    }

    /// naga's `RayIntersection` (`SpecialTypes::ray_intersection`).
    fn ray_intersection_type(&mut self) -> Result<Handle<naga::Type>, Error> {
        if let Some(t) = self.module.special_types.ray_intersection {
            return Ok(t);
        }
        let (u, f, v2, b) = (
            self.ty_of(Type::U32)?,
            self.ty_of(Type::F32)?,
            self.ty_of(Type::vector(iridia_ir::Scalar::F32, 2))?,
            self.ty_of(Type::BOOL)?,
        );
        let transform = self.ty(TypeInner::Matrix {
            columns: VectorSize::Quad,
            rows: VectorSize::Tri,
            scalar: naga::Scalar::F32,
        });
        let members = vec![
            member("kind", u, 0),
            member("t", f, 4),
            member("instance_custom_data", u, 8),
            member("instance_index", u, 12),
            member("sbt_record_offset", u, 16),
            member("geometry_index", u, 20),
            member("primitive_index", u, 24),
            member("barycentrics", v2, 28),
            member("front_face", b, 36),
            member("object_to_world", transform, 48),
            member("world_to_object", transform, 112),
        ];
        let t = self.module.types.insert(
            naga::Type {
                name: Some("RayIntersection".to_string()),
                inner: TypeInner::Struct { members, span: 176 },
            },
            Span::UNDEFINED,
        );
        self.module.special_types.ray_intersection = Some(t);
        Ok(t)
    }

    /// The locals of the function's queries, each a `ray_query` and the `RayDesc` of its ray.
    pub(super) fn declare_queries(&mut self) -> Result<(), Error> {
        let queries = self.f.queries.clone();
        if queries.is_empty() {
            return Ok(());
        }
        let ray_desc = self.ray_desc_type()?;
        let query_ty = self.ty(TypeInner::RayQuery { vertex_return: false });
        for (n, q) in queries.iter().enumerate() {
            let local = |name: alloc::string::String, ty| LocalVariable {
                name: Some(name),
                ty,
                init: None,
            };
            let query = self
                .func
                .local_variables
                .append(local(format!("q{n}"), query_ty), Span::UNDEFINED);
            let ray = self
                .func
                .local_variables
                .append(local(format!("q{n}_ray"), ray_desc), Span::UNDEFINED);
            self.queries.push(Query {
                query,
                ray,
                flags: q.flags,
            });
        }
        Ok(())
    }

    fn ray_query(&self, q: QueryId) -> Result<Query, Error> {
        self.queries
            .get(q.0 as usize)
            .copied()
            .ok_or_else(|| err(format!("ray query q{}, which the function does not declare", q.0)))
    }

    /// A ray query's operation: its start, abort or commit.
    pub(super) fn ray_query_node(&mut self, n: &Node) -> Result<(), Error> {
        let (q, fun) = match n {
            Node::RayQueryInit { query, scene, ray } => {
                if scene.index.is_some() {
                    return Err(err(
                        "a ray query in an element of an array of acceleration structures is not lowered yet",
                    ));
                }
                let q = self.ray_query(*query)?;
                let global = self.global(scene.id)?;
                let acceleration_structure = self.expr(E::GlobalVariable(global));
                // the flags the query is declared with join the ray's
                let flags = self.get(ray.flags)?;
                let flags = if q.flags == 0 {
                    flags
                } else {
                    let declared = self.literal_bits(iridia_ir::Scalar::U32, u64::from(q.flags))?;
                    self.binary(Bo::InclusiveOr, flags, declared)
                };
                let components = vec![
                    flags,
                    self.get(ray.mask)?,
                    self.get(ray.t_min)?,
                    self.get(ray.t_max)?,
                    self.get(ray.origin)?,
                    self.get(ray.direction)?,
                ];
                let ty = self.ray_desc_type()?;
                let descriptor = self.expr(E::Compose { ty, components });
                // the ray is kept for the reads of it
                self.store_local(q.ray, descriptor)?;
                (
                    q,
                    RayQueryFunction::Initialize {
                        acceleration_structure,
                        descriptor,
                    },
                )
            }
            Node::RayQueryAbort(query) => (self.ray_query(*query)?, RayQueryFunction::Terminate),
            Node::RayQueryCommitTriangle(query) => {
                (self.ray_query(*query)?, RayQueryFunction::ConfirmIntersection)
            }
            Node::RayQueryCommitProcedural { query, t } => (
                self.ray_query(*query)?,
                RayQueryFunction::GenerateIntersection { hit_t: self.get(*t)? },
            ),
            _ => return Err(err("not a ray query operation")),
        };
        let query = self.expr(E::LocalVariable(q.query));
        self.statement(Statement::RayQuery { query, fun })
    }

    /// `RayQueryProceed`: the statement, and its result.
    pub(super) fn ray_query_proceed(&mut self, query: QueryId) -> Result<Handle<E>, Error> {
        let q = self.ray_query(query)?;
        let pointer = self.expr(E::LocalVariable(q.query));
        let result = self.expr(E::RayQueryProceedResult);
        self.statement(Statement::RayQuery {
            query: pointer,
            fun: RayQueryFunction::Proceed { result },
        })?;
        Ok(result)
    }

    /// `RayQueryState`: a field of the kept ray, or of the candidate or committed intersection.
    pub(super) fn ray_query_state(&mut self, query: QueryId, field: QueryField) -> Result<Handle<E>, Error> {
        let q = self.ray_query(query)?;
        let traced = |l: &mut Self, index: u32| -> Handle<E> {
            let base = l.expr(E::LocalVariable(q.ray));
            let pointer = l.expr(E::AccessIndex { base, index });
            l.expr(E::Load { pointer })
        };
        let lane = |l: &mut Self, base: Handle<E>, index: u8| {
            l.expr(E::AccessIndex {
                base,
                index: u32::from(index),
            })
        };
        let (committed, hit) = match field {
            QueryField::RayFlags => return Ok(traced(self, RAY_FLAGS)),
            QueryField::RayTMin => return Ok(traced(self, RAY_T_MIN)),
            QueryField::WorldRayOrigin(n) => {
                let v = traced(self, RAY_ORIGIN);
                return Ok(lane(self, v, n));
            }
            QueryField::WorldRayDirection(n) => {
                let v = traced(self, RAY_DIRECTION);
                return Ok(lane(self, v, n));
            }
            QueryField::CandidateNonOpaque => {
                return Err(err(
                    "whether a ray query's candidate procedural primitive is opaque is not lowered: naga's intersection does not say",
                ));
            }
            QueryField::CommittedStatus => (true, None),
            QueryField::CandidateType => (false, None),
            QueryField::Committed(h) => (true, Some(h)),
            QueryField::Candidate(h) => (false, Some(h)),
        };
        self.ray_intersection_type()?;
        let pointer = self.expr(E::LocalVariable(q.query));
        let intersection = self.expr(E::RayQueryGetIntersection {
            query: pointer,
            committed,
        });
        let at = |l: &mut Self, index: u32| {
            l.expr(E::AccessIndex {
                base: intersection,
                index,
            })
        };
        let Some(hit) = hit else {
            let kind = at(self, KIND);
            if committed {
                // nothing (0), a triangle (1), a procedural primitive (2): naga's kinds of a committed intersection
                return Ok(kind);
            }
            // a non-opaque triangle (0) or a procedural primitive (1): naga's candidate kinds are 1 and 3
            let triangle = self.literal_bits(iridia_ir::Scalar::U32, KIND_TRIANGLE)?;
            let is_triangle = self.binary(Bo::Equal, kind, triangle);
            let zero = self.literal_bits(iridia_ir::Scalar::U32, 0)?;
            let one = self.literal_bits(iridia_ir::Scalar::U32, 1)?;
            return Ok(self.select(is_triangle, zero, one));
        };
        Ok(match hit {
            HitField::T => at(self, T),
            HitField::InstanceIndex => at(self, INSTANCE_INDEX),
            HitField::InstanceId => at(self, INSTANCE_CUSTOM_DATA),
            HitField::GeometryIndex => at(self, GEOMETRY_INDEX),
            HitField::PrimitiveIndex => at(self, PRIMITIVE_INDEX),
            HitField::ContributionToHitGroup => at(self, SBT_RECORD_OFFSET),
            HitField::FrontFace => at(self, FRONT_FACE),
            HitField::Barycentrics(n) => {
                let pair = at(self, BARYCENTRICS);
                lane(self, pair, n)
            }
            // a 3 by 4 matrix's row and column: naga's is four columns of three rows
            HitField::ObjectToWorld { row, column } | HitField::WorldToObject { row, column } => {
                let index = if matches!(hit, HitField::ObjectToWorld { .. }) {
                    OBJECT_TO_WORLD
                } else {
                    WORLD_TO_OBJECT
                };
                let matrix = at(self, index);
                let column = lane(self, matrix, column);
                lane(self, column, row)
            }
            // the kept ray in the hit's object space, as Direct3D defines it: the hit's world-to-object transform of
            // the ray's origin (a point, w 1) or direction (w 0)
            HitField::ObjectRayOrigin(n) | HitField::ObjectRayDirection(n) => {
                let (member, w) = if matches!(hit, HitField::ObjectRayOrigin(_)) {
                    (RAY_ORIGIN, 1.0)
                } else {
                    (RAY_DIRECTION, 0.0)
                };
                let world = traced(self, member);
                let w = self.f32s(Type::F32, w)?;
                let ty = self.ty_of(Type::vector(iridia_ir::Scalar::F32, 4))?;
                let homogeneous = self.expr(E::Compose {
                    ty,
                    components: vec![world, w],
                });
                let transform = at(self, WORLD_TO_OBJECT);
                let object = self.binary(Bo::Multiply, transform, homogeneous);
                lane(self, object, n)
            }
        })
    }
}
