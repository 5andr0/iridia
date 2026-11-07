//! Inline ray tracing (DXR 1.1's `RayQuery`, dx.op 178 to 212, 214 and 215): each `AllocateRayQuery` a query of the
//! function, declared with its constant flags, and its operations the IR's on that query. The IR's query holds the ray
//! it traces, so every state read is one operation (the naga-path lift keeps the ray in a local beside naga's query,
//! and refuses the reads of the ray in object space).

use iridia_ir::{HitField, Node, Op, QueryDecl, QueryField, QueryId, RayDesc, Type};

use super::{Lift, Lv};
use crate::Result;
use dxil_reader::module::{Instruction, ValueId};

impl Lift<'_> {
    /// `AllocateRayQuery(flags)`: a query of the function, declared with the constant flags.
    pub(crate) fn allocate_ray_query(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let Some(flags) = self.const_arg(args, 1).and_then(|f| u32::try_from(f).ok()) else {
            return self.refuse("a ray query declared with flags that are not constant");
        };
        let q = QueryId(u32::try_from(self.g.queries.len()).unwrap_or(u32::MAX));
        self.g.queries.push(QueryDecl { flags });
        self.define_result(inst, Lv::Query(q))
    }

    /// The query a handle value names.
    fn ray_query(&mut self, v: ValueId) -> Result<QueryId> {
        match self.get(v)? {
            Lv::Query(q) => Ok(q),
            _ => self.refuse("a ray query operation on something that is not a ray query"),
        }
    }

    /// A constant lane, row or column operand of a state read.
    fn query_index(&self, args: &[ValueId], n: usize) -> Result<u8> {
        self.const_arg(args, n)
            .and_then(|c| u8::try_from(c).ok())
            .map_or_else(|| self.refuse("a ray query field read at a computed index"), Ok)
    }

    /// The ray query operations: `TraceRayInline` (179), `Proceed` (180), `Abort` (181), the commits (182, 183) and
    /// the reads of the ray and of the candidate or committed hit (184 to 212, 214, 215).
    pub(crate) fn ray_query_op(&mut self, inst: &Instruction, code: u64, args: &[ValueId]) -> Result<()> {
        let query = self.ray_query(self.arg(args, 1)?)?;
        match code {
            // TraceRayInline(query, scene, flags, mask, origin, t min, direction, t max)
            179 => {
                let (scene, _) = self.handle(self.arg(args, 2)?)?;
                let flags = self.uint(self.arg(args, 3)?)?;
                let mask = self.uint(self.arg(args, 4)?)?;
                let origin = self.float3(args, 5)?;
                let t_min = self.float(self.arg(args, 8)?)?;
                let direction = self.float3(args, 9)?;
                let t_max = self.float(self.arg(args, 12)?)?;
                self.g.node(Node::RayQueryInit {
                    query,
                    scene,
                    ray: RayDesc {
                        flags,
                        mask,
                        origin,
                        t_min,
                        direction,
                        t_max,
                    },
                });
                Ok(())
            }
            180 => {
                let more = self.g.value(Type::BOOL, Op::RayQueryProceed(query));
                self.define_result(inst, Lv::Value(more))
            }
            181 => {
                self.g.node(Node::RayQueryAbort(query));
                Ok(())
            }
            182 => {
                self.g.node(Node::RayQueryCommitTriangle(query));
                Ok(())
            }
            183 => {
                let t = self.float(self.arg(args, 2)?)?;
                self.g.node(Node::RayQueryCommitProcedural { query, t });
                Ok(())
            }
            _ => {
                let field = self.query_field(code, args)?;
                let x = self.g.value(field.ty(), Op::RayQueryState(query, field));
                self.define_result(inst, Lv::Value(x))
            }
        }
    }

    /// The field a state read (184 to 212, 214, 215) reads.
    fn query_field(&self, code: u64, args: &[ValueId]) -> Result<QueryField> {
        let lane = |l: &Self| l.query_index(args, 2);
        // the candidate's fields, then the committed hit's
        let side = |h: HitField, committed: bool| {
            if committed {
                QueryField::Committed(h)
            } else {
                QueryField::Candidate(h)
            }
        };
        let field = match code {
            184 => QueryField::CommittedStatus,
            185 => QueryField::CandidateType,
            // a 3 by 4 matrix's row and column
            186..=189 => {
                let (row, column) = (lane(self)?, self.query_index(args, 3)?);
                let h = if matches!(code, 186 | 188) {
                    HitField::ObjectToWorld { row, column }
                } else {
                    HitField::WorldToObject { row, column }
                };
                side(h, code >= 188)
            }
            190 => QueryField::CandidateNonOpaque,
            191 | 192 => side(HitField::FrontFace, code == 192),
            193 | 194 => side(HitField::Barycentrics(lane(self)?), code == 194),
            195 => QueryField::RayFlags,
            196 => QueryField::WorldRayOrigin(lane(self)?),
            197 => QueryField::WorldRayDirection(lane(self)?),
            198 => QueryField::RayTMin,
            199 | 200 => side(HitField::T, code == 200),
            201 | 207 => side(HitField::InstanceIndex, code == 207),
            202 | 208 => side(HitField::InstanceId, code == 208),
            203 | 209 => side(HitField::GeometryIndex, code == 209),
            204 | 210 => side(HitField::PrimitiveIndex, code == 210),
            205 | 211 => side(HitField::ObjectRayOrigin(lane(self)?), code == 211),
            206 | 212 => side(HitField::ObjectRayDirection(lane(self)?), code == 212),
            214 | 215 => side(HitField::ContributionToHitGroup, code == 215),
            other => return self.refuse(format!("ray query dx.op {other}")),
        };
        if !field.in_range() {
            return self.refuse(format!(
                "ray query field {field:?} past its lanes, rows or columns"
            ));
        }
        Ok(field)
    }
}
