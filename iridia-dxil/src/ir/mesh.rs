//! A mesh program: a thread group that writes one meshlet, its counts (`SetMeshOutputCounts`), each primitive's
//! vertex indices (`EmitIndices`) and each vertex's outputs (`StoreVertexOutput`), as the IR's meshlet stores, and
//! reads its task payload (`GetMeshPayload`, a pointer to a variable of the payload space). An amplification program's
//! `DispatchMesh`, the IR's mesh launch of the variable its payload lives in.

use iridia_ir::{
    Node, Op, Scalar, Space, Type, Var,
    interface::{Stage, StageFacts},
};

use super::{Lift, Lv, Pointer};
use crate::Result;
use dxil_reader::module::ValueId;

impl Lift<'_> {
    /// The mesh `dx.op`s: `SetMeshOutputCounts` (168), `EmitIndices` (169), `GetMeshPayload` (170),
    /// `StoreVertexOutput` (171), `StorePrimitiveOutput` (172) and `DispatchMesh` (173).
    pub(crate) fn mesh_op(&mut self, code: u64, args: &[ValueId], result: Option<ValueId>) -> Result<()> {
        if code == 173 {
            return self.dispatch_mesh(args);
        }
        if self.setup.interface.stage != Stage::Mesh {
            return self.refuse("a mesh operation outside a mesh program");
        }
        match code {
            // SetMeshOutputCounts(vertices, primitives)
            168 => {
                let vertices = self.uint(self.arg(args, 1)?)?;
                let primitives = self.uint(self.arg(args, 2)?)?;
                self.g.node(Node::SetMeshCounts { vertices, primitives });
                Ok(())
            }
            // EmitIndices(primitive, the vertices of its topology)
            169 => {
                let StageFacts::Mesh(facts) = self.setup.interface.facts.clone() else {
                    return self.refuse("a mesh program without its facts");
                };
                let primitive = self.uint(self.arg(args, 1)?)?;
                let n = facts.topology.indices() as usize;
                let mut indices = Vec::with_capacity(n);
                for k in 0..n {
                    indices.push(self.uint(self.arg(args, 2usize.saturating_add(k))?)?);
                }
                let lanes =
                    u8::try_from(n).map_err(|_| crate::error(self.at, "a primitive of 256 vertices"))?;
                let indices = self
                    .g
                    .value(Type::vector(Scalar::U32, lanes), Op::Compose(indices));
                self.g.node(Node::StoreIndices { primitive, indices });
                Ok(())
            }
            // StoreVertexOutput(element, row, column, value, vertex)
            171 => {
                let (slot, position, value) = self.output_lane(args)?;
                let vertex = self.uint(self.arg(args, 5)?)?;
                self.g.node(Node::StoreVertex {
                    vertex,
                    slot,
                    mask: 1 << position,
                    value,
                });
                Ok(())
            }
            // GetMeshPayload(): a pointer to the payload, laid out in bytes
            170 => {
                let StageFacts::Mesh(facts) = self.setup.interface.facts.clone() else {
                    return self.refuse("a mesh program without its facts");
                };
                if facts.payload == 0 {
                    return self.refuse("a task payload read where the facts state none");
                }
                let var = match self.mesh_payload {
                    Some(v) => v,
                    None => {
                        let id = iridia_ir::VarId(u32::try_from(self.g.vars.len()).unwrap_or(u32::MAX));
                        self.g.vars.push(Var {
                            name: format!("payload{}", id.0),
                            ty: Type::U32,
                            len: Some(facts.payload.div_ceil(4)),
                            space: Space::Payload,
                            init: Vec::new(),
                        });
                        self.bytes.insert(id);
                        self.mesh_payload = Some(id);
                        id
                    }
                };
                let v = result.ok_or_else(|| crate::error(self.at, "GetMeshPayload of no result"))?;
                self.define(
                    v,
                    Lv::Pointer(Pointer {
                        var,
                        at: 0,
                        index: None,
                    }),
                )
            }
            172 => self.refuse("a mesh program's per-primitive outputs are not lifted onto the IR yet"),
            other => self.refuse(format!("mesh operation {other}")),
        }
    }

    /// `DispatchMesh(x, y, z, payload)`: the IR's mesh launch of the variable the payload is.
    fn dispatch_mesh(&mut self, args: &[ValueId]) -> Result<()> {
        if self.setup.interface.stage != Stage::Amplification {
            return self.refuse("DispatchMesh outside an amplification program");
        }
        let mut groups = Vec::with_capacity(3);
        for k in 1..=3 {
            groups.push(self.uint(self.arg(args, k)?)?);
        }
        let groups = self.g.value(Type::vector(Scalar::U32, 3), Op::Compose(groups));
        let payload = match self.get(self.arg(args, 4)?)? {
            Lv::Pointer(Pointer {
                var,
                at: 0,
                index: None,
            }) if self.bytes.contains(&var) => var,
            _ => return self.refuse("DispatchMesh of a payload that is not a whole variable"),
        };
        self.g.node(Node::DispatchMesh { groups, payload });
        Ok(())
    }
}
