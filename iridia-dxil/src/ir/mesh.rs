//! A mesh program: a thread group that writes one meshlet, its counts (`SetMeshOutputCounts`), each primitive's
//! vertex indices (`EmitIndices`) and each vertex's outputs (`StoreVertexOutput`), as the IR's meshlet stores.

use iridia_ir::{
    Node, Op, Scalar, Type,
    interface::{Stage, StageFacts},
};

use super::Lift;
use crate::Result;
use dxil_reader::module::ValueId;

impl Lift<'_> {
    /// The mesh `dx.op`s: `SetMeshOutputCounts` (168), `EmitIndices` (169), `GetMeshPayload` (170),
    /// `StoreVertexOutput` (171), `StorePrimitiveOutput` (172) and `DispatchMesh` (173).
    pub(crate) fn mesh_op(&mut self, code: u64, args: &[ValueId]) -> Result<()> {
        if self.setup.interface.stage != Stage::Mesh && code != 173 {
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
            170 => self.refuse("a mesh program's task payload is not lifted onto the IR yet"),
            172 => self.refuse("a mesh program's per-primitive outputs are not lifted onto the IR yet"),
            173 => self.refuse("DispatchMesh is not lifted onto the IR yet"),
            other => self.refuse(format!("mesh operation {other}")),
        }
    }
}
