//! The stages whose programs read arrays of control points or vertices: hull, domain and geometry. One point's input
//! (`LoadInput`'s vertex), a hull program's output control points (`LoadOutputControlPoint`), the patch constants
//! (`LoadPatchConstant`, `StorePatchConstant`), the values the pipeline gives (`DomainLocation`, `PrimitiveID`,
//! `OutputControlPointID`, `GSInstanceID`) and a geometry program's vertices (`EmitStream`, `CutStream`), each as the
//! naga-path lift ([`crate::lift`]) reads it.

use iridia_ir::{
    Builtin, Node, Op, Scalar, Type, ValueId as IrValue,
    interface::{Stage, StageFacts, Varying},
};

use super::{Lift, Lv};
use crate::Result;
use crate::lift::stages::patch_cell;
use dxil_reader::module::{Instruction, ValueId};

impl Lift<'_> {
    /// Whether the program's inputs are arrays of points (a hull, domain or geometry program): `LoadInput` names the
    /// point it reads.
    pub(crate) fn reads_points(&self) -> bool {
        matches!(
            self.setup.interface.stage,
            Stage::Hull | Stage::Domain | Stage::Geometry
        )
    }

    /// The patch constants of the program: a hull program's outputs, a domain program's inputs.
    fn patch_constants(&self) -> &[Varying] {
        match &self.setup.interface.facts {
            StageFacts::Hull(h) => &h.patch_constants,
            StageFacts::Domain(d) => &d.patch_constants,
            _ => &[],
        }
    }

    /// A lane read from a varying of scalar `scalar` as the call's result type: a flag read as an integer is all ones
    /// where it holds.
    pub(crate) fn varying_result(&mut self, inst: &Instruction, x: IrValue, scalar: Scalar) -> Result<()> {
        let want = inst
            .result
            .map(|(_, ty)| self.carried_type(ty))
            .transpose()?
            .flatten()
            .ok_or_else(|| crate::error(self.at, "a varying read of no type"))?;
        let r = if scalar == Scalar::Bool && want.scalar != Scalar::Bool {
            let (on, off) = (self.u32_lit(u32::MAX), self.u32_lit(0));
            let r = self.select(Type::U32, x, on, off);
            self.retype(r, want)?
        } else {
            self.retype(x, want)?
        };
        self.define_result(inst, Lv::Value(r))
    }

    /// The patch-constant element a call's arguments 1 to 3 name: the patch-constant varying, and the lane of it.
    fn patch_lane(&self, args: &[ValueId], what: &str) -> Result<(usize, u8)> {
        let (Some(id), Some(row), Some(col)) = (
            self.const_arg(args, 1),
            self.const_arg(args, 2),
            self.const_arg(args, 3),
        ) else {
            return self.refuse(format!(
                "a patch constant {what} at a computed element, row or column"
            ));
        };
        let Some(element) = self
            .setup
            .patch_elements
            .iter()
            .find(|e| i64::from(e.id) == id)
            .cloned()
        else {
            return self.refuse(format!("patch constant element {id}"));
        };
        let (register, lane) = self.cell_of(element.row, element.column, row, col)?;
        match patch_cell(self.patch_constants(), register, lane) {
            Some(cell) => Ok(cell),
            None => self.refuse(format!(
                "patch constant {} has no lane at register {register}",
                element.name
            )),
        }
    }

    /// `LoadPatchConstant(element, row, column)` in a domain program: one lane of a patch constant.
    pub(crate) fn load_patch_constant(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        if self.setup.interface.stage != Stage::Domain {
            return self.refuse("a patch constant read outside a domain program");
        }
        let (varying, position) = self.patch_lane(args, "read")?;
        let scalar = self
            .patch_constants()
            .get(varying)
            .map(|v| iridia_ir::value_type(v.value).scalar)
            .ok_or_else(|| crate::error(self.at, "a patch constant past the interface"))?;
        let slot =
            u32::try_from(varying).map_err(|_| crate::error(self.at, "a patch constant slot past 2^32"))?;
        let x = self
            .g
            .value(Type::scalar(scalar), Op::LoadPatch(slot, 1 << position));
        self.varying_result(inst, x, scalar)
    }

    /// `StorePatchConstant(element, row, column, value)` in a hull program's patch-constant function: one lane of a
    /// patch constant.
    pub(crate) fn store_patch_constant(&mut self, args: &[ValueId]) -> Result<()> {
        if self.setup.interface.stage != Stage::Hull {
            return self.refuse("a patch constant written outside a hull program");
        }
        let (varying, position) = self.patch_lane(args, "written")?;
        let scalar = self
            .patch_constants()
            .get(varying)
            .map(|v| iridia_ir::value_type(v.value).scalar)
            .ok_or_else(|| crate::error(self.at, "a patch constant past the interface"))?;
        let x = self.value(self.arg(args, 4)?)?;
        let x = self.flag_bits(x);
        let x = self.retype(x, Type::scalar(scalar))?;
        let slot =
            u32::try_from(varying).map_err(|_| crate::error(self.at, "a patch constant slot past 2^32"))?;
        self.g.node(Node::StorePatch(slot, 1 << position, x));
        Ok(())
    }

    /// `x`, a flag written to a varying, as D3D's all ones or zero; any other value as it is.
    pub(crate) fn flag_bits(&mut self, x: IrValue) -> IrValue {
        if self.g.ty(x).is_some_and(|t| t.scalar == Scalar::Bool) {
            let (on, off) = (self.u32_lit(u32::MAX), self.u32_lit(0));
            self.select(Type::U32, x, on, off)
        } else {
            x
        }
    }

    /// `LoadOutputControlPoint(element, row, column, point)` in a hull program's patch-constant function: one lane of
    /// an output control point.
    pub(crate) fn load_output_control_point(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        if self.setup.interface.stage != Stage::Hull {
            return self.refuse("an output control point read outside a hull program");
        }
        let (Some(id), Some(row), Some(col)) = (
            self.const_arg(args, 1),
            self.const_arg(args, 2),
            self.const_arg(args, 3),
        ) else {
            return self.refuse("an output control point read at a computed element, row or column");
        };
        let Some(element) = self
            .setup
            .output_elements
            .iter()
            .find(|e| i64::from(e.id) == id)
            .cloned()
        else {
            return self.refuse(format!("output element {id}"));
        };
        let (register, lane) = self.cell_of(element.row, element.column, row, col)?;
        let Some((varying, position)) = self.setup.output_cell(register, lane) else {
            return self.refuse(format!(
                "output {} has no lane at register {register}",
                element.name
            ));
        };
        let scalar = self
            .setup
            .interface
            .outputs
            .get(varying)
            .map(|v| iridia_ir::value_type(v.value).scalar)
            .ok_or_else(|| crate::error(self.at, "an output past the interface"))?;
        let point = self.uint(self.arg(args, 4)?)?;
        let slot = u32::try_from(varying).map_err(|_| crate::error(self.at, "an output slot past 2^32"))?;
        let x = self
            .g
            .value(Type::scalar(scalar), Op::LoadOutputAt(slot, 1 << position, point));
        self.varying_result(inst, x, scalar)
    }

    /// `DomainLocation(component)`: a lane of the point's place in its patch.
    pub(crate) fn domain_location(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let Some(component) = self
            .const_arg(args, 1)
            .and_then(|c| u8::try_from(c).ok())
            .filter(|c| *c < 3)
        else {
            return self.refuse("a domain location read at a computed component");
        };
        let location = self
            .g
            .value(Type::vector(Scalar::F32, 3), Op::Builtin(Builtin::DomainLocation));
        let lane = self
            .g
            .value(Type::F32, Op::Swizzle(location, [component, 0, 0, 0]));
        self.varying_result(inst, lane, Scalar::F32)
    }

    /// `PrimitiveID`, `OutputControlPointID` and `GSInstanceID`: a `u32` the pipeline gives.
    pub(crate) fn pipeline_value(&mut self, inst: &Instruction, builtin: Builtin) -> Result<()> {
        let x = self.g.value(Type::U32, Op::Builtin(builtin));
        self.varying_result(inst, x, Scalar::U32)
    }

    /// `EmitStream`, `CutStream` and `EmitThenCutStream` (97 to 99) on stream 0: the outputs as they are stored now
    /// become the next vertex, and the next vertex after a cut begins a strip.
    pub(crate) fn stream_op(&mut self, code: u64, args: &[ValueId]) -> Result<()> {
        if self.setup.interface.stage != Stage::Geometry {
            return self.refuse("a vertex emitted outside a geometry program");
        }
        if self.const_arg(args, 1) != Some(0) {
            return self.refuse("a vertex emitted to a stream other than 0");
        }
        if code != 98 {
            self.g.node(Node::Emit(0));
        }
        if code != 97 {
            self.g.node(Node::Cut(0));
        }
        Ok(())
    }
}
