//! The stages whose programs read arrays of control points or vertices: hull, domain and geometry. The registers that
//! hold one point's varyings (`v[point][register]`, `vicp`, `vocp`) and the patch constants (`vpc`), the values the
//! pipeline gives (`vOutputControlPointID`, `vDomain`, `vPrim`, `vGSInstanceID`, a hull phase's instance), a geometry
//! program's vertices (`emit`, `cut`), and a hull program's two functions: its control-point phase as the entry (the
//! input points passed through where it has none), and its fork and join phases, in order, as the patch constant
//! function, each phase's instances a loop whose counter is the phase's instance number.

use iridia_ir::{
    Binary, Builder, Builtin, Cmp, End, Function, Op, Scalar, Type, ValueId,
    interface::{StageFacts, Varying},
};

use super::{Exit, Lift, every, lanes_of, scalar_of};
use crate::Result;
use crate::lift::setup::PhaseKind;
use dxbc_reader::decode::{Instruction, Operand, Register};

/// Where a control point register's varyings are: the input points, or a hull program's output points.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Points {
    Input,
    Output,
}

/// The registers a list of varyings is placed in, in order.
fn registers_of(list: &[Varying]) -> Vec<u32> {
    let mut r: Vec<u32> = list
        .iter()
        .filter_map(|v| v.provenance.d3d.map(|d| d.register))
        .filter(|&r| r != u32::MAX)
        .collect();
    r.sort_unstable();
    r.dedup();
    r
}

impl Lift<'_> {
    /// Whether the program's input registers are arrays of points (a hull, domain or geometry program).
    pub(super) fn reads_points(&self) -> bool {
        matches!(
            self.stage,
            iridia_ir::interface::Stage::Hull
                | iridia_ir::interface::Stage::Domain
                | iridia_ir::interface::Stage::Geometry
        )
    }

    /// `value`, a varying's `width` lanes, as raw 32-bit lanes: a bool as D3D's all ones or zero.
    pub(super) fn raw_bits(&mut self, value: ValueId, width: u8) -> Result<ValueId> {
        let ty = self.ty(value)?;
        if ty.scalar == Scalar::Bool {
            let ones = self
                .b
                .literal(Type::vector(Scalar::U32, width), [u64::from(u32::MAX); 4]);
            let none = self.b.literal(Type::vector(Scalar::U32, width), [0; 4]);
            self.select(value, ones, none)
        } else {
            Ok(self.b.value(Type::vector(Scalar::U32, width), Op::Bitcast(value)))
        }
    }

    /// Register `register`'s four raw lanes composed from the varyings of `list` it holds, each read by `read` (its
    /// slot and lane mask); zero where it holds none.
    fn compose_register(
        &mut self,
        list: &[Varying],
        register: u32,
        read: &dyn Fn(u32, u8) -> Op,
    ) -> Result<ValueId> {
        let zero = self.b.u32(0);
        let mut row = [zero; 4];
        for (slot, v) in (0u32..).zip(list) {
            let Some(d3d) = v.provenance.d3d.filter(|d| d.register == register) else {
                continue;
            };
            let width = v.value.width;
            let value = self.b.value(
                Type::vector(scalar_of(v.value.scalar), width),
                read(slot, every(width)),
            );
            let bits = self.raw_bits(value, width)?;
            for (k, l) in (0u8..).zip(lanes_of(d3d.mask)).take(usize::from(width)) {
                let lane = if width == 1 { bits } else { self.lane(bits, k)? };
                if let Some(at) = row.get_mut(usize::from(l)) {
                    *at = lane;
                }
            }
        }
        self.op(Op::Compose(row.to_vec()))
    }

    /// The row `index` names among `rows` (by register number): a fixed index its row, a computed one chosen among
    /// them; zero for an index none has.
    fn choose_row(&mut self, rows: Vec<(u32, ValueId)>, index: ValueId) -> Result<ValueId> {
        let mut v = self.splat_bits(Scalar::U32, 4, 0);
        for (r, row) in rows {
            let number = self.b.u32(r);
            let here = self.binary(Binary::UCmp(Cmp::Eq), index, number)?;
            v = self.select(here, row, v)?;
        }
        Ok(v)
    }

    /// A control point's (or a geometry program's input vertex's) register `v[point][register]`, `vicp` or `vocp`.
    pub(super) fn point_register(&mut self, op: &Operand, points: Points) -> Result<ValueId> {
        if op.indices.len() != 2 {
            return self.refuse(format!(
                "{:?} with {} indices, where a point and a register are taken",
                op.register,
                op.indices.len()
            ));
        }
        let point = self.index_value(&self.index_of(op, 0)?.clone())?;
        let list = match points {
            Points::Input => self.setup.interface.inputs.clone(),
            Points::Output => self.setup.interface.outputs.clone(),
        };
        let read = move |slot: u32, mask: u8| match points {
            Points::Input => Op::LoadInputAt(slot, mask, point),
            Points::Output => Op::LoadOutputAt(slot, mask, point),
        };
        let register = self.index_of(op, 1)?.clone();
        if register.relative.is_none() {
            return self.compose_register(&list, register.base, &read);
        }
        let index = self.index_value(&register)?;
        let mut rows = Vec::new();
        for r in registers_of(&list) {
            rows.push((r, self.compose_register(&list, r, &read)?));
        }
        self.choose_row(rows, index)
    }

    /// A patch constant register `vpc[register]`: a domain program's input, or in a hull program's join phase the
    /// output register the fork phases wrote.
    pub(super) fn patch_register(&mut self, op: &Operand) -> Result<ValueId> {
        let register = self.index_of(op, 0)?.clone();
        let list = match &self.setup.interface.facts {
            StageFacts::Domain(d) => d.patch_constants.clone(),
            StageFacts::Hull(h) => h.patch_constants.clone(),
            _ => return self.refuse("a patch constant read in a program that has none"),
        };
        let row = |l: &mut Self, r: u32| -> Result<ValueId> {
            if l.patch_fn {
                match l.outputs.get(&r).copied() {
                    Some(o) => Ok(l.b.value(Type::U32X4, Op::LoadVar(o))),
                    None => Ok(l.splat_bits(Scalar::U32, 4, 0)),
                }
            } else {
                l.compose_register(&list, r, &|slot, mask| Op::LoadPatch(slot, mask))
            }
        };
        if register.relative.is_none() {
            return row(self, register.base);
        }
        let index = self.index_value(&register)?;
        let mut rows = Vec::new();
        for r in registers_of(&list) {
            rows.push((r, row(self, r)?));
        }
        self.choose_row(rows, index)
    }

    /// A register whose first lanes are a value the pipeline gives (the rest zero).
    pub(super) fn pipeline_value(&mut self, register: Register) -> Result<ValueId> {
        let zero = self.b.u32(0);
        let builtin = match register {
            Register::OutputControlPointId => Builtin::OutputControlPoint,
            Register::PrimitiveId => Builtin::PrimitiveId,
            Register::GsInstanceId => Builtin::GsInstance,
            Register::InputDomainPoint => {
                let location = self
                    .b
                    .value(Type::vector(Scalar::F32, 3), Op::Builtin(Builtin::DomainLocation));
                let bits = self.raw_bits(location, 3)?;
                let mut lanes = Vec::new();
                for k in 0..3 {
                    lanes.push(self.lane(bits, k)?);
                }
                lanes.push(zero);
                return self.op(Op::Compose(lanes));
            }
            Register::ForkInstanceId | Register::JoinInstanceId => {
                let kind = if register == Register::ForkInstanceId {
                    PhaseKind::Fork
                } else {
                    PhaseKind::Join
                };
                return match self.phase_instance {
                    Some((k, four)) if k == kind => Ok(four),
                    _ => self.refuse(format!("{register:?} read outside a {kind:?} phase")),
                };
            }
            other => return self.refuse(format!("{other:?} is not a value the pipeline gives")),
        };
        let value = self.b.value(Type::U32, Op::Builtin(builtin));
        self.op(Op::Compose(vec![value, zero, zero, zero]))
    }

    /// `emit`, `cut` and `emitthencut` (and their `_stream` forms): a geometry program's vertex is its outputs as they
    /// are stored now.
    pub(super) fn emit_or_cut(&mut self, ins: &Instruction, name: &str) -> Result<()> {
        let stream = match ins.operands.first() {
            Some(m) => self.index_of(m, 0)?.base,
            None => 0,
        };
        if stream != 0 {
            return self.refuse(format!("a vertex of stream {stream}: only stream 0 is carried"));
        }
        if name.starts_with("emit") {
            self.store_outputs()?;
            self.b.emit(0);
        }
        if name.starts_with("cut") || name.starts_with("emitthencut") {
            self.b.cut(0);
        }
        Ok(())
    }

    /// End the function being built (its outputs stored and returning, where its code does not end otherwise), and
    /// start the builder of the next, `next`.
    pub(super) fn end_function(&mut self, next: &str) -> Result<Function> {
        if let Some(frame) = self.frames.last() {
            return self.refuse(format!("{frame:?} is not closed at the end of the function"));
        }
        let end = match self.ended.take() {
            Some((end, _)) => end,
            None if self.exit_kind == Exit::Bare => End::Return,
            None => {
                self.epilogue()?;
                End::Return
            }
        };
        std::mem::replace(&mut self.b, Builder::new(next))
            .finish(end)
            .map_err(|e| crate::error(self.word, e))
    }

    /// Begin another function of the program: the registers and resources declared again in it.
    fn begin_function(&mut self) -> Result<()> {
        self.temps.clear();
        self.indexable.clear();
        self.inputs.clear();
        self.outputs.clear();
        self.specials.clear();
        self.depth = None;
        self.coverage = None;
        self.resources = super::resources::Resources::default();
        self.frames.clear();
        self.ended = None;
        self.declare()
    }

    /// A hull program's two functions: the control-point phase (or the points passed through) and the patch constant
    /// function its fork and join phases make.
    pub(super) fn hull(&mut self) -> Result<(Function, Function)> {
        let phases = self.setup.phases.clone();
        self.declare()?;
        match phases.iter().find(|p| p.kind == PhaseKind::ControlPoint) {
            Some(p) => self.body_range(p.words.clone())?,
            None => self.pass_through()?,
        }
        let entry = self.end_function("patch_constant")?;
        self.patch_fn = true;
        self.begin_function()?;
        for phase in phases.iter().filter(|p| p.kind != PhaseKind::ControlPoint) {
            let zero = self.b.u32(0);
            let (label, params) = self
                .b
                .open_loop(&[zero])
                .map_err(|e| crate::error(self.word, e))?;
            let [instance] = params[..] else {
                return self.refuse("a phase loop without its counter");
            };
            let count = self.b.u32(phase.instances);
            let done = self.binary(Binary::UCmp(Cmp::Ge), instance, count)?;
            self.b.open_if(done);
            self.b
                .close(End::Br(label, vec![]), &[])
                .map_err(|e| crate::error(self.word, e))?;
            let four = self.op(Op::Compose(vec![instance, zero, zero, zero]))?;
            self.phase_instance = Some((phase.kind, four));
            let block = self.b.open_block();
            self.exit_kind = Exit::Phase(block);
            self.body_range(phase.words.clone())?;
            if let Some(frame) = self.frames.last() {
                return self.refuse(format!("a hull phase that ends inside {frame:?}"));
            }
            let end = match self.ended.take() {
                Some((end, _)) => end,
                None => End::Yield(vec![]),
            };
            self.b.close(end, &[]).map_err(|e| crate::error(self.word, e))?;
            let one = self.b.u32(1);
            let next = self.binary(Binary::IAdd, instance, one)?;
            self.b
                .close(End::Continue(label, vec![next]), &[])
                .map_err(|e| crate::error(self.word, e))?;
        }
        self.phase_instance = None;
        self.exit_kind = Exit::Epilogue;
        let patch = self.end_function("main")?;
        Ok((entry, patch))
    }

    /// A hull program without a control-point phase: each output point is the input point of its number, an output
    /// register's lanes taking the input varying placed in the same register lanes. The registers are what the
    /// epilogue stores as the outputs, as a control-point phase's writes are.
    fn pass_through(&mut self) -> Result<()> {
        let point = self.b.value(Type::U32, Op::Builtin(Builtin::OutputControlPoint));
        let inputs = self.setup.interface.inputs.clone();
        let outputs = self.setup.interface.outputs.clone();
        for o in &outputs {
            let Some(d3d) = o.provenance.d3d else {
                return self.refuse(format!("output {} has no register", o.label.name));
            };
            let found = (0u32..).zip(&inputs).find(|(_, i)| {
                i.provenance
                    .d3d
                    .is_some_and(|x| x.register == d3d.register && x.mask == d3d.mask)
            });
            let Some((from, i)) = found else {
                return self.refuse(format!(
                    "output {} of a hull program with no control-point phase is no input's",
                    o.label.name
                ));
            };
            let Some(&var) = self.outputs.get(&d3d.register) else {
                return self.refuse(format!("output register o{} is not declared", d3d.register));
            };
            let lanes = lanes_of(d3d.mask);
            let read = self.b.value(
                Type::vector(scalar_of(i.value.scalar), i.value.width),
                Op::LoadInputAt(from, every(i.value.width), point),
            );
            let bits = self.raw_bits(read, i.value.width)?;
            let bits = self.swizzle(bits, &(0u8..).take(lanes.len()).collect::<Vec<_>>())?;
            self.store_lanes(var, d3d.mask, bits)?;
        }
        Ok(())
    }
}
