//! Derived programs: a pixel program cut after one of its instructions, and a vertex output run as an invocation of its
//! own. No meaning is assigned to the chosen value: recognizing a useful cut is the caller's business. Each derived
//! program keeps the source hash of the complete original bytecode, and states what it is in its lifter identity and notes.

use std::collections::BTreeMap;

use iridia_ir::{Op, Program, Scalar, Type, interface::Stage};

use super::{Lift, Options};
use crate::{Instruction, Opcode, Register, Result, error, signature::varying};

/// A register value immediately after one original instruction. `lanes` order is the projected RGB order.
/// The word offset must identify an instruction, and all three distinct temporary lanes must be initialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Projection {
    /// Offset, in 32-bit words from the version token, of the instruction the value is read after.
    pub after_word: usize,
    /// Index of the temporary register `rN` to read.
    pub temporary: u32,
    /// Which lanes (0 x, 1 y, 2 z, 3 w) of the temporary become red, green and blue; three distinct lanes, each already
    /// written by the cut.
    pub lanes: [u8; 3],
}

/// What a lift derives from its program.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Derive {
    /// A cut after one instruction: `false` keeps the prefix and gives its lanes as RGB and alpha one, `true` keeps the
    /// whole program and replaces the cut's lanes with a new input.
    pub(super) cut: Option<(Projection, bool)>,
    /// Pad the colour output's lanes the program does not write with (0, 0, 0, 1).
    pub(super) pad_output: bool,
    /// The rows of the float constant bank a relative read reaches, where the program's stage does not say: a vertex
    /// program's slice runs as a pixel invocation and reads the vertex constant file.
    pub(super) float_bank: Option<u32>,
}

impl Lift<'_> {
    /// Whether `i` is the instruction a cut follows, and where it is, the cut is made: the prefix is the program (the
    /// lift stops) or the cut's lanes are replaced by an input's. A cut inside flow control or a subroutine is refused.
    pub(super) fn cut_after(&mut self, i: &Instruction) -> Result<bool> {
        let Some((cut, tail)) = self.derive.cut else {
            return Ok(false);
        };
        if i.word != cut.after_word {
            return Ok(false);
        }
        if !self.flow.is_empty() || !self.calls.is_empty() {
            return Err(error(
                cut.after_word,
                "a projection after an instruction inside flow control or a subroutine",
            ));
        }
        self.flush()?;
        let mask = cut.lanes.iter().fold(0u8, |m, n| m | (1 << n));
        let value = self.register(Register::Temporary(cut.temporary), mask)?;
        self.notes.push(format!(
            "derived register projection after word {}: r{} lanes {:?}; {}",
            cut.after_word,
            cut.temporary,
            cut.lanes,
            if tail {
                "original tail with replacement input"
            } else {
                "prefix only; alpha one"
            }
        ));
        if tail {
            let slot = u32::try_from(self.signature.interface.inputs.len())
                .map_err(|_| error(cut.after_word, "more than 4294967295 inputs"))?;
            let mut input = varying("DERIVED_RGB", 0, 7, 31, None);
            input.used = true;
            self.signature.interface.inputs.push(input);
            let arg = self
                .b
                .value(Type::vector(Scalar::F32, 3), Op::LoadInput(slot, 0b111));
            let mut parts = Vec::new();
            for n in 0u8..4 {
                parts.push(self.lane(value, n)?);
            }
            for (n, &lane) in (0u8..).zip(&cut.lanes) {
                let part = self.lane(arg, n)?;
                *parts.get_mut(usize::from(lane)).ok_or_else(|| {
                    error(
                        cut.after_word,
                        format!("projection lane {lane}, where a register has lanes 0 to 3"),
                    )
                })? = part;
            }
            let replaced = self.op(Op::Compose(parts))?;
            self.store(Register::Temporary(cut.temporary), mask, replaced, None)?;
            return Ok(false);
        }
        let mut parts = Vec::new();
        for &lane in &cut.lanes {
            parts.push(self.lane(value, lane)?);
        }
        parts.push(self.b.f32(1.0));
        let rgb = self.op(Op::Compose(parts))?;
        self.registers.retain(|r, _| !matches!(r, Register::Target(_)));
        self.store(Register::Target(0), 15, rgb, None)?;
        Ok(true)
    }

    /// The colour output's lanes the program does not write set to (0, 0, 0, 1), after the last instruction.
    pub(super) fn pad_output(&mut self) -> Result<()> {
        if !self.derive.pad_output {
            return Ok(());
        }
        let &(_, mask) = self
            .registers
            .get(&Register::Target(0))
            .ok_or_else(|| error(0, "projected output has no writes"))?;
        if mask != 15 {
            let zero = self.b.f32(0.0);
            let one = self.b.f32(1.0);
            let padding = self.op(Op::Compose(vec![zero, zero, zero, one]))?;
            self.store(Register::Target(0), 15 & !mask, padding, None)?;
        }
        Ok(())
    }
}

/// Pixel-only: lift the original instruction prefix through `cut`, returning its selected lanes as RGB and alpha one.
/// Prior discards remain; subsequent operations and discards are excluded. This is a derived program, whose notes name
/// the cut, whose lifter identity includes the projection, and whose source hash still identifies the complete original
/// bytecode.
pub fn project(bytes: &[u8], cut: Projection) -> Result<Program> {
    project_with(bytes, cut, &Options::default())
}

/// [`project`] with the stated [`Options`].
pub fn project_with(bytes: &[u8], cut: Projection, options: &Options) -> Result<Program> {
    projected(bytes, cut, false, options)
}

/// Pixel-only: lift the complete original program, replacing just the cut's three register lanes with the new float3
/// input `DERIVED_RGB0` (signature register 31). This runs the original downstream operations, including normalization
/// and discards, rather than a rewritten equation. Other register lanes and the prefix remain. An original input v31 is
/// refused. The caller supplies projected RGB in that new input's xyz lanes.
pub fn reencode(bytes: &[u8], cut: Projection) -> Result<Program> {
    reencode_with(bytes, cut, &Options::default())
}

/// [`reencode`] with the stated [`Options`].
pub fn reencode_with(bytes: &[u8], cut: Projection, options: &Options) -> Result<Program> {
    projected(bytes, cut, true, options)
}

fn projected(bytes: &[u8], cut: Projection, tail: bool, options: &Options) -> Result<Program> {
    let source = crate::read(bytes)?;
    if source.stage != Stage::Pixel {
        return Err(error(cut.after_word, "register RGB projections are pixel-only"));
    }
    if source.version.major != 3 {
        return Err(error(
            cut.after_word,
            format!("register RGB projections are of ps_3_0, not {}", source.version),
        ));
    }
    if cut.lanes.iter().any(|&n| n > 3)
        || cut.lanes[0] == cut.lanes[1]
        || cut.lanes[0] == cut.lanes[2]
        || cut.lanes[1] == cut.lanes[2]
        || !source.instructions.iter().any(|i| i.word == cut.after_word)
    {
        return Err(error(
            cut.after_word,
            "invalid projection lanes or instruction boundary",
        ));
    }
    if tail && source.inputs.iter().any(|i| i.register == Register::Input(31)) {
        return Err(error(
            cut.after_word,
            "projection input register v31 already occupied",
        ));
    }
    let derive = Derive {
        cut: Some((cut, tail)),
        pad_output: false,
        float_bank: None,
    };
    let mut program = super::lift_shader(&source, bytes, options, derive)?;
    // A derived value and its original program share the source hash, but are distinct cache artifacts.
    program.provenance.lifter = format!(
        "{}/{}:word={}:r{}:{:?}",
        super::LIFTER,
        if tail { "reencode" } else { "projection" },
        cut.after_word,
        cut.temporary,
        cut.lanes
    );
    Ok(program)
}

/// Project selected lanes of a declared `vs_3_0` output into a derived invocation program.
///
/// The complete original program must lift successfully. A backwards register/lane slice retains only the writes that
/// reach the selected output. It preserves original arithmetic, swizzles, masks and local `def` values; it does not
/// interpret the output semantic. The result is pixel-stage IR for an independent invocation, with original vertex inputs
/// and constant-register bindings. `Target0` keeps selected lanes in their original RGBA positions; other lanes are
/// explicitly placeholder (0, 0, 0, 1). Other original outputs are excluded. Full original source hash, a distinct lifter
/// identity and a derivation note keep this separate from the vertex program. Its original declarations remain in the
/// interface; `used` and graph dependencies describe actual reads.
///
/// A program with flow control is refused: the slice follows straight-line register writes.
pub fn project_vertex_output(bytes: &[u8], register: u32, mask: u8) -> Result<Program> {
    project_vertex_output_with(bytes, register, mask, &Options::default())
}

/// [`project_vertex_output`] with the stated [`Options`].
pub fn project_vertex_output_with(
    bytes: &[u8],
    register: u32,
    mask: u8,
    options: &Options,
) -> Result<Program> {
    let mut source = crate::read(bytes)?;
    if source.stage != Stage::Vertex {
        return Err(error(0, "vertex output projection requires vs_3_0"));
    }
    if source.version.major != 3 {
        return Err(error(
            0,
            format!("vertex output projection is of vs_3_0, not {}", source.version),
        ));
    }
    let declared = source
        .outputs
        .iter()
        .filter(|o| o.register == Register::Output(register))
        .fold(0, |m, o| m | o.mask);
    if mask == 0 || mask & !declared != 0 {
        return Err(error(0, "vertex output projection mask is empty or not declared"));
    }
    if let Some(i) = source.instructions.iter().find(|i| flow_control(i.opcode)) {
        return Err(error(
            i.word,
            format!("vertex output projection of a program with {}", i.opcode.name()),
        ));
    }
    // Validate the full program first: slicing never hides an unsupported instruction or invalid dead output.
    super::lift_shader(&source, bytes, options, Derive::default())?;
    let root = Register::Output(register);
    let mut live = BTreeMap::from([(root, mask)]);
    let mut kept = Vec::new();
    for instruction in source.instructions.iter().rev() {
        let Some(mut destination) = instruction.destination else {
            continue;
        };
        let needed = live.get(&destination.register).copied().unwrap_or(0);
        let written = needed & destination.mask;
        if written == 0 {
            continue;
        }
        live.insert(destination.register, needed & !destination.mask);
        destination.mask = written;
        let mut selected = instruction.clone();
        selected.destination = Some(destination);
        for (operand, &s) in selected.sources.iter().enumerate() {
            *live.entry(s.register).or_default() |= s.read_mask(selected.source_mask(operand));
            // a relative constant also reads the lane of a0 it is addressed by
            if let Some(crate::Index::Address(lane)) = s.relative {
                *live.entry(Register::Address).or_default() |= 1 << lane;
            }
        }
        if destination.register == root
            && let Some(d) = selected.destination.as_mut()
        {
            d.register = Register::Target(0);
        }
        kept.push(selected);
    }
    kept.reverse();
    source.instructions = kept;
    source.outputs.clear();
    source.stage = Stage::Pixel;
    let derive = Derive {
        cut: None,
        pad_output: true,
        float_bank: Some(crate::VERTEX_CONSTANTS),
    };
    let mut program = super::lift_shader(&source, bytes, options, derive)?;
    program.provenance.target = "vs_3_0/output-invocation".into();
    program.provenance.lifter = format!("{}/vertex-output:o{register}:mask={mask}", super::LIFTER);
    program.notes.push(format!("derived invocation of original vs_3_0 o{register}, mask {mask:#x}; selected lanes keep original RGBA positions; unselected lanes are placeholders (0,0,0,1), not source outputs"));
    program.notes.sort();
    Ok(program)
}

/// Whether the opcode is flow control: it changes which instructions run, or where the program ends.
fn flow_control(opcode: Opcode) -> bool {
    use Opcode::*;
    matches!(
        opcode,
        If | IfC
            | Else
            | EndIf
            | Rep
            | EndRep
            | Loop
            | EndLoop
            | Break
            | BreakC
            | BreakP
            | Call
            | CallNz
            | Ret
            | Label
    )
}
