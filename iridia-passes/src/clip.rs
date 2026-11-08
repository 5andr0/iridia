//! Clip distances carried as plain varyings, for a pipeline whose clipper cannot take them (WebGPU without its
//! `clip-distances` feature).
//!
//! [`clip_distances_as_varyings`] makes each `SV_ClipDistance` output of a vertex program an ordinary varying of the
//! same label and width, perspective-interpolated; [`discard_clipped`] gives the pixel program a matching input for
//! each and a discard before each of its returns where any lane is negative. A distance linear in the clip position
//! interpolates perspective-correct to its value at the pixel, so this keeps what a clipper keeps, decided at pixel
//! centres. A varying is linked by its label (as Direct3D links stages), so a writer that links the two programs
//! finds each pair without being told where it sits.

use alloc::{format, string::String, vec, vec::Vec};

use iridia_ir::{
    Binary, Body, Cmp, End, Function, Node, Op, Precision, Program, Region, Type, ValueDef, ValueId,
    interface::{
        InterpMode, InterpSampling, Interpolation, Label, ScalarType, Stage, SystemValue, Value, Varying,
        VaryingProvenance,
    },
};

/// The varyings a vertex program carries its clip distances in: each one's label and width (its lanes are the distances).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClipVaryings {
    /// The varyings, in the interface's order.
    pub varyings: Vec<(Label, u8)>,
}

impl ClipVaryings {
    /// How many distances there are, over all the varyings.
    pub fn count(&self) -> u32 {
        self.varyings.iter().map(|(_, w)| u32::from(*w)).sum()
    }
}

fn same_label(a: &Label, b: &Label) -> bool {
    a.index == b.index && a.name.eq_ignore_ascii_case(&b.name)
}

/// Each clip distance output of the vertex program made a plain varying (its system value dropped; its label, width and
/// stores as they were), and which they are; `None` where the program has none. The program's code is not touched.
///
/// Refused: a program of another stage, and clip distance outputs of one label (the pixel program could not tell them
/// apart).
pub fn clip_distances_as_varyings(program: &mut Program) -> Result<Option<ClipVaryings>, String> {
    if program.interface.stage != Stage::Vertex {
        return Err(format!(
            "clip distances as varyings: {} program has no clip distances to carry",
            program.interface.stage.with_article()
        ));
    }
    let mut clips = ClipVaryings::default();
    for v in &mut program.interface.outputs {
        if v.system != Some(SystemValue::ClipDistance) {
            continue;
        }
        if clips.varyings.iter().any(|(l, _)| same_label(l, &v.label)) {
            return Err(format!(
                "clip distances as varyings: two clip distance outputs are labelled {}{}",
                v.label.name, v.label.index
            ));
        }
        v.system = None;
        clips.varyings.push((v.label.clone(), v.value.width));
    }
    Ok((!clips.varyings.is_empty()).then_some(clips))
}

/// The id of a new value of `ty`.
fn push(values: &mut Vec<ValueDef>, ty: Type, op: Op) -> Result<ValueId, String> {
    let id = u32::try_from(values.len()).map_err(|_| String::from("discard clipped: too many values"))?;
    values.push(ValueDef {
        ty,
        op,
        origin: None,
        precision: Precision::Full,
    });
    Ok(ValueId(id))
}

/// The test that some clip distance is below zero, as new values (placed in `nodes`) and the bool that holds it, one
/// lane of each input at a time: a NaN is not below zero.
fn below_zero(
    values: &mut Vec<ValueDef>,
    nodes: &mut Vec<Node>,
    inputs: &[(u32, u8)],
) -> Result<Option<ValueId>, String> {
    let zero = push(values, Type::F32, Op::Literal([0; 4]))?;
    let mut any: Option<ValueId> = None;
    let mut zero_placed = false;
    for &(slot, width) in inputs {
        for lane in 0..width {
            let x = push(values, Type::F32, Op::LoadInput(slot, 1u8 << lane))?;
            let below = push(values, Type::BOOL, Op::Binary(Binary::FCmp(Cmp::Lt), x, zero))?;
            if !zero_placed {
                nodes.push(Node::Value(zero));
                zero_placed = true;
            }
            nodes.push(Node::Value(x));
            nodes.push(Node::Value(below));
            any = Some(match any {
                None => below,
                Some(earlier) => {
                    let both = push(values, Type::BOOL, Op::Binary(Binary::Or, earlier, below))?;
                    nodes.push(Node::Value(both));
                    both
                }
            });
        }
    }
    Ok(any)
}

/// Every region that returns gets the guard at its end: made afresh there, since a value is defined where its region
/// runs.
fn guard(values: &mut Vec<ValueDef>, r: &mut Region, inputs: &[(u32, u8)]) -> Result<(), String> {
    for n in &mut r.nodes {
        for inner in n.regions_mut() {
            guard(values, inner, inputs)?;
        }
    }
    if r.end == Some(End::Return) {
        let mut test = Vec::new();
        if let Some(cond) = below_zero(values, &mut test, inputs)? {
            r.nodes.extend(test);
            r.nodes.push(Node::If {
                cond,
                accept: Region {
                    nodes: vec![Node::Demote],
                    end: Some(End::Yield(Vec::new())),
                },
                reject: Region {
                    nodes: Vec::new(),
                    end: Some(End::Yield(Vec::new())),
                },
                results: Vec::new(),
            });
        }
    }
    Ok(())
}

/// The pixel program given an input for each of the producer's `clips` (after its own, perspective-interpolated at the
/// pixel centre, no Direct3D register: nothing in Direct3D feeds a clip distance to a pixel program) and, before each
/// return, a discard where any lane of them is below zero. The discard is last, after every sample the program takes:
/// it demotes the invocation, which a derivative in its quad still sees.
///
/// Refused: a program of another stage, one whose body is a control flow graph (run [`crate::prepare`] first), and one
/// that already has an input of a label a clip varying has.
pub fn discard_clipped(program: &mut Program, clips: &ClipVaryings) -> Result<(), String> {
    if program.interface.stage != Stage::Pixel {
        return Err(format!(
            "discard clipped: {} program discards nothing a clipper keeps",
            program.interface.stage.with_article()
        ));
    }
    if !matches!(program.entry.body, Body::Regions(_)) {
        return Err(String::from(
            "discard clipped: the body is a control flow graph (prepare the program first)",
        ));
    }
    for (label, _) in &clips.varyings {
        if program
            .interface
            .inputs
            .iter()
            .any(|v| same_label(&v.label, label))
        {
            return Err(format!(
                "discard clipped: the pixel program already has an input labelled {}{}",
                label.name, label.index
            ));
        }
    }
    let mut slots = Vec::new();
    for (label, width) in &clips.varyings {
        let slot = u32::try_from(program.interface.inputs.len())
            .map_err(|_| String::from("discard clipped: too many inputs"))?;
        program.interface.inputs.push(Varying {
            label: label.clone(),
            system: None,
            value: Value {
                scalar: ScalarType::Float,
                width: *width,
            },
            interp: Some(Interpolation {
                mode: InterpMode::Perspective,
                sampling: InterpSampling::Center,
            }),
            used: true,
            provenance: VaryingProvenance::default(),
        });
        slots.push((slot, *width));
    }
    let Function { values, body, .. } = &mut program.entry;
    match body {
        Body::Regions(top) => guard(values, top, &slots),
        Body::Cfg(_) => Ok(()),
    }
}
