//! The vertex inputs a vertex program and the pixel program drawn after it need, lane by lane.
use alloc::{collections::BTreeMap, vec::Vec};

use iridia_ir::{
    Program,
    interface::{Label, Stage},
};

use crate::reads::{GraphError, graph_values};

use super::{Node, NodeId, Values, labelled};

/// The lanes of each program's inputs that can change the draw, each program's lanes by its interface's inputs. A
/// vertex input that only makes a varying the pixel program never reads is not needed. The vertex program's position,
/// clip and cull distances and other system outputs, and both programs' discards and writes, are always needed. The
/// walk is conservative: a condition or a texel coordinate a lane reaches counts, and constants are not specialised.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexDemand {
    /// The lanes of each vertex input that can change the draw: an index of the vertex program's
    /// [`Interface::inputs`](iridia_ir::interface::Interface::inputs) to a mask of lanes (bit 0 is x). An input with
    /// no entry is not needed.
    pub vertex: BTreeMap<usize, u8>,
    /// The lanes of each pixel input the pixel program reads, indexed the same way over the pixel program's inputs.
    pub pixel: BTreeMap<usize, u8>,
}

/// Why the input demand of a vertex and pixel pair could not be determined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DemandError {
    /// The supplied programs are not a vertex stage followed by a pixel stage.
    Stages {
        /// The first program's stage.
        vertex: Stage,
        /// The second program's stage.
        pixel: Stage,
    },
    /// A stage's value graph could not be followed.
    Graph {
        /// The stage being analysed.
        stage: Stage,
        /// The graph's refusal.
        error: GraphError,
    },
    /// The program reads a varying its interface does not declare.
    MissingInput {
        /// The pixel input index.
        input: usize,
    },
    /// No vertex output has the label of a pixel input that is read.
    UnfedInput {
        /// The pixel input label.
        label: Label,
    },
    /// The vertex output lacks a lane the pixel input reads.
    MissingLane {
        /// The pixel input label.
        label: Label,
        /// The lane, zero for x.
        lane: u8,
    },
}

impl core::fmt::Display for DemandError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Stages { .. } => f.write_str("vertex demand requires a vertex and a pixel program"),
            Self::Graph { error, .. } => core::fmt::Display::fmt(error, f),
            Self::MissingInput { input } => write!(
                f,
                "the pixel program reads input {input}, which its interface does not have"
            ),
            Self::UnfedInput { label } => write!(
                f,
                "pixel input {}{} has no vertex output",
                label.name, label.index
            ),
            Self::MissingLane { label, lane } => write!(
                f,
                "pixel input {}{} lane {lane} has no vertex write",
                label.name, label.index
            ),
        }
    }
}

impl core::error::Error for DemandError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Graph { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// The bit of `lane` in a mask of lanes; none past the eighth.
fn bit(lane: u8) -> u8 {
    1u8.checked_shl(u32::from(lane)).unwrap_or(0)
}

/// The input lanes the graph reaches from `roots`, as masks by input.
fn inputs(values: &Values, roots: impl IntoIterator<Item = NodeId>) -> BTreeMap<usize, u8> {
    let mut masks = BTreeMap::new();
    for id in values.reached_from(roots) {
        if let Some(Node::Input { input, lane }) = values.get(id) {
            *masks.entry(*input).or_default() |= bit(*lane);
        }
    }
    masks
}

impl VertexDemand {
    /// The vertex input lanes the pair needs: the pixel program's input lanes it reads, followed back through the
    /// vertex outputs of the same label and lane. A read the vertex program has no output for, or a program the walk
    /// cannot follow, is an error, never a sign that an input is not needed.
    pub fn of_programs(vertex: &Program, pixel: &Program) -> Result<Self, DemandError> {
        if vertex.stage != Stage::Vertex || pixel.stage != Stage::Pixel {
            return Err(DemandError::Stages {
                vertex: vertex.stage,
                pixel: pixel.stage,
            });
        }
        let vg = graph_values(vertex).map_err(|error| DemandError::Graph {
            stage: vertex.stage,
            error,
        })?;
        let pg = graph_values(pixel).map_err(|error| DemandError::Graph {
            stage: pixel.stage,
            error,
        })?;
        let pixel_inputs = inputs(&pg, pg.roots());
        let mut roots: Vec<NodeId> = vg
            .outputs()
            .iter()
            .filter(|((output, _), _)| {
                vertex
                    .interface
                    .outputs
                    .get(*output)
                    .is_some_and(|o| o.system.is_some())
            })
            .map(|(_, &node)| node)
            .chain(vg.discards().iter().flatten().copied())
            .chain(vg.writes().iter().flatten().copied())
            .collect();
        for (&input, &mask) in &pixel_inputs {
            let varying = pixel
                .interface
                .inputs
                .get(input)
                .ok_or(DemandError::MissingInput { input })?;
            // a system input comes from the rasterizer; the vertex program's system outputs are needed anyway
            if varying.system.is_some() {
                continue;
            }
            let output = labelled(&vertex.interface.outputs, &varying.label).ok_or_else(|| {
                DemandError::UnfedInput {
                    label: varying.label.clone(),
                }
            })?;
            for lane in (0..varying.value.width).filter(|&l| mask & bit(l) != 0) {
                roots.push(vg.output(output, lane).ok_or_else(|| DemandError::MissingLane {
                    label: varying.label.clone(),
                    lane,
                })?);
            }
        }
        Ok(Self {
            vertex: inputs(&vg, roots),
            pixel: pixel_inputs,
        })
    }
}
