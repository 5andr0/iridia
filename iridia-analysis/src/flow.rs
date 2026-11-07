//! What a program's outputs are made of: a dataflow query over its code, with no game in it.
//!
//! [`Values`] is the program's value graph, lane by lane: each output lane, each discard's conditions and what each
//! write is made of, as a node made of inputs, texels, constants and the operations between them. Variables are seen
//! through (a load is the value the last store left), a construct whose ways leave different values is a
//! [`Node::Merge`] that names what decides it, a loop-carried value a [`Node::Loop`] (the graph's only cycles) and a
//! bitcast of the same lanes is no operation (the bits are the value). An operation and a literal also have the scalar
//! type they are of ([`Values::kind`]), and an operation the type it reads its operands in ([`Values::reads`]): where
//! that differs from an operand's own, a bitcast is between them. Equal nodes of the same types are one node, so a
//! value used twice is shared, and two operations alike but for the type they read in (a comparison of the same bits
//! as floats and as integers) are two.
//!
//! [`Flow`] is what may reach each output lane and the discard, and how ([`Via`]): an input lane or a texel lane
//! through arithmetic (its value), only as what decides which texel is read, or only through a condition. It is a
//! may-reach over-approximation: a constant that is zero at run time still carries what it multiplies, and the answer
//! is never "must reach".
//!
//! Not followed, and stated: a write to a storage buffer or image into a later read of it (a read is the resource's
//! own value), nor an atomic's change to such memory. What each such write is made of is kept ([`Values::writes`]), so
//! a source that reaches only a write is still found. An atomic on a variable (thread-group memory) is followed: a
//! later load sees what it left. A geometry program's emitted vertices and a ray tracing program's payload are not
//! followed: such a program is an error.
use alloc::{collections::BTreeMap, collections::BTreeSet, vec::Vec};

use iridia_ir::interface::{Interface, Label, ResourceId, Stage, Varying};
pub use iridia_ir::{AtomicOp, Binary, FloatToInt, QuadSwap, Scalar, Ternary, Unary, WaveOp, WaveScan};

mod demand;
mod trivial;
mod walk;

pub use demand::{DemandError, VertexDemand};

/// A node of a [`Values`] graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

/// A literal's value: integers, floats and booleans (0 or 1) alike. Equal when the bits are.
#[derive(Clone, Copy, Debug)]
pub struct Literal(pub f64);
impl PartialEq for Literal {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for Literal {}
impl PartialOrd for Literal {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Literal {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.to_bits().cmp(&other.0.to_bits())
    }
}
impl core::hash::Hash for Literal {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// How a texel is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Read {
    /// Filtered at a coordinate.
    Sample,
    /// The four texels around a coordinate, of this lane.
    Gather(u8),
    /// One texel by its integer address.
    Load,
}

/// Where in a buffer a word is.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum At {
    /// The word's place in the buffer: its byte offset over 4 (a constant buffer's row `r`, lane `l` is `4r + l`).
    Component(u32),
    /// A word of an element the program indexes at run time.
    Indexed {
        /// The word's place for index 0: its byte offset over 4.
        word: u32,
        /// The bytes each step of the address's index moves (16 for a constant buffer's rows, 1 for a byte address);
        /// 0 where the address is fixed and only an element of an array of buffers is chosen at run time.
        stride: u32,
        /// The values of the indices, outermost first: the element of an array of buffers, then the address's.
        index: Vec<NodeId>,
    },
}

/// A wave operation ([`Op::Wave`]), as [`iridia_ir::Wave`] names it, without its operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Wave {
    /// Which active invocations' condition holds.
    Ballot,
    /// The value of the active invocation of the lowest number.
    First,
    /// The value another invocation has: the operands are the value and the invocation.
    Lane,
    /// The value another invocation of the quad has: the operands are the value and the invocation.
    QuadLane,
    /// The value the invocation's neighbour in its quad has.
    QuadSwap(QuadSwap),
    /// An operation over the active invocations' values.
    Collective(WaveOp, WaveScan),
}

/// An operation of the program, lane by lane ([`Node::Op`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    /// A one-operand operation; [`Unary::Any`] and [`Unary::All`] take every lane of their operand.
    Unary(Unary),
    /// A two-operand operation.
    Binary(Binary),
    /// A three-operand operation.
    Ternary(Ternary),
    /// A bit field put into an integer: the operands are the base, the inserted value, the offset and the count.
    BitInsert,
    /// A choice: the operands are the condition, the accepted value and the rejected one.
    Select,
    /// A dot product: the operands are the first vector's lanes, then the second's.
    Dot,
    /// A conversion to the node's scalar type ([`Values::kind`]) from the one it reads, by the rule (not a bitcast,
    /// which leaves a value as it is).
    Convert(FloatToInt),
    /// One of the operands chosen by an index computed at run time: the candidates, then the index.
    Index,
    /// A lane of a bitcast to fewer lanes: the bits of the operands, lowest first.
    Join,
    /// A lane of a bitcast to more lanes: part `.0` of the operand's bits, lowest first.
    Part(u8),
    /// An operation across the invocations of a wave.
    Wave(Wave),
    /// What an atomic operation on a variable leaves: the operands are the value before, the operand and, for a
    /// compare-exchange, the comparand.
    Atomic(AtomicOp),
}

/// A value the pipeline gives a program that is not one of its interface's input varyings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Supplied {
    /// `SV_OutputControlPointID`: the output control point a hull program's control-point phase computes.
    OutputControlPoint,
    /// `SV_PrimitiveID`: the patch a hull or domain program runs for.
    Primitive,
    /// `SV_DomainLocation`: where in the patch a domain program's vertex is.
    DomainLocation,
    /// A domain program's patch constant, by its index in the domain facts' `patch_constants` (its hull program's
    /// patch-constant result).
    PatchConstant(usize),
    /// The instance of a hull program's patch constant phase.
    PhaseInstance,
    /// Element `.0` of a mesh program's task payload, which its amplification program handed it.
    Payload(u32),
}

/// What a value lane is made of.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Node {
    /// A lane of an input varying: its index in `Interface::inputs`.
    Input {
        /// Index into `Interface::inputs`.
        input: usize,
        /// The lane of the input, 0 for x.
        lane: u8,
    },
    /// A lane of a value the pipeline supplies ([`Supplied`]): a source of its own, not an input lane.
    Supplied {
        /// Which supplied value.
        value: Supplied,
        /// The lane of the value, 0 for x.
        lane: u8,
    },
    /// A word of a buffer (a constant buffer's, a storage buffer's) or a lane of a pipeline state parameter (at its
    /// lane's place).
    Global {
        /// The interface resource.
        resource: ResourceId,
        /// Where in it the word is.
        at: At,
    },
    /// A lane of a texel a texture read returns. `coordinate` is the coordinate's (or address's) lanes in order (an
    /// array layer as its last); `controls` the rest that decides which texel: the element of an array of textures or
    /// samplers, the level, bias or gradients, the compare value, the clamp, the offset, the sample.
    Texel {
        /// The texture's interface resource.
        resource: ResourceId,
        /// Which lane (0 for x) of the returned texel.
        lane: u8,
        /// How the texel is read.
        read: Read,
        /// The coordinate's (or address's) lanes in order.
        coordinate: Vec<NodeId>,
        /// The values that decide the texel besides the coordinate.
        controls: Vec<NodeId>,
    },
    /// A lane of what the program asks of a resource that is not its contents: an image's size, layers, levels or
    /// samples, a sample's position, a buffer's length.
    Query {
        /// The resource the query is about.
        resource: ResourceId,
        /// The lane of the answer.
        lane: u8,
        /// The values the query takes: the element of an array of resources, the level, the sample.
        operands: Vec<NodeId>,
    },
    /// The hidden counter of a structured buffer, as an increment or decrement reads it.
    Counter {
        /// The buffer.
        resource: ResourceId,
    },
    /// A constant.
    Literal(Literal),
    /// An undefined value: a variable read before any store.
    Undef,
    /// An operation on other nodes.
    Op {
        /// The operation.
        op: Op,
        /// Its operands in the order the operation takes them.
        operands: Vec<NodeId>,
    },
    /// After a construct whose ways leave different values here: one of `arms`, as `condition` (the branch's
    /// condition or switch selector; for a block or loop the conditions of its exits) decides.
    Merge {
        /// The nodes the construct is decided by.
        condition: Vec<NodeId>,
        /// The value each way leaves.
        arms: Vec<NodeId>,
    },
    /// A value a loop carries: one of `arms` (its value entering the loop and after each way an iteration goes
    /// round), for as many iterations as `exit` (the conditions the loop is left by) decides.
    Loop {
        /// The loop's value entering it and after each way an iteration goes round.
        arms: Vec<NodeId>,
        /// The conditions the loop is left by.
        exit: Vec<NodeId>,
    },
}

/// A node's scalar types: its result's ([`Values::kind`]) and, for an operation, the one it reads its operands in
/// ([`Values::reads`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Kinds {
    result: Option<Scalar>,
    reads: Option<Scalar>,
}

/// A program as a value graph ([`Node`]): each output lane's node, each discard's conditions and what each write is
/// made of. A hull program's patch constant function is walked after its entry, reading the entry's outputs as its
/// output control points (each point the same lanes: which point is not followed); its results are the
/// [`Values::patch_constants`].
#[derive(Clone, Debug)]
pub struct Values {
    stage: Stage,
    nodes: Vec<Node>,
    /// Each node's scalar types ([`Values::kind`], [`Values::reads`]).
    kinds: Vec<Kinds>,
    /// (index in `Interface::outputs`, lane) -> its value.
    outputs: BTreeMap<(usize, u8), NodeId>,
    /// A hull program's patch constants: (index in the hull facts' `patch_constants`, lane) -> its value.
    patch_constants: BTreeMap<(usize, u8), NodeId>,
    /// The conditions guarding each discard, outermost first; empty for one taken unconditionally.
    discards: Vec<Vec<NodeId>>,
    /// What each write not followed into a later read is made of.
    writes: Vec<Vec<NodeId>>,
}

impl Values {
    /// The stage of the program the graph is of.
    pub fn stage(&self) -> Stage {
        self.stage
    }

    /// The number of nodes: every [`NodeId`] of the graph is below it.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no node.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Every node with its id, in id order.
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, &Node)> + '_ {
        self.nodes.iter().enumerate().map(|(i, n)| (node_id(i), n))
    }

    /// A hull program's patch constants: (index in the hull facts' `patch_constants`, lane) -> its value.
    pub fn patch_constants(&self) -> &BTreeMap<(usize, u8), NodeId> {
        &self.patch_constants
    }

    /// The node `id` names. A `NodeId` is made only by the graph it names a node of; one from another graph is the
    /// caller's bug, as an index past a slice is ([`Values::get`] is the checked form).
    ///
    /// # Panics
    ///
    /// Panics if `id` came from another `Values` graph with more nodes than this one.
    #[allow(clippy::indexing_slicing)] // a NodeId is made only by the graph it indexes
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// The node `id` names, `None` where this graph has no node of that number (an id from another graph).
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(usize::try_from(id.0).ok()?)
    }

    /// The scalar type of node `id` where the program states one: an operation's result's (a comparison's is
    /// [`Scalar::Bool`]) and a literal's. `None` for the other nodes, whose lanes are bits each use reads in its own
    /// type (an input, a texel, a buffer word, a merge or a loop's value), and for an operation the walk makes of
    /// memory (a load or store through a run-time index, an atomic's result).
    pub fn kind(&self, id: NodeId) -> Option<Scalar> {
        self.kinds.get(usize::try_from(id.0).ok()?)?.result
    }

    /// The scalar type operation `id` reads its operands in, where the program states one: an arithmetic
    /// operation's, a comparison's (whose result is [`Scalar::Bool`]), a conversion's source type, a select's accepted
    /// and rejected values'. An operand of another type has its bits read as this type: a bitcast is between them.
    /// `None` for a node that is no operation and for an operation the walk makes of memory ([`Values::kind`]).
    pub fn reads(&self, id: NodeId) -> Option<Scalar> {
        self.kinds.get(usize::try_from(id.0).ok()?)?.reads
    }

    /// The node of output `output` (its index in `Interface::outputs`) lane `lane`, where the program has it: every
    /// lane of every output, a lane the program never stores as the zero it is when the program returns.
    pub fn output(&self, output: usize, lane: u8) -> Option<NodeId> {
        self.outputs.get(&(output, lane)).copied()
    }

    /// Every output lane, by (index in `Interface::outputs`, lane).
    pub fn outputs(&self) -> &BTreeMap<(usize, u8), NodeId> {
        &self.outputs
    }

    /// The node of the output labelled `label`, lane `lane`.
    pub fn output_labelled(&self, interface: &Interface, label: &Label, lane: u8) -> Option<NodeId> {
        self.output(labelled(&interface.outputs, label)?, lane)
    }

    /// The conditions guarding each discard, outermost first (empty: taken whenever it is reached).
    pub fn discards(&self) -> &[Vec<NodeId>] {
        &self.discards
    }

    /// What each write the graph does not follow into a later read is made of, in program order: a storage buffer's
    /// or image's (its value, its address's run-time indices or coordinate), an atomic's (its operands and its
    /// address's indices), a mesh program's vertices, indices and counts and an amplification program's launch, each
    /// with the conditions it is under, outermost first.
    pub fn writes(&self) -> &[Vec<NodeId>] {
        &self.writes
    }

    /// The nodes the program's results start from: every output lane's, every patch constant's, every discard's
    /// conditions and what every write is made of ([`Values::writes`]).
    pub fn roots(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.outputs
            .values()
            .copied()
            .chain(self.patch_constants.values().copied())
            .chain(self.discards.iter().flatten().copied())
            .chain(self.writes.iter().flatten().copied())
    }

    /// Every node `roots` are made of, the roots included: the nodes reached through [`Values::operands`].
    pub fn reached_from(&self, roots: impl IntoIterator<Item = NodeId>) -> BTreeSet<NodeId> {
        let mut seen = BTreeSet::new();
        let mut pending: Vec<NodeId> = roots.into_iter().collect();
        while let Some(id) = pending.pop() {
            if seen.insert(id) {
                pending.extend(self.operands(id));
            }
        }
        seen
    }

    /// The nodes `id` is made of, as [`Node`] lists them: operands, a texel's coordinate and controls, a word's
    /// indices, a query's operands, a merge's conditions and arms, a loop's arms and exits.
    pub fn operands(&self, id: NodeId) -> Vec<NodeId> {
        match self.node(id) {
            Node::Input { .. }
            | Node::Supplied { .. }
            | Node::Counter { .. }
            | Node::Literal(_)
            | Node::Undef => Vec::new(),
            Node::Global { at, .. } => match at {
                At::Component(_) => Vec::new(),
                At::Indexed { index, .. } => index.clone(),
            },
            Node::Texel {
                coordinate, controls, ..
            } => coordinate.iter().chain(controls).copied().collect(),
            Node::Query { operands, .. } | Node::Op { operands, .. } => operands.clone(),
            Node::Merge { condition, arms } => condition.iter().chain(arms).copied().collect(),
            Node::Loop { arms, exit } => arms.iter().chain(exit).copied().collect(),
        }
    }
}

/// How a source reaches what it reaches, weakest first: only through a condition, only as what decides which texel is
/// read (or what a query asks), or through its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Via {
    /// Only through a condition: a branch or loop exit deciding the value, or a discard's.
    Control,
    /// Only as what decides which texel is read: its coordinate, level, offset or compare value.
    Coordinate,
    /// Through arithmetic, selects, swizzles, conversions and variables: the value is computed from it.
    Value,
}

/// One input lane: its index in `Interface::inputs` and the lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lane {
    /// Index into `Interface::inputs`.
    pub input: usize,
    /// The lane of the input, 0 for x.
    pub lane: u8,
}

/// A lane of a texel read from a resource of the program of `stage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TexelLane {
    /// The stage whose program reads the texture.
    pub stage: Stage,
    /// The texture's interface resource in that program.
    pub resource: ResourceId,
    /// The lane of the texel, 0 for x.
    pub lane: u8,
}

/// A word of a buffer or pipeline state parameter the program of `stage` reads at a place it states. Row `r`, lane
/// `l` of a constant buffer is word `4r + l`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalWord {
    /// The stage whose program reads the word.
    pub stage: Stage,
    /// The interface resource in that program.
    pub resource: ResourceId,
    /// The word's place in it.
    pub word: u32,
}

/// What may reach one output lane or the discard, each source with the strongest [`Via`] of its paths.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reach {
    /// The input lanes that may reach it, each with the strongest way.
    pub inputs: BTreeMap<Lane, Via>,
    /// The texel lanes that may reach it, each with the strongest way.
    pub texels: BTreeMap<TexelLane, Via>,
    /// The words of buffers and state parameters that may reach it, each with the strongest way. A buffer read at an
    /// index computed at run time is not among them: its index is, as what decides the word.
    pub words: BTreeMap<GlobalWord, Via>,
}

/// What may reach each output lane and each discard of a program ([`Reach`]). A may-reach over-approximation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Flow {
    outputs: BTreeMap<(usize, u8), Reach>,
    discard: Reach,
}

impl Flow {
    /// The flow of `program`. An error names what the walk cannot follow ([`Values::of_program`]), never a silent "no
    /// dependency".
    pub fn of_program(program: &iridia_ir::Program) -> Result<Flow, alloc::string::String> {
        Ok(Flow::of_values(&Values::of_program(program)?))
    }

    /// The flow of a value graph.
    pub fn of_values(values: &Values) -> Flow {
        let mut flow = Flow::default();
        for (&key, &node) in &values.outputs {
            flow.outputs.insert(key, reach(values, &[(node, Via::Value)]));
        }
        let conditions: Vec<(NodeId, Via)> = values
            .discards
            .iter()
            .flatten()
            .map(|&c| (c, Via::Control))
            .collect();
        flow.discard = reach(values, &conditions);
        flow
    }

    /// The input lanes that may reach output `label` lane `lane` (`SV_Target0`.w is `SV_Target`, 0, lane 3).
    pub fn reaching(&self, interface: &Interface, label: &Label, lane: u8) -> Vec<(Lane, Via)> {
        self.reach_of(interface, label, lane)
            .map(|r| r.inputs.iter().map(|(l, v)| (*l, *v)).collect())
            .unwrap_or_default()
    }

    /// The texel lanes that may reach output `label` lane `lane`.
    pub fn texels_reaching(&self, interface: &Interface, label: &Label, lane: u8) -> Vec<(TexelLane, Via)> {
        self.reach_of(interface, label, lane)
            .map(|r| r.texels.iter().map(|(t, v)| (*t, *v)).collect())
            .unwrap_or_default()
    }

    /// Everything that may reach output `label` lane `lane`, `None` where the program has no such output lane.
    pub fn reach_of(&self, interface: &Interface, label: &Label, lane: u8) -> Option<&Reach> {
        self.outputs.get(&(labelled(&interface.outputs, label)?, lane))
    }

    /// Every output lane's reach, by (index in `Interface::outputs`, lane).
    pub fn outputs(&self) -> &BTreeMap<(usize, u8), Reach> {
        &self.outputs
    }

    /// The input lanes that may reach a discard's condition: whether the pixel is drawn at all.
    pub fn discard(&self) -> &BTreeMap<Lane, Via> {
        &self.discard.inputs
    }

    /// Everything that may reach a discard's condition.
    pub fn discard_reach(&self) -> &Reach {
        &self.discard
    }
}

/// The index of the varying labelled `label` (D3D's semantic names are case-insensitive).
pub(crate) fn labelled(varyings: &[Varying], label: &Label) -> Option<usize> {
    varyings
        .iter()
        .position(|v| v.label.index == label.index && v.label.name.eq_ignore_ascii_case(&label.name))
}

fn strongest<K: Ord>(map: &mut BTreeMap<K, Via>, key: K, via: Via) {
    let entry = map.entry(key).or_insert(via);
    *entry = (*entry).max(via);
}

/// What reaches from `roots`, each with the via its path starts with: a path's via is its weakest step, a source's
/// the strongest of its paths. A texel's coordinate and controls and a query's operands are reached as
/// [`Via::Coordinate`], a merge's condition and a loop's exits as [`Via::Control`].
fn reach(values: &Values, roots: &[(NodeId, Via)]) -> Reach {
    let mut out = Reach::default();
    let mut best: BTreeMap<NodeId, Via> = BTreeMap::new();
    let mut work: Vec<(NodeId, Via)> = Vec::new();
    let push = |best: &mut BTreeMap<NodeId, Via>, work: &mut Vec<(NodeId, Via)>, id: NodeId, via: Via| {
        if best.get(&id).is_none_or(|b| *b < via) {
            best.insert(id, via);
            work.push((id, via));
        }
    };
    for &(id, via) in roots {
        push(&mut best, &mut work, id, via);
    }
    while let Some((id, via)) = work.pop() {
        if best.get(&id).is_some_and(|b| *b > via) {
            continue;
        }
        match values.node(id) {
            Node::Input { input, lane } => strongest(
                &mut out.inputs,
                Lane {
                    input: *input,
                    lane: *lane,
                },
                via,
            ),
            Node::Global { resource, at } => match at {
                At::Component(word) => strongest(
                    &mut out.words,
                    GlobalWord {
                        stage: values.stage,
                        resource: *resource,
                        word: *word,
                    },
                    via,
                ),
                At::Indexed { index, .. } => {
                    for &i in index {
                        push(&mut best, &mut work, i, via);
                    }
                }
            },
            Node::Texel {
                resource,
                lane,
                coordinate,
                controls,
                ..
            } => {
                strongest(
                    &mut out.texels,
                    TexelLane {
                        stage: values.stage,
                        resource: *resource,
                        lane: *lane,
                    },
                    via,
                );
                for &c in coordinate.iter().chain(controls) {
                    push(&mut best, &mut work, c, via.min(Via::Coordinate));
                }
            }
            Node::Query { operands, .. } => {
                for &o in operands {
                    push(&mut best, &mut work, o, via.min(Via::Coordinate));
                }
            }
            Node::Supplied { .. } | Node::Counter { .. } | Node::Literal(_) | Node::Undef => {}
            Node::Op { operands, .. } => {
                for &o in operands {
                    push(&mut best, &mut work, o, via);
                }
            }
            Node::Merge { condition, arms } => {
                for &a in arms {
                    push(&mut best, &mut work, a, via);
                }
                for &c in condition {
                    push(&mut best, &mut work, c, via.min(Via::Control));
                }
            }
            Node::Loop { arms, exit } => {
                for &a in arms {
                    push(&mut best, &mut work, a, via);
                }
                for &c in exit {
                    push(&mut best, &mut work, c, via.min(Via::Control));
                }
            }
        }
    }
    out
}

/// A graph's node number: a graph of 2^32 nodes is more memory than a machine has, so the number saturates there.
fn node_id(index: usize) -> NodeId {
    NodeId(u32::try_from(index).unwrap_or(u32::MAX))
}
