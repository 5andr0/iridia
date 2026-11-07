//! The walk that builds a [`Values`] graph: the program's regions run once each with the state of its variables and
//! outputs as nodes, every construct's ways joined where they meet and every loop body walked once.
use alloc::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    format,
    string::String,
    vec,
    vec::Vec,
};

use iridia_ir::{
    Address, AtomicOp, Body, Builtin, End, Function, GatherOffset, LabelId, Level, Memory, Node as Step,
    Op as IrOp, Program, Query, Region, Res, Scalar, Space, Unary, ValueId, VarId, Wave as IrWave,
    interface::{ScalarType, Stage, StageFacts, Varying},
};

use super::{At, Kinds, Literal, Node, NodeId, Op, Read, Supplied, Values, Wave, node_id};

type Outputs = BTreeMap<(usize, u8), NodeId>;

/// The graph being built: its nodes, their types and the interned ones by what they are.
#[derive(Default)]
struct Graph {
    nodes: Vec<Node>,
    kinds: Vec<Kinds>,
    interned: BTreeMap<(Node, Kinds), NodeId>,
    #[cfg(test)]
    loop_walks: usize,
}

impl Graph {
    /// The node of these types: one already made, or a new one.
    fn typed(&mut self, node: Node, kinds: Kinds) -> NodeId {
        let key = (node, kinds);
        if let Some(&id) = self.interned.get(&key) {
            return id;
        }
        let id = node_id(self.nodes.len());
        self.nodes.push(key.0.clone());
        self.kinds.push(kinds);
        self.interned.insert(key, id);
        id
    }

    /// A node of no stated type ([`Values::kind`]).
    fn intern(&mut self, node: Node) -> NodeId {
        self.typed(node, Kinds::default())
    }

    /// An operation of result type `kind` that reads its operands as `reads` ([`Values::reads`]).
    fn op(&mut self, op: Op, kind: Option<Scalar>, reads: Option<Scalar>, operands: Vec<NodeId>) -> NodeId {
        self.typed(Node::Op { op, operands }, Kinds { result: kind, reads })
    }

    fn literal(&mut self, v: f64, kind: Option<Scalar>) -> NodeId {
        self.typed(
            Node::Literal(Literal(v)),
            Kinds {
                result: kind,
                reads: None,
            },
        )
    }

    /// One of `arms` as `condition` decides; the one value where they agree.
    fn merge(&mut self, condition: Vec<NodeId>, arms: Vec<NodeId>) -> NodeId {
        let arms = distinct(arms);
        if let [only] = arms[..] {
            return only;
        }
        self.intern(Node::Merge { condition, arms })
    }

    /// A loop's value, whose arms and exits are filled once the loop is walked: a node of its own, never interned (a
    /// node equal to it is made of it).
    fn fresh(&mut self) -> NodeId {
        let id = node_id(self.nodes.len());
        self.nodes.push(Node::Loop {
            arms: Vec::new(),
            exit: Vec::new(),
        });
        self.kinds.push(Kinds::default());
        id
    }
}

/// The values in order, each once.
fn distinct(values: Vec<NodeId>) -> Vec<NodeId> {
    let mut out: Vec<NodeId> = Vec::with_capacity(values.len());
    for v in values {
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

/// The memory the walk follows, as nodes: every variable's lanes (an array's elements' lanes, element after element)
/// and the outputs stored so far (a hull program's patch constant function's: the patch constants).
#[derive(Clone, Debug, PartialEq, Eq)]
struct State {
    vars: Vec<Vec<NodeId>>,
    outputs: Outputs,
}

impl State {
    fn var(&self, var: VarId) -> Result<&Vec<NodeId>, String> {
        self.vars
            .get(var.0 as usize)
            .ok_or_else(|| format!("variable {} is not the function's", var.0))
    }

    fn var_mut(&mut self, var: VarId) -> Result<&mut Vec<NodeId>, String> {
        self.vars
            .get_mut(var.0 as usize)
            .ok_or_else(|| format!("variable {} is not the function's", var.0))
    }
}

/// A lane the walk holds that a loop may carry round.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Slot {
    /// Lane `.1` of variable `.0`.
    Var(usize, usize),
    /// An output lane.
    Output((usize, u8)),
    /// Lane `.1` of the loop's parameter `.0`.
    Param(usize, usize),
}

/// A way out of a construct to its end: the state, its results' lanes and the conditions it leaves under (those of
/// the branches between the construct and the exit).
type Exit = (State, Vec<Vec<NodeId>>, Vec<NodeId>);

/// A way to a construct's end that it leaves by: the state and its results' lanes.
type Way = (State, Vec<Vec<NodeId>>);

/// A construct an exit names by its label.
struct Target {
    label: LabelId,
    /// The length of the control stack where the construct starts.
    base: usize,
    is_loop: bool,
    /// The ways out to its end.
    exits: Vec<Exit>,
    /// A loop's ways round: the state and the parameters' lanes.
    continues: Vec<Way>,
    /// A loop's conditions of the other ways that decide how often it runs: its continues, and the exits, returns and
    /// discards that leave it for beyond it.
    left: Vec<NodeId>,
}

/// What a function's walk gives: its outputs (a patch constant function's patch constants), discards and writes.
struct Walked {
    outputs: Outputs,
    discards: Vec<Vec<NodeId>>,
    writes: Vec<Vec<NodeId>>,
}

struct Walker<'a> {
    program: &'a Program,
    f: &'a Function,
    g: &'a mut Graph,
    /// Each value's lanes, as the walk last computed it.
    lanes: Vec<Option<Vec<NodeId>>>,
    /// The conditions of the branches the walk is in, outermost first.
    control: Vec<NodeId>,
    /// A distinct identity for each active control scope. A branch and its later survival gate can name the same
    /// condition with opposite outcomes; they must not be factored as a shared guard of all returns.
    control_scopes: Vec<usize>,
    next_scope: usize,
    targets: Vec<Target>,
    /// Each return's conditions and the outputs it returns.
    returns: Vec<(Vec<NodeId>, Vec<usize>, Outputs)>,
    discards: Vec<Vec<NodeId>>,
    writes: Vec<Vec<NodeId>>,
    /// Each way out that skips what follows it (a return, a discard, an exit or continue of a construct around it):
    /// the conditions it is under, and the place on the target stack of the construct it goes to (`None` for one that
    /// leaves the function). What a construct is followed by runs only where such a way in it was not taken, up to
    /// that construct's end ([`Walker::gate`]).
    leaves: Vec<(Vec<NodeId>, Option<usize>)>,
    /// The entry's outputs, which a hull program's patch constant function reads as its output control points.
    main_outputs: Option<&'a Outputs>,
    /// Walking a hull program's patch constant function: its stores are patch constants.
    patch: bool,
}

impl Values {
    /// The value graph of `program`: its entry, and a hull program's patch constant function after it. A body that is
    /// a control flow graph is structured first ([`iridia_passes::structurize`], on a copy). An error names what the
    /// walk cannot follow: a geometry program's emitted vertices and a ray tracing program's payload are not
    /// followed.
    pub fn of_program(program: &Program) -> Result<Values, String> {
        let stage = program.stage;
        if stage == Stage::Geometry {
            return Err("a geometry program's emitted vertices are not followed".into());
        }
        if stage.is_ray_tracing() {
            return Err(format!(
                "{} program's payload is not followed",
                stage.with_article()
            ));
        }
        let mut g = Graph::default();
        let entry = structured(&program.entry)?;
        let outputs = zeros(&mut g, &program.interface.outputs);
        let main = Walker::new(program, &entry, &mut g, None, false).run(outputs)?;
        let mut values = Values {
            stage,
            nodes: Vec::new(),
            kinds: Vec::new(),
            outputs: BTreeMap::new(),
            patch_constants: BTreeMap::new(),
            discards: main.discards,
            writes: main.writes,
        };
        if let Some(patch) = &program.patch {
            let StageFacts::Hull(hull) = &program.interface.facts else {
                return Err(format!(
                    "{} program with a patch constant function",
                    stage.with_article()
                ));
            };
            let patch = structured(patch)?;
            let constants = zeros(&mut g, &hull.patch_constants);
            let walked = Walker::new(program, &patch, &mut g, Some(&main.outputs), true).run(constants)?;
            values.patch_constants = walked.outputs;
            values.discards.extend(walked.discards);
            values.writes.extend(walked.writes);
        }
        values.outputs = main.outputs;
        values.nodes = g.nodes;
        values.kinds = g.kinds;
        super::trivial::simplify(&mut values);
        Ok(values)
    }
}

/// `f` with a `Regions` body: itself, or a structured copy of a control flow graph.
fn structured(f: &Function) -> Result<Cow<'_, Function>, String> {
    match f.body {
        Body::Regions(_) => Ok(Cow::Borrowed(f)),
        Body::Cfg(_) => {
            let mut copy = f.clone();
            iridia_passes::structurize(&mut copy)
                .map_err(|e| format!("the control flow graph of {} is not structured: {e}", f.name))?;
            Ok(Cow::Owned(copy))
        }
    }
}

/// Every lane of the varyings as the zero it is before the program stores it.
fn zeros(g: &mut Graph, varyings: &[Varying]) -> Outputs {
    let mut out = BTreeMap::new();
    for (i, v) in varyings.iter().enumerate() {
        let zero = g.literal(0.0, Some(scalar_of(v.value.scalar)));
        for lane in 0..v.value.width {
            out.insert((i, lane), zero);
        }
    }
    out
}

/// The IR's scalar of an interface scalar type.
fn scalar_of(t: ScalarType) -> Scalar {
    match t {
        ScalarType::Bool => Scalar::Bool,
        ScalarType::Sint => Scalar::I32,
        ScalarType::Uint => Scalar::U32,
        ScalarType::Float => Scalar::F32,
        ScalarType::Double => Scalar::F64,
    }
}

/// A literal lane's bits as the number they are of `scalar`.
fn literal_value(bits: u64, scalar: Scalar) -> f64 {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = bits.to_le_bytes();
    match scalar {
        Scalar::Bool => f64::from(u8::from(bits != 0)),
        Scalar::U8 => f64::from(b0),
        Scalar::U16 => f64::from(u16::from_le_bytes([b0, b1])),
        Scalar::U32 => f64::from(u32::from_le_bytes([b0, b1, b2, b3])),
        // a whole number past 2^53 is the float nearest it, as the source's literal of it is
        Scalar::U64 => bits as f64,
        Scalar::I8 => f64::from(i8::from_le_bytes([b0])),
        Scalar::I16 => f64::from(i16::from_le_bytes([b0, b1])),
        Scalar::I32 => f64::from(i32::from_le_bytes([b0, b1, b2, b3])),
        Scalar::I64 => i64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]) as f64,
        Scalar::F16 => half::f16::from_le_bytes([b0, b1]).to_f64(),
        Scalar::F32 => f64::from(f32::from_le_bytes([b0, b1, b2, b3])),
        Scalar::F64 => f64::from_le_bytes([b0, b1, b2, b3, b4, b5, b6, b7]),
    }
}

/// The lanes of a mask, in ascending order.
fn mask_lanes(mask: u8) -> impl Iterator<Item = u8> {
    (0u8..8).filter(move |l| mask >> l & 1 != 0)
}

/// Lane `k` as the graph states it (a value has at most four, a variable some thousands).
fn lane_u8(k: usize) -> u8 {
    u8::try_from(k).unwrap_or(u8::MAX)
}

fn lane_u32(k: usize) -> u32 {
    u32::try_from(k).unwrap_or(u32::MAX)
}

impl<'a> Walker<'a> {
    fn new(
        program: &'a Program,
        f: &'a Function,
        g: &'a mut Graph,
        main_outputs: Option<&'a Outputs>,
        patch: bool,
    ) -> Self {
        Walker {
            program,
            f,
            g,
            lanes: vec![None; f.values.len()],
            control: Vec::new(),
            control_scopes: Vec::new(),
            next_scope: 0,
            targets: Vec::new(),
            returns: Vec::new(),
            discards: Vec::new(),
            writes: Vec::new(),
            leaves: Vec::new(),
            main_outputs,
            patch,
        }
    }

    /// The function walked from its start, with `outputs` as the outputs before it stores any.
    fn run(mut self, outputs: Outputs) -> Result<Walked, String> {
        let Body::Regions(body) = &self.f.body else {
            return Err(format!("function {} is not structured", self.f.name));
        };
        let vars = self.initial_vars();
        let state = State { vars, outputs };
        // falling off the end of the function is its return
        if let Some((end, _)) = self.region(body, state)? {
            self.returns
                .push((self.control.clone(), self.control_scopes.clone(), end.outputs));
        }
        let common: BTreeSet<usize> = self
            .returns
            .first()
            .map(|(_, scopes, _)| {
                scopes
                    .iter()
                    .copied()
                    .filter(|s| self.returns.iter().all(|(_, scopes, _)| scopes.contains(s)))
                    .collect()
            })
            .unwrap_or_default();
        // the outputs: the last return's, each earlier one's where its conditions held
        let mut result: Option<Outputs> = None;
        for (conditions, scopes, outputs) in core::mem::take(&mut self.returns).into_iter().rev() {
            let conditions: Vec<NodeId> = conditions
                .into_iter()
                .zip(scopes)
                .filter_map(|(c, scope)| (!common.contains(&scope)).then_some(c))
                .collect();
            result = Some(match result {
                Some(later) if !conditions.is_empty() => outputs
                    .iter()
                    .map(|(&key, &early)| {
                        let node = match later.get(&key) {
                            Some(&late) => self.g.merge(conditions.clone(), vec![early, late]),
                            None => early,
                        };
                        (key, node)
                    })
                    .collect(),
                _ => outputs,
            });
        }
        Ok(Walked {
            outputs: result.unwrap_or_default(),
            discards: self.discards,
            writes: self.writes,
        })
    }

    /// Every variable's lanes before the function stores it: an initialized one's literals, workgroup memory's zero, a
    /// mesh program's task payload as its amplification program handed it, and undefined for the rest.
    fn initial_vars(&mut self) -> Vec<Vec<NodeId>> {
        let stage = self.program.stage;
        let f = self.f;
        f.vars
            .iter()
            .map(|var| {
                let width = usize::from(var.ty.lanes);
                let elements = var.len.map_or(1, |n| n as usize);
                let mut lanes = Vec::with_capacity(width.saturating_mul(elements).min(1 << 16));
                let kind = Some(var.ty.scalar);
                for e in 0..elements {
                    for k in 0..width {
                        let node = match (var.init.get(e).and_then(|l| l.get(k)), var.space) {
                            (Some(&bits), _) => self.g.literal(literal_value(bits, var.ty.scalar), kind),
                            (None, Space::Workgroup) => self.g.literal(0.0, kind),
                            (None, Space::Payload) if stage == Stage::Mesh => self.g.intern(Node::Supplied {
                                value: Supplied::Payload(lane_u32(e)),
                                lane: lane_u8(k),
                            }),
                            (None, _) => self.g.intern(Node::Undef),
                        };
                        lanes.push(node);
                    }
                }
                lanes
            })
            .collect()
    }

    /// The lanes of value `v` as the walk last computed it.
    fn get(&self, v: ValueId) -> Result<Vec<NodeId>, String> {
        self.lanes
            .get(v.0 as usize)
            .cloned()
            .flatten()
            .ok_or_else(|| format!("%{} is used where the walk has not computed it", v.0))
    }

    /// The lanes of value `v`, a scalar repeated where it has one lane and `width` are wanted.
    fn broadcast(&self, v: ValueId, width: usize) -> Result<Vec<NodeId>, String> {
        let lanes = self.get(v)?;
        Ok(match lanes[..] {
            [only] if width > 1 => vec![only; width],
            _ => lanes,
        })
    }

    /// The first lane of value `v` (an index's, a scalar's).
    fn scalar_lane(&self, v: ValueId) -> Result<NodeId, String> {
        self.get(v)?
            .first()
            .copied()
            .ok_or_else(|| format!("%{} has no lanes", v.0))
    }

    fn set(&mut self, v: ValueId, lanes: Vec<NodeId>) {
        if let Some(slot) = self.lanes.get_mut(v.0 as usize) {
            *slot = Some(lanes);
        }
    }

    /// The scalar type of value `v`.
    fn scalar(&self, v: ValueId) -> Option<Scalar> {
        self.f.ty(v).map(|t| t.scalar)
    }

    /// The index value `v` is, where it is an integer literal: its word as unsigned (a negative index is past the
    /// end).
    fn literal_index(&self, v: ValueId) -> Option<u32> {
        match self.f.value(v) {
            Some(d) if d.ty.scalar.is_integer() => match d.op {
                IrOp::Literal(bits) => u32::try_from(bits.first()? & 0xffff_ffff).ok(),
                _ => None,
            },
            _ => None,
        }
    }

    /// The lanes of the element of an array of resources a resource names: none for the resource itself.
    fn element(&self, r: &Res) -> Result<Vec<NodeId>, String> {
        match r.index {
            Some(i) => self.get(i),
            None => Ok(Vec::new()),
        }
    }

    /// A write not followed into a later read: what it is made of, under the conditions it is under.
    fn write(&mut self, mut made: Vec<NodeId>) {
        made.extend(&self.control);
        self.writes.push(made);
    }

    /// Lane by lane `op` of the lanes of `operands`, each broadcast to `width`.
    fn lanewise(
        &mut self,
        op: Op,
        width: usize,
        kinds: (Option<Scalar>, Option<Scalar>),
        operands: &[ValueId],
    ) -> Result<Vec<NodeId>, String> {
        let per: Vec<Vec<NodeId>> = operands
            .iter()
            .map(|&o| self.broadcast(o, width))
            .collect::<Result<_, _>>()?;
        (0..width)
            .map(|k| {
                let lanes: Vec<NodeId> = per
                    .iter()
                    .map(|p| {
                        p.get(k)
                            .copied()
                            .ok_or("an operand with fewer lanes than its result")
                    })
                    .collect::<Result<_, _>>()?;
                Ok(self.g.op(op, kinds.0, kinds.1, lanes))
            })
            .collect()
    }

    /// The lanes `lanes` of one point of a stage whose inputs are arrays, the point `at`: the lanes themselves where
    /// the point is a literal, otherwise each chosen by it (every point is the same lanes: which point is not
    /// followed).
    fn at_point(&mut self, lanes: Vec<NodeId>, at: ValueId) -> Result<Vec<NodeId>, String> {
        if self.literal_index(at).is_some() {
            return Ok(lanes);
        }
        let index = self.scalar_lane(at)?;
        Ok(lanes
            .into_iter()
            .map(|l| self.g.op(Op::Index, None, None, vec![l, index]))
            .collect())
    }

    /// The lanes element `index` of array variable `var` holds in `state`: the element's own where the index is a
    /// literal (zero or undefined past the end, by the rule), otherwise each lane chosen among the elements' by it.
    fn load_element(
        &mut self,
        state: &State,
        var: VarId,
        index: ValueId,
        oob: iridia_ir::OutOfBounds,
    ) -> Result<Vec<NodeId>, String> {
        let def = self
            .f
            .vars
            .get(var.0 as usize)
            .ok_or_else(|| format!("variable {} is not the function's", var.0))?;
        let (width, elements) = (usize::from(def.ty.lanes), def.len.map_or(1, |n| n as usize));
        let memory = state.var(var)?.clone();
        if let Some(i) = self.literal_index(index) {
            let start = (i as usize).saturating_mul(width);
            return Ok(match memory.get(start..start.saturating_add(width)) {
                Some(lanes) if (i as usize) < elements => lanes.to_vec(),
                _ => {
                    let past = match oob {
                        iridia_ir::OutOfBounds::Zero => self.g.literal(0.0, Some(def.ty.scalar)),
                        iridia_ir::OutOfBounds::Undef => self.g.intern(Node::Undef),
                    };
                    vec![past; width]
                }
            });
        }
        let index = self.scalar_lane(index)?;
        (0..width)
            .map(|k| {
                let candidates: Vec<NodeId> = (0..elements)
                    .map(|e| {
                        memory
                            .get(e.saturating_mul(width).saturating_add(k))
                            .copied()
                            .ok_or("an element past its variable")
                    })
                    .collect::<Result<_, _>>()?;
                Ok(if let [only] = candidates[..] {
                    only
                } else {
                    let mut operands = distinct(candidates);
                    operands.push(index);
                    self.g.op(Op::Index, None, None, operands)
                })
            })
            .collect()
    }

    /// `value` stored into element `index` of array variable `var`: the element's lanes where the index is a literal
    /// (nothing past the end), otherwise each element's lanes either as they were or the value, as the index decides.
    fn store_element(
        &mut self,
        state: &mut State,
        var: VarId,
        index: ValueId,
        value: Vec<NodeId>,
    ) -> Result<(), String> {
        let def = self
            .f
            .vars
            .get(var.0 as usize)
            .ok_or_else(|| format!("variable {} is not the function's", var.0))?;
        let (width, elements) = (usize::from(def.ty.lanes), def.len.map_or(1, |n| n as usize));
        if let Some(i) = self.literal_index(index) {
            let memory = state.var_mut(var)?;
            if (i as usize) < elements {
                let start = (i as usize).saturating_mul(width);
                for (k, lane) in value.into_iter().enumerate().take(width) {
                    if let Some(slot) = memory.get_mut(start.saturating_add(k)) {
                        *slot = lane;
                    }
                }
            }
            return Ok(());
        }
        let index = self.scalar_lane(index)?;
        let mut memory = state.var(var)?.clone();
        for e in 0..elements {
            for (k, &lane) in value.iter().enumerate().take(width) {
                let at = e.saturating_mul(width).saturating_add(k);
                let old = *memory.get(at).ok_or("a store past its variable")?;
                if old == lane {
                    continue;
                }
                // a store through a run-time index may leave this element as it was
                let stored = self.g.op(Op::Index, None, None, vec![old, lane, index]);
                if let Some(slot) = memory.get_mut(at) {
                    *slot = stored;
                }
            }
        }
        *state.var_mut(var)? = memory;
        Ok(())
    }

    /// The words of a buffer a load of `width` lanes of `scalar` reads from `address` of `resource`: each word's own
    /// node at a place the program states, or at an index it computes. A 64-bit lane is its two words.
    fn words(
        &mut self,
        resource: &Res,
        address: &Address,
        width: usize,
        scalar: Scalar,
    ) -> Result<Vec<NodeId>, String> {
        let mut index = self.element(resource)?;
        let (offset, stride) = match address.index {
            None => (address.offset, 0),
            Some(i) => match self.literal_index(i) {
                Some(n) => (address.offset.wrapping_add(address.stride.wrapping_mul(n)), 0),
                None => {
                    index.push(self.scalar_lane(i)?);
                    (address.offset, address.stride)
                }
            },
        };
        let first = offset / 4;
        let per: u32 = if scalar.bits() > 32 { 2 } else { 1 };
        let word = |g: &mut Graph, w: u32| {
            let at = if index.is_empty() {
                At::Component(w)
            } else {
                At::Indexed {
                    word: w,
                    stride,
                    index: index.clone(),
                }
            };
            g.intern(Node::Global {
                resource: resource.id,
                at,
            })
        };
        let mut out = Vec::with_capacity(width);
        for k in 0..width {
            let w = first.saturating_add(lane_u32(k).saturating_mul(per));
            let lane = if per == 2 {
                let parts = vec![word(self.g, w), word(self.g, w.saturating_add(1))];
                self.g.op(Op::Join, None, None, parts)
            } else {
                word(self.g, w)
            };
            out.push(lane);
        }
        Ok(out)
    }

    /// The lanes of a texel read: one node per lane of the result.
    fn texel(
        &mut self,
        resource: &Res,
        read: Read,
        coordinate: Vec<NodeId>,
        controls: Vec<NodeId>,
        width: usize,
    ) -> Vec<NodeId> {
        (0..width)
            .map(|lane| {
                self.g.intern(Node::Texel {
                    resource: resource.id,
                    lane: lane_u8(lane),
                    read,
                    coordinate: coordinate.clone(),
                    controls: controls.clone(),
                })
            })
            .collect()
    }

    /// A texel offset fixed in the program, as literal lanes; none for no offset.
    fn fixed_offset(&mut self, offset: [i8; 3]) -> Vec<NodeId> {
        if offset == [0; 3] {
            return Vec::new();
        }
        offset
            .iter()
            .map(|&o| self.g.literal(f64::from(o), Some(Scalar::I32)))
            .collect()
    }

    /// The lanes of value `v`, its operation run on `state` (an atomic on a variable changes it).
    fn evaluate(&mut self, v: ValueId, state: &mut State) -> Result<Vec<NodeId>, String> {
        let f = self.f;
        let def = f
            .value(v)
            .ok_or_else(|| format!("%{} is not a value of the function", v.0))?;
        let ty = def.ty;
        let width = usize::from(ty.lanes);
        let kind = Some(ty.scalar);
        Ok(match &def.op {
            IrOp::Literal(bits) => bits
                .iter()
                .take(width)
                .map(|&b| self.g.literal(literal_value(b, ty.scalar), kind))
                .collect(),
            IrOp::Undef => vec![self.g.intern(Node::Undef); width],
            IrOp::Param => return Err(format!("parameter %{} is placed as a node", v.0)),
            IrOp::Unary(u, a) => {
                let reads = self.scalar(*a);
                if matches!(u, Unary::Any | Unary::All) {
                    let lanes = self.get(*a)?;
                    vec![self.g.op(Op::Unary(*u), kind, reads, lanes)]
                } else {
                    self.lanewise(Op::Unary(*u), width, (kind, reads), &[*a])?
                }
            }
            IrOp::Binary(b, x, y) => {
                self.lanewise(Op::Binary(*b), width, (kind, self.scalar(*x)), &[*x, *y])?
            }
            IrOp::Ternary(t, a, b, c) => {
                self.lanewise(Op::Ternary(*t), width, (kind, self.scalar(*a)), &[*a, *b, *c])?
            }
            IrOp::BitInsert {
                base,
                insert,
                offset,
                count,
            } => self.lanewise(
                Op::BitInsert,
                width,
                (kind, kind),
                &[*base, *insert, *offset, *count],
            )?,
            IrOp::Select { cond, accept, reject } => {
                self.lanewise(Op::Select, width, (kind, kind), &[*cond, *accept, *reject])?
            }
            IrOp::Compose(parts) => {
                let mut out = Vec::with_capacity(width);
                for p in parts {
                    out.extend(self.get(*p)?);
                }
                out
            }
            IrOp::Swizzle(a, pick) => {
                let lanes = self.get(*a)?;
                pick.iter()
                    .take(width)
                    .map(|&p| {
                        lanes
                            .get(usize::from(p))
                            .copied()
                            .ok_or("a swizzle past its vector")
                    })
                    .collect::<Result<_, _>>()?
            }
            IrOp::Splat(a) => vec![self.scalar_lane(*a)?; width],
            IrOp::Dot(a, b) => {
                let mut operands = self.get(*a)?;
                operands.extend(self.get(*b)?);
                vec![self.g.op(Op::Dot, kind, self.scalar(*a), operands)]
            }
            IrOp::Convert(a, rule) => {
                self.lanewise(Op::Convert(*rule), width, (kind, self.scalar(*a)), &[*a])?
            }
            IrOp::Bitcast(a) => self.bitcast(*a, width, kind)?,
            IrOp::LoadInput(slot, mask) => mask_lanes(*mask)
                .map(|lane| {
                    self.g.intern(Node::Input {
                        input: *slot as usize,
                        lane,
                    })
                })
                .collect(),
            IrOp::LoadInputAt(slot, mask, at) => {
                let lanes = mask_lanes(*mask)
                    .map(|lane| {
                        self.g.intern(Node::Input {
                            input: *slot as usize,
                            lane,
                        })
                    })
                    .collect();
                self.at_point(lanes, *at)?
            }
            IrOp::LoadOutputAt(slot, mask, at) => {
                let main = self
                    .main_outputs
                    .ok_or("an output control point read outside a hull program's patch constant function")?;
                let zero = self.g.literal(0.0, kind);
                let lanes = mask_lanes(*mask)
                    .map(|lane| main.get(&(*slot as usize, lane)).copied().unwrap_or(zero))
                    .collect();
                self.at_point(lanes, *at)?
            }
            IrOp::LoadPatch(slot, mask) => {
                if self.patch {
                    let zero = self.g.literal(0.0, kind);
                    mask_lanes(*mask)
                        .map(|lane| {
                            state
                                .outputs
                                .get(&(*slot as usize, lane))
                                .copied()
                                .unwrap_or(zero)
                        })
                        .collect()
                } else {
                    mask_lanes(*mask)
                        .map(|lane| {
                            self.g.intern(Node::Supplied {
                                value: Supplied::PatchConstant(*slot as usize),
                                lane,
                            })
                        })
                        .collect()
                }
            }
            IrOp::Builtin(b) => {
                let value = match b {
                    Builtin::OutputControlPoint => Supplied::OutputControlPoint,
                    Builtin::PrimitiveId => Supplied::Primitive,
                    Builtin::DomainLocation => Supplied::DomainLocation,
                    Builtin::PhaseInstance => Supplied::PhaseInstance,
                    Builtin::GsInstance => return Err("a geometry program's instance is not followed".into()),
                };
                (0..width)
                    .map(|k| {
                        self.g.intern(Node::Supplied {
                            value,
                            lane: lane_u8(k),
                        })
                    })
                    .collect()
            }
            IrOp::LoadVar(var) => state.var(*var)?.iter().take(width).copied().collect(),
            IrOp::LoadElement(var, index, oob) => self.load_element(state, *var, *index, *oob)?,
            IrOp::LoadState(r) => (0..width)
                .map(|k| {
                    self.g.intern(Node::Global {
                        resource: *r,
                        at: At::Component(lane_u32(k)),
                    })
                })
                .collect(),
            IrOp::LoadBuffer {
                resource, address, ..
            } => self.words(resource, address, width, ty.scalar)?,
            IrOp::Sample {
                texture,
                sampler,
                coord,
                level,
                compare,
                offset,
                clamp,
            } => {
                let coordinate = self.get(*coord)?;
                let mut controls = self.element(texture)?;
                controls.extend(self.element(sampler)?);
                controls.extend(self.fixed_offset(*offset));
                match level {
                    Level::Auto | Level::Zero => {}
                    Level::Bias(x) | Level::Exact(x) => controls.extend(self.get(*x)?),
                    Level::Grad(x, y) => {
                        controls.extend(self.get(*x)?);
                        controls.extend(self.get(*y)?);
                    }
                }
                for c in compare.iter().chain(clamp) {
                    controls.extend(self.get(*c)?);
                }
                self.texel(texture, Read::Sample, coordinate, controls, width)
            }
            IrOp::Gather {
                texture,
                sampler,
                coord,
                component,
                compare,
                offset,
            } => {
                let coordinate = self.get(*coord)?;
                let mut controls = self.element(texture)?;
                controls.extend(self.element(sampler)?);
                match offset {
                    GatherOffset::Fixed(o) => controls.extend(self.fixed_offset(*o)),
                    GatherOffset::Dynamic(o) => controls.extend(self.get(*o)?),
                }
                if let Some(c) = compare {
                    controls.extend(self.get(*c)?);
                }
                self.texel(texture, Read::Gather(*component), coordinate, controls, width)
            }
            IrOp::LoadTexel {
                texture,
                coord,
                level,
                sample,
                offset,
            } => {
                let coordinate = self.get(*coord)?;
                let mut controls = self.element(texture)?;
                controls.extend(self.fixed_offset(*offset));
                for c in level.iter().chain(sample) {
                    controls.extend(self.get(*c)?);
                }
                self.texel(texture, Read::Load, coordinate, controls, width)
            }
            IrOp::Query(r, q) => {
                let mut operands = self.element(r)?;
                if let Query::Size(Some(x)) | Query::SamplePosition(x) = q {
                    operands.extend(self.get(*x)?);
                }
                (0..width)
                    .map(|k| {
                        self.g.intern(Node::Query {
                            resource: r.id,
                            lane: lane_u8(k),
                            operands: operands.clone(),
                        })
                    })
                    .collect()
            }
            IrOp::Atomic {
                memory,
                op,
                value,
                compare,
            } => self.atomic(state, memory, *op, *value, *compare, width, ty.scalar)?,
            IrOp::Counter(r, _) => {
                let made = self.element(r)?;
                self.write(made);
                vec![self.g.intern(Node::Counter { resource: r.id }); width]
            }
            IrOp::ReportHit { .. } => return Err("a reported hit is not followed".into()),
            IrOp::RayQueryProceed(_) | IrOp::RayQueryState(_, _) => {
                return Err("a ray query is not followed".into());
            }
            IrOp::Wave(w) => match w {
                IrWave::Ballot(c) => {
                    let operands = self.get(*c)?;
                    vec![self.g.op(Op::Wave(Wave::Ballot), kind, self.scalar(*c), operands); width]
                }
                IrWave::First(a) => self.lanewise(Op::Wave(Wave::First), width, (kind, kind), &[*a])?,
                IrWave::QuadSwap(a, d) => {
                    self.lanewise(Op::Wave(Wave::QuadSwap(*d)), width, (kind, kind), &[*a])?
                }
                IrWave::Lane(a, l) => self.lanewise(Op::Wave(Wave::Lane), width, (kind, kind), &[*a, *l])?,
                IrWave::QuadLane(a, l) => {
                    self.lanewise(Op::Wave(Wave::QuadLane), width, (kind, kind), &[*a, *l])?
                }
                IrWave::Collective { op, scan, value } => self.lanewise(
                    Op::Wave(Wave::Collective(*op, *scan)),
                    width,
                    (kind, kind),
                    &[*value],
                )?,
            },
        })
    }

    /// The lanes of a bitcast of `a` to `width` lanes: its own where the counts agree, else each lane the bits of
    /// those it is made of ([`Op::Join`], [`Op::Part`]).
    fn bitcast(&mut self, a: ValueId, width: usize, kind: Option<Scalar>) -> Result<Vec<NodeId>, String> {
        let lanes = self.get(a)?;
        let reads = self.scalar(a);
        let n = lanes.len();
        if n == width || n == 0 || width == 0 {
            return Ok(lanes);
        }
        if n > width {
            let group = n.checked_div(width).unwrap_or(1).max(1);
            return Ok(lanes
                .chunks(group)
                .take(width)
                .map(|c| self.g.op(Op::Join, kind, reads, c.to_vec()))
                .collect());
        }
        let parts = width.checked_div(n).unwrap_or(1).max(1);
        (0..width)
            .map(|j| {
                let from = *lanes
                    .get(j.checked_div(parts).unwrap_or(0))
                    .ok_or("a bitcast past its operand")?;
                let part = lane_u8(j.checked_rem(parts).unwrap_or(0));
                Ok(self.g.op(Op::Part(part), kind, reads, vec![from]))
            })
            .collect()
    }

    /// An atomic operation: what it is made of kept as a write; the memory's value before it as its result; and
    /// for a variable (memory the walk follows), what it leaves there.
    #[allow(clippy::too_many_arguments)]
    fn atomic(
        &mut self,
        state: &mut State,
        memory: &Memory,
        op: AtomicOp,
        value: ValueId,
        compare: Option<ValueId>,
        width: usize,
        scalar: Scalar,
    ) -> Result<Vec<NodeId>, String> {
        let operand = self.get(value)?;
        let comparand = match compare {
            Some(c) => Some(self.get(c)?),
            None => None,
        };
        let mut made = Vec::new();
        match memory {
            Memory::Var(_, index) => {
                if let Some(i) = index {
                    made.extend(self.get(*i)?);
                }
            }
            Memory::Buffer(r, address) => {
                made.extend(self.element(r)?);
                if let Some(i) = address.index {
                    made.extend(self.get(i)?);
                }
            }
            Memory::Image(r, coord) => {
                made.extend(self.element(r)?);
                made.extend(self.get(*coord)?);
            }
        }
        made.extend(&operand);
        made.extend(comparand.iter().flatten());
        self.write(made);
        match memory {
            Memory::Buffer(r, address) => self.words(r, address, width, scalar),
            Memory::Image(r, coord) => {
                let coordinate = self.get(*coord)?;
                let controls = self.element(r)?;
                Ok(self.texel(r, Read::Load, coordinate, controls, width))
            }
            Memory::Var(var, index) => {
                let before = match index {
                    Some(i) => self.load_element(state, *var, *i, iridia_ir::OutOfBounds::Zero)?,
                    None => state.var(*var)?.iter().take(width).copied().collect(),
                };
                let after: Vec<NodeId> = before
                    .iter()
                    .enumerate()
                    .map(|(k, &old)| {
                        let with = operand
                            .get(k)
                            .or(operand.first())
                            .copied()
                            .ok_or("an atomic of no lanes")?;
                        Ok(match op {
                            AtomicOp::Exchange => with,
                            AtomicOp::CompareExchange => {
                                let c = comparand
                                    .as_ref()
                                    .and_then(|c| c.get(k).or(c.first()).copied())
                                    .ok_or("a compare-exchange with no comparand")?;
                                self.g.op(Op::Atomic(op), None, None, vec![old, with, c])
                            }
                            _ => self.g.op(Op::Atomic(op), None, None, vec![old, with]),
                        })
                    })
                    .collect::<Result<_, String>>()?;
                match index {
                    Some(i) => self.store_element(state, *var, *i, after)?,
                    None => {
                        let lanes = state.var_mut(*var)?;
                        for (slot, lane) in lanes.iter_mut().zip(after) {
                            *slot = lane;
                        }
                    }
                }
                Ok(before)
            }
        }
    }

    /// Lanes of an output (a patch constant in a patch constant function) stored: the lanes of `mask` from the value's
    /// lanes in order.
    fn store_output(&mut self, state: &mut State, slot: u32, mask: u8, v: ValueId) -> Result<(), String> {
        let lanes = self.get(v)?;
        for (lane, node) in mask_lanes(mask).zip(lanes) {
            state.outputs.insert((slot as usize, lane), node);
        }
        Ok(())
    }

    /// Run region `r` from `state`: the state it ends in and the lanes of the values it yields, `None` where every
    /// path through it leaves (a return, a discard, an exit to a construct around it).
    fn region(&mut self, r: &Region, state: State) -> Result<Option<Way>, String> {
        let mut state = state;
        for step in &r.nodes {
            match step {
                Step::RayQueryInit { .. }
                | Step::RayQueryAbort(_)
                | Step::RayQueryCommitTriangle(_)
                | Step::RayQueryCommitProcedural { .. } => {
                    return Err("a ray query is not followed".into());
                }
                Step::Value(v) => {
                    let lanes = self.evaluate(*v, &mut state)?;
                    self.set(*v, lanes);
                }
                Step::StoreVar(var, v) => {
                    let lanes = self.get(*v)?;
                    let memory = state.var_mut(*var)?;
                    for (slot, lane) in memory.iter_mut().zip(lanes) {
                        *slot = lane;
                    }
                }
                Step::StoreOutput(slot, mask, v) => {
                    if self.patch {
                        return Err("an output stored by a patch constant function".into());
                    }
                    self.store_output(&mut state, *slot, *mask, *v)?;
                }
                Step::StorePatch(slot, mask, v) => {
                    if !self.patch {
                        return Err("a patch constant stored outside a patch constant function".into());
                    }
                    self.store_output(&mut state, *slot, *mask, *v)?;
                }
                // a demoted invocation's later stores have no effect: its path ends here as a discard's does
                Step::Demote => {
                    self.leave_loops_from(0);
                    self.discards.push(self.control.clone());
                    self.leaves.push((self.control.clone(), None));
                    return Ok(None);
                }
                Step::StoreElement(var, index, v, _) => {
                    let lanes = self.get(*v)?;
                    self.store_element(&mut state, *var, *index, lanes)?;
                }
                Step::StoreBuffer {
                    resource,
                    address,
                    value,
                } => {
                    let mut made = self.get(*value)?;
                    made.extend(self.element(resource)?);
                    if let Some(i) = address.index {
                        made.extend(self.get(i)?);
                    }
                    self.write(made);
                }
                Step::StoreTexel { image, coord, value } => {
                    let mut made = self.get(*coord)?;
                    made.extend(self.get(*value)?);
                    made.extend(self.element(image)?);
                    self.write(made);
                }
                Step::Barrier(_) => {}
                Step::Emit(_) | Step::Cut(_) => {
                    return Err("a geometry program's emitted vertices are not followed".into());
                }
                Step::SetMeshCounts { vertices, primitives } => {
                    let mut made = self.get(*vertices)?;
                    made.extend(self.get(*primitives)?);
                    self.write(made);
                }
                Step::StoreVertex { vertex, value, .. } => {
                    let mut made = self.get(*vertex)?;
                    made.extend(self.get(*value)?);
                    self.write(made);
                }
                Step::StoreIndices { primitive, indices } => {
                    let mut made = self.get(*primitive)?;
                    made.extend(self.get(*indices)?);
                    self.write(made);
                }
                Step::DispatchMesh { groups, payload } => {
                    let mut made = self.get(*groups)?;
                    made.extend(state.var(*payload)?);
                    self.write(made);
                }
                Step::TraceRay { .. } | Step::CallShader { .. } => {
                    return Err("a ray tracing call is not followed".into());
                }
                Step::If {
                    cond,
                    accept,
                    reject,
                    results,
                } => {
                    let c = self.get(*cond)?;
                    let (base, leaves, depth) = (self.control.len(), self.leaves.len(), self.targets.len());
                    self.push_control(&c);
                    let a = self.region(accept, state.clone())?;
                    let accept_control = self.control.get(base..).unwrap_or_default().to_vec();
                    // Opposite outcomes share the predicate, but not the control scope. In particular, returns
                    // from both arms must retain the predicate that chooses between their output values.
                    self.truncate_control(base);
                    self.push_control(&c);
                    let r = self.region(reject, state)?;
                    let reject_control = self.control.get(base..).unwrap_or_default().to_vec();
                    let survival = match (a.is_some(), r.is_some()) {
                        (true, false) => Some(accept_control),
                        (false, true) => Some(reject_control),
                        _ => None,
                    };
                    self.truncate_control(base);
                    match self.join(&c, a.into_iter().chain(r).collect(), results)? {
                        Some(s) => state = s,
                        None => return Ok(None),
                    }
                    if let Some(survival) = survival {
                        // Only this arm reaches what follows. Decisions inside the other arm choose among its
                        // exits, not whether this arm survives. Keep the surviving arm's own partial exits.
                        self.push_control(&survival);
                    } else {
                        self.gate(base, leaves, depth);
                    }
                }
                Step::Switch {
                    label,
                    selector,
                    cases,
                    default,
                    results,
                } => {
                    let c = self.get(*selector)?;
                    let (base, leaves, depth) = (self.control.len(), self.leaves.len(), self.targets.len());
                    self.targets.push(Target::new(*label, base, false));
                    let mut ends = Vec::new();
                    for body in cases.iter().map(|c| &c.body).chain([default]) {
                        self.push_control(&c);
                        let end = self.region(body, state.clone())?;
                        let end_control = self.control.get(base..).unwrap_or_default().to_vec();
                        let target = self.targets.last_mut().ok_or("the switch's own target")?;
                        let exits = core::mem::take(&mut target.exits);
                        ends.extend(self.join_exits(end, &end_control, exits)?);
                        // what a case's ways out gated ends with the case
                        self.truncate_control(base);
                    }
                    self.targets.pop().ok_or("the switch's own target")?;
                    self.truncate_control(base);
                    // Inner decisions choose each case's value. Only the selector chooses between the cases.
                    match self.join(&c, ends, results)? {
                        Some(s) => state = s,
                        None => return Ok(None),
                    }
                    self.gate(base, leaves, depth);
                }
                Step::Block { label, body, results } => {
                    let (base, leaves, depth) = (self.control.len(), self.leaves.len(), self.targets.len());
                    self.targets.push(Target::new(*label, base, false));
                    let end = self.region(body, state)?;
                    let end_control = self.control.get(base..).unwrap_or_default().to_vec();
                    // what the body's ways out gated in it ends with it, as a branch's condition at its end
                    self.truncate_control(base);
                    let target = self.targets.pop().ok_or("the block's own target")?;
                    // First join the explicit exits. Their inner guards choose among those exits. The path
                    // reaching the body's end then chooses between that result and the fallthrough value.
                    let end = self.join_exits(end, &end_control, target.exits)?;
                    match self.join(&[], end.into_iter().collect(), results)? {
                        Some(s) => state = s,
                        None => return Ok(None),
                    }
                    self.gate(base, leaves, depth);
                }
                Step::Loop {
                    label,
                    params,
                    inits,
                    body,
                    results,
                } => {
                    let (base, leaves, depth) = (self.control.len(), self.leaves.len(), self.targets.len());
                    match self.run_loop(*label, params, inits, body, results, state)? {
                        Some(s) => state = s,
                        None => return Ok(None),
                    }
                    self.gate(base, leaves, depth);
                }
            }
        }
        match &r.end {
            Some(End::Yield(values)) => {
                let lanes = values.iter().map(|v| self.get(*v)).collect::<Result<_, _>>()?;
                Ok(Some((state, lanes)))
            }
            Some(End::Br(label, values)) => {
                let lanes = values.iter().map(|v| self.get(*v)).collect::<Result<_, _>>()?;
                let at = self.target(*label)?;
                self.leave_loops_from(at.saturating_add(1));
                let control = self.control.clone();
                let target = self.targets.get_mut(at).ok_or("an exit's target")?;
                let conditions = control.get(target.base..).unwrap_or_default().to_vec();
                target.exits.push((state, lanes, conditions));
                self.leaves.push((control, Some(at)));
                Ok(None)
            }
            Some(End::Continue(label, values)) => {
                let lanes = values.iter().map(|v| self.get(*v)).collect::<Result<_, _>>()?;
                let at = self.target(*label)?;
                // going round is the continue's decision, as leaving is an exit's
                self.leave_loops_from(at);
                let target = self.targets.get_mut(at).ok_or("a continue's target")?;
                if !target.is_loop {
                    return Err(format!("a continue of label {}, which is no loop's", label.0));
                }
                target.continues.push((state, lanes));
                self.leaves.push((self.control.clone(), Some(at)));
                Ok(None)
            }
            Some(End::Return) => {
                self.leave_loops_from(0);
                self.returns
                    .push((self.control.clone(), self.control_scopes.clone(), state.outputs));
                self.leaves.push((self.control.clone(), None));
                Ok(None)
            }
            Some(End::Kill) => {
                self.leave_loops_from(0);
                self.discards.push(self.control.clone());
                self.leaves.push((self.control.clone(), None));
                Ok(None)
            }
            Some(End::IgnoreHit | End::AcceptHitAndEndSearch) => {
                Err("an any-hit program's decision is not followed".into())
            }
            Some(End::Unreachable) => Ok(None),
            None => Err("a region with no end".into()),
        }
    }

    /// The place on the target stack of the construct `label` names.
    fn target(&self, label: LabelId) -> Result<usize, String> {
        self.targets
            .iter()
            .rposition(|t| t.label == label)
            .ok_or_else(|| format!("an exit to label {}, which no construct around it has", label.0))
    }

    /// A way out of the loops from place `from` on the target stack inwards (or round the one at `from`, for a continue
    /// of it): each one's iterations are decided by it, under the conditions it is under there.
    fn leave_loops_from(&mut self, from: usize) {
        let control = &self.control;
        for t in self.targets.iter_mut().skip(from) {
            if t.is_loop {
                t.left.extend(control.get(t.base..).unwrap_or_default());
            }
        }
    }

    /// After a construct that started with `base` conditions, `leaves` ways out recorded and `depth` constructs around
    /// it: what follows it runs only where none of the ways out of it to beyond it was taken (a return, a discard, an
    /// exit or continue of a construct around it), so their conditions inside it decide what follows, up to the end of
    /// the construct each goes to (there the walk drops them, as it drops a branch's condition at its end).
    fn gate(&mut self, base: usize, leaves: usize, depth: usize) {
        let mut gates = Vec::new();
        for (conditions, to) in self.leaves.get(leaves..).unwrap_or_default() {
            if to.is_none_or(|t| t < depth) {
                gates.extend(conditions.get(base..).unwrap_or_default());
            }
        }
        for c in distinct(gates) {
            if !self.control.contains(&c) {
                self.push_control(&[c]);
            }
        }
    }

    fn push_control(&mut self, conditions: &[NodeId]) {
        for &condition in conditions {
            self.control.push(condition);
            self.control_scopes.push(self.next_scope);
            self.next_scope = self.next_scope.saturating_add(1);
        }
    }

    fn truncate_control(&mut self, base: usize) {
        self.control.truncate(base);
        self.control_scopes.truncate(base);
    }

    /// Join the explicit exits inside one body before choosing between them and its fallthrough. A case that has
    /// just one exit carries that value: its survival guards do not choose between it and a different case's value.
    fn join_exits(
        &mut self,
        end: Option<Way>,
        end_control: &[NodeId],
        exits: Vec<Exit>,
    ) -> Result<Option<Way>, String> {
        let condition = distinct(exits.iter().flat_map(|(_, _, c)| c.clone()).collect());
        let exits = self.join_ways(&condition, exits.into_iter().map(|(s, r, _)| (s, r)).collect())?;
        self.join_ways(end_control, end.into_iter().chain(exits).collect())
    }

    /// The state after `ends` meet, `condition` deciding which, and the construct's `results` as their ways give them:
    /// a lane they disagree on is a merge, a lane they agree on keeps its value. `None` where no way reaches the end.
    fn join(
        &mut self,
        condition: &[NodeId],
        ends: Vec<Way>,
        results: &[ValueId],
    ) -> Result<Option<State>, String> {
        let Some((out, yielded)) = self.join_ways(condition, ends)? else {
            return Ok(None);
        };
        if yielded.len() != results.len() {
            return Err(format!(
                "a construct of {} results left with {} values",
                results.len(),
                yielded.len()
            ));
        }
        for (&r, lanes) in results.iter().zip(yielded) {
            self.set(r, lanes);
        }
        Ok(Some(out))
    }

    fn join_ways(&mut self, condition: &[NodeId], ends: Vec<Way>) -> Result<Option<Way>, String> {
        let mut ends = ends.into_iter();
        let Some((mut out, mut yielded)) = ends.next() else {
            return Ok(None);
        };
        let rest: Vec<Way> = ends.collect();
        if !rest.is_empty() {
            for (i, lanes) in out.vars.iter_mut().enumerate() {
                if rest.iter().all(|(s, _)| s.vars.get(i) == Some(lanes)) {
                    continue;
                }
                for (k, lane) in lanes.iter_mut().enumerate() {
                    let others = rest
                        .iter()
                        .filter_map(|(s, _)| s.vars.get(i).and_then(|v| v.get(k)).copied());
                    if others.clone().all(|o| o == *lane) {
                        continue;
                    }
                    let arms = core::iter::once(*lane).chain(others).collect();
                    *lane = self.g.merge(condition.to_vec(), arms);
                }
            }
            for (key, lane) in &mut out.outputs {
                let others = rest.iter().filter_map(|(s, _)| s.outputs.get(key).copied());
                if others.clone().all(|o| o == *lane) {
                    continue;
                }
                let arms = core::iter::once(*lane).chain(others).collect();
                *lane = self.g.merge(condition.to_vec(), arms);
            }
            for (j, lanes) in yielded.iter_mut().enumerate() {
                for (k, lane) in lanes.iter_mut().enumerate() {
                    let others: Vec<NodeId> = rest
                        .iter()
                        .map(|(_, y)| {
                            y.get(j)
                                .and_then(|l| l.get(k))
                                .copied()
                                .ok_or("ways out of a construct that give its results different lanes")
                        })
                        .collect::<Result<_, _>>()?;
                    if others.iter().all(|o| o == lane) {
                        continue;
                    }
                    let arms = core::iter::once(*lane).chain(others).collect();
                    *lane = self.g.merge(condition.to_vec(), arms);
                }
            }
        }
        Ok(Some((out, yielded)))
    }

    /// Install a header phi for every memory, output and parameter lane before walking the body. Once the
    /// backedges have filled their arms, the graph removes phis whose value does not change. Discovering changed
    /// lanes by retrying the body instead costs exponentially many walks for nested loops.
    #[allow(clippy::too_many_arguments)]
    fn run_loop(
        &mut self,
        label: LabelId,
        params: &[ValueId],
        inits: &[ValueId],
        body: &Region,
        results: &[ValueId],
        entry: State,
    ) -> Result<Option<State>, String> {
        let base = self.control.len();
        let initial: Vec<Vec<NodeId>> = inits.iter().map(|v| self.get(*v)).collect::<Result<_, _>>()?;
        if initial.len() != params.len() {
            return Err(format!(
                "a loop of {} parameters entered with {} values",
                params.len(),
                initial.len()
            ));
        }
        let mut start = entry.clone();
        let mut start_params = initial.clone();
        let mut phis = Vec::new();
        for (i, lanes) in start.vars.iter_mut().enumerate() {
            for (k, lane) in lanes.iter_mut().enumerate() {
                let id = self.g.fresh();
                phis.push((Slot::Var(i, k), id, *lane));
                *lane = id;
            }
        }
        for (key, lane) in &mut start.outputs {
            let id = self.g.fresh();
            phis.push((Slot::Output(*key), id, *lane));
            *lane = id;
        }
        for (i, lanes) in start_params.iter_mut().enumerate() {
            for (k, lane) in lanes.iter_mut().enumerate() {
                let id = self.g.fresh();
                phis.push((Slot::Param(i, k), id, *lane));
                *lane = id;
            }
        }
        for (&p, lanes) in params.iter().zip(&start_params) {
            self.set(p, lanes.clone());
        }
        self.targets.push(Target::new(label, base, true));
        #[cfg(test)]
        {
            self.g.loop_walks += 1;
        }
        let end = self.region(body, start)?;
        self.truncate_control(base);
        let mut target = self.targets.pop().ok_or("the loop's own target")?;
        if let Some((s, r)) = end {
            target.exits.push((s, r, Vec::new()));
        }
        let exit = distinct(
            target
                .exits
                .iter()
                .flat_map(|(_, _, c)| c.iter().copied())
                .chain(target.left.iter().copied())
                .collect(),
        );
        for (slot, id, before) in phis {
            let of = |s: &State, p: &[Vec<NodeId>]| -> Option<NodeId> {
                match slot {
                    Slot::Var(i, k) => s.vars.get(i)?.get(k).copied(),
                    Slot::Output(key) => s.outputs.get(&key).copied(),
                    Slot::Param(i, k) => p.get(i)?.get(k).copied(),
                }
            };
            let mut arms = vec![before];
            arms.extend(target.continues.iter().filter_map(|(s, p)| of(s, p)));
            if let Some(node) = self.g.nodes.get_mut(id.0 as usize) {
                *node = Node::Loop {
                    arms: distinct(arms),
                    exit: exit.clone(),
                };
            }
        }
        let ends = target.exits.into_iter().map(|(s, r, _)| (s, r)).collect();
        self.join(&exit, ends, results)
    }
}

impl Target {
    fn new(label: LabelId, base: usize, is_loop: bool) -> Target {
        Target {
            label,
            base,
            is_loop,
            exits: Vec::new(),
            continues: Vec::new(),
            left: Vec::new(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::flow::{Flow, Lane, Via};
    use core::fmt::Write as _;
    use iridia_ir::interface::{Interface, Label, ShaderFormat, Value, Varying};
    use iridia_ir::{Conventions, FloatEnv, Provenance};
    use proptest::prelude::*;

    fn nested(depth: usize) -> Program {
        let mut text = String::from(
            "iridia-ir 1\nfn main {\nvar v: f32\n%0: f32 = input 0.x\n%1: f32 = input 0.y\n%2: f32 = f32(0.0)\n%3: bool = fcmp.lt %0, %2\nv = %2\n",
        );
        for label in 0..depth {
            writeln!(text, "loop '{label} () = () {{").unwrap();
        }
        for label in (0..depth).rev() {
            let value = 4 + 2 * label;
            writeln!(text, "%{value}: f32 = load v\n%{}: f32 = fadd %{value}, %0\nv = %{}\nif %3 {{\ncontinue '{label}\n}} else {{\nbr '{label}\n}}\nunreachable\n}}", value + 1, value + 1).unwrap();
        }
        let value = 4 + 2 * depth;
        writeln!(
            text,
            "%{value}: f32 = load v\noutput 0.x = %{value}\noutput 0.y = %1\nreturn\n}}"
        )
        .unwrap();
        let varying = |name: &str| Varying {
            label: Label {
                name: name.into(),
                index: 0,
            },
            system: None,
            value: Value {
                scalar: ScalarType::Float,
                width: 2,
            },
            interp: None,
            used: true,
            provenance: Default::default(),
        };
        let program = Program {
            stage: Stage::Pixel,
            entry: iridia_ir::text::parse(&text).unwrap(),
            patch: None,
            workgroup: None,
            interface: Interface {
                stage: Stage::Pixel,
                inputs: vec![varying("TEXCOORD")],
                outputs: vec![varying("SV_Target")],
                resources: Vec::new(),
                facts: Default::default(),
            },
            conventions: Conventions::D3D,
            float: FloatEnv::default(),
            provenance: Provenance {
                format: ShaderFormat::Dxbc,
                target: "ps_5_0".into(),
                lifter: "text".into(),
                lifter_version: 1,
                source_hash: None,
                instruction_count: None,
                policies: Vec::new(),
            },
            notes: Vec::new(),
        };
        iridia_ir::validate(&program).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
        program
    }

    proptest! {
        #[test]
        fn nested_loops_are_walked_once(depth in 1usize..65) {
            let program = nested(depth);
            let mut graph = Graph::default();
            let outputs = zeros(&mut graph, &program.interface.outputs);
            Walker::new(&program, &program.entry, &mut graph, None, false).run(outputs).unwrap();
            prop_assert_eq!(graph.loop_walks, depth);
            let values = Values::of_program(&program).unwrap();
            let flow = Flow::of_values(&values);
            prop_assert_eq!(flow.outputs()[&(0, 0)].inputs.get(&Lane { input: 0, lane: 0 }), Some(&Via::Value));
            prop_assert_eq!(flow.outputs()[&(0, 1)].inputs.len(), 1);
            prop_assert_eq!(flow.outputs()[&(0, 1)].inputs.get(&Lane { input: 0, lane: 1 }), Some(&Via::Value));
        }
    }

    #[test]
    fn invariant_computations_keep_their_identity_across_a_loop() {
        let mut program = nested(1);
        program.interface.inputs[0].value.width = 4;
        program.entry = iridia_ir::text::parse(
            "iridia-ir 1
fn main {
  var stable: f32
  var result: f32
  %0: f32 = input 0.x
  %1: f32 = input 0.y
  %2: f32 = input 0.z
  %3: f32 = f32(0.0)
  %4: bool = fcmp.lt %2, %3
  %5: f32 = fadd %0, %1
  stable = %0
  result = %5
  loop '0 () = () {
    %6: f32 = load stable
    %7: f32 = fadd %6, %1
    result = %7
    if %4 {
      continue '0
    } else {
      yield
    }
    yield
  }
  %8: f32 = load result
  output 0.x = %8
  return
}",
        )
        .unwrap();
        iridia_ir::validate(&program).unwrap();
        let values = Values::of_program(&program).unwrap();
        let flow = Flow::of_values(&values);
        let reach = &flow.outputs()[&(0, 0)];
        assert_eq!(reach.inputs.len(), 2);
        assert_eq!(reach.inputs.get(&Lane { input: 0, lane: 0 }), Some(&Via::Value));
        assert_eq!(reach.inputs.get(&Lane { input: 0, lane: 1 }), Some(&Via::Value));
    }

    #[test]
    fn a_nested_exit_does_not_choose_an_outer_blocks_result() {
        let mut program = nested(1);
        program.entry = iridia_ir::text::parse(
            "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f32 = input 0.y
  %2: f32 = f32(0.0)
  %3: f32 = f32(1.0)
  %4: bool = fcmp.lt %0, %2
  %5: bool = fcmp.lt %1, %2
  %7: u32 = u32(0)
  block '0 {
    if %4 {
      yield
    } else {
      loop '1 (%8: u32) = (%7) {
        %9: u32 = u32(1)
        %10: u32 = iadd %8, %9
        %11: u32 = u32(2)
        %12: bool = ucmp.lt %10, %11
        %13: bool = and %5, %12
        if %13 {
          continue '1 %10
        } else {
          yield
        }
        br '0 %3
      }
      unreachable
    }
    yield %2
  } -> %6: f32
  output 0.x = %6
  return
}",
        )
        .unwrap();
        iridia_ir::validate(&program).unwrap();
        let flow = Flow::of_program(&program).unwrap();
        let reach = &flow.outputs()[&(0, 0)];
        assert_eq!(reach.inputs.len(), 1, "{reach:?}");
        assert_eq!(reach.inputs.get(&Lane { input: 0, lane: 0 }), Some(&Via::Control));
    }

    #[test]
    fn explicit_return_alternatives_keep_their_selector() {
        for guarded in [false, true] {
            for switch in [false, true] {
                let guard = if guarded {
                    "  if %4 {\n    kill\n  } else {\n    yield\n  }\n"
                } else {
                    ""
                };
                let alternatives = if switch {
                    "  %6: u32 = bitcast %1
  switch '0 %6 {
  case 0:
    output 0.x = %3
    return
  case 1:
    output 0.x = %2
    return
  default:
    output 0.x = %3
    return
  }"
                } else {
                    "  if %5 {
    output 0.x = %3
    return
  } else {
    output 0.x = %2
    return
  }"
                };
                let mut program = nested(1);
                program.entry = iridia_ir::text::parse(&format!(
                    "iridia-ir 1
fn main {{
  %0: f32 = input 0.x
  %1: f32 = input 0.y
  %2: f32 = f32(0.0)
  %3: f32 = f32(1.0)
  %4: bool = fcmp.lt %0, %2
{guard}  %5: bool = fcmp.lt %1, %2
{alternatives}
  unreachable
}}"
                ))
                .unwrap();
                iridia_ir::validate(&program).unwrap();
                let flow = Flow::of_program(&program).unwrap();
                let reach = &flow.outputs()[&(0, 0)];
                assert_eq!(reach.inputs.len(), 1, "guarded={guarded}, switch={switch}");
                assert_eq!(reach.inputs.get(&Lane { input: 0, lane: 1 }), Some(&Via::Control));
                assert_eq!(flow.discard_reach().inputs.len(), usize::from(guarded));
            }
        }
    }

    #[test]
    fn a_cases_survival_guard_does_not_choose_between_cases() {
        let mut program = nested(1);
        program.entry = iridia_ir::text::parse(
            "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f32 = input 0.y
  %2: f32 = f32(0.0)
  %3: f32 = f32(1.0)
  %4: bool = fcmp.lt %1, %2
  %5: u32 = bitcast %0
  switch '0 %5 {
  case 0:
    if %4 {
      kill
    } else {
      yield
    }
    output 0.x = %2
    br '0
  default:
    output 0.x = %3
    br '0
  }
  return
}",
        )
        .unwrap();
        iridia_ir::validate(&program).unwrap();
        let flow = Flow::of_program(&program).unwrap();
        assert_eq!(flow.outputs()[&(0, 0)].inputs.len(), 1);
        assert_eq!(
            flow.outputs()[&(0, 0)].inputs.get(&Lane { input: 0, lane: 0 }),
            Some(&Via::Control)
        );
        assert_eq!(flow.discard_reach().inputs.len(), 2);
    }

    #[test]
    fn shared_survival_guards_do_not_choose_between_opposite_returns() {
        let mut program = nested(1);
        program.entry = iridia_ir::text::parse(
            "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f32 = input 0.y
  %2: f32 = f32(0.0)
  %3: f32 = f32(1.0)
  %4: bool = fcmp.lt %0, %2
  if %4 {
    kill
  } else {
    yield
  }
  %5: bool = fcmp.lt %1, %2
  if %5 {
    output 0.x = %3
    return
  } else {
    yield
  }
  output 0.x = %2
  return
}",
        )
        .unwrap();
        iridia_ir::validate(&program).unwrap();
        let flow = Flow::of_program(&program).unwrap();
        let reach = &flow.outputs()[&(0, 0)];
        assert_eq!(reach.inputs.len(), 1);
        assert_eq!(reach.inputs.get(&Lane { input: 0, lane: 1 }), Some(&Via::Control));
        assert!(
            flow.discard_reach()
                .inputs
                .contains_key(&Lane { input: 0, lane: 0 })
        );
    }
}
