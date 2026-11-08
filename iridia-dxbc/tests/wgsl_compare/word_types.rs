//! Floating data keeps its interpretation through bitcasts and lane transport. Integer arithmetic and numeric
//! conversions remain integer. Test-only resource aliases identify each executed store without changing executor APIs.
use std::collections::BTreeMap;

use iridia_ir::interface::{
    BufferKind, Resource, ResourceId, ResourceKind, SampleType, ScalarType, StorageTarget,
};
use iridia_ir::{
    Binary, Body, End, Function, LabelId, Node, Op, Program, Region, Scalar, Type, ValueId, VarId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum WordType {
    Float32,
    Integer,
    /// A word only a literal (or nothing) ever gave: both sides hold the same bits, whatever they mean, so it joins any
    /// origin it is merged with and compares exactly on its own.
    Constant,
    Unknown,
}

impl WordType {
    fn declared(scalar: Scalar) -> Self {
        match scalar {
            Scalar::F32 => Self::Float32,
            s if s.is_integer() || s == Scalar::Bool => Self::Integer,
            _ => Self::Unknown,
        }
    }

    pub(super) fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Constant, x) | (x, Self::Constant) => x,
            (a, b) if a == b => a,
            _ => Self::Unknown,
        }
    }
}

const UNKNOWN: [WordType; 4] = [WordType::Unknown; 4];

/// What a typed view or storage image declares its texels to be: floats for a float view or image, integers for an integer one.
/// A lane whose stored value no trace resolves (a control-flow result) is the declared type; a lane the trace resolves keeps it.
fn declared_texel(resource: &Resource) -> Option<WordType> {
    match &resource.kind {
        ResourceKind::Storage {
            target: StorageTarget::Buffer(BufferKind::Typed(view)),
            ..
        } => Some(match view.scalar {
            ScalarType::Float => WordType::Float32,
            ScalarType::Sint | ScalarType::Uint | ScalarType::Bool => WordType::Integer,
            ScalarType::Double => WordType::Unknown,
        }),
        ResourceKind::Storage {
            target: StorageTarget::Image { sample, .. },
            ..
        } => Some(match sample {
            SampleType::Float | SampleType::Depth => WordType::Float32,
            SampleType::Sint | SampleType::Uint => WordType::Integer,
        }),
        _ => None,
    }
}

/// What gives a value its content beyond its operation, over a whole function (no order of execution is used, so what it says
/// holds for any): the values stores give each variable (`StoreVar`, `StoreElement`), whether anything else can write one (a
/// payload another stage fills), and the values that reach each control-flow parameter (a construct's results by each region's
/// yield and each `Br` to it, a loop's parameters by its inits and each `Continue`). A parameter nothing here reaches (a
/// function in the `Cfg` phase) has no entry.
struct Flow {
    stored: Vec<Vec<ValueId>>,
    opaque: Vec<bool>,
    reaching: BTreeMap<ValueId, Vec<ValueId>>,
}

impl Flow {
    fn of(function: &Function) -> Self {
        let mut flow = Flow {
            stored: vec![Vec::new(); function.vars.len()],
            opaque: vec![false; function.vars.len()],
            reaching: BTreeMap::new(),
        };
        let Body::Regions(top) = &function.body else {
            // Block parameters of a control flow graph are not followed: every parameter is unknown
            if let Body::Cfg(blocks) = &function.body {
                for block in blocks {
                    flow.nodes(&block.nodes, &BTreeMap::new(), &BTreeMap::new());
                }
            }
            return flow;
        };
        // the results a label's exits give, and the parameters its `Continue` gives
        let mut results = BTreeMap::new();
        let mut params = BTreeMap::new();
        Self::labels(&top.nodes, &mut results, &mut params);
        flow.region(top, &results, &params);
        flow
    }

    fn labels(
        nodes: &[Node],
        results: &mut BTreeMap<LabelId, Vec<ValueId>>,
        params: &mut BTreeMap<LabelId, Vec<ValueId>>,
    ) {
        for node in nodes {
            match node {
                Node::Block {
                    label, results: r, ..
                }
                | Node::Switch {
                    label, results: r, ..
                } => {
                    results.insert(*label, r.clone());
                }
                Node::Loop {
                    label,
                    params: p,
                    results: r,
                    ..
                } => {
                    results.insert(*label, r.clone());
                    params.insert(*label, p.clone());
                }
                _ => {}
            }
            for region in node.regions() {
                Self::labels(&region.nodes, results, params);
            }
        }
    }

    fn reach(&mut self, to: &[ValueId], from: &[ValueId]) {
        for (to, from) in to.iter().zip(from) {
            self.reaching.entry(*to).or_default().push(*from);
        }
    }

    fn region(
        &mut self,
        region: &Region,
        results: &BTreeMap<LabelId, Vec<ValueId>>,
        params: &BTreeMap<LabelId, Vec<ValueId>>,
    ) {
        self.nodes(&region.nodes, results, params);
        match &region.end {
            Some(End::Br(label, values)) => {
                if let Some(to) = results.get(label) {
                    self.reach(to, values);
                }
            }
            Some(End::Continue(label, values)) => {
                if let Some(to) = params.get(label) {
                    self.reach(to, values);
                }
            }
            _ => {}
        }
    }

    fn nodes(
        &mut self,
        nodes: &[Node],
        results: &BTreeMap<LabelId, Vec<ValueId>>,
        params: &BTreeMap<LabelId, Vec<ValueId>>,
    ) {
        for node in nodes {
            match node {
                Node::StoreVar(var, value) | Node::StoreElement(var, _, value, _) => {
                    if let Some(list) = self.stored.get_mut(var.0 as usize) {
                        list.push(*value);
                    }
                }
                Node::DispatchMesh { payload: var, .. }
                | Node::TraceRay { payload: var, .. }
                | Node::CallShader { parameter: var, .. } => {
                    if let Some(opaque) = self.opaque.get_mut(var.0 as usize) {
                        *opaque = true;
                    }
                }
                Node::Loop {
                    params: carried,
                    inits,
                    ..
                } => self.reach(carried, inits),
                _ => {}
            }
            // the construct's own results: what each of its regions yields
            let own = match node {
                Node::If { results, .. }
                | Node::Switch { results, .. }
                | Node::Block { results, .. }
                | Node::Loop { results, .. } => Some(results),
                _ => None,
            };
            for inner in node.regions() {
                if let (Some(own), Some(End::Yield(values))) = (own, &inner.end) {
                    self.reach(own, values);
                }
                self.region(inner, results, params);
            }
        }
    }

    fn values(&self, var: VarId) -> &[ValueId] {
        self.stored.get(var.0 as usize).map_or(&[], Vec::as_slice)
    }

    fn is_opaque(&self, var: VarId) -> bool {
        self.opaque.get(var.0 as usize).copied().unwrap_or(true)
    }
}

struct Origins<'a> {
    function: &'a Function,
    flow: Flow,
    cache: Vec<Option<[WordType; 4]>>,
    solved: bool,
}

impl<'a> Origins<'a> {
    fn new(function: &'a Function) -> Self {
        Self {
            function,
            flow: Flow::of(function),
            cache: Vec::new(),
            solved: false,
        }
    }

    fn cached(&self, id: ValueId) -> [WordType; 4] {
        self.cache
            .get(id.0 as usize)
            .copied()
            .flatten()
            .unwrap_or(UNKNOWN)
    }

    fn literal_is(&self, id: ValueId, bits: u32) -> bool {
        self.function.value(id).is_some_and(|v| {
            v.ty.scalar.is_integer()
                && v.ty.scalar.bits() == 32
                && matches!(&v.op, Op::Literal(words) if words.iter().take(usize::from(v.ty.lanes)).all(|&w| w == u64::from(bits)))
        })
    }

    /// The DXBC movc idiom's mask: a literal zero/all-ones select, optionally repeated into a vector.
    /// No range inference or other bitwise operation is a mask proof.
    fn mask_select(&self, mut id: ValueId) -> bool {
        for _ in 0..self.function.values.len() {
            match self.function.value(id).map(|v| &v.op) {
                Some(Op::Splat(a)) => id = *a,
                // Compose of the same scalar is another spelling of Splat (as lifted in Riders 1027).
                Some(Op::Compose(parts)) if !parts.is_empty() && parts.iter().all(|a| *a == parts[0]) => {
                    id = parts[0];
                }
                Some(Op::Select { accept, reject, .. }) => {
                    return (self.literal_is(*accept, 0) && self.literal_is(*reject, u32::MAX))
                        || (self.literal_is(*reject, 0) && self.literal_is(*accept, u32::MAX));
                }
                _ => return false,
            }
        }
        false
    }

    fn resolve(&self, id: ValueId, ty: Type, op: &Op) -> [WordType; 4] {
        match op {
            Op::Bitcast(a) => {
                if ty.scalar.bits() == 32 && self.function.ty(*a).is_some_and(|t| t.scalar.bits() == 32) {
                    self.cached(*a)
                } else {
                    UNKNOWN
                }
            }
            Op::Splat(a) => [self.cached(*a)[0]; 4],
            Op::Swizzle(a, lanes) => {
                let source = self.cached(*a);
                std::array::from_fn(|i| {
                    source
                        .get(usize::from(lanes[i]))
                        .copied()
                        .unwrap_or(WordType::Unknown)
                })
            }
            Op::Compose(parts) => {
                let mut out = UNKNOWN;
                let mut to = out.iter_mut();
                for &part in parts {
                    let lanes = self.function.ty(part).map_or(0, |t| usize::from(t.lanes));
                    for view in self.cached(part).into_iter().take(lanes) {
                        if let Some(slot) = to.next() {
                            *slot = view;
                        }
                    }
                }
                out
            }
            Op::Select { accept, reject, .. } => {
                let a = self.cached(*accept);
                let b = self.cached(*reject);
                std::array::from_fn(|i| a[i].merge(b[i]))
            }
            Op::Binary(Binary::And, a, b) => {
                let other = if self.mask_select(*a) {
                    Some(*b)
                } else if self.mask_select(*b) {
                    Some(*a)
                } else {
                    None
                };
                other.map_or([WordType::Integer; 4], |other| {
                    self.cached(other).map(|origin| match origin {
                        // a masked float stays a float, a masked constant a constant, anything else an integer
                        WordType::Float32 | WordType::Constant => origin,
                        _ => WordType::Integer,
                    })
                })
            }
            // What a variable holds is whatever some store of it gave it (no reaching-definition order is used, so this holds for
            // any order): the stores' origins merged, lane by lane. A variable nothing stores to holds its initial constants, or
            // is undefined.
            Op::LoadVar(var) | Op::LoadElement(var, ..) => {
                if self.flow.is_opaque(*var) {
                    return UNKNOWN;
                }
                self.flow
                    .values(*var)
                    .iter()
                    .fold([WordType::Constant; 4], |held, stored| {
                        let stored = self.cached(*stored);
                        std::array::from_fn(|i| held[i].merge(stored[i]))
                    })
            }
            // A control-flow parameter is whatever reaches it, merged lane by lane; one nothing is known to reach is unknown.
            Op::Param => match self.flow.reaching.get(&id) {
                Some(sources) if !sources.is_empty() => {
                    sources.iter().fold([WordType::Constant; 4], |held, source| {
                        let source = self.cached(*source);
                        std::array::from_fn(|i| held[i].merge(source[i]))
                    })
                }
                _ => UNKNOWN,
            },
            Op::Undef => UNKNOWN,
            // A literal is the same bits on both sides, whatever they mean: it takes the origin of what it is merged with.
            Op::Literal(_) if ty.scalar != Scalar::F32 => [WordType::Constant; 4],
            // Operations which compute a new value, including Convert, establish its numeric interpretation.
            _ => [WordType::declared(ty.scalar); 4],
        }
    }

    /// Every value's origin, from the least solution of the equations `resolve` states: all values start with the origin that
    /// joins any other, and a pass lifts each to the join of what it held and what its operation and its sources now give,
    /// until a pass lifts none. Each step only raises a value in a lattice of three levels, so this ends; a cycle (a
    /// loop-carried lane that carries itself) adds nothing of its own.
    fn solve(&mut self) {
        if self.solved {
            return;
        }
        self.solved = true;
        self.cache = vec![Some([WordType::Constant; 4]); self.function.values.len()];
        loop {
            let mut lifted = false;
            for (i, def) in self.function.values.iter().enumerate() {
                let now = self.resolve(ValueId(i as u32), def.ty, &def.op);
                let held = self.cached(ValueId(i as u32));
                let joined = std::array::from_fn(|lane| held[lane].merge(now[lane]));
                if joined != held {
                    self.cache[i] = Some(joined);
                    lifted = true;
                }
            }
            if !lifted {
                return;
            }
        }
    }

    fn value(&mut self, id: ValueId) -> [WordType; 4] {
        self.solve();
        self.cached(id)
    }
}

#[derive(Clone, Debug)]
pub(super) struct StoreOrigin {
    pub buffer: ResourceId,
    pub lanes: [WordType; 4],
}

/// Only the software observer sees these extra resource IDs; its backend aliases them to the original buffers.
/// The original IR is still used for WARP and for a hard, per-vector execution equality check.
pub(super) fn instrument(program: &Program) -> Result<(Program, BTreeMap<ResourceId, StoreOrigin>), String> {
    fn nodes(
        body: &mut [Node],
        origins: &mut Origins<'_>,
        resources: &mut Vec<Resource>,
        originals: &BTreeMap<ResourceId, Resource>,
        sites: &mut BTreeMap<ResourceId, StoreOrigin>,
        next: &mut u32,
    ) -> Result<(), String> {
        for node in body {
            let store = match node {
                Node::StoreBuffer { resource, value, .. } => Some((resource, *value, false)),
                Node::StoreTexel { image, value, .. } => Some((image, *value, true)),
                _ => None,
            };
            if let Some((resource, value, texel)) = store {
                let original = originals
                    .get(&resource.id)
                    .ok_or("a store's resource is missing")?;
                let mut alias = original.clone();
                let mut lanes = origins.value(value);
                if texel && let Some(declared) = declared_texel(original) {
                    for lane in &mut lanes {
                        if *lane == WordType::Unknown {
                            *lane = declared;
                        }
                    }
                }
                let id = ResourceId(*next);
                *next = next
                    .checked_add(1)
                    .ok_or("no resource ID remains for a store observer")?;
                sites.insert(
                    id,
                    StoreOrigin {
                        buffer: resource.id,
                        lanes,
                    },
                );
                alias.id = id;
                alias.provenance.d3d = None;
                alias.provenance.counter_of = None;
                alias.provenance.name = format!("store_origin_{}", id.0);
                resources.push(alias);
                resource.id = id;
            }
            for region in node.regions_mut() {
                nodes(&mut region.nodes, origins, resources, originals, sites, next)?;
            }
        }
        Ok(())
    }
    let mut out = program.clone();
    let originals: BTreeMap<_, _> = program
        .interface
        .resources
        .iter()
        .map(|r| (r.id, r.clone()))
        .collect();
    let mut next = originals
        .keys()
        .map(|id| id.0)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("no resource ID remains for a store observer")?;
    let mut sites = BTreeMap::new();
    let mut origins = Origins::new(&program.entry);
    match &mut out.entry.body {
        Body::Regions(region) => nodes(
            &mut region.nodes,
            &mut origins,
            &mut out.interface.resources,
            &originals,
            &mut sites,
            &mut next,
        )?,
        Body::Cfg(blocks) => {
            for block in blocks {
                nodes(
                    &mut block.nodes,
                    &mut origins,
                    &mut out.interface.resources,
                    &originals,
                    &mut sites,
                    &mut next,
                )?;
            }
        }
    }
    Ok((out, sites))
}

#[cfg(test)]
mod tests {
    use super::super::{Class, compute_class};
    use super::*;
    use iridia_ir::{Address, Op, Precision, Region, Res, Type, ValueDef, ValueId};

    #[test]
    fn integer_stores_do_not_accept_adjacent_float_bit_patterns() {
        let words_from_u32_stores = (0x3f80_0000u32, 0x3f80_0001u32);
        assert_eq!(
            compute_class(
                WordType::Integer,
                words_from_u32_stores.0,
                words_from_u32_stores.1
            ),
            Class::Beyond
        );
    }

    #[test]
    fn unknown_stores_are_exact_and_declared_float_stores_keep_float_tolerance() {
        assert_eq!(
            compute_class(WordType::Unknown, 0x3f80_0000, 0x3f80_0001),
            Class::Beyond
        );
        assert_eq!(
            compute_class(WordType::Float32, 0x3f80_0000, 0x3f80_0001),
            Class::Ulps4
        );
        assert_eq!(stored_origin(&[(Type::F32, Op::Param)]), WordType::Unknown);
    }

    fn with_stores(scalars: &[Scalar]) -> Function {
        Function {
            name: "main".into(),
            vars: Vec::new(),
            queries: Vec::new(),
            values: scalars
                .iter()
                .map(|&scalar| ValueDef {
                    ty: Type::scalar(scalar),
                    op: Op::Literal([0; 4]),
                    origin: None,
                    precision: Precision::Full,
                })
                .collect(),
            body: Body::Regions(Region {
                nodes: scalars
                    .iter()
                    .enumerate()
                    .map(|(i, _)| Node::StoreBuffer {
                        resource: Res::of(ResourceId(0)),
                        address: Address::fixed(0),
                        value: ValueId(i as u32),
                    })
                    .collect(),
                end: Some(iridia_ir::End::Return),
            }),
        }
    }

    fn stored_origin(values: &[(Type, Op)]) -> WordType {
        let mut function = with_stores(&[]);
        function.values = values
            .iter()
            .map(|(ty, op)| ValueDef {
                ty: *ty,
                op: op.clone(),
                origin: None,
                precision: Precision::Full,
            })
            .collect();
        function.body = Body::Regions(Region {
            nodes: vec![Node::StoreBuffer {
                resource: Res::of(ResourceId(0)),
                address: Address::fixed(0),
                value: ValueId((values.len() - 1) as u32),
            }],
            end: Some(iridia_ir::End::Return),
        });
        let lanes = Origins::new(&function).value(ValueId((values.len() - 1) as u32));
        lanes
            .into_iter()
            .take(usize::from(values.last().unwrap().0.lanes))
            .reduce(WordType::merge)
            .unwrap_or(WordType::Unknown)
    }

    #[test]
    fn bitcasting_float_data_to_words_keeps_its_floating_origin() {
        let view = stored_origin(&[
            (Type::F32, Op::Literal([0x3f80_0000, 0, 0, 0])),
            (Type::U32, Op::Bitcast(ValueId(0))),
        ]);
        assert_eq!(view, WordType::Float32);
        assert_eq!(compute_class(view, 0x3f80_0000, 0x3f80_0001), Class::Ulps4);
    }

    #[test]
    fn numeric_conversion_of_float_data_produces_integer_words() {
        let view = stored_origin(&[
            (Type::F32, Op::Literal([0x3f80_0000, 0, 0, 0])),
            (
                Type::U32,
                Op::Convert(ValueId(0), iridia_ir::FloatToInt::Saturate),
            ),
        ]);
        assert_eq!(view, WordType::Integer);
        assert_eq!(compute_class(view, 0x3f80_0000, 0x3f80_0001), Class::Beyond);
    }

    fn masked_float() -> Vec<(Type, Op)> {
        vec![
            (Type::F32, Op::Literal([0x3f80_0000; 4])),
            (Type::U32, Op::Bitcast(ValueId(0))),
            (Type::BOOL, Op::Literal([1; 4])),
            (Type::U32, Op::Literal([u64::from(u32::MAX); 4])),
            (Type::U32, Op::Literal([0; 4])),
            (
                Type::U32,
                Op::Select {
                    cond: ValueId(2),
                    accept: ValueId(3),
                    reject: ValueId(4),
                },
            ),
            (Type::U32, Op::Binary(Binary::And, ValueId(1), ValueId(5))),
        ]
    }

    #[test]
    fn literal_mask_select_transports_float_bits_in_either_operand_order() {
        for reverse_arms in [false, true] {
            for reverse_operands in [false, true] {
                let mut values = masked_float();
                if reverse_arms {
                    values[5].1 = Op::Select {
                        cond: ValueId(2),
                        accept: ValueId(4),
                        reject: ValueId(3),
                    };
                }
                if reverse_operands {
                    values[6].1 = Op::Binary(Binary::And, ValueId(5), ValueId(1));
                }
                assert_eq!(stored_origin(&values), WordType::Float32);
            }
        }
        for mask in [Op::Splat(ValueId(5)), Op::Compose(vec![ValueId(5), ValueId(5)])] {
            let mut values = masked_float();
            values[6] = (Type::vector(Scalar::U32, 2), mask);
            values.push((Type::vector(Scalar::U32, 2), Op::Splat(ValueId(1))));
            values.push((
                Type::vector(Scalar::U32, 2),
                Op::Binary(Binary::And, ValueId(6), ValueId(7)),
            ));
            assert_eq!(stored_origin(&values), WordType::Float32);
        }
    }

    #[test]
    fn other_bit_operations_and_unproved_masks_stay_integer() {
        for op in [Binary::Or, Binary::Xor] {
            let mut values = masked_float();
            values[6].1 = Op::Binary(op, ValueId(1), ValueId(5));
            assert_eq!(stored_origin(&values), WordType::Integer);
        }
        for mask in [Op::Literal([u64::from(u32::MAX); 4]), Op::LoadInput(0, 1)] {
            let mut values = masked_float();
            values[5].1 = mask;
            assert_eq!(stored_origin(&values), WordType::Integer);
        }
        for arm in [
            Op::Literal([u64::from(u32::MAX - 1); 4]),
            Op::Binary(Binary::IAdd, ValueId(4), ValueId(4)),
        ] {
            let mut values = masked_float();
            values[3].1 = arm;
            assert_eq!(stored_origin(&values), WordType::Integer);
        }
        let mut values = masked_float();
        values[1].1 = Op::Convert(ValueId(0), iridia_ir::FloatToInt::Saturate);
        assert_eq!(stored_origin(&values), WordType::Integer);
        // A mixed composition is not a splat of the proved select, even if its other lane is all ones.
        let mut values = masked_float();
        values[6] = (
            Type::vector(Scalar::U32, 2),
            Op::Compose(vec![ValueId(5), ValueId(3)]),
        );
        values.push((Type::vector(Scalar::U32, 2), Op::Splat(ValueId(1))));
        values.push((
            Type::vector(Scalar::U32, 2),
            Op::Binary(Binary::And, ValueId(6), ValueId(7)),
        ));
        assert_eq!(stored_origin(&values), WordType::Integer);
    }

    #[test]
    fn lane_transport_preserves_each_origin_and_integer_arithmetic_stops_it() {
        let float = (Type::F32, Op::Literal([0x3f80_0000, 0, 0, 0]));
        let bits = (Type::U32, Op::Bitcast(ValueId(0)));
        let integer = (
            Type::U32,
            Op::Binary(iridia_ir::Binary::IAdd, ValueId(1), ValueId(1)),
        );
        let composed = (
            Type::vector(Scalar::U32, 2),
            Op::Compose(vec![ValueId(1), ValueId(2)]),
        );
        let values = [float.clone(), bits.clone(), integer.clone(), composed];
        let mut lane = values.to_vec();
        lane.push((Type::U32, Op::Swizzle(ValueId(3), [0; 4])));
        assert_eq!(stored_origin(&lane), WordType::Float32);
        *lane.last_mut().unwrap() = (Type::U32, Op::Swizzle(ValueId(3), [1; 4]));
        assert_eq!(stored_origin(&lane), WordType::Integer);
        assert_eq!(stored_origin(&values), WordType::Unknown);
        assert_eq!(
            stored_origin(&[
                float,
                bits,
                integer,
                (
                    Type::U32,
                    Op::Binary(iridia_ir::Binary::IAdd, ValueId(1), ValueId(2)),
                ),
            ]),
            WordType::Integer
        );
    }

    #[test]
    fn deep_bitcast_chains_are_iterative_and_a_cycle_adds_no_origin_of_its_own() {
        let mut values = vec![(Type::F32, Op::Literal([0; 4]))];
        for i in 1..4096 {
            values.push((
                if i % 2 == 0 { Type::F32 } else { Type::U32 },
                Op::Bitcast(ValueId(i - 1)),
            ));
        }
        assert_eq!(stored_origin(&values), WordType::Float32);
        assert_eq!(
            stored_origin(&[
                (Type::U32, Op::Bitcast(ValueId(1))),
                (Type::U32, Op::Bitcast(ValueId(0))),
            ]),
            WordType::Constant
        );
    }

    fn storage(target: StorageTarget) -> Resource {
        Resource {
            id: ResourceId(0),
            kind: ResourceKind::Storage {
                target,
                access: iridia_ir::interface::Access::Write,
            },
            count: 1,
            provenance: Default::default(),
        }
    }

    #[test]
    fn a_typed_view_or_image_declares_the_type_of_a_lane_no_trace_resolves() {
        let view = |scalar| {
            storage(StorageTarget::Buffer(BufferKind::Typed(
                iridia_ir::interface::Value { scalar, width: 4 },
            )))
        };
        assert_eq!(declared_texel(&view(ScalarType::Float)), Some(WordType::Float32));
        assert_eq!(declared_texel(&view(ScalarType::Uint)), Some(WordType::Integer));
        assert_eq!(declared_texel(&view(ScalarType::Sint)), Some(WordType::Integer));
        let image = |sample| {
            storage(StorageTarget::Image {
                dim: iridia_ir::interface::TextureDim::D2,
                arrayed: false,
                sample,
            })
        };
        assert_eq!(declared_texel(&image(SampleType::Float)), Some(WordType::Float32));
        assert_eq!(declared_texel(&image(SampleType::Uint)), Some(WordType::Integer));
        // a raw or structured buffer says nothing about its words
        assert_eq!(
            declared_texel(&storage(StorageTarget::Buffer(BufferKind::Raw))),
            None
        );
    }

    /// The origin of the load `values.last()` of a function whose variables are `vars` (all vec4<u32>, the second an array of
    /// two), after `stores` of values (by index) to variables (by index; an element store goes to element 0).
    fn loaded_origin(values: &[(Type, Op)], stores: &[(u32, usize)]) -> WordType {
        use iridia_ir::{OutOfBounds, Var};
        let mut function = with_stores(&[]);
        function.vars = vec![
            Var::new("r0", Type::U32X4),
            Var {
                len: Some(2),
                ..Var::new("x0", Type::U32X4)
            },
        ];
        function.values = values
            .iter()
            .map(|(ty, op)| ValueDef {
                ty: *ty,
                op: op.clone(),
                origin: None,
                precision: Precision::Full,
            })
            .collect();
        let mut nodes: Vec<Node> = stores
            .iter()
            .map(|&(var, value)| {
                let value = ValueId(value as u32);
                if var == 0 {
                    Node::StoreVar(VarId(0), value)
                } else {
                    Node::StoreElement(VarId(1), value, value, OutOfBounds::Zero)
                }
            })
            .collect();
        nodes.push(Node::StoreBuffer {
            resource: Res::of(ResourceId(0)),
            address: Address::fixed(0),
            value: ValueId((values.len() - 1) as u32),
        });
        function.body = Body::Regions(Region {
            nodes,
            end: Some(iridia_ir::End::Return),
        });
        Origins::new(&function)
            .value(ValueId((values.len() - 1) as u32))
            .into_iter()
            .reduce(WordType::merge)
            .unwrap_or(WordType::Unknown)
    }

    fn float_vector() -> (Type, Op) {
        (Type::F32X4, Op::Binary(Binary::FAdd, ValueId(0), ValueId(0)))
    }

    #[test]
    fn a_variable_holds_what_its_stores_gave_it() {
        let v4 = Type::U32X4;
        let seed = (Type::F32X4, Op::Param);
        // two float results stored to one variable: a load of it is a float
        let two_floats = [
            seed.clone(),
            float_vector(),
            (Type::F32X4, Op::Binary(Binary::FSub, ValueId(0), ValueId(0))),
            (v4, Op::LoadVar(VarId(0))),
        ];
        assert_eq!(loaded_origin(&two_floats, &[(0, 1), (0, 2)]), WordType::Float32);
        // a literal is the same bits on both sides: it joins the float
        let with_literal = [
            seed.clone(),
            float_vector(),
            (v4, Op::Literal([0, 0, 0x3f80_0000, 0])),
            (v4, Op::LoadVar(VarId(0))),
        ];
        assert_eq!(loaded_origin(&with_literal, &[(0, 1), (0, 2)]), WordType::Float32);
        // an integer result among them leaves nothing to say
        let with_integer = [
            seed.clone(),
            float_vector(),
            (v4, Op::Binary(Binary::IAdd, ValueId(2), ValueId(2))),
            (v4, Op::LoadVar(VarId(0))),
        ];
        assert_eq!(loaded_origin(&with_integer, &[(0, 1), (0, 2)]), WordType::Unknown);
        // nothing stored: the initial value, a constant; only a literal stored: a constant
        let loaded = [seed.clone(), (v4, Op::LoadVar(VarId(0)))];
        assert_eq!(loaded_origin(&loaded, &[]), WordType::Constant);
        let literal = [(v4, Op::Literal([1, 2, 3, 4])), (v4, Op::LoadVar(VarId(0)))];
        assert_eq!(loaded_origin(&literal, &[(0, 0)]), WordType::Constant);
        // an element of an array is every element's stores
        let element = [
            seed,
            float_vector(),
            (
                v4,
                Op::LoadElement(VarId(1), ValueId(0), iridia_ir::OutOfBounds::Zero),
            ),
        ];
        assert_eq!(loaded_origin(&element, &[(1, 1)]), WordType::Float32);
        // a constant is no origin of its own
        assert_eq!(WordType::Constant.merge(WordType::Integer), WordType::Integer);
        assert_eq!(WordType::Float32.merge(WordType::Integer), WordType::Unknown);
        assert_eq!(WordType::Constant.merge(WordType::Constant), WordType::Constant);
        assert_eq!(
            compute_class(WordType::Constant, 0x3f80_0000, 0x3f80_0001),
            Class::Beyond
        );
    }

    /// The origin of value `%id` of the IR text `body` (a function `main` over the inputs `input 0.x` and `input 0.y`).
    fn text_origin(body: &str, id: u32) -> WordType {
        let text = format!(
            "iridia-ir 1
fn main {{
{body}
  return
}}
"
        );
        let function = iridia_ir::text::parse(&text).unwrap_or_else(|e| {
            panic!(
                "{e}
{text}"
            )
        });
        assert_eq!(
            function.value(ValueId(id)).map(|v| v.ty.lanes),
            Some(1),
            "the text's %{id} is not the value id {id}"
        );
        Origins::new(&function)
            .value(ValueId(id))
            .into_iter()
            .next()
            .unwrap()
    }

    /// Integer data whose bits look like denormal floats (small values) stays integer data: one bit off is a difference, however
    /// the origin analysis reaches the word (arithmetic, a join with another integer, a literal, a variable, a loop), and
    /// only a word every definition of which is a float result gets a float tolerance.
    #[test]
    fn integer_words_that_look_like_denormals_still_fail_on_one_bit() {
        let prelude = "  %0: f32 = input 0.x
  %1: u32 = u32(7)
  %2: bool = ucmp.lt %1, %1
  %3: f32 = fadd %0, %0
  %4: u32 = bitcast %3
  %5: u32 = iadd %1, %1
  %6: u32 = u32(1)";
        let joined = |accept: &str, reject: &str| {
            format!(
                "{prelude}
  if %2 {{
    yield {accept}
  }} else {{
    yield {reject}
  }} -> %7: u32"
            )
        };
        // an integer result (1 is the denormal 1.4e-45 as a float): exact, and one bit off is Beyond
        for (body, id) in [
            (joined("%5", "%5"), 7),
            // a join with a literal or with float bits: the integer arm keeps the word out of the float tolerance
            (joined("%5", "%6"), 7),
            (joined("%5", "%4"), 7),
            (prelude.to_owned(), 5),
        ] {
            let origin = text_origin(&body, id);
            assert!(
                matches!(origin, WordType::Integer | WordType::Unknown),
                "{body}: {origin:?}"
            );
            assert_eq!(compute_class(origin, 1, 2), Class::Beyond);
            assert_eq!(compute_class(origin, 0, 0x8000_0000), Class::Beyond);
        }
        // a literal alone is a constant, compared exactly too
        assert_eq!(
            compute_class(text_origin(&joined("%6", "%6"), 7), 1, 2),
            Class::Beyond
        );
        // float bits are the only origin that has a tolerance: the same two words are within it
        let float = text_origin(&joined("%4", "%4"), 7);
        assert_eq!(float, WordType::Float32);
        assert_ne!(compute_class(float, 1, 2), Class::Beyond);
        assert_ne!(compute_class(float, 0, 0x8000_0000), Class::Beyond);
    }

    #[test]
    fn a_construct_result_is_what_its_arms_and_exits_give_it() {
        let prelude = "  %0: f32 = input 0.x
  %1: u32 = u32(7)
  %2: bool = ucmp.lt %1, %1
  %3: f32 = fadd %0, %0
  %4: u32 = bitcast %3
  %5: u32 = iadd %1, %1
  %6: u32 = u32(0)";
        let arms = |accept: &str, reject: &str| {
            format!(
                "{prelude}
  if %2 {{
    yield {accept}
  }} else {{
    yield {reject}
  }} -> %7: u32"
            )
        };
        // two arms of float bits: a float; a float and a literal: a float; a float and an integer result: unknown
        assert_eq!(text_origin(&arms("%4", "%4"), 7), WordType::Float32);
        assert_eq!(text_origin(&arms("%4", "%6"), 7), WordType::Float32);
        assert_eq!(text_origin(&arms("%4", "%5"), 7), WordType::Unknown);
        assert_eq!(text_origin(&arms("%6", "%6"), 7), WordType::Constant);
        // a loop's parameter is its init and every continue; its result is the body's yield and every exit
        let looped = |init: &str, next: &str| {
            format!(
                "{prelude}
  loop '0 (%7: u32) = ({init}) {{
    %8: bool = ucmp.ge %7, %1
    if %8 {{
      br '0 %7
    }} else {{
      yield
    }}
    continue '0 {next}
  }} -> %9: u32"
            )
        };
        assert_eq!(text_origin(&looped("%6", "%4"), 7), WordType::Float32);
        assert_eq!(text_origin(&looped("%4", "%4"), 7), WordType::Float32);
        assert_eq!(text_origin(&looped("%6", "%5"), 7), WordType::Integer);
        assert_eq!(text_origin(&looped("%4", "%5"), 7), WordType::Unknown);
        // the exit `br '0 %7` gives the loop's result the parameter's origin
        assert_eq!(text_origin(&looped("%6", "%4"), 9), WordType::Float32);
        assert_eq!(text_origin(&looped("%6", "%5"), 9), WordType::Integer);
    }

    #[test]
    fn executed_writes_choose_each_word_without_poisoning_untaken_stores() {
        use super::super::{Cell, Memory};
        use iridia_exec::Backend;
        let buffer = ResourceId(0);
        let mut memory = Memory {
            words: BTreeMap::from([(buffer, vec![0; 2])]),
            aliases: BTreeMap::from([
                (
                    ResourceId(1),
                    StoreOrigin {
                        buffer,
                        lanes: [
                            WordType::Float32,
                            WordType::Integer,
                            WordType::Unknown,
                            WordType::Unknown,
                        ],
                    },
                ),
                (
                    ResourceId(2),
                    StoreOrigin {
                        buffer,
                        lanes: [WordType::Integer; 4],
                    },
                ),
            ]),
            ..Memory::default()
        };
        memory
            .store_buffer(ResourceId(1), 0, [0x3f80_0000; 4], 3)
            .unwrap();
        // The executor queries an aliased store's bounds before writing. Reads must see the same buffer too.
        assert_eq!(memory.buffer_words(ResourceId(1)).unwrap(), 2);
        assert_eq!(
            memory.read_buffer(ResourceId(1), 0, 2).unwrap(),
            [0x3f80_0000, 0x3f80_0000, 0, 0]
        );
        assert_eq!(memory.origins[&(buffer, 0)], WordType::Float32);
        assert_eq!(memory.origins[&(buffer, 1)], WordType::Integer);
        // An integer site exists, but its branch has not executed. It cannot change word 0's view.
        assert_eq!(
            compute_class(memory.origins[&(buffer, 0)], 0x3f80_0000, 0x3f80_0001),
            Class::Ulps4
        );
        // An actually executed write of another origin makes only that word unknown and exact.
        memory
            .store_buffer(ResourceId(2), 0, [0x3f80_0000; 4], 1)
            .unwrap();
        assert_eq!(memory.origins[&(buffer, 0)], WordType::Unknown);
        assert_eq!(
            compute_class(memory.origins[&(buffer, 0)], 0x3f80_0000, 0x3f80_0001),
            Class::Beyond
        );
        assert_eq!(memory.stored[&Cell::Buffer(buffer, 0)].2, 2);
    }
}
