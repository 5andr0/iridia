//! Floating data keeps its interpretation through bitcasts and lane transport. Integer arithmetic and numeric
//! conversions remain integer. Test-only resource aliases identify each executed store without changing executor APIs.
use std::collections::BTreeMap;

use iridia_ir::interface::{Resource, ResourceId};
use iridia_ir::{Binary, Body, Function, Node, Op, Program, Scalar, Type, ValueId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum WordType {
    Float32,
    Integer,
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
        if self == other { self } else { Self::Unknown }
    }
}

const UNKNOWN: [WordType; 4] = [WordType::Unknown; 4];

struct Origins<'a> {
    function: &'a Function,
    cache: Vec<Option<[WordType; 4]>>,
    active: Vec<bool>,
}

impl<'a> Origins<'a> {
    fn new(function: &'a Function) -> Self {
        Self {
            function,
            cache: vec![None; function.values.len()],
            active: vec![false; function.values.len()],
        }
    }

    fn cached(&self, id: ValueId) -> [WordType; 4] {
        self.cache
            .get(id.0 as usize)
            .copied()
            .flatten()
            .unwrap_or(UNKNOWN)
    }

    fn dependencies(&self, op: &Op) -> Vec<ValueId> {
        match op {
            Op::Bitcast(a) | Op::Swizzle(a, _) | Op::Splat(a) => vec![*a],
            Op::Compose(parts) => parts.clone(),
            Op::Select { accept, reject, .. } => vec![*accept, *reject],
            Op::Binary(Binary::And, a, b) => vec![*a, *b],
            _ => Vec::new(),
        }
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

    fn resolve(&self, ty: Type, op: &Op) -> [WordType; 4] {
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
                    self.cached(other).map(|origin| {
                        if origin == WordType::Float32 {
                            origin
                        } else {
                            WordType::Integer
                        }
                    })
                })
            }
            // A control-flow parameter or mutable load needs reaching-definition provenance we do not have.
            Op::Undef | Op::Param | Op::LoadVar(_) | Op::LoadElement(..) => UNKNOWN,
            // Operations which compute a new value, including Convert, establish its numeric interpretation.
            _ => [WordType::declared(ty.scalar); 4],
        }
    }

    fn value(&mut self, id: ValueId) -> [WordType; 4] {
        let mut work = vec![(id, false)];
        while let Some((id, ready)) = work.pop() {
            if self.cache.get(id.0 as usize).is_none_or(Option::is_some) {
                continue;
            }
            let Some(def) = self.function.value(id) else {
                continue;
            };
            if ready {
                let view = self.resolve(def.ty, &def.op);
                if let Some(slot) = self.cache.get_mut(id.0 as usize) {
                    *slot = Some(view);
                }
            } else if self.active.get(id.0 as usize).copied().unwrap_or(false) {
                if let Some(slot) = self.cache.get_mut(id.0 as usize) {
                    *slot = Some(UNKNOWN);
                }
            } else {
                if let Some(active) = self.active.get_mut(id.0 as usize) {
                    *active = true;
                }
                work.push((id, true));
                work.extend(self.dependencies(&def.op).into_iter().map(|v| (v, false)));
            }
        }
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
                Node::StoreBuffer { resource, value, .. } => Some((resource, *value)),
                Node::StoreTexel { image, value, .. } => Some((image, *value)),
                _ => None,
            };
            if let Some((resource, value)) = store {
                let mut alias = originals
                    .get(&resource.id)
                    .ok_or("a store's resource is missing")?
                    .clone();
                let id = ResourceId(*next);
                *next = next
                    .checked_add(1)
                    .ok_or("no resource ID remains for a store observer")?;
                sites.insert(
                    id,
                    StoreOrigin {
                        buffer: resource.id,
                        lanes: origins.value(value),
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
        let integer = (Type::U32, Op::Literal([1, 0, 0, 0]));
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
    fn deep_bitcast_chains_are_iterative_and_cycles_stay_unknown() {
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
            WordType::Unknown
        );
    }

    #[test]
    fn executed_writes_choose_each_word_without_poisoning_untaken_stores() {
        use super::super::Memory;
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
        assert_eq!(memory.stored[&(buffer, 0)].2, 2);
    }
}
