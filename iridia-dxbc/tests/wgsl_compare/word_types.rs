//! A buffer word gets float tolerance only when every store's declared IR type is f32.
use std::collections::BTreeMap;

use iridia_ir::interface::ResourceId;
use iridia_ir::{Body, Function, Node, Program, Scalar};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WordType {
    Float32,
    Integer,
    Unknown,
}

impl WordType {
    fn declared(scalar: Option<Scalar>) -> Self {
        match scalar {
            Some(Scalar::F32) => Self::Float32,
            Some(s) if s.is_integer() || s == Scalar::Bool => Self::Integer,
            _ => Self::Unknown,
        }
    }
}

fn nodes(function: &Function, body: &[Node], out: &mut BTreeMap<ResourceId, WordType>) {
    for node in body {
        let stored = match node {
            Node::StoreBuffer { resource, value, .. } => Some((resource.id, *value)),
            Node::StoreTexel { image, value, .. } => Some((image.id, *value)),
            _ => None,
        };
        if let Some((resource, value)) = stored {
            let ty = WordType::declared(function.ty(value).map(|t| t.scalar));
            out.entry(resource)
                .and_modify(|old| {
                    if *old != ty {
                        *old = WordType::Unknown;
                    }
                })
                .or_insert(ty);
        }
        for region in node.regions() {
            nodes(function, &region.nodes, out);
        }
    }
}

fn function_stores(function: &Function, out: &mut BTreeMap<ResourceId, WordType>) {
    match &function.body {
        Body::Regions(region) => nodes(function, &region.nodes, out),
        Body::Cfg(blocks) => {
            for block in blocks {
                nodes(function, &block.nodes, out);
            }
        }
    }
}

pub(super) fn stores(program: &Program) -> BTreeMap<ResourceId, WordType> {
    let mut out = BTreeMap::new();
    for function in std::iter::once(&program.entry).chain(program.patch.as_ref()) {
        function_stores(function, &mut out);
    }
    out
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

    #[test]
    fn declared_store_types_are_used_and_mixed_stores_remain_unknown() {
        for (scalars, expected) in [
            (vec![Scalar::U32], WordType::Integer),
            (vec![Scalar::F32, Scalar::F32], WordType::Float32),
            (vec![Scalar::F32, Scalar::U32, Scalar::F32], WordType::Unknown),
        ] {
            let mut views = BTreeMap::new();
            function_stores(&with_stores(&scalars), &mut views);
            assert_eq!(views.get(&ResourceId(0)), Some(&expected));
        }
    }
}
