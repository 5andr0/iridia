//! Necessary IR shape of the source-specific quantized-float rows. The per-word numerical proof stays in their
//! evidence; this check makes a changed lift invalidate the excuse even when its container hash is unchanged.
use iridia_ir::{Binary, Body, FloatToInt, Function, Node, Op, Type, Unary, ValueId};

fn op(f: &Function, id: ValueId, ty: Type) -> Option<&Op> {
    f.value(id).filter(|v| v.ty == ty).map(|v| &v.op)
}

fn round_trip(f: &Function, id: ValueId) -> bool {
    let Some(Op::Convert(float, FloatToInt::Saturate)) = op(f, id, Type::U32) else {
        return false;
    };
    let Some(Op::Convert(bits, FloatToInt::Undef)) = op(f, *float, Type::F32) else {
        return false;
    };
    let Some(Op::Bitcast(source)) = op(f, *bits, Type::U32) else {
        return false;
    };
    matches!(op(f, *source, Type::F32), Some(Op::Binary(Binary::FAdd, ..)))
}

fn half(f: &Function, id: ValueId) -> bool {
    matches!(op(f, id, Type::U32), Some(Op::Unary(Unary::F32ToF16, source)) if f.ty(*source) == Some(Type::F32))
}

fn upper_half(f: &Function, id: ValueId) -> bool {
    matches!(op(f, id, Type::U32), Some(Op::Binary(Binary::Shl, source, shift))
        if half(f, *source) && matches!(op(f, *shift, Type::U32), Some(Op::Literal(words)) if words[0] == 16))
}

fn packed(f: &Function, id: ValueId) -> bool {
    matches!(op(f, id, Type::U32), Some(Op::Binary(Binary::Or, a, b))
        if (upper_half(f, *a) && half(f, *b)) || (upper_half(f, *b) && half(f, *a)))
}

/// One unpacked vec3 store with the proved chain and one packed-half vec2 store, with no other global store.
pub(super) fn check(f: &Function) -> Result<(), &'static str> {
    fn nodes(f: &Function, body: &[Node], counts: &mut [usize; 2]) -> bool {
        for n in body {
            match n {
                Node::StoreBuffer { value, .. } => {
                    let Some(Op::Compose(parts)) = f.value(*value).map(|v| &v.op) else {
                        return false;
                    };
                    if parts.len() == 3 && parts.iter().all(|v| round_trip(f, *v)) {
                        counts[0] += 1;
                    } else if parts.len() == 2 && parts.iter().all(|v| packed(f, *v)) {
                        counts[1] += 1;
                    } else {
                        return false;
                    }
                }
                Node::StoreTexel { .. } => return false,
                _ => {}
            }
            if n.regions().iter().any(|r| !nodes(f, &r.nodes, counts)) {
                return false;
            }
        }
        true
    }
    let mut counts = [0, 0];
    let matches = match &f.body {
        Body::Regions(r) => nodes(f, &r.nodes, &mut counts),
        Body::Cfg(blocks) => blocks.iter().all(|b| nodes(f, &b.nodes, &mut counts)),
    };
    if matches && counts == [1, 1] {
        Ok(())
    } else {
        Err("the lifted quantized-float stores changed shape: remove the row")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iridia_ir::interface::ResourceId;
    use iridia_ir::{Address, End, Precision, Region, Res, ValueDef};

    fn example() -> Function {
        let mut f = Function {
            name: "main".into(),
            vars: Vec::new(),
            queries: Vec::new(),
            values: Vec::new(),
            body: Body::Regions(Region {
                nodes: Vec::new(),
                end: Some(End::Return),
            }),
        };
        let mut value = |ty, op| {
            let id = ValueId(f.values.len() as u32);
            f.values.push(ValueDef {
                ty,
                op,
                origin: None,
                precision: Precision::Full,
            });
            id
        };
        let zero = value(Type::F32, Op::Literal([0; 4]));
        let add = value(Type::F32, Op::Binary(Binary::FAdd, zero, zero));
        let bits = value(Type::U32, Op::Bitcast(add));
        let float = value(Type::F32, Op::Convert(bits, FloatToInt::Undef));
        let integer = value(Type::U32, Op::Convert(float, FloatToInt::Saturate));
        let unpacked = value(
            Type::vector(iridia_ir::Scalar::U32, 3),
            Op::Compose(vec![integer; 3]),
        );
        let lo = value(Type::U32, Op::Unary(Unary::F32ToF16, add));
        let shift = value(Type::U32, Op::Literal([16; 4]));
        let hi = value(Type::U32, Op::Binary(Binary::Shl, lo, shift));
        let word = value(Type::U32, Op::Binary(Binary::Or, hi, lo));
        let packed = value(
            Type::vector(iridia_ir::Scalar::U32, 2),
            Op::Compose(vec![word; 2]),
        );
        f.body = Body::Regions(Region {
            nodes: [unpacked, packed]
                .into_iter()
                .map(|value| Node::StoreBuffer {
                    resource: Res::of(ResourceId(0)),
                    address: Address::fixed(0),
                    value,
                })
                .collect(),
            end: Some(End::Return),
        });
        f
    }

    #[test]
    fn source_specific_store_shape_must_remain_the_proved_round_trip() {
        assert!(check(&example()).is_ok());
        for index in [2, 3, 4] {
            let mut changed = example();
            changed.values[index].op = Op::Literal([0; 4]);
            assert!(check(&changed).unwrap_err().contains("remove the row"));
        }
        let mut changed = example();
        changed.values[4].op = Op::Convert(ValueId(3), FloatToInt::Undef);
        assert!(check(&changed).is_err());
        let mut changed = example();
        changed.values[2].ty = Type::I32;
        assert!(check(&changed).is_err());
    }

    #[test]
    fn retaining_one_good_store_does_not_excuse_other_changed_stores() {
        let mut changed = example();
        if let Body::Regions(r) = &mut changed.body {
            r.nodes.push(r.nodes[0].clone());
        }
        assert!(check(&changed).unwrap_err().contains("remove the row"));
        let mut changed = example();
        changed.values[9].op = Op::Literal([0; 4]);
        assert!(check(&changed).is_err());
    }

    #[test]
    fn quantized_rows_require_the_exact_name_and_hash() {
        let rows: Vec<_> = crate::wgsl_compare::COMPUTE_EXPECTED
            .iter()
            .filter(|(_, _, why)| *why == crate::wgsl_compare::WARP_QUANTIZED_FLOAT_BITS)
            .collect();
        assert_eq!(
            rows.len(),
            65,
            "48 rows of capture 81015009 and 17 of capture b857e8d2"
        );
        let unique: std::collections::BTreeSet<_> = rows.iter().map(|(n, h, _)| (*n, *h)).collect();
        assert_eq!(unique.len(), 65);
        for (name, hash, _) in rows {
            assert!(hash.starts_with("sha256:") && hash.len() == 71);
            assert_eq!(
                crate::wgsl_compare::compute_expected(name, hash),
                Some(crate::wgsl_compare::WARP_QUANTIZED_FLOAT_BITS)
            );
            assert!(crate::wgsl_compare::compute_expected(name, "sha256:0").is_none());
        }
    }
}
