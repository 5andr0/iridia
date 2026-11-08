//! Temporary bridge for the old `iridia-exec` WGSL oracle. Remove with that oracle in cut-over step 3/5.
//! Only the writer's named pure helpers are inlined; every other call is refused. Tests validate the actual
//! emitted text before this bridge, and WARP tests execute that text with its calls intact.
use super::oracle_naga as naga_fork;
use naga_fork::{Block, Expression as E, Function, Handle, LocalVariable, Span, Statement as S};
use std::collections::BTreeMap;

type Map = BTreeMap<Handle<E>, Handle<E>>;
type Calls = BTreeMap<Handle<E>, (Handle<Function>, Vec<Handle<E>>)>;

fn allowed(name: &str) -> bool {
    // naga appends an underscore to names ending in a digit when writing WGSL.
    let name = name.strip_suffix('_').unwrap_or(name);
    [
        "iridia_f32_to_f16_",
        "iridia_f16_to_f32_",
        "iridia_fmod_",
        "iridia_umulhi_",
        "iridia_smulhi_",
        "iridia_extract_",
        "iridia_insert_",
    ]
    .iter()
    .any(|p| {
        name.strip_prefix(p)
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    })
}

fn expr(e: &E, map: &Map, locals: &BTreeMap<Handle<LocalVariable>, Handle<LocalVariable>>) -> E {
    let mut e = e.clone();
    let sub = |h: &mut Handle<E>| *h = map[h];
    match &mut e {
        E::Literal(_)
        | E::Constant(_)
        | E::Override(_)
        | E::ZeroValue(_)
        | E::FunctionArgument(_)
        | E::GlobalVariable(_)
        | E::AtomicResult { .. }
        | E::WorkGroupUniformLoadResult { .. }
        | E::SubgroupBallotResult
        | E::SubgroupOperationResult { .. } => {}
        E::LocalVariable(h) => *h = locals[h],
        E::Compose { components, .. } => components.iter_mut().for_each(sub),
        E::Access { base, index } => {
            sub(base);
            sub(index);
        }
        E::AccessIndex { base, .. } => sub(base),
        E::Splat { value, .. } => sub(value),
        E::Swizzle { vector, .. } => sub(vector),
        E::Load { pointer } => sub(pointer),
        E::Unary { expr, .. } | E::Derivative { expr, .. } | E::As { expr, .. } | E::ArrayLength(expr) => {
            sub(expr)
        }
        E::Binary { left, right, .. } => {
            sub(left);
            sub(right);
        }
        E::Select {
            condition,
            accept,
            reject,
        } => {
            sub(condition);
            sub(accept);
            sub(reject);
        }
        E::Relational { argument, .. } => sub(argument),
        E::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            sub(arg);
            arg1.iter_mut().chain(arg2).chain(arg3).for_each(sub);
        }
        E::ImageQuery { image, query } => {
            sub(image);
            if let naga_fork::ImageQuery::Size { level } = query {
                level.iter_mut().for_each(sub);
            }
        }
        E::ImageSample {
            image,
            sampler,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref,
            ..
        } => {
            sub(image);
            sub(sampler);
            sub(coordinate);
            array_index
                .iter_mut()
                .chain(offset)
                .chain(depth_ref)
                .for_each(sub);
            match level {
                naga_fork::SampleLevel::Auto | naga_fork::SampleLevel::Zero => {}
                naga_fork::SampleLevel::Exact(h) | naga_fork::SampleLevel::Bias(h) => sub(h),
                naga_fork::SampleLevel::Gradient { x, y } => {
                    sub(x);
                    sub(y);
                }
            }
        }
        E::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            sub(image);
            sub(coordinate);
            array_index.iter_mut().chain(sample).chain(level).for_each(sub);
        }
        other => panic!("the pure-helper test bridge does not handle {other:?}"),
    }
    e
}

fn calls(block: &Block, out: &mut Calls) {
    for s in block.iter() {
        match s {
            S::Call {
                function,
                arguments,
                result: Some(r),
            } => {
                out.insert(*r, (*function, arguments.clone()));
            }
            S::Block(b) => calls(b, out),
            S::If { accept, reject, .. } => {
                calls(accept, out);
                calls(reject, out);
            }
            S::Loop { body, continuing, .. } => {
                calls(body, out);
                calls(continuing, out);
            }
            S::Switch { cases, .. } => {
                for case in cases {
                    calls(&case.body, out);
                }
            }
            _ => {}
        }
    }
}

fn block(source: &Block, map: &Map, expanded: &BTreeMap<Handle<E>, Block>, helper: bool) -> Block {
    let mut out = Block::new();
    for (s, span) in source.span_iter() {
        let s = match s {
            S::Emit(range) => {
                for h in range.clone() {
                    out.push(
                        S::Emit(naga_fork::Range::new_from_bounds(map[&h], map[&h])),
                        *span,
                    );
                }
                continue;
            }
            S::Call { result: Some(h), .. } => {
                out.extend_block(expanded[h].clone());
                continue;
            }
            S::Return { .. } if helper => continue,
            S::Return { value } => S::Return {
                value: value.map(|h| map[&h]),
            },
            S::Store { pointer, value } => S::Store {
                pointer: map[pointer],
                value: map[value],
            },
            S::If {
                condition,
                accept,
                reject,
            } => S::If {
                condition: map[condition],
                accept: block(accept, map, expanded, helper),
                reject: block(reject, map, expanded, helper),
            },
            S::Block(b) => S::Block(block(b, map, expanded, helper)),
            S::Switch { selector, cases } => S::Switch {
                selector: map[selector],
                cases: cases
                    .iter()
                    .map(|case| naga_fork::SwitchCase {
                        value: case.value,
                        body: block(&case.body, map, expanded, helper),
                        fall_through: case.fall_through,
                    })
                    .collect(),
            },
            S::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => S::ImageStore {
                image: map[image],
                coordinate: map[coordinate],
                array_index: array_index.map(|h| map[&h]),
                value: map[value],
            },
            S::Atomic {
                pointer,
                fun,
                value,
                result,
            } => S::Atomic {
                pointer: map[pointer],
                fun: atomic(*fun, map),
                value: map[value],
                result: result.map(|h| map[&h]),
            },
            S::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
                result,
            } => S::ImageAtomic {
                image: map[image],
                coordinate: map[coordinate],
                array_index: array_index.map(|h| map[&h]),
                fun: atomic(*fun, map),
                value: map[value],
                result: result.map(|h| map[&h]),
            },
            S::WorkGroupUniformLoad { pointer, result } => S::WorkGroupUniformLoad {
                pointer: map[pointer],
                result: map[result],
            },
            S::SubgroupBallot { result, predicate } => S::SubgroupBallot {
                result: map[result],
                predicate: predicate.map(|h| map[&h]),
            },
            S::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => S::SubgroupCollectiveOperation {
                op: *op,
                collective_op: *collective_op,
                argument: map[argument],
                result: map[result],
            },
            S::SubgroupGather {
                mode,
                argument,
                result,
            } => S::SubgroupGather {
                mode: gather(*mode, map),
                argument: map[argument],
                result: map[result],
            },
            S::Loop {
                body,
                continuing,
                break_if,
            } => S::Loop {
                body: block(body, map, expanded, helper),
                continuing: block(continuing, map, expanded, helper),
                break_if: break_if.map(|h| map[&h]),
            },
            S::Break | S::Continue | S::Kill | S::ControlBarrier(_) | S::MemoryBarrier(_) => s.clone(),
            other => panic!("the pure-helper test bridge does not handle {other:?}"),
        };
        out.push(s, *span);
    }
    out
}

fn atomic(fun: naga_fork::AtomicFunction, map: &Map) -> naga_fork::AtomicFunction {
    match fun {
        naga_fork::AtomicFunction::Exchange { compare } => naga_fork::AtomicFunction::Exchange {
            compare: compare.map(|h| map[&h]),
        },
        other => other,
    }
}

fn gather(mode: naga_fork::GatherMode, map: &Map) -> naga_fork::GatherMode {
    use naga_fork::GatherMode as G;
    match mode {
        G::Broadcast(h) => G::Broadcast(map[&h]),
        G::Shuffle(h) => G::Shuffle(map[&h]),
        G::ShuffleDown(h) => G::ShuffleDown(map[&h]),
        G::ShuffleUp(h) => G::ShuffleUp(map[&h]),
        G::ShuffleXor(h) => G::ShuffleXor(map[&h]),
        G::QuadBroadcast(h) => G::QuadBroadcast(map[&h]),
        G::BroadcastFirst | G::QuadSwap(_) => mode,
    }
}

fn append(
    source: &Function,
    arguments: Option<&[Handle<E>]>,
    functions: &naga_fork::Arena<Function>,
    to: &mut Function,
) -> (Map, Block) {
    let mut locals = BTreeMap::new();
    for (h, l) in source.local_variables.iter() {
        let mut l = l.clone();
        l.init = None;
        locals.insert(h, to.local_variables.append(l, Span::UNDEFINED));
    }
    let mut sites = BTreeMap::new();
    calls(&source.body, &mut sites);
    let mut map = Map::new();
    let mut expanded = BTreeMap::new();
    for (h, e) in source.expressions.iter() {
        let new = match (e, arguments) {
            (E::FunctionArgument(n), Some(args)) => args[*n as usize],
            (E::CallResult(function), _) => {
                let callee = &functions[*function];
                assert!(
                    allowed(callee.name.as_deref().unwrap_or("")),
                    "refused non-helper call: {:?}",
                    callee.name
                );
                let (target, args) = &sites[&h];
                assert_eq!(target, function);
                let args: Vec<_> = args.iter().map(|a| map[a]).collect();
                let (inner, body) = append(callee, Some(&args), functions, to);
                let Some(S::Return { value: Some(result) }) = callee.body.last() else {
                    panic!("helper has no final result")
                };
                expanded.insert(h, body);
                inner[result]
            }
            _ => to.expressions.append(expr(e, &map, &locals), Span::UNDEFINED),
        };
        map.insert(h, new);
    }
    for (h, l) in source.local_variables.iter() {
        to.local_variables[locals[&h]].init = l.init.map(|h| map[&h]);
    }
    let body = block(&source.body, &map, &expanded, arguments.is_some());
    (map, body)
}

pub fn helpers(module: &mut naga_fork::Module) {
    for entry in &mut module.entry_points {
        // Do not rewrite unrelated programs (including the old oracle's unsupported calls).
        let mut sites = Calls::new();
        calls(&entry.function.body, &mut sites);
        if sites.is_empty() {
            continue;
        }
        let source = std::mem::take(&mut entry.function);
        let mut target = Function {
            name: source.name.clone(),
            arguments: source.arguments.clone(),
            result: source.result.clone(),
            ..Function::default()
        };
        let (_, body) = append(&source, None, &module.functions, &mut target);
        target.body = body;
        entry.function = target;
    }
}

#[test]
fn helper_allowlist_accepts_only_the_numbered_writer_names() {
    assert!(allowed("iridia_f32_to_f16_0_"));
    assert!(allowed("iridia_umulhi_12_"));
    for other in [
        "user_function",
        "iridia_f32_to_f16_",
        "iridia_f32_to_f16_0_extra",
        "iridia_f32_to_f16_0__",
        "iridia_unknown_0_",
    ] {
        assert!(!allowed(other), "{other}");
    }
}
