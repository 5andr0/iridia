//! Material feature decisions, with the caller's explicitly supplied uniform words.
use alloc::{
    collections::{BTreeMap, BTreeSet},
    string::String,
    vec,
    vec::Vec,
};

use iridia_ir::{
    Address, Body, Denormals, Function, Node, Op, Program, Region, Res, Type, Unary, ValueDef, ValueId,
    interface::{ResourceId, ResourceKind},
};

/// Known raw constant-buffer words, keyed by resource and byte offset.
pub type Words = BTreeMap<(ResourceId, u32), u32>;
mod known;
mod lanes;

type Word = (ResourceId, u32);
type Dependencies = Option<BTreeSet<Word>>;

/// Why material specialisation could not produce a validated, fully decided copy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpecialiseError {
    /// The source or transformed program fails IR validation.
    InvalidProgram {
        /// Whether validation followed the transformation.
        transformed: bool,
        /// The validator's diagnostics.
        problems: Vec<iridia_ir::Problem>,
    },
    /// A supplied word names no resource.
    MissingResource(ResourceId),
    /// A supplied word names a resource other than a constant buffer.
    NotConstantBuffer(ResourceId),
    /// A supplied word is unaligned or outside its buffer.
    WordOutOfBounds {
        /// The buffer.
        resource: ResourceId,
        /// The supplied byte offset.
        offset: u32,
    },
    /// The program cannot be prepared for specialisation.
    Unsupported {
        /// The preparation pass's explanation.
        reason: String,
    },
    /// A decision wholly fixed by supplied words uses an operation the fold cannot evaluate.
    UnresolvedDecision {
        /// The condition in the prepared function.
        value: ValueId,
    },
}

impl core::fmt::Display for SpecialiseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidProgram {
                transformed,
                problems,
            } => write!(
                f,
                "{} program invalid: {problems:?}",
                if *transformed { "specialised" } else { "source" }
            ),
            Self::MissingResource(id) => write!(f, "known word names absent resource {}", id.0),
            Self::NotConstantBuffer(id) => write!(f, "resource {} is not a constant buffer", id.0),
            Self::WordOutOfBounds { resource, offset } => {
                write!(f, "known word {}+{offset} outside constant buffer", resource.0)
            }
            Self::Unsupported { reason } => f.write_str(reason),
            Self::UnresolvedDecision { value } => write!(
                f,
                "decision %{} fixed by the stated words was not evaluated",
                value.0
            ),
        }
    }
}
impl core::error::Error for SpecialiseError {}

/// A specialised copy and the evidence for its eliminated resource reads.
#[must_use]
#[derive(Clone, Debug)]
pub struct Specialised {
    /// The validated program, with feature decisions taken and dead values removed.
    pub program: Program,
    /// The supplied words selected as fully stated feature decisions.
    pub words: Words,
    /// Resource declarations removed because the resulting program no longer reads them.
    pub unread: Vec<ResourceId>,
    /// Number of value results replaced by known values (including scalar words of vector loads).
    pub folded: usize,
}

fn validate(p: &Program, transformed: bool) -> Result<(), SpecialiseError> {
    iridia_ir::validate(p).map_err(|problems| SpecialiseError::InvalidProgram {
        transformed,
        problems,
    })
}

fn prepare(source: &Program, words: &Words) -> Result<Program, SpecialiseError> {
    validate(source, false)?;
    for &(id, offset) in words.keys() {
        let r = source
            .interface
            .resource(id)
            .ok_or(SpecialiseError::MissingResource(id))?;
        let ResourceKind::ConstantBuffer { layout } = &r.kind else {
            return Err(SpecialiseError::NotConstantBuffer(id));
        };
        if offset % 4 != 0 || offset.checked_add(4).is_none_or(|end| end > layout.size) {
            return Err(SpecialiseError::WordOutOfBounds { resource: id, offset });
        }
    }
    if source.patch.is_some() {
        return Err(SpecialiseError::Unsupported {
            reason: "material specialisation of a patch constant function is not implemented".into(),
        });
    }
    let mut p = source.clone();
    if matches!(p.entry.body, Body::Cfg(_)) {
        crate::structurize(&mut p.entry).map_err(|reason| SpecialiseError::Unsupported { reason })?;
    }
    crate::promote(&mut p.entry).map_err(|reason| SpecialiseError::Unsupported { reason })?;
    crate::simplify(&mut p.entry);
    crate::fold(&mut p.entry);
    Ok(p)
}

fn constant(f: &Function, id: ValueId) -> Option<u32> {
    match f.value(id)?.op {
        Op::Literal(bits) => u32::try_from(*bits.first()?).ok(),
        _ => None,
    }
}

fn address(f: &Function, r: Res, a: Address) -> Option<u32> {
    if r.index.is_some_and(|id| constant(f, id) != Some(0)) {
        return None;
    }
    let index = match a.index {
        Some(id) => constant(f, id)?,
        None => 0,
    };
    Some(a.offset.wrapping_add(a.stride.wrapping_mul(index)) & !3)
}

/// Dependencies of one lane. The explicit stack keeps long source expression chains off the host stack.
fn dependencies(
    f: &Function,
    id: ValueId,
    lane: u8,
    known: &Words,
    aliases: &lanes::Aliases,
    cache: &mut BTreeMap<(ValueId, u8), Dependencies>,
) -> Dependencies {
    let mut stack = vec![((id, lane), false)];
    while let Some((key @ (id, lane), expanded)) = stack.pop() {
        if !expanded && cache.contains_key(&key) {
            continue;
        }
        let Some(d) = f.value(id) else {
            cache.insert(key, None);
            continue;
        };
        let same = |v: ValueId| {
            (
                v,
                if f.ty(v).is_some_and(|t| t.lanes == 1) {
                    0
                } else {
                    lane
                },
            )
        };
        let operands: Option<Vec<(ValueId, u8)>> = match &d.op {
            Op::Literal(_) => Some(Vec::new()),
            Op::Param => lanes::root(aliases, id, lane).map(|root| vec![root]),
            Op::LoadBuffer {
                resource,
                address: at,
                ..
            } if d.ty.scalar.bits() == 32 => {
                let found = address(f, *resource, *at)
                    .and_then(|offset| offset.checked_add(u32::from(lane).checked_mul(4)?))
                    .map(|offset| (resource.id, offset));
                cache.insert(
                    key,
                    found
                        .filter(|w| known.contains_key(w))
                        .map(|w| BTreeSet::from([w])),
                );
                continue;
            }
            Op::Swizzle(v, picks) => picks.get(usize::from(lane)).map(|&l| vec![(*v, l)]),
            Op::Splat(v) => Some(vec![(*v, 0)]),
            Op::Compose(parts) => {
                let mut l = lane;
                parts.iter().find_map(|&v| {
                    let n = f.ty(v)?.lanes;
                    if l < n {
                        Some(vec![(v, l)])
                    } else {
                        l = l.saturating_sub(n);
                        None
                    }
                })
            }
            Op::Dot(a, b) => Some(
                [a, b]
                    .into_iter()
                    .flat_map(|&v| (0..f.ty(v).map_or(0, |t| t.lanes)).map(move |l| (v, l)))
                    .collect(),
            ),
            Op::Unary(Unary::Any | Unary::All, v) => {
                Some((0..f.ty(*v).map_or(0, |t| t.lanes)).map(|l| (*v, l)).collect())
            }
            Op::Bitcast(v) if f.ty(*v).is_none_or(|t| t.scalar.bits() != d.ty.scalar.bits()) => None,
            Op::Unary(..)
            | Op::Binary(..)
            | Op::Ternary(..)
            | Op::BitInsert { .. }
            | Op::Select { .. }
            | Op::Convert(..)
            | Op::Bitcast(_) => Some(d.op.operands().into_iter().map(same).collect()),
            _ => None,
        };
        let Some(operands) = operands else {
            cache.insert(key, None);
            continue;
        };
        if !expanded {
            // A cyclic dependency is unknown; expanded acyclic nodes replace this provisional entry.
            cache.insert(key, None);
            stack.push((key, true));
            stack.extend(
                operands
                    .into_iter()
                    .filter(|o| !cache.contains_key(o))
                    .map(|o| (o, false)),
            );
        } else {
            let result = operands.iter().try_fold(BTreeSet::new(), |mut out, o| {
                out.extend(cache.get(o)?.as_ref()?.iter().copied());
                Some(out)
            });
            cache.insert(key, result);
        }
    }
    cache.get(&(id, lane)).cloned().flatten()
}

fn decisions(f: &Function) -> Vec<ValueId> {
    fn region(f: &Function, r: &Region, out: &mut Vec<ValueId>) {
        for n in &r.nodes {
            match n {
                Node::If {
                    cond, accept, reject, ..
                } if accept != reject => out.push(*cond),
                Node::Switch { selector, .. } => out.push(*selector),
                Node::Value(v) => {
                    if let Some(d) = f.value(*v)
                        && let Op::Select { cond, accept, reject } = d.op
                        && accept != reject
                    {
                        out.push(cond);
                    }
                }
                _ => {}
            }
            for r in n.regions() {
                region(f, r, out);
            }
        }
    }
    let mut out = Vec::new();
    if let Body::Regions(r) = &f.body {
        region(f, r, &mut out);
    }
    out
}

fn selected(f: &Function, known: &Words) -> Words {
    let aliases = lanes::aliases(f);
    let mut cache = BTreeMap::new();
    let mut words = BTreeSet::new();
    for id in decisions(f) {
        let found = (0..f.ty(id).map_or(0, |t| t.lanes)).try_fold(BTreeSet::new(), |mut out, lane| {
            out.extend(dependencies(f, id, lane, known, &aliases, &mut cache)?);
            Some(out)
        });
        if let Some(found) = found {
            words.extend(found);
        }
    }
    known
        .iter()
        .filter(|(key, _)| words.contains(key))
        .map(|(&k, &v)| (k, v))
        .collect()
}

/// The supplied words that decide a branch or select without runtime input. Ordinary colours and scales stay
/// uniforms. A condition using unknown words, input lanes, texture samples or loop state selects no words.
pub fn feature_words(source: &Program, known: &Words) -> Result<Words, SpecialiseError> {
    let p = prepare(source, known)?;
    Ok(selected(&p.entry, known))
}

fn replace_reads(f: &mut Function, words: &Words) -> usize {
    let mut replacements = BTreeMap::new();
    let mut count = 0usize;
    for index in 0..f.values.len() {
        let Some(d) = f.values.get(index).cloned() else {
            continue;
        };
        let Op::LoadBuffer {
            resource,
            address: at,
            oob,
        } = d.op
        else {
            continue;
        };
        if d.ty.scalar.bits() != 32 {
            continue;
        }
        let Some(offset) = address(f, resource, at) else {
            continue;
        };
        let lanes: Vec<_> = (0..d.ty.lanes)
            .map(|l| offset.checked_add(u32::from(l).saturating_mul(4)))
            .collect();
        if !lanes
            .iter()
            .any(|o| o.is_some_and(|o| words.contains_key(&(resource.id, o))))
        {
            continue;
        }
        let mut parts = Vec::new();
        let mut placed = Vec::new();
        for offset in lanes.into_iter().flatten() {
            let op = if let Some(&bits) = words.get(&(resource.id, offset)) {
                count = count.saturating_add(1);
                Op::Literal([u64::from(bits), 0, 0, 0])
            } else {
                Op::LoadBuffer {
                    resource,
                    address: Address::fixed(offset),
                    oob,
                }
            };
            let id = ValueId(u32::try_from(f.values.len()).unwrap_or(u32::MAX));
            f.values.push(ValueDef {
                ty: Type {
                    scalar: d.ty.scalar,
                    lanes: 1,
                },
                op,
                origin: d.origin,
                precision: d.precision,
            });
            parts.push(id);
            placed.push(Node::Value(id));
        }
        if let Some(d) = f.values.get_mut(index) {
            d.op = Op::Compose(parts);
        }
        replacements.insert(ValueId(u32::try_from(index).unwrap_or(u32::MAX)), placed);
    }
    fn insert(r: &mut Region, replacements: &BTreeMap<ValueId, Vec<Node>>) {
        let mut nodes = Vec::new();
        for mut n in core::mem::take(&mut r.nodes) {
            if let Node::Value(id) = n
                && let Some(placed) = replacements.get(&id)
            {
                nodes.extend(placed.clone());
            }
            for r in n.regions_mut() {
                insert(r, replacements);
            }
            nodes.push(n);
        }
        r.nodes = nodes;
    }
    if let Body::Regions(r) = &mut f.body {
        insert(r, &replacements);
    }
    count
}

/// Fold fully known pure operations through the same numeric definition the reference executor uses.
/// This chooses that reference's result, including its portable transcendental backend and its default of preserving
/// denormals where the program leaves them open. Reduced-precision permissions remain explicit refusals.
fn reference_values(f: &mut Function, denormals: Denormals) -> usize {
    let aliases = lanes::aliases(f);
    let known = known::values(f, &aliases, denormals);
    let mut count = 0usize;
    for (index, d) in f.values.iter_mut().enumerate() {
        if matches!(d.op, Op::Literal(_) | Op::Param) {
            continue;
        }
        let Ok(index) = u32::try_from(index) else { break };
        let mut bits = [0; 4];
        let complete = bits
            .iter_mut()
            .enumerate()
            .take(usize::from(d.ty.lanes))
            .all(|(lane, out)| {
                let Ok(lane) = u8::try_from(lane) else {
                    return false;
                };
                let Some(Some(word)) = known.get(&(ValueId(index), lane)) else {
                    return false;
                };
                *out = *word;
                true
            });
        if complete {
            d.op = Op::Literal(bits);
            count = count.saturating_add(1);
        }
    }
    count
}

fn take_selects(f: &mut Function) {
    for index in 0..f.values.len() {
        let Some(d) = f.values.get(index) else {
            continue;
        };
        let Op::Select { cond, accept, reject } = d.op else {
            continue;
        };
        let Some(c) = f.value(cond) else {
            continue;
        };
        let Op::Literal(bits) = c.op else {
            continue;
        };
        let first = bits.first().copied().unwrap_or(0) != 0;
        if bits
            .iter()
            .take(usize::from(c.ty.lanes))
            .any(|&b| (b != 0) != first)
        {
            continue;
        }
        if let Some(d) = f.values.get_mut(index) {
            d.op = Op::Swizzle(if first { accept } else { reject }, [0, 1, 2, 3]);
        }
    }
}

/// Fold fully stated material feature decisions into a validated copy, by [`iridia_ir::eval`], the reference
/// executor's numeric definition (one portable `libm` backend, denormals preserved where unspecified). This fixes
/// that reference result rather than a platform math library's choice. Reduced-precision decisions are refused.
/// Only words selected by [`feature_words`]
/// are frozen; all other uniform values remain runtime inputs. A selected decision the fold cannot take returns
/// [`SpecialiseError::UnresolvedDecision`]. The source is never modified.
pub fn specialise(source: &Program, known: &Words) -> Result<Specialised, SpecialiseError> {
    let mut p = prepare(source, known)?;
    let words = selected(&p.entry, known);
    let aliases = lanes::aliases(&p.entry);
    let required: BTreeSet<_> = decisions(&p.entry)
        .into_iter()
        .filter(|&id| {
            let mut cache = BTreeMap::new();
            (0..p.entry.ty(id).map_or(0, |t| t.lanes))
                .try_fold(BTreeSet::new(), |mut out, l| {
                    out.extend(dependencies(&p.entry, id, l, &words, &aliases, &mut cache)?);
                    Some(out)
                })
                .is_some_and(|d| !d.is_empty())
        })
        .collect();
    let mut folded = replace_reads(&mut p.entry, &words);
    loop {
        let before = p.entry.clone();
        let n = crate::fold(&mut p.entry).saturating_add(reference_values(&mut p.entry, p.float.f32));
        folded = folded.saturating_add(n);
        take_selects(&mut p.entry);
        super::specialise::fold_control(&mut p.entry);
        crate::simplify(&mut p.entry);
        if p.entry == before {
            break;
        }
    }
    if let Some(value) = decisions(&p.entry).into_iter().find(|id| required.contains(id)) {
        return Err(SpecialiseError::UnresolvedDecision { value });
    }
    let read = super::specialise::read(&p.entry);
    let unread = p
        .interface
        .resources
        .iter()
        .filter(|r| !read.contains(&r.id))
        .map(|r| r.id)
        .collect();
    p.interface.resources.retain(|r| read.contains(&r.id));
    validate(&p, true)?;
    Ok(Specialised {
        program: p,
        words,
        unread,
        folded,
    })
}
