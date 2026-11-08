//! Metadata checks made before allocating naga's expression arena. Only operations placed in active regions are
//! considered; an unused value-table entry is not an operation the lowering will visit. Builders retain defensive
//! checks for invalid IR and for their own handle/scope invariants.
use super::{Error, err, memory::Atomics, naga_scalar, size};
use crate::{BindingPlan, EntryIo};
use alloc::{format, vec::Vec};
use iridia_ir::{
    AtomicOp, Binary, End, FloatToInt, Function, GatherOffset, LabelId, Level, Memory, MinMaxNan, Node, Op,
    Program, Query, QueryField, Region, Res, SaturateNan, Scalar, Space, Type, Unary, ValueId, Wave, WaveOp,
    WaveScan,
    interface::{BufferKind, ResourceId, ResourceKind, SampleType, ScalarType, StorageTarget, TextureDim},
};
pub(super) const IMAGE_ATOMIC_RESULT: &str = "a texel atomic whose old value is read, or a compare-exchange of a texel, has no naga form: naga's image atomic gives no old value and takes no comparand";

/// WGSL's sample-level forms, shared with the expression builder so the upfront scan cannot retain an older rule.
pub(super) fn sample_level(
    dim: TextureDim,
    depth: bool,
    level: Level,
    offset: bool,
    compare: bool,
) -> Result<(), Error> {
    match (dim, level) {
        (TextureDim::D1, Level::Auto) => {}
        (TextureDim::D1, Level::Zero | Level::Exact(_)) if !offset && !compare => {}
        (TextureDim::D1, other) => {
            let at = match other {
                Level::Zero | Level::Exact(_) => "an explicit level with an offset or a comparison",
                Level::Bias(_) => "a biased level",
                Level::Grad(..) | Level::Auto => "explicit derivatives",
            };
            return Err(err(format!("a sample of a 1D texture at {at} has no WGSL form")));
        }
        (_, Level::Bias(_)) if depth => {
            return Err(err("a biased sample of a depth texture has no WGSL form"));
        }
        (_, Level::Grad(..)) if depth && compare => {
            return Err(err(
                "a comparison sample at explicit derivatives has no WGSL form",
            ));
        }
        _ => {}
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Shape {
    dim: TextureDim,
    arrayed: bool,
    multisampled: bool,
    sample: SampleType,
    storage: bool,
}
impl Shape {
    fn coordinate_lanes(self) -> u8 {
        match self.dim {
            TextureDim::D1 => 1,
            TextureDim::D2 => 2,
            TextureDim::D3 | TextureDim::Cube => 3,
        }
    }
    fn texel_lanes(self) -> Result<u8, Error> {
        if self.dim == TextureDim::Cube {
            Err(err("a texel address of a cube texture"))
        } else {
            Ok(self.coordinate_lanes())
        }
    }
    fn levels(self) -> bool {
        !self.storage && !self.multisampled
    }
}
struct Check<'a> {
    p: &'a Program,
    read: &'a [bool],
    io: &'a EntryIo,
}
impl Check<'_> {
    fn f(&self) -> &Function {
        &self.p.entry
    }
    fn ty(&self, v: ValueId) -> Result<Type, Error> {
        self.f().ty(v).ok_or_else(|| err(format!("%{} has no type", v.0)))
    }
    fn kind(&self, id: ResourceId) -> Result<&ResourceKind, Error> {
        self.p
            .interface
            .resource(id)
            .map(|r| &r.kind)
            .ok_or_else(|| err(format!("resource {} is not in the interface", id.0)))
    }
    fn typed(&self, id: ResourceId) -> Result<bool, Error> {
        Ok(matches!(
            self.kind(id)?,
            ResourceKind::Buffer(BufferKind::Typed(_))
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Typed(_)),
                    ..
                }
        ))
    }
    fn words(&self, id: ResourceId) -> Result<bool, Error> {
        Ok(matches!(
            self.kind(id)?,
            ResourceKind::Buffer(BufferKind::Raw | BufferKind::Structured { .. })
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(BufferKind::Raw | BufferKind::Structured { .. }),
                    ..
                }
        ))
    }
    fn shape(&self, r: Res) -> Result<Shape, Error> {
        Ok(match *self.kind(r.id)? {
            ResourceKind::Texture {
                dim,
                arrayed,
                multisampled,
                sample,
            } => Shape {
                dim,
                arrayed,
                multisampled,
                sample,
                storage: false,
            },
            ResourceKind::Storage {
                target: StorageTarget::Image { dim, arrayed, sample },
                ..
            } => Shape {
                dim,
                arrayed,
                multisampled: false,
                sample,
                storage: true,
            },
            ref other => return Err(err(format!("an image operation on a {other:?}"))),
        })
    }
    fn index(&self, v: ValueId) -> Result<(), Error> {
        let t = self.ty(v)?;
        if !matches!(t, Type::U32 | Type::I32) {
            return Err(err(format!("an index of {t}")));
        }
        Ok(())
    }
    fn integer_address(&self, v: ValueId) -> Result<Type, Error> {
        let t = self.ty(v)?;
        if !matches!(t.scalar, Scalar::I32 | Scalar::U32) {
            return Err(err(format!("an integer address of {t}")));
        }
        Ok(t)
    }
    fn sample_address(&self, s: Shape, v: ValueId) -> Result<(), Error> {
        let t = self.ty(v)?;
        if t.lanes != s.coordinate_lanes().saturating_add(u8::from(s.arrayed)) {
            return Err(err(format!(
                "a sample coordinate of {t} for a {:?} texture",
                s.dim
            )));
        }
        Ok(())
    }
    fn texel_address(
        &self,
        s: Shape,
        v: ValueId,
        sample: Option<ValueId>,
        level: Option<ValueId>,
    ) -> Result<(), Error> {
        let n = s.texel_lanes()?;
        let t = self.integer_address(v)?;
        if t.lanes != n.saturating_add(u8::from(s.arrayed)) {
            return Err(err(format!("a texel address of {t} for a {:?} image", s.dim)));
        }
        if s.levels()
            && let Some(l) = level
        {
            self.integer_address(l)?;
        }
        if s.multisampled {
            match sample {
                Some(v) => {
                    self.integer_address(v)?;
                }
                None => return Err(err("a load of a multisampled texture without its sample")),
            }
        }
        Ok(())
    }
    fn nan(&self, t: Type) -> Result<(), Error> {
        if !matches!(t.scalar, Scalar::F32 | Scalar::F64) {
            return Err(err(format!("a NaN test of {t} is not lowered yet")));
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments, reason = "the operands of one IR sample")]
    fn sample(
        &self,
        r: Res,
        coord: ValueId,
        level: Level,
        compare: Option<ValueId>,
        offset: [i8; 3],
        clamp: Option<ValueId>,
    ) -> Result<(), Error> {
        if clamp.is_some() {
            return Err(err("a sample with a level clamp has no WGSL form"));
        }
        let s = self.shape(r)?;
        if matches!(s.sample, SampleType::Sint | SampleType::Uint) {
            return Err(err("a filtered sample of an integer texture has no WGSL form"));
        }
        self.sample_address(s, coord)?;
        if offset != [0; 3] && s.dim == TextureDim::Cube {
            return Err(err("a texel offset of a cube sample"));
        }
        let depth = s.sample == SampleType::Depth;
        sample_level(s.dim, depth, level, offset != [0; 3], compare.is_some())?;
        match (level, compare.is_some()) {
            (Level::Auto | Level::Zero, _) => {}
            (other, true) => {
                return Err(err(format!(
                    "a comparison sample at level {other:?} has no WGSL form"
                )));
            }
            (Level::Exact(v), false) if depth => {
                let value = self.f().value(v).and_then(|d| match d.op {
                    Op::Literal(bits) if d.ty == Type::F32 => u32::try_from(bits[0]).ok().map(f32::from_bits),
                    _ => None,
                });
                if !value.is_some_and(|v| (0u8..=16).any(|n| f32::from(n) == v)) {
                    return Err(err(
                        "a depth sample at a computed or fractional level has no WGSL form",
                    ));
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn atomic(
        &self,
        v: ValueId,
        memory: Memory,
        op: AtomicOp,
        compare: Option<ValueId>,
    ) -> Result<(), Error> {
        match memory {
            Memory::Image(r, coord) if !self.typed(r.id)? => {
                if compare.is_some() || self.read.get(v.0 as usize).copied().unwrap_or(true) {
                    return Err(err(IMAGE_ATOMIC_RESULT));
                }
                let s = self.shape(r)?;
                if !s.storage {
                    return Err(err("an atomic of a texture that is not a storage image"));
                }
                match op {
                    AtomicOp::Add | AtomicOp::Sub | AtomicOp::And | AtomicOp::Or | AtomicOp::Xor => {}
                    AtomicOp::SMin | AtomicOp::SMax if s.sample == SampleType::Sint => {}
                    AtomicOp::UMin | AtomicOp::UMax if s.sample != SampleType::Sint => {}
                    AtomicOp::Exchange | AtomicOp::CompareExchange => {
                        return Err(err(
                            "an exchange of a texel has no naga form: naga's image atomic takes no exchange",
                        ));
                    }
                    other => {
                        return Err(err(format!(
                            "a {other:?} of a storage image of {:?} texels has no naga form: an image's minimum and maximum are of its texels' signedness",
                            s.sample
                        )));
                    }
                }
                self.texel_address(s, coord, None, None)?;
            }
            Memory::Buffer(r, address) => {
                if !self.words(r.id)? {
                    return Err(err(format!("a word atomic of resource {}", r.id.0)));
                }
                if let Some(v) = address.index {
                    self.index(v)?;
                }
            }
            Memory::Image(_, coord) => self.index(coord)?,
            Memory::Var(_, index) => {
                if let Some(v) = index {
                    self.index(v)?;
                }
            }
        }
        Ok(())
    }
    fn value(&self, v: ValueId) -> Result<(), Error> {
        let def = self
            .f()
            .value(v)
            .ok_or_else(|| err(format!("%{} is not a value", v.0)))?;
        let t = def.ty;
        check_type(t)?;
        match &def.op {
            Op::Literal(_) if t.scalar == Scalar::F16 => return Err(err("f16 literals are not lowered yet")),
            Op::Unary(u, a) => {
                let from = self.ty(*a)?;
                match u {
                    Unary::IsNan | Unary::FSign => self.nan(from)?,
                    Unary::IsInf if !matches!(from.scalar, Scalar::F32 | Scalar::F64) => {
                        return Err(err(format!("an infinity test of {from} is not lowered yet")));
                    }
                    Unary::Saturate(SaturateNan::Zero)
                    | Unary::ClampRange {
                        nan: SaturateNan::Zero,
                        ..
                    } => self.nan(from)?,
                    _ => {}
                }
            }
            Op::Binary(Binary::FRem, _, _) if t.scalar != Scalar::F32 => {
                return Err(err(format!("a remainder of {t} is not lowered yet")));
            }
            Op::Binary(Binary::FMin(nan) | Binary::FMax(nan), a, _) if *nan != MinMaxNan::Any => {
                self.nan(self.ty(*a)?)?
            }
            Op::Convert(a, rule) => {
                let from = self.ty(*a)?;
                if from.scalar.is_float() && t.scalar.is_integer() && *rule == FloatToInt::Saturate {
                    if !matches!(from.scalar, Scalar::F32 | Scalar::F64) {
                        return Err(err(format!(
                            "a saturating conversion from {from} is not lowered yet"
                        )));
                    }
                    if !matches!(t.scalar, Scalar::I32 | Scalar::U32) {
                        return Err(err(format!(
                            "a saturating conversion to {} is not lowered yet",
                            t.scalar.name()
                        )));
                    }
                }
            }
            Op::Bitcast(a) => {
                let from = self.ty(*a)?;
                if from.lanes != t.lanes {
                    let fb = from.scalar.bits();
                    let tb = t.scalar.bits();
                    if u32::from(from.lanes).saturating_mul(fb) != u32::from(t.lanes).saturating_mul(tb) {
                        return Err(err(format!("a bitcast of {from} to {t}")));
                    }
                    if !matches!((fb, tb), (32, 64) | (64, 32)) {
                        return Err(err(format!("a bitcast of {from} to {t} is not lowered yet")));
                    }
                }
            }
            Op::LoadState(r) => {
                return Err(err(format!(
                    "pipeline state r{} is read: specialise it first (a state block is not lowered yet)",
                    r.0
                )));
            }
            Op::LoadInput(slot, _) => {
                let carried = self
                    .io
                    .arguments
                    .iter()
                    .find(|a| a.varying == *slot as usize)
                    .ok_or_else(|| err(format!("input {slot} is not an argument of the entry point")))?
                    .value;
                if carried.scalar == ScalarType::Bool
                    && t.scalar != Scalar::Bool
                    && (!t.scalar.is_integer() || t.lanes != 1)
                {
                    return Err(err(format!("a flag input read as {t}")));
                }
            }
            Op::LoadInputAt(..) => return Err(err("a read of one vertex's input is not lowered yet")),
            Op::LoadOutputAt(..) => {
                return Err(err("a read of one control point's output is not lowered yet"));
            }
            Op::LoadPatch(..) => return Err(err("a patch constant read is not lowered yet")),
            Op::Builtin(b) => return Err(err(format!("builtin {b:?} is not lowered yet"))),
            Op::ReportHit { .. } => {
                return Err(err("an intersection program's reported hit is not lowered yet"));
            }
            Op::RayQueryState(_, QueryField::CandidateNonOpaque) => {
                return Err(err(
                    "whether a ray query's candidate procedural primitive is opaque is not lowered: naga's intersection does not say",
                ));
            }
            Op::Wave(Wave::Collective { op, scan, .. })
                if *scan != WaveScan::Reduce && !matches!(op, WaveOp::Add | WaveOp::Mul) =>
            {
                return Err(err(format!("a wave scan of {op:?} has no naga form")));
            }
            Op::Wave(Wave::QuadLane(_, lane))
                if !self
                    .f()
                    .value(*lane)
                    .is_some_and(|d| matches!(d.op, Op::Literal(_))) =>
            {
                return Err(err("a quad read of a computed lane is not lowered yet"));
            }
            Op::Atomic {
                memory, op, compare, ..
            } => self.atomic(v, *memory, *op, *compare)?,
            Op::LoadBuffer {
                resource, address, ..
            } => {
                if self.words(resource.id)? {
                    if t.scalar.bits() != 32 {
                        return Err(err(format!("a buffer read of {t} is not lowered yet")));
                    }
                    if let Some(i) = address.index {
                        self.index(i)?;
                    }
                } else {
                    let offset = address.offset;
                    let stride = address.stride;
                    if t.scalar.bits() != 32
                        || !offset.is_multiple_of(4)
                        || (!stride.is_multiple_of(16) && address.index.is_some())
                    {
                        return Err(err(format!(
                            "a buffer read of {t} at stride {stride}, offset {offset} is not lowered yet"
                        )));
                    }
                    if ((offset % 16) / 4).saturating_add(u32::from(t.lanes)) > 4 {
                        return Err(err(format!(
                            "a buffer read of {t} across a row at offset {offset}"
                        )));
                    }
                    let rows = match self.kind(resource.id)? {
                        ResourceKind::ConstantBuffer { layout } => layout.size.div_ceil(16),
                        _ => {
                            return Err(err(format!(
                                "resource {} is not a constant buffer",
                                resource.id.0
                            )));
                        }
                    };
                    if let Some(i) = address.index {
                        self.index(i)?;
                    } else if offset / 16 >= rows {
                        return Err(err(format!(
                            "a read of row {} of a buffer of {rows}",
                            offset / 16
                        )));
                    }
                }
            }
            Op::Sample {
                texture,
                coord,
                level,
                compare,
                offset,
                clamp,
                ..
            } => self.sample(*texture, *coord, *level, *compare, *offset, *clamp)?,
            Op::Gather {
                texture,
                coord,
                component,
                offset,
                ..
            } => {
                let s = self.shape(*texture)?;
                match offset {
                    GatherOffset::Dynamic(_) => {
                        return Err(err("a gather at a computed offset has no WGSL form"));
                    }
                    GatherOffset::Fixed(o) if *o != [0; 3] && s.dim == TextureDim::Cube => {
                        return Err(err("a texel offset of a cube gather"));
                    }
                    _ => {}
                }
                if s.sample == SampleType::Depth && *component != 0 {
                    return Err(err(format!("a gather of lane {component} of a depth texture")));
                }
                self.sample_address(s, *coord)?;
            }
            Op::LoadTexel {
                texture,
                coord,
                level,
                sample,
                ..
            } => {
                if self.typed(texture.id)? {
                    if t.lanes != 4 || t.scalar.bits() != 32 {
                        return Err(err(format!("a typed buffer read of {t}")));
                    }
                    self.index(*coord)?;
                } else {
                    self.texel_address(self.shape(*texture)?, *coord, *sample, *level)?;
                }
            }
            Op::Query(r, Query::Elements) => match self.kind(r.id)? {
                ResourceKind::Buffer(_)
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(_),
                    ..
                } => {}
                other => return Err(err(format!("the elements of a {other:?}"))),
            },
            Op::Query(r, q) => {
                let s = self.shape(*r)?;
                match q {
                    Query::Size(level) => {
                        let lanes = match s.dim {
                            TextureDim::D1 => 1,
                            TextureDim::D3 => 3,
                            _ => 2,
                        };
                        if t != Type::vector(Scalar::U32, lanes) && !(lanes == 1 && t == Type::U32) {
                            return Err(err(format!("a size query of {t} for a {:?} image", s.dim)));
                        }
                        if s.levels()
                            && let Some(l) = level
                        {
                            self.integer_address(*l)?;
                        }
                    }
                    Query::Layers if s.arrayed => {}
                    Query::Levels if s.levels() => {}
                    Query::Samples if s.multisampled => {}
                    Query::SamplePosition(i) if s.multisampled => {
                        self.integer_address(*i)?;
                    }
                    other => {
                        let what = match other {
                            Query::Size(_) => "size",
                            Query::Layers => "layers",
                            Query::Levels => "mip levels",
                            Query::Samples => "samples",
                            Query::SamplePosition(_) => "sample position",
                            Query::Elements => "elements",
                        };
                        let image = match (s.storage, s.multisampled, s.arrayed) {
                            (true, ..) => "storage image",
                            (_, true, true) => "multisampled array texture",
                            (_, true, false) => "multisampled texture",
                            (_, false, true) => "array texture",
                            _ => "texture",
                        };
                        return Err(err(format!(
                            "a query of the {what} of a {:?} {image}, which has none",
                            s.dim
                        )));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
    fn region(&self, r: &Region) -> Result<(), Error> {
        for node in &r.nodes {
            match node {
                Node::Value(v) => self.value(*v)?,
                Node::StoreBuffer {
                    resource,
                    value,
                    address,
                } => {
                    if !self.words(resource.id)? {
                        return Err(err(format!("a buffer write of resource {}", resource.id.0)));
                    }
                    let t = self.ty(*value)?;
                    if t.scalar.bits() != 32 {
                        return Err(err(format!("a buffer write of {t} is not lowered yet")));
                    }
                    if let Some(i) = address.index {
                        self.index(i)?;
                    }
                }
                Node::StoreTexel { image, coord, value } => {
                    if self.typed(image.id)? {
                        let t = self.ty(*value)?;
                        if t.lanes != 4 || t.scalar.bits() != 32 {
                            return Err(err(format!("a typed buffer write of {t}")));
                        }
                        self.index(*coord)?;
                    } else {
                        let s = self.shape(*image)?;
                        if !s.storage {
                            return Err(err("a texel store of a texture that is not a storage image"));
                        }
                        self.texel_address(s, *coord, None, None)?;
                    }
                }
                Node::StorePatch(..) => return Err(err("a patch constant store is not lowered yet")),
                Node::Emit(_) | Node::Cut(_) => {
                    return Err(err("a geometry program's emit and cut are not lowered yet"));
                }
                Node::SetMeshCounts { .. } | Node::StoreVertex { .. } | Node::StoreIndices { .. } => {
                    return Err(err("a mesh program's meshlet is not lowered yet"));
                }
                Node::DispatchMesh { .. } => {
                    return Err(err("an amplification program's mesh launch is not lowered yet"));
                }
                Node::TraceRay { .. } | Node::CallShader { .. } => {
                    return Err(err("a traced ray or a shader call is not lowered yet"));
                }
                Node::RayQueryInit { scene, .. } if scene.index.is_some() => {
                    return Err(err(
                        "a ray query in an element of an array of acceleration structures is not lowered yet",
                    ));
                }
                _ => {}
            }
            for child in node.regions() {
                self.region(child)?;
            }
        }
        if matches!(r.end, Some(End::IgnoreHit | End::AcceptHitAndEndSearch)) {
            return Err(err("an any-hit program's decision is not lowered yet"));
        }
        Ok(())
    }
}
fn check_type(t: Type) -> Result<(), Error> {
    naga_scalar(t.scalar)?;
    if t.lanes != 1 {
        size(t.lanes)?;
    }
    Ok(())
}

/// Resolve exits lexically, including shadowed labels, before building naga's local/result handles.
struct ExitTarget {
    label: LabelId,
    kind: super::Kind,
    results: usize,
    params: Option<usize>,
}
fn arity(expected: usize, actual: usize) -> Result<(), Error> {
    if expected != actual {
        return Err(err(
            "an exit with another number of values than its construct takes",
        ));
    }
    Ok(())
}
fn exits(r: &Region, stack: &mut Vec<ExitTarget>, yields: Option<usize>) -> Result<(), Error> {
    use super::Kind;
    for node in &r.nodes {
        if let Node::Loop { params, inits, .. } = node {
            arity(params.len(), inits.len())?;
        }
        let construct = match node {
            Node::Block { label, results, .. } => Some(ExitTarget {
                label: *label,
                kind: Kind::Block,
                results: results.len(),
                params: None,
            }),
            Node::Loop {
                label,
                params,
                results,
                ..
            } => Some(ExitTarget {
                label: *label,
                kind: Kind::Loop,
                results: results.len(),
                params: Some(params.len()),
            }),
            Node::Switch { label, results, .. } => Some(ExitTarget {
                label: *label,
                kind: Kind::Switch,
                results: results.len(),
                params: None,
            }),
            _ => None,
        };
        let labelled = construct.is_some();
        if let Some(c) = construct {
            stack.push(c);
        }
        let count = match node {
            Node::If { results, .. }
            | Node::Block { results, .. }
            | Node::Loop { results, .. }
            | Node::Switch { results, .. } => results.len(),
            _ => 0,
        };
        for child in node.regions() {
            exits(child, stack, Some(count))?;
        }
        if labelled {
            stack.pop();
        }
    }
    match &r.end {
        None => return Err(err("a region with no end")),
        Some(End::Yield(vs)) if yields.is_none() && !vs.is_empty() => {
            return Err(err("a yield of values at the end of the function"));
        }
        Some(End::Yield(vs)) => arity(yields.unwrap_or(0), vs.len())?,
        Some(End::Br(label, vs) | End::Continue(label, vs)) => {
            let continues = matches!(r.end, Some(End::Continue(..)));
            let (at, target) = stack
                .iter()
                .enumerate()
                .rev()
                .find(|(_, target)| target.label == *label)
                .ok_or_else(|| {
                    err(format!(
                        "an exit to label {}, which no enclosing construct has",
                        label.0
                    ))
                })?;
            if continues && target.kind != Kind::Loop {
                return Err(err("a continue to a construct that is not a loop"));
            }
            arity(
                if continues {
                    target.params.unwrap_or(0)
                } else {
                    target.results
                },
                vs.len(),
            )?;
            if !continues && target.kind == Kind::Block {
                return Err(err(
                    "a break out of a block: run iridia_passes::break_flags first",
                ));
            }
            let nearest = stack
                .iter()
                .rposition(|target| target.kind == Kind::Loop || (!continues && target.kind == Kind::Switch));
            if nearest != Some(at) {
                return Err(err(
                    "an exit past an enclosing construct: run iridia_passes::break_flags first",
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn program(
    p: &Program,
    plan: &BindingPlan,
    body: &Region,
    read: &[bool],
    atomics: &Atomics,
    counted: &[ResourceId],
    io: &EntryIo,
) -> Result<(), Error> {
    for r in &p.interface.resources {
        if matches!(r.kind, ResourceKind::PipelineState(_)) {
            continue;
        }
        if r.count == 0 {
            return Err(err("an array of no resources"));
        }
        if !plan.bindings.iter().any(|b| b.resource == r.id) {
            return Err(err(format!(
                "resource {} ({}) has no binding in the plan",
                r.id.0, r.provenance.name
            )));
        }
        match r.kind {
            ResourceKind::ConstantBuffer { ref layout } if layout.size == 0 => {
                return Err(err("an empty constant buffer"));
            }
            ResourceKind::Texture {
                arrayed: true,
                multisampled: true,
                ..
            } => return Err(err("a multisampled array texture has no WGSL form")),
            ResourceKind::Texture {
                dim: TextureDim::D1,
                arrayed: true,
                ..
            } => return Err(err("a 1D array texture has no WGSL form")),
            ResourceKind::Storage {
                target: StorageTarget::Image { dim, arrayed, sample },
                ..
            } => {
                if dim == TextureDim::D1 && arrayed {
                    return Err(err("a 1D array storage image has no WGSL form"));
                }
                if sample == SampleType::Depth
                    || (sample == SampleType::Float && atomics.resources.contains_key(&r.id))
                {
                    return Err(err(format!(
                        "a storage image of {sample:?} texels an atomic reaches"
                    )));
                }
                if dim == TextureDim::Cube {
                    return Err(err("a cube storage image"));
                }
            }
            ResourceKind::ConstantBuffer { .. }
            | ResourceKind::Texture { .. }
            | ResourceKind::Storage { .. }
            | ResourceKind::Buffer(_)
            | ResourceKind::Sampler { .. }
            | ResourceKind::AccelerationStructure => {}
            ref other => return Err(err(format!("a {other:?} resource is not lowered yet"))),
        }
    }
    for id in counted {
        if !plan.counters.iter().any(|b| b.resource == *id) {
            return Err(err(format!(
                "the counter of resource {} has no binding in the plan",
                id.0
            )));
        }
    }
    for (n, v) in p.entry.vars.iter().enumerate() {
        if v.space == Space::Payload {
            return Err(err("a mesh program's task payload is not lowered yet"));
        }
        let atomic = u32::try_from(n)
            .ok()
            .is_some_and(|n| atomics.vars.contains_key(&n));
        if atomic && (v.space != Space::Workgroup || v.ty.lanes != 1) {
            return Err(err(format!(
                "an atomic of {} variable {} is not lowered yet",
                if v.space == Space::Workgroup {
                    "the vector"
                } else {
                    "the private"
                },
                v.name
            )));
        }
        check_type(v.ty)?;
        if v.len == Some(0) {
            return Err(err("an array of no elements"));
        }
        if v.space == Space::Workgroup && !v.init.is_empty() {
            return Err(err(format!("workgroup variable {} has an initial value", v.name)));
        }
        if v.ty.scalar == Scalar::F16 && !v.init.is_empty() {
            return Err(err("f16 literals are not lowered yet"));
        }
    }
    Check { p, read, io }.region(body)?;
    exits(body, &mut Vec::new(), None)
}
