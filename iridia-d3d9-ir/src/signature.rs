//! What a program's declarations, implied registers and constant table say about its interface: the varyings by
//! semantic, the float, integer and boolean constant banks with the CTAB layout of their registers, a texture and
//! sampler per sampler, and the pipeline state the code reads.

use std::collections::BTreeMap;

use iridia_ir::interface::{
    BufferLayout, Constant as Field, ConstantType, D3dBinding, D3dClass, D3dSignature, Interface, InterpMode,
    InterpSampling, Interpolation, Label, Majorness, Resource, ResourceId, ResourceKind, ResourceProvenance,
    SampleType, ScalarType, Stage, StageFacts, StateParam, SystemValue, TextureDim, Value, Varying,
    VaryingProvenance,
};

use crate::{Constant, Register, Result, Shader, error, model::usage};

/// D3DDECLUSAGE's names, by usage number.
const SEMANTICS: [&str; 14] = [
    "POSITION",
    "BLENDWEIGHT",
    "BLENDINDICES",
    "NORMAL",
    "PSIZE",
    "TEXCOORD",
    "TANGENT",
    "BINORMAL",
    "TESSFACTOR",
    "POSITIONT",
    "COLOR",
    "FOG",
    "DEPTH",
    "SAMPLE",
];

/// The float constant registers of a vertex program, the rows a relative read can reach.
pub const VERTEX_CONSTANTS: u32 = 256;

/// The float constant registers of a pixel 3.0 program, the rows a read relative to `aL` can reach.
pub const PIXEL_CONSTANTS: u32 = 224;

/// The partial-precision note: a request the lifters compute at `f32`.
pub const PARTIAL_PRECISION: &str = "partial-precision requests computed at f32";

/// A constant bank: the resource and its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bank {
    /// The constant buffer resource.
    pub resource: ResourceId,
    /// Its registers: past the last register read or allocated, the whole bank where a read is relative.
    pub rows: u32,
}

/// A sampler's texture and sampler resources.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sampler {
    /// The sampler state.
    pub sampler: ResourceId,
    /// The texture for each dimension it can have: the declared one, or for a pixel 1.x texture stage whose dimension
    /// is not known, one each for 2D, cube and volume.
    pub textures: Vec<(TextureDim, ResourceId)>,
    /// The stage's `StateParam::TextureDim`, where there is more than one texture to pick from.
    pub dim_state: Option<ResourceId>,
}

/// A program's interface before its code is lowered: the inputs (none used yet), the resources, and no outputs (see
/// [`outputs`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// The interface: inputs in declaration order, the banks, a texture (or one per possible dimension) and a sampler
    /// per sampler register, and pipeline state as the lift reads it.
    pub interface: Interface,
    /// The input slot of each input register, with its declared mask.
    pub inputs: BTreeMap<Register, (u32, u8)>,
    /// The float constant bank, where the program reads or allocates a register of it.
    pub bank: Option<Bank>,
    /// The integer constant bank (`iN`, a `vec4<i32>` each), where the program reads one.
    pub integers: Option<Bank>,
    /// The boolean constant bank (`bN`, a 32-bit boolean each, packed), where the program reads one.
    pub booleans: Option<Bank>,
    /// The samplers by register (texture stage, in pixel 1.x).
    pub samplers: BTreeMap<u32, Sampler>,
    /// What the declarations show that the lift cannot state exactly.
    pub notes: Vec<String>,
}

impl Signature {
    /// The pipeline state resource of `param`, declared the first time it is asked for.
    pub fn state(&mut self, param: StateParam) -> Result<ResourceId> {
        let found = self
            .interface
            .resources
            .iter()
            .find(|r| r.kind == ResourceKind::PipelineState(param))
            .map(|r| r.id);
        match found {
            Some(id) => Ok(id),
            None => resource(
                &mut self.interface,
                ResourceKind::PipelineState(param),
                format!("{param:?}"),
                None,
            ),
        }
    }
}

fn value(width: u8) -> Value {
    Value {
        scalar: ScalarType::Float,
        width,
    }
}

/// How many lanes a write mask names.
pub fn lanes_in(mask: u8) -> u8 {
    u8::try_from(mask.count_ones()).unwrap_or(4)
}

/// The semantic name of a declaration usage.
fn semantic(usage: u8, at: usize) -> Result<&'static str> {
    SEMANTICS.get(usize::from(usage)).copied().ok_or_else(|| {
        error(
            at,
            format!(
                "declaration usage {usage}, where D3DDECLUSAGE is 0 to {}",
                SEMANTICS.len().saturating_sub(1)
            ),
        )
    })
}

/// A float varying of `name` and `index` with the lanes of `mask`, at register `register`, unused.
pub fn varying(name: &str, index: u32, mask: u8, register: u32, system: Option<SystemValue>) -> Varying {
    Varying {
        label: Label {
            name: name.into(),
            index,
        },
        system,
        value: value(lanes_in(mask)),
        interp: None,
        used: false,
        provenance: VaryingProvenance {
            d3d: Some(D3dSignature {
                register,
                mask,
                read_write_mask: 0,
            }),
        },
    }
}

/// The registers one element of a CTAB type takes: a row-major matrix's rows, a column-major one's columns, one
/// otherwise.
fn lines(t: &crate::Type) -> u16 {
    match t.class {
        2 => t.rows,
        3 => t.columns,
        _ => 1,
    }
}

/// A float constant's layout. The bank's register layout is exact even for an optimized partial array: CTAB's
/// unshortened type is retained by `read`, and the interface describes the allocation the program binds.
fn field(c: &Constant) -> Result<Field> {
    let t = &c.ty;
    if c.register_set != 2
        || t.kind != 3
        || !t.members.is_empty()
        || !(1..=4).contains(&t.rows)
        || !(1..=4).contains(&t.columns)
        || t.elements == 0
    {
        return Err(error(
            0,
            format!("CTAB {}: non-float/structured/empty type {:?}", c.name, t),
        ));
    }
    // rows and columns are 1 to 4, checked above
    let lanes = |n: u16| u8::try_from(n).map_err(|_| error(0, format!("CTAB {}: {n} lanes", c.name)));
    let element = match t.class {
        0 if t.rows == 1 && t.columns == 1 => ConstantType::Scalar(ScalarType::Float),
        1 if t.rows == 1 => ConstantType::Vector(value(lanes(t.columns)?)),
        2 | 3 => ConstantType::Matrix {
            scalar: ScalarType::Float,
            rows: lanes(t.rows)?,
            columns: lanes(t.columns)?,
            major: if t.class == 2 {
                Majorness::Row
            } else {
                Majorness::Column
            },
            stride: 16,
        },
        _ => return Err(error(0, format!("CTAB {}: parameter class {}", c.name, t.class))),
    };
    let registers = lines(t);
    // a single matrix the compiler allocated fewer lines than it has (its last rows or columns are never read) is
    // described by the lines allocated, as a partial array is
    let partial_matrix = t.elements == 1 && (1..registers).contains(&c.count);
    let element = match element {
        ConstantType::Matrix {
            scalar,
            rows,
            columns,
            major,
            stride,
        } if partial_matrix => {
            let kept = lanes(c.count)?;
            let (rows, columns) = match major {
                Majorness::Row => (kept, columns),
                Majorness::Column => (rows, kept),
            };
            ConstantType::Matrix {
                scalar,
                rows,
                columns,
                major,
                stride,
            }
        }
        element => element,
    };
    let Some(count) = c
        .count
        .checked_div(registers)
        .filter(|&n| partial_matrix || (c.count.is_multiple_of(registers) && n <= t.elements))
    else {
        return Err(error(
            0,
            format!(
                "CTAB {}: an allocation of {} registers is not a whole number of the type's elements ({} registers each, at most {})",
                c.name, c.count, registers, t.elements
            ),
        ));
    };
    let ty = if t.elements > 1 {
        ConstantType::Array {
            element: Box::new(element),
            count: count.into(),
            stride: u32::from(registers).saturating_mul(16),
        }
    } else {
        element
    };
    with_default(c, ty, u32::from(c.register).saturating_mul(16))
}

/// An integer (`register_set` 1, a `vec4<i32>` register each) or boolean (0, four bytes each) constant's layout: its
/// registers, as scalars or vectors of the declared columns.
fn integer_field(c: &Constant) -> Result<Field> {
    let t = &c.ty;
    let (scalar, size) = match (c.register_set, t.kind) {
        (0, 1) => (ScalarType::Bool, 4),
        (1, 2) => (ScalarType::Sint, 16),
        _ => {
            return Err(error(
                0,
                format!("CTAB {}: register set {} of type {:?}", c.name, c.register_set, t),
            ));
        }
    };
    if !t.members.is_empty() || t.rows != 1 || !(1..=4).contains(&t.columns) || t.class > 1 {
        return Err(error(
            0,
            format!("CTAB {}: a structured or matrix type {:?}", c.name, t),
        ));
    }
    let columns = u8::try_from(t.columns).unwrap_or(4);
    let element = if columns == 1 || scalar == ScalarType::Bool {
        ConstantType::Scalar(scalar)
    } else {
        ConstantType::Vector(Value {
            scalar,
            width: columns,
        })
    };
    let ty = if c.count > 1 {
        ConstantType::Array {
            element: Box::new(element),
            count: c.count.into(),
            stride: size,
        }
    } else {
        element
    };
    with_default(c, ty, u32::from(c.register).saturating_mul(size))
}

fn with_default(c: &Constant, ty: ConstantType, offset: u32) -> Result<Field> {
    let default_size = ty
        .size()
        .ok_or_else(|| error(0, "CTAB normalized type size overflow"))? as usize;
    let default = c
        .default
        .as_ref()
        .map(|bytes| {
            bytes.get(..default_size).map(<[u8]>::to_vec).ok_or_else(|| {
                error(
                    0,
                    format!(
                        "CTAB {}: a default of {} bytes, where its type takes {default_size}",
                        c.name,
                        bytes.len()
                    ),
                )
            })
        })
        .transpose()?;
    Ok(Field {
        name: c.name.clone(),
        offset,
        ty,
        default,
    })
}

fn resource(
    interface: &mut Interface,
    kind: ResourceKind,
    name: String,
    binding: Option<(D3dClass, u32)>,
) -> Result<ResourceId> {
    let id = ResourceId(
        u32::try_from(interface.resources.len()).map_err(|_| error(0, "more than 4294967295 resources"))?,
    );
    interface.resources.push(Resource {
        id,
        kind,
        count: 1,
        provenance: ResourceProvenance {
            name,
            d3d: binding.map(|(class, register)| D3dBinding {
                class,
                register,
                space: 0,
            }),
            counter_of: None,
            layout: None,
        },
    });
    Ok(id)
}

/// A constant bank of `rows` registers of `stride` bytes, named `name`, at `cb{register}`.
fn bank(
    s: &mut Signature,
    name: &str,
    register: u32,
    rows: u32,
    stride: u32,
    fields: Vec<Field>,
) -> Result<Bank> {
    let layout = BufferLayout {
        size: rows.saturating_mul(stride).next_multiple_of(16),
        constants: fields,
    };
    let id = resource(
        &mut s.interface,
        ResourceKind::ConstantBuffer { layout },
        name.into(),
        Some((D3dClass::ConstantBuffer, register)),
    )?;
    Ok(Bank { resource: id, rows })
}

/// The interface the declarations, the implied registers and the constant table state, the CTAB refusals included.
/// `texture_dims` gives the dimension of pixel 1.x texture stages the host has stated; a stage it does not name gets a
/// texture of each dimension and a `StateParam::TextureDim` to pick by.
pub fn signature(source: &Shader, texture_dims: &BTreeMap<u32, TextureDim>) -> Result<Signature> {
    let mut s = Signature {
        interface: Interface {
            stage: source.stage,
            inputs: Vec::new(),
            outputs: Vec::new(),
            resources: Vec::new(),
            facts: StageFacts::None,
        },
        inputs: BTreeMap::new(),
        bank: None,
        integers: None,
        booleans: None,
        samplers: BTreeMap::new(),
        notes: Vec::new(),
    };
    for i in &source.inputs {
        let register = match i.register {
            Register::Input(n) | Register::Texture(n) => n,
            other => return Err(error(0, format!("{} is not an input register", other.name()))),
        };
        let mut v = varying(semantic(i.usage, 0)?, i.index.into(), i.mask, register, None);
        v.interp = (source.stage == Stage::Pixel).then_some(Interpolation {
            mode: InterpMode::Perspective,
            sampling: if i.centroid {
                InterpSampling::Centroid
            } else {
                InterpSampling::Center
            },
        });
        if i.partial_precision {
            s.notes.push(PARTIAL_PRECISION.into());
        }
        let slot =
            u32::try_from(s.interface.inputs.len()).map_err(|_| error(0, "more than 4294967295 inputs"))?;
        s.inputs.insert(i.register, (slot, i.mask));
        s.interface.inputs.push(v);
    }
    // the rows of each bank the code reads; a relative read can reach the whole bank
    let (mut rows, mut integers, mut booleans) = (0, 0, 0);
    let float_bank = if source.stage == Stage::Vertex {
        VERTEX_CONSTANTS
    } else {
        PIXEL_CONSTANTS
    };
    let reads = source
        .instructions
        .iter()
        .flat_map(|i| i.sources.iter().chain(i.predicate.iter()));
    for src in reads {
        match src.register {
            Register::Constant(_) if src.relative.is_some() => rows = rows.max(float_bank),
            Register::Constant(n) if !source.definitions.contains_key(&n) => {
                rows = rows.max(n.saturating_add(1))
            }
            Register::Integer(n) if !source.integers.contains_key(&n) => {
                integers = integers.max(n.saturating_add(1))
            }
            Register::Boolean(n) if !source.booleans.contains_key(&n) => {
                booleans = booleans.max(n.saturating_add(1))
            }
            _ => {}
        }
    }
    // a matrix operation reads the registers after its second source's
    for i in &source.instructions {
        if let (Some(extra), Some(Register::Constant(n))) = (
            i.matrix_rows().map(|r| r.saturating_sub(1)),
            i.sources.get(1).map(|s| s.register),
        ) {
            let last = n.saturating_add(extra);
            if i.sources.get(1).is_some_and(|s| s.relative.is_some()) {
                rows = rows.max(float_bank);
            } else if !source.definitions.contains_key(&last) {
                rows = rows.max(last.saturating_add(1));
            }
        }
    }
    let (mut fields, mut integer_fields, mut boolean_fields) = (Vec::new(), Vec::new(), Vec::new());
    if let Some(ctab) = &source.constants {
        for c in &ctab.constants {
            let end = u32::from(c.register).saturating_add(u32::from(c.count));
            match c.register_set {
                0 => {
                    booleans = booleans.max(end);
                    boolean_fields.push(integer_field(c)?);
                }
                1 => {
                    integers = integers.max(end);
                    integer_fields.push(integer_field(c)?);
                }
                2 => {
                    rows = rows.max(end);
                    fields.push(field(c)?);
                    let t = &c.ty;
                    if u32::from(c.count) < u32::from(t.elements).saturating_mul(u32::from(lines(t))) {
                        s.notes.push(format!(
                            "CTAB {} retains its declared type; its optimized allocation binds {} registers",
                            c.name, c.count
                        ));
                    }
                }
                3 => {
                    if c.count != 1 || c.ty.class != 4 || !matches!(c.ty.kind, 10..=14) {
                        return Err(error(
                            0,
                            format!("CTAB {}: sampler array or type {:?}", c.name, c.ty),
                        ));
                    }
                    // pixel 1.x declares no samplers, so its table is all there is to check against
                    if !source.pixel_1x() {
                        let expected = match c.ty.kind {
                            12 => TextureDim::D2,
                            13 => TextureDim::D3,
                            14 => TextureDim::Cube,
                            _ => {
                                return Err(error(0, format!("CTAB {}: sampler type {}", c.name, c.ty.kind)));
                            }
                        };
                        if source.samplers.get(&u32::from(c.register)) != Some(&expected) {
                            return Err(error(0, format!("CTAB {} sampler disagrees with DCL", c.name)));
                        }
                    }
                }
                _ => {
                    return Err(error(
                        0,
                        format!("CTAB {}: register set {} not lifted", c.name, c.register_set),
                    ));
                }
            }
        }
    }
    if rows > 0 {
        s.bank = Some(bank(&mut s, "float_constants", 0, rows, 16, fields)?);
    }
    if integers > 0 {
        s.integers = Some(bank(&mut s, "int_constants", 1, integers, 16, integer_fields)?);
    }
    if booleans > 0 {
        s.booleans = Some(bank(&mut s, "bool_constants", 2, booleans, 4, boolean_fields)?);
    }
    for (&slot, &declared) in &source.samplers {
        if declared == TextureDim::D1 {
            return Err(error(0, "1D sampler"));
        }
        let name = source
            .constants
            .as_ref()
            .and_then(|c| {
                c.constants
                    .iter()
                    .find(|c| c.register_set == 3 && u32::from(c.register) == slot)
            })
            .map_or_else(|| format!("s{slot}"), |c| c.name.clone());
        let dims = if !source.pixel_1x() {
            vec![declared]
        } else if let Some(&dim) = texture_dims.get(&slot) {
            vec![dim]
        } else {
            vec![TextureDim::D2, TextureDim::Cube, TextureDim::D3]
        };
        let mut textures = Vec::new();
        for &dim in &dims {
            let kind = ResourceKind::Texture {
                dim,
                arrayed: false,
                multisampled: false,
                sample: SampleType::Float,
            };
            let texture_name = if dims.len() > 1 {
                format!("{name}_{dim:?}")
            } else {
                name.clone()
            };
            let texture = resource(
                &mut s.interface,
                kind,
                texture_name,
                Some((D3dClass::ShaderResource, slot)),
            )?;
            textures.push((dim, texture));
        }
        let sampler = resource(
            &mut s.interface,
            ResourceKind::Sampler { comparison: false },
            name,
            Some((D3dClass::Sampler, slot)),
        )?;
        let dim_state = if textures.len() > 1 {
            Some(s.state(StateParam::TextureDim(slot))?)
        } else {
            None
        };
        s.samplers.insert(
            slot,
            Sampler {
                sampler,
                textures,
                dim_state,
            },
        );
    }
    Ok(s)
}

/// The outputs of a program whose registers were written in the lanes `written` gives: a vertex program's declared
/// or implied outputs in order, or a pixel program's written colour targets in register order (`r0` in pixel 1.x)
/// and its depth. Each is the register it is read from with its varying, marked used; a vertex program with no
/// `POSITION0`, and a lane of an output that is never written, are refused (`at` is where the refusal points).
pub fn outputs(
    source: &Shader,
    written: &BTreeMap<Register, u8>,
    at: usize,
) -> Result<Vec<(Register, Varying)>> {
    let outgoing: Vec<_> = if source.stage == Stage::Vertex {
        source
            .outputs
            .iter()
            .map(|o| {
                let system =
                    (o.usage == usage::POSITION && o.index == 0).then_some(SystemValue::ClipPosition);
                let register = match o.register {
                    Register::Output(n) | Register::Color(n) | Register::TexCoord(n) => n,
                    _ => 0,
                };
                Ok((
                    o.register,
                    varying(semantic(o.usage, at)?, o.index.into(), o.mask, register, system),
                ))
            })
            .collect::<Result<_>>()?
    } else if source.pixel_1x() {
        let mut out = vec![(
            Register::Temporary(0),
            varying("SV_Target", 0, 15, 0, Some(SystemValue::Target(0))),
        )];
        if written.contains_key(&Register::Depth) {
            out.push((
                Register::Depth,
                varying("SV_Depth", 0, 1, 0, Some(SystemValue::Depth)),
            ));
        }
        out
    } else {
        written
            .iter()
            .filter_map(|(&r, &mask)| match r {
                Register::Target(n) => {
                    Some((r, varying("SV_Target", n, mask, n, Some(SystemValue::Target(n)))))
                }
                Register::Depth => Some((r, varying("SV_Depth", 0, 1, 0, Some(SystemValue::Depth)))),
                _ => None,
            })
            .collect()
    };
    if source.stage == Stage::Vertex
        && !outgoing
            .iter()
            .any(|(_, v)| v.system == Some(SystemValue::ClipPosition))
    {
        let what = if source.fixed_outputs() {
            "vertex program never writes oPos"
        } else {
            "vertex program has no declared POSITION0 clip output"
        };
        return Err(error(0, what));
    }
    let mut out = Vec::new();
    for (r, mut v) in outgoing {
        let mask = v
            .provenance
            .d3d
            .as_ref()
            .ok_or_else(|| error(at, "output signature missing"))?
            .mask;
        let &done = written
            .get(&r)
            .ok_or_else(|| error(at, format!("uninitialized output {}", r.name())))?;
        if mask & !done != 0 {
            return Err(error(
                at,
                format!("{}: uninitialized output lanes {:#x}", r.name(), mask & !done),
            ));
        }
        v.used = true;
        out.push((r, v));
    }
    Ok(out)
}

/// Mark lanes `required` of input register `register` read: refused where the program does not declare them.
pub fn read_input(s: &mut Signature, register: Register, required: u8, at: usize) -> Result<u32> {
    let &(slot, mask) = s
        .inputs
        .get(&register)
        .ok_or_else(|| error(at, format!("undeclared input {}", register.name())))?;
    if required & !mask != 0 {
        return Err(error(
            at,
            format!("{}: read of undeclared lanes {required:#x}", register.name()),
        ));
    }
    let input = s.interface.inputs.get_mut(slot as usize).ok_or_else(|| {
        error(
            at,
            format!(
                "input {} is input {slot} of a list that is shorter",
                register.name()
            ),
        )
    })?;
    input.used = true;
    if let Some(d) = &mut input.provenance.d3d {
        d.read_write_mask |= required;
    }
    Ok(slot)
}
