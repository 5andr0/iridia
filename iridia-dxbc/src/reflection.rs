//! Shader Model 4 and 5 RDEF and signature records ([`dxbc_reader`]'s, as the file stores them) mapped into the
//! interface vocabulary; no material or binding policy. D3D's enum numbers stop here: a value the mapping does not know is an error naming
//! it.
use crate::{Result, error, read};
use dxbc_reader::decode::Declared;
use dxbc_reader::{ProgramType, RdefBinding, RdefType, SignatureElement};
use iridia::d3d;
use iridia::{
    Access, BufferKind, BufferLayout, Constant, ConstantType, D3dBinding, D3dClass, Interface, Majorness,
    Member, Resource, ResourceId, ResourceKind, ResourceProvenance, SampleType, ScalarType, Stage,
    StorageTarget, TextureDim, Value, Varying,
};
use std::collections::{BTreeMap, BTreeSet};

/// The pipeline stage of a program type.
pub(crate) fn stage_of(kind: ProgramType) -> Stage {
    match kind {
        ProgramType::Pixel => Stage::Pixel,
        ProgramType::Vertex => Stage::Vertex,
        ProgramType::Geometry => Stage::Geometry,
        ProgramType::Hull => Stage::Hull,
        ProgramType::Domain => Stage::Domain,
        ProgramType::Compute => Stage::Compute,
    }
}

/// A signature element in the vocabulary, or the reason it is not mapped.
pub fn varying(e: &SignatureElement, stage: Stage, output: bool) -> Result<Varying> {
    let element = d3d::Element {
        name: &e.name,
        index: e.index,
        system_value: e.system_value,
        component_type: e.component_type,
        register: e.register,
        mask: e.mask,
        read_write_mask: e.read_write_mask,
    };
    d3d::varying(stage, output, &element).map_err(|detail| error(e.at, detail.to_string()))
}

/// The signature in whichever of these parts the container has, mapped.
fn signature_of(
    directory: &[dxcontainer::Chunk<'_>],
    tags: &[&[u8; 4]],
    stage: Stage,
    output: bool,
) -> Result<Vec<Varying>> {
    let tag = tags
        .iter()
        .find(|t| directory.iter().any(|c| &&c.kind == *t))
        .or(tags.first())
        .ok_or_else(|| error(0, "a signature with no part name"))?;
    dxbc_reader::signature(dxbc_reader::chunk(directory, tag).map_err(read)?, tag)
        .map_err(read)?
        .iter()
        .map(|e| varying(e, stage, output))
        .collect()
}

/// Read a container's ISGN/OSGN signatures alone, for a container whose RDEF was stripped (as some engines ship
/// them): no resources, so the caller states the constant layout from its own records. Interpolation is the
/// program's to state: [`crate::lift`] reads it from the declarations.
pub fn signatures(bytes: &[u8]) -> Result<Interface> {
    let directory = dxbc_reader::parts(bytes).map_err(read)?;
    let stage = stage_of(dxbc_reader::program_type(&directory).map_err(read)?);
    Ok(Interface {
        stage,
        inputs: signature_of(&directory, &[b"ISGN", b"ISG1"], stage, false)?,
        outputs: signature_of(&directory, &[b"OSGN", b"OSG1", b"OSG5"], stage, true)?,
        resources: Vec::new(),
        facts: Default::default(),
    })
}

/// `D3D_SHADER_INPUT_TYPE`'s name for a message.
fn input_type_name(kind: u32) -> &'static str {
    match kind {
        0 => "D3D_SIT_CBUFFER",
        1 => "D3D_SIT_TBUFFER",
        2 => "D3D_SIT_TEXTURE",
        3 => "D3D_SIT_SAMPLER",
        4 => "D3D_SIT_UAV_RWTYPED",
        5 => "D3D_SIT_STRUCTURED",
        6 => "D3D_SIT_UAV_RWSTRUCTURED",
        7 => "D3D_SIT_BYTEADDRESS",
        8 => "D3D_SIT_UAV_RWBYTEADDRESS",
        9 => "D3D_SIT_UAV_APPEND_STRUCTURED",
        10 => "D3D_SIT_UAV_CONSUME_STRUCTURED",
        11 => "D3D_SIT_UAV_RWSTRUCTURED_WITH_COUNTER",
        12 => "D3D_SIT_RTACCELERATIONSTRUCTURE",
        13 => "D3D_SIT_UAV_FEEDBACKTEXTURE",
        _ => "an unknown resource type",
    }
}

/// What one RDEF binding record means.
struct Binding<'a>(&'a RdefBinding);
impl Binding<'_> {
    /// A texture's or typed view's value type, from its return type (normalized formats return floats).
    fn scalar(&self) -> std::result::Result<ScalarType, String> {
        match self.0.returns {
            1 | 2 | 5 => Ok(ScalarType::Float),
            3 => Ok(ScalarType::Sint),
            4 => Ok(ScalarType::Uint),
            r => Err(format!("{}: resource return type {r} is not mapped", self.0.name)),
        }
    }
    fn sample(&self) -> std::result::Result<SampleType, String> {
        Ok(match self.scalar()? {
            ScalarType::Sint => SampleType::Sint,
            ScalarType::Uint => SampleType::Uint,
            _ => SampleType::Float,
        })
    }
    /// A texture dimension's shape: dimensionality, array, multisampled.
    fn shape(&self) -> std::result::Result<(TextureDim, bool, bool), String> {
        Ok(match self.0.dimension {
            2 => (TextureDim::D1, false, false),
            3 => (TextureDim::D1, true, false),
            4 => (TextureDim::D2, false, false),
            5 => (TextureDim::D2, true, false),
            6 => (TextureDim::D2, false, true),
            7 => (TextureDim::D2, true, true),
            8 => (TextureDim::D3, false, false),
            9 => (TextureDim::Cube, false, false),
            10 => (TextureDim::Cube, true, false),
            d => return Err(format!("{}: resource dimension {d} is not mapped", self.0.name)),
        })
    }
    /// A typed buffer's element: its scalar type and, from the flags' component bits, its width.
    fn element(&self) -> std::result::Result<Value, String> {
        Ok(Value {
            scalar: self.scalar()?,
            width: u8::try_from((self.0.flags >> 2) & 3)
                .unwrap_or(3)
                .saturating_add(1),
        })
    }
    fn stride(&self) -> std::result::Result<BufferKind, String> {
        match self.0.samples {
            0 => Err(format!("{}: a structured buffer without a stride", self.0.name)),
            stride => Ok(BufferKind::Structured { stride }),
        }
    }
    /// What the record stands for, leaving constant buffers (whose layout is a table apart) to the caller.
    fn kind(&self) -> std::result::Result<(ResourceKind, D3dClass), String> {
        let storage = |target| ResourceKind::Storage {
            target,
            access: Access::ReadWrite,
        };
        Ok(match self.0.kind {
            // a tbuffer is read as a typed buffer of four-lane rows (float where the record states no return type)
            1 => (
                ResourceKind::Buffer(BufferKind::Typed(self.element().unwrap_or(Value {
                    scalar: ScalarType::Float,
                    width: 4,
                }))),
                D3dClass::ShaderResource,
            ),
            2 if self.0.dimension == 1 => (
                ResourceKind::Buffer(BufferKind::Typed(self.element()?)),
                D3dClass::ShaderResource,
            ),
            2 => {
                let (dim, arrayed, multisampled) = self.shape()?;
                (
                    ResourceKind::Texture {
                        dim,
                        arrayed,
                        multisampled,
                        sample: self.sample()?,
                    },
                    D3dClass::ShaderResource,
                )
            }
            3 => (
                ResourceKind::Sampler {
                    comparison: self.0.flags & 2 != 0,
                },
                D3dClass::Sampler,
            ),
            4 if self.0.dimension == 1 => (
                storage(StorageTarget::Buffer(BufferKind::Typed(self.element()?))),
                D3dClass::UnorderedAccess,
            ),
            4 => {
                let (dim, arrayed, multisampled) = self.shape()?;
                if multisampled {
                    return Err(format!("{}: a multisampled unordered-access view", self.0.name));
                }
                (
                    storage(StorageTarget::Image {
                        dim,
                        arrayed,
                        sample: self.sample()?,
                    }),
                    D3dClass::UnorderedAccess,
                )
            }
            5 => (ResourceKind::Buffer(self.stride()?), D3dClass::ShaderResource),
            6 => (
                storage(StorageTarget::Buffer(self.stride()?)),
                D3dClass::UnorderedAccess,
            ),
            7 => (ResourceKind::Buffer(BufferKind::Raw), D3dClass::ShaderResource),
            8 => (
                storage(StorageTarget::Buffer(BufferKind::Raw)),
                D3dClass::UnorderedAccess,
            ),
            kind => {
                return Err(format!(
                    "{}: {} ({kind}) is not mapped",
                    self.0.name,
                    input_type_name(kind)
                ));
            }
        })
    }
}

/// A struct type with `by` taken from each member's offset; `None` for another type or a member before `by`.
fn rebased(ty: &ConstantType, by: u32) -> Option<ConstantType> {
    let ConstantType::Struct { members } = ty else {
        return None;
    };
    let members = members
        .iter()
        .map(|m| {
            Some(Member {
                name: m.name.clone(),
                offset: m.offset.checked_sub(by)?,
                ty: m.ty.clone(),
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ConstantType::Struct { members })
}

/// A stored type record in the vocabulary, under the buffer's packing.
fn constant_type(ty: &RdefType, packing: d3d::Packing) -> Result<ConstantType> {
    let mut members = Vec::new();
    for m in &ty.members {
        members.push(Member {
            name: m.name.clone(),
            offset: m.offset,
            ty: constant_type(&m.ty, packing)?,
        });
    }
    let (class, scalar) = (ty.class, u32::from(ty.scalar));
    let (rows, columns) = (u32::from(ty.rows), u32::from(ty.columns));
    let fail = |detail: String| error(ty.word, detail);
    let element = match class {
        0..=3 => {
            let scalar = d3d::scalar_type(scalar).map_err(|e| fail(e.to_string()))?;
            match class {
                0 if (rows, columns) == (1, 1) => ConstantType::Scalar(scalar),
                1 if rows == 1 => d3d::vector(scalar, columns).map_err(|e| fail(e.to_string()))?,
                2 => packing
                    .matrix(scalar, rows, columns, Majorness::Row)
                    .map_err(|e| fail(e.to_string()))?,
                3 => packing
                    .matrix(scalar, rows, columns, Majorness::Column)
                    .map_err(|e| fail(e.to_string()))?,
                _ => return Err(fail(format!("a {rows} by {columns} variable of class {class}"))),
            }
        }
        5 => ConstantType::Struct { members },
        4 => {
            return Err(fail(
                "a constant-buffer variable of an object type (D3D_SVC_OBJECT)".into(),
            ));
        }
        c => return Err(fail(format!("variable class {c} is not mapped"))),
    };
    match ty.elements {
        0 => Ok(element),
        n => packing
            .array(element, u32::from(n))
            .map_err(|e| fail(e.to_string())),
    }
}

/// The stage an RDEF's target (its version token) names, and whether it is Shader Model 5.1's, where this reader reads
/// it: a program of any stage, of Shader Model 4.0, 4.1, 5.0 or 5.1.
fn rdef_stage(target: u32) -> Option<(Stage, bool)> {
    let stage = match target >> 16 {
        0xffff => Stage::Pixel,
        0xfffe => Stage::Vertex,
        0x4753 => Stage::Geometry,
        0x4853 => Stage::Hull,
        0x4453 => Stage::Domain,
        0x4353 => Stage::Compute,
        _ => return None,
    };
    match target & 0xffff {
        0x0400 | 0x0410 | 0x0500 => Some((stage, false)),
        0x0501 => Some((stage, true)),
        _ => None,
    }
}

/// A container's reflected resources where it has an RDEF this reader reads (a Shader Model 4.0 to 5.1 program's, of
/// any stage): `None` where it has no RDEF chunk or one of another target (the lift then names its resources from
/// their declarations, as without reflection). An RDEF of a target it reads that doesn't parse is an error, never
/// taken as absent.
pub fn resource_reflection(bytes: &[u8]) -> Result<Option<Vec<Resource>>> {
    let Some((_, sm51)) = dxbc_reader::rdef_target(bytes)
        .map_err(read)?
        .and_then(rdef_stage)
    else {
        return Ok(None);
    };
    let rdef = dxbc_reader::rdef(bytes)
        .map_err(read)?
        .ok_or_else(|| error(0, "missing reflection chunk RDEF"))?;
    resources(&rdef, sm51, || declared_rows(bytes)).map(Some)
}

/// Read a pixel, vertex or compute program's RDEF resource and constant-buffer metadata and its ISGN/OSGN signatures
/// (the stage is the RDEF's own target, so reflection reads a container without its code). A hull, domain or
/// geometry program is refused: its interface is more than its signatures ([`resource_reflection`] reads its
/// resources).
pub fn reflection(bytes: &[u8]) -> Result<Interface> {
    let directory = dxbc_reader::parts(bytes).map_err(read)?;
    let rdef = dxbc_reader::rdef(bytes)
        .map_err(read)?
        .ok_or_else(|| error(0, "missing reflection chunk RDEF"))?;
    let (program_stage, sm51) = rdef_stage(rdef.target)
        .filter(|(stage, _)| matches!(stage, Stage::Pixel | Stage::Vertex | Stage::Compute))
        .ok_or_else(|| {
            error(
                4,
                "expected the RDEF of a Shader Model 4.0 to 5.1 pixel, vertex or compute program",
            )
        })?;
    let resources = resources(&rdef, sm51, || declared_rows(bytes))?;
    // a compute program has no varyings: its invocation ids are system values its declarations state, and its
    // signature parts (where it has them) are empty
    let (inputs, outputs) = if program_stage == Stage::Compute {
        (Vec::new(), Vec::new())
    } else {
        (
            signature_of(&directory, &[b"ISGN", b"ISG1"], program_stage, false)?,
            signature_of(&directory, &[b"OSGN", b"OSG1", b"OSG5"], program_stage, true)?,
        )
    };
    Ok(Interface {
        stage: program_stage,
        inputs,
        outputs,
        resources,
        facts: Default::default(),
    })
}

/// The rows each constant-buffer declaration of a container's code states, by its first register and its space; none
/// where the container has no code.
fn declared_rows(bytes: &[u8]) -> Result<BTreeMap<(u32, u32), u32>> {
    let directory = dxbc_reader::parts(bytes).map_err(read)?;
    if !directory.iter().any(|c| matches!(&c.kind, b"SHDR" | b"SHEX")) {
        return Ok(BTreeMap::new());
    }
    let words = dxbc_reader::code(bytes).map_err(read)?;
    let decoded = dxbc_reader::decode::decode(&words).map_err(read)?;
    Ok(decoded
        .declarations
        .iter()
        .filter_map(|d| match &d.kind {
            Declared::ConstantBuffer { rows, .. } => d.kind.registers().map(|r| ((r.lower, r.space), *rows)),
            _ => None,
        })
        .collect())
}

/// A constant-buffer or tbuffer binding's name and buffer type (0 or 1, as the buffers state it); `None` for another
/// binding.
fn buffer(r: &RdefBinding) -> Option<(&str, u32)> {
    (r.kind <= 1).then(|| (r.name.as_str(), u32::from(r.kind == 1)))
}

/// Which layout each constant-buffer and tbuffer binding takes (an index into `layouts`; `None` for another
/// binding): the one of its name and buffer type. Two buffers can share a name (fxc keeps a `cbuffer C` declared twice
/// in one namespace as two buffers named alike) and the RDEF does not say which binding is which: fxc lists the
/// bindings by register and the buffers in the order declared. Such bindings are told apart by the rows their
/// declarations in the code state (`declared`, read only then), which the layout must hold; where that leaves more
/// than one way and the layouts differ, the program is refused.
fn buffer_layouts(
    bindings: &[RdefBinding],
    layouts: &[(String, BufferLayout, bool, u32)],
    declared: impl FnOnce() -> Result<BTreeMap<(u32, u32), u32>>,
) -> Result<Vec<Option<usize>>> {
    let mut taken = vec![None; bindings.len()];
    let mut declared = Some(declared);
    let mut rows = BTreeMap::new();
    let mut done = BTreeSet::new();
    for key in bindings.iter().filter_map(buffer) {
        if !done.insert(key) {
            continue;
        }
        let what = if key.1 == 1 { "tbuffer" } else { "constant buffer" };
        let mut open: Vec<usize> = (0..bindings.len())
            .filter(|&b| bindings.get(b).and_then(buffer) == Some(key))
            .collect();
        let mut free: Vec<usize> = (0..layouts.len())
            .filter(|&l| {
                layouts
                    .get(l)
                    .is_some_and(|(name, _, _, table)| (name.as_str(), *table) == key)
            })
            .collect();
        let word = |b: usize| bindings.get(b).map_or(0, |r| r.word);
        if open.len() > free.len() {
            let b = open.get(free.len()).copied().unwrap_or(0);
            return Err(error(word(b), format!("{}: a {what} without its layout", key.0)));
        }
        if open.len() > 1 || free.len() > 1 {
            if let Some(read) = declared.take() {
                rows = read()?;
            }
            // whether layout `l` holds the rows binding `b`'s declaration states (any, where it has none)
            let holds = |b: usize, l: usize| {
                let need = bindings
                    .get(b)
                    .and_then(|r| rows.get(&(r.register, r.space)))
                    .copied()
                    .unwrap_or(0);
                layouts
                    .get(l)
                    .is_some_and(|(_, layout, _, _)| layout.size.div_ceil(16) >= need)
            };
            // a binding only one free layout can hold takes it, until none is left so
            while let Some((i, j)) = open.iter().enumerate().find_map(|(i, &b)| {
                let mut fits = free.iter().enumerate().filter(|&(_, &l)| holds(b, l));
                match (fits.next(), fits.next()) {
                    (Some((j, _)), None) => Some((i, j)),
                    _ => None,
                }
            }) {
                let (b, l) = (open.remove(i), free.remove(j));
                if let Some(t) = taken.get_mut(b) {
                    *t = Some(l);
                }
            }
            // the rest take the layouts in order only where those are alike and hold every one of them
            let alike = free.windows(2).all(
                |w| matches!(w, [x, y] if layouts.get(*x).map(|l| &l.1) == layouts.get(*y).map(|l| &l.1)),
            );
            if let Some(&b) = open.first()
                && !(alike && open.iter().all(|&b| free.iter().all(|&l| holds(b, l))))
            {
                return Err(error(
                    word(b),
                    format!(
                        "{} {what}s named {}: their declarations do not tell which binding takes which layout",
                        open.len(),
                        key.0
                    ),
                ));
            }
        }
        for (b, l) in open.into_iter().zip(free) {
            if let Some(t) = taken.get_mut(b) {
                *t = Some(l);
            }
        }
    }
    Ok(taken)
}

/// An RDEF's resources: each binding record with the layout of its constant buffer or structured element (`declared`
/// the rows the code's constant-buffer declarations state, read where two buffers share a name).
fn resources(
    rdef: &dxbc_reader::Rdef,
    sm51: bool,
    declared: impl FnOnce() -> Result<BTreeMap<(u32, u32), u32>>,
) -> Result<Vec<Resource>> {
    for binding in &rdef.bindings {
        if (binding.count == 0 && !sm51) || binding.flags & !0x1f != 0 {
            return Err(error(
                binding.word,
                format!(
                    "{}: bind count {} and flags {:#x} are not mapped",
                    binding.name, binding.count, binding.flags
                ),
            ));
        }
    }
    // RDEF buffer types: cbuffer (0), tbuffer (1), structured element (3).
    // The last uses storage packing; the other two use constant registers.
    let mut layouts: Vec<(String, BufferLayout, bool, u32)> = Vec::new();
    for buffer in &rdef.buffers {
        let table = match buffer.kind {
            kind @ (0 | 1 | 3) => kind,
            kind => {
                return Err(error(
                    buffer.word,
                    format!(
                        "{}: buffer type {kind} ({}) is not mapped",
                        buffer.name,
                        match kind {
                            2 => "interface pointers",
                            _ => "unknown",
                        }
                    ),
                ));
            }
        };
        let packing = if table == 3 {
            d3d::Packing::Structured
        } else {
            d3d::Packing::Constant
        };
        let mut constants = Vec::new();
        for variable in &buffer.variables {
            let packed = |ty: &ConstantType| {
                if table == 3 {
                    ty.size()
                        .and_then(|n| n.checked_next_multiple_of(d3d::Packing::Structured.alignment(ty)))
                } else {
                    ty.size()
                }
            };
            let stated = constant_type(&variable.ty, packing)?;
            let ty = if packed(&stated) == Some(variable.size) {
                stated
            } else {
                // a struct whose members' offsets include the variable's own offset in the buffer, as a hull
                // program's record of a struct it does not use can state them (Sekiro's SAT_MtdParam: its members
                // from 384 of a 384-byte struct at 384, where the material's pixel programs have them from 0): taken
                // with those offsets made the struct's own, where that is what packs to the stated size
                match rebased(&stated, variable.offset) {
                    Some(r) if packed(&r) == Some(variable.size) => r,
                    _ => {
                        return Err(error(
                            variable.word,
                            format!(
                                "variable {} is {} bytes, its type packs to {:?}",
                                variable.name,
                                variable.size,
                                packed(&stated)
                            ),
                        ));
                    }
                }
            };
            constants.push(Constant {
                name: variable.name.clone(),
                offset: variable.offset,
                ty,
                default: variable.default.clone(),
            });
        }
        layouts.push((
            buffer.name.clone(),
            BufferLayout {
                size: buffer.size,
                constants,
            },
            false,
            table,
        ));
    }
    let taken = buffer_layouts(&rdef.bindings, &layouts, declared)?;
    let mut resources = Vec::new();
    for (i, record) in rdef.bindings.iter().enumerate() {
        let binding = Binding(record);
        let fail = |detail: String| error(record.word, detail);
        let mut table_layout = None;
        let (kind, class) = if record.kind == 0 || record.kind == 1 {
            // a constant buffer takes a layout of type 0, a tbuffer one of type 1
            let tbuffer = record.kind == 1;
            let (_, layout, claimed, _) = taken
                .get(i)
                .copied()
                .flatten()
                .and_then(|l| layouts.get_mut(l))
                .ok_or_else(|| fail(format!("{}: a buffer without its layout", record.name)))?;
            *claimed = true;
            if tbuffer {
                table_layout = Some(layout.clone());
                binding.kind().map_err(fail)?
            } else {
                (
                    ResourceKind::ConstantBuffer {
                        layout: layout.clone(),
                    },
                    D3dClass::ConstantBuffer,
                )
            }
        } else {
            let kind = binding.kind().map_err(fail)?;
            if matches!(record.kind, 5 | 6 | 9 | 10 | 11)
                && let Some((_, layout, claimed, _)) = layouts.iter_mut().find(|(name, _, _, table)| {
                    // fxc's SM5.1 resource-array binding names the range, but its structured
                    // element layout names the first descriptor (for example lights_out[0]).
                    *table == 3
                        && (*name == record.name
                            || (sm51
                                && record.count != 1
                                && name.strip_suffix("[0]") == Some(record.name.as_str())))
                })
            {
                if std::mem::replace(claimed, true) {
                    return Err(fail(format!(
                        "{}: two bindings of one structured layout",
                        record.name
                    )));
                }
                if layout.size != record.samples {
                    return Err(fail(format!(
                        "{}: element layout {} disagrees with stride {}",
                        record.name, layout.size, record.samples
                    )));
                }
                table_layout = Some(layout.clone());
            }
            kind
        };
        resources.push(Resource {
            id: ResourceId(u32::try_from(i).map_err(|_| error(0, "more than 4294967295 resources"))?),
            kind,
            count: if sm51 && record.count == 0 {
                Resource::UNBOUNDED
            } else {
                record.count
            },
            provenance: ResourceProvenance {
                name: record.name.clone(),
                d3d: Some(D3dBinding {
                    class,
                    register: record.register,
                    space: record.space,
                }),
                counter_of: None,
                layout: table_layout,
            },
        });
    }
    if let Some((name, ..)) = layouts.iter().find(|(_, _, claimed, _)| !claimed) {
        return Err(error(
            0,
            format!("constant buffer {name} has no resource binding"),
        ));
    }
    Ok(resources)
}

#[cfg(test)]
mod structured_tests {
    use iridia::{ConstantType, Majorness, ScalarType};
    const BYTES: &[u8] = include_bytes!("../tests/fixtures/structured_layout.dxbc");
    // SDK 10.0.26100.0 compiler dumps: entries stride 96, indices 4, lookup 12.
    // The fixture distinguishes cbuffer offsets (values 16, rows 64) from storage offsets (4, 16).
    #[test]
    fn structured_layout_survives_reflection_and_lifting() {
        check(&super::reflection(BYTES).expect("reflection"));
        check(&crate::lift(BYTES).expect("lift").interface);
    }

    fn check(interface: &iridia::Interface) {
        let layout = |name: &str| {
            interface
                .resources
                .iter()
                .find(|r| r.provenance.name == name)
                .expect("named resource")
                .provenance
                .layout
                .as_ref()
                .expect("element layout")
        };
        let entry = layout("entries");
        assert_eq!(entry.size, 96);
        let members = entry.constants[0].ty.members();
        assert_eq!(
            members
                .iter()
                .map(|m| (m.name.as_str(), m.offset))
                .collect::<Vec<_>>(),
            [
                ("scalar", 0),
                ("values", 4),
                ("rows", 16),
                ("columns", 40),
                ("nested", 64)
            ]
        );
        assert!(matches!(
            members[1].ty,
            ConstantType::Array {
                count: 3,
                stride: 4,
                ..
            }
        ));
        assert!(matches!(
            members[2].ty,
            ConstantType::Matrix {
                stride: 12,
                major: Majorness::Row,
                ..
            }
        ));
        assert!(matches!(
            members[3].ty,
            ConstantType::Matrix {
                stride: 8,
                major: Majorness::Column,
                ..
            }
        ));
        assert!(matches!(
            members[4].ty,
            ConstantType::Array {
                count: 2,
                stride: 16,
                ..
            }
        ));
        let nested = members[4].ty.members();
        assert_eq!(
            (nested[0].name.as_str(), nested[0].offset, nested[1].offset),
            ("direction", 0, 12)
        );
        assert_eq!(layout("indices").size, 4);
        assert_eq!(
            layout("indices").constants[0].ty,
            ConstantType::Scalar(ScalarType::Uint)
        );
        assert_eq!(layout("lookup").size, 12);
        assert_eq!(
            layout("lookup").constants[0].ty,
            ConstantType::Vector(iridia::Value {
                scalar: ScalarType::Uint,
                width: 3
            })
        );
    }
    #[test]
    fn structured_tail_padding_is_kept_in_the_element_stride() {
        // Both compiler dumps give 16 bytes although the last data member ends at byte 12.
        let bytes = include_bytes!("../tests/fixtures/structured_tail.dxbc");
        let reflected = super::reflection(bytes).expect("tail reflection");
        let layout = reflected
            .resources
            .iter()
            .find(|r| r.provenance.name == "entries")
            .expect("entries")
            .provenance
            .layout
            .as_ref()
            .expect("layout");
        assert_eq!(layout.size, 16);
        assert_eq!(layout.constants[0].ty.size(), Some(12));
        let members = layout.constants[0].ty.members();
        assert_eq!(members[0].ty, ConstantType::Scalar(ScalarType::Double));
        assert_eq!((members[0].offset, members[1].offset), (0, 8));
    }
}

#[cfg(test)]
mod shader_model_4_tests {
    use iridia::{Constant, ConstantType, Interface, ResourceKind, ScalarType, Value};

    // tests/fixtures/reflection_sm4.hlsl: ps_4_0, as declared (the first `cbuffer A::C` at b1), with that buffer at b9
    // (fxc then lists its binding after the other's), and with the two buffers alike in size and declaration
    const PLAIN: &[u8] = include_bytes!("../tests/fixtures/reflection_sm4.dxbc");
    const SWAPPED: &[u8] = include_bytes!("../tests/fixtures/reflection_sm4.swapped.dxbc");
    const ALIKE: &[u8] = include_bytes!("../tests/fixtures/reflection_sm4.alike.dxbc");

    /// The size and constants (name, offset) of the constant buffer at `register`.
    fn buffer(interface: &Interface, register: u32) -> (u32, Vec<Constant>) {
        let resource = interface
            .resources
            .iter()
            .find(|r| r.provenance.d3d.as_ref().is_some_and(|d| d.register == register))
            .unwrap_or_else(|| panic!("a resource at register {register}"));
        let ResourceKind::ConstantBuffer { layout } = &resource.kind else {
            panic!("a constant buffer at register {register}")
        };
        (layout.size, layout.constants.clone())
    }

    fn named(constants: &[Constant]) -> Vec<(&str, u32)> {
        constants.iter().map(|c| (c.name.as_str(), c.offset)).collect()
    }

    #[test]
    fn minimum_precision_constants_are_their_32_bit_types() {
        for interface in [
            super::reflection(PLAIN).unwrap(),
            crate::lift(PLAIN).unwrap().interface,
        ] {
            let (size, constants) = buffer(&interface, 0);
            assert_eq!(size, 96);
            let tys: Vec<&ConstantType> = constants.iter().map(|c| &c.ty).collect();
            assert_eq!(
                tys[..2],
                [
                    &ConstantType::Vector(Value {
                        scalar: ScalarType::Float,
                        width: 4
                    }),
                    &ConstantType::Vector(Value {
                        scalar: ScalarType::Sint,
                        width: 2
                    }),
                ]
            );
            // fxc's 52 bytes: three 16-byte registers and the last element's 4
            assert!(matches!(
                tys[2],
                ConstantType::Array { element, count: 4, stride: 16 }
                    if **element == ConstantType::Scalar(ScalarType::Uint)
            ));
            assert_eq!(tys[2].size(), Some(52));
        }
    }

    #[test]
    fn same_named_buffers_take_the_layouts_their_declarations_hold() {
        for (bytes, first) in [(PLAIN, 1), (SWAPPED, 9)] {
            for interface in [
                super::reflection(bytes).unwrap(),
                crate::lift(bytes).unwrap().interface,
            ] {
                let (size, constants) = buffer(&interface, first);
                assert_eq!((size, named(&constants)), (16, vec![("A::pad", 0), ("A::x", 4)]));
                let (size, constants) = buffer(&interface, 2);
                assert_eq!(
                    (size, named(&constants)),
                    (48, vec![("A::skip", 0), ("A::y", 32)])
                );
            }
        }
    }

    #[test]
    fn same_named_buffers_nothing_tells_apart_are_refused() {
        for e in [
            super::reflection(ALIKE).unwrap_err(),
            crate::lift(ALIKE).unwrap_err(),
        ] {
            assert!(
                e.to_string()
                    .contains("2 constant buffers named A::C: their declarations do not tell which binding takes which layout"),
                "{e}"
            );
        }
    }
}
