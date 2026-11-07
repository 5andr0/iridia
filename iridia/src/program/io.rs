//! How a program's interface is carried by its entry point: the one place that says which module argument or result
//! member a varying is, with its name, type and binding. The validator checks a module against it and a lifter builds
//! the module from it.

use naga::Binding;

use iridia_ir::target::{self, BuiltIn};

use crate::{Interface, ScalarType, Stage, StageFacts, Value, Varying};

pub use iridia_ir::target::{IoType, carried, varying_name};

/// One argument or result member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IoItem {
    /// `<NAME><index>` after the label, or `clip_distances`.
    pub name: String,
    /// The type of the member.
    pub ty: IoType,
    /// The naga binding it has: a location or a built-in.
    pub binding: Binding,
    /// Which varyings of the interface's list (inputs for an argument, outputs for a result) it carries: one, or every
    /// clip distance.
    pub varyings: Vec<usize>,
}

/// The entry point's arguments (one per input varying) and result members (one per output varying, the clip distances
/// being one together), in interface order. A varying nothing reads or writes (`used` false) that no binding exists
/// for is left out: the interface lists every element of the container's signature, the module carries what it can.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryIo {
    /// One per input varying that is used or has a binding.
    pub arguments: Vec<IoItem>,
    /// One per output varying (the clip distances together), in interface order.
    pub results: Vec<IoItem>,
}

/// naga's binding for a target's.
fn naga_binding(b: target::Binding) -> Binding {
    match b {
        target::Binding::BuiltIn(b) => Binding::BuiltIn(match b {
            BuiltIn::Position => naga::BuiltIn::Position { invariant: false },
            BuiltIn::FrontFacing => naga::BuiltIn::FrontFacing,
            BuiltIn::VertexIndex => naga::BuiltIn::VertexIndex,
            BuiltIn::InstanceIndex => naga::BuiltIn::InstanceIndex,
            BuiltIn::PrimitiveIndex => naga::BuiltIn::PrimitiveIndex,
            BuiltIn::SampleIndex => naga::BuiltIn::SampleIndex,
            BuiltIn::SampleMask => naga::BuiltIn::SampleMask,
            BuiltIn::FragDepth => naga::BuiltIn::FragDepth,
            BuiltIn::WorkgroupId => naga::BuiltIn::WorkGroupId,
            BuiltIn::LocalInvocationId => naga::BuiltIn::LocalInvocationId,
            BuiltIn::GlobalInvocationId => naga::BuiltIn::GlobalInvocationId,
            BuiltIn::LocalInvocationIndex => naga::BuiltIn::LocalInvocationIndex,
            BuiltIn::SubgroupInvocationId => naga::BuiltIn::SubgroupInvocationId,
            BuiltIn::SubgroupSize => naga::BuiltIn::SubgroupSize,
            BuiltIn::ClipDistances => naga::BuiltIn::ClipDistances,
        }),
        target::Binding::Location {
            location,
            interpolation,
            sampling,
        } => Binding::Location {
            location,
            interpolation: interpolation.map(|i| match i {
                crate::InterpMode::Perspective => naga::Interpolation::Perspective,
                crate::InterpMode::Linear => naga::Interpolation::Linear,
                crate::InterpMode::Flat => naga::Interpolation::Flat,
            }),
            sampling: match sampling {
                crate::InterpSampling::Center => None,
                crate::InterpSampling::Centroid => Some(naga::Sampling::Centroid),
                crate::InterpSampling::Sample => Some(naga::Sampling::Sample),
            },
            blend_src: None,
            per_primitive: false,
        },
    }
}

/// One argument or result member of a stage function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FnItem {
    /// The name of the argument or member: after the varying it carries (`<NAME><index>`), or the pipeline value's own.
    pub name: String,
    /// The type of the member.
    pub ty: IoType,
    /// The varying it carries, by index into `list`; `None` for the ids the pipeline supplies.
    pub varying: Option<usize>,
    /// The list `varying` indexes.
    pub list: ItemList,
}

/// What a [`FnItem`] carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemList {
    /// The interface's inputs.
    Inputs,
    /// The interface's outputs (a hull program's patch-constant function reads the output control points).
    Outputs,
    /// The patch constants: a hull program's patch-constant result, a domain program's patch-constant arguments.
    PatchConstants,
    /// A value the pipeline supplies, named by the item (`SV_OutputControlPointID`, `SV_PrimitiveID`,
    /// `SV_DomainLocation`, `SV_GSInstanceID`, a geometry program's `SV_VertexCount` and `SV_StripStart`), or a ray
    /// tracing program's payload and attributes.
    Pipeline,
}

/// A function of a program that has no entry point: its name, arguments and result members (a structure `Out` when
/// there are any).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionSig {
    /// The function's name in the module.
    pub name: &'static str,
    /// Its arguments, in order.
    pub arguments: Vec<FnItem>,
    /// Its result members, in order; the function returns a structure `Out` when there are any.
    pub result: Vec<FnItem>,
}

/// The functions a program of a stage naga has no entry point for is made of; `None` for a stage that is an entry
/// point. The arguments carry no bindings: they are named after the varying they carry, as an entry point's are.
///
/// - A vertex program whose outputs feed a tessellator: `main(inputs) -> Out`.
/// - A hull program: `main(inputs as arrays of the input control points, SV_OutputControlPointID: u32,
///   SV_PrimitiveID: u32) -> Out`, the control-point phase, run for each output control point of the patch; and `patch_constant(inputs as arrays of the input
///   control points, outputs as arrays of the output control points under `out_` and the name, SV_PrimitiveID: u32) -> Out`,
///   the patch-constant phase, run once for a patch, its result the patch constants.
/// - A domain program: `main(patch constants, inputs as arrays of the input control points, SV_DomainLocation:
///   vec3<f32>, SV_PrimitiveID: u32) -> Out`.
/// - A geometry program: `main(inputs as arrays of the primitive's vertices, SV_PrimitiveID: u32, SV_GSInstanceID:
///   u32) -> Out`, the result each output varying as an array of the most vertices an invocation emits, then
///   `SV_VertexCount` (how many were emitted) and `SV_StripStart` (per vertex, 1 where it begins a strip).
/// - A ray tracing program (ray generation, intersection, any-hit, closest-hit, miss, callable): `main(the ray's
///   system values it reads, payload, attributes)` returning nothing, `payload` for every stage but ray generation and
///   intersection, `attributes` for any-hit and closest-hit; the payload and attributes are pointers to structures the
///   shader defines ([`IoType::Memory`]). What the shader does to the ray tracing pipeline is calls to the module's
///   functions named by [`crate::rt`] (`TraceRay`, `ReportHit`, `CallShader`, `IgnoreHit`, `AcceptHitAndEndSearch`),
///   which have no body: they are the pipeline's.
pub fn stage_functions(interface: &Interface) -> Result<Option<Vec<FunctionSig>>, String> {
    let stage = interface.stage;
    let values = |list: &[Varying], output: bool, of: ItemList| -> Vec<FnItem> {
        list.iter()
            .enumerate()
            .map(|(i, v)| FnItem {
                name: varying_name(&v.label),
                ty: IoType::Value(carried(stage, output, v)),
                varying: Some(i),
                list: of,
            })
            .collect()
    };
    let arrays = |list: &[Varying], output: bool, prefix: &str, n: u32, of: ItemList| -> Vec<FnItem> {
        list.iter()
            .enumerate()
            .map(|(i, v)| FnItem {
                name: format!("{prefix}{}", varying_name(&v.label)),
                ty: IoType::Array(carried(stage, output, v), n),
                varying: Some(i),
                list: of,
            })
            .collect()
    };
    let scalar = |name: &str, width: u8, scalar: ScalarType| FnItem {
        name: name.into(),
        ty: IoType::Value(Value { scalar, width }),
        varying: None,
        list: ItemList::Pipeline,
    };
    let (inputs, outputs, patch) = (ItemList::Inputs, ItemList::Outputs, ItemList::PatchConstants);
    Ok(match (&interface.stage, &interface.facts) {
        (Stage::Vertex, StageFacts::VertexWithoutPosition) => Some(vec![FunctionSig {
            name: "main",
            arguments: values(&interface.inputs, false, inputs),
            result: values(&interface.outputs, true, outputs),
        }]),
        (Stage::Hull, StageFacts::Hull(h)) => {
            let mut main_args = arrays(&interface.inputs, false, "", h.input_control_points, inputs);
            main_args.push(scalar("SV_OutputControlPointID", 1, ScalarType::Uint));
            // the control-point phase may read the patch's number too (`vPrim`)
            main_args.push(scalar("SV_PrimitiveID", 1, ScalarType::Uint));
            let mut constant_args = arrays(&interface.inputs, false, "", h.input_control_points, inputs);
            constant_args.extend(arrays(
                &interface.outputs,
                true,
                "out_",
                h.output_control_points,
                outputs,
            ));
            constant_args.push(scalar("SV_PrimitiveID", 1, ScalarType::Uint));
            Some(vec![
                FunctionSig {
                    name: "main",
                    arguments: main_args,
                    result: values(&interface.outputs, true, outputs),
                },
                FunctionSig {
                    name: "patch_constant",
                    arguments: constant_args,
                    result: values(&h.patch_constants, true, patch),
                },
            ])
        }
        (Stage::Domain, StageFacts::Domain(d)) => {
            let mut args = values(&d.patch_constants, false, patch);
            args.extend(arrays(
                &interface.inputs,
                false,
                "",
                d.input_control_points,
                inputs,
            ));
            args.push(scalar("SV_DomainLocation", 3, ScalarType::Float));
            args.push(scalar("SV_PrimitiveID", 1, ScalarType::Uint));
            Some(vec![FunctionSig {
                name: "main",
                arguments: args,
                result: values(&interface.outputs, true, outputs),
            }])
        }
        (Stage::Hull | Stage::Domain, _) => {
            return Err(format!(
                "{} program without its patch facts",
                stage.with_article()
            ));
        }
        (Stage::Geometry, StageFacts::Geometry(g)) => {
            let mut args = arrays(&interface.inputs, false, "", g.input.vertices(), inputs);
            args.push(scalar("SV_PrimitiveID", 1, ScalarType::Uint));
            args.push(scalar("SV_GSInstanceID", 1, ScalarType::Uint));
            // the emitted vertices: each output varying for every vertex the invocation may emit, how many it did, and
            // whether each starts a new strip (the first, and the first after a cut)
            let mut result = arrays(&interface.outputs, true, "", g.max_vertices, outputs);
            result.push(scalar("SV_VertexCount", 1, ScalarType::Uint));
            result.push(FnItem {
                name: "SV_StripStart".into(),
                ty: IoType::Array(
                    Value {
                        scalar: ScalarType::Uint,
                        width: 1,
                    },
                    g.max_vertices,
                ),
                varying: None,
                list: ItemList::Pipeline,
            });
            Some(vec![FunctionSig {
                name: "main",
                arguments: args,
                result,
            }])
        }
        (Stage::Geometry, _) => return Err("a geometry program without its facts".into()),
        // a ray tracing program: the ray's system values it reads, then the payload (a hit, miss or callable program
        // has one, `payload`: what the traced ray carries, or a callable's parameter) and the attributes of the hit
        // (`attributes`, a triangle's barycentrics or what an intersection program reported)
        (stage, StageFacts::None | StageFacts::RayTracing(_)) if stage.is_ray_tracing() => {
            let mut arguments = values(&interface.inputs, false, inputs);
            let memory = |name: &str| FnItem {
                name: name.into(),
                ty: IoType::Memory,
                varying: None,
                list: ItemList::Pipeline,
            };
            if matches!(
                stage,
                Stage::AnyHit | Stage::ClosestHit | Stage::Miss | Stage::Callable
            ) {
                arguments.push(memory("payload"));
            }
            if matches!(stage, Stage::AnyHit | Stage::ClosestHit) {
                arguments.push(memory("attributes"));
            }
            Some(vec![FunctionSig {
                name: "main",
                arguments,
                result: Vec::new(),
            }])
        }
        (stage, facts) if stage.is_ray_tracing() => {
            return Err(format!("{} program with {facts:?}", stage.with_article()));
        }
        _ => None,
    })
}

/// The members of a mesh program's vertex type, one for each output varying, bound as a vertex program's outputs are.
pub fn mesh_vertex_io(interface: &Interface) -> Result<Vec<IoItem>, String> {
    let mut vertex = interface.clone();
    vertex.stage = Stage::Vertex;
    vertex.inputs.clear();
    vertex.facts = StageFacts::None;
    Ok(entry_io(&vertex)?.results)
}

/// The entry point's arguments and result members for `interface` ([`iridia_ir::target::entry_io`], with naga's
/// bindings). `Err` says why the module cannot carry it (a system value no stage binding exists for).
pub fn entry_io(interface: &Interface) -> Result<EntryIo, String> {
    let io = target::entry_io(interface)?;
    let item = |i: target::IoItem| IoItem {
        name: i.name,
        ty: i.ty,
        binding: naga_binding(i.binding),
        varyings: i.varyings,
    };
    Ok(EntryIo {
        arguments: io.arguments.into_iter().map(item).collect(),
        results: io.results.into_iter().map(item).collect(),
    })
}
