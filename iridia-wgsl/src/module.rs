//! The module: the program's resources as globals where a binding plan puts them, the entry point `main` that carries its
//! interface, and the helper functions its operations call.

use alloc::{
    collections::{BTreeMap, BTreeSet},
    format,
    string::{String, ToString},
    vec::Vec,
};

use iridia_ir::interface::{
    Access, BufferKind, Resource, ResourceKind, SampleType, Stage, StageFacts, StorageTarget, TextureDim,
    Value,
};
use iridia_ir::target::{Bind, Binding, BindingPlan, BuiltIn, IoItem, IoType, entry_io};
use iridia_ir::{Body, Program, Scalar, Space, Type};

use crate::emit::{Member, R, Writer};
use crate::helpers;
use crate::memory::{Atomics, counted};
use crate::names::Names;
use crate::profile::{Directives, Profile};
use crate::text::{self, err};

/// The member a result item is, with the binding and type its pipeline gives it.
pub(crate) struct Linked {
    /// The binding the member has: its consumer's where it is read, else the program's own.
    pub binding: Binding,
    /// The value the member has: the consumer's where it differs from what the program writes.
    pub value: Option<Value>,
}

/// The name of the private table of Direct3D's standard sample positions.
pub(crate) const SAMPLE_POSITIONS: &str = "sample_positions";

/// The rows of Direct3D's standard sample positions, of every standard count (1 + 2 + 4 + 8 + 16).
pub(crate) const STANDARD_ROWS: u32 = 31;

/// WGSL's text of a texture's type, or why it has none.
fn texture_type(dim: TextureDim, arrayed: bool, multisampled: bool, sample: SampleType) -> R<String> {
    if multisampled && (arrayed || dim != TextureDim::D2) {
        return Err(err("a multisampled array texture has no WGSL form"));
    }
    if dim == TextureDim::D1 && arrayed {
        return Err(err("a 1D array texture has no WGSL form"));
    }
    let d = match dim {
        TextureDim::D1 => "1d",
        TextureDim::D2 => "2d",
        TextureDim::D3 => "3d",
        TextureDim::Cube => "cube",
    };
    let array = if arrayed { "_array" } else { "" };
    Ok(match sample {
        SampleType::Depth => {
            if matches!(dim, TextureDim::D1 | TextureDim::D3) {
                return Err(err(format!("a depth texture of {dim:?} has no WGSL form")));
            }
            if multisampled {
                "texture_depth_multisampled_2d".to_string()
            } else {
                format!("texture_depth_{d}{array}")
            }
        }
        other => {
            let t = match other {
                SampleType::Float => "f32",
                SampleType::Sint => "i32",
                _ => "u32",
            };
            if multisampled {
                format!("texture_multisampled_2d<{t}>")
            } else {
                format!("texture_{d}{array}<{t}>")
            }
        }
    })
}

/// A storage image's type: of 32-bit lanes of its sample type, four of them (`rgba32float`, `rgba32uint`, `rgba32sint`), or
/// one where an atomic reaches it (`r32uint`, `r32sint`, with WGSL's `atomic` access, which loads and stores too).
fn storage_image(
    dim: TextureDim,
    arrayed: bool,
    sample: SampleType,
    access: Access,
    atomic: bool,
) -> R<String> {
    if dim == TextureDim::D1 && arrayed {
        return Err(err("a 1D array storage image has no WGSL form"));
    }
    let format = match (sample, atomic) {
        (SampleType::Sint, true) => "r32sint",
        (SampleType::Uint, true) => "r32uint",
        (SampleType::Float, true) => {
            return Err(err("a storage image of Float texels an atomic reaches"));
        }
        (SampleType::Sint, false) => "rgba32sint",
        (SampleType::Uint, false) => "rgba32uint",
        (SampleType::Float, false) => "rgba32float",
        (SampleType::Depth, _) => return Err(err("a depth storage image has no WGSL form")),
    };
    let d = match dim {
        TextureDim::D1 => "1d",
        TextureDim::D2 => "2d",
        TextureDim::D3 => "3d",
        TextureDim::Cube => return Err(err("a cube storage image")),
    };
    let array = if arrayed { "_array" } else { "" };
    let access = match access {
        _ if atomic => "atomic",
        Access::Read => "read",
        Access::Write => "write",
        Access::ReadWrite => "read_write",
    };
    Ok(format!("texture_storage_{d}{array}<{format}, {access}>"))
}

impl<'a> Writer<'a> {
    /// The text of the whole module for `program`.
    pub(crate) fn module(
        program: &'a Program,
        plan: &'a BindingPlan,
        profile: &'a Profile,
        link: impl FnOnce(&[IoItem]) -> R<Vec<Linked>>,
    ) -> R<String> {
        let f = &program.entry;
        let Body::Regions(body) = &f.body else {
            return Err(err("a graph body: structurize it first"));
        };
        let stage_attribute = match program.stage {
            Stage::Vertex => "@vertex".to_string(),
            Stage::Pixel => "@fragment".to_string(),
            Stage::Compute => {
                let [x, y, z] = program
                    .workgroup
                    .ok_or_else(|| err("a compute program without its thread group size"))?;
                format!("@compute @workgroup_size({x}, {y}, {z})")
            }
            other => {
                return Err(err(format!(
                    "{} program is not written yet",
                    other.with_article()
                )));
            }
        };
        if program.interface.facts == StageFacts::VertexWithoutPosition {
            return Err(err(
                "a vertex program that writes no position feeds a tessellator: it is a function, not a WGSL entry point",
            ));
        }
        let mut io = entry_io(&program.interface).map_err(err)?;
        let linked = link(&io.results)?;
        // the names of the arguments and result members are the labels', made WGSL identifiers
        let mut names = Names::default();
        for name in ["Out", "main", SAMPLE_POSITIONS] {
            names.reserve(name);
        }
        for item in io.arguments.iter_mut().chain(io.results.iter_mut()) {
            item.name = names.unique(&item.name);
        }
        let mut members = Vec::new();
        for (item, link) in io.results.iter().zip(linked) {
            let (carried, clip) = match item.ty {
                IoType::Value(v) => (v, false),
                IoType::ClipDistances(n) => (
                    Value {
                        scalar: iridia_ir::interface::ScalarType::Float,
                        width: u8::try_from(n)
                            .map_err(|_| err(format!("{}: {n} clip distances", item.name)))?,
                    },
                    true,
                ),
                ref other => return Err(err(format!("{}: {other:?} in an entry point", item.name))),
            };
            members.push(Member {
                name: item.name.clone(),
                binding: link.binding,
                carried,
                linked: link.value.unwrap_or(carried),
                slots: item.varyings.clone(),
                clip,
            });
        }
        let mut w = Writer {
            program,
            f,
            plan,
            profile,
            names,
            io,
            members,
            globals: Vec::new(),
            counters: Vec::new(),
            vars: Vec::new(),
            atomics: Atomics::of(f)?,
            helpers: BTreeSet::new(),
            out: String::new(),
            depth: 1,
            next: 0,
            defined: alloc::vec![false; f.values.len()],
            hidden: BTreeSet::new(),
            constructs: Vec::new(),
            lanes: BTreeMap::new(),
            split: BTreeMap::new(),
            gathered: None,
            read: crate::flow::read_values(f, body),
            sites: crate::origin::Sites::of(f, body),
            forward: BTreeMap::new(),
            outputs: alloc::vec![None; program.interface.outputs.len()],
            decls: Vec::new(),
        };
        let result_struct = w.result_struct()?;
        let globals = w.declare_resources()?;
        let (workgroup_vars, private_vars) = w.declare_vars()?;
        w.declare_outputs()?;
        // the body, then the function around it
        w.region(body, crate::flow::Yield::Top)?;
        let body_text = core::mem::take(&mut w.out);
        let mut text = String::new();
        if program.interface.facts == StageFacts::EarlyDepthStencil {
            text.push_str("// early_depth_stencil: WGSL has no early depth/stencil guarantee.\n");
        }
        text.push_str(&w.enables(&body_text)?);
        for d in &w.decls {
            text.push_str(d);
            text.push('\n');
        }
        text.push_str(&result_struct);
        text.push_str(&globals);
        text.push_str(&workgroup_vars);
        // a literal infinity or NaN is a call that no helper list recorded
        if [&body_text, &globals, &private_vars, &result_struct]
            .iter()
            .any(|t| t.contains("iridia_f32_from_bits("))
        {
            w.helpers.insert(helpers::Helper::F32FromBits);
        }
        let defs = helpers::definitions(&w.helpers);
        if !defs.is_empty() {
            text.push_str(&defs);
            text.push('\n');
        }
        text.push_str(&stage_attribute);
        text.push('\n');
        text.push_str(&w.signature()?);
        text.push_str(&private_vars);
        text.push_str(&body_text);
        text.push_str("}\n");
        Ok(text)
    }

    /// The `enable` directives the text needs.
    fn enables(&self, body: &str) -> R<String> {
        let mut out = String::new();
        let float16 = self.f.values.iter().any(|v| v.ty.scalar == Scalar::F16)
            || self.f.vars.iter().any(|v| v.ty.scalar == Scalar::F16);
        if float16 {
            out.push_str("enable f16;\n");
        }
        let int16 = self
            .f
            .values
            .iter()
            .any(|v| matches!(v.ty.scalar, Scalar::I16 | Scalar::U16))
            || self
                .f
                .vars
                .iter()
                .any(|v| matches!(v.ty.scalar, Scalar::I16 | Scalar::U16));
        if int16 {
            out.push_str("enable wgpu_int16;\n");
        }
        let bindings = self
            .io
            .arguments
            .iter()
            .map(|a| &a.binding)
            .chain(self.members.iter().map(|m| &m.binding));
        let (mut clip, mut primitive) = (false, false);
        for b in bindings {
            match b {
                Binding::BuiltIn(BuiltIn::ClipDistances) => clip = true,
                Binding::BuiltIn(BuiltIn::PrimitiveIndex) => primitive = true,
                _ => {}
            }
        }
        if clip {
            out.push_str("enable clip_distances;\n");
        }
        if primitive {
            out.push_str("enable primitive_index;\n");
        }
        if self.program.interface.resources.iter().any(Resource::is_array) {
            out.push_str("enable wgpu_binding_array;\n");
        }
        if self.profile.directives == Directives::Standard
            && iridia_ir::target::requirements(self.program)
                .iter()
                .any(|r| r.capability == iridia_ir::target::Capability::Subgroup)
        {
            out.push_str("enable subgroups;\n");
        }
        // The source program takes these derivatives where it takes them, in divergent control flow as often as not
        // (Direct3D leaves the value past a non-uniform branch to the hardware); a browser refuses the text unless told
        // not to check. The directive restates what the source does: hoisting a sample out of its branch would change
        // the program.
        if self.profile.directives == Directives::Standard
            && [
                "textureSample(",
                "textureSampleBias(",
                "textureSampleCompare(",
                "dpdx",
                "dpdy",
                "fwidth",
            ]
            .iter()
            .any(|call| body.contains(call))
        {
            out.push_str("diagnostic(off, derivative_uniformity);\n");
        }
        if !out.is_empty() {
            out.push('\n');
        }
        Ok(out)
    }

    /// The structure of the result, one member per output; nothing for a program with none.
    fn result_struct(&mut self) -> R<String> {
        if self.members.is_empty() {
            return Ok(String::new());
        }
        let mut out = String::from("struct Out {\n");
        for m in &self.members {
            let ty = if m.clip {
                format!("array<f32, {}>", m.carried.width)
            } else {
                text::ty(iridia_ir::value_type(m.linked))?
            };
            out.push_str(&format!(
                "    {}{}: {ty},\n",
                crate::io::attributes(&m.binding),
                m.name
            ));
        }
        out.push_str("}\n\n");
        Ok(out)
    }

    /// `fn main(...) -> Out {`, the arguments one to a line.
    fn signature(&mut self) -> R<String> {
        let mut args = Vec::new();
        for item in &self.io.arguments {
            let ty = match item.ty {
                IoType::Value(v) => text::ty(iridia_ir::value_type(v))?,
                ref other => return Err(err(format!("{}: {other:?} in an entry point", item.name))),
            };
            args.push(format!(
                "    {}{}: {ty}",
                crate::io::attributes(&item.binding),
                item.name
            ));
        }
        let result = if self.members.is_empty() { "" } else { " -> Out" };
        Ok(if args.is_empty() {
            format!("fn main(){result} {{\n")
        } else {
            format!("fn main(\n{}\n){result} {{\n", args.join(",\n"))
        })
    }

    /// The globals of the resources and of the hidden counters, with the bindings the plan gives them.
    fn declare_resources(&mut self) -> R<String> {
        let mut out = String::new();
        let resources: Vec<Resource> = self
            .program
            .interface
            .resources
            .iter()
            .filter(|r| !matches!(r.kind, ResourceKind::PipelineState(_)))
            .cloned()
            .collect();
        for r in &resources {
            let bind: &Bind = self
                .plan
                .bindings
                .iter()
                .find(|b| b.resource == r.id)
                .ok_or_else(|| {
                    err(format!(
                        "resource {} ({}) has no binding in the plan",
                        r.id.0, r.provenance.name
                    ))
                })?;
            let name = self.names.unique(&bind.name);
            let (space, ty) = match &r.kind {
                ResourceKind::ConstantBuffer { layout } => {
                    let rows = layout.size.div_ceil(16);
                    if rows == 0 {
                        return Err(err("an empty constant buffer"));
                    }
                    ("var<uniform> ".to_string(), format!("array<vec4<u32>, {rows}>"))
                }
                ResourceKind::Texture {
                    dim,
                    arrayed,
                    multisampled,
                    sample,
                } => (
                    String::from("var "),
                    texture_type(*dim, *arrayed, *multisampled, *sample)?,
                ),
                ResourceKind::Sampler { comparison } => (
                    "var ".to_string(),
                    if *comparison {
                        "sampler_comparison"
                    } else {
                        "sampler"
                    }
                    .to_string(),
                ),
                ResourceKind::Storage {
                    target: StorageTarget::Image { dim, arrayed, sample },
                    access,
                } => {
                    let atomic = self.atomics.resources.contains_key(&r.id);
                    (
                        "var ".to_string(),
                        storage_image(*dim, *arrayed, *sample, *access, atomic)?,
                    )
                }
                // WGSL has no write-only storage buffer: a buffer the program writes is read-write
                ResourceKind::Buffer(kind) => {
                    ("var<storage, read> ".to_string(), self.buffer_type(r.id, *kind)?)
                }
                ResourceKind::Storage {
                    target: StorageTarget::Buffer(kind),
                    access,
                } => (
                    format!(
                        "var<storage, {}> ",
                        if *access == Access::Read {
                            "read"
                        } else {
                            "read_write"
                        }
                    ),
                    self.buffer_type(r.id, *kind)?,
                ),
                other => return Err(err(format!("a {other:?} resource is not written yet"))),
            };
            // an array of resources is one binding array at its first register (unsized where the program does not size it);
            // a binding array of buffers is one of structures, each with one member `data`: the buffer's array
            let ty = if r.is_array() {
                let element = if space.starts_with("var<") {
                    let element = self.names.unique(&format!("{name}_element"));
                    self.decls
                        .push(format!("struct {element} {{\n    data: {ty},\n}}\n"));
                    element
                } else {
                    ty
                };
                if r.count == Resource::UNBOUNDED {
                    format!("binding_array<{element}>")
                } else {
                    format!("binding_array<{element}, {}>", r.count)
                }
            } else {
                ty
            };
            out.push_str(&format!(
                "@group({}) @binding({}) {space}{name}: {ty};\n",
                bind.group, bind.binding
            ));
            self.globals.push((r.id, name));
        }
        // the hidden counter of each view whose counter the program changes, where the plan binds it
        for id in counted(self.f) {
            let bind = self
                .plan
                .counters
                .iter()
                .find(|b| b.resource == id)
                .ok_or_else(|| {
                    err(format!(
                        "the counter of resource {} has no binding in the plan",
                        id.0
                    ))
                })?;
            let name = self.names.unique(&bind.name);
            out.push_str(&format!(
                "@group({}) @binding({}) var<storage, read_write> {name}: array<atomic<u32>>;\n",
                bind.group, bind.binding
            ));
            self.counters.push((id, name));
        }
        if !out.is_empty() {
            out.push('\n');
        }
        Ok(out)
    }

    /// A buffer's array: of 32-bit words (atomic words where an atomic reaches it) for a raw or structured buffer, of
    /// `vec4<u32>` elements for a typed one (structures of four atomic words where an atomic reaches it).
    fn buffer_type(&mut self, id: iridia_ir::interface::ResourceId, kind: BufferKind) -> R<String> {
        let atomic = self.atomics.resources.get(&id).copied();
        Ok(match (kind, atomic) {
            (BufferKind::Raw | BufferKind::Structured { .. }, Some(s)) => {
                format!("array<atomic<{}>>", s.name())
            }
            (BufferKind::Raw | BufferKind::Structured { .. }, None) => "array<u32>".to_string(),
            (BufferKind::Typed(_), Some(s)) => {
                let name = if s == Scalar::I32 {
                    "AtomicWordsI32"
                } else {
                    "AtomicWordsU32"
                };
                self.names.reserve(name);
                self.words_structs(name, s);
                format!("array<{name}>")
            }
            (BufferKind::Typed(_), None) => "array<vec4<u32>>".to_string(),
        })
    }

    /// Declare the structure of four atomic words a typed buffer an atomic reaches is made of (once).
    fn words_structs(&mut self, name: &str, s: Scalar) {
        let decl = format!(
            "struct {name} {{\n    w0: atomic<{0}>,\n    w1: atomic<{0}>,\n    w2: atomic<{0}>,\n    w3: atomic<{0}>,\n}}\n",
            s.name()
        );
        if !self.decls.contains(&decl) {
            self.decls.push(decl);
        }
    }

    /// Direct3D's standard sample positions, in pixels from the pixel's centre, of every standard count in turn (1, 2, 4, 8 and
    /// 16 samples: [`STANDARD_ROWS`] rows, count `n`'s from row `n - 1`): a private table, declared once.
    pub(crate) fn sample_positions(&mut self) -> R<&'static str> {
        let mut rows = Vec::new();
        for count in [1u32, 2, 4, 8, 16] {
            for index in 0..count {
                let [x, y] = iridia_ir::standard_sample_position(count, index)
                    .ok_or_else(|| err(format!("no standard position of sample {index} of {count}")))?;
                let lane = |l: f32| text::literal(Scalar::F32, u64::from(l.to_bits()));
                rows.push(format!("vec2<f32>({}, {})", lane(x)?, lane(y)?));
            }
        }
        if rows.len() != STANDARD_ROWS as usize {
            return Err(err(format!("{} standard sample positions", rows.len())));
        }
        let ty = format!("array<vec2<f32>, {STANDARD_ROWS}>");
        let decl = format!(
            "var<private> {SAMPLE_POSITIONS}: {ty} = {ty}({});\n",
            rows.join(", ")
        );
        if !self.decls.contains(&decl) {
            self.decls.push(decl);
        }
        Ok(SAMPLE_POSITIONS)
    }

    /// The variables of the function: the workgroup ones as globals, the private ones as locals at the top of the entry
    /// point.
    fn declare_vars(&mut self) -> R<(String, String)> {
        let (mut workgroup, mut private) = (String::new(), String::new());
        for (n, var) in self.f.vars.iter().enumerate() {
            if var.space != Space::Private && var.space != Space::Workgroup {
                return Err(err(format!(
                    "{:?} variable {} is not written yet",
                    var.space, var.name
                )));
            }
            let is_workgroup = var.space == Space::Workgroup;
            // a variable an atomic reaches is made of atomic words, which WGSL has in workgroup memory only
            let atomic = u32::try_from(n)
                .ok()
                .and_then(|n| self.atomics.vars.get(&n).copied());
            let element = match atomic {
                Some(s) if is_workgroup && var.ty.lanes == 1 => format!("atomic<{}>", s.name()),
                Some(_) => {
                    return Err(err(format!(
                        "an atomic of {} variable {} is not written yet",
                        if is_workgroup { "the vector" } else { "the private" },
                        var.name
                    )));
                }
                None => text::ty(var.ty)?,
            };
            let ty = match var.len {
                None => element.clone(),
                Some(0) => return Err(err("an array of no elements")),
                Some(len) => format!("array<{element}, {len}>"),
            };
            let name = self.names.unique(&var.name);
            self.vars.push(name.clone());
            if is_workgroup {
                // workgroup memory starts as zero in WGSL, as the IR's does
                if !var.init.is_empty() {
                    return Err(err(format!(
                        "workgroup variable {} has an initial value",
                        var.name
                    )));
                }
                workgroup.push_str(&format!("var<workgroup> {name}: {ty};\n"));
                continue;
            }
            let init = if var.init.is_empty() {
                String::new()
            } else {
                let mut elements: Vec<String> = var
                    .init
                    .iter()
                    .map(|bits| text::value(var.ty, *bits))
                    .collect::<Result<_, _>>()?;
                let value = match (var.len, elements.as_slice()) {
                    (None, [one]) => one.clone(),
                    _ => {
                        // elements past those given start as zero
                        let len = var.len.unwrap_or(1) as usize;
                        let zero = format!("{element}()");
                        while elements.len() < len {
                            elements.push(zero.clone());
                        }
                        format!("{ty}({})", elements.join(", "))
                    }
                };
                format!(" = {value}")
            };
            private.push_str(&format!("    var {name}: {ty}{init};\n"));
        }
        if !workgroup.is_empty() {
            workgroup.push('\n');
        }
        Ok((workgroup, private))
    }

    /// A local per output slot a member carries: zero before its first store.
    fn declare_outputs(&mut self) -> R<()> {
        let slots: Vec<usize> = self
            .members
            .iter()
            .flat_map(|m| m.slots.iter().copied())
            .collect();
        let mut declared = String::new();
        for slot in slots {
            let t: Type = self.slot_type(slot)?;
            let name = format!("o{slot}");
            declared.push_str(&format!("    var {name}: {};\n", text::ty(t)?));
            if let Some(o) = self.outputs.get_mut(slot) {
                *o = Some(name);
            }
        }
        self.out.push_str(&declared);
        Ok(())
    }
}
