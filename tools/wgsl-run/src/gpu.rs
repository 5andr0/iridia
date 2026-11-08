//! The device (DX12's WARP adapter) and one job's run on it: the resources the entry point uses made from the job's,
//! the pipeline, the dispatch or draw, and what it wrote read back.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use naga::{
    AddressSpace, ArraySize, Handle, ImageClass, ImageDimension, ScalarKind, ShaderStage, StorageAccess,
    StorageFormat, TypeInner,
};
use wgpu::util::DeviceExt as _;

use crate::job::{Invocation, Job, Reply, ResourceKind, Written};
use crate::wrap::{self, Wrapped};

/// The device jobs run on.
pub struct Gpu {
    handle: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// The adapter's name, as replies state it.
    pub adapter: String,
    /// How long a job's GPU work may take.
    timeout: Duration,
    /// The device's features.
    features: wgpu::Features,
}

/// Why a job did not finish: an error to reply with, or a GPU that did not finish its work (the process cannot go on).
pub enum Failure {
    Error(String),
    TimedOut(String),
}

impl From<String> for Failure {
    fn from(e: String) -> Self {
        Failure::Error(e)
    }
}

impl From<&str> for Failure {
    fn from(e: &str) -> Self {
        Failure::Error(e.to_owned())
    }
}

/// The full-screen triangle the pixel programs' quads are drawn with.
const TRIANGLE: &str = "@vertex
fn main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = select(-1.0, 3.0, i == 1u);
    let y = select(1.0, -3.0, i == 2u);
    return vec4<f32>(x, y, 0.5, 1.0);
}
";

impl Gpu {
    /// The WARP adapter of DX12 and a device with every feature it has that needs no opt-in; shaders compiled by the
    /// DXC of `dxc` (a path to `dxcompiler.dll`, with `dxil.dll` beside it) or by FXC where `None`.
    pub fn new(dxc: Option<String>, timeout: Duration) -> Result<Gpu, String> {
        let shader_compiler = match dxc {
            Some(dxc_path) => wgpu::Dx12Compiler::DynamicDxc { dxc_path },
            None => wgpu::Dx12Compiler::Fxc,
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::DX12,
            flags: wgpu::InstanceFlags::empty(),
            backend_options: wgpu::BackendOptions {
                dx12: wgpu::Dx12BackendOptions {
                    shader_compiler,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::DX12));
        let names: Vec<String> = adapters.iter().map(|a| a.get_info().name).collect();
        let adapter = adapters
            .into_iter()
            .find(|a| a.get_info().device_type == wgpu::DeviceType::Cpu)
            .ok_or_else(|| format!("no WARP adapter among DX12's: {names:?}"))?;
        let info = adapter.get_info();
        let mut features = wgpu::Features::empty();
        for (name, flag) in adapter.features().iter_names() {
            if !name.starts_with("EXPERIMENTAL") {
                features |= flag;
            }
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("wgsl-run"),
            required_features: features,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("the WARP device: {e}"))?;
        // errors are caught by the error scopes around each job; one outside them is reported, not a panic
        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| eprintln!("wgsl-run: uncaptured: {e}")));
        Ok(Gpu {
            handle: adapter,
            device,
            queue,
            adapter: format!(
                "{} ({:?}, {}; {features:?})",
                info.name, info.backend, info.driver_info
            ),
            timeout,
            features,
        })
    }

    /// Run a job.
    pub fn run(&self, job: &Job) -> Result<Reply, Failure> {
        let wrapped = wrap::wrap(&job.wgsl, job.entry.as_deref(), job.workgroup_init)?;
        let scopes = [
            self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
            self.device.push_error_scope(wgpu::ErrorFilter::Internal),
            self.device.push_error_scope(wgpu::ErrorFilter::Validation),
        ];
        let ran = self.run_wrapped(job, &wrapped);
        let mut errors = Vec::new();
        for scope in scopes.into_iter().rev() {
            if let Some(e) = pollster::block_on(scope.pop()) {
                errors.push(e.to_string());
            }
        }
        if !errors.is_empty() {
            return Err(Failure::Error(format!("wgpu: {}", errors.join("; "))));
        }
        ran
    }

    fn run_wrapped(&self, job: &Job, w: &Wrapped) -> Result<Reply, Failure> {
        let stage = w.stage;
        let visibility = match stage {
            ShaderStage::Fragment => wgpu::ShaderStages::FRAGMENT,
            _ => wgpu::ShaderStages::COMPUTE,
        };
        let uses = &w.info.get_entry_point(w.entry_index);
        // the resources the entry point uses, by group: each one's layout entry and binding resource
        let mut groups: BTreeMap<u32, Vec<(wgpu::BindGroupLayoutEntry, Bound)>> = BTreeMap::new();
        let mut readbacks = Vec::new();
        let invocations = if stage == ShaderStage::Compute {
            0
        } else {
            u32::try_from(job.invocations.len()).map_err(|_| "too many invocations".to_string())?
        };
        if stage == ShaderStage::Fragment && invocations % 4 != 0 {
            return Err(format!("{invocations} pixel invocations: a whole number of quads is needed").into());
        }
        for (handle, g) in w.module.global_variables.iter() {
            if uses[handle].is_empty() {
                continue;
            }
            let Some(binding) = g.binding else {
                continue;
            };
            let (entry, bound) = if binding.group == wrap::GROUP {
                let words = if binding.binding == wrap::INPUTS {
                    self.inputs(job, w)?
                } else {
                    vec![0; (invocations * w.output_words()).max(1) as usize]
                };
                let buffer = self.buffer(&words, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
                let read_only = binding.binding == wrap::INPUTS;
                (
                    layout_entry(
                        binding.binding,
                        visibility,
                        buffer_type(wgpu::BufferBindingType::Storage { read_only }),
                        None,
                    ),
                    Bound::One(One::Buffer(buffer)),
                )
            } else {
                let resource = job
                    .resources
                    .iter()
                    .find(|r| r.group == binding.group && r.binding == binding.binding)
                    .ok_or_else(|| {
                        format!(
                            "the job has no resource for @group({}) @binding({}) ({})",
                            binding.group,
                            binding.binding,
                            g.name.as_deref().unwrap_or("unnamed")
                        )
                    })?;
                self.resource(
                    &w.module,
                    g,
                    binding.group,
                    binding.binding,
                    visibility,
                    &resource.kind,
                    &mut readbacks,
                )?
            };
            groups.entry(binding.group).or_default().push((entry, bound));
        }
        let last = groups.keys().next_back().copied();
        let mut layouts = Vec::new();
        for group in 0..=last.unwrap_or(0) {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = groups
                .get(&group)
                .map(|g| g.iter().map(|(e, _)| *e).collect())
                .unwrap_or_default();
            layouts.push(
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &entries,
                    }),
            );
        }
        let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts
            .iter()
            .take(if last.is_some() { layouts.len() } else { 0 })
            .map(Some)
            .collect();
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &layout_refs,
                immediate_size: 0,
            });
        let mut bind_groups = Vec::new();
        for (group, layout) in layouts.iter().enumerate().take(layout_refs.len()) {
            let empty = Vec::new();
            let members = groups.get(&(group as u32)).unwrap_or(&empty);
            let arrays: Vec<Elements> = members.iter().map(|(_, b)| Elements::of(b)).collect();
            let entries: Vec<wgpu::BindGroupEntry> = members
                .iter()
                .zip(&arrays)
                .map(|((e, b), elements)| wgpu::BindGroupEntry {
                    binding: e.binding,
                    resource: match b {
                        Bound::One(one) => one.resource(),
                        Bound::Array(_) => elements.resource(),
                    },
                })
                .collect();
            bind_groups.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &entries,
            }));
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("program"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(w.module.clone())),
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        match stage {
            ShaderStage::Fragment => {
                let pipeline = self
                    .device
                    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                        label: Some("program"),
                        layout: Some(&pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                                label: Some("triangle"),
                                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(TRIANGLE)),
                            }),
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            buffers: &[],
                        },
                        primitive: wgpu::PrimitiveState::default(),
                        depth_stencil: None,
                        multisample: wgpu::MultisampleState::default(),
                        fragment: Some(wgpu::FragmentState {
                            module: &module,
                            entry_point: Some(&w.entry),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: wgpu::TextureFormat::R8Unorm,
                                blend: None,
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview_mask: None,
                        cache: None,
                    });
                if invocations > 0 {
                    let target = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("quads"),
                        size: wgpu::Extent3d {
                            width: invocations / 2,
                            height: 2,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::R8Unorm,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[],
                    });
                    let view = target.create_view(&Default::default());
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(&pipeline);
                    for (k, g) in bind_groups.iter().enumerate() {
                        pass.set_bind_group(k as u32, g, &[]);
                    }
                    pass.draw(0..3, 0..1);
                }
            }
            _ => {
                let pipeline = self
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("program"),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some(&w.entry),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            zero_initialize_workgroup_memory: !w.zeroes_workgroup_memory,
                            ..Default::default()
                        },
                        cache: None,
                    });
                let groups = if stage == ShaderStage::Compute {
                    job.dispatch.ok_or("a compute program's job states no dispatch")?
                } else {
                    [invocations, 1, 1]
                };
                if groups.iter().all(|&n| n > 0) {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    for (k, g) in bind_groups.iter().enumerate() {
                        pass.set_bind_group(k as u32, g, &[]);
                    }
                    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                }
            }
        }
        // what the run wrote: the harness's results, then the program's writable resources
        let outputs = groups.get(&wrap::GROUP).and_then(|g| {
            g.iter().find_map(|(e, b)| match b {
                Bound::One(One::Buffer(buffer)) if e.binding == wrap::OUTPUTS => Some(buffer),
                _ => None,
            })
        });
        let mut copies = Vec::new();
        if let Some(buffer) = outputs {
            copies.push(self.copy_buffer(&mut encoder, buffer));
        }
        for r in &readbacks {
            copies.push(match r {
                Readback::Buffer { buffer, .. } => self.copy_buffer(&mut encoder, buffer),
                Readback::Texture { texture, lanes, .. } => self.copy_texture(&mut encoder, texture, *lanes),
            });
        }
        self.queue.submit([encoder.finish()]);
        let mut read = Vec::new();
        for (staging, layout) in &copies {
            read.push(self.read(staging, layout)?);
        }
        let mut reply = Reply {
            adapter: self.adapter.clone(),
            ..Reply::default()
        };
        let mut read = read.into_iter();
        if outputs.is_some() {
            let words = read.next().ok_or("no results read")?;
            let per = w.output_words() as usize;
            for k in 0..invocations as usize {
                let at = &words[k * per..(k + 1) * per];
                let returned = at[0] == 1;
                let mut results = Vec::new();
                if returned {
                    let mut i = 1;
                    for &n in &w.results {
                        results.push(at[i..i + n as usize].to_vec());
                        i += n as usize;
                    }
                }
                reply.invocations.push(Invocation { returned, results });
            }
        }
        for (r, words) in readbacks.iter().zip(read) {
            let at = match r {
                Readback::Buffer { at, .. } | Readback::Texture { at, .. } => *at,
            };
            reply.resources.push(Written {
                group: at.group,
                binding: at.binding,
                element: at.element,
                words,
            });
        }
        Ok(reply)
    }

    /// The harness's argument words, every invocation's in order, each checked against the entry point's arguments.
    fn inputs(&self, job: &Job, w: &Wrapped) -> Result<Vec<u32>, String> {
        let mut words = Vec::new();
        for (k, invocation) in job.invocations.iter().enumerate() {
            if invocation.len() != w.arguments.len() {
                return Err(format!(
                    "invocation {k} gives {} arguments, the entry point takes {}",
                    invocation.len(),
                    w.arguments.len()
                ));
            }
            for (a, (given, &want)) in invocation.iter().zip(&w.arguments).enumerate() {
                if given.len() != want as usize {
                    return Err(format!(
                        "invocation {k}'s argument {a} has {} words, its type takes {want}",
                        given.len()
                    ));
                }
                words.extend_from_slice(given);
            }
        }
        if words.is_empty() {
            words.push(0);
        }
        Ok(words)
    }

    fn buffer(&self, words: &[u32], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &bytes,
            usage,
        })
    }

    /// The layout entry and resource of a global of the program, from the job's resource: a binding array's elements,
    /// or the one resource. A writable one is noted to read back.
    #[allow(clippy::too_many_arguments)]
    fn resource(
        &self,
        module: &naga::Module,
        g: &naga::GlobalVariable,
        group: u32,
        binding: u32,
        visibility: wgpu::ShaderStages,
        kind: &ResourceKind,
        readbacks: &mut Vec<Readback>,
    ) -> Result<(wgpu::BindGroupLayoutEntry, Bound), String> {
        let name = g.name.as_deref().unwrap_or("unnamed");
        let at = |element| At {
            group,
            binding,
            element,
        };
        match (&module.types[g.ty].inner, kind) {
            (TypeInner::BindingArray { base, size }, ResourceKind::Array(elements)) => {
                let given =
                    u32::try_from(elements.len()).map_err(|_| format!("{name}: too many elements"))?;
                let count = match size {
                    ArraySize::Constant(n) if n.get() == given => *n,
                    ArraySize::Constant(n) => {
                        return Err(format!("{name}: a binding array of {n} given {given} elements"));
                    }
                    ArraySize::Dynamic => NonZeroU32::new(given)
                        .ok_or_else(|| format!("{name}: a binding array given no elements"))?,
                    ArraySize::Pending(_) => {
                        return Err(format!("{name}: a binding array of an override's size"));
                    }
                };
                let mut ty = None;
                let mut ones = Vec::new();
                for (k, e) in (0..).zip(elements) {
                    let (t, one) = self.element(module, name, g.space, *base, e, at(Some(k)), readbacks)?;
                    if ty.is_some_and(|ty| ty != t) {
                        return Err(format!("{name}: element {k} binds as {t:?}, the first as {ty:?}"));
                    }
                    ty = Some(t);
                    ones.push(one);
                }
                let ty = ty.ok_or_else(|| format!("{name}: a binding array given no elements"))?;
                // wgpu panics, rather than failing validation, on an array its device has no feature for
                let needs = match ty {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { .. },
                        ..
                    } => {
                        wgpu::Features::BUFFER_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                    }
                    wgpu::BindingType::Buffer { .. } => wgpu::Features::BUFFER_BINDING_ARRAY,
                    wgpu::BindingType::StorageTexture { .. } => {
                        wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                    }
                    _ => wgpu::Features::TEXTURE_BINDING_ARRAY,
                };
                if !self.features.contains(needs) {
                    return Err(format!(
                        "{name}: a binding array of {ty:?} needs {:?}, which the device does not have",
                        needs.difference(self.features)
                    ));
                }
                Ok((
                    layout_entry(binding, visibility, ty, Some(count)),
                    Bound::Array(ones),
                ))
            }
            (TypeInner::BindingArray { .. }, _) => Err(format!(
                "{name}: a binding array given one resource, not an array"
            )),
            (_, ResourceKind::Array(_)) => Err(format!("{name}: one resource given an array")),
            _ => {
                let (ty, one) = self.element(module, name, g.space, g.ty, kind, at(None), readbacks)?;
                Ok((layout_entry(binding, visibility, ty, None), Bound::One(one)))
            }
        }
    }

    /// The binding type and resource of one resource (a global, or an element of a binding array) of type `ty` in
    /// `space`, from the job's; a writable one is noted to read back.
    #[allow(clippy::too_many_arguments)]
    fn element(
        &self,
        module: &naga::Module,
        name: &str,
        space: AddressSpace,
        ty: Handle<naga::Type>,
        kind: &ResourceKind,
        at: At,
        readbacks: &mut Vec<Readback>,
    ) -> Result<(wgpu::BindingType, One), String> {
        let inner = &module.types[ty].inner;
        match (space, inner, kind) {
            (AddressSpace::Uniform, _, ResourceKind::Buffer { words }) => {
                let size = inner.size(module.to_ctx()) as usize;
                let mut words = words.clone();
                words.resize(size.div_ceil(4).max(1), 0);
                let buffer = self.buffer(&words, wgpu::BufferUsages::UNIFORM);
                Ok((buffer_type(wgpu::BufferBindingType::Uniform), One::Buffer(buffer)))
            }
            (AddressSpace::Storage { access }, _, ResourceKind::Buffer { words }) => {
                let read_only = !access.contains(StorageAccess::STORE);
                let words = if words.is_empty() { vec![0] } else { words.clone() };
                let buffer = self.buffer(&words, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
                if !read_only {
                    readbacks.push(Readback::Buffer {
                        buffer: buffer.clone(),
                        at,
                    });
                }
                Ok((
                    buffer_type(wgpu::BufferBindingType::Storage { read_only }),
                    One::Buffer(buffer),
                ))
            }
            (AddressSpace::Handle, TypeInner::Sampler { comparison }, ResourceKind::Sampler { compare }) => {
                let compare = match (comparison, compare.as_deref()) {
                    (false, _) => None,
                    (true, f) => Some(compare_function(f.unwrap_or("less"))?),
                };
                let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: None,
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Nearest,
                    min_filter: wgpu::FilterMode::Nearest,
                    mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                    compare,
                    ..Default::default()
                });
                // every sampler filters to the nearest texel: bound as one that does not filter, it may gather an
                // integer texture too (a float texture is bound unfilterable to match)
                let ty = if *comparison {
                    wgpu::SamplerBindingType::Comparison
                } else {
                    wgpu::SamplerBindingType::NonFiltering
                };
                Ok((wgpu::BindingType::Sampler(ty), One::Sampler(sampler)))
            }
            (
                AddressSpace::Handle,
                TypeInner::Image { dim, arrayed, class },
                ResourceKind::Texture {
                    texel,
                    size,
                    layers,
                    samples,
                    per_sample,
                },
            ) => self.texture(
                name,
                *dim,
                *arrayed,
                *class,
                (*texel, per_sample.as_deref()),
                *size,
                (*layers, *samples),
                at,
                readbacks,
            ),
            (space, inner, kind) => Err(format!(
                "@group({}) @binding({}){} {name}: a {space:?} {inner:?} given {}",
                at.group,
                at.binding,
                at.element.map(|k| format!(" element {k}")).unwrap_or_default(),
                match kind {
                    ResourceKind::Buffer { .. } => "a buffer",
                    ResourceKind::Texture { .. } => "a texture",
                    ResourceKind::Sampler { .. } => "a sampler",
                    ResourceKind::Array(_) => "an array",
                }
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn texture(
        &self,
        name: &str,
        dim: ImageDimension,
        arrayed: bool,
        class: ImageClass,
        (texel, per_sample): ([u32; 4], Option<&[[u32; 4]]>),
        size: Option<[u32; 3]>,
        (layers, samples): (Option<u32>, Option<u32>),
        at: At,
        readbacks: &mut Vec<Readback>,
    ) -> Result<(wgpu::BindingType, One), String> {
        let multi = matches!(
            class,
            ImageClass::Sampled { multi: true, .. } | ImageClass::Depth { multi: true }
        );
        if multi {
            return self.multisampled(name, dim, arrayed, class, texel, per_sample, size, samples);
        }
        if samples.is_some() || per_sample.is_some() {
            return Err(format!(
                "{name}: samples given for a texture that is not multisampled"
            ));
        }
        let [width, height, depth] = size.unwrap_or([4, 4, 4]);
        let (dimension, view_dimension, extent) = match (dim, arrayed) {
            (ImageDimension::D1, false) => (
                wgpu::TextureDimension::D1,
                wgpu::TextureViewDimension::D1,
                [width, 1, 1],
            ),
            (ImageDimension::D2, false) => (
                wgpu::TextureDimension::D2,
                wgpu::TextureViewDimension::D2,
                [width, height, 1],
            ),
            (ImageDimension::D2, true) => (
                wgpu::TextureDimension::D2,
                wgpu::TextureViewDimension::D2Array,
                [width, height, layers.unwrap_or(2)],
            ),
            (ImageDimension::D3, false) => (
                wgpu::TextureDimension::D3,
                wgpu::TextureViewDimension::D3,
                [width, height, depth],
            ),
            (ImageDimension::Cube, false) => (
                wgpu::TextureDimension::D2,
                wgpu::TextureViewDimension::Cube,
                [width, width, 6],
            ),
            (ImageDimension::Cube, true) => (
                wgpu::TextureDimension::D2,
                wgpu::TextureViewDimension::CubeArray,
                [width, width, layers.unwrap_or(12)],
            ),
            (d, a) => {
                return Err(format!(
                    "{name}: a {d:?} texture{}",
                    if a { " array" } else { "" }
                ));
            }
        };
        let extent = wgpu::Extent3d {
            width: extent[0],
            height: extent[1],
            depth_or_array_layers: extent[2],
        };
        let (format, binding_type, lanes, storage) = match class {
            ImageClass::Sampled { kind, .. } => {
                let (format, sample_type) = match kind {
                    ScalarKind::Float => (
                        wgpu::TextureFormat::Rgba32Float,
                        wgpu::TextureSampleType::Float { filterable: false },
                    ),
                    ScalarKind::Sint => (wgpu::TextureFormat::Rgba32Sint, wgpu::TextureSampleType::Sint),
                    ScalarKind::Uint => (wgpu::TextureFormat::Rgba32Uint, wgpu::TextureSampleType::Uint),
                    other => return Err(format!("{name}: a texture of {other:?}")),
                };
                (
                    format,
                    wgpu::BindingType::Texture {
                        sample_type,
                        view_dimension,
                        multisampled: false,
                    },
                    4,
                    false,
                )
            }
            ImageClass::Depth { .. } => (
                wgpu::TextureFormat::Depth32Float,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension,
                    multisampled: false,
                },
                1,
                false,
            ),
            ImageClass::Storage { format, access } => {
                let (f, lanes) =
                    storage_format(format).ok_or_else(|| format!("{name}: storage format {format:?}"))?;
                let access = if access.contains(StorageAccess::ATOMIC) {
                    wgpu::StorageTextureAccess::Atomic
                } else if access.contains(StorageAccess::LOAD | StorageAccess::STORE) {
                    wgpu::StorageTextureAccess::ReadWrite
                } else if access.contains(StorageAccess::STORE) {
                    wgpu::StorageTextureAccess::WriteOnly
                } else {
                    wgpu::StorageTextureAccess::ReadOnly
                };
                (
                    f,
                    wgpu::BindingType::StorageTexture {
                        access,
                        format: f,
                        view_dimension,
                    },
                    lanes,
                    true,
                )
            }
            ImageClass::External => return Err(format!("{name}: an external texture")),
        };
        let depth_format = format == wgpu::TextureFormat::Depth32Float;
        let mut usage = wgpu::TextureUsages::TEXTURE_BINDING;
        if depth_format {
            usage |= wgpu::TextureUsages::RENDER_ATTACHMENT;
        } else {
            usage |= wgpu::TextureUsages::COPY_DST;
        }
        if storage {
            usage = wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(name),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension,
            format,
            usage,
            view_formats: &[],
        });
        if depth_format {
            // a depth texture cannot be written from a buffer: each layer is cleared to the depth
            let depth = f32::from_bits(texel[0]);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            for layer in 0..extent.depth_or_array_layers {
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(depth),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            self.queue.submit([encoder.finish()]);
        } else {
            let per_layer = (extent.width * extent.height) as usize;
            let row: Vec<u8> = texel[..lanes as usize]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            let data: Vec<u8> = row.repeat(per_layer * extent.depth_or_array_layers as usize);
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(extent.width * 4 * lanes),
                    rows_per_image: Some(extent.height),
                },
                extent,
            );
        }
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(view_dimension),
            ..Default::default()
        });
        if storage {
            readbacks.push(Readback::Texture {
                texture: texture.clone(),
                lanes,
                at,
            });
        }
        Ok((binding_type, One::Texture(texture, view)))
    }

    /// A multisampled 2D texture (colour in a 32-bit format, or depth): made, then drawn into once with a full-screen
    /// triangle whose fragment shader runs for each sample and writes the sample's own texel, since a multisampled texture
    /// cannot be uploaded to. `texel` is every sample's unless `per_sample` gives each its own.
    #[allow(clippy::too_many_arguments)]
    fn multisampled(
        &self,
        name: &str,
        dim: ImageDimension,
        arrayed: bool,
        class: ImageClass,
        texel: [u32; 4],
        per_sample: Option<&[[u32; 4]]>,
        size: Option<[u32; 3]>,
        samples: Option<u32>,
    ) -> Result<(wgpu::BindingType, One), String> {
        if dim != ImageDimension::D2 || arrayed {
            return Err(format!(
                "{name}: a multisampled {dim:?} texture{} (only a 2D one that is not arrayed)",
                if arrayed { " array" } else { "" }
            ));
        }
        let [width, height, _] = size.unwrap_or([4, 4, 4]);
        let count = samples.unwrap_or(4);
        let (format, sample_type, fragment) = match class {
            ImageClass::Sampled {
                kind: ScalarKind::Float,
                ..
            } => (
                wgpu::TextureFormat::Rgba32Float,
                wgpu::TextureSampleType::Float { filterable: false },
                "@location(0) vec4<f32> { return bitcast<vec4<f32>>(texels.v[s]); }",
            ),
            ImageClass::Sampled {
                kind: ScalarKind::Sint,
                ..
            } => (
                wgpu::TextureFormat::Rgba32Sint,
                wgpu::TextureSampleType::Sint,
                "@location(0) vec4<i32> { return bitcast<vec4<i32>>(texels.v[s]); }",
            ),
            ImageClass::Sampled {
                kind: ScalarKind::Uint,
                ..
            } => (
                wgpu::TextureFormat::Rgba32Uint,
                wgpu::TextureSampleType::Uint,
                "@location(0) vec4<u32> { return texels.v[s]; }",
            ),
            ImageClass::Depth { .. } => (
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureSampleType::Depth,
                "@builtin(frag_depth) f32 { return bitcast<f32>(texels.v[s].x); }",
            ),
            other => return Err(format!("{name}: a multisampled {other:?} texture")),
        };
        let depth = format == wgpu::TextureFormat::Depth32Float;
        if count < 2 {
            return Err(format!(
                "{name}: {count} samples, where a multisampled texture has at least 2"
            ));
        }
        let flags = self.handle.get_texture_format_features(format).flags;
        if !flags.sample_count_supported(count) {
            let supported: Vec<u32> = [1, 2, 4, 8, 16]
                .into_iter()
                .filter(|n| flags.sample_count_supported(*n))
                .collect();
            return Err(format!(
                "{name}: {count} samples of {format:?}, where the adapter supports {supported:?}"
            ));
        }
        let texels: Vec<[u32; 4]> = match per_sample {
            Some(each) if each.len() == count as usize => each.to_vec(),
            Some(each) => {
                return Err(format!(
                    "{name}: {} per-sample texels for {count} samples",
                    each.len()
                ));
            }
            None => vec![texel; count as usize],
        };
        let mut words = [0u32; 64];
        for (slot, word) in words.iter_mut().zip(texels.iter().flatten()) {
            *slot = *word;
        }
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(name),
            size: extent,
            mip_level_count: 1,
            sample_count: count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniform = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sample texels"),
            contents: &words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>(),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let source = format!(
            "struct Texels {{ v: array<vec4<u32>, 16> }}
@group(0) @binding(0) var<uniform> texels: Texels;
@fragment
fn main(@builtin(sample_index) s: u32) -> {fragment}
"
        );
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sample fill"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[layout_entry(
                    0,
                    wgpu::ShaderStages::FRAGMENT,
                    buffer_type(wgpu::BufferBindingType::Uniform),
                    None,
                )],
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let target = [Some(wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sample fill"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("triangle"),
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(TRIANGLE)),
                    }),
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: depth.then(|| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    targets: if depth { &[] } else { &target },
                }),
                multiview_mask: None,
                cache: None,
            });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let colour = [Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: if depth { &[] } else { &colour },
                depth_stencil_attachment: depth.then_some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        Ok((
            wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: true,
            },
            One::Texture(texture, view),
        ))
    }

    /// A buffer's contents copied to one the host can map, and how to read it.
    fn copy_buffer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
    ) -> (wgpu::Buffer, Layout) {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        (staging, Layout::Words)
    }

    /// A texture's texels copied to a buffer the host can map (rows padded to 256 bytes), and how to read it.
    fn copy_texture(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        lanes: u32,
    ) -> (wgpu::Buffer, Layout) {
        let size = texture.size();
        let row = size.width * 4 * lanes;
        let padded = row.div_ceil(256) * 256;
        let rows = size.height * size.depth_or_array_layers;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(padded * rows),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );
        (staging, Layout::Rows { row, padded, rows })
    }

    /// The words of a mapped copy.
    fn read(&self, staging: &wgpu::Buffer, layout: &Layout) -> Result<Vec<u32>, Failure> {
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(self.timeout),
        }) {
            Ok(_) => {}
            Err(wgpu::PollError::Timeout) => {
                return Err(Failure::TimedOut(format!(
                    "the GPU did not finish within {} s",
                    self.timeout.as_secs()
                )));
            }
            Err(e) => return Err(Failure::Error(format!("waiting for the GPU: {e}"))),
        }
        let bytes = slice
            .get_mapped_range()
            .map_err(|e| Failure::Error(format!("mapping a copy: {e}")))?;
        let words = |b: &[u8]| -> Vec<u32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect()
        };
        let out = match layout {
            Layout::Words => words(&bytes),
            Layout::Rows { row, padded, rows } => {
                let mut out = Vec::new();
                for r in 0..*rows as usize {
                    let at = r * *padded as usize;
                    out.extend(words(&bytes[at..at + *row as usize]));
                }
                out
            }
        };
        drop(bytes);
        staging.unmap();
        Ok(out)
    }
}

/// How a mapped copy holds its words.
enum Layout {
    Words,
    Rows { row: u32, padded: u32, rows: u32 },
}

/// Where a resource is: its group and binding, and its element of a binding array.
#[derive(Clone, Copy)]
struct At {
    group: u32,
    binding: u32,
    element: Option<u32>,
}

/// A writable resource to read back after the run.
enum Readback {
    Buffer {
        buffer: wgpu::Buffer,
        at: At,
    },
    Texture {
        texture: wgpu::Texture,
        lanes: u32,
        at: At,
    },
}

/// One resource.
enum One {
    Buffer(wgpu::Buffer),
    #[allow(dead_code, reason = "the texture lives as long as its view is bound")]
    Texture(wgpu::Texture, wgpu::TextureView),
    Sampler(wgpu::Sampler),
}

impl One {
    fn resource(&self) -> wgpu::BindingResource<'_> {
        match self {
            One::Buffer(buffer) => buffer.as_entire_binding(),
            One::Texture(_, view) => wgpu::BindingResource::TextureView(view),
            One::Sampler(s) => wgpu::BindingResource::Sampler(s),
        }
    }
}

/// What a binding holds: one resource, or a binding array's elements (all of one kind).
enum Bound {
    One(One),
    Array(Vec<One>),
}

/// A binding array's elements as the slices a bind group entry takes.
#[derive(Default)]
struct Elements<'a> {
    buffers: Vec<wgpu::BufferBinding<'a>>,
    views: Vec<&'a wgpu::TextureView>,
    samplers: Vec<&'a wgpu::Sampler>,
}

impl<'a> Elements<'a> {
    fn of(bound: &'a Bound) -> Elements<'a> {
        let mut e = Elements::default();
        if let Bound::Array(ones) = bound {
            for one in ones {
                match one {
                    One::Buffer(b) => e.buffers.push(b.as_entire_buffer_binding()),
                    One::Texture(_, view) => e.views.push(view),
                    One::Sampler(s) => e.samplers.push(s),
                }
            }
        }
        e
    }

    fn resource(&self) -> wgpu::BindingResource<'_> {
        if !self.views.is_empty() {
            wgpu::BindingResource::TextureViewArray(&self.views)
        } else if !self.samplers.is_empty() {
            wgpu::BindingResource::SamplerArray(&self.samplers)
        } else {
            wgpu::BindingResource::BufferArray(&self.buffers)
        }
    }
}

fn buffer_type(ty: wgpu::BufferBindingType) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}

fn layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
    count: Option<NonZeroU32>,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count,
    }
}

fn compare_function(name: &str) -> Result<wgpu::CompareFunction, String> {
    Ok(match name {
        "never" => wgpu::CompareFunction::Never,
        "less" => wgpu::CompareFunction::Less,
        "less_equal" => wgpu::CompareFunction::LessEqual,
        "greater" => wgpu::CompareFunction::Greater,
        "greater_equal" => wgpu::CompareFunction::GreaterEqual,
        "equal" => wgpu::CompareFunction::Equal,
        "not_equal" => wgpu::CompareFunction::NotEqual,
        "always" => wgpu::CompareFunction::Always,
        other => return Err(format!("comparison {other}")),
    })
}

/// The texture format of a storage format, and its lanes; `None` for one this harness does not make.
fn storage_format(f: StorageFormat) -> Option<(wgpu::TextureFormat, u32)> {
    Some(match f {
        StorageFormat::R32Uint => (wgpu::TextureFormat::R32Uint, 1),
        StorageFormat::R32Sint => (wgpu::TextureFormat::R32Sint, 1),
        StorageFormat::R32Float => (wgpu::TextureFormat::R32Float, 1),
        StorageFormat::Rg32Uint => (wgpu::TextureFormat::Rg32Uint, 2),
        StorageFormat::Rg32Sint => (wgpu::TextureFormat::Rg32Sint, 2),
        StorageFormat::Rg32Float => (wgpu::TextureFormat::Rg32Float, 2),
        StorageFormat::Rgba32Uint => (wgpu::TextureFormat::Rgba32Uint, 4),
        StorageFormat::Rgba32Sint => (wgpu::TextureFormat::Rgba32Sint, 4),
        StorageFormat::Rgba32Float => (wgpu::TextureFormat::Rgba32Float, 4),
        _ => return None,
    })
}
