//! A Direct3D device on the CPU for running test suites: resources, samplers, constant buffers and descriptor ranges
//! as a suite's runner declares them, answering a lifted program's requests through `iridia-exec`'s two vocabularies:
//! [`SlotRequests`] (`iridia_exec::Sampler`, by bind slot, for the vertex and pixel programs `iridia_exec::draw` runs)
//! and [`IdRequests`] (`iridia_exec::Backend`, by the program's resource ids, for compute dispatches).
//!
//! Reads outside a resource return 0 (D3D11.3 22.4.6, 22.4.12); a resource nothing is bound to reads 0 too (7.18.17).
//! Writes outside one are dropped (5.3.11.3, "out of bounds writes ... do nothing").
pub mod format;
pub mod resource;
pub mod sample;

use std::collections::{BTreeMap, BTreeSet};

use iridia::{D3dClass, Interface, ResourceKind, StorageTarget, TextureDim};
use iridia_exec::{
    Atomic, AtomicAdd, BufferAtomic, Dim as ExecDim, Element, ImageAtomic, ImageAtomicOp, ImageLoad,
    ImageLod, ImageQuery, ImageSample, ImageStore, Load, QueryKind, Store,
};

use crate::vkd3d::cases::Descriptor;
use resource::{Dim, Kind, Resource};
use sample::{Lod, Request, Sampler};

/// The device's state: its resources by kind and slot, samplers by slot, constant buffers by slot (32-bit words), and
/// the descriptor ranges that say which resource a register reads.
#[derive(Clone, Debug, Default)]
pub struct Device {
    /// Resources by kind and slot.
    pub resources: BTreeMap<(Kind, u32), Resource>,
    /// Samplers by slot.
    pub samplers: BTreeMap<u32, Sampler>,
    /// Constant buffers by slot, as words.
    pub constants: BTreeMap<u32, Vec<u32>>,
    /// The descriptor ranges; `None` for the default (register `n` of space 0 reads slot `n`).
    pub descriptors: Option<Vec<Descriptor>>,
}

/// The class of view a descriptor range names.
fn range_kind(d: &Descriptor) -> Kind {
    if d.kind == "uav" { Kind::Uav } else { Kind::Srv }
}

/// Which resource a register of a class and space reads.
fn resolve(descriptors: &Option<Vec<Descriptor>>, kind: Kind, register: u32, space: u32) -> Option<u32> {
    match descriptors {
        None => (space == 0).then_some(register),
        Some(ranges) => ranges.iter().find_map(|d| {
            (range_kind(d) == kind
                && d.space == space
                && (d.register..d.register.saturating_add(d.count)).contains(&register))
            .then(|| d.target + (register - d.register))
        }),
    }
}

/// The resource kind of a D3D register class, for views.
fn view_kind(class: D3dClass) -> Option<Kind> {
    match class {
        D3dClass::ShaderResource => Some(Kind::Srv),
        D3dClass::UnorderedAccess => Some(Kind::Uav),
        _ => None,
    }
}

impl Device {
    /// The rows of constant buffer `slot`, `rows` of them, unwritten words 0.
    pub fn constant_rows(&self, slot: u32, rows: usize) -> Vec<[u32; 4]> {
        let words = self.constants.get(&slot).map(Vec::as_slice).unwrap_or(&[]);
        (0..rows)
            .map(|r| [0, 1, 2, 3].map(|l| words.get(r * 4 + l).copied().unwrap_or(0)))
            .collect()
    }

    /// The last register of a class and space a resource is bound at.
    pub fn last_register(&self, kind: Kind, space: u32) -> Option<u32> {
        match &self.descriptors {
            None => (space == 0)
                .then(|| {
                    self.resources
                        .keys()
                        .filter(|(k, _)| *k == kind)
                        .map(|(_, slot)| *slot)
                        .max()
                })
                .flatten(),
            Some(ranges) => ranges
                .iter()
                .filter(|d| range_kind(d) == kind && d.space == space)
                .filter_map(|d| d.count.checked_sub(1).map(|n| d.register.saturating_add(n)))
                .max(),
        }
    }

    /// The views programs with these interfaces can read, by kind and slot: every element of each of their arrays of
    /// views that a register resolves to (an array of no stated size as far as the last register of its class and
    /// space anything is bound at).
    pub fn views_read(&self, interfaces: &[&Interface]) -> BTreeSet<(Kind, u32)> {
        let mut read = BTreeSet::new();
        for r in interfaces.iter().flat_map(|i| &i.resources) {
            let Some(d) = r.provenance.d3d else { continue };
            let Some(kind) = view_kind(d.class) else { continue };
            let end = self
                .last_register(kind, d.space)
                .map_or(0, |last| last.saturating_sub(d.register).saturating_add(1));
            for k in 0..r.count.min(end) {
                if let Some(slot) = resolve(&self.descriptors, kind, d.register.saturating_add(k), d.space) {
                    read.insert((kind, slot));
                }
            }
        }
        read
    }

    /// The resource a view register reads.
    pub fn view(&self, kind: Kind, register: u32, space: u32) -> Option<&Resource> {
        let slot = resolve(&self.descriptors, kind, register, space)?;
        self.resources.get(&(kind, slot))
    }

    fn view_mut(&mut self, kind: Kind, register: u32, space: u32) -> Option<&mut Resource> {
        let slot = resolve(&self.descriptors, kind, register, space)?;
        self.resources.get_mut(&(kind, slot))
    }
}

/// The dimension a request names, and whether it is a 1D texture.
fn one_d(dim: ExecDim) -> bool {
    dim == ExecDim::D1
}

fn lod_of(lod: ImageLod) -> Lod {
    match lod {
        ImageLod::Implicit => Lod::None(0.0),
        ImageLod::Bias(b) => Lod::None(b),
        ImageLod::Level(l) => Lod::Level(l),
        ImageLod::Gradients(g) => Lod::Gradients(g, 0.0),
        ImageLod::GradientsBias(g, b) => Lod::Gradients(g, b),
    }
}

/// A sample of `r` through sampler `s` as an executor asks it.
fn sample_of(r: &Resource, s: &Sampler, req: &ImageSample<u32>) -> [f32; 4] {
    let request = Request {
        coord: req.coord,
        layer: req.layer,
        lod: lod_of(req.lod),
        offset: req.offset,
        gather: req.gather,
        reference: req.depth_ref,
    };
    let mut s = *s;
    if req.depth_ref.is_none() && matches!(s.reduction, sample::Reduction::Comparison(_)) {
        // a comparison sampler read without a reference (`Sample` on a comparison sampler) reads the texels
        s.reduction = sample::Reduction::Standard;
    }
    sample::sample(r, &s, &request, one_d(req.dim))
}

/// A size, level-count, layer-count or sample-count query of `r` (D3D11.3 22.4.14, `resinfo`).
fn query_of(r: Option<&Resource>, kind: QueryKind, level: Option<u32>, dim: ExecDim) -> [u32; 4] {
    let Some(r) = r else {
        return [0; 4];
    };
    match kind {
        QueryKind::Levels => [r.levels, 0, 0, 0],
        QueryKind::Samples => [r.samples, 0, 0, 0],
        QueryKind::Layers => [
            if r.dim == Dim::Cube {
                r.layers / 6
            } else {
                r.layers
            },
            0,
            0,
            0,
        ],
        QueryKind::Size => {
            let level = level.unwrap_or(0);
            if level >= r.levels {
                return [0; 4];
            }
            let [w, h, d] = r.level_dims(level);
            match dim {
                ExecDim::D1 => [w, 0, 0, 0],
                ExecDim::D2 | ExecDim::Cube => [w, h, 0, 0],
                ExecDim::D3 => [w, h, d, 0],
            }
        }
    }
}

/// An atomic operation on the 32-bit word at byte `at` of `r` (`compare` the comparand of a compare-exchange): the word
/// as it was. An operation past the end (`at` is `None`) changes nothing and returns 0, D3D's rule.
fn atomic_on(
    r: &mut Resource,
    at: Option<usize>,
    op: ImageAtomicOp,
    signed: bool,
    value: u32,
    compare: u32,
) -> u32 {
    let Some(at) = at else {
        return 0;
    };
    let old = r.word(at);
    r.set_word(at, op.apply(old, value, compare, signed));
    old
}

/// A store of `value`'s lanes `mask` marks to a texel or buffer element of a resource: typed through its format, a
/// structured or raw buffer's consecutive words from the byte address. A structured buffer's element (`address[0]`) is
/// `stride` bytes, the stride of the program that addresses it.
fn store_to(
    r: &mut Resource,
    stride: Option<u32>,
    address: [u32; 3],
    layer: Option<u32>,
    value: [u32; 4],
    mask: u8,
) {
    if r.dim == Dim::Buffer && (stride.is_some() || r.raw) {
        let base = match stride {
            Some(stride) => address[0] as usize * stride as usize + address[1] as usize,
            None => address[0] as usize * 4,
        };
        for (k, word) in value.into_iter().enumerate() {
            if mask >> k & 1 == 1 {
                r.set_word(base + k * 4, word);
            }
        }
        return;
    }
    let layer = layer.unwrap_or(if r.dim == Dim::D3 { 0 } else { address[2] });
    let z = if r.dim == Dim::D3 { address[2] } else { 0 };
    let at = [i64::from(address[0]), i64::from(address[1]), i64::from(z)];
    if let Some(offset) = r.texel_offset(0, layer, at) {
        let size = r.format.size as usize;
        let format = r.format;
        if let Some(bytes) = r.data.get_mut(offset..offset + size) {
            format.write(value, mask, bytes);
        }
    }
}

/// The word a 32-bit atomic on a texel or buffer element operates on: its byte offset. A structured buffer's element
/// (`address[0]`) is `stride` bytes, as in [`store_to`].
fn atomic_offset(r: &Resource, stride: Option<u32>, address: [u32; 3], layer: Option<u32>) -> Option<usize> {
    if r.dim == Dim::Buffer {
        return Some(match (stride, r.raw) {
            (Some(stride), _) => address[0] as usize * stride as usize + address[1] as usize,
            (None, true) => address[0] as usize * 4,
            (None, false) => address[0] as usize * r.format.size as usize,
        })
        .filter(|&o| o + 4 <= r.data.len());
    }
    let layer = layer.unwrap_or(if r.dim == Dim::D3 { 0 } else { address[2] });
    let z = if r.dim == Dim::D3 { address[2] } else { 0 };
    r.texel_offset(
        0,
        layer,
        [i64::from(address[0]), i64::from(address[1]), i64::from(z)],
    )
}

/// The requests of the vertex and pixel programs of a draw, in `iridia_exec::Sampler`'s bind-slot vocabulary. A slot
/// names a register of whichever class the programs bind there: `classes` says which, for the requests (a buffer
/// read, a size) that do not.
#[derive(Debug)]
pub struct SlotRequests<'a> {
    /// The device.
    pub device: &'a mut Device,
    /// The view class each slot the programs bind a buffer at reads.
    classes: BTreeMap<u32, Kind>,
    /// The structure stride of each slot the programs bind a structured buffer at. Direct3D steps through a structured
    /// buffer by the stride the program declares, whatever the view's (vkd3d's srv-structuredbuffer reads a 16-byte
    /// view as 32-byte elements); the view's stride stands only where no program states one.
    strides: BTreeMap<u32, u32>,
}

impl<'a> SlotRequests<'a> {
    /// The requests on `device` of programs with these interfaces; programs that bind a shader resource and an
    /// unordered-access view at one slot number cannot be told apart in this vocabulary, and are refused, as are two
    /// structure strides at one slot.
    pub fn new(device: &'a mut Device, interfaces: &[&Interface]) -> Result<Self, String> {
        let mut classes = BTreeMap::new();
        let mut strides = BTreeMap::new();
        for i in interfaces {
            for r in &i.resources {
                let Some(d) = r.provenance.d3d else { continue };
                if d.space != 0 {
                    return Err(format!(
                        "{} is in register space {}, which the bind-slot requests of a draw do not name",
                        r.provenance.name, d.space
                    ));
                }
                let Some(kind) = view_kind(d.class) else { continue };
                if let Some(old) = classes.insert(d.register, kind)
                    && old != kind
                {
                    return Err(format!(
                        "t{0} and u{0} are both bound, which the bind-slot requests of a draw cannot tell apart",
                        d.register
                    ));
                }
                let (ResourceKind::Buffer(iridia::BufferKind::Structured { stride })
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(iridia::BufferKind::Structured { stride }),
                    ..
                }) = r.kind
                else {
                    continue;
                };
                if let Some(old) = strides.insert(d.register, stride)
                    && old != stride
                {
                    return Err(format!(
                        "slot {} is read as structures of {old} and of {stride} bytes",
                        d.register
                    ));
                }
            }
        }
        Ok(SlotRequests {
            device,
            classes,
            strides,
        })
    }

    fn kind(&self, slot: u32) -> Kind {
        self.classes.get(&slot).copied().unwrap_or(Kind::Srv)
    }

    /// The structure stride the programs address the buffer at `slot` by: their own, else the view's.
    fn stride(&self, slot: u32, view: &Resource) -> Option<u32> {
        self.strides.get(&slot).copied().or(view.stride)
    }
}

impl iridia_exec::Sampler for SlotRequests<'_> {
    fn sample(&mut self, request: iridia_exec::Sample) -> Result<[f32; 4], String> {
        self.sample_image(ImageSample {
            texture: request.texture,
            sampler: request.sampler,
            dim: ExecDim::D2,
            coord: [request.uv[0], request.uv[1], 0.0],
            layer: request.layer,
            lod: request.lod.into(),
            offset: [0; 3],
            gather: None,
            depth_ref: None,
            depth: false,
        })
    }

    fn sample_image(&mut self, request: ImageSample<u32>) -> Result<[f32; 4], String> {
        let s = self
            .device
            .samplers
            .get(&request.sampler)
            .copied()
            .ok_or_else(|| format!("s{}: no sampler declared", request.sampler))?;
        Ok(match self.device.view(Kind::Srv, request.texture, 0) {
            Some(r) => sample_of(r, &s, &request),
            None => [0.0; 4],
        })
    }

    fn load(&mut self, request: Load) -> Result<[u32; 4], String> {
        let kind = self.kind(request.texture);
        let Some(r) = self.device.view(kind, request.texture, 0) else {
            return Ok([0; 4]);
        };
        if r.dim == Dim::Buffer {
            let x = i64::from(request.texel[0]);
            return Ok(r.texel(0, 0, [x, 0, 0]).map_or([0; 4], |t| r.mapped(t)));
        }
        Ok(sample::load(r, request.texel, None, request.mip, None, [0; 3]))
    }

    fn load_image(&mut self, request: ImageLoad<u32>) -> Result<[u32; 4], String> {
        let kind = if request.storage { Kind::Uav } else { Kind::Srv };
        let Some(r) = self.device.view(kind, request.texture, 0) else {
            return Ok([0; 4]);
        };
        if r.dim == Dim::Buffer {
            let x = i64::from(request.texel[0]);
            return Ok(r.texel(0, 0, [x, 0, 0]).map_or([0; 4], |t| r.mapped(t)));
        }
        let (texel, layer) = if r.dim == Dim::D3 {
            (request.texel, None)
        } else {
            (
                [request.texel[0], request.texel[1], 0],
                request.layer.or(Some(request.texel[2])),
            )
        };
        Ok(sample::load(
            r,
            texel,
            layer,
            request.mip.unwrap_or(0),
            request.sample,
            [0; 3],
        ))
    }

    fn query_image(&mut self, request: ImageQuery<u32>) -> Result<[u32; 4], String> {
        let kind = self.kind(request.texture);
        Ok(query_of(
            self.device.view(kind, request.texture, 0),
            request.kind,
            request.level,
            request.dim,
        ))
    }

    fn store(&mut self, request: Store) -> Result<(), String> {
        let program = self.strides.get(&request.view).copied();
        if let Some(r) = self.device.view_mut(Kind::Uav, request.view, 0) {
            let stride = program.or(r.stride);
            store_to(r, stride, request.address, None, request.value, request.mask);
        }
        Ok(())
    }

    fn store_image(&mut self, request: ImageStore<u32>) -> Result<(), String> {
        let mask = (1u8 << request.lanes.min(4)) - 1;
        if let Some(r) = self.device.view_mut(Kind::Uav, request.view, 0) {
            store_to(r, r.stride, request.texel, request.layer, request.value, mask);
        }
        Ok(())
    }

    fn atomic_add(&mut self, request: AtomicAdd) -> Result<u32, String> {
        self.atomic(Atomic {
            view: request.view,
            address: request.address,
            op: ImageAtomicOp::Add,
            signed: false,
            value: request.value,
            compare: 0,
        })
    }

    fn atomic(&mut self, request: Atomic) -> Result<u32, String> {
        let program = self.strides.get(&request.view).copied();
        let Some(r) = self.device.view_mut(Kind::Uav, request.view, 0) else {
            return Ok(0);
        };
        let at = atomic_offset(r, program.or(r.stride), request.address, None);
        Ok(atomic_on(
            r,
            at,
            request.op,
            request.signed,
            request.value,
            request.compare,
        ))
    }

    fn atomic_image(&mut self, request: ImageAtomic<u32>) -> Result<u32, String> {
        let Some(r) = self.device.view_mut(Kind::Uav, request.view, 0) else {
            return Ok(0);
        };
        let at = atomic_offset(r, r.stride, request.texel, request.layer);
        Ok(atomic_on(
            r,
            at,
            request.op,
            request.signed,
            request.value,
            request.compare,
        ))
    }

    fn structured(&mut self, slot: u32, element: u32, offset: u32) -> Result<u32, String> {
        let kind = self.kind(slot);
        let Some(r) = self.device.view(kind, slot, 0) else {
            return Ok(0);
        };
        let at = match self.stride(slot, r) {
            Some(stride) => element as usize * stride as usize + offset as usize,
            None => element as usize * 4 + offset as usize,
        };
        Ok(r.word(at))
    }

    fn buffer_elements(&mut self, slot: u32) -> Option<u32> {
        let kind = self.kind(slot);
        Some(self.device.view(kind, slot, 0).map_or(0, |r| {
            let bytes = r.width.saturating_mul(r.element_size());
            match self.stride(slot, r) {
                // as many of the program's structures as the view's bytes hold
                Some(stride) => bytes / stride.max(1),
                None if r.raw => bytes / 4,
                None => r.width,
            }
        }))
    }
}

/// Where a program's resource is on the device.
#[derive(Clone, Copy, Debug)]
struct Binding {
    kind: Kind,
    /// The register of the first element.
    register: u32,
    space: u32,
    /// How many elements (consecutive registers): 1 for a view that is not an array, [`iridia::Resource::UNBOUNDED`]
    /// for an array of no stated size.
    count: u32,
    /// Whether the executor holds the buffer's words: its reads and writes do not reach the device until the
    /// dispatch ends.
    held: Held,
}

/// How the executor holds a view's words for a dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// It does not: the device answers each read and write (images, and typed buffers outside arrays).
    No,
    /// The buffer's bytes as words (structured and raw buffers).
    Words,
    /// Four 32-bit lanes per element, through the view's format (a typed buffer in an array of them: the executor
    /// holds an array's elements, and carries a typed element as four lanes).
    Lanes,
}

/// The requests of a compute (or mesh) program, in `iridia_exec::Backend`'s vocabulary of the program's resources.
#[derive(Debug)]
pub struct IdRequests<'a> {
    /// The device.
    pub device: &'a mut Device,
    bindings: BTreeMap<iridia::ResourceId, Binding>,
    /// Samplers by resource: the register of the first element, how many.
    samplers: BTreeMap<iridia::ResourceId, (u32, u32)>,
    /// The hidden counters the program updates (a raw buffer of one word each), by resource: the view they count.
    counters: BTreeMap<iridia::ResourceId, iridia::ResourceId>,
}

impl<'a> IdRequests<'a> {
    /// The requests of a program with `interface` on `device`.
    pub fn new(device: &'a mut Device, interface: &Interface) -> Result<Self, String> {
        let mut bindings = BTreeMap::new();
        let mut samplers = BTreeMap::new();
        let mut counters = BTreeMap::new();
        for r in &interface.resources {
            let Some(d) = r.provenance.d3d else {
                if let Some(view) = r.provenance.counter_of {
                    counters.insert(r.id, view);
                    continue;
                }
                return Err(format!("{} has no D3D binding", r.provenance.name));
            };
            match (&r.kind, view_kind(d.class)) {
                (ResourceKind::Sampler { .. }, _) => {
                    samplers.insert(r.id, (d.register, r.count));
                }
                (ResourceKind::ConstantBuffer { .. }, _) => {}
                (kind, Some(view)) => {
                    let buffer = match kind {
                        ResourceKind::Buffer(b)
                        | ResourceKind::Storage {
                            target: StorageTarget::Buffer(b),
                            ..
                        } => Some(b),
                        _ => None,
                    };
                    let held = match buffer {
                        Some(iridia::BufferKind::Structured { .. } | iridia::BufferKind::Raw) => Held::Words,
                        Some(iridia::BufferKind::Typed(_)) if r.count != 1 => Held::Lanes,
                        _ => Held::No,
                    };
                    bindings.insert(
                        r.id,
                        Binding {
                            kind: view,
                            register: d.register,
                            space: d.space,
                            count: r.count,
                            held,
                        },
                    );
                }
                _ => {}
            }
        }
        for (counter, view) in &counters {
            if !bindings.get(view).is_some_and(|b| b.kind == Kind::Uav) {
                return Err(format!(
                    "resource {} counts resource {}, not an unordered-access view",
                    counter.0, view.0
                ));
            }
        }
        Ok(IdRequests {
            device,
            bindings,
            samplers,
            counters,
        })
    }

    /// The words of every structured and raw buffer the program reads and writes, for the executor to hold, and of
    /// every hidden counter it updates (the count of each element of the view it counts, 0 for one nothing is
    /// bound to).
    pub fn held_words(&self) -> BTreeMap<iridia::ResourceId, Vec<u32>> {
        let mut out = BTreeMap::new();
        for (counter, view) in &self.counters {
            let Some(b) = self.bindings.get(view) else {
                continue;
            };
            let counts = (0..self.reach(b))
                .map(|k| self.view_of(b, k).map_or(0, Resource::count))
                .collect();
            out.insert(*counter, counts);
        }
        for (id, b) in &self.bindings {
            if b.held == Held::Words && b.count == 1 {
                out.insert(*id, self.words_of(b, 0).unwrap_or_default());
            }
        }
        out
    }

    /// The words of every element of every array of structured and raw buffers the program reads and writes, by
    /// resource and element, for the executor to hold: the elements a resource is bound to (an element no descriptor
    /// names is missing, and reading it stops the run, as an unbound descriptor is undefined in D3D12).
    pub fn held_elements(&self) -> BTreeMap<(iridia::ResourceId, u32), Vec<u32>> {
        let mut out = BTreeMap::new();
        for (id, b) in &self.bindings {
            if b.held != Held::No && b.count != 1 {
                for k in self.elements_of(b) {
                    if let Some(words) = self.words_of(b, k) {
                        out.insert((*id, k), words);
                    }
                }
            }
        }
        out
    }

    /// The elements of an array binding a resource is bound to: an array of no stated size reaches as far as the
    /// last register of its class and space anything is bound at.
    fn elements_of(&self, b: &Binding) -> impl Iterator<Item = u32> + '_ {
        let b = *b;
        (0..self.reach(&b)).filter(move |k| self.view_of(&b, *k).is_some())
    }

    /// How many elements of a binding there can be views at: its count, or for an array of no stated size as far as
    /// the last register of its class and space anything is bound at.
    fn reach(&self, b: &Binding) -> u32 {
        let end = self
            .device
            .last_register(b.kind, b.space)
            .map_or(0, |last| last.saturating_sub(b.register).saturating_add(1));
        b.count.min(end)
    }

    /// The words the executor holds of element `element` of a binding: the bytes as words, or each typed element's
    /// four lanes.
    fn words_of(&self, b: &Binding, element: u32) -> Option<Vec<u32>> {
        let r = self.view_of(b, element)?;
        Some(match b.held {
            Held::Lanes => (0..r.width)
                .flat_map(|i| {
                    sample::load(
                        r,
                        [i32::try_from(i).unwrap_or(i32::MAX), 0, 0],
                        None,
                        0,
                        None,
                        [0; 3],
                    )
                })
                .collect(),
            Held::Words | Held::No => r
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect(),
        })
    }

    /// Put the words the executor held back into the unordered-access views they came from.
    pub fn give_back(
        &mut self,
        words: &BTreeMap<iridia::ResourceId, Vec<u32>>,
        elements: &BTreeMap<(iridia::ResourceId, u32), Vec<u32>>,
    ) {
        let all = words
            .iter()
            .map(|(id, w)| ((*id, 0), w))
            .chain(elements.iter().map(|(k, w)| (*k, w)));
        for ((id, element), w) in all {
            let Some(b) = self.bindings.get(&id).copied() else {
                continue;
            };
            if b.kind != Kind::Uav {
                continue;
            }
            let Some(r) = self.view_mut(&b, element) else {
                continue;
            };
            match b.held {
                Held::Lanes => {
                    for (i, lanes) in (0u32..).zip(w.as_chunks::<4>().0) {
                        store_to(r, None, [i, 0, 0], None, *lanes, 0xf);
                    }
                }
                Held::Words | Held::No => {
                    for (k, word) in w.iter().enumerate() {
                        r.set_word(k * 4, *word);
                    }
                }
            }
        }
        // after the buffers: a counter in its view's own buffer is the word the counter updates left
        for (counter, view) in &self.counters.clone() {
            let (Some(counts), Some(b)) = (words.get(counter), self.bindings.get(view).copied()) else {
                continue;
            };
            for (k, n) in (0u32..).zip(counts) {
                if let Some(r) = self.view_mut(&b, k) {
                    r.set_count(*n);
                }
            }
        }
    }

    fn binding(&self, id: iridia::ResourceId) -> Result<Binding, String> {
        self.bindings
            .get(&id)
            .copied()
            .ok_or_else(|| format!("resource {} is not a view", id.0))
    }

    /// Element `element` of a binding: the view at the register that many past its first; none past its end.
    fn view_of(&self, b: &Binding, element: u32) -> Option<&Resource> {
        if element >= b.count {
            return None;
        }
        self.device
            .view(b.kind, b.register.checked_add(element)?, b.space)
    }

    fn view_mut(&mut self, b: &Binding, element: u32) -> Option<&mut Resource> {
        if element >= b.count {
            return None;
        }
        self.device
            .view_mut(b.kind, b.register.checked_add(element)?, b.space)
    }

    fn resource(&self, id: iridia::ResourceId) -> Result<Option<&Resource>, String> {
        self.element(Element { id, index: 0 })
    }

    fn element(&self, e: Element) -> Result<Option<&Resource>, String> {
        let b = self.binding(e.id)?;
        Ok(self.view_of(&b, e.index))
    }

    fn element_mut(&mut self, id: iridia::ResourceId, index: u32) -> Result<Option<&mut Resource>, String> {
        let b = self.binding(id)?;
        Ok(self.view_mut(&b, index))
    }

    fn resource_mut(&mut self, id: iridia::ResourceId) -> Result<Option<&mut Resource>, String> {
        self.element_mut(id, 0)
    }
}

impl iridia_exec::Backend for IdRequests<'_> {
    fn sample(&mut self, r: ImageSample<Element>) -> Result<[f32; 4], String> {
        let (first, count) = *self
            .samplers
            .get(&r.sampler.id)
            .ok_or_else(|| format!("resource {} is not a sampler", r.sampler.id.0))?;
        if r.sampler.index >= count {
            return Err(format!(
                "sampler {}: element {} of an array of {count}",
                r.sampler.id.0, r.sampler.index
            ));
        }
        let slot = first.saturating_add(r.sampler.index);
        let s = self
            .device
            .samplers
            .get(&slot)
            .copied()
            .ok_or_else(|| format!("s{slot}: no sampler declared"))?;
        let request = ImageSample {
            texture: 0,
            sampler: 0,
            dim: r.dim,
            coord: r.coord,
            layer: r.layer,
            lod: r.lod,
            offset: r.offset,
            gather: r.gather,
            depth_ref: r.depth_ref,
            depth: r.depth,
        };
        Ok(match self.element(r.texture)? {
            Some(res) => sample_of(res, &s, &request),
            None => [0.0; 4],
        })
    }

    fn load(&mut self, r: ImageLoad<Element>) -> Result<[u32; 4], String> {
        let Some(res) = self.element(r.texture)? else {
            return Ok([0; 4]);
        };
        if res.dim == Dim::Buffer {
            return Ok(res
                .texel(0, 0, [i64::from(r.texel[0]), 0, 0])
                .map_or([0; 4], |t| res.mapped(t)));
        }
        let (texel, layer) = if res.dim == Dim::D3 {
            (r.texel, None)
        } else {
            ([r.texel[0], r.texel[1], 0], r.layer.or(Some(r.texel[2])))
        };
        Ok(sample::load(
            res,
            texel,
            layer,
            r.mip.unwrap_or(0),
            r.sample,
            [0; 3],
        ))
    }

    fn query(&mut self, r: ImageQuery<Element>) -> Result<[u32; 4], String> {
        Ok(query_of(self.element(r.texture)?, r.kind, r.level, r.dim))
    }

    fn store_image(&mut self, r: ImageStore<iridia::ResourceId>) -> Result<(), String> {
        let mask = (1u8 << r.lanes.min(4)) - 1;
        if let Some(res) = self.element_mut(r.view, r.index)? {
            store_to(res, res.stride, r.texel, r.layer, r.value, mask);
        }
        Ok(())
    }

    fn atomic_image(&mut self, r: ImageAtomic<iridia::ResourceId>) -> Result<u32, String> {
        let Some(res) = self.element_mut(r.view, r.index)? else {
            return Ok(0);
        };
        let at = atomic_offset(res, res.stride, r.texel, r.layer);
        Ok(atomic_on(res, at, r.op, r.signed, r.value, r.compare))
    }

    fn store_buffer(
        &mut self,
        view: iridia::ResourceId,
        word: u32,
        value: [u32; 4],
        mask: u8,
    ) -> Result<(), String> {
        let Some(res) = self.resource_mut(view)? else {
            return Ok(());
        };
        // a typed buffer's element is four lanes of the executor's words: word `word + k` is lane (word + k) % 4 of
        // element (word + k) / 4
        for (k, v) in value.into_iter().enumerate() {
            if mask >> k & 1 == 1 {
                let w = word.saturating_add(k as u32);
                let mut lanes = [0; 4];
                lanes[(w % 4) as usize] = v;
                store_to(res, None, [w / 4, 0, 0], None, lanes, 1 << (w % 4));
            }
        }
        Ok(())
    }

    fn read_buffer(&mut self, view: iridia::ResourceId, word: u32, count: u8) -> Result<[u32; 4], String> {
        let Some(res) = self.resource(view)? else {
            return Ok([0; 4]);
        };
        let mut out = [0u32; 4];
        for (k, slot) in out.iter_mut().enumerate().take(usize::from(count)) {
            let w = word + k as u32;
            let lanes = res
                .texel(0, 0, [i64::from(w / 4), 0, 0])
                .map_or([0; 4], |t| res.mapped(t));
            *slot = lanes[(w % 4) as usize];
        }
        Ok(out)
    }

    fn buffer_words(&mut self, view: iridia::ResourceId) -> Result<u32, String> {
        Ok(self.resource(view)?.map_or(0, |r| r.width.saturating_mul(4)))
    }

    fn atomic_add(&mut self, view: iridia::ResourceId, word: u32, value: u32) -> Result<u32, String> {
        self.atomic_buffer(BufferAtomic {
            view: Element { id: view, index: 0 },
            word,
            op: ImageAtomicOp::Add,
            signed: false,
            value,
            compare: 0,
        })
    }

    fn atomic_buffer(&mut self, r: BufferAtomic) -> Result<u32, String> {
        let Some(res) = self.element_mut(r.view.id, r.view.index)? else {
            return Ok(0);
        };
        let at = atomic_offset(res, None, [r.word / 4, 0, 0], None);
        Ok(atomic_on(res, at, r.op, r.signed, r.value, r.compare))
    }
}

/// The constant buffers of a program with `interface`, as `rows(register, space, rows)` gives each register's:
/// an array's elements at consecutive registers, the ones `rows` has.
pub fn constant_buffers(
    interface: &Interface,
    rows: impl Fn(u32, u32, usize) -> Option<Vec<[u32; 4]>>,
) -> iridia_exec::Resources {
    let mut out = iridia_exec::Resources::default();
    for r in &interface.resources {
        let (ResourceKind::ConstantBuffer { layout }, Some(d)) = (&r.kind, r.provenance.d3d) else {
            continue;
        };
        let n = (layout.size as usize).div_ceil(16).max(1);
        if r.count == 1 {
            out.constants.insert(
                r.id,
                rows(d.register, d.space, n).unwrap_or_else(|| vec![[0; 4]; n]),
            );
            continue;
        }
        // an array of no stated size: as far as the registers `rows` answers for, in the 64K D3D allows
        for k in 0..r.count.min(1 << 16) {
            match d.register.checked_add(k).and_then(|reg| rows(reg, d.space, n)) {
                Some(words) => {
                    out.constant_elements.insert((r.id, k), words);
                }
                None if r.count == iridia::Resource::UNBOUNDED => break,
                // an element of a stated array nothing is bound to reads 0, as an unbound constant buffer does
                None => {
                    out.constant_elements.insert((r.id, k), vec![[0; 4]; n]);
                }
            }
        }
    }
    out
}

/// Run `groups` thread groups of a compute program with `interface` on `device`, its constant buffers
/// `resources`: structured and raw buffers (and arrays of them) and hidden counters held by the executor for the
/// dispatch and put back into the device's unordered-access views after it.
pub fn dispatch(
    exe: &iridia_exec::Executable,
    interface: &Interface,
    device: &mut Device,
    resources: &iridia_exec::Resources,
    groups: [u32; 3],
    wave: u32,
) -> Result<(), Dispatch> {
    let bound = exe.bind_resources(resources).map_err(Dispatch::Failed)?;
    let mut requests = IdRequests::new(device, interface).map_err(Dispatch::Unbindable)?;
    let mut held = requests.held_words();
    let mut elements = requests.held_elements();
    let run = exe.dispatch_elements(&bound, &mut requests, &mut held, &mut elements, groups, wave);
    // what the groups that ran wrote is the device's, as a GPU's writes before a fault are
    requests.give_back(&held, &elements);
    run.map(|_| ()).map_err(Dispatch::Failed)
}

/// Why a dispatch did not run to its end.
#[derive(Clone, Debug)]
pub enum Dispatch {
    /// The device cannot bind the program's interface.
    Unbindable(String),
    /// Binding or running it failed.
    Failed(String),
}

/// The dimension of a texture resource kind, for the size queries a program states.
pub fn texture_dim(dim: TextureDim) -> ExecDim {
    match dim {
        TextureDim::D1 => ExecDim::D1,
        TextureDim::D2 => ExecDim::D2,
        TextureDim::D3 => ExecDim::D3,
        TextureDim::Cube => ExecDim::Cube,
    }
}
