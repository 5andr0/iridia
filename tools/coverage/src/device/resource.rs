//! A resource as vkd3d's runner makes it: a buffer or a texture of some dimension, format, size, layers and levels,
//! its bytes in the runner's order (every layer of level 0, then of level 1, and so on; rows tightly packed).
use super::format::{Class, Format, named};
use crate::vkd3d::cases::ResourceDesc;

/// What a resource is bound as.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    /// A shader resource view (`t<n>`).
    Srv,
    /// An unordered-access view (`u<n>`).
    Uav,
    /// A render target (`SV_Target<n>`).
    Rtv,
    /// The depth-stencil target.
    Dsv,
    /// A vertex buffer.
    Vb,
}

impl Kind {
    /// The kind vkd3d's name for it names.
    pub fn named(name: &str) -> Result<Kind, String> {
        Ok(match name {
            "srv" => Kind::Srv,
            "uav" => Kind::Uav,
            "rtv" => Kind::Rtv,
            "dsv" => Kind::Dsv,
            "vb" => Kind::Vb,
            _ => return Err(format!("a resource of type {name}")),
        })
    }
}

/// A resource's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dim {
    /// A buffer.
    Buffer,
    /// A 2D texture, or an array of them.
    D2,
    /// A volume.
    D3,
    /// A cube map (six layers).
    Cube,
}

/// One component of a view's component mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Swizzle {
    /// Component `n` of the texel.
    Component(u8),
    /// Zero.
    Zero,
    /// One.
    One,
}

/// A resource.
#[derive(Clone, Debug)]
pub struct Resource {
    /// What it is bound as.
    pub kind: Kind,
    /// Its slot.
    pub slot: u32,
    /// Its format; `unknown` for a structured or raw buffer.
    pub format: Format,
    /// Its shape.
    pub dim: Dim,
    /// Width in texels; a buffer's in elements (bytes for a vertex buffer).
    pub width: u32,
    /// Height in texels.
    pub height: u32,
    /// Depth in texels.
    pub depth: u32,
    /// Array layers.
    pub layers: u32,
    /// Mip levels.
    pub levels: u32,
    /// Samples per texel.
    pub samples: u32,
    /// A structured buffer's element size in bytes.
    pub stride: Option<u32>,
    /// A raw (byte-address) buffer.
    pub raw: bool,
    /// The view's component mapping.
    pub swizzle: [Swizzle; 4],
    /// The least level of detail a sample reads.
    pub min_lod_clamp: f32,
    /// The bytes.
    pub data: Vec<u8>,
    /// An unordered-access view's hidden counter, if it has one.
    pub counter: Counter,
    /// A target the runner made itself.
    pub default: bool,
}

/// Where an unordered-access view's hidden counter is (`D3D12_BUFFER_UAV::CounterOffsetInBytes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Counter {
    /// The view has none.
    None,
    /// In a resource of its own, holding this count.
    Separate(u32),
    /// The word at this byte offset of the view's own resource (a D3D12 view whose counter resource is the buffer).
    At(usize),
}

/// `size` at mip level `level`: halved per level, at least 1.
pub fn level_size(size: u32, level: u32) -> u32 {
    (size >> level.min(31)).max(1)
}

impl Resource {
    /// The resource a case's description makes.
    pub fn new(desc: &ResourceDesc) -> Result<Resource, String> {
        let kind = Kind::named(&desc.kind)?;
        let format = match (&desc.format, desc.stride) {
            (_, Some(_)) => named("unknown")?,
            (Some(name), None) => named(name)?,
            (None, None) if kind == Kind::Vb => named("unknown")?,
            (None, None) => return Err("a resource with no format".into()),
        };
        let dim = match desc.dimension.as_deref() {
            Some("buffer") => Dim::Buffer,
            Some("2d") | None => Dim::D2,
            Some("3d") => Dim::D3,
            Some("cube") => Dim::Cube,
            Some(d) => return Err(format!("a resource of dimension {d}")),
        };
        let swizzle = match &desc.swizzle {
            None => [0, 1, 2, 3].map(Swizzle::Component),
            Some(s) => {
                let mut out = [Swizzle::Zero; 4];
                for (k, c) in s.chars().take(4).enumerate() {
                    out[k] = match c {
                        'r' => Swizzle::Component(0),
                        'g' => Swizzle::Component(1),
                        'b' => Swizzle::Component(2),
                        'a' => Swizzle::Component(3),
                        '0' => Swizzle::Zero,
                        '1' => Swizzle::One,
                        _ => return Err(format!("the component mapping {s}")),
                    };
                }
                out
            }
        };
        let mut r = Resource {
            kind,
            slot: desc.slot,
            format,
            dim,
            width: desc.width,
            height: desc.height.max(1),
            depth: desc.depth.max(1),
            layers: desc.layers.max(1),
            levels: desc.levels.max(1),
            samples: desc.samples.max(1),
            stride: desc.stride,
            raw: desc.raw,
            swizzle,
            min_lod_clamp: desc.min_lod_clamp.map_or(0.0, f32::from_bits),
            data: desc.data.iter().flat_map(|w| w.to_le_bytes()).collect(),
            // vkd3d's D3D12 runner makes the buffer its own counter resource, at offset 0
            counter: if desc.counter {
                Counter::At(0)
            } else {
                Counter::None
            },
            default: desc.default,
        };
        if kind == Kind::Vb {
            r.width = u32::try_from(r.data.len()).map_err(|_| "a vertex buffer past 4 GiB")?;
            return Ok(r);
        }
        if r.samples > 1 && r.levels > 1 {
            return Err("a multisampled texture of more than one level".into());
        }
        let size = r.total_bytes();
        if r.data.len() > size {
            return Err(format!("{} bytes of data for a resource of {size}", r.data.len()));
        }
        r.data.resize(size, 0);
        Ok(r)
    }

    /// Bytes per element: the format's texel, a structured buffer's stride, a raw buffer's word.
    pub fn element_size(&self) -> u32 {
        match (self.stride, self.raw) {
            (Some(s), _) => s,
            (None, true) => 4,
            (None, false) => self.format.size.max(1),
        }
    }

    /// How many bytes every subresource takes together.
    pub fn total_bytes(&self) -> usize {
        if self.dim == Dim::Buffer {
            return (self.width as usize).saturating_mul(self.element_size() as usize);
        }
        (0..self.levels)
            .map(|l| self.level_bytes(l).saturating_mul(self.layers as usize))
            .sum::<usize>()
    }

    /// The size of level `level`: width, height, depth.
    pub fn level_dims(&self, level: u32) -> [u32; 3] {
        [
            level_size(self.width, level),
            if matches!(self.dim, Dim::Buffer) {
                1
            } else {
                level_size(self.height, level)
            },
            if self.dim == Dim::D3 {
                level_size(self.depth, level)
            } else {
                1
            },
        ]
    }

    /// Bytes of one layer of level `level`.
    /// The bytes of one layer of level `level`: a multisampled texture's texel is its samples, together.
    fn level_bytes(&self, level: u32) -> usize {
        let [w, h, d] = self.level_dims(level);
        (w as usize) * (h as usize) * (d as usize) * (self.element_size() as usize) * (self.samples as usize)
    }

    /// The byte offset of subresource (`level`, `layer`): the runner's order is every layer of a level before the
    /// next level.
    pub fn subresource(&self, level: u32, layer: u32) -> usize {
        let before: usize = (0..level)
            .map(|l| self.level_bytes(l) * self.layers as usize)
            .sum();
        before + self.level_bytes(level) * layer as usize
    }

    /// The byte offset of texel (x, y, z) of subresource (`level`, `layer`), or `None` outside it; a multisampled
    /// texture's first sample.
    pub fn texel_offset(&self, level: u32, layer: u32, at: [i64; 3]) -> Option<usize> {
        self.sample_offset(level, layer, at, 0)
    }

    /// The byte offset of sample `sample` of texel (x, y, z) of subresource (`level`, `layer`), or `None` outside it.
    pub fn sample_offset(&self, level: u32, layer: u32, at: [i64; 3], sample: u32) -> Option<usize> {
        if level >= self.levels || layer >= self.layers || sample >= self.samples {
            return None;
        }
        let [w, h, d] = self.level_dims(level);
        let [x, y, z] = at;
        if x < 0 || y < 0 || z < 0 || x >= i64::from(w) || y >= i64::from(h) || z >= i64::from(d) {
            return None;
        }
        let size = self.element_size() as usize;
        let index = (z as usize * h as usize + y as usize) * w as usize + x as usize;
        let samples = self.samples as usize;
        Some(self.subresource(level, layer) + (index * samples + sample as usize) * size)
    }

    /// The resource resolved to one sample a texel, as vkd3d's runners resolve a multisampled target before they
    /// read it back (`ResolveSubresource`, Direct3D 9's `StretchRect`): each texel the average of its samples, a
    /// float or UNORM format's lanes as floats. An integer, typeless or depth format has no average and is refused.
    pub fn resolved(&self) -> Result<Resource, String> {
        if self.samples <= 1 {
            return Ok(self.clone());
        }
        let size = self.format.size as usize;
        let samples = self.samples as usize;
        let mut out = self.clone();
        out.samples = 1;
        out.data = vec![0; out.total_bytes()];
        for (texel, into) in self.data.chunks(size * samples).zip(out.data.chunks_mut(size)) {
            let read: Vec<[u32; 4]> = texel
                .chunks(size)
                .map(|sample| self.format.read(sample))
                .collect();
            self.format.write(self.format.resolve(&read)?, 0xf, into);
        }
        Ok(out)
    }

    /// The four lanes a view reads of sample `sample` of texel (x, y, z) of subresource (`level`, `layer`) before
    /// its component mapping; `None` outside it.
    pub fn sample_texel(&self, level: u32, layer: u32, at: [i64; 3], sample: u32) -> Option<[u32; 4]> {
        let at = self.sample_offset(level, layer, at, sample)?;
        let size = self.format.size as usize;
        self.data.get(at..at + size).map(|b| self.format.read(b))
    }

    /// The four lanes a view reads of texel (x, y, z) of subresource (`level`, `layer`) before its component mapping;
    /// `None` outside it.
    pub fn texel(&self, level: u32, layer: u32, at: [i64; 3]) -> Option<[u32; 4]> {
        let at = self.texel_offset(level, layer, at)?;
        let size = self.format.size as usize;
        self.data.get(at..at + size).map(|b| self.format.read(b))
    }

    /// The lanes of a texel through the view's component mapping.
    pub fn mapped(&self, lanes: [u32; 4]) -> [u32; 4] {
        let one = match self.format.class {
            Class::Uint | Class::Sint => 1,
            _ => 1f32.to_bits(),
        };
        self.swizzle.map(|s| match s {
            Swizzle::Component(c) => lanes[usize::from(c)],
            Swizzle::Zero => 0,
            Swizzle::One => one,
        })
    }

    /// The 32-bit word at byte `offset` of a buffer (0 past its end).
    pub fn word(&self, offset: usize) -> u32 {
        self.data
            .get(offset..offset + 4)
            .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// The view's hidden count; 0 for a view without a counter.
    pub fn count(&self) -> u32 {
        match self.counter {
            Counter::None => 0,
            Counter::Separate(n) => n,
            Counter::At(at) => self.word(at),
        }
    }

    /// Set the view's hidden count; nothing for a view without a counter.
    pub fn set_count(&mut self, n: u32) {
        match self.counter {
            Counter::None => {}
            Counter::Separate(_) => self.counter = Counter::Separate(n),
            Counter::At(at) => self.set_word(at, n),
        }
    }

    /// Write the 32-bit word at byte `offset` of a buffer (nothing past its end).
    pub fn set_word(&mut self, offset: usize, value: u32) {
        if let Some(b) = self.data.get_mut(offset..offset + 4) {
            b.copy_from_slice(&value.to_le_bytes());
        }
    }
}
