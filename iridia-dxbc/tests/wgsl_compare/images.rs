//! The images a compute program binds, as the harness binds them (tools/wgsl-run): every sampled texture uniform, a storage
//! texture holding the words its stores and atomics change, each 4 texels a side with one level (a 1D texture 4 wide, a 3D
//! one 4 deep, an array 2 layers, a cube 6, a cube array 12). An element of an array of images is an image of its own.
use std::collections::BTreeMap;

use iridia_exec::{
    Element, ImageAtomic, ImageAtomicOp, ImageLoad, ImageQuery, ImageSample, ImageStore, QueryKind,
};
use iridia_ir::interface::{ResourceId, TextureDim};

/// The shape of an image: its dimensionality and whether it is an array.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shape {
    pub dim: TextureDim,
    pub arrayed: bool,
}

impl Shape {
    /// Its width, its height and its slices: the layers of an array, the faces of a cube, the depth of a 3D texture, else 1.
    fn extent(self) -> [u32; 3] {
        let slices = match (self.dim, self.arrayed) {
            (TextureDim::Cube, false) => 6,
            (TextureDim::Cube, true) => 12,
            (TextureDim::D3, _) => 4,
            (_, true) => 2,
            _ => 1,
        };
        [4, if self.dim == TextureDim::D1 { 1 } else { 4 }, slices]
    }

    /// How many layers it has, as a query for them says.
    fn layers(self) -> u32 {
        match (self.dim, self.arrayed) {
            (TextureDim::Cube, false) => 6,
            (TextureDim::Cube, true) => 12,
            (_, true) => 2,
            _ => 1,
        }
    }

    /// How many texels it has.
    pub fn texels(self) -> usize {
        self.extent().iter().product::<u32>() as usize
    }

    /// The index of the texel at `at` (x, y, z) and `layer`, where it is inside the image.
    fn index(self, at: [i64; 3], layer: Option<i64>) -> Option<usize> {
        let [w, h, n] = self.extent().map(i64::from);
        let comps = match self.dim {
            TextureDim::D1 => 1,
            TextureDim::D2 => 2,
            TextureDim::D3 | TextureDim::Cube => 3,
        };
        if at[..comps]
            .iter()
            .zip([w, h, n])
            .any(|(a, e)| !(0..e).contains(a))
        {
            return None;
        }
        // an array's layer is a slice of its own; a 3D texture's depth is its slice; a cube's third coordinate is a face
        let slice = match (self.dim, self.arrayed) {
            (TextureDim::D3, _) => at[2],
            (TextureDim::Cube, _) => at[2],
            (_, true) => {
                let layer = layer?;
                if !(0..n).contains(&layer) {
                    return None;
                }
                layer
            }
            _ => 0,
        };
        Some(((slice * h + at[1].max(0)) * w + at[0]) as usize)
    }
}

/// A sampled texture: its uniform texel.
#[derive(Clone, Copy, Debug)]
struct Sampled {
    texel: [u32; 4],
    shape: Shape,
}

/// A storage texture: four lanes of each texel, in the host's order (a slice, then a row, then a column).
#[derive(Clone, Debug)]
struct Stored {
    texels: Vec<[u32; 4]>,
    shape: Shape,
}

#[derive(Clone, Default)]
pub(super) struct Images {
    sampled: BTreeMap<(ResourceId, u32), Sampled>,
    storage: BTreeMap<(ResourceId, u32), Stored>,
}

impl Images {
    pub fn add_sampled(&mut self, id: ResourceId, element: u32, texel: [u32; 4], shape: Shape) {
        self.sampled.insert((id, element), Sampled { texel, shape });
    }

    pub fn add_storage(&mut self, id: ResourceId, element: u32, texel: [u32; 4], shape: Shape) {
        let texels = vec![texel; shape.texels()];
        self.storage.insert((id, element), Stored { texels, shape });
    }

    /// The first `lanes` lanes of every texel of storage image `element` of `id`, as the host reads them back.
    pub fn words(&self, id: ResourceId, element: u32, lanes: usize) -> Option<Vec<u32>> {
        let s = self.storage.get(&(id, element))?;
        Some(s.texels.iter().flat_map(|t| t[..lanes].iter().copied()).collect())
    }

    /// Whether every storage image holds the texels it holds in `other`.
    pub fn storage_equals(&self, other: &Images) -> bool {
        self.storage.len() == other.storage.len()
            && self
                .storage
                .iter()
                .all(|(key, s)| other.storage.get(key).is_some_and(|o| o.texels == s.texels))
    }

    /// How many texels storage image `element` of `id` has.
    pub fn texels(&self, id: ResourceId, element: u32) -> Option<usize> {
        self.storage.get(&(id, element)).map(|s| s.texels.len())
    }

    fn shape(&self, e: Element) -> Option<Shape> {
        self.sampled
            .get(&(e.id, e.index))
            .map(|s| s.shape)
            .or_else(|| self.storage.get(&(e.id, e.index)).map(|s| s.shape))
    }

    pub fn sample(&mut self, r: ImageSample<Element>) -> Result<[f32; 4], String> {
        let t = self
            .sampled
            .get(&(r.texture.id, r.texture.index))
            .ok_or_else(|| format!("no texture {}[{}]", r.texture.id.0, r.texture.index))?;
        let c = t.texel.map(f32::from_bits);
        // a depth texture: its depth is the first lane, and a comparison sample gives 1 where `depth_ref <= depth`
        // (`wgsl_run::COMPARE`) and 0 elsewhere, in every lane of a gather as of the sample
        if r.depth || r.depth_ref.is_some() {
            let depth = c[0];
            let value = r
                .depth_ref
                .map_or(depth, |reference| if reference <= depth { 1.0 } else { 0.0 });
            return Ok([value; 4]);
        }
        Ok(match r.gather {
            Some(k) => [c[usize::from(k)]; 4],
            None => c,
        })
    }

    pub fn load(&mut self, r: ImageLoad<Element>) -> Result<[u32; 4], String> {
        if r.mip.unwrap_or(0) != 0 {
            return Ok([0; 4]);
        }
        let at = r.texel.map(i64::from);
        let layer = r.layer.map(i64::from);
        if r.storage {
            let s = self
                .storage
                .get(&(r.texture.id, r.texture.index))
                .ok_or_else(|| format!("no storage texture {}[{}]", r.texture.id.0, r.texture.index))?;
            return Ok(s.shape.index(at, layer).map_or([0; 4], |i| s.texels[i]));
        }
        let t = self
            .sampled
            .get(&(r.texture.id, r.texture.index))
            .ok_or_else(|| format!("no texture {}[{}]", r.texture.id.0, r.texture.index))?;
        Ok(if t.shape.index(at, layer).is_some() {
            t.texel
        } else {
            [0; 4]
        })
    }

    pub fn query(&mut self, r: ImageQuery<Element>) -> Result<[u32; 4], String> {
        let shape = self
            .shape(r.texture)
            .ok_or_else(|| format!("no image {}[{}]", r.texture.id.0, r.texture.index))?;
        Ok(match r.kind {
            QueryKind::Size if r.level.unwrap_or(0) > 0 => [0; 4],
            QueryKind::Size => [4, 4, 4, 0],
            QueryKind::Levels => [1, 0, 0, 0],
            QueryKind::Layers => [shape.layers(), 0, 0, 0],
            QueryKind::Samples => [1, 0, 0, 0],
        })
    }

    /// Store to the image `view` (the original's id, not a store observer's alias): the index of the texel written, where
    /// the store is inside the image.
    pub fn store(&mut self, r: ImageStore<ResourceId>) -> Result<Option<usize>, String> {
        let s = self
            .storage
            .get_mut(&(r.view, r.index))
            .ok_or_else(|| format!("no storage texture {}[{}]", r.view.0, r.index))?;
        // a store outside the image changes nothing
        let at = s.shape.index(r.texel.map(i64::from), r.layer.map(i64::from));
        if let Some(i) = at {
            s.texels[i][..usize::from(r.lanes)].copy_from_slice(&r.value[..usize::from(r.lanes)]);
        }
        Ok(at)
    }

    /// An atomic on the image `view`: the word it held, and the index of the texel, where it is inside the image.
    pub fn atomic(&mut self, r: ImageAtomic<ResourceId>) -> Result<(u32, Option<usize>), String> {
        let s = self
            .storage
            .get_mut(&(r.view, r.index))
            .ok_or_else(|| format!("no storage texture {}[{}]", r.view.0, r.index))?;
        let Some(i) = s.shape.index(r.texel.map(i64::from), r.layer.map(i64::from)) else {
            return Ok((0, None));
        };
        let old = s.texels[i][0];
        let new = match r.op {
            ImageAtomicOp::Add => old.wrapping_add(r.value),
            ImageAtomicOp::Sub => old.wrapping_sub(r.value),
            ImageAtomicOp::And => old & r.value,
            ImageAtomicOp::Or => old | r.value,
            ImageAtomicOp::Xor => old ^ r.value,
            ImageAtomicOp::Min if r.signed => (old as i32).min(r.value as i32) as u32,
            ImageAtomicOp::Min => old.min(r.value),
            ImageAtomicOp::Max if r.signed => (old as i32).max(r.value as i32) as u32,
            ImageAtomicOp::Max => old.max(r.value),
            ImageAtomicOp::Exchange => r.value,
            ImageAtomicOp::CompareExchange => {
                if old == r.compare {
                    r.value
                } else {
                    old
                }
            }
        };
        s.texels[i][0] = new;
        Ok((old, Some(i)))
    }
}
