//! A program's root signature (the container's `RTS0` part, version 1.0 or 1.1: d3d12.h's `D3D12_ROOT_SIGNATURE_DESC`
//! as serialised): the parameters, in order, and the static samplers, which say which registers the offloader's
//! descriptor tables, root constants and root descriptors fill.

/// A descriptor range's class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeKind {
    /// Shader resource views.
    Srv,
    /// Unordered-access views.
    Uav,
    /// Constant-buffer views.
    Cbv,
    /// Samplers.
    Sampler,
}

/// A range of a descriptor table.
#[derive(Clone, Copy, Debug)]
pub struct Range {
    /// Its class.
    pub kind: RangeKind,
    /// How many descriptors; `u32::MAX` for unbounded.
    pub count: u32,
    /// The first register.
    pub base: u32,
    /// The register space.
    pub space: u32,
    /// The offset in the table of its first descriptor; `u32::MAX` to follow the previous range.
    pub offset: u32,
}

/// A root parameter.
#[derive(Clone, Debug)]
pub enum Parameter {
    /// A descriptor table.
    Table(Vec<Range>),
    /// 32-bit constants at a register: register, space, how many.
    Constants(u32, u32, u32),
    /// A root descriptor of a class at a register: class, register, space.
    Descriptor(RangeKind, u32, u32),
}

/// A static sampler (`D3D12_STATIC_SAMPLER_DESC`).
#[derive(Clone, Copy, Debug)]
#[allow(missing_docs, reason = "d3d12.h's names")]
pub struct StaticSampler {
    pub filter: u32,
    pub address: [u32; 3],
    pub mip_lod_bias: f32,
    pub max_anisotropy: u32,
    pub comparison: u32,
    pub border: u32,
    pub min_lod: f32,
    pub max_lod: f32,
    pub register: u32,
    pub space: u32,
}

/// A root signature.
#[derive(Clone, Debug, Default)]
pub struct RootSignature {
    /// The parameters, in order.
    pub parameters: Vec<Parameter>,
    /// The static samplers.
    pub samplers: Vec<StaticSampler>,
}

fn word(b: &[u8], at: usize) -> Result<u32, String> {
    b.get(at..at + 4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .ok_or_else(|| format!("RTS0: ends at byte {at}"))
}

/// The root signature a container carries, if it carries one.
pub fn of(container: &[u8]) -> Result<Option<RootSignature>, String> {
    let parts = dxcontainer::chunks(container).map_err(|e| format!("RTS0: {e}"))?;
    let Some(part) = parts.iter().find(|c| &c.kind == b"RTS0") else {
        return Ok(None);
    };
    let b: &[u8] = part.data;
    let version = word(b, 0)?;
    if version != 1 && version != 2 {
        return Err(format!("RTS0: version {version}"));
    }
    let (count, at) = (word(b, 4)? as usize, word(b, 8)? as usize);
    let (samplers, sampler_at) = (word(b, 12)? as usize, word(b, 16)? as usize);
    let mut out = RootSignature::default();
    let class = |t: u32| -> Result<RangeKind, String> {
        Ok(match t {
            0 => RangeKind::Srv,
            1 => RangeKind::Uav,
            2 => RangeKind::Cbv,
            3 => RangeKind::Sampler,
            _ => return Err(format!("RTS0: range type {t}")),
        })
    };
    for i in 0..count {
        let p = at + i * 12;
        let (kind, offset) = (word(b, p)?, word(b, p + 8)? as usize);
        out.parameters.push(match kind {
            0 => {
                let (n, ranges) = (word(b, offset)? as usize, word(b, offset + 4)? as usize);
                let size = if version == 1 { 20 } else { 24 };
                let mut list = Vec::with_capacity(n);
                for r in 0..n {
                    let q = ranges + r * size;
                    list.push(Range {
                        kind: class(word(b, q)?)?,
                        count: word(b, q + 4)?,
                        base: word(b, q + 8)?,
                        space: word(b, q + 12)?,
                        offset: word(b, q + if version == 1 { 16 } else { 20 })?,
                    });
                }
                Parameter::Table(list)
            }
            1 => Parameter::Constants(word(b, offset)?, word(b, offset + 4)?, word(b, offset + 8)?),
            2 => Parameter::Descriptor(RangeKind::Cbv, word(b, offset)?, word(b, offset + 4)?),
            3 => Parameter::Descriptor(RangeKind::Srv, word(b, offset)?, word(b, offset + 4)?),
            4 => Parameter::Descriptor(RangeKind::Uav, word(b, offset)?, word(b, offset + 4)?),
            t => return Err(format!("RTS0: parameter type {t}")),
        });
    }
    for i in 0..samplers {
        let q = sampler_at + i * 52;
        out.samplers.push(StaticSampler {
            filter: word(b, q)?,
            address: [word(b, q + 4)?, word(b, q + 8)?, word(b, q + 12)?],
            mip_lod_bias: f32::from_bits(word(b, q + 16)?),
            max_anisotropy: word(b, q + 20)?,
            comparison: word(b, q + 24)?,
            border: word(b, q + 28)?,
            min_lod: f32::from_bits(word(b, q + 32)?),
            max_lod: f32::from_bits(word(b, q + 36)?),
            register: word(b, q + 40)?,
            space: word(b, q + 44)?,
        });
    }
    Ok(Some(out))
}
