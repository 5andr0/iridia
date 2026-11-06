//! The DXGI formats vkd3d's runner names: how many bytes a texel takes, what a shader reads of one (four lanes, as a
//! view returns them) and what a write stores. The conversions are the Direct3D 11.3 functional specification's
//! (section 3.2, "Data Conversion"): UNORM by `round(saturate(x) * (2^n - 1))` to nearest even, and back as `c / (2^n -
//! 1)`; an integer format of fewer bits than 32 takes the value's low bits; a missing component reads 0, and a missing
//! alpha 1 (section 5.3.4, "Default values for missing components").

/// How a format's lanes are read and written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// 32-bit floats.
    Float,
    /// Unsigned integers.
    Uint,
    /// Signed integers.
    Sint,
    /// Unsigned normalized integers.
    Unorm,
    /// No type: a structured or raw buffer's words, or a typeless 32-bit texel read as it is stored.
    Typeless,
    /// A depth format: one 32-bit float, and with a stencil a byte in the next word.
    Depth,
}

/// A format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    /// vkd3d's name for it.
    pub name: &'static str,
    /// Bytes per texel.
    pub size: u32,
    /// Components stored.
    pub components: u8,
    /// Bits per component.
    pub bits: u8,
    /// How they are read.
    pub class: Class,
    /// The components in memory are blue, green, red, alpha.
    pub bgra: bool,
    /// A depth format sampled for comparison (vkd3d's `r32-float-shadow`).
    pub shadow: bool,
}

const fn f(name: &'static str, size: u32, components: u8, bits: u8, class: Class) -> Format {
    Format {
        name,
        size,
        components,
        bits,
        class,
        bgra: false,
        shadow: false,
    }
}

/// Every format vkd3d's shader runner names (tests/shader_runner.c, `format_info`).
pub const FORMATS: [Format; 19] = [
    f("unknown", 0, 1, 32, Class::Typeless),
    f("r32g32b32a32-float", 16, 4, 32, Class::Float),
    f("r32g32b32a32-uint", 16, 4, 32, Class::Uint),
    f("r32g32b32a32-sint", 16, 4, 32, Class::Sint),
    f("r32g32b32-float", 12, 3, 32, Class::Float),
    f("r32g32-float", 8, 2, 32, Class::Float),
    f("r32g32-uint", 8, 2, 32, Class::Uint),
    f("r32g32-sint", 8, 2, 32, Class::Sint),
    f("d32-float-s8x24-uint", 8, 2, 32, Class::Depth),
    // the stencil of a d32-float-s8x24-uint texture, read as a uint in green
    f("x32-typeless-g8x24-uint", 8, 2, 32, Class::Depth),
    f("r8g8b8a8-unorm", 4, 4, 8, Class::Unorm),
    f("r8g8b8a8-uint", 4, 4, 8, Class::Uint),
    f("r32-typeless", 4, 1, 32, Class::Typeless),
    Format {
        shadow: true,
        ..f("r32-float-shadow", 4, 1, 32, Class::Float)
    },
    f("r32-float", 4, 1, 32, Class::Float),
    f("r32-uint", 4, 1, 32, Class::Uint),
    f("r32-sint", 4, 1, 32, Class::Sint),
    f("r8-uint", 1, 1, 8, Class::Uint),
    Format {
        bgra: true,
        ..f("b8g8r8a8-unorm", 4, 4, 8, Class::Unorm)
    },
];

/// The depth formats a depth-stencil target takes.
pub const D32: Format = f("d32-float", 4, 1, 32, Class::Depth);

/// The format vkd3d names `name`.
pub fn named(name: &str) -> Result<Format, String> {
    if name == D32.name {
        return Ok(D32);
    }
    FORMATS
        .iter()
        .find(|f| f.name == name)
        .copied()
        .ok_or_else(|| format!("the format {name}, which this runner does not know"))
}

/// `x` to an `n`-bit UNORM value: saturated, scaled and rounded to nearest even; NaN is 0 (3.2.3.6).
fn unorm(x: f32, n: u8) -> u32 {
    let max = ((1u64 << n) - 1) as f64;
    if x.is_nan() {
        return 0;
    }
    let v = f64::from(x).clamp(0.0, 1.0) * max;
    let r = v.round_ties_even();
    r as u32
}

impl Format {
    /// The four lanes a view reads of the texel `bytes` (as many as [`Format::size`]): f32 bits for a float or UNORM
    /// format, the integers for an integer one, each missing component 0 and a missing alpha 1.
    pub fn read(&self, bytes: &[u8]) -> [u32; 4] {
        let word = |i: usize| -> u32 {
            bytes
                .get(i * 4..i * 4 + 4)
                .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let one = match self.class {
            Class::Float | Class::Unorm | Class::Depth => 1f32.to_bits(),
            _ => 1,
        };
        let mut lanes = [0, 0, 0, one];
        match (self.class, self.bits) {
            (Class::Depth, _) if self.name == "x32-typeless-g8x24-uint" => {
                // the stencil, in green
                lanes = [0, word(1) & 0xff, 0, 1];
            }
            (_, 32) => {
                for (k, lane) in lanes.iter_mut().enumerate().take(usize::from(self.components)) {
                    *lane = word(k);
                }
            }
            (Class::Unorm, 8) => {
                for (k, lane) in lanes.iter_mut().enumerate().take(usize::from(self.components)) {
                    *lane = (f32::from(bytes.get(k).copied().unwrap_or(0)) / 255.0).to_bits();
                }
                if self.bgra {
                    lanes.swap(0, 2);
                }
            }
            (_, 8) => {
                for (k, lane) in lanes.iter_mut().enumerate().take(usize::from(self.components)) {
                    *lane = u32::from(bytes.get(k).copied().unwrap_or(0));
                }
            }
            _ => {}
        }
        lanes
    }

    /// The bytes a write of `lanes` (as a shader writes them: f32 bits or integers) stores in a texel, the lanes `mask`
    /// marks over `old`'s.
    pub fn write(&self, lanes: [u32; 4], mask: u8, old: &mut [u8]) {
        let mut lanes = lanes;
        if self.bgra {
            lanes.swap(0, 2);
        }
        let mask = if self.bgra {
            (mask & 0b1010) | ((mask & 1) << 2) | ((mask >> 2) & 1)
        } else {
            mask
        };
        for (k, lane) in lanes.into_iter().enumerate().take(usize::from(self.components)) {
            if mask >> k & 1 == 0 {
                continue;
            }
            match (self.class, self.bits) {
                (_, 32) => {
                    if let Some(b) = old.get_mut(k * 4..k * 4 + 4) {
                        b.copy_from_slice(&lane.to_le_bytes());
                    }
                }
                (Class::Unorm, n) => {
                    if let Some(b) = old.get_mut(k) {
                        *b = unorm(f32::from_bits(lane), n) as u8;
                    }
                }
                (_, 8) => {
                    if let Some(b) = old.get_mut(k) {
                        *b = lane as u8;
                    }
                }
                _ => {}
            }
        }
    }

    /// The scalar a sampler reads of a depth or shadow texel: its first lane as a float.
    pub fn depth(&self, bytes: &[u8]) -> f32 {
        f32::from_bits(self.read(bytes)[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unorm_rounds_to_nearest_even_and_saturates() {
        let fmt = named("r8g8b8a8-unorm").unwrap();
        let mut texel = [0u8; 4];
        fmt.write(
            [
                0.5f32.to_bits(),
                2.0f32.to_bits(),
                (-1f32).to_bits(),
                f32::NAN.to_bits(),
            ],
            0xf,
            &mut texel,
        );
        // 0.5 * 255 = 127.5, to even 128
        assert_eq!(texel, [128, 255, 0, 0]);
        assert_eq!(f32::from_bits(fmt.read(&texel)[0]), 128.0 / 255.0);
    }

    #[test]
    fn missing_components_read_zero_and_alpha_one() {
        let fmt = named("r32-uint").unwrap();
        assert_eq!(fmt.read(&7u32.to_le_bytes()), [7, 0, 0, 1]);
        let fmt = named("r32g32-float").unwrap();
        let bytes: Vec<u8> = [2f32, 3f32].iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(
            fmt.read(&bytes),
            [2f32.to_bits(), 3f32.to_bits(), 0, 1f32.to_bits()]
        );
    }

    #[test]
    fn bgra_swaps_red_and_blue() {
        let fmt = named("b8g8r8a8-unorm").unwrap();
        let mut texel = [0u8; 4];
        fmt.write([1f32.to_bits(), 0, 0, 0], 0b0001, &mut texel);
        assert_eq!(texel, [0, 0, 255, 0]);
        assert_eq!(fmt.read(&texel)[0], 1f32.to_bits());
    }
}
