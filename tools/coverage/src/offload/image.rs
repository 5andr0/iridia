//! The offload-test-suite's image check: `offloader -r <target>` writes the render target as a 16-bit RGBA PNG
//! (lib/Image/Image.cpp `writePNG`: each float channel `floor(v * 65536)` clamped), and `imgdiff` reads it and the
//! golden image back with libpng's simplified reader as 8-bit sRGB RGB (`loadPNG`, `PNG_FORMAT_RGB`), then compares
//! the two pixel by pixel by CIE76 distance (lib/Image/ImageComparators.cpp) under a test's rules.
//!
//! Here the render target never becomes a file: its 16-bit values go through the same 8-bit conversion as the golden
//! image's decoded ones, so both sides are converted alike. That conversion is the sRGB transfer function with
//! nearest rounding; libpng computes it through its gamma tables, which can differ from it by one code value, a CIE76
//! distance well under the 2.3 "visible" threshold the rules count by.
use std::fmt::Write as _;

use crate::inflate::inflate;

/// A decoded PNG: 16-bit RGBA, top row first.
#[derive(Clone, Debug)]
pub struct Rgba16 {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Four 16-bit channels per pixel.
    pub pixels: Vec<[u16; 4]>,
}

/// Decode a PNG of 16-bit RGBA (colour type 6, depth 16, not interlaced), as the golden images are.
pub fn decode_png(file: &[u8]) -> Result<Rgba16, String> {
    if !file.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("not a PNG".into());
    }
    let mut at = 8;
    let (mut width, mut height) = (0u32, 0u32);
    let mut idat = Vec::new();
    while at + 8 <= file.len() {
        let len = u32::from_be_bytes(file[at..at + 4].try_into().expect("4 bytes")) as usize;
        let kind = &file[at + 4..at + 8];
        let body = file.get(at + 8..at + 8 + len).ok_or("PNG: chunk ends")?;
        match kind {
            b"IHDR" => {
                width = u32::from_be_bytes(body[0..4].try_into().expect("4 bytes"));
                height = u32::from_be_bytes(body[4..8].try_into().expect("4 bytes"));
                let (depth, colour, interlace) = (body[8], body[9], body[12]);
                if depth != 16 || colour != 6 || interlace != 0 {
                    return Err(format!(
                        "PNG: depth {depth}, colour type {colour}, interlace {interlace}: only 16-bit RGBA is read"
                    ));
                }
            }
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
        at += 12 + len;
    }
    let stride = width as usize * 8;
    let raw = inflate(
        idat.get(2..).ok_or("PNG: no data")?,
        (stride + 1) * height as usize,
    )?;
    if raw.len() < (stride + 1) * height as usize {
        return Err("PNG: data ends".into());
    }
    let mut prev = vec![0u8; stride];
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height as usize {
        let row = &raw[y * (stride + 1)..(y + 1) * (stride + 1)];
        let filter = row[0];
        let mut cur = row[1..].to_vec();
        for i in 0..stride {
            let a = if i >= 8 { cur[i - 8] } else { 0 };
            let b = prev[i];
            let c = if i >= 8 { prev[i - 8] } else { 0 };
            cur[i] = cur[i].wrapping_add(match filter {
                0 => 0,
                1 => a,
                2 => b,
                3 => ((u16::from(a) + u16::from(b)) / 2) as u8,
                4 => {
                    let p = i16::from(a) + i16::from(b) - i16::from(c);
                    let (pa, pb, pc) = (
                        (p - i16::from(a)).abs(),
                        (p - i16::from(b)).abs(),
                        (p - i16::from(c)).abs(),
                    );
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    }
                }
                f => return Err(format!("PNG: filter {f}")),
            });
        }
        for px in cur.as_chunks::<8>().0 {
            pixels.push([0, 1, 2, 3].map(|k| u16::from_be_bytes([px[2 * k], px[2 * k + 1]])));
        }
        prev = cur;
    }
    Ok(Rgba16 {
        width,
        height,
        pixels,
    })
}

/// `ColorUtils::toInt<uint16_t>`: a float channel as `writePNG` stores it.
pub fn to_u16(v: f32) -> u16 {
    let d = f64::from(v);
    if d.is_nan() {
        return 0;
    }
    (d * 65536.0).floor().clamp(0.0, 65535.0) as u16
}

fn srgb_encode(linear: f64) -> f64 {
    if linear <= 0.003_130_8 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// libpng's simplified reader on a 16-bit linear RGBA pixel, asked for 8-bit sRGB RGB with no background: opaque
/// pixels are sRGB-encoded, transparent ones stay black (the zeroed buffer), the rest are composed on black.
pub fn to_srgb8(p: [u16; 4]) -> [u8; 3] {
    let alpha8 = ((u32::from(p[3]) * 255 + 32767) / 65535) as u8;
    if alpha8 == 0 {
        return [0; 3];
    }
    let a = f64::from(p[3]) / 65535.0;
    [0, 1, 2].map(|k| {
        let c = f64::from(p[k]) / 65535.0;
        if alpha8 == 255 {
            (srgb_encode(c) * 255.0).round().clamp(0.0, 255.0) as u8
        } else {
            // the premultiplied linear 8-bit component, then composed on black and encoded
            let component = (c * a * 255.0).round();
            (srgb_encode(component / 255.0) * 255.0).round().clamp(0.0, 255.0) as u8
        }
    })
}

// Color.cpp: the colour spaces and the CIE76 distance.

const D65: [f64; 3] = [95.047, 100.000, 108.883];

fn xyz_to_lab_component(v: f64) -> f64 {
    const E: f64 = 216.0 / 24389.0;
    const K: f64 = 24389.0 / 27.0;
    if v > E {
        v.powf(1.0 / 3.0)
    } else {
        (K * v + 16.0) / 116.0
    }
}

fn lab(rgb: [f64; 3]) -> [f64; 3] {
    let m = [
        0.4124564, 0.3575761, 0.1804375, 0.2126729, 0.7151522, 0.0721750, 0.0193339, 0.1191920, 0.9503041,
    ];
    let x = rgb[0] * m[0] + rgb[1] * m[1] + rgb[2] * m[2];
    let y = rgb[0] * m[3] + rgb[1] * m[4] + rgb[2] * m[5];
    let z = rgb[0] * m[6] + rgb[1] * m[7] + rgb[2] * m[8];
    let fx = xyz_to_lab_component(x / D65[0]);
    let fy = xyz_to_lab_component(y / D65[1]);
    let fz = xyz_to_lab_component(z / D65[2]);
    [(116.0 * fy - 16.0).max(0.0), 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// `Color::cie75Distance` of two 8-bit RGB pixels (as `compareImages` translates them, `v / 255`).
pub fn distance(l: [u8; 3], r: [u8; 3]) -> f64 {
    let f = |p: [u8; 3]| p.map(|c| f64::from(c) / 255.0);
    let (a, b) = (lab(f(l)), lab(f(r)));
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A rule of a test's rules file (`CompareCheck`).
#[derive(Clone, Debug)]
pub enum Rule {
    /// `Furthest`: no pixel further than this.
    Furthest(f64),
    /// `RMS` of all pixels.
    Rms(f64),
    /// `DiffRMS`: of the visibly different pixels.
    DiffRms(f64),
    /// `PixelPercent`: at most this percent visibly different.
    PixelPercent(f64),
    /// `Intervals`: the histogram's buckets, each at most this percent.
    Intervals(Vec<f64>),
}

/// The rules a rules file's YAML (as JSON of strings) states.
pub fn rules(v: &serde_json::Value) -> Result<Vec<Rule>, String> {
    let list = v.as_array().ok_or("rules: not a sequence")?;
    let num = |x: &serde_json::Value| -> Result<f64, String> {
        x.as_str()
            .ok_or("rules: not a scalar")?
            .trim()
            .parse::<f64>()
            .map_err(|e| format!("rules: {e}"))
    };
    list.iter()
        .map(|r| {
            let ty = r.get("Type").and_then(|t| t.as_str()).ok_or("rules: no Type")?;
            if ty == "Intervals" {
                let vals = r.get("Vals").and_then(|v| v.as_array()).ok_or("rules: no Vals")?;
                return Ok(Rule::Intervals(vals.iter().map(num).collect::<Result<_, _>>()?));
            }
            let val = num(r.get("Val").ok_or("rules: no Val")?)?;
            Ok(match ty {
                "Furthest" => Rule::Furthest(val),
                "RMS" => Rule::Rms(val),
                "DiffRMS" => Rule::DiffRms(val),
                "PixelPercent" => Rule::PixelPercent(val),
                other => return Err(format!("rules: unknown Type {other}")),
            })
        })
        .collect()
}

/// `imgdiff` of two 8-bit RGB images under `rules` (the default rule `Furthest 2.3` when there are none): pass or the
/// text it prints.
pub fn compare(expected: &[[u8; 3]], actual: &[[u8; 3]], rules: &[Rule]) -> Result<(), String> {
    if expected.len() != actual.len() {
        return Err("imgdiff: error: Image sizes do not match".into());
    }
    const VISIBLE: f64 = 2.3;
    let (mut furthest, mut rms, mut diff_rms) = (0f64, 0f64, 0f64);
    let (mut count, mut visible) = (0u64, 0u64);
    let mut histogram = [0u64; 10];
    for (e, a) in expected.iter().zip(actual) {
        let d = distance(*e, *a);
        if d > VISIBLE {
            visible += 1;
            diff_rms += d;
            histogram[(d - VISIBLE).clamp(0.0, 9.0) as usize] += 1;
        }
        furthest = furthest.max(d);
        rms += d;
        count += 1;
    }
    rms = (rms / count as f64).sqrt();
    diff_rms = (diff_rms / visible as f64).sqrt();
    let default = [Rule::Furthest(VISIBLE)];
    let rules = if rules.is_empty() { &default[..] } else { rules };
    let mut error = None;
    for r in rules {
        let failed = match r {
            Rule::Furthest(v) => (furthest > *v)
                .then(|| format!("Furthest color distance check failed: {furthest} above threshold {v}")),
            Rule::Rms(v) => (rms > *v).then(|| format!("RMS check failed. RMS {rms} above threshold {v}")),
            Rule::DiffRms(v) => (diff_rms > *v).then(|| {
                format!("Differing RMS check failed. RMS of differing pixels {diff_rms} above threshold {v}")
            }),
            Rule::PixelPercent(v) => {
                let pct = visible as f64 / count as f64 * 100.0;
                (pct > *v).then(|| {
                    format!("PixelPercent check failed. Difference percent {pct} above threshold {v}")
                })
            }
            Rule::Intervals(vals) => (0..10).find_map(|i| {
                if i >= vals.len() {
                    (histogram[i] != 0)
                        .then(|| format!("Interval[{i}]: Contains non-zero value: {}", histogram[i]))
                } else {
                    let pct = histogram[i] as f64 / count as f64 * 100.0;
                    (pct > vals[i])
                        .then(|| format!("Interval[{i}]: Out of range. Maximum: {} Actual: {pct}", vals[i]))
                }
            }),
        };
        if failed.is_some() {
            error = failed;
            break;
        }
    }
    let Some(error) = error else {
        return Ok(());
    };
    let mut text = String::new();
    let _ = writeln!(text, "RMS Difference: {rms}");
    let _ = writeln!(text, "Furthest Pixel Difference: {furthest}");
    let _ = writeln!(
        text,
        "Pixels with visible differences: {visible} {}%",
        visible as f64 / count as f64 * 100.0
    );
    let _ = writeln!(text, "Total Pixels: {count}");
    let _ = writeln!(text, "Error: {error}");
    Err(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours() {
        assert_eq!(to_u16(1.0), 65535);
        assert_eq!(to_u16(0.5), 32768);
        assert_eq!(to_srgb8([65535, 0, 32768, 65535]), [255, 0, 188]);
        assert_eq!(distance([10, 20, 30], [10, 20, 30]), 0.0);
        // Color.cpp divides XYZ of RGB in [0, 1] by a white point scaled to 100: white is Y = 0.01, L* = 116 * 0.01^(1/3)
        // - 16, so black to white is about 8.99 (not 100)
        let white = 116.0 * 0.01f64.cbrt() - 16.0;
        assert!(
            (distance([0, 0, 0], [255, 255, 255]) - white).abs() < 1e-4,
            "{white}"
        );
    }
}
