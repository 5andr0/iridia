//! The offloader's result checks (lib/Support/Check.cpp): `BufferExact` compares bytes, `BufferFloatULP` and
//! `BufferFloatEpsilon` compare floats with its NaN, signed-zero, denormal and half-float rules; a failure is reported
//! with the text `verifyResult` writes, which some tests check.
use half::f16;

use super::pipeline::{Buffer, Check, DataFormat, Rule, buffer_document};

const F16_SIGN: u16 = 0x8000;
const F16_EXP: u16 = 0x7c00;
const F16_MANT: u16 = 0x03ff;

fn denorm32(f: f32) -> bool {
    f.is_subnormal()
}

fn denorm64(f: f64) -> bool {
    f.is_subnormal()
}

fn f16_nan(v: u16) -> bool {
    v & F16_EXP == F16_EXP && v & F16_MANT != 0
}

fn f16_value(v: u16) -> f32 {
    f16::from_bits(v).to_f32()
}

fn double_epsilon(s: f64, r: f64, epsilon: f64, any: bool) -> bool {
    if s == r {
        return true;
    }
    if s.is_nan() || r.is_nan() {
        return s.is_nan() && r.is_nan();
    }
    if any && denorm64(r) && s == 0.0 && s.is_sign_negative() == r.is_sign_negative() {
        return true;
    }
    (s - r).abs() < epsilon
}

fn double_ulp(s: f64, r: f64, ulp: u32, any: bool, zero: f64) -> bool {
    if s == r {
        return true;
    }
    if s.is_nan() || r.is_nan() {
        return s.is_nan() && r.is_nan();
    }
    if any && denorm64(r) && s == 0.0 && s.is_sign_negative() == r.is_sign_negative() {
        return true;
    }
    if zero > 0.0 && (s.abs() <= zero || r.abs() <= zero) {
        return (s - r).abs() <= zero;
    }
    let diff = s.to_bits().wrapping_sub(r.to_bits()) as i64;
    diff.unsigned_abs() <= u64::from(ulp)
}

fn float_epsilon(s: f32, r: f32, epsilon: f32, any: bool) -> bool {
    if s == r {
        return true;
    }
    if s.is_nan() || r.is_nan() {
        return s.is_nan() && r.is_nan();
    }
    if any && denorm32(r) && s == 0.0 && s.is_sign_negative() == r.is_sign_negative() {
        return true;
    }
    (s - r).abs() < epsilon
}

fn float_ulp(s: f32, r: f32, ulp: u32, any: bool, zero: f32) -> bool {
    if s == r {
        return true;
    }
    if s.is_nan() || r.is_nan() {
        return s.is_nan() && r.is_nan();
    }
    if any && denorm32(r) && s == 0.0 && s.is_sign_negative() == r.is_sign_negative() {
        return true;
    }
    if zero > 0.0 && (s.abs() <= zero || r.abs() <= zero) {
        return (s - r).abs() <= zero;
    }
    let diff = s.to_bits().wrapping_sub(r.to_bits()) as i32;
    diff.unsigned_abs() <= ulp
}

fn half_epsilon(s: u16, r: u16, epsilon: f32) -> bool {
    if s & !F16_SIGN == 0 && r & !F16_SIGN == 0 {
        return true;
    }
    if s == r {
        return true;
    }
    if f16_nan(s) || f16_nan(r) {
        return f16_nan(s) && f16_nan(r);
    }
    (f16_value(s) - f16_value(r)).abs() < epsilon
}

fn half_ulp(s: u16, r: u16, ulp: u32, zero: f32) -> bool {
    if s & !F16_SIGN == 0 && r & !F16_SIGN == 0 {
        return true;
    }
    if s == r {
        return true;
    }
    if f16_nan(s) || f16_nan(r) {
        return f16_nan(s) && f16_nan(r);
    }
    let ordered = |h: u16| -> i32 {
        if h & F16_SIGN != 0 {
            i32::from(!h)
        } else {
            i32::from(h | F16_SIGN)
        }
    };
    if zero > 0.0 {
        let (a, b) = (f16_value(s), f16_value(r));
        if a.abs() <= zero || b.abs() <= zero {
            return (a - b).abs() <= zero;
        }
    }
    (ordered(s) - ordered(r)).unsigned_abs() <= ulp
}

/// Every element of `actual` against `expected`, by `same`; mismatches written as `testAll` writes them.
fn all<const N: usize>(
    actual: &Buffer,
    expected: &Buffer,
    same: impl Fn([u8; N], [u8; N]) -> bool,
    show: impl Fn([u8; N]) -> String,
    log: &mut String,
) -> bool {
    if actual.array_size != expected.array_size || actual.size != expected.size {
        return false;
    }
    for (a, e) in actual.data.iter().zip(&expected.data) {
        let a = a[..actual.size].as_chunks::<N>().0;
        let e = e[..expected.size].as_chunks::<N>().0;
        let mut passed = true;
        for (i, (&x, &y)) in a.iter().zip(e).enumerate() {
            if !same(x, y) {
                log.push_str(&format!(
                    "  Mismatch at element {i}: expected={}, actual={}\n",
                    show(y),
                    show(x)
                ));
                passed = false;
            }
        }
        if !passed {
            return false;
        }
    }
    true
}

/// `std::hexfloat` as libstdc++ writes it for a `float` or `double`: `0x1.8p+1`.
fn hexfloat(v: f64) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan" } else { "nan" }.into();
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf" } else { "inf" }.into();
    }
    let sign = if v.is_sign_negative() { "-" } else { "" };
    if v == 0.0 {
        return format!("{sign}0x0p+0");
    }
    let bits = v.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i64;
    let mant = bits & ((1 << 52) - 1);
    // glibc's %a writes a subnormal unnormalised, 0x0.<mantissa>p-1022
    let (lead, e) = if exp == 0 { (0, -1022) } else { (1, exp - 1023) };
    let mut digits = format!("{mant:013x}");
    while digits.ends_with('0') {
        digits.pop();
    }
    let point = if digits.is_empty() {
        String::new()
    } else {
        format!(".{digits}")
    };
    format!(
        "{sign}0x{lead}{point}p{}{}",
        if e < 0 { '-' } else { '+' },
        e.abs()
    )
}

/// `operator<<` of a float with the default stream format (`%g` with six digits).
fn stream_float(v: f64) -> String {
    super::yaml::format_g(v)
}

/// The comparison of one result: `None` when it passes, otherwise the error text `verifyResult` makes.
pub fn verify(check: &Check, actual: &Buffer, expected: &Buffer) -> Option<String> {
    let any = check.denorm == "Any";
    let mut log = String::new();
    let passed = match check.rule {
        Rule::BufferExact => {
            actual.array_size == expected.array_size
                && actual.size == expected.size
                && actual
                    .data
                    .iter()
                    .zip(&expected.data)
                    .all(|(a, e)| a[..actual.size] == e[..expected.size])
        }
        Rule::BufferFloatULP | Rule::BufferFloatEpsilon => {
            let ulp = check.rule == Rule::BufferFloatULP;
            match actual.format {
                DataFormat::Float64 => all::<8>(
                    actual,
                    expected,
                    |a, e| {
                        let (a, e) = (f64::from_le_bytes(a), f64::from_le_bytes(e));
                        if ulp {
                            double_ulp(a, e, check.ulpt, any, check.zero_tolerance)
                        } else {
                            double_epsilon(a, e, check.epsilon, any)
                        }
                    },
                    |x| {
                        let v = f64::from_le_bytes(x);
                        format!("{} ({})", stream_float(v), hexfloat(v))
                    },
                    &mut log,
                ),
                DataFormat::Float32 | DataFormat::Depth32 => all::<4>(
                    actual,
                    expected,
                    |a, e| {
                        let (a, e) = (f32::from_le_bytes(a), f32::from_le_bytes(e));
                        if ulp {
                            float_ulp(a, e, check.ulpt, any, check.zero_tolerance as f32)
                        } else {
                            float_epsilon(a, e, check.epsilon as f32, any)
                        }
                    },
                    |x| {
                        let v = f64::from(f32::from_le_bytes(x));
                        format!("{} ({})", stream_float(v), hexfloat(v))
                    },
                    &mut log,
                ),
                DataFormat::Float16 => all::<2>(
                    actual,
                    expected,
                    |a, e| {
                        let (a, e) = (u16::from_le_bytes(a), u16::from_le_bytes(e));
                        if ulp {
                            half_ulp(a, e, check.ulpt, check.zero_tolerance as f32)
                        } else {
                            half_epsilon(a, e, check.epsilon as f32)
                        }
                    },
                    |x| format!("0x{:x}", u16::from_le_bytes(x)),
                    &mut log,
                ),
                _ => false,
            }
        }
    };
    if passed {
        return None;
    }
    let mut out = log;
    out.push_str(&format!("Test failed: {}\n", check.name));
    match check.rule {
        Rule::BufferExact => out.push_str("Comparison Rule: BufferExact\n"),
        Rule::BufferFloatULP => {
            out.push_str(&format!("Comparison Rule: BufferFloatULP\nULP: {}\n", check.ulpt))
        }
        Rule::BufferFloatEpsilon => out.push_str(&format!(
            "Comparison Rule: BufferFloatEpsilon\nEpsilon: {}\n",
            stream_float(check.epsilon)
        )),
    }
    out.push_str("Expected:\n");
    out.push_str(&buffer_document(expected));
    out.push_str("Got:\n");
    out.push_str(&buffer_document(actual));
    out.push_str("Full Hex representation of Expected Buffer Values:\n");
    out.push_str(&hex_buffer(expected));
    out.push('\n');
    out.push_str("Full Hex representation of Actual Buffer Values:\n");
    out.push_str(&hex_buffer(actual));
    out.push('\n');
    Some(out)
}

/// `getBufferStr`: every element's bits in lower-case hexadecimal.
fn hex_buffer(b: &Buffer) -> String {
    let n = b.format.size();
    let one = |d: &[u8]| -> String {
        let items: Vec<String> = d[..b.size.min(d.len())]
            .chunks_exact(n)
            .map(|c| {
                let mut w = [0u8; 8];
                w[..n].copy_from_slice(c);
                format!("0x{:x}", u64::from_le_bytes(w))
            })
            .collect();
        if items.is_empty() {
            String::new()
        } else {
            format!("[ {} ]", items.join(", "))
        }
    };
    if b.array_size > 1 {
        b.data
            .iter()
            .map(|d| format!(" - {}", one(d)))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        b.data.last().map(|d| one(d)).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulps_and_signs() {
        assert!(float_ulp(1.0, f32::from_bits(1.0f32.to_bits() + 1), 1, true, 0.0));
        assert!(!float_ulp(
            1.0,
            f32::from_bits(1.0f32.to_bits() + 2),
            1,
            true,
            0.0
        ));
        // a denormal expected, a zero of its sign got, with denormals Any
        assert!(float_ulp(0.0, f32::from_bits(1), 0, true, 0.0));
        assert!(!float_ulp(-0.0, f32::from_bits(1), 0, true, 0.0));
        assert!(half_ulp(0x8000, 0x0000, 0, 0.0));
        // half ULPs across zero count through the ordering, where -0 and +0 are two steps (Check.cpp's ToOrdered)
        assert!(half_ulp(0x8001, 0x0001, 3, 0.0));
        assert!(!half_ulp(0x8001, 0x0001, 2, 0.0));
    }

    #[test]
    fn hexfloat_as_libstdcxx() {
        assert_eq!(hexfloat(1.0), "0x1p+0");
        assert_eq!(hexfloat(3.0), "0x1.8p+1");
        assert_eq!(hexfloat(-0.5), "-0x1p-1");
        assert_eq!(hexfloat(f64::from_bits(1)), "0x0.0000000000001p-1022");
    }
}
