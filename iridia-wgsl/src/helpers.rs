//! Helper functions of the text: the operations that take more than an expression (`f32tof16`, a remainder that is exact,
//! the high half of a product) are one WGSL function each, written once at the top of a module that calls them. They take
//! and give scalars; a vector operand is taken lane by lane at the call.
//!
//! Each is integer arithmetic where it can be, so every device gives the same bits (`pack2x16float` needs `f16` and leaves
//! the rounding open).

use alloc::collections::BTreeSet;
use alloc::string::String;

/// A helper function a program's text may call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Helper {
    /// `f32` to the bits of the nearest binary16.
    F32ToF16,
    /// The `f32` of the binary16 in the low bits.
    F16ToF32,
    /// C's `fmod`, exact.
    Fmod,
    /// The high half of an unsigned 32-bit product.
    UMulHigh,
    /// The high half of a signed 32-bit product.
    SMulHigh,
    /// The `f32` with the bits of a `u32` that a compiler cannot fold: WGSL makes a constant infinity or NaN an error.
    F32FromBits,
}

impl Helper {
    /// The function's name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Helper::F32ToF16 => "iridia_f32_to_f16",
            Helper::F16ToF32 => "iridia_f16_to_f32",
            Helper::Fmod => "iridia_fmod",
            Helper::UMulHigh => "iridia_umul_high",
            Helper::SMulHigh => "iridia_smul_high",
            Helper::F32FromBits => "iridia_f32_from_bits",
        }
    }

    /// The helpers this one calls.
    fn calls(self) -> &'static [Helper] {
        match self {
            Helper::SMulHigh => &[Helper::UMulHigh],
            Helper::Fmod => &[Helper::F32FromBits],
            _ => &[],
        }
    }

    /// The function's text.
    pub(crate) fn source(self) -> &'static str {
        match self {
            Helper::F32ToF16 => F32_TO_F16,
            Helper::F16ToF32 => F16_TO_F32,
            Helper::Fmod => FMOD,
            Helper::UMulHigh => UMUL_HIGH,
            Helper::SMulHigh => SMUL_HIGH,
            Helper::F32FromBits => F32_FROM_BITS,
        }
    }
}

/// The text of `used` and the helpers they call, in a fixed order.
pub(crate) fn definitions(used: &BTreeSet<Helper>) -> String {
    let mut all = used.clone();
    for h in used {
        all.extend(h.calls().iter().copied());
    }
    all.iter()
        .map(|h| h.source())
        .collect::<alloc::vec::Vec<_>>()
        .join("\n")
}

/// The `f32` with these bits. A call is not a constant expression: `bitcast<f32>(0x7f800000u)` is, and WGSL refuses a
/// constant that is an infinity or a NaN (Tint and naga both). This makes the text valid, not exact: WGSL gives no
/// guarantee about an infinity or a NaN at run time, so a browser's device may return other bits than these.
pub(crate) const F32_FROM_BITS: &str = "fn iridia_f32_from_bits(bits: u32) -> f32 {
    return bitcast<f32>(bits);
}
";

/// The nearest binary16 to an `f32`, in the low 16 bits. Ties go to even; a NaN is the quiet NaN 0x7e00, a value beyond the
/// range is an infinity, a binary16 denormal is kept.
pub(crate) const F32_TO_F16: &str = "\
fn iridia_f32_to_f16(x: f32) -> u32 {
    let w = bitcast<u32>(x);
    let sign = (w >> 16u) & 0x8000u;
    let exp = (w >> 23u) & 0xffu;
    let man = w & 0x7fffffu;
    let signed_infinity = sign | 0x7c00u;
    let special = select(0x7e00u, signed_infinity, man == 0u);
    // a normal half keeps the exponent less 112 and drops 13 bits; below that the implicit bit joins a significand
    // shifted further, a denormal
    let normal = exp > 112u;
    let sig = select(man | 0x800000u, man, normal);
    let shift = select(126u - exp, 13u, normal);
    let top = select(0u, (exp - 112u) << 10u, normal);
    let half = sig >> shift;
    let rest = sig & ((1u << shift) - 1u);
    let mid = 1u << (shift - 1u);
    let tie_odd = rest == mid && (half & 1u) == 1u;
    let up = select(0u, 1u, rest > mid || tie_odd);
    // a carry out of the significand moves into the exponent, to an infinity at the top
    let rounded = sign | ((top | half) + up);
    var r = select(rounded, sign, exp < 102u);
    r = select(r, signed_infinity, exp >= 143u);
    return select(r, special, exp == 0xffu);
}
";

/// The `f32` of a binary16 in the low 16 bits (the bits above ignored), exact.
pub(crate) const F16_TO_F32: &str = "\
fn iridia_f16_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let exp = (h >> 10u) & 0x1fu;
    let man = h & 0x3ffu;
    let fraction = man << 13u;
    let normal = ((exp + 112u) << 23u) | fraction;
    // a denormal, normalised: its leading bit becomes the implicit one
    let lead = countLeadingZeros(man) - 21u;
    let denormal_bits = ((113u - lead) << 23u) | (((man << lead) & 0x3ffu) << 13u);
    let denormal = select(denormal_bits, 0u, man == 0u);
    let special = 0x7f800000u | fraction;
    var r = select(normal, denormal, exp == 0u);
    r = select(r, special, exp == 0x1fu);
    return bitcast<f32>(sign | r);
}
";

/// C's `fmod` of `f32`, exact: WGSL's `%` is `x - y * trunc(x / y)` with each step rounded, so the remainder is a long
/// division of the significands in integers, one bit of the exponents' difference a step. A NaN where `x` is infinite or a
/// NaN or `y` is zero or a NaN; `x` where `|x| < |y|` or `y` is infinite.
pub(crate) const FMOD: &str = "\
fn iridia_fmod(x: f32, y: f32) -> f32 {
    let bx = bitcast<u32>(x);
    let by = bitcast<u32>(y);
    let ax = bx & 0x7fffffffu;
    let ay = by & 0x7fffffffu;
    let sign = bx & 0x80000000u;
    let nan = ax >= 0x7f800000u || ay == 0u || ay > 0x7f800000u;
    let unchanged = ax < ay || ay == 0x7f800000u;
    // each significand normalised to its top bit at bit 23, its exponent (32 more than the biased one, so a denormal's
    // stays positive) lowered by the shift
    let ex0 = ax >> 23u;
    let ey0 = ay >> 23u;
    let mx0 = (ax & 0x7fffffu) | select(0x800000u, 0u, ex0 == 0u);
    let my0 = (ay & 0x7fffffu) | select(0x800000u, 0u, ey0 == 0u);
    let upx = countLeadingZeros(mx0) - 8u;
    let upy = countLeadingZeros(my0) - 8u;
    let mx = mx0 << upx;
    let my = my0 << upy;
    let ex = max(ex0, 1u) + 32u - upx;
    let ey = max(ey0, 1u) + 32u - upy;
    var m = mx;
    var d = select(ex - ey, 0u, nan || unchanged);
    loop {
        if (d == 0u) {
            break;
        }
        m = select(m, m - my, m >= my) << 1u;
        d = d - 1u;
    }
    let r = select(m, m - my, m >= my);
    // the remainder (below 2^24, at y's exponent) normalised; a biased exponent below 1 makes it a denormal, shifted down
    let up = countLeadingZeros(r) - 8u;
    let rn = r << up;
    let e = ey - up;
    let normal_bits = ((e - 32u) << 23u) | (rn & 0x7fffffu);
    let denormal_bits = rn >> min(33u - e, 31u);
    var bits = select(denormal_bits, normal_bits, e >= 33u);
    bits = select(bits, 0u, r == 0u) | sign;
    let kept = select(bitcast<f32>(bits), x, unchanged);
    return select(kept, iridia_f32_from_bits(0x7fc00000u), nan);
}
";

/// The high half of the 64-bit product of two `u32`, from 16-bit halves.
pub(crate) const UMUL_HIGH: &str = "\
fn iridia_umul_high(a: u32, b: u32) -> u32 {
    let al = a & 0xffffu;
    let ah = a >> 16u;
    let bl = b & 0xffffu;
    let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    // the middle column: the carry into the high half, each part below 2^16
    let carry = ((ll >> 16u) + (lh & 0xffffu) + (hl & 0xffffu)) >> 16u;
    return hh + (lh >> 16u) + (hl >> 16u) + carry;
}
";

/// The high half of the 64-bit product of two `i32`: the unsigned one less each operand where the other is negative.
pub(crate) const SMUL_HIGH: &str = "\
fn iridia_smul_high(a: i32, b: i32) -> i32 {
    let ua = bitcast<u32>(a);
    let ub = bitcast<u32>(b);
    return bitcast<i32>(iridia_umul_high(ua, ub) - select(0u, ub, a < 0) - select(0u, ua, b < 0));
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, string::ToString};

    const ALL: [Helper; 6] = [
        Helper::F32ToF16,
        Helper::F16ToF32,
        Helper::Fmod,
        Helper::UMulHigh,
        Helper::SMulHigh,
        Helper::F32FromBits,
    ];

    /// Each helper, with what it calls, is a module upstream naga reads and validates. The bits it gives are held to the IR's
    /// definition by `iridia-dxbc`'s `exec_wgsl`, which runs them on WARP.
    #[test]
    fn every_helper_is_valid_wgsl() {
        for h in ALL {
            let text = format!(
                "{}\n@compute @workgroup_size(1) fn main() {{}}\n",
                definitions(&BTreeSet::from([h]))
            );
            let module = naga::front::wgsl::parse_str(&text)
                .unwrap_or_else(|e| panic!("{}: {}\n{text}", h.name(), e.emit_to_string(&text)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}: {e:?}\n{text}", h.name()));
        }
    }

    #[test]
    fn each_helper_defines_the_function_it_is_named_for() {
        for h in ALL {
            assert!(
                h.source().starts_with(&format!("fn {}(", h.name())),
                "{}",
                h.name()
            );
        }
        let both = definitions(&BTreeSet::from([Helper::SMulHigh]));
        assert!(both.contains("fn iridia_umul_high("), "{}", both.to_string());
    }
}
