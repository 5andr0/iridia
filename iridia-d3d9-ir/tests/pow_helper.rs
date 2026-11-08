//! The `pow` the WARP comparison swaps in (`iridia_warp_client::substitute_pow`) to tell a program that differs from WARP
//! only at `pow` of a zero base and an exponent of 0 or less from one that differs elsewhere: it must be WGSL's `pow` at every
//! other argument, or an agreement after the swap proves nothing about the zero base. Both `pow`s run on WARP in one program
//! over a grid of bases and exponents (signed zeros, a denormal, the extremes, negative values, infinities, NaN): they agree
//! in every bit (any NaN with any NaN) except at a zero base with an exponent of 0 or less, where the helper gives libm's
//! value (`pow(0, 0)` 1, `pow(0, y < 0)` +inf), the value `iridia-exec-ir` gives.
mod harness;
mod inputs;
mod warp;

use iridia_warp_client as wgsl_run;

const GRID: [f32; 20] = [
    0.0,
    -0.0,
    1e-45,
    -1e-45,
    1e-40,
    1.175_494_4e-38,
    0.5,
    1.0,
    2.0,
    3.7,
    1e10,
    3.4e38,
    f32::INFINITY,
    -0.5,
    -1.0,
    -2.0,
    -3.7,
    f32::NEG_INFINITY,
    f32::NAN,
    -1e10,
];

const PROGRAM: &str = "
@fragment
fn main(@location(0) a: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(pow(a.x, a.y), iridia_pow(a.x, a.y), 0.0f, 0.0f);
}
";

#[test]
fn the_swapped_in_pow_is_pow_except_at_a_zero_base_with_an_exponent_of_0_or_less() {
    let Some(harness) = warp::harness() else {
        eprintln!("tools.wgsl_run is not set: skipped");
        return;
    };
    let wgsl = wgsl_run::substitute_pow(PROGRAM).expect("substitutes");
    let pairs: Vec<(f32, f32)> = GRID
        .iter()
        .flat_map(|&x| GRID.iter().map(move |&y| (x, y)))
        .collect();
    assert_eq!(pairs.len() % 4, 0, "a pixel program runs in quads");
    let invocations: Vec<_> = pairs
        .iter()
        .map(|&(x, y)| serde_json::json!([[x.to_bits(), y.to_bits(), 0, 0]]))
        .collect();
    let reply = harness
        .run(&serde_json::json!({"id": 0, "wgsl": wgsl, "invocations": invocations, "resources": []}))
        .expect("runs");
    let got = reply["invocations"].as_array().expect("invocations");
    assert_eq!(got.len(), pairs.len());
    let mut zero_base = 0;
    for (&(x, y), g) in pairs.iter().zip(got) {
        let lane = |k: usize| {
            f32::from_bits(u32::try_from(g["results"][0][k].as_u64().expect("a word")).expect("32 bits"))
        };
        let (builtin, helper) = (lane(0), lane(1));
        // WARP flushes a denormal to zero, so the helper sees a denormal exponent as 0
        let seen = if y.is_subnormal() { 0.0 } else { y };
        if x == 0.0 && y <= 0.0 {
            zero_base += 1;
            let defined = if seen == 0.0 { 1.0 } else { f32::INFINITY };
            assert_eq!(
                helper.to_bits(),
                defined.to_bits(),
                "pow({x:?}, {y:?}): the helper gives {helper:?}, libm's value is {defined:?}"
            );
        } else {
            assert!(
                builtin.to_bits() == helper.to_bits() || (builtin.is_nan() && helper.is_nan()),
                "pow({x:?}, {y:?}): the builtin gives {builtin:?} ({:#010x}), the helper {helper:?} ({:#010x})",
                builtin.to_bits(),
                helper.to_bits()
            );
        }
    }
    // the grid has a zero base of each sign with every exponent of 0 or less
    assert_eq!(zero_base, 2 * GRID.iter().filter(|&&y| y <= 0.0).count());
}
