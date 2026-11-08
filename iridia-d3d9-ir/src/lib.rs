//! Reads raw Direct3D 9 token streams of every shader model (vertex 1.1 to 3.0, pixel 1.1 to 3.0) and lifts them
//! into iridia's shader IR ([`iridia_ir::Program`]).
//!
//! The records are read by `d3d9-reader`; [`read`] checks every register against its version's banks and makes the
//! inputs and outputs a version implies by register explicit. [`signature`] states the interface the declarations,
//! registers and constant table give, and [`lift`] builds the program: one variable per register, each instruction's
//! operations in order, the outputs stored at the end.
//!
//! # Special inputs
//!
//! What the lift does at an input a Microsoft instruction page gives a value for (or says nothing about), and why. The rule:
//! follow the document where it states one value; where it contradicts itself, follow the prose and the hardware; where it is
//! silent, keep a deliberate recorded policy. Sources: Microsoft Learn
//! `direct3dhlsl/<op>---vs` and `---ps` pages (win32-pr, the instruction reference), fetched 2026-11-08; Wine's bug 10423 and
//! wined3d's zero-source commits as the driver-behaviour evidence. "Lift" is `iridia-d3d9-ir/src/lift/arith.rs`; "IR" is
//! `iridia-ir/src/eval.rs` (the executor both writers are held to); "WGSL" is what a builtin does where the IR leaves the case
//! to the target.
//!
//! | op (versions) | input | Microsoft's stated result | lift today | verdict |
//! |---|---|---|---|---|
//! | `log` (vs/ps 2.0+, vs 1.1) | `abs(src) == 0` | `-FLT_MAX` (code fragment; nothing in the prose contradicts it) | `FLog2(abs(x))`: IR `log2(0)` = `-inf`; WGSL `log2(0)` = `-inf` or undefined | **fixed**: `select(x == 0, -FLT_MAX, log2(abs(x)))`, one row of `tests/assembled/ps_2_0_exp_log.asm` and of `vs_1_1_expp_logp.asm` each |
//! | `logp` (vs 1.1+) | `abs(src) == 0` | `-FLT_MAX` (code fragment) | as `log` (`FLog2`, partial precision) | **fixed** with `log` |
//! | `nrm` (2.0+) | `x*x + y*y + z*z == 0` | `f = FLT_MAX`, so the lanes are `src * FLT_MAX` (0 for a zero lane) | `select(squared == 0, f32::MAX, inverseSqrt(squared))` | matches, nothing to do (a test row exists in `assembled`: vs_2_0_pow_nrm_lrp) |
//! | `rcp` (2.0+, vs 1.1) | `src == 0` | code fragment: `FLT_MAX`; prose beside it: "A source of 0.0 yields infinity" (the page contradicts itself) | `FDiv(1, x)`: `+inf` for +0, `-inf` for -0 | **doc inconsistent**: pseudocode and prose disagree; Wine's wined3d returns `FLT_MAX` (Verbeet 2010); hardware gives infinity. Decision needed, default: leave (IEEE, = the prose) |
//! | `rcp` | `src == 1` | exactly 1.0 | `1/1` = 1 | matches |
//! | `rsq` (2.0+, vs 1.1) | `abs(src) == 0` | code fragment: `FLT_MAX`; prose: "A source of 0.0 yields infinity" | `FInverseSqrt(abs(x))` = `+inf` | same contradiction as `rcp` (vs; the ps page states only the prose, infinity): leave, say so |
//! | `rsq` | `abs(src) == 1` | exactly 1.0 | `inverseSqrt(1)` = 1 | matches (IR exact; WGSL builtin within its ULP bound, WARP checked) |
//! | `exp` (vs 1.1+, ps via `exp`) | any | `pow(2, src)`; at least 21 bits; no special case stated | `FExp2` | silent: leave |
//! | `expp` (vs) | vs 1.1: `(2^floor(v), v - floor(v), 2^v, 1)`; 2.0+: `2^v` in every lane (10 bits) | as stated | `expp()` builds exactly those lanes | matches |
//! | `pow` (2.0+) | `abs(src0) == 0`, any exponent | `dest = pow(abs(src0), src1)`; "could be expanded as exp(src1 * log(src0))"; no value for a zero base is stated | `FPow(abs(x), y)`, no guard: IR `libm::pow` (`pow(0,0)` = 1, `pow(0,y<0)` = +inf, `pow(0,y>0)` = 0); WGSL `pow` is undefined for `x = 0, y <= 0`; WARP gives NaN | **doc silent** about the value. The "may be expanded" form with the documented `log(0) = -FLT_MAX` gives 1 / +inf / 0, which is libm's. Drivers disagree at 0^0 (Wine bug 10423: software VP 0, hardware VP 1; later commits: NaN on most GPUs; wined3d returns 0 for any zero base). Leave the lift: the zero base stays undefined, as the IR's `FPow` and WGSL's `pow` leave it. In the WARP comparison it is a class of its own: a program with a lane beyond tolerance whose WGSL has `pow` is run again with `pow` giving a zero base libm's value, and counts when every lane then agrees (`tests/warp`, helper held to `pow` by `tests/pow_helper.rs`). Proof pair: `pow(0, -0.0)` is 1 in the IR executor and NaN on WARP |
//! | `lit` (vs 1.1+) | `src.w` outside +-127.9961 | clamped to +-`MAXPOWER` | `select`s as the code fragment | matches |
//! | `lit` | `src.x <= 0` / `src.y <= 0` | `y = 0` / `z = 0` (and `0^0` not reached: `pow` runs only for `src.y > 0`) | selects as the fragment; NaN compares false, as the fragment | matches |
//! | `sincos` (vs 2.0+) | angle outside [-pi, +pi] | "must be within [-pi, +pi]"; the result outside is not stated | `FSin`/`FCos` of the value | silent: leave (undefined input) |
//! | `sincos` | unwritten lanes | vs 2.x: `dest.y/.z` undefined; vs 3.0: not touched | lanes written as the mask says | matches (only x, y, xy accepted; others refused) |
//! | `mova`, `mov` to `a0` | halves | (`mova` to nearest, the reference rasterizer rounds a half away from zero; vs 1.x `mov` floors) | `Rounding::NearestAway` / `Floor` | matches the reference as the lift's comment says |
//! | `mul`, `mad`, `dp*` | `0 * inf`, `0 * nan` | pseudocode is plain `a * b`; no special case | `Options::multiply`: `ZeroWins` default (DXVK/wine), `Ieee` by choice | doc silent: leave (a policy the lifter documents; held to Windows by the vkd3d cases) |
//! | `min`, `max` | a NaN operand | pseudocode `(a < b) ? a : b` and `(a >= b) ? a : b`: a failed compare gives `src1`, so `min(x, NaN)` is NaN and `min(NaN, x)` is `x` | `MinMaxNan::MinNum`: the non-NaN operand either way | **doc pseudocode differs for a NaN `src1`** (NaN vs the other operand); the page gives it only as pseudocode, and the lifter chose DXVK's behaviour on purpose (`d3d9.min_max_nan=min_num` is on every program's provenance). A decision for the coordinator; default: leave |
//! | `sgn` (2.0+) | `src < 0` / `== 0` / else | -1 / 0 / 1 (NaN falls to 1) | selects exactly that | matches |
//! | `cmp` (ps 1.2+) | `src0 >= 0` | `src1` if `src0 >= 0`, else `src2` (NaN: `src2`) | `FCmp(Ge)` + `select` | matches |
//! | `slt`, `sge` | compare | `(a < b) ? 1 : 0`, `(a >= b) ? 1 : 0` | `FCmp` + `select` | matches |
//! | `frc` | any | `x - floor(x)` | `FFract` | matches (WGSL `fract` is the same definition) |
//! | `dst` | any | `(1, s0.y * s1.y, s0.z, s1.w)` | `Compose([1, y*y, z, w])`, the product by the multiply rule | matches |
//! | `texdepth` (ps 1.4) | `r5.g == 0` | "If r5.g = 0, the result of r5.r / r5.g = 1.0" | `x / min(y, 1)`: x/0 = inf or NaN | **fixed**: `select(y == 0, 1, x / min(y, 1))`, `tests/assembled/ps_1_4_texdepth.asm` |
//! | `texm3x2depth` (ps 1.3) | `w == 0` | `t = 1.0` where `w == 0`, else `z / w` | `select(w == 0, 1, z / w)` | matches |
//! | `texm3x3spec`, `texm3x3vspec` | `N * N == 0` | reflection `2 * ((N*E)/(N*N)) * N - E`; zero normal not stated | as the formula | silent: leave |
//! | `texld` ps 1.4 `_dw` / `_dz`, projected samples | divisor 0 | not stated | `FDiv` by w / z | silent: leave |
//! | `texkill` | lane `< 0` | the first three lanes (ps 1.x; all four from 2.0 by the HLSL `clip` correspondence), "if x,y,z < 0 cancel" | `Any(lane < 0)` then `demote` | matches |
//! | `lrp`, `dp2add`, `abs`, `mad`, `bem` | any | formulas only (`src2 + src0 * (src1 - src2)` is stated as equal to `src0 * src1 + (1 - src0) * src2`); no special case | as the formulas (`lrp` the second form) | matches / silent |
//!
//! Pages read: pow, log, logp, exp, expp, rcp, rsq, nrm, lit, sincos, dst, mul, mad, min, abs, sgn, cmp, frc, sge, lrp, dp2add, bem,
//! texkill, texdepth, texm3x2depth, texm3x3spec, texld (ps 1.4) (vs and ps where they differ; `slt`, `max`, `sub`, `add` follow
//! their neighbours by name and were not read). Not read: the other ps 1.x texture instructions (`tex`, `texcoord`, `texcrd`,
//! `texbem`, `texbeml`, `texreg2*`, `texdp3*`, `texm3x2pad`/`tex`, `texm3x3pad`/`tex`/`vspec`) and `phase`: they have no special input
//! values in the pages' summaries; their formulas are held by the assembled fixtures.
//!
//! ## Decisions
//!
//! The rule above gives these (2026-11-08):
//!
//! - `log`, `logp` and `texdepth` are fixed to the stated values (the table's "fixed" rows), each with a fixture row in
//!   `tests/assembled` whose `; run:` line states the documented value; the assembled test also runs both writers' WGSL of
//!   each fixture on WARP, so the `select` is checked in the WGSL as well.
//! - `rcp` and `rsq` stay IEEE infinity: the prose, the pixel-shader page and the hardware say infinity; the pseudocode's
//!   `FLT_MAX` is what wined3d chose (Verbeet 2010), recorded here and not followed.
//! - `min`/`max` keep `min_num`, a recorded policy with its provenance on every program (`d3d9.min_max_nan=min_num`); the
//!   pseudocode differs only for a NaN `src1`.
//! - `pow` (a zero base), `exp`, `sincos` outside [-pi, +pi] and `mul` of `0 * inf` are left as the documents leave them:
//!   silent, and a deliberate policy where the lifter has one (`Options::multiply`).
//!
//! The pages not read (the other ps 1.x texture instructions, `phase`) have no special input values in the pages' summaries;
//! their formulas are held by the assembled fixtures.
#![deny(missing_docs)]
#![deny(clippy::disallowed_methods, clippy::disallowed_types)]
#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation
    )
)]

mod lift;
mod model;
mod sampler;
mod signature;

pub use d3d9_reader::{Comment, Constant, ConstantTable, Type, TypeMember};
pub use lift::{
    LIFTER, LIFTER_VERSION, Options, Projection, Ps1x, lift, lift_with, project, project_vertex_output,
    project_vertex_output_with, project_with, reencode, reencode_with,
};
pub use model::{
    Comparison, Destination, Index, Input, Instruction, Opcode, Output, Register, Shader, Source, Version,
    read, usage,
};
pub use sampler::sampler_states;
pub use signature::{
    Bank, PARTIAL_PRECISION, PIXEL_CONSTANTS, Sampler, Signature, VERTEX_CONSTANTS, lanes_in, outputs,
    read_input, signature, varying,
};

/// A refusal states the word offset in the raw token stream and the unsupported or malformed feature.
pub type Error = iridia_ir::Refusal;
/// The result of every function in this crate: `Ok` or a refusal that names the word offset.
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn error(at: usize, detail: impl Into<String>) -> Error {
    Error::new("D3D9", iridia_ir::Unit::Word, at, detail)
}

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;

/// Frame the first raw SM3 vertex or pixel program in a container's remaining bytes.
/// Returns its stage and exact bytes through END; subsequent container metadata is not consumed.
/// `d3d9-reader` reads every record up to END, so END is never searched for inside operands or comments; an
/// operation the lifter does not support is retained as bytecode for [`lift`] to refuse.
pub fn frame(bytes: &[u8]) -> Result<(iridia_ir::interface::Stage, &[u8])> {
    use iridia_ir::interface::Stage;
    let (program, span) = d3d9_reader::frame(bytes).map_err(|e| error(e.word, e.detail))?;
    let stage = match program.version.token() {
        0xfffe_0300 => Stage::Vertex,
        0xffff_0300 => Stage::Pixel,
        _ => {
            return Err(error(
                0,
                format!("version {}: only vs_3_0 and ps_3_0 are framed", program.version),
            ));
        }
    };
    Ok((stage, span))
}
