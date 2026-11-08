#![allow(clippy::unwrap_used)]
//! The WGSL `iridia_naga` writes, run on Direct3D 12's WARP adapter through wgpu (the harness `tools.wgsl_run`, an
//! executor iridia does not write), against the IR executor (`ProgramShader::from_ir`): each vertex and pixel program of
//! a corpus the IR lift takes, after the passes the gate's P column runs, on the same input registers and constant
//! buffers, in `exec_warp`'s environment (uniform textures, point-filtering samplers clamped, so a sample is independent
//! of its coordinate). A vertex program runs once per vector; a pixel program as a 2x2 quad per vector, its lanes' float
//! inputs stepping along x and y as a rasterizer's do, so its derivatives are the quad's own. Every lane the executor
//! writes is compared, classed by how far apart (Direct3D leaves `exp`, `log`, `sin`, `cos`, `rsqrt` and `rcp` to a few
//! ulp): one further apart than 1e-3 relative, a discard on one side only, or a lane the WGSL does not carry fails.
//! Lanes within 2^-20 of each other where either is below 2^-10 are counted apart as "near zero", not failed: Riders
//! Republic's programs 2133, 2135 and their kin compute `1 - (a * 0.85 + 0.15)`, which the executor makes 0 and WARP
//! -2^-26. Not a fused `mad`: DXC folds the constants of that expression written in WGSL, `(1 - 0.15) - a * 0.85`
//! (`tools/wgsl-run/probes/mad.py`: WARP's words equal that evaluation for 4,096 of 4,096 values of `a`, the unfused
//! one for 2,621 and `fma` then subtract for 2,317), while the same expression as fxc's `mad` and `add` run on WARP
//! natively is the executor's bit for bit (`exec_warp.rs`, `a_mad_and_a_packed_quotient_agree_with_warp_bit_for_bit`).
//! The difference is DXC's fast-math reassociation of the WGSL, which WGSL allows, not a lift.
//!
//! A lane further apart than 1e-3 that is `fract(x * c)` for a constant `c` of magnitude 1e3 or more is counted as
//! "amplified", by source, where the difference implies `x`s no further apart than WGSL lets the operation that rounds
//! `x` be off by (`wgsl_run::amplified`): Elden Ring's ambient occlusion packs depth as `fract(saturate(a / b) * 65025)`,
//! where one ulp of the quotient is 1/256 of the lane. WARP's division is correctly rounded (`probes/div.py`: 4,096 of
//! 4,096 quotients, and `fract(saturate(a / b) * 65025)` apart for none), and the pack's `div_sat`, `mul` and `frc` as
//! fxc writes them are the executor's bit for bit on WARP; so the quotient differs through its numerator, a sum the
//! program accumulates (inferred, not isolated: its other lanes differ by up to 1e-5 relative in the same sums, where DXC
//! may reassociate as above), a few ulp off, which the factor turns into a lane 1/256 or 2/256 off. A lane that only
//! depends on such a value further down still fails, its line naming the amplification: Unity programs 03688, 03689 and 04137 to 04183 sum value noise of
//! `fract(sin(dot(p, k)) * 43758.5453)` hashes, and WARP's `sin` of arguments from 269 to 55424 is not correctly
//! rounded (tools/wgsl-run/probes/sin.py: 0 of 64 correctly rounded, median 9,103 ulp off, at most 296,737), which the
//! factor turns into hashes more than 1e-3 apart for 64 of 64; WARP running program 03688's own bytecode gives the
//! WGSL's values, not the executor's.
//!
//! Needs `tools.wgsl_run` (and `tools.wgsl_run_dxc`, the `dxcompiler.dll` that compiles the programs that need Shader
//! Model 6), and a corpus folder. `shader.diff_limit` caps the programs of a corpus.
mod common;
mod inputs;
mod wgsl_run;

mod wgsl_compare;
use wgsl_compare::{UNPROVABLE_BARRIERS, corpus, corpus_over, unprovable_barrier};

#[test]
fn sekiro_programs_run_as_naga_wgsl_on_warp() {
    corpus(inputs::Game::Sekiro);
}
#[test]
fn riders_republic_programs_run_as_naga_wgsl_on_warp() {
    corpus(inputs::Game::Riders);
}
#[test]
fn unity_programs_run_as_naga_wgsl_on_warp() {
    corpus(inputs::Game::Sledding);
}
#[test]
fn elden_programs_run_as_naga_wgsl_on_warp() {
    corpus(inputs::Game::Elden);
}

/// A folder of containers outside the configured corpora (`shader.outside`); without it, nothing to run.
#[test]
fn outside_programs_run_as_naga_wgsl_on_warp() {
    let Some(root) = inputs::debug("shader.outside").map(std::path::PathBuf::from) else {
        return;
    };
    corpus_over(&common::folder(&root));
}

/// A barrier failure is excused by the program's name and source hash together: a program of the same name with other
/// source, or other source of the same hash under another name, is a fault.
#[test]
fn a_barrier_failure_is_excused_only_for_a_listed_program() {
    let (name, hash, _) = UNPROVABLE_BARRIERS[0];
    assert!(unprovable_barrier(&format!("folder/{name}"), hash).is_some());
    assert!(unprovable_barrier(name, "sha256:0").is_none());
    assert!(unprovable_barrier(name, "-").is_none());
    assert!(unprovable_barrier("GXLightOrganize4.cpo", hash).is_none());
    for (n, h, why) in UNPROVABLE_BARRIERS {
        assert!(h.starts_with("sha256:") && h.len() == 71, "{n}: {h}");
        assert!(!why.is_empty());
    }
}
