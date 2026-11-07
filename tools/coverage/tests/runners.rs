//! The suites' cases that prove what the runners' device binds and draws: unordered-access counters, arrays of views
//! and draw parameters run through `iridia-exec` to the results the suites expect.
use std::path::{Path, PathBuf};

use iridia_coverage::inflate::read_text;
use iridia_coverage::{offload, vkd3d};

fn suite(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("suites").join(path)
}

/// The outcome of each offload test named, by name.
fn offload_outcomes(tests: &[&str]) -> Vec<(String, String)> {
    let text = read_text(&suite("offload/offload.json.gz")).expect("the offload case file");
    let mut file = offload::cases::read(&text).expect("the offload case file");
    file.cases.retain(|c| tests.contains(&c.test.as_str()));
    assert_eq!(
        file.cases.len(),
        tests.len(),
        "every test named is in the case file"
    );
    let mut out: Vec<(String, String)> = offload::run::run_all(&file, 4, None)
        .into_iter()
        .map(|r| (r.test, format!("{}: {}", r.outcome.class(), r.outcome.detail())))
        .collect();
    out.sort();
    out
}

fn assert_pass(outcomes: &[(String, String)]) {
    for (test, outcome) in outcomes {
        assert!(outcome.starts_with("pass"), "{test}: {outcome}");
    }
}

/// `IncrementCounter` and `DecrementCounter` update the hidden counter the offloader reads back as `Counter:` (one
/// buffer) and `Counters:` (an array of them).
#[test]
fn the_offload_counter_tests_read_their_counters_back() {
    assert_pass(&offload_outcomes(&[
        "Feature/StructuredBuffer/inc_counter.test",
        "Feature/StructuredBuffer/dec_counter.test",
        "Feature/StructuredBuffer/inc_counter_array.test",
        "Feature/StructuredBuffer/inc_counter_array_imm_idx.test",
    ]));
}

/// Arrays of views, stated and unbounded, bind element by element at consecutive registers.
#[test]
fn the_offload_resource_array_tests_bind_every_element() {
    assert_pass(&offload_outcomes(&[
        "Feature/ResourceArrays/array-global.test",
        "Feature/ResourceArrays/unbounded-array.test",
        "Feature/ResourceArrays/multi-dim-array-global.test",
    ]));
}

/// The runs of vkd3d test file `test` in case file `cases`.
fn vkd3d_runs(cases: &str, test: &str) -> Vec<vkd3d::run::RunResult> {
    let text = read_text(&suite(cases)).expect("the vkd3d case file");
    let mut file = vkd3d::cases::read(&text).expect("the vkd3d case file");
    file.cases.retain(|r| r.file == test);
    assert!(!file.cases.is_empty(), "{test} is in {cases}");
    vkd3d::run::run_all(&file, 4).0
}

/// That every probe of the runs that vkd3d's runner selects holds, and that there is one.
fn assert_selected_pass(runs: &[vkd3d::run::RunResult]) {
    let selected = runs.iter().flat_map(|r| &r.probes).filter(|p| p.selected).count();
    assert!(selected > 0, "no probe of {} runs is selected", runs.len());
    for r in runs {
        for p in r.probes.iter().filter(|p| p.selected) {
            assert_eq!(
                p.outcome.class(),
                "pass",
                "{} {} {} line {}: {:?}",
                r.file,
                r.runner,
                r.model,
                p.line,
                p.outcome
            );
        }
    }
}

/// vkd3d's `counter_buffer` resource is its own counter: the probe of its first word reads the count.
#[test]
fn the_vkd3d_counter_runs_count() {
    assert_selected_pass(&vkd3d_runs("vkd3d/dxil.json.gz", "uav-counter.shader_test"));
}

/// SV_VertexID and SV_InstanceID count from 0 whatever vertex and instance a draw starts at, and the vertex buffers
/// are read from the first vertex on.
#[test]
fn vertex_and_instance_ids_count_from_the_draws_start() {
    for cases in ["vkd3d/dxbc.json.gz", "vkd3d/dxil.json.gz"] {
        assert_selected_pass(&vkd3d_runs(cases, "instance-id.shader_test"));
    }
}

/// A `[descriptors]` section maps registers to resources, and an empty one is the default mapping again. (Its runs
/// at shader model 6.6 read through `ResourceDescriptorHeap`, which a later section of the file tests.)
#[test]
fn descriptor_sections_map_and_reset() {
    for cases in ["vkd3d/dxbc.json.gz", "vkd3d/dxil.json.gz"] {
        let mut runs = vkd3d_runs(cases, "descriptors.shader_test");
        runs.retain(|r| r.model != "6.6");
        assert!(!runs.is_empty(), "{cases}: runs below shader model 6.6");
        assert_selected_pass(&runs);
    }
}

/// A draw that reads a view a step before could not write is not judged: texture-load.shader_test copies a
/// multisampled target to a texture and draws with it, and each probe after has the outcome of the draw that made the
/// target, whether that draw ran or not.
#[test]
fn a_draw_that_reads_what_could_not_be_written_is_not_judged() {
    for cases in ["vkd3d/dxbc.json.gz", "vkd3d/dxil.json.gz"] {
        let runs = vkd3d_runs(cases, "texture-load.shader_test");
        let mut checked = 0;
        for r in &runs {
            let at = |line| r.probes.iter().find(|p| p.line == line).map(|p| &p.outcome);
            let Some(made) = at(171) else { continue };
            for line in [191, 194, 197, 200] {
                assert_eq!(
                    at(line),
                    Some(made),
                    "{} {} {} line {line}",
                    r.file,
                    r.runner,
                    r.model
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "{cases}: the multisampled runs of texture-load.shader_test"
        );
    }
}

/// A DXIL `i1` converted to a float or extended is the integer 1 (or -1, signed) where true: vkd3d's tests that turn
/// comparisons and booleans into numbers.
#[test]
fn dxil_booleans_convert_as_integers() {
    for test in [
        "all.shader_test",
        "bool-cast.shader_test",
        "cast-to-uint.shader_test",
        "logic-operations.shader_test",
    ] {
        assert_selected_pass(&vkd3d_runs("vkd3d/dxil.json.gz", test));
    }
}

/// DXBC's `ld` reads the texel its immediate offset (`aoffimmi`) moves it to.
#[test]
fn dxbc_loads_take_their_texel_offset() {
    for test in [
        "swizzle-constant-prop.shader_test",
        "texture-load-offset.shader_test",
    ] {
        assert_selected_pass(&vkd3d_runs("vkd3d/dxbc.json.gz", test));
    }
}

/// A DXIL program compiled with `-denorm preserve` keeps 32-bit denormals and one compiled with `-denorm ftz` flushes
/// them: arithmetic-float-uniform.shader_test's `denorm` sections at shader model 6.2 and up.
#[test]
fn dxil_denormal_modes_are_the_programs() {
    let runs = vkd3d_runs("vkd3d/dxil.json.gz", "arithmetic-float-uniform.shader_test");
    let mut checked = 0;
    for r in &runs {
        for p in r.probes.iter().filter(|p| [552, 556, 561, 564].contains(&p.line)) {
            assert_eq!(
                p.outcome.class(),
                "pass",
                "{} {} line {}: {:?}",
                r.runner,
                r.model,
                p.line,
                p.outcome
            );
            checked += 1;
        }
    }
    assert!(checked >= 8, "the denorm probes of two models or more, {checked}");
}

/// A Direct3D 9 program leaves it open whether 32-bit denormals are flushed, and vkd3d's d3d9 runs expect them flushed
/// (arithmetic-float-uniform.shader_test's probes after "Denormal behavior"): the run again with them flushed holds
/// there, and says so.
#[test]
fn d3d9_runs_hold_with_denormals_flushed() {
    let runs = vkd3d_runs("vkd3d/d3d9.json.gz", "arithmetic-float-uniform.shader_test");
    let mut checked = 0;
    for r in &runs {
        for p in r.probes.iter().filter(|p| [500, 503].contains(&p.line)) {
            assert_eq!(
                p.outcome,
                vkd3d::run::Outcome::PassFlushed,
                "{} {} line {}",
                r.runner,
                r.model,
                p.line
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 4, "the denormal probes at shader models 2.0 and 3.0");
}

/// Direct3D 9's alpha test writes a pixel only where its first target's alpha passes the comparison with the
/// reference, each of the eight comparisons.
#[test]
fn d3d9_alpha_test_compares_the_alpha() {
    assert_selected_pass(&vkd3d_runs("vkd3d/d3d9.json.gz", "alpha-test.shader_test"));
}

/// Direct3D 9's user clip planes clip in clip space, enabled and disabled by index.
#[test]
fn d3d9_clip_planes_clip() {
    assert_selected_pass(&vkd3d_runs("vkd3d/d3d9.json.gz", "clip-planes.shader_test"));
}

/// Direct3D 9's point size draws a point as the square of its size, from the state or the vertex program's `PSIZE`,
/// held to the state's range; a point sprite's texture coordinates are each pixel's place on it.
#[test]
fn d3d9_points_take_their_size_and_sprite_coordinates() {
    for test in [
        "ffp-point-size.shader_test",
        "shader-point-size.shader_test",
        "point-sprite.shader_test",
    ] {
        assert_selected_pass(&vkd3d_runs("vkd3d/d3d9.json.gz", test));
    }
}

/// A primitive's leading vertex picks the target slice (`SV_RenderTargetArrayIndex`) and the viewport
/// (`SV_ViewportArrayIndex`) it draws to, and the pixel program reads the index as written.
#[test]
fn target_and_viewport_indices_pick_the_slice_and_viewport() {
    for cases in ["vkd3d/dxbc.json.gz", "vkd3d/dxil.json.gz"] {
        for test in ["rt-array-index.shader_test", "vp-array-index.shader_test"] {
            assert_selected_pass(&vkd3d_runs(cases, test));
        }
    }
}

/// Direct3D 9 samples pixel `(x, y)` at `(x, y)`, half a pixel before a later API: the linear texture coordinate of
/// pixel (320, 0) is the quad's at x = 0, y = 1 exactly; its flat shade mode makes the colours the first vertex's and
/// leaves the texture coordinates interpolated; and `vPos` reads the pixel's whole coordinates.
#[test]
fn d3d9_draws_sample_at_whole_pixels_and_shade_colours_flat() {
    for test in ["shade-mode.shader_test", "pixel-position.shader_test"] {
        assert_selected_pass(&vkd3d_runs("vkd3d/d3d9.json.gz", test));
    }
}

/// A Direct3D 9 draw runs again from the IR, which must write what the WGSL's draw writes: the colour (`v#`) and
/// texture coordinate (`t#`) inputs share register numbers, and `vPos` and `vFace` have none, so the IR's
/// varyings take the WGSL's register view.
#[test]
fn d3d9_draws_from_the_ir_as_from_its_wgsl() {
    for test in [
        "many-semantics.shader_test",
        "sm1-interstage-interface.shader_test",
        "is-front-face.shader_test",
    ] {
        assert_selected_pass(&vkd3d_runs("vkd3d/d3d9.json.gz", test));
    }
}

/// A sample reduces its coordinate to its address mode's range before scaling it to texels (7.18.6), and a NaN left
/// is 0 (3.2.4.1): `tex2Dproj` with w = 0 samples at infinity, which clamps to the edge texel where linear filtering
/// of the infinite coordinate itself gives NaN.
#[test]
fn an_infinite_coordinate_samples_the_edge_texel_it_clamps_to() {
    assert_selected_pass(&vkd3d_runs("vkd3d/d3d9.json.gz", "sampler.shader_test"));
    // the Shader Model 4 runs: 5.0's has a `sample_clamp_feedback` iridia does not lift
    let mut runs = vkd3d_runs("vkd3d/dxbc.json.gz", "sampler.shader_test");
    runs.retain(|r| r.model == "4.0");
    assert_selected_pass(&runs);
}

/// A draw steps through a structured buffer by the stride its program declares, not the view's: vkd3d's
/// srv-structuredbuffer reads a view of 16-byte structures as 32-byte `float4[2]` elements.
#[test]
fn a_draw_addresses_a_structured_buffer_by_the_programs_stride() {
    for cases in ["vkd3d/dxbc.json.gz", "vkd3d/dxil.json.gz"] {
        assert_selected_pass(&vkd3d_runs(cases, "srv-structuredbuffer.shader_test"));
    }
}
