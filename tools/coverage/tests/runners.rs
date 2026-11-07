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

/// The offloader's mesh pipelines (a mesh and a pixel program, `DispatchGroupCount` groups, its default rasterizer
/// state) and multisampled targets (`SampleCount`, resolved before the read back) draw and reach the image comparison.
/// The comparison itself needs a checkout of offload-golden-images, which the suite does not copy: the coverage job
/// holds it, making the table (`docs/coverage`, where these four pass) with the checkout and failing where it differs.
/// The amplification pipeline is refused, iridia-dxil not lifting amplification programs, and the stencil one is not
/// modelled.
#[test]
fn the_offload_mesh_and_multisampled_pipelines_draw_to_their_image_comparisons() {
    let outcomes = offload_outcomes(&[
        "Graphics/MSAA/MeshShaderTriangle.test",
        "Graphics/MSAA/SimpleTriangle.test",
        "Graphics/MeshShaders/SimpleAmplification.test",
        "Graphics/MeshShaders/SimpleLines.test",
        "Graphics/MeshShaders/SimpleTriangle.test",
        "Graphics/MeshShaders/StencilState.test",
    ]);
    for (test, outcome) in &outcomes {
        let want = match test.as_str() {
            "Graphics/MeshShaders/SimpleAmplification.test" => "refused: ",
            "Graphics/MeshShaders/StencilState.test" => "not-modelled: a stencil test",
            _ => "not-modelled: no checkout of offload-golden-images",
        };
        assert!(outcome.starts_with(want), "{test}: {outcome}");
        if want == "refused: " {
            assert!(outcome.contains("amplification"), "{test}: {outcome}");
        }
    }
}

/// Every atomic operation of an unordered-access view, not the add alone, reaches the device: vkd3d's uav-atomics
/// (InterlockedAnd, Or, Xor, Min and Max signed and unsigned, Exchange, CompareExchange and CompareStore, of typed and
/// structured views) held to its expected values, recorded from Direct3D on Windows, on every DXBC and DXIL run, and
/// the offload suite's `Interlocked*.resources.32` tests (each operation through an array of views) held to theirs.
/// Refused by name, by the lifters the runners execute (`iridia_dxbc::lift` and `iridia_dxil::lift`, not the IR's):
/// the section at line 122, a compare-exchange and compare-store of a `RWTexture2D` (neither lifter has an atomic
/// compare of a texture, only of a buffer), and the section at line 149, which carries on
/// with the texture it leaves (until line 161 binds a buffer in its place); and on the DXBC runs the section at line
/// 197, `atomic_umax` and `atomic_umin` of an `RWBuffer<int>`, the DXBC lifter taking an atomic's signedness from its
/// view (the IR's lowering to WGSL writes such a minimum or maximum as a compare-exchange loop).
#[test]
fn every_atomic_operation_of_a_view_reaches_the_device() {
    let mut seen = Vec::new();
    let mut wrong = Vec::new();
    for format in ["dxbc", "dxil"] {
        for r in vkd3d_runs(&format!("vkd3d/{format}.json.gz"), "uav-atomics.shader_test") {
            for p in &r.probes {
                let (class, why) = match p.line {
                    122..161 if format == "dxbc" => ("refused", "on a texture view"),
                    122..161 => ("refused", "not an unordered-access buffer"),
                    197..249 if format == "dxbc" => ("refused", "on memory whose atomics are signed"),
                    _ => ("pass", ""),
                };
                if p.outcome.class() != class || !p.outcome.detail().contains(why) {
                    wrong.push(format!(
                        "{} {} line {}: {:?}",
                        r.runner, r.model, p.line, p.outcome
                    ));
                }
                seen.push((format, p.line));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{}",
        wrong.join(
            "
"
        )
    );
    // the first probe of each of the seven sections, on both formats
    for format in ["dxbc", "dxil"] {
        for line in [39, 86, 125, 153, 201, 253, 375] {
            assert!(
                seen.contains(&(format, line)),
                "the {format} probe at line {line} is run"
            );
        }
    }
    let outcomes = offload_outcomes(&[
        "Feature/HLSLLib/InterlockedAnd.resources.32.test",
        "Feature/HLSLLib/InterlockedExchange.resources.32.test",
        "Feature/HLSLLib/InterlockedMax.resources.32.test",
        "Feature/HLSLLib/InterlockedMin.resources.32.test",
        "Feature/HLSLLib/InterlockedOr.resources.32.test",
        "Feature/HLSLLib/InterlockedXor.resources.32.test",
    ]);
    for (test, outcome) in &outcomes {
        assert_eq!(outcome, "pass: ", "{test}");
    }
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

/// A vector probe steps 16 bytes through its target, as vkd3d reads any vector as a `vec4`: `probe uav-2 (4)
/// f32(-3.0, 22.0)` reads words 16 and 17, not 8 and 9.
#[test]
fn a_vector_probe_steps_a_vec4() {
    for cases in ["vkd3d/dxbc.json.gz", "vkd3d/dxil.json.gz"] {
        let mut runs = vkd3d_runs(cases, "uav-rwstructuredbuffer.shader_test");
        // the 6.2 run's 16-bit structures are refused (the next test)
        runs.retain(|r| r.model != "6.2");
        assert_selected_pass(&runs);
    }
}

/// iridia-dxil refuses the 16-bit lanes of a raw or structured buffer, which it does not pack two to a word, where it
/// wrote each as a word of its own (uav-rwstructuredbuffer's `half`/`int16_t` structures at 6.2).
#[test]
fn sixteen_bit_lanes_of_a_structured_buffer_are_refused() {
    let runs = vkd3d_runs("vkd3d/dxil.json.gz", "uav-rwstructuredbuffer.shader_test");
    let sixteen: Vec<_> = runs.iter().filter(|r| r.model == "6.2" && r.selected()).collect();
    assert_eq!(sixteen.len(), 1, "the 16-bit structures run at 6.2");
    for r in sixteen {
        let why = r.deciding().map(|p| p.outcome.detail()).unwrap_or_default();
        assert_eq!(r.class(), "refused", "{} {}: {why}", r.runner, r.model);
        assert!(
            why.contains("16-bit lanes of a raw or structured buffer"),
            "{why}"
        );
    }
}

/// `Texture2DMS::GetDimensions` reads the sample count from the fourth lane of DXIL's `getDimensions`, which is the
/// level count only for a texture that is not multisampled.
#[test]
fn a_multisampled_texture_states_its_sample_count() {
    assert_selected_pass(&vkd3d_runs("vkd3d/dxil.json.gz", "getdimensions.shader_test"));
}

/// A dispatch's store to a typed buffer reaches every lane it marks: the executor's word `4 * element + lane` is that
/// lane of that element (barriers.shader_test's `u[0] = 42` writes all four lanes).
#[test]
fn a_dispatch_stores_every_lane_of_a_typed_buffer_element() {
    assert_selected_pass(&vkd3d_runs("vkd3d/dxil.json.gz", "barriers.shader_test"));
}

/// A constant row of four minimum-precision lanes (`cbufferLoadLegacy.f16` with four lanes) holds each lane's half in
/// the low 16 bits of its word: arithmetic-float-uniform's `min16float u[4]` reads 0x4180 as 2.75.
#[test]
fn a_minimum_precision_constant_is_the_half_in_its_word() {
    let runs = vkd3d_runs("vkd3d/dxil.json.gz", "arithmetic-float-uniform.shader_test");
    let probes: Vec<_> = runs
        .iter()
        .filter(|r| r.model == "6.0")
        .flat_map(|r| &r.probes)
        .filter(|p| p.line == 359)
        .collect();
    assert_eq!(probes.len(), 1);
    assert_eq!(probes[0].outcome.class(), "pass", "{:?}", probes[0].outcome);
}

/// A hull program without a control-point phase passes its input control points through: fxc leaves the phase out
/// where the program returns `input[i]`, and the domain program then reads the vertex program's outputs, not zeros.
#[test]
fn a_hull_program_without_a_control_point_phase_passes_its_points_through() {
    assert_selected_pass(&vkd3d_runs("vkd3d/dxbc.json.gz", "tessellation.shader_test"));
}

/// Direct3D 9's shader model 2 clamps a vertex program's colours to [0, 1] on their way to the pixel program, and 3
/// does not: color-clamping reads (-2, 1, 2, -1) / 10 as (0, 0.1, 0.1, 0) at 2.0 and as itself at 3.0.
#[test]
fn shader_model_2_clamps_colours_between_the_stages() {
    let runs = vkd3d_runs("vkd3d/d3d9.json.gz", "color-clamping.shader_test");
    assert_eq!(
        runs.iter().filter(|r| r.selected()).count(),
        2,
        "the 2.0 and 3.0 runs"
    );
    assert_selected_pass(&runs);
}

/// A pixel input reads the output lanes of its own components: a Direct3D 9 input declared `v0.xw` (interface-packing,
/// the signature mask not contiguous) reads COLOR's x and w, where pairing lanes by count read its x and y.
#[test]
fn a_pixel_input_reads_the_lanes_of_its_components() {
    assert_selected_pass(&vkd3d_runs("vkd3d/d3d9.json.gz", "interface-packing.shader_test"));
}

/// Direct3D 9 reads a pixel input its vertex program leaves unwritten: a texture coordinate as 0, and at shader model
/// 2 a colour as white (uninitialized-varyings), where a later API's pipeline does not link. The file's probe of the
/// white is `winbug(amd)`, every Windows driver but AMD's giving it, so the cases leave it out as a probe of the
/// hardware; it is put back here.
#[test]
fn direct3d_9_reads_unwritten_inputs_as_zero_and_shader_model_2_colours_as_white() {
    use vkd3d::cases::{Step, What};
    let text = read_text(&suite("vkd3d/d3d9.json.gz")).expect("the vkd3d case file");
    let mut file = vkd3d::cases::read(&text).expect("the vkd3d case file");
    file.cases
        .retain(|r| r.file == "uninitialized-varyings.shader_test");
    let two = file
        .cases
        .iter_mut()
        .find(|r| r.model == "2.0")
        .expect("the 2.0 run");
    assert!(
        matches!(
            two.steps.last(),
            Some(Step {
                line: 43,
                what: What::Draw { .. }
            })
        ),
        "the 2.0 run ends with the colour draw at line 43"
    );
    two.steps.push(Step {
        line: 44,
        what: What::Probe {
            probe: [0, 0, 1, 1],
            target: "rtv-0".into(),
            kind: "f32".into(),
            count: 4,
            expect: vec![1f32.to_bits(); 4],
            ulps: 0,
            layer: 0,
        },
    });
    let runs = vkd3d::run::run_all(&file, 4).0;
    assert_selected_pass(&runs);
    let two = runs.iter().find(|r| r.model == "2.0").expect("the 2.0 run");
    assert!(
        two.probes.iter().any(|p| p.line == 44 && p.selected),
        "the white probe is selected"
    );
}

/// A Direct3D 9 `ps_1_x` `tex` reads its texture stage's state, which neither WGSL nor `iridia-exec` reads: a draw
/// makes the program again for its own state, the stage's projection flag (`projected`, ps1-projection) and the
/// dimension of the texture bound at the stage (ps1-sampler's 2D, volume and cube sections, to line 107; its later
/// sections wait on a `ps_1_x` program's colour output being saturated).
#[test]
fn a_direct3d_9_draw_specialises_its_programs_by_the_texture_stage_state() {
    assert_selected_pass(&vkd3d_runs("vkd3d/d3d9.json.gz", "ps1-projection.shader_test"));
    let runs = vkd3d_runs("vkd3d/d3d9.json.gz", "ps1-sampler.shader_test");
    let mut lines = Vec::new();
    for r in &runs {
        for p in r.probes.iter().filter(|p| p.selected && p.line <= 107) {
            assert_eq!(
                p.outcome.class(),
                "pass",
                "{} {} line {}: {:?}",
                r.file,
                r.model,
                p.line,
                p.outcome
            );
            lines.push(p.line);
        }
    }
    // the volume's probes and the cube's
    for line in [67, 70, 92, 107] {
        assert!(lines.contains(&line), "the probe at line {line} is selected");
    }
}

/// Direct3D 9's fog, as vkd3d-shader applies it: vertex fog from the vertex program's coordinate, and table fog,
/// linear, exp and exp2, from the position's z with an orthographic projection and its w otherwise (fog.shader_test).
#[test]
fn direct3d_9_fogs_the_first_target_by_its_mode_and_source() {
    let runs = vkd3d_runs("vkd3d/d3d9.json.gz", "fog.shader_test");
    assert_selected_pass(&runs);
    // vertex fog, linear, exp and exp2 from z, and linear from w
    let lines: Vec<u64> = runs
        .iter()
        .flat_map(|r| &r.probes)
        .filter(|p| p.selected)
        .map(|p| p.line)
        .collect();
    for line in [69, 76, 83, 90, 148] {
        assert!(lines.contains(&line), "the probe at line {line} is selected");
    }
}

/// Direct3D 12's depth bounds pass a pixel only where the depth buffer's value is within them, ends included, and a
/// program that forces early tests (`[earlydepthstencil]`) runs only where they pass, so its UAV store is not made
/// where they fail (depth-bounds.shader_test).
#[test]
fn the_depth_bounds_and_early_tests_hold_back_a_pixel_and_its_program() {
    assert_selected_pass(&vkd3d_runs("vkd3d/dxbc.json.gz", "depth-bounds.shader_test"));
}

/// A multisampled target is drawn sample by sample at Direct3D's standard positions: a centroid input is read at the
/// first covered sample where the primitive does not cover the pixel's every sample (sm1-centroid, whose expected
/// offsets are the 4x pattern's), the sample mask and the covered samples make `SV_Coverage` and the samples written,
/// a probe reads the target resolved (coverage.shader_test's input section, lines 49 to 77), and a load of a
/// multisampled texture reads its sample (texture-load's `Texture2DMS` loads, lines 171 to 200).
#[test]
fn a_multisampled_target_is_drawn_and_read_by_its_samples() {
    assert_selected_pass(&vkd3d_runs("vkd3d/d3d9.json.gz", "sm1-centroid.shader_test"));
    assert_selected_pass(&vkd3d_runs("vkd3d/dxbc.json.gz", "texture-load.shader_test"));
    let mut seen = Vec::new();
    for r in vkd3d_runs("vkd3d/dxil.json.gz", "coverage.shader_test") {
        // lines 18 and 22 read the coverage a program writes, which the executor does not hand the draw yet
        for p in r.probes.iter().filter(|p| p.selected && p.line >= 49) {
            assert_eq!(
                p.outcome.class(),
                "pass",
                "{} {} line {}: {:?}",
                r.file,
                r.model,
                p.line,
                p.outcome
            );
            seen.push(p.line);
        }
    }
    for line in [49, 50, 72, 77] {
        assert!(seen.contains(&line), "the probe at line {line} is selected");
    }
}

/// A mesh dispatch runs the mesh program's thread groups and draws the primitives they emit with the pixel program,
/// held to vkd3d's expected values: mesh.shader_test's sections without an amplification program (lines 35, 66, 101
/// and 143: the output counts the first thread sets, vertices and indices written by several threads, and each
/// coordinate by a thread of its own). Refused by name: the pixel program reading primitive outputs (line 205, whose
/// probe is `winbug(amd)`, every Windows driver but AMD's giving it, so the cases leave it out as a probe of the
/// hardware; it is put back here), since the IR carries no per-primitive outputs, and the sections with an
/// amplification program, since iridia-dxil does not lift amplification programs or a mesh program's payload.
#[test]
fn a_mesh_dispatch_draws_the_primitives_its_program_emits() {
    use vkd3d::cases::{Step, What};
    let text = read_text(&suite("vkd3d/dxil.json.gz")).expect("the vkd3d case file");
    let mut file = vkd3d::cases::read(&text).expect("the vkd3d case file");
    file.cases.retain(|r| r.file == "mesh.shader_test");
    assert!(!file.cases.is_empty(), "mesh.shader_test has runs");
    for case in &mut file.cases {
        let at = case
            .steps
            .iter()
            .position(|s| s.line == 199 && matches!(s.what, What::DispatchMesh { .. }))
            .expect("the primitive output section's dispatch at line 199");
        case.steps.insert(
            at + 1,
            Step {
                line: 205,
                what: What::Probe {
                    probe: [0, 0, 640, 480],
                    target: "rtv-0".into(),
                    kind: "f32".into(),
                    count: 4,
                    expect: [0.25f32, 0.5, 0.75, 1.0].map(f32::to_bits).to_vec(),
                    ulps: 0,
                    layer: 0,
                },
            },
        );
    }
    let runs = vkd3d::run::run_all(&file, 4).0;
    assert!(
        runs.iter().any(|r| r.selected()),
        "a run of mesh.shader_test is selected"
    );
    let mut seen = Vec::new();
    for r in &runs {
        for p in &r.probes {
            let (class, why) = match p.line {
                ..205 => ("pass", ""),
                205 => ("refused", "per-primitive outputs"),
                _ => ("not modelled", "amplification"),
            };
            assert_eq!(
                p.outcome.class(),
                class,
                "{} {} line {}: {:?}",
                r.file,
                r.model,
                p.line,
                p.outcome
            );
            assert!(
                p.outcome.detail().contains(why),
                "line {}: {:?}",
                p.line,
                p.outcome
            );
            seen.push(p.line);
        }
    }
    for line in [35, 66, 101, 143, 205] {
        assert!(seen.contains(&line), "the probe at line {line} is run");
    }
}
