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

/// That every probe of the runs that vkd3d's runner selects holds.
fn assert_selected_pass(runs: &[vkd3d::run::RunResult]) {
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
