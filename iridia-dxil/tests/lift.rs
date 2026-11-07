#![allow(clippy::unwrap_used)]
//! Lifting the fixtures (dxc 1.7, default optimisation and -Od): each lifts, validates and writes WGSL.
use iridia_dxil::lift;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}

/// Fixtures whose program computes with a double (a 64-bit float made from or split into two 32-bit words): WGSL has no 64-bit
/// float, so the lifter refuses each by name. The refusal is the outcome asserted, not a skipped file.
const DOUBLE_FIXTURES: [&str; 1] = ["structured_tail.dxil"];

/// Fixtures whose program reads and writes 16-bit lanes of raw or structured buffers, which this lifter refuses by name
/// (the IR lift takes them: tests/ir_warp.rs). The refusal is the outcome asserted, not a skipped file.
const SIXTEEN_BIT_LANE_FIXTURES: [&str; 1] = ["buffer_lanes16.dxil"];

/// Fixtures of an amplification pipeline (an amplification program, a mesh program that reads its task payload), which
/// this lifter refuses by name (the IR lift takes them: tests/amplification.rs). The refusal is the outcome asserted.
const AMPLIFICATION_FIXTURES: [&str; 5] = [
    "amplification.dxil",
    "amplification_mesh.dxil",
    "payload_halves.dxil",
    "payload_layout.dxil",
    "payload_layout_mesh.dxil",
];

/// Fixtures that gather a depth texture's channel other than red, which WGSL's depth gather does not read: the lifter
/// refuses each by name (iridia-dxbc's tests/depth_gather.rs checks both lifts). The refusal is the outcome asserted.
const DEPTH_GATHER_FIXTURES: [&str; 1] = ["gather_cmp_green.dxil"];

/// Fixtures that read the render target's sample count, which is the pipeline's, not the program's: refused by name
/// (iridia-dxbc's tests/sample_count.rs checks every lift). The refusal is the outcome asserted.
const SAMPLE_COUNT_FIXTURES: [&str; 1] = ["sample_count.dxil"];

/// Fixtures that are a library of several programs, whose programs are the per-export fixtures beside them.
const WHOLE_LIBRARIES: [&str; 1] = ["ray_subobjects.dxil"];

#[test]
fn every_fixture_lifts() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".dxil"))
        // a whole library (every program of it, with its subobjects) is not one program: the lifter takes a library of
        // one, and the per-export containers of it are lifted instead
        .filter(|n| n != "ray_subobjects.dxil")
        .collect();
    names.sort();
    let mut failures = Vec::new();
    for name in &names {
        // a whole library holds several programs: `lift` refuses it by name, and each program of it is a container of
        // its own (ray_subobjects.<export>.dxil), which this loop lifts
        if WHOLE_LIBRARIES.contains(&name.as_str()) {
            match lift(&fixture(name)) {
                Err(e) if e.to_string().contains("not exactly one entry point") => {}
                other => failures.push(format!(
                    "{name}: expected the refusal of a whole library, got {other:?}"
                )),
            }
            continue;
        }
        if AMPLIFICATION_FIXTURES.contains(&name.as_str()) {
            match lift(&fixture(name)) {
                Err(e) if e.to_string().contains("amplification programs are not lifted") => {}
                other => failures.push(format!(
                    "{name}: expected the refusal of an amplification pipeline, got {other:?}"
                )),
            }
            continue;
        }
        if SAMPLE_COUNT_FIXTURES.contains(&name.as_str()) {
            match lift(&fixture(name)) {
                Err(e)
                    if e.to_string()
                        .contains("the sample count of the rasterizer is the pipeline's") => {}
                other => failures.push(format!(
                    "{name}: expected the refusal of the rasterizer's sample count, got {other:?}"
                )),
            }
            continue;
        }
        if DEPTH_GATHER_FIXTURES.contains(&name.as_str()) {
            match lift(&fixture(name)) {
                Err(e) if e.to_string().contains("a gather of lane 1 of a depth texture") => {}
                other => failures.push(format!(
                    "{name}: expected the refusal of a depth gather of lane 1, got {other:?}"
                )),
            }
            continue;
        }
        if DOUBLE_FIXTURES.contains(&name.as_str()) {
            match lift(&fixture(name)) {
                Err(e) if e.to_string().contains("double-precision floats are not lifted") => {}
                other => failures.push(format!(
                    "{name}: expected the refusal of double-precision floats, got {other:?}"
                )),
            }
            continue;
        }
        if SIXTEEN_BIT_LANE_FIXTURES.contains(&name.as_str()) {
            match lift(&fixture(name)) {
                Err(e)
                    if e.to_string()
                        .contains("16-bit lanes of a raw or structured buffer are not lifted") => {}
                other => failures.push(format!(
                    "{name}: expected the refusal of 16-bit buffer lanes, got {other:?}"
                )),
            }
            continue;
        }
        match lift(&fixture(name)) {
            Err(e) => failures.push(format!("{name}: lift: {e}")),
            Ok(p) => {
                if let Err(problems) = p.validate() {
                    failures.push(format!(
                        "{name}: validate: {}",
                        problems
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join("; ")
                    ));
                } else {
                    // a program of a stage with no WGSL stage is written as its functions
                    let written = if iridia::stage_functions(&p.interface).is_ok_and(|f| f.is_some()) {
                        iridia::emit::wgsl_functions(&p)
                    } else {
                        iridia::emit::wgsl(&p)
                    };
                    match written {
                        Ok(_) => {}
                        // WebGPU has no ray tracing pipeline: those programs are refused by name
                        Err(iridia::ProgramError::Unsupported { detail, .. })
                            if detail.contains("ray tracing pipeline") => {}
                        // WGSL's texture atomics return no old value and have no exchange: naga's writer refuses those by name
                        Err(iridia::ProgramError::Unsupported { detail, .. })
                            if detail.contains("texture atomic result")
                                || detail.contains("texture atomic exchange") => {}
                        Err(e) => failures.push(format!("{name}: wgsl: {e}")),
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
