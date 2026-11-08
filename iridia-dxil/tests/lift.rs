#![allow(clippy::unwrap_used)]
//! Lifting the fixtures (dxc 1.7, default optimisation and -Od): each lifts, validates, is prepared and written as WGSL
//! by iridia-wgsl, and naga reads the text back.
use iridia_dxil::ir::lift;
use iridia_ir::interface::Stage;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}

/// Fixtures the lift refuses by name, with the reason. The refusal is the outcome asserted, not a skipped file.
const LIFT_REFUSED: [(&str, &str); 2] = [
    // a library of several programs: its programs are the per-export fixtures beside it
    ("ray_subobjects.dxil", "expected exactly one entry point"),
    // the sample count is the pipeline's, not the program's (iridia-dxbc's tests/sample_count.rs checks every lift)
    (
        "sample_count.dxil",
        "the sample count of the rasterizer is the pipeline's",
    ),
];

/// Fixtures the writers refuse by name, with the reason: what WGSL has no form for. The refusal is the outcome asserted.
const WRITE_REFUSED: [(&str, &str); 7] = [
    // WGSL's depth gather reads the red channel (iridia-dxbc's tests/depth_gather.rs checks both lifts)
    ("gather_cmp_green.dxil", "a gather of lane 1 of a depth texture"),
    // WGSL's texture atomics return no old value
    ("image_atomic_results.dxil", "whose old value is read"),
    ("image_atomic_results.od.dxil", "whose old value is read"),
    ("image_atomic_results.66.dxil", "whose old value is read"),
    // a ray query's candidate says whether it is opaque, which WGSL's query does not
    (
        "ray_query_object.dxil",
        "candidate procedural primitive is not opaque",
    ),
    // a vertex program with no position feeds a tessellator: it is a function, not an entry point
    ("tessellation.vs.dxil", "feeds a tessellator"),
    ("tessellation.vs.od.dxil", "feeds a tessellator"),
];

/// The text as wgpu's naga reads it: parsed and validated with every capability.
fn read_back(text: &str) -> Result<(), String> {
    let module = naga_hlsl::front::wgsl::parse_str(text).map_err(|e| e.emit_to_string(text))?;
    naga_hlsl::valid::Validator::new(
        naga_hlsl::valid::ValidationFlags::all(),
        naga_hlsl::valid::Capabilities::all(),
    )
    .validate(&module)
    .map(|_| ())
    .map_err(|e| e.to_string())
}

#[test]
fn every_fixture_lifts() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".dxil"))
        .collect();
    names.sort();
    let mut failures = Vec::new();
    let mut written = 0;
    for name in &names {
        let bytes = fixture(name);
        let mut program = match lift(&bytes) {
            Ok(p) => p,
            Err(e) => {
                match LIFT_REFUSED.iter().find(|(n, _)| n == name) {
                    Some((_, reason)) if e.to_string().contains(reason) => {}
                    _ => failures.push(format!("{name}: lift: {e}")),
                }
                continue;
            }
        };
        if let Some((_, reason)) = LIFT_REFUSED.iter().find(|(n, _)| n == name) {
            failures.push(format!("{name}: expected the lift to refuse: {reason}"));
            continue;
        }
        if let Err(problems) = iridia_ir::validate(&program) {
            failures.push(format!("{name}: validate: {problems:?}"));
            continue;
        }
        if let Err(e) = iridia_passes::prepare(&mut program) {
            failures.push(format!("{name}: prepare: {e}"));
            continue;
        }
        let refusal = iridia_wgsl::wgsl(&program, &iridia_wgsl::Profile::NAGA).map_err(|e| e.to_string());
        let expected = WRITE_REFUSED.iter().find(|(n, _)| n == name);
        match (refusal, expected) {
            (Ok(text), None) => match read_back(&text) {
                Ok(()) => written += 1,
                Err(e) => failures.push(format!("{name}: naga does not read the text back: {e}")),
            },
            (Err(e), Some((_, reason))) if e.contains(reason) => {}
            // neither IR writer writes the functions of a stage with no entry point (hull, domain, geometry, mesh,
            // amplification, ray tracing) yet; the refusal names the stage, and a writer that learns to write one makes
            // the program here a text to check, so this fails until its fixtures are written and read back
            (Err(e), None)
                if !matches!(program.stage, Stage::Vertex | Stage::Pixel | Stage::Compute)
                    && e.ends_with(&format!("{:?} program is not written yet", program.stage)) => {}
            (Err(e), _) => failures.push(format!("{name}: wgsl: {e}")),
            (Ok(_), Some((_, reason))) => {
                failures.push(format!("{name}: expected the writer to refuse: {reason}"))
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    assert!(written > 80, "{written} fixtures written");
}
