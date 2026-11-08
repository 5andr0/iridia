#![allow(clippy::unwrap_used)]

//! WGSL starts workgroup memory zeroed, and wgsl-run makes that so in the program's module (loops over the group's
//! invocations: `tools/wgsl-run/src/workgroup.rs`) instead of leaving it to wgpu's HLSL, which assigns each array whole in
//! the first invocation and takes WARP's compiler about a minute for 3,072 words. These hold the harness to both: a program
//! reads every workgroup word before it writes any and sees zeros, the same words as with wgpu's own initialisation, and a
//! program with a large array runs within the harness's GPU wait. Needs `tools.wgsl_run`.

mod inputs;
mod wgsl_run;

use serde_json::{Value, json};

/// Every kind of workgroup variable the initialisation handles: a lone scalar, an array, arrays of arrays of vectors, a
/// structure with an array, a vector and an atomic in it, and lone atomics. Each invocation sums all the words it can see
/// before the barrier and writes the sum (which must be 0), then leaves the memory dirty for the next group on the same
/// core, so an initialisation that is missing shows as a nonzero sum there.
fn program(extra: &str, extra_sum: &str) -> String {
    format!(
        "struct S {{ a: array<u32, 5>, b: vec4<u32>, c: atomic<u32> }};
var<workgroup> scalar: u32;
var<workgroup> plain: array<u32, 100>;
var<workgroup> nested: array<array<vec2<u32>, 3>, 7>;
var<workgroup> record: S;
var<workgroup> one: atomic<u32>;
{extra}
@group(0) @binding(0) var<storage, read_write> sums: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) i: u32) {{
    var sum = scalar;
    for (var k = 0u; k < 100u; k++) {{ sum += plain[k]; }}
    for (var a = 0u; a < 7u; a++) {{
        for (var b = 0u; b < 3u; b++) {{ sum += nested[a][b].x + nested[a][b].y; }}
    }}
    for (var k = 0u; k < 5u; k++) {{ sum += record.a[k]; }}
    sum += record.b.x + record.b.y + record.b.z + record.b.w;
    sum += atomicLoad(&record.c) + atomicLoad(&one);
    {extra_sum}
    workgroupBarrier();
    sums[group.x * 64u + i] = sum;
    scalar = i + 1u;
    plain[i] = 0xffffffffu;
    plain[i + 36u] = 0xffffffffu;
    nested[i % 7u][i % 3u] = vec2<u32>(i + 1u, 7u);
    record.a[i % 5u] = i + 1u;
    record.b = vec4<u32>(i + 1u);
    atomicAdd(&record.c, i + 1u);
    atomicAdd(&one, 1u);
}}
"
    )
}

fn job(wgsl: &str, init: Option<&str>) -> Value {
    let mut job = json!({
        "wgsl": wgsl,
        "dispatch": [16, 1, 1],
        "resources": [{ "group": 0, "binding": 0, "buffer": { "words": vec![0xdead_beefu32; 16 * 64] } }],
    });
    if let Some(init) = init {
        job["workgroup_init"] = json!(init);
    }
    job
}

fn words(reply: &Value) -> Vec<u64> {
    reply["resources"][0]["words"]
        .as_array()
        .unwrap_or_else(|| panic!("no words in {reply}"))
        .iter()
        .map(|w| w.as_u64().unwrap())
        .collect()
}

#[test]
fn the_modules_loops_zero_what_wgpus_assignment_does() {
    let Some(harness) = wgsl_run::harness() else {
        println!("tools.wgsl_run is not configured: skipped");
        return;
    };
    let wgsl = program("", "");
    let loops = words(&harness.run(&job(&wgsl, None)).unwrap());
    let wgpu = words(&harness.run(&job(&wgsl, Some("wgpu"))).unwrap());
    assert_eq!(loops.len(), 16 * 64);
    assert!(
        loops.iter().all(|&w| w == 0),
        "the harness's initialisation left words: {:?}",
        loops
            .iter()
            .enumerate()
            .filter(|(_, w)| **w != 0)
            .take(8)
            .collect::<Vec<_>>()
    );
    assert_eq!(loops, wgpu, "the two initialisations give different words");
}

#[test]
fn a_program_with_a_large_array_runs_within_the_gpu_wait() {
    let Some(harness) = wgsl_run::harness() else {
        println!("tools.wgsl_run is not configured: skipped");
        return;
    };
    // 4,096 words took WARP 115 s to compile with wgpu's assignment, against the harness's 30 s wait
    let wgsl = program(
        "var<workgroup> large: array<u32, 4096>;",
        "for (var k = i; k < 4096u; k += 64u) { sum += large[k]; } workgroupBarrier(); for (var k = i; k < 4096u; k += 64u) { large[k] = 0xffffffffu; }",
    );
    let started = std::time::Instant::now();
    let reply = harness.run(&job(&wgsl, None)).unwrap_or_else(|e| panic!("{e}"));
    assert!(words(&reply).iter().all(|&w| w == 0));
    assert!(
        started.elapsed().as_secs() < 20,
        "took {:?}: the large array is initialised by something slow again",
        started.elapsed()
    );
}
