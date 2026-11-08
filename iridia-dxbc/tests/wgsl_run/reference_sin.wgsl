// An accurate `sin` for WGSL (Cephes' `sinf`: quarter-turn reduction in three parts, then the minimax polynomials of sin and
// cos on [-pi/4, pi/4]). On WARP it is within 8.5e-8 of the f64 reference over -1000..1000, where WARP's builtin `sin` is off by
// up to 9.6e-5 over the same arguments (past D3D's -100 pi..100 pi bound for the largest) (tools/wgsl-run/probes/sin_ref.py measures both). A double-float sin was tried
// first: DXC and FXC reassociate its error-free transforms away. `shader.sin=reference` appends this to a program's WGSL.
fn iridia_sin(x: f32) -> f32 {
    let k = round(x * 0.63661977236758134f);
    var r = x - k * 1.5703125f;
    r = r - k * 4.837512969970703125e-4f;
    r = r - k * 7.54978995489188216e-8f;
    let s = r * r;
    let sp = r + r * s * (-1.6666654611e-1f + s * (8.3321608736e-3f + s * -1.9515295891e-4f));
    let cp = 1.0f - 0.5f * s + s * s * (4.166664568298827e-2f + s * (-1.388731625493765e-3f + s * 2.443315711809948e-5f));
    let q = i32(k - 4.0f * floor(k * 0.25f));
    if q == 0 { return sp; }
    if q == 1 { return cp; }
    if q == 2 { return -sp; }
    return -cp;
}
fn iridia_sin4(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(iridia_sin(v.x), iridia_sin(v.y), iridia_sin(v.z), iridia_sin(v.w));
}
