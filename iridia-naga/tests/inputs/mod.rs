//! Optional independent execution of the emitted helper calls, declared beside the tests that use it.
primase_testdata::inputs! {
    games {}
    installs {}
    keys {}
    tools {
        WgslRun = "wgsl_run" {
            about: "the WGSL entry-point harness (tools/wgsl-run), running the emitted helper calls through wgpu on Direct3D 12 WARP",
            source: "independent: wgpu and Microsoft's WARP rasterizer",
            gate: false,
        },
        WgslRunDxc = "wgsl_run_dxc" {
            about: "the optional dxcompiler.dll used by the WGSL harness; FXC compiles these 32-bit fixtures when absent",
            source: "third-party: Microsoft's DirectX Shader Compiler",
            gate: false,
        },
    }
    oracle_sets {}
    declared {}
    debug {}
}
