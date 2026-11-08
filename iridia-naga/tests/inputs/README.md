# Independent execution tools

The optional WARP harness executes emitted helper calls without the temporary native-oracle inliner.

<!-- real-data inputs: made by primase_testdata::doc::tables from tests/inputs/mod.rs -->

#### Configuration keys

| key | gate | about | source |
| --- | --- | --- | --- |
| `tools.wgsl_run` | - | the WGSL entry-point harness (tools/wgsl-run), running the emitted helper calls through wgpu on Direct3D 12 WARP | independent: wgpu and Microsoft's WARP rasterizer |
| `tools.wgsl_run_dxc` | - | the optional dxcompiler.dll used by the WGSL harness; FXC compiles these 32-bit fixtures when absent | third-party: Microsoft's DirectX Shader Compiler |
| `scratch.root` | - | where real-data tests write their outputs (the OS temporary folder when unset) | - |

<!-- end of real-data inputs -->
