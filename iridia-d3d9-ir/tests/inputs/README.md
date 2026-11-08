# Real-data inputs

What this crate's real-data tests read, as tests/inputs/mod.rs declares them: the keys of the configuration file
`GAME_DATA` names (its layout is described by [primase-testdata](https://github.com/Kalmarv/primase-testdata)), the oracle sets and the files of each, and the debug keys
`GAME_DATA_DEBUG` sets. A test whose input is not configured skips with a line naming its key.

<!-- real-data inputs: made by primase_testdata::doc::tables from tests/inputs/mod.rs -->

#### Configuration keys

| key | gate | about | source |
| --- | --- | --- | --- |
| `tools.wgsl_run` | - | the WGSL harness that runs a WGSL program's entry point on Direct3D 12's WARP adapter through wgpu (tools/wgsl-run, built with `cargo build --release` there) | independent: wgpu and Microsoft's WARP rasterizer, through tools/wgsl-run of this repository (its own workspace, outside iridia's) |
| `tools.wgsl_run_dxc` | - | the `dxcompiler.dll` the WGSL harness compiles with (DXC 1.8.2502 or later, `dxil.dll` beside it); FXC compiles where it is not configured | third-party: Microsoft's DirectX Shader Compiler release build |
| `scratch.root` | - | where real-data tests write their outputs (the OS temporary folder when unset) | - |

#### Oracle sets (`<oracles.root>/<set>/<game>/`)

| set | about | source |
| --- | --- | --- |
| `oracles.d3d-disassembler` | Microsoft's D3D shader disassembly of a game's shader containers (`NNNN.dxbc` with its listing) | independent: Microsoft's D3D Shader Disassembler, run over the user's own install |

#### Declared files

| key | status | gate | about | source |
| --- | --- | --- | --- | --- |
| `oracles.d3d-disassembler.general/sm3` | live | yes | privately extracted raw ps_3_0 (`pixel.bin`) and two vs_3_0 programs (`vertex.bin`, `vertex-layered.bin`), each with Microsoft's `.asm` disassembly; tests/disassembly.rs checks all executable operands, declarations, definitions and CTAB allocations | independent: Microsoft d3dcompiler_47.dll D3DDisassemble via iridia-d3d9/tests/oracles/disassemble.py, on a shipped Shaun White Skateboarding SDB2 v39 pixel shader (record 5) and vertex shaders (records 5 and 150, ekshadersPChigh.bin); raw bytes and disassembly stay outside the repository |
| `oracles.d3d-disassembler.shaun-white/sdb2` | live | - | a folder of every distinct vertex (`.vs`) and pixel (`.ps`) program of the three SDB2 bundles (ekshadersPChigh, ekshadersPCmed, ekshadersPClow), each raw token stream with Microsoft's `.asm` disassembly beside it; tests/corpus.rs | independent: Microsoft d3dcompiler_47.dll D3DDisassemble over the programs of the user's own Shaun White Skateboarding install (extracted by a read-only script); raw bytes and disassembly stay outside the repository |

#### Debug keys (`GAME_DATA_DEBUG`)

| key | about |
| --- | --- |
| `shader.graphics` | `1` runs every program in the WARP comparison (tests/warp); `0` none; unset runs a fixed sample (one program in `shader.graphics_every`, 128 by default, by file-name hash), or every program `shader.only` names |
| `shader.graphics_every` | the WARP comparison's sample takes one program in this many (128 by default; `shader.graphics`) |
| `shader.only` | `\|`-separated parts of program names: only the programs of a corpus with one of them in the name run on WARP |
| `shader.lanes` | a part of a program's name: the WARP comparison prints that program's WGSL and every lane that is not exact |
| `shader.dump` | a folder: the WGSL of each program that runs on WARP is written to it, named by the program and the writer |
| `shader.pow` | `defined`: the WARP comparison swaps `pow` in each program's WGSL for one that gives a zero base and an exponent of 0 or less libm's value (WGSL leaves it undefined), to tell programs whose differences are that case from the others |
| `shader.sin` | `reference`: the WARP comparison swaps `sin` in each program's WGSL for an accurate one (within 8.5e-8 of the reference over -1000..1000; WARP's builtin is off by up to 9.6e-5 over the same arguments), to tell programs whose differences are WARP's `sin` from the others; a program with a `sin` or `cos` of another form is a gap |

<!-- end of real-data inputs -->
