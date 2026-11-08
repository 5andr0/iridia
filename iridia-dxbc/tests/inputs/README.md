# Real-data inputs

What this crate's real-data tests read, as tests/inputs/mod.rs declares them: the keys of the configuration file
`GAME_DATA` names (its layout is described by [primase-testdata](https://github.com/Kalmarv/primase-testdata)), the oracle sets and the files of each, and the debug keys
`GAME_DATA_DEBUG` sets. A test whose input is not configured skips with a line naming its key.

<!-- real-data inputs: made by primase_testdata::doc::tables from tests/inputs/mod.rs -->

#### Configuration keys

| key | gate | about | source |
| --- | --- | --- | --- |
| `tools.fxc` | - | Microsoft's `fxc` shader compiler executable | third-party: Microsoft's fxc |
| `tools.vkd3d_compiler` | - | vkd3d's `vkd3d-compiler` program (its Shader Model 4 and 5 reader and disassembler), with vkd3d's command line | third-party: vkd3d (LGPL-2.1-or-later), built from source and run as a program, never linked |
| `tools.vkd3d_dxbc` | - | vkd3d's `vkd3d-dxbc` program (its container reader, digest check and writer), with vkd3d's command line | third-party: vkd3d (LGPL-2.1-or-later), built from source and run as a program, never linked |
| `tools.tint` | - | Dawn's `tint` compiler program (Chrome's WGSL reader, `tint --format wgsl <file>`), which the WGSL comparison holds a writer's browser-profile text to (`shader.tint`) | third-party: Google's Dawn, built from source and run as a program, never linked |
| `tools.warp` | yes | the WARP oracle program that runs a shader on Microsoft's WARP rasterizer | third-party: Microsoft's WARP rasterizer, through a thin host program that is not part of this repository |
| `tools.wgsl_run` | - | the WGSL harness that runs a WGSL program's entry point on Direct3D 12's WARP adapter through wgpu (tools/wgsl-run, built with `cargo build --release` there) | independent: wgpu and Microsoft's WARP rasterizer, through tools/wgsl-run of this repository (its own workspace, outside iridia's) |
| `tools.wgsl_run_dxc` | - | the `dxcompiler.dll` the WGSL harness compiles with (DXC 1.8.2502 or later, wgpu 30's minimum, `dxil.dll` beside it): the programs that need Shader Model 6 (subgroup operations, 16-bit and 64-bit types); FXC compiles the others where it is not configured | third-party: Microsoft's DirectX Shader Compiler release build |
| `scratch.root` | - | where real-data tests write their outputs (the OS temporary folder when unset) | - |

#### Oracle sets (`<oracles.root>/<set>/<game>/`)

| set | about | source |
| --- | --- | --- |
| `oracles.corpus` | the games' own files cut from their installs (shader containers, EBX documents, Havok tagfiles, DCX files, CAS frames) | the user's own installs: files extracted as stored |
| `oracles.d3d-disassembler` | Microsoft's D3D shader disassembly of a game's shader containers (`NNNN.dxbc` with its listing) | independent: Microsoft's D3D Shader Disassembler, run over the user's own install |

#### Declared files

| key | status | gate | about | source |
| --- | --- | --- | --- | --- |
| `oracles.corpus.elden/shaders` | live | - | a folder (searched recursively) of the DXIL containers of Elden Ring's Shader Model 6 shader bundles (one file per container), lifted, validated and written as WGSL, the refusals counted by reason with an example program; iridia-dxil/tests/lift_corpus.rs | the user's own install: the containers of its `/shader/*.shaderbnd.dcx` binders and `/shader/*.shaderbdlebnd.dcx` bundles (ray-tracing `[rt]` bundles included); primase's [corpus_elden_shaders example](https://github.com/Kalmarv/primase/tree/main/worlds/fromsoft/examples/corpus_elden_shaders.rs) extracts them |
| `oracles.corpus.general/dxbc` | live | yes | Sekiro's 390 pixel programs (`.ppo`), Microsoft's disassembly of each (`.ppo.asm`) and the bindings, inputs and constant layout read from that text (`.ppo.reflection.json`); iridia-dxbc/tests/disassembly.rs, reflection.rs; schema in iridia-dxbc/README.md | independent: the game's own programs (`corpus.sekiro/shaders`) disassembled by Microsoft's fxc 10.0.26100 (`fxc /nologo /dumpbin`), the reflection read from the disassembly's text by iridia-dxbc/tests/oracles/dxbc_corpus.py |
| `oracles.corpus.general/slim` | live | yes | the slim shader set: 8,287 DXBC, DXIL and Direct3D 9 programs (`programs/<family>/<sha256>.<family>`, Microsoft's listing beside each, `slim.tsv` saying why each is there) that reach every region and branch side of the lifters the full pool of 108,394 reaches; tests/ir_gate.rs gates its DXIL programs | independent: primase-oracles' `producers/slim-set`, a set cover of the disassembly features of the oracle home's game corpora, its disassembler sets, ten outside projects' test corpora and FidelityFX SDK v1.1.4 |
| `oracles.corpus.riders/shaders` | live | - | a folder (searched recursively) of the DXBC containers of Riders Republic's shader pack (ShaderDataDX11.bin, split at its containers), lifted, validated and written as WGSL, the refusals counted by reason; iridia-dxbc/tests/lift_corpus.rs | the user's own install: the programs split from its shader pack |
| `oracles.corpus.sekiro/d3d11-shaders` | live | - | a folder of the DXBC containers of Sekiro's Direct3D 11 programs of every stage (`<hash>.<stage>.dxbc`: 9,680, its hull, domain and geometry programs among them), each one's reflection held to Microsoft's listing of it; iridia-dxbc/tests/reflection.rs | the user's own install: the programs extracted from its shader archives |
| `oracles.corpus.sekiro/shaders` | live | - | a folder (searched recursively) of the DXBC containers of Sekiro's extracted pixel and vertex programs (`.ppo`, `.vpo`), lifted, validated and written as WGSL, the refusals counted by reason; iridia-dxbc/tests/lift_corpus.rs; the same programs are used in primase's [detail_derive test](https://github.com/Kalmarv/primase/tree/main/worlds/fromsoft/tests/detail_derive.rs) | the user's own install: the programs extracted from its shader archives |
| `oracles.corpus.skate/shaders` | live | - | a folder (searched recursively) of the DXIL containers of skate.'s CompiledBytecode resources (one file per container), lifted, validated and written as WGSL, the refusals counted by reason with an example program; iridia-dxil/tests/lift_corpus.rs | the user's own install: the containers of its superbundles' CompiledBytecode resources; primase's [corpus_skate_shaders example](https://github.com/Kalmarv/primase/tree/main/worlds/reskate/examples/corpus_skate_shaders.rs) extracts them |
| `oracles.corpus.sledding/shaders` | live | - | a folder (searched recursively) of the DXBC containers of a Unity player's compiled pixel programs, read out of its shader assets, lifted, validated and written as WGSL, the refusals counted by reason; iridia-dxbc/tests/lift_corpus.rs | the user's own install: the programs read from its shader assets |
| `oracles.d3d-disassembler.riders/shader-pack` | live | - | a folder of DXBC containers `NNNN.dxbc` (Riders Republic's ShaderDataDX11.bin, Steam build 25394987, split at its containers: 2,443) each with `NNNN.txt`, Microsoft's D3D Shader Disassembler text of it; iridia-dxbc/tests/disassembly.rs | independent: Microsoft's D3D Shader Disassembler, run over the user's own install's pack (kept in a private cache) |

#### Debug keys (`GAME_DATA_DEBUG`)

| key | about |
| --- | --- |
| `vkd3d.limit` | how many containers of a corpus the vkd3d comparison reads (200 spread evenly by default; `all` reads every one) |
| `shader.diff_limit` | how many programs of a corpus the WARP comparison runs |
| `shader.lanes` | a part of a program's name: the WGSL comparison prints that program's WGSL and every lane that is not exact |
| `shader.warp_all` | `1` compares every program the WARP oracle can bind |
| `shader.tint` | `1` also writes each program's browser-profile WGSL and holds it to `tools.tint` |
| `shader.tint_keep` | a folder that keeps the WGSL Tint refuses, with what Tint said (`shader.tint`) |
| `shader.worst` | a file name prefix: each corpus's programs and their worst class are written to `<prefix>-<corpus>.txt` |
| `shader.census` | `1`: the compute census (compute_census.rs) lifts every compute program of the corpus and counts them by what a comparison would have to account for |
| `shader.compute` | `0` runs no compute program; unset runs a fixed sample (one program in 16 by file-name hash, and every program `COMPUTE_EXPECTED` names) with the vertex and pixel programs; `sample` runs that sample alone; `1` runs every compute program of a corpus with them, as they are bound by `exec_wgsl` (buffer-only and group-shared programs); `only` runs every one and nothing else |
| `shader.compute_runs` | how many times each compute job runs on WARP (2 by default): a word that differs between the runs is one the program leaves unordered |
| `shader.dump` | a folder: the WGSL of each program that runs is written to it, named by the program |
| `shader.only` | `\|`-separated parts of program names: only the programs of a corpus with one of them in the name run |
| `shader.writer` | `wgsl` runs the WGSL of iridia-wgsl in the WARP comparison; `naga` (the default) runs iridia-naga's |
| `shader.outside` | a folder of containers outside the configured corpora (walked for every DXBC container in it) that the IR gate and the WARP comparison run |

<!-- end of real-data inputs -->
