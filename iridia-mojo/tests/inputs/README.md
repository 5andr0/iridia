# Real-data inputs

What this crate's real-data tests read, as tests/inputs/mod.rs declares them: the keys of the configuration file
`GAME_DATA` names (its layout is described by [primase-testdata](https://github.com/Kalmarv/primase-testdata)), the oracle sets and the files of each, and the debug keys
`GAME_DATA_DEBUG` sets. A test whose input is not configured skips with a line naming its key.

<!-- real-data inputs: made by primase_testdata::doc::tables from tests/inputs/mod.rs -->

#### Configuration keys

| key | gate | about | source |
| --- | --- | --- | --- |
| `scratch.root` | - | where real-data tests write their outputs (the OS temporary folder when unset) | - |

#### Oracle sets (`<oracles.root>/<set>/<game>/`)

| set | about | source |
| --- | --- | --- |
| `oracles.corpus` | shader programs as compilers and games wrote them | the user's own installs and public shader corpora, as each program's record states |
| `oracles.d3d-disassembler` | Microsoft's D3D shader disassembly of a game's shader containers (`NNNN.dxbc` with its listing) | independent: Microsoft's D3D Shader Disassembler, run over the user's own install |
| `oracles.mojoshader` | MonoGame's GLSL of Direct3D 9 programs: `programs.jsonl`, one line per program of a set (its name there and SHA-256, then the GLSL, stage, version, hints and uniforms, or MojoShader's refusal), and `MANIFEST.toml` | independent: MonoGame's MojoShader (`libmojoshader_64.dll` of dotnet-mgfxc 3.8.1.303) run by iridia-mojo/tests/oracles/mojoshader.py over another oracle set's programs, with the edits MonoGame's effect compiler makes |

#### Declared files

| key | status | gate | about | source |
| --- | --- | --- | --- | --- |
| `oracles.corpus.general/slim` | live | - | the slim shader set; its Direct3D 9 programs (`programs/d3d9/<sha256>.d3d9`) are the bytes of `oracles.mojoshader.general/slim-d3d9`'s pairs | independent: primase-oracles' `producers/slim-set`, a set cover of the disassembly features of the oracle home's game corpora, its disassembler sets, ten outside projects' test corpora and FidelityFX SDK v1.1.4 |
| `oracles.d3d-disassembler.general/sm3` | live | - | a raw ps_3_0 and two vs_3_0 programs (`pixel.bin`, `vertex.bin`, `vertex-layered.bin`): the bytes of `oracles.mojoshader.general/sm3`'s pairs | independent: Microsoft d3dcompiler_47.dll D3DDisassemble via iridia-d3d9/tests/oracles/disassemble.py, on shipped Shaun White Skateboarding SDB2 v39 programs; raw bytes stay outside the repository |
| `oracles.d3d-disassembler.shaun-white/sdb2` | live | - | every distinct vertex (`.vs`) and pixel (`.ps`) program of Shaun White Skateboarding's three SDB2 bundles: the bytes of `oracles.mojoshader.shaun-white/sdb2`'s pairs | independent: Microsoft d3dcompiler_47.dll D3DDisassemble over the programs of the user's own Shaun White Skateboarding install (extracted by a read-only script); raw bytes stay outside the repository |
| `oracles.mojoshader.general/sm3` | live | - | MonoGame's GLSL of the three programs of `oracles.d3d-disassembler.general/sm3`; tests/oracle.rs | independent: MonoGame's MojoShader build (dotnet-mgfxc 3.8.1.303's libmojoshader_64.dll) via iridia-mojo/tests/oracles/mojoshader.py |
| `oracles.mojoshader.general/slim-d3d9` | live | - | MonoGame's GLSL of the 557 Direct3D 9 programs of `oracles.corpus.general/slim` (MojoShader refuses 106); tests/oracle.rs | independent: MonoGame's MojoShader build (dotnet-mgfxc 3.8.1.303's libmojoshader_64.dll) via iridia-mojo/tests/oracles/mojoshader.py |
| `oracles.mojoshader.shaun-white/sdb2` | live | - | MonoGame's GLSL of the 7,306 programs of `oracles.d3d-disassembler.shaun-white/sdb2`; tests/oracle.rs | independent: MonoGame's MojoShader build (dotnet-mgfxc 3.8.1.303's libmojoshader_64.dll) via iridia-mojo/tests/oracles/mojoshader.py, over the programs of the user's own install; nothing of them goes into the repository |

<!-- end of real-data inputs -->
