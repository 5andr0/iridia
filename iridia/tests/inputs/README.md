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

#### Declared files

| key | status | gate | about | source |
| --- | --- | --- | --- | --- |
| `oracles.corpus.general/slim` | live | yes | the slim shader set: 8,286 DXBC, DXIL and Direct3D 9 programs (`programs/<family>/<sha256>.<family>`, Microsoft's listing beside each, `slim.tsv` saying why each is there) that reach every region and branch side of the lifters the full pool of 108,394 reaches; tests/slim.rs, the pre-push gate | independent: primase-oracles' `producers/slim-set`, a set cover of the disassembly features of the oracle home's game corpora, its disassembler sets, ten outside projects' test corpora and FidelityFX SDK v1.1.4 |

<!-- end of real-data inputs -->
