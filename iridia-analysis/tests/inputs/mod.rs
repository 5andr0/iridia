//! The real-data inputs this crate's tests read, each with what it is, where it comes from and whether a full run
//! needs it (`primase_testdata::inputs!`; README.md here lists them). Integration tests reach it
//! with `mod inputs;`, and tests/real_data_inputs.rs checks that every query the crate's sources write is declared
//! here. Declare an input here when a test first asks for it.

primase_testdata::inputs! {
    games {
        Elden = "elden",
        General = "general",
    }
    installs {}
    keys {}
    tools {}
    oracle_sets {
        Corpus = "corpus" {
            about: "shader programs as compilers and games wrote them",
            source: "the user's own installs and public shader corpora, as each program's record states",
        },
    }
    declared {
        Query::Oracle(OracleSet::Corpus, Game::Elden, "shaders") => {
            status: Live,
            gate: false,
            about: r#"a folder (searched recursively) of the DXIL containers of Elden Ring's Shader Model 6 shader bundles (one file per container), each one's analyses on the IR held to the older analyses and to tests/expected/elden.tsv; tests/oracle.rs"#,
            source: r#"the user's own install: the containers of its `/shader/*.shaderbnd.dcx` binders and `/shader/*.shaderbdlebnd.dcx` bundles (ray-tracing `[rt]` bundles included); primase's [corpus_elden_shaders example](https://github.com/Kalmarv/primase/tree/main/worlds/fromsoft/examples/corpus_elden_shaders.rs) extracts them"#,
        },
        Query::Oracle(OracleSet::Corpus, Game::General, "slim") => {
            status: Live,
            gate: true,
            about: r#"the slim shader set: 8,287 DXBC, DXIL and Direct3D 9 programs (`programs/<family>/<sha256>.<family>`, Microsoft's listing beside each, `slim.tsv` saying why each is there) that reach every region and branch side of the lifters the full pool of 108,394 reaches; tests/oracle.rs holds each one's analyses on the IR to the older analyses and to tests/expected/slim-<family>.tsv; tests/matrix_lines.rs compares 24 constants in six integer-matrix DXBC programs with the older matrix analysis"#,
            source: r#"independent: primase-oracles' `producers/slim-set`, a set cover of the disassembly features of the oracle home's game corpora, its disassembler sets, ten outside projects' test corpora and FidelityFX SDK v1.1.4"#,
        },
    }
    debug {}
}
