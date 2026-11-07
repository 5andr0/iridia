//! The real-data inputs this crate's tests read, each with what it is, where it comes from and whether a full run
//! needs it (`primase_testdata::inputs!`; README.md here lists them). Integration tests reach it
//! with `mod inputs;`, unit tests with `#[cfg(test)] #[path = "../tests/inputs/mod.rs"] mod inputs;` in the
//! crate root, and tests/real_data_inputs.rs checks that every query the crate's sources write is declared
//! here. Declare an input here when a test first asks for it.

primase_testdata::inputs! {
    games {
        General = "general",
        ShaunWhite = "shaun-white",
    }
    installs {}
    keys {}
    tools {}
    oracle_sets {
        Corpus = "corpus" {
            about: "shader programs as compilers and games wrote them",
            source: "the user's own installs and public shader corpora, as each program's record states",
        },
        D3dDisassembler = "d3d-disassembler" {
            about: "Microsoft's D3D shader disassembly of a game's shader containers (`NNNN.dxbc` with its listing)",
            source: "independent: Microsoft's D3D Shader Disassembler, run over the user's own install",
        },
        MojoShader = "mojoshader" {
            about: "MonoGame's GLSL of Direct3D 9 programs: `programs.jsonl`, one line per program of a set (its name there and SHA-256, then the GLSL, stage, version, hints and uniforms, or MojoShader's refusal), and `MANIFEST.toml`",
            source: "independent: MonoGame's MojoShader (`libmojoshader_64.dll` of dotnet-mgfxc 3.8.1.303) run by iridia-mojo/tests/oracles/mojoshader.py over another oracle set's programs, with the edits MonoGame's effect compiler makes",
        },
    }
    declared {
        Query::Oracle(OracleSet::Corpus, Game::General, "slim") => {
            status: Live,
            gate: false,
            about: r#"the slim shader set; its Direct3D 9 programs (`programs/d3d9/<sha256>.d3d9`) are the bytes of `oracles.mojoshader.general/slim-d3d9`'s pairs"#,
            source: r#"independent: primase-oracles' `producers/slim-set`, a set cover of the disassembly features of the oracle home's game corpora, its disassembler sets, ten outside projects' test corpora and FidelityFX SDK v1.1.4"#,
        },
        Query::Oracle(OracleSet::D3dDisassembler, Game::General, "sm3") => {
            status: Live,
            gate: false,
            about: r#"a raw ps_3_0 and two vs_3_0 programs (`pixel.bin`, `vertex.bin`, `vertex-layered.bin`): the bytes of `oracles.mojoshader.general/sm3`'s pairs"#,
            source: r#"independent: Microsoft d3dcompiler_47.dll D3DDisassemble via iridia-d3d9/tests/oracles/disassemble.py, on shipped Shaun White Skateboarding SDB2 v39 programs; raw bytes stay outside the repository"#,
        },
        Query::Oracle(OracleSet::D3dDisassembler, Game::ShaunWhite, "sdb2") => {
            status: Live,
            gate: false,
            about: r#"every distinct vertex (`.vs`) and pixel (`.ps`) program of Shaun White Skateboarding's three SDB2 bundles: the bytes of `oracles.mojoshader.shaun-white/sdb2`'s pairs"#,
            source: r#"independent: Microsoft d3dcompiler_47.dll D3DDisassemble over the programs of the user's own Shaun White Skateboarding install (extracted by a read-only script); raw bytes stay outside the repository"#,
        },
        Query::Oracle(OracleSet::MojoShader, Game::General, "sm3") => {
            status: Live,
            gate: false,
            about: r#"MonoGame's GLSL of the three programs of `oracles.d3d-disassembler.general/sm3`; tests/oracle.rs"#,
            source: r#"independent: MonoGame's MojoShader build (dotnet-mgfxc 3.8.1.303's libmojoshader_64.dll) via iridia-mojo/tests/oracles/mojoshader.py"#,
        },
        Query::Oracle(OracleSet::MojoShader, Game::General, "slim-d3d9") => {
            status: Live,
            gate: false,
            about: r#"MonoGame's GLSL of the 557 Direct3D 9 programs of `oracles.corpus.general/slim` (MojoShader refuses 106); tests/oracle.rs"#,
            source: r#"independent: MonoGame's MojoShader build (dotnet-mgfxc 3.8.1.303's libmojoshader_64.dll) via iridia-mojo/tests/oracles/mojoshader.py"#,
        },
        Query::Oracle(OracleSet::MojoShader, Game::ShaunWhite, "sdb2") => {
            status: Live,
            gate: false,
            about: r#"MonoGame's GLSL of the 7,306 programs of `oracles.d3d-disassembler.shaun-white/sdb2`; tests/oracle.rs"#,
            source: r#"independent: MonoGame's MojoShader build (dotnet-mgfxc 3.8.1.303's libmojoshader_64.dll) via iridia-mojo/tests/oracles/mojoshader.py, over the programs of the user's own install; nothing of them goes into the repository"#,
        },
    }
    debug {}
}
