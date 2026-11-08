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
    tools {
        WgslRun = "wgsl_run" {
            about: "the WGSL harness that runs a WGSL program's entry point on Direct3D 12's WARP adapter through wgpu (tools/wgsl-run, built with `cargo build --release` there)",
            source: "independent: wgpu and Microsoft's WARP rasterizer, through tools/wgsl-run of this repository (its own workspace, outside iridia's)",
            gate: false,
        },
        WgslRunDxc = "wgsl_run_dxc" {
            about: "the `dxcompiler.dll` the WGSL harness compiles with (DXC 1.8.2502 or later, `dxil.dll` beside it); FXC compiles where it is not configured",
            source: "third-party: Microsoft's DirectX Shader Compiler release build",
            gate: false,
        },
    }
    oracle_sets {
        D3dDisassembler = "d3d-disassembler" {
            about: "Microsoft's D3D shader disassembly of a game's shader containers (`NNNN.dxbc` with its listing)",
            source: "independent: Microsoft's D3D Shader Disassembler, run over the user's own install",
        },
    }
    declared {
        Query::Oracle(OracleSet::D3dDisassembler, Game::General, "sm3") => {
            status: Live,
            gate: true,
            about: r#"privately extracted raw ps_3_0 (`pixel.bin`) and two vs_3_0 programs (`vertex.bin`, `vertex-layered.bin`), each with Microsoft's `.asm` disassembly; tests/disassembly.rs checks all executable operands, declarations, definitions and CTAB allocations"#,
            source: r#"independent: Microsoft d3dcompiler_47.dll D3DDisassemble via iridia-d3d9/tests/oracles/disassemble.py, on a shipped Shaun White Skateboarding SDB2 v39 pixel shader (record 5) and vertex shaders (records 5 and 150, ekshadersPChigh.bin); raw bytes and disassembly stay outside the repository"#,
        },
        Query::Oracle(OracleSet::D3dDisassembler, Game::ShaunWhite, "sdb2") => {
            status: Live,
            gate: false,
            about: r#"a folder of every distinct vertex (`.vs`) and pixel (`.ps`) program of the three SDB2 bundles (ekshadersPChigh, ekshadersPCmed, ekshadersPClow), each raw token stream with Microsoft's `.asm` disassembly beside it; tests/corpus.rs"#,
            source: r#"independent: Microsoft d3dcompiler_47.dll D3DDisassemble over the programs of the user's own Shaun White Skateboarding install (extracted by a read-only script); raw bytes and disassembly stay outside the repository"#,
        },
    }
    debug {
        "shader.graphics" => "`1` runs every program in the WARP comparison (tests/warp); `0` none; unset runs a fixed sample (one program in `shader.graphics_every`, 128 by default, by file-name hash), or every program `shader.only` names",
        "shader.graphics_every" => "the WARP comparison's sample takes one program in this many (128 by default; `shader.graphics`)",
        "shader.only" => "`|`-separated parts of program names: only the programs of a corpus with one of them in the name run on WARP",
        "shader.lanes" => "a part of a program's name: the WARP comparison prints that program's WGSL and every lane that is not exact",
        "shader.dump" => "a folder: the WGSL of each program that runs on WARP is written to it, named by the program and the writer",
        "shader.pow" => "`defined`: the WARP comparison swaps `pow` in each program's WGSL for one that gives a zero base and an exponent of 0 or less libm's value (WGSL leaves it undefined), to tell programs whose differences are that case from the others",
        "shader.sin" => "`reference`: the WARP comparison swaps `sin` in each program's WGSL for an accurate one (within 8.5e-8 of the reference over -1000..1000; WARP's builtin is off by up to 9.6e-5 over the same arguments), to tell programs whose differences are WARP's `sin` from the others; a program with a `sin` or `cos` of another form is a gap",
    }
}
