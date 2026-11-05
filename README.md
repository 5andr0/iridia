# iridia

A set of Rust libraries that translate compiled Direct3D shaders into [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga)
modules. The modules can be validated, inspected and written out as WGSL, SPIR-V or any other format naga supports.

## Features

- Lifts Shader Model 4 and 5 bytecode (DXBC, `fxc` output) of every stage
- Lifts Shader Model 6 bytecode (DXIL, `dxc` output) of every stage except amplification, including mesh and ray tracing programs
- Lifts Direct3D 9 `vs_3_0` and `ps_3_0` token streams with their `CTAB` constant table
- Lifts the GLSL that MojoShader writes for Shader Model 2 and 3 programs
- Reads a container's reflection and signatures without lifting its program
- Keeps semantic names, register packing and constant buffer layouts in a typed interface beside the module
- Writes out Direct3D's numeric behaviour where it differs from WGSL, such as integer division by zero
- Writes WGSL for a given pipeline, with chosen resource bindings and vertex outputs matched to the pixel program's inputs
- Analyses a lifted program without running it: what each output lane is made of, which constant words it reads, and whether a constant buffer holds a matrix by rows or by columns
- Specialises a program for stated constant values by folding them into a copy
- Runs lifted programs on the CPU with a reference executor, including geometry, tessellation, mesh and ray tracing stages
- Validates every lifted program with naga's validator and checks its interface against the entry point
- Refuses unsupported constructs by name, with the word, bit or line where they occur, and records approximations as notes on the program
- Records each program's source hash and lifter version, so caches can tell when a lift is out of date
- Contains no `unsafe` code and does no I/O in the library crates

## Getting Started

### Requirements

Rust 1.99 or newer (edition 2024).

### Installation

| Crate | Description |
|-------|-------------|
| [`iridia`](iridia) | The lifted program type: a naga module with its interface, validation, WGSL writers and analyses |
| [`iridia-dxbc`](iridia-dxbc) | Lifts DXBC, Shader Model 4 and 5 |
| [`iridia-dxil`](iridia-dxil) | Lifts DXIL, Shader Model 6 |
| [`iridia-d3d9`](iridia-d3d9) | Lifts Direct3D 9 `vs_3_0` and `ps_3_0` token streams |
| [`iridia-mojo`](iridia-mojo) | Lifts MojoShader's GLSL |
| [`iridia-interface`](iridia-interface) | Serializable interface and fixed-function state data, without naga |
| [`iridia-exec`](iridia-exec) | CPU reference executor for lifted programs |

All seven crates are published on crates.io at version 0.4.1:

```sh
cargo add iridia iridia-dxbc
```

The `main` branch is ahead of that release. Shader Model 5.1 lifting and the other changes listed under "Unreleased"
in [CHANGELOG.md](CHANGELOG.md) are only available from the repository:

```sh
cargo add --git https://github.com/Kalmarv/iridia iridia iridia-dxbc
```

The modules are built with [primase-naga](https://github.com/Kalmarv/primase-naga), a fork of naga 30.0.1. Code that
uses `Program::module` directly must depend on the same fork, for example with `cargo add primase-naga --rename naga`.

### Usage

Lift a Shader Model 5 shader, list the inputs it reads and write it as WGSL (`iridia/examples/lift.rs`):

```rust,no_run
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args().nth(1).ok_or("usage: lift <shader.dxbc>")?;
    let container = std::fs::read(path)?;

    // Shader Model 4 or 5; `iridia_dxil::lift` takes Shader Model 6 the same way
    let program = iridia_dxbc::lift(&container)?;
    println!(
        "{:?} program, target {}",
        program.stage, program.provenance.target
    );
    for input in program.interface.inputs.iter().filter(|input| input.used) {
        println!("reads {}{}", input.label.name, input.label.index);
    }

    // the same program as WGSL; `program.module` is a `naga::Module` for any other naga backend
    print!("{}", iridia::emit::wgsl(&program)?);
    Ok(())
}
```

Other examples, which take either a DXBC or a DXIL container:

```sh
cargo run -p iridia --example to_wgsl -- shader.dxbc
cargo run -p iridia --example to_spirv -- shader.dxil out.spv
```

Run the tests with `cargo test --workspace`. Tests that need game files, Microsoft's compilers or the WARP rasterizer
skip unless `GAME_DATA` points at a [primase-testdata](https://github.com/Kalmarv/primase-testdata) configuration; each
crate's `tests/inputs/README.md` lists those inputs.

Each crate's README describes what it reads and what it refuses. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
checks and for reporting a shader the crates cannot read. Licensed under GPL-3.0-or-later.
