//! Translate a compiled Direct3D shader to SPIR-V, through naga's SPIR-V writer.
//!
//! ```text
//! cargo run -p iridia --example to_spirv -- path/to/shader.dxbc out.spv
//! ```
//!
//! `iridia_naga::module` lowers an IR program to a `naga::Module`, so any writer naga has can take it; this one enables
//! naga's `spv-out` feature. The writer is naga's: it refuses what SPIR-V output cannot express, and the module is
//! written as the program was lifted, in Direct3D's clip-space and texture conventions
//! ([`iridia_ir::Program::conventions`]), which a pipeline that wants Vulkan's applies where it meets the program.

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let (Some(path), Some(out)) = (args.next(), args.next()) else {
        eprintln!("usage: to_spirv <shader.dxbc|shader.dxil> <out.spv>");
        return ExitCode::from(2);
    };
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("{}: {e}", path.to_string_lossy());
            return ExitCode::from(2);
        }
    };
    let lifted = if bytes.windows(4).any(|w| w == b"DXIL") {
        iridia_dxil::ir::lift(&bytes)
    } else {
        iridia_dxbc::ir::lift(&bytes)
    };
    let mut program = match lifted {
        Ok(program) => program,
        Err(refusal) => {
            eprintln!("refused: {refusal}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(refusal) = iridia_passes::prepare(&mut program) {
        eprintln!("refused: {refusal}");
        return ExitCode::FAILURE;
    }
    if let Err(problems) = iridia_ir::validate(&program) {
        for problem in problems {
            eprintln!("invalid: {problem:?}");
        }
        return ExitCode::FAILURE;
    }
    let module = match iridia_naga::BindingPlan::d3d(&program.interface)
        .map_err(iridia_naga::Error)
        .and_then(|plan| iridia_naga::module(&program, &plan))
    {
        Ok(module) => module,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let info = match iridia_naga::validate(&module, &iridia_naga::Profile::NAGA) {
        Ok(info) => info,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let words = match naga_spv::back::spv::write_vec(
        &module,
        &info,
        &naga_spv::back::spv::Options::default(),
        None,
    ) {
        Ok(words) => words,
        Err(e) => {
            eprintln!("spirv: {e}");
            return ExitCode::FAILURE;
        }
    };
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if let Err(e) = std::fs::write(&out, bytes) {
        eprintln!("{}: {e}", out.to_string_lossy());
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}
