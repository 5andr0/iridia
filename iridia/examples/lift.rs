use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args().nth(1).ok_or("usage: lift <shader.dxbc>")?;
    let container = std::fs::read(path)?;

    // Shader Model 4 or 5; `iridia_dxil::ir::lift` takes Shader Model 6 the same way
    let mut program = iridia_dxbc::ir::lift(&container)?;
    println!(
        "{:?} program, target {}",
        program.stage, program.provenance.target
    );
    for input in program.interface.inputs.iter().filter(|input| input.used) {
        println!("reads {}{}", input.label.name, input.label.index);
    }

    // the form the writers take, then the same program as WGSL
    iridia_passes::prepare(&mut program)?;
    print!("{}", iridia_wgsl::wgsl(&program, &iridia_wgsl::Profile::BROWSER)?);
    Ok(())
}
