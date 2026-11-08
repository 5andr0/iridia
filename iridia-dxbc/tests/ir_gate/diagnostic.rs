//! Single-container diagnostic, compiled only for the example executable.
use super::*;

/// One program's lifts and runs, for a disagreement the gate prints: `IR_GATE_ONE=<container> cargo run --release -p
/// iridia-dxbc --example ir_gate`. A and B as text (not with `IR_GATE_QUIET`), then each
/// vector's runs of A, B and C, the first request C makes otherwise than A, and C on a quad of the first lane's inputs;
/// `IR_GATE_C_WGSL=<file>` runs that WGSL as C instead of the lowering's (to try a change to it); `IR_GATE_P=1` prints
/// P as text and the lowering's WGSL.
pub(crate) fn run() {
    let path = std::env::var("IR_GATE_ONE").expect("IR_GATE_ONE names a container");
    let bytes = std::fs::read(&path).unwrap();
    let (old, p) = lift_both(&bytes);
    let old = old.unwrap();
    let quiet = std::env::var_os("IR_GATE_QUIET").is_some();
    if !quiet {
        println!(
            "A (old lift) as WGSL:\n{}",
            iridia::emit::wgsl(&old).unwrap_or_else(|e| e.to_string())
        );
    }
    let p = p.unwrap();
    if !quiet {
        println!("B (IR lift):\n{}", iridia_ir::text::print(&p.entry));
    }
    let p = structured(p).unwrap();
    if std::env::var_os("IR_GATE_P").is_some() {
        match passed(p.clone()) {
            Ok(pp) => {
                println!(
                    "P (after the passes):
{}",
                    iridia_ir::text::print(&pp.entry)
                );
                match iridia_naga::wgsl(&pp, &iridia_naga::Profile::NAGA) {
                    Ok(w) => println!(
                        "C's WGSL:
{w}"
                    ),
                    Err(e) => println!("C's WGSL: {}", e.0),
                }
            }
            Err(e) => println!("P: {e}"),
        }
    }
    // C, where the lowering takes the program
    let c_program = passed(p.clone()).ok().and_then(|pp| match lower_c(&old, &pp) {
        Ok(c) => Some(c),
        Err(Unlowered::Refused(e) | Unlowered::Unbuilt(e)) => {
            println!("C: {e}");
            None
        }
    });
    if old.interface.stage == Stage::Compute {
        return one_compute(&old, &p, c_program.as_ref());
    }
    let a = ProgramShader::new(old).unwrap();
    let b = ProgramShader::from_ir(p).unwrap();
    let c_shader = c_program.map(|c| ProgramShader::new(c).unwrap());
    for c in 0..CASES {
        let quad = a.needs_quad() || b.needs_quad();
        let case = gate_case(&a, c, quad, QuadInputs::Own);
        println!(
            "case {c}: inputs {:x?}",
            case.iter().map(|q| &q.inputs).collect::<Vec<_>>()
        );
        let (mut sa, mut sb) = (Hashed::for_shader(&a, c), Hashed::for_shader(&b, c));
        let (x, y) = (run_asking(&a, &case, &mut sa), run_asking(&b, &case, &mut sb));
        println!("  A {x:x?}\n  B {y:x?}");
        if let Some(cs) = &c_shader {
            let mut sc = Hashed::for_shader(cs, c);
            let z = run_asking(cs, &case, &mut sc);
            println!("  C {z:x?}");
            for (k, (p, q)) in sa.asked.iter().zip(&sc.asked).enumerate() {
                if p != q {
                    println!("  request {k} differs:\n    A {p}\n    C {q}");
                    break;
                }
            }
            if sa.asked.len() != sc.asked.len() {
                println!("  A made {} requests, C {}", sa.asked.len(), sc.asked.len());
            }
            // the quad with every lane the first lane's inputs (no lane leaves a loop before the others)
            if quad {
                let first = gate_case(&a, c, quad, QuadInputs::First);
                let (mut ta, mut tc) = (Hashed::for_shader(&a, c), Hashed::for_shader(cs, c));
                let (xa, xc) = (run_asking(&a, &first, &mut ta), run_asking(cs, &first, &mut tc));
                println!(
                    "  first lane's inputs everywhere: A and C agree: {}",
                    same_run(&xa, &xc, false)
                );
            }
        }
        if !quiet {
            for (k, (p, q)) in sa.asked.iter().zip(&sb.asked).enumerate() {
                if p != q {
                    println!("  request {k} differs:\n    A {p}\n    B {q}");
                }
            }
            if c == 0 || x != y {
                println!("  A asked {:#?}\n  B asked {:#?}", sa.asked, sb.asked);
            }
        }
        if sa.asked.len() != sb.asked.len() {
            println!("  A made {} requests, B {}", sa.asked.len(), sb.asked.len());
        }
    }
}

/// [`run`] for a compute program: each case's memory after A, B and C, and the first words where B or C leaves
/// other memory than A.
fn one_compute(old: &iridia::Program, p: &iridia_ir::Program, c: Option<&iridia::Program>) {
    let a = Executable::new(old).unwrap();
    let b = Executable::from_ir(p).unwrap();
    let c = c.map(|c| (Executable::new(c).unwrap(), c));
    for case in 0..CASES {
        let x = dispatch(&a, &old.interface, case);
        let y = dispatch(&b, &old.interface, case);
        let z = c
            .as_ref()
            .map(|(c, program)| dispatch(c, &program.interface, case));
        println!("case {case}:");
        let Ok(x) = x else {
            println!("  A {x:?}");
            continue;
        };
        for (column, y) in [("B", Some(y)), ("C", z)] {
            match y {
                None => {}
                Some(Err(e)) => println!("  {column} {e}"),
                Some(Ok(y)) => memory_differences(column, &x, &y),
            }
        }
    }
}

/// Up to a few of the places where `y` leaves other memory than `x`.
fn memory_differences(column: &str, x: &Memory, y: &Memory) {
    let mut lines = Vec::new();
    for (id, words) in &x.buffers {
        match y.buffers.get(id) {
            None => lines.push(format!("buffer {id:?}: missing")),
            Some(other) => {
                for (k, (p, q)) in words.iter().zip(other).enumerate() {
                    if p != q {
                        lines.push(format!("buffer {id:?} word {k}: A {p:x} {column} {q:x}"));
                    }
                }
            }
        }
    }
    for (key, words) in &x.elements {
        if y.elements.get(key) != Some(words) {
            lines.push(format!(
                "element {key:?}: A {words:x?} {column} {:x?}",
                y.elements.get(key)
            ));
        }
    }
    for (key, texel) in &x.texels {
        if y.texels.get(key) != Some(texel) {
            lines.push(format!(
                "texel {key:?}: A {texel:x?} {column} {:x?}",
                y.texels.get(key)
            ));
        }
    }
    for key in y.texels.keys().filter(|k| !x.texels.contains_key(k)) {
        lines.push(format!("texel {key:?}: only {column} wrote it"));
    }
    if x.counters != y.counters {
        lines.push(format!(
            "counters: A {:x?} {column} {:x?}",
            x.counters, y.counters
        ));
    }
    let total = lines.len();
    for l in lines.iter().take(12) {
        println!("  {l}");
    }
    if total > 12 {
        println!("  ... {total} differences in all");
    }
    if total == 0 {
        println!("  {column} agrees");
    }
}
