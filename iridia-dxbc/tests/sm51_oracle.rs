//! Each .asm is fxc 10.0.26100.0 -dumpbin of the adjacent .51.dxbc, compiled from its HLSL.
#![allow(clippy::unwrap_used)]
use dxbc_reader::{Operand, Register, Selection};

fn register(r: Register) -> &'static str {
    match r {
        Register::Temp => "r",
        Register::Input => "v",
        Register::Output => "o",
        Register::Constant => "CB",
        Register::Texture => "T",
        Register::Sampler => "S",
        Register::Uav => "U",
        Register::ThreadId => "vThreadID",
        Register::Null => "null",
        Register::Immediate => "literal",
        other => panic!("extend the oracle for {other:?}"),
    }
}
fn operand(op: &Operand, relative: bool) -> String {
    let mut s = register(op.register).to_string();
    for (n, index) in op.indices.iter().enumerate() {
        let ix = if let Some(r) = &index.relative {
            let r = operand(r, true);
            if index.base == 0 {
                r
            } else {
                format!("{r}+{}", index.base)
            }
        } else {
            index.base.to_string()
        };
        if n == 0 {
            s.push_str(&ix);
        } else {
            s.push_str(&format!("[{ix}]"));
        }
    }
    if relative {
        let lane = match op.selection {
            Selection::Scalar(c) => c,
            Selection::Swizzle(s) => s[0],
            _ => panic!("relative index without a lane"),
        };
        s.push('.');
        s.push(char::from(b"xyzw"[usize::from(lane)]));
    }
    s
}
fn split_operands(s: &str) -> Vec<&str> {
    if s.trim().is_empty() {
        return Vec::new();
    }
    let mut depth = 0;
    let mut start = 0;
    let mut result = Vec::new();
    for (at, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                result.push(s[start..at].trim());
                start = at + 1;
            }
            _ => {}
        }
    }
    result.push(s[start..].trim());
    result
}
fn assembly_operand(s: &str) -> String {
    let s = s.trim().trim_start_matches('-').trim_start_matches('|');
    if s.starts_with("l(") {
        return "literal".into();
    }
    let mut result = String::new();
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            '.' | '{' | '|' if depth == 0 => break,
            _ => {}
        }
        if !c.is_whitespace() {
            result.push(c);
        }
    }
    result
}

#[test]
fn every_sm51_instruction_matches_fxc_registers_and_indices() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sm51");
    let mut total = 0;
    let mut operands = 0;
    for name in [
        "spaces",
        "textures",
        "constants",
        "images",
        "structured",
        "unbounded",
        "nonuniform",
        "unbounded_buffers",
        "unbounded_constants",
        "samplers",
    ] {
        let bytes = std::fs::read(root.join(format!("{name}.51.dxbc"))).unwrap();
        let asm = std::fs::read_to_string(root.join(format!("{name}.51.asm"))).unwrap();
        let (instructions, count) = compare(name, &bytes, &asm);
        total += instructions;
        operands += count;
    }
    println!("total: {total} instructions, {operands} operands");
}

fn compare(name: &str, bytes: &[u8], asm: &str) -> (usize, usize) {
    let decoded = dxbc_reader::decode(&dxbc_reader::code(bytes).unwrap()).unwrap();
    let lines: Vec<_> = asm
        .lines()
        .map(str::trim)
        .skip_while(|s| *s != "ps_5_1" && *s != "cs_5_1")
        .skip(1)
        .filter(|s| !s.is_empty() && !s.starts_with("//") && !s.starts_with("dcl_"))
        .collect();
    assert_eq!(
        decoded.instructions.len(),
        lines.len(),
        "{name}: instruction count"
    );
    let mut count = 0;
    for (ins, line) in decoded.instructions.iter().zip(lines) {
        let (opcode, args) = line.split_once(' ').unwrap_or((line, ""));
        let opcode = opcode
            .split('(')
            .next()
            .unwrap()
            .trim_end_matches("_indexable")
            .trim_end_matches("_nz")
            .trim_end_matches("_z");
        let opcode = if let Some(flags) = opcode.strip_prefix("sync_") {
            let mut control = 0;
            for flag in flags.split('_') {
                control |= match flag {
                    "t" => 1,
                    "g" => 2,
                    "ugroup" => 4,
                    "uglobal" => 8,
                    other => panic!("unknown sync flag {other}"),
                };
            }
            assert_eq!(ins.control & 15, control, "{name}: {line}");
            "sync"
        } else {
            opcode
        };
        assert_eq!(ins.name(), opcode, "{name} word {}: {line}", ins.word);
        let args = split_operands(args);
        assert_eq!(ins.operands.len(), args.len(), "{name} word {}: {line}", ins.word);
        for (op, text) in ins.operands.iter().zip(args) {
            assert_eq!(
                operand(op, false),
                assembly_operand(text),
                "{name} word {}: {line}",
                ins.word
            );
            assert_eq!(op.non_uniform, text.contains("nonuniform"), "{name}: {text}");
            count += 1;
        }
    }
    println!(
        "{name}: {} instructions, {count} operand registers/indices",
        decoded.instructions.len()
    );
    (decoded.instructions.len(), count)
}

/// Private binaries stay outside the repository. This optional directory contains .cs.dxbc and .cs.asm pairs.
#[test]
fn private_sm51_decoder_reflection_and_lift() {
    let Some(root) = std::env::var_os("IRIDIA_SM51_CORPUS") else {
        eprintln!("SKIP private SM5.1 oracle: IRIDIA_SM51_CORPUS is unset");
        return;
    };
    let root = std::path::PathBuf::from(root);
    let mut files: Vec<_> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "dxbc"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no private DXBC fixtures");
    for path in files {
        let name = path.file_name().unwrap().to_str().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let asm = std::fs::read_to_string(path.with_extension("asm")).unwrap();
        compare(name, &bytes, &asm);
        match iridia_dxbc::resource_reflection(&bytes) {
            Ok(Some(resources)) => println!(
                "{name}: reflection {} resources; validation {:?}",
                resources.len(),
                iridia_dxbc::reflection(&bytes).map(|i| i.verify())
            ),
            result => println!("{name}: reflection {result:?}"),
        }
        match iridia_dxbc::lift(&bytes) {
            Ok(program) => println!(
                "{name}: lifted {} resources; validation {:?}",
                program.interface.resources.len(),
                program.validate().map(|_| ())
            ),
            Err(error) => println!("{name}: lift refused: {error}"),
        }
    }
}
