#![allow(clippy::unwrap_used)]
//! Programs in MojoShader's dialect lifted onto the IR (`iridia_mojo::ir`): they validate, print and read back as the
//! same text, take the passes, are written as WGSL and have the interface (a digest of it) and notes the older lift,
//! which they were once held to, gave them; what the dialect does not have is refused by name, in the older lift's
//! words; and the GLSL MonoGame's MojoShader made of the fixtures' Direct3D 9 programs (`tests/fixtures`) runs as those
//! programs do (`tests/pairing`).
mod inputs;
mod pairing;
mod programs;

use std::path::{Path, PathBuf};

use iridia_ir::interface::Stage;
use iridia_mojo::{Hints, ir};
use programs::{CONTROL, GAMMA, VERTEX, hints};
use sha2::{Digest, Sha256};

/// FNV-1a of `text`, as 16 hex digits.
fn fnv(text: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// The digest of each program's interface (its `Debug` text) and its notes, as the older lift's gave them while the
/// two were held equal.
const OLDER: [(&str, &str, &[&str]); 3] = [
    ("gamma.ps", "98bbb2828e2c86d2", &[]),
    (
        "vertex.vs",
        "1429271d3a1fa697",
        &[
            "the clip-space fix for OpenGL that MojoShader ends a vertex program with (posFixup: flip y, half-pixel offset, depth 0..1 to -1..1) is dropped: the program keeps Direct3D's clip space",
        ],
    ),
    ("control.ps", "07f70fd7e1c3ecfd", &[]),
];

fn programs() -> [(&'static str, &'static str, Stage, Hints); 3] {
    [
        ("gamma.ps", GAMMA, Stage::Pixel, hints()),
        ("vertex.vs", VERTEX, Stage::Vertex, hints()),
        ("control.ps", CONTROL, Stage::Pixel, Hints::default()),
    ]
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// The fixtures' GLSL, as `tests/oracles/mojoshader.py` wrote it.
fn fixture_lines() -> Vec<pairing::Line> {
    pairing::lines(&std::fs::read_to_string(fixtures().join("mojoshader/programs.jsonl")).unwrap())
}

#[test]
fn the_programs_lift_print_take_the_passes_and_are_written() {
    for (name, source, stage, hints) in programs() {
        let p = ir::lift(source, stage, &hints).unwrap_or_else(|e| panic!("{name}: {e}"));
        iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let text = iridia_ir::text::print(&p.entry);
        let read = iridia_ir::text::parse(&text).unwrap_or_else(|e| panic!("{name}: {e}\n{text}"));
        assert_eq!(iridia_ir::text::print(&read), text, "{name}");
        let (_, digest, notes) = OLDER.iter().find(|o| o.0 == name).unwrap();
        assert_eq!(fnv(&format!("{:?}", p.interface)), *digest, "{name}");
        assert_eq!(p.notes, *notes, "{name}");
        assert_eq!(p.provenance.lifter, ir::LIFTER);
        assert_eq!(
            p.provenance.policies,
            [
                "glsl.clamp01=saturate",
                "glsl.float_to_int=saturate",
                "glsl.inversesqrt_log2=magnitude",
                "glsl.normalize_zero=zero",
                "glsl.uniform_oob=zero"
            ]
        );
        let mut prepared = p.clone();
        iridia_passes::prepare(&mut prepared).unwrap_or_else(|e| panic!("{name}: {e}"));
        iridia_ir::validate(&prepared).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        iridia_naga::wgsl(&prepared, &iridia_naga::Profile::BROWSER)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn the_vertex_program_drops_the_clip_space_fix_of_the_api() {
    let p = ir::lift(VERTEX, Stage::Vertex, &hints()).unwrap();
    assert!(p.notes.iter().any(|n| n.contains("posFixup")), "{:?}", p.notes);
    assert!(!iridia_ir::text::print(&p.entry).contains("posFixup"));
}

/// What the dialect does not have is refused by line and reason, in the words the older lift used.
#[test]
fn what_the_dialect_does_not_have_is_refused_as_the_older_lift_refused_it() {
    let with = |body: &str| {
        format!(
            "uniform vec4 ps_uniforms_vec4[1];\nvec4 ps_r0;\n#define ps_oC0 gl_FragColor\nvoid main()\n{{\n{body}\n}}\n"
        )
    };
    let gamma = |from: &str, to: &str| {
        assert!(GAMMA.contains(from), "{from}");
        GAMMA.replacen(from, to, 1)
    };
    let cases = [
        (with("ps_r0 = tanh(ps_r0);"), Stage::Pixel, "tanh"),
        (with("while (true) { }"), Stage::Pixel, "while"),
        (with("ps_oC0 = gl_FragCoord;"), Stage::Pixel, "gl_FragCoord"),
        ("#version 450\nvoid main() {}".into(), Stage::Pixel, "version"),
        (with("ps_r0 = vec4(1.0);"), Stage::Compute, "not in the dialect"),
        (
            VERTEX.replace(
                "gl_Position.xy += posFixup.zw * gl_Position.ww;",
                "gl_Position.xy += posFixup.zw;",
            ),
            Stage::Vertex,
            "posFixup",
        ),
        (
            gamma("ps_r0 = ps_r0 * ps_v0;", "ps_r0 = ps_r0.xy * ps_v0;"),
            Stage::Pixel,
            "arithmetic on vectors of 2 and 4 lanes",
        ),
        (
            CONTROL.replacen("? ps_r1 : ps_c8", "? ps_r1 : ps_c8.xyz", 1),
            Stage::Pixel,
            "differ in type or width",
        ),
        (
            gamma("ps_r1.x = log2(ps_r0.x);", "ps_r1.x = dot(ps_r0, ps_r0.xyz);"),
            Stage::Pixel,
            "dot of vectors of different widths",
        ),
        (
            gamma("vec4 ps_r1;\n", "vec4 ps_r1;\nvec4 ps_r1;\n"),
            Stage::Pixel,
            "ps_r1 declared twice",
        ),
        (
            gamma("ps_r1.x = log2(ps_r0.x);", "ps_r1.x = 1e999;"),
            Stage::Pixel,
            "1e999, which is not finite",
        ),
        (with("ps_r0 = texture2D(ps_s0, ps_r0.xy);"), Stage::Pixel, "ps_s0"),
    ];
    let said = [
        "MojoShader GLSL at line 6: the function tanh",
        "MojoShader GLSL at line 6: while",
        "MojoShader GLSL at line 6: gl_FragCoord",
        "MojoShader GLSL at line 1: the preprocessor directive #version",
        "MojoShader GLSL at line 0: a Compute program is not in the dialect",
        "MojoShader GLSL at line 25: posFixup used outside the clip-space fix MojoShader ends a vertex program with",
        "MojoShader GLSL at line 20: arithmetic on vectors of 2 and 4 lanes",
        "MojoShader GLSL at line 25: the two sides of ?: differ in type or width",
        "MojoShader GLSL at line 21: dot of vectors of different widths",
        "MojoShader GLSL at line 11: ps_r1 declared twice",
        "MojoShader GLSL at line 21: the number 1e999, which is not finite",
        "MojoShader GLSL at line 6: texture2D of ps_s0, which is not a sampler",
    ];
    assert_eq!(cases.len(), said.len());
    for ((source, stage, words), said) in cases.into_iter().zip(said) {
        let new = ir::lift(&source, stage, &hints()).unwrap_err();
        assert!(new.detail.contains(words), "{words}: {new}");
        assert_eq!(new.to_string(), said, "{words}");
    }
}

/// What the IR lift reads of the dialect beyond the older lift (`break`, cube and volume samples) it refuses where it
/// is wrong, by name.
#[test]
fn a_break_or_a_sample_out_of_place_is_refused_by_name() {
    let refused = |source: String, stage| ir::lift(&source, stage, &hints()).unwrap_err().detail;
    let with = |body: &str| {
        format!(
            "uniform vec4 ps_uniforms_vec4[1];\nvec4 ps_r0;\n#define ps_oC0 gl_FragColor\nvoid main()\n{{\n{body}\n}}\n"
        )
    };
    let gamma = |from: &str, to: &str| {
        assert!(GAMMA.contains(from), "{from}");
        GAMMA.replacen(from, to, 1)
    };
    assert!(refused(with("break;"), Stage::Pixel).contains("break outside a loop"));
    let after = CONTROL.replacen(
        "\t\tps_r0 = ps_r0 + ps_c0;\n",
        "\t\tbreak;\n\t\tps_r0 = ps_r0 + ps_c0;\n",
        1,
    );
    assert_ne!(after, CONTROL);
    assert!(refused(after, Stage::Pixel).contains("a statement after break"));
    let cube_of_2d = refused(
        gamma("texture2D(ps_s0, ps_v1.xy)", "textureCube(ps_s0, ps_v1.xyz)"),
        Stage::Pixel,
    );
    assert!(
        cube_of_2d.contains("textureCube of ps_s0, which texture2D samples"),
        "{cube_of_2d}"
    );
    let lanes = refused(
        gamma("uniform sampler2D", "uniform samplerCube").replacen("texture2D(", "textureCube(", 1),
        Stage::Pixel,
    );
    assert!(
        lanes.contains("textureCube at a coordinate that is not 3 floats"),
        "{lanes}"
    );
}

/// The version gate (`iridia_ir::gate`): what the IR lift makes of the programs here and of the fixtures' GLSL is
/// hashed under `ir::LIFTER_VERSION` in `tests/ir-lifted.hashes`. A change to the output without a bump of the version
/// fails here, and so does a bump without the table of that version; the current table is written to the scratch root's
/// `mojoshader-ir-lifted.hashes`.
#[test]
fn what_the_ir_lift_makes_of_its_programs_is_the_versions() {
    let mut outcomes: Vec<_> = programs()
        .into_iter()
        .map(|(name, source, stage, hints)| {
            let outcome = ir::lift(source, stage, &hints)
                .map(|p| iridia_ir::gate::lifted(&p))
                .map_err(|e| e.to_string());
            (name.to_owned(), outcome)
        })
        .collect();
    for line in fixture_lines() {
        let glsl = line.glsl.clone().unwrap();
        let outcome = ir::lift(&glsl, line.stage(), &line.hints())
            .map(|p| iridia_ir::gate::lifted(&p))
            .map_err(|e| e.to_string());
        outcomes.push((format!("fixtures/{}", line.name), outcome));
    }
    let current = iridia_ir::gate::table(ir::LIFTER, ir::LIFTER_VERSION, &outcomes);
    let committed =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ir-lifted.hashes"))
            .unwrap_or_default();
    if let Err(what) = iridia_ir::gate::check(&committed, &current) {
        let new = inputs::scratch().join("mojoshader-ir-lifted.hashes");
        std::fs::write(&new, &current).unwrap();
        panic!("{what}; the current table is {}", new.display());
    }
}

/// Each fixture's GLSL beside the Direct3D 9 program it was made from, run in both executors on seeded inputs: the
/// class each pair gives.
#[test]
fn the_fixtures_glsl_runs_as_the_d3d9_programs_it_was_made_from() {
    let mut classes = Vec::new();
    for line in fixture_lines() {
        let bytes = std::fs::read(fixtures().join("d3d9").join(&line.name)).unwrap();
        let sha: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(
            sha, line.sha256,
            "{} is not the program the GLSL was made from",
            line.name
        );
        let (class, example) = pairing::pair(&line.name, &bytes, &line, 32);
        assert!(class.starts_with("agree"), "{class}: {example}");
        classes.push((line.name.clone(), class));
    }
    let expected = [
        ("colour.ps", pairing::AGREE),
        ("flow.ps", pairing::AGREE),
        ("reach.ps", pairing::AGREE),
        ("simple.ps", pairing::AGREE),
        ("simple.vs", pairing::AGREE),
        // MojoShader prints `sincos`'s range reduction, 0.159154937 and 6.28318548, as 0.159155 and 6.283185
        ("skin.vs", pairing::PRINTED),
        ("step.vs", pairing::AGREE),
    ];
    let expected: Vec<(String, String)> = expected
        .into_iter()
        .map(|(n, c)| (n.to_owned(), c.to_owned()))
        .collect();
    assert_eq!(classes, expected);
}
