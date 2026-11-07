#![allow(clippy::unwrap_used)]
//! Programs in MojoShader's dialect lifted onto the IR (`iridia_mojo::ir`): they validate, print and read back as the
//! same text, take the passes, are written as WGSL and have the older lift's interface; what the dialect does not have
//! is refused by name, as the older lift refuses it; and the GLSL MonoGame's MojoShader made of the fixtures' Direct3D
//! 9 programs (`tests/fixtures`) runs as those programs do (`tests/pairing`).
mod inputs;
mod pairing;
mod programs;

use std::path::{Path, PathBuf};

use iridia::Stage;
use iridia_mojo::{Hints, ir};
use programs::{CONTROL, GAMMA, VERTEX, hints};
use sha2::{Digest, Sha256};

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
        let old = iridia_mojo::lift(source, stage, &hints).unwrap();
        assert_eq!(p.interface, old.interface, "{name}");
        assert_eq!(p.notes, old.notes, "{name}");
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

/// What both lifts refuse, they refuse alike: the same line and reason.
#[test]
fn what_the_dialect_does_not_have_is_refused_as_the_older_lift_refuses_it() {
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
    for (source, stage, words) in cases {
        let new = ir::lift(&source, stage, &hints()).unwrap_err();
        let old = iridia_mojo::lift(&source, stage, &hints()).unwrap_err();
        assert!(new.detail.contains(words), "{words}: {new}");
        assert_eq!(new.to_string(), old.to_string(), "{words}");
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
/// class each pair gives. `colour.ps`'s `pow` is naga's in the older lift, which `iridia-exec` computes otherwise; the
/// older lift reads none of `reach.ps` (a cube and a volume sample, `break`), `skin.vs` or `step.vs` (the fog varying,
/// a float).
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
    let refused = format!("{}; the older lift (A) refuses it", pairing::AGREE);
    let expected = [
        (
            "colour.ps",
            format!("{}; the older lift (A) does not", pairing::AGREE),
        ),
        ("flow.ps", pairing::AGREE.to_owned()),
        ("reach.ps", refused.clone()),
        ("simple.ps", pairing::AGREE.to_owned()),
        ("simple.vs", pairing::AGREE.to_owned()),
        // MojoShader prints `sincos`'s range reduction, 0.159154937 and 6.28318548, as 0.159155 and 6.283185
        (
            "skin.vs",
            format!("{}; the older lift (A) refuses it", pairing::PRINTED),
        ),
        ("step.vs", refused),
    ];
    let expected: Vec<(String, String)> = expected.into_iter().map(|(n, c)| (n.to_owned(), c)).collect();
    assert_eq!(classes, expected);
}
