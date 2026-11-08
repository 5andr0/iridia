#![allow(clippy::unwrap_used)]
//! `iridia_passes::clip_distances_as_varyings` and `discard_clipped` on the corpora's real programs: every vertex program
//! with a clip distance output is made carry them as plain varyings and written for a browser (which has no clipper for
//! them: refused before the pass), and pixel programs given the matching inputs and discards are written and validated
//! the same way. A program the browser profile still refuses is counted by its reason, none of which may be the clip
//! distances. Needs the corpus oracle set's `sekiro/shaders` or `elden/shaders`; `shader.diff_limit` caps the programs.
mod common;
mod inputs;

use std::collections::BTreeMap;

use iridia_ir::interface::{Stage, SystemValue};
use iridia_naga::Profile;
use iridia_passes::{ClipVaryings, clip_distances_as_varyings, discard_clipped};

fn has_dxil(bytes: &[u8]) -> bool {
    let word = |at: usize| -> Option<usize> {
        let b = bytes.get(at..at.checked_add(4)?)?;
        usize::try_from(u32::from_le_bytes(b.try_into().ok()?)).ok()
    };
    let Some(count) = word(28) else {
        return false;
    };
    (0..count.min(64)).any(|k| {
        word(32 + 4 * k)
            .and_then(|at| bytes.get(at..at.checked_add(4)?))
            .is_some_and(|k| k == b"DXIL")
    })
}

fn lifted(bytes: &[u8]) -> Result<iridia_ir::Program, String> {
    let mut p = if has_dxil(bytes) {
        iridia_dxil::ir::lift(bytes)
    } else {
        iridia_dxbc::ir::lift(bytes)
    }
    .map_err(|e| format!("the IR lift: {e}"))?;
    iridia_passes::prepare(&mut p)?;
    iridia_ir::validate(&p).map_err(|e| format!("{e:?}"))?;
    Ok(p)
}

/// The text read back and validated by naga's front end.
fn accepted(text: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(text).map_err(|e| e.emit_to_string(text))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .map(|_| ())
    .map_err(|e| format!("{e:?}"))
}

fn clips_of(p: &iridia_ir::Program) -> bool {
    p.interface
        .outputs
        .iter()
        .any(|v| v.system == Some(SystemValue::ClipDistance))
}

fn corpus(game: inputs::Game) {
    let Some(corpus) = common::corpus(game) else {
        return;
    };
    let (mut vertex, mut written, mut pixel_written) = (0usize, 0usize, 0usize);
    let mut refused: BTreeMap<String, usize> = BTreeMap::new();
    let mut first_clips: Option<ClipVaryings> = None;
    let mut pixels = Vec::new();
    for file in corpus.names.iter().take(common::diff_limit()) {
        let Ok(p) = lifted(&corpus.read(file)) else {
            continue;
        };
        match p.interface.stage {
            Stage::Vertex if clips_of(&p) => {
                vertex += 1;
                // refused before the pass, by the clip distances
                if let Err(e) = iridia_naga::wgsl(&p, &Profile::BROWSER) {
                    assert!(
                        e.to_string().contains("clip distance"),
                        "{}: {e}",
                        corpus.at(file)
                    );
                }
                let mut q = p.clone();
                let clips = clip_distances_as_varyings(&mut q)
                    .unwrap()
                    .expect("a clip distance output");
                iridia_ir::validate(&q).unwrap();
                assert!(!clips_of(&q));
                match iridia_naga::wgsl(&q, &Profile::BROWSER) {
                    Ok(text) => {
                        accepted(&text).unwrap_or_else(|e| panic!("{}: {e}\n{text}", corpus.at(file)));
                        written += 1;
                        first_clips.get_or_insert(clips);
                    }
                    Err(e) => {
                        let why = common::reason(&e.to_string());
                        assert!(!why.contains("clip distance"), "{}: {e}", corpus.at(file));
                        *refused.entry(why).or_default() += 1;
                    }
                }
            }
            Stage::Pixel if pixels.len() < 200 => pixels.push(p),
            _ => {}
        }
    }
    println!(
        "{}: {vertex} vertex programs with clip distances, {written} written for a browser after the pass",
        corpus.name
    );
    for (why, n) in &refused {
        println!("  {n:5} still refused: {why}");
    }
    let Some(clips) = first_clips else {
        return;
    };
    // the pixel side: every pixel program of the sample takes the first program's varyings and discards on them
    let mut pixel_refused: BTreeMap<String, usize> = BTreeMap::new();
    for mut p in pixels {
        let before = p.interface.inputs.len();
        if let Err(e) = discard_clipped(&mut p, &clips) {
            *pixel_refused.entry(common::reason(&e)).or_default() += 1;
            continue;
        }
        assert_eq!(p.interface.inputs.len(), before + clips.varyings.len());
        iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&p.entry)));
        match iridia_naga::wgsl(&p, &Profile::BROWSER) {
            Ok(text) => {
                accepted(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
                assert!(text.contains("discard;"), "{text}");
                pixel_written += 1;
            }
            Err(e) => *pixel_refused.entry(common::reason(&e.to_string())).or_default() += 1,
        }
    }
    println!(
        "  {pixel_written} pixel programs given the discard, written for a browser and accepted by naga"
    );
    for (why, n) in &pixel_refused {
        println!("  {n:5} pixel programs not written: {why}");
    }
}

#[test]
fn sekiro_clip_distances_as_varyings() {
    corpus(inputs::Game::Sekiro);
}

#[test]
fn elden_clip_distances_as_varyings() {
    corpus(inputs::Game::Elden);
}
