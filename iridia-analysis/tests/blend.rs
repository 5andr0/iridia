//! The blend recogniser on the IR, on a program Microsoft's compiler wrote from hand-written HLSL
//! (`iridia/tests/fixtures/blend.hlsl`): the layers, weights and layer values it reads are the source's, worked by hand
//! below, and what it reads computes what the executor computes running the program, bit for bit.
#![allow(clippy::unwrap_used)]
use std::collections::{BTreeMap, BTreeSet};

use iridia_analysis::blend::{Over, Scalar, Term};
use iridia_analysis::flow::{At, Node, NodeId, Values};
use iridia_exec::probe::AnsweredReads;
use iridia_exec::{ImageSample, PixelShader, ProgramShader};
use iridia_ir::Program;
use iridia_ir::interface::ResourceKind;

fn program() -> Program {
    iridia_dxbc::ir::lift(include_bytes!("../../iridia/tests/fixtures/blend_over.dxbc")).unwrap()
}

/// What a leaf is, by the source's names: `colour.x`, `uv.y`, `base_map.g`, `Material[5]` (a word of the buffer).
fn name(program: &Program, values: &Values, n: NodeId) -> String {
    let lane = |l: u8| ["x", "y", "z", "w"][usize::from(l)];
    match values.node(n) {
        Node::Input { input, lane: l } => {
            let label = &program.interface.inputs[*input].label;
            let short = match label.name.as_str() {
                "TEXCOORD" => "uv",
                "COLOR" => "colour",
                other => other,
            };
            format!("{short}.{}", lane(*l))
        }
        Node::Texel {
            resource, lane: l, ..
        } => {
            let r = program
                .interface
                .resources
                .iter()
                .find(|r| r.id == *resource)
                .unwrap();
            format!("{}.{}", r.provenance.name, ["r", "g", "b", "a"][usize::from(*l)])
        }
        Node::Global {
            resource,
            at: At::Component(c),
        } => {
            let r = program
                .interface
                .resources
                .iter()
                .find(|r| r.id == *resource)
                .unwrap();
            format!("{}[{c}]", r.provenance.name)
        }
        other => format!("{other:?}"),
    }
}

/// A scalar as text, each operation in brackets, leaves by [`name`].
fn text(program: &Program, values: &Values, s: &Scalar) -> String {
    fn at(program: &Program, values: &Values, s: &Scalar, i: usize) -> String {
        let t = |k: &usize| at(program, values, s, *k);
        match &s.terms[i] {
            Term::Leaf(n) => name(program, values, *n),
            Term::Constant(c) => format!("{c}"),
            Term::Add([a, b]) => format!("({} + {})", t(a), t(b)),
            Term::Subtract([a, b]) => format!("({} - {})", t(a), t(b)),
            Term::Multiply([a, b]) => format!("({} * {})", t(a), t(b)),
            Term::Min([a, b]) => format!("min({}, {})", t(a), t(b)),
            Term::Max([a, b]) => format!("max({}, {})", t(a), t(b)),
            Term::Saturate(a) => format!("saturate({})", t(a)),
        }
    }
    at(program, values, s, s.terms.len() - 1)
}

fn scalar(program: &Program, values: &Values, n: NodeId) -> String {
    text(program, values, &Scalar::of(values, n, &BTreeSet::new()))
}

/// The layers, worked by hand from the source and the compiler's listing (`fxc /dumpbin`): fxc writes each lerp as
/// `mad(weight, layer + -below, below)`, the last under `mad_sat`; `1 - sharpness * x` as `mad_sat(-sharpness, x, 1)`;
/// `2 * (1 - c) - m` as `mad(1 - c, 2, -m)`.
#[test]
fn layers_weights_and_values_are_the_sources() {
    let program = program();
    let values = Values::of_program(&program).unwrap();
    let mut weights = Vec::new();
    for (lane, channel) in ["r", "g", "b"].into_iter().enumerate() {
        let node = values.output(0, lane as u8).unwrap();
        let over = Over::under(&values, node).unwrap();
        assert_eq!(
            scalar(&program, &values, over.base),
            format!("base_map.{channel}")
        );
        assert_eq!(over.layers.len(), 2);
        assert_eq!(
            scalar(&program, &values, over.layers[0].value),
            format!("(layer_map.{channel} * 1.8)")
        );
        assert_eq!(
            scalar(&program, &values, over.layers[1].value),
            format!("top_map.{channel}")
        );
        // the lane is the top layer's interpolation, saturated, times the tint
        let top = over.layers[1].node;
        let outside = Scalar::of(&values, node, &BTreeSet::from([top]));
        assert_eq!(
            text(&program, &values, &outside).replace(&name(&program, &values, top), "top"),
            format!("(saturate(top) * Material[{lane}])")
        );
        weights.push([over.layers[0].weight, over.layers[1].weight]);
    }
    // one weight per layer, shared by the lanes
    assert!(weights.iter().all(|w| *w == weights[0]));
    let c = "saturate(colour.x)";
    assert_eq!(
        scalar(&program, &values, weights[0][0]),
        format!(
            "((Material[4] * (saturate((((-1 * Material[3]) * (((1 - {c}) * 2) - (1 - mask_map.r))) + 1)) - {c})) + {c})"
        )
    );
    assert_eq!(
        scalar(&program, &values, weights[0][1]),
        "min(max(colour.y, 0.25), 0.75)"
    );
}

/// The executor's register view of the fixture: v1 the uv, v2 the colour; t0 to t3 the four maps.
struct Case {
    colour: [f32; 4],
    texels: [[f32; 4]; 4],
    words: [f32; 8],
}

/// A leaf's value in `case`, by what the node is.
fn leaf(program: &Program, values: &Values, case: &Case, n: NodeId) -> Option<f32> {
    match values.node(n) {
        Node::Input { input, lane } => {
            let label = &program.interface.inputs[*input].label;
            (label.name == "COLOR").then(|| case.colour[usize::from(*lane)])
        }
        Node::Texel { resource, lane, .. } => {
            let r = program.interface.resources.iter().find(|r| r.id == *resource)?;
            let slot = r.provenance.d3d.as_ref()?.register as usize;
            Some(case.texels[slot][usize::from(*lane)])
        }
        Node::Global {
            at: At::Component(c), ..
        } => case.words.get(*c as usize).copied(),
        _ => None,
    }
}

fn run(program: &Program, case: &Case) -> [f32; 4] {
    let shader = ProgramShader::from_ir(program.clone()).unwrap();
    let inputs = [
        [0; 4],
        [0.25f32.to_bits(), 0.75f32.to_bits(), 0, 0],
        case.colour.map(f32::to_bits),
    ];
    let words = [
        [0, 1, 2, 3].map(|k| case.words[k].to_bits()),
        [4, 5, 6, 7].map(|k| case.words[k].to_bits()),
    ];
    let texels = case.texels;
    let sample = move |s: &ImageSample<u32>| Ok(texels[s.texture as usize]);
    let out = shader
        .execute(&inputs, &[(0, &words)], &mut AnsweredReads::new(sample))
        .unwrap();
    out.outputs[0].map(f32::from_bits)
}

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) || (a == 0.0 && b == 0.0)
}

/// What the recogniser reads computes the program's output, bit for bit: each lane as one [`Scalar`], and each lane as
/// its [`Over`] chain (`w * (layer - below) + below`, as fxc rounds it) inside what the lane makes of the top layer.
/// The cases put weights inside, at and past 0 and 1, colours past 1 (`layer_map` times 1.8), and NaN and infinity
/// where the guards decide.
#[test]
fn what_it_reads_computes_what_the_program_does() {
    let program = program();
    let values = Values::of_program(&program).unwrap();
    assert!(
        program
            .interface
            .resources
            .iter()
            .filter(|r| matches!(r.kind, ResourceKind::Texture { .. }))
            .count()
            == 4
    );
    let colours = [
        [0.3, 0.6, 0.0, 1.0],
        [0.9, 0.1, 0.0, 1.0],
        [-0.5, 2.0, 0.0, 1.0],
        [1.5, -1.0, 0.0, 1.0],
        [f32::NAN, f32::NAN, 0.0, 1.0],
        [f32::INFINITY, 0.5, 0.0, 1.0],
        [1.0 / 3.0, 0.7, 0.0, 1.0],
    ];
    let texel_sets = [
        [
            [0.2, 0.4, 0.6, 1.0],
            [0.9, 0.5, 0.1, 1.0],
            [0.1, 0.8, 0.3, 1.0],
            [0.25, 0.0, 0.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.5, 1.0],
            [0.3, 0.7, 0.95, 1.0],
            [0.6, 0.6, 0.0, 1.0],
            [0.9, 0.0, 0.0, 1.0],
        ],
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.5, 0.25, 0.125, 1.0],
            [f32::NAN, 0.0, 0.0, 1.0],
        ],
    ];
    let word_sets = [
        [1.0, 0.8, 0.6, 1.4, 0.6, 0.0, 0.0, 0.0],
        [0.5, 1.0, 2.0, 2.8, 0.8, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, f32::INFINITY, 1.0, 0.0, 0.0, 0.0],
    ];
    let mut checked = 0;
    for colour in colours {
        for texels in texel_sets {
            for words in word_sets {
                let case = Case {
                    colour,
                    texels,
                    words,
                };
                let out = run(&program, &case);
                let at = |n: NodeId| leaf(&program, &values, &case, n);
                for lane in 0..3u8 {
                    let node = values.output(0, lane).unwrap();
                    let whole = Scalar::of(&values, node, &BTreeSet::new());
                    let got = whole.evaluate(at);
                    let want = out[usize::from(lane)];
                    assert!(same(got, want), "lane {lane} as one scalar: {got} for {want}");
                    let over = Over::under(&values, node).unwrap();
                    let value = |n: NodeId| Scalar::of(&values, n, &BTreeSet::new()).evaluate(at);
                    let mut below = value(over.base);
                    let mut layered = BTreeMap::new();
                    for layer in &over.layers {
                        let (w, l) = (value(layer.weight), value(layer.value));
                        below = w * (l - below) + below;
                        layered.insert(layer.node, below);
                    }
                    let top = over.layers.last().unwrap().node;
                    let outside = Scalar::of(&values, node, &BTreeSet::from([top]));
                    let got = outside.evaluate(|n| if n == top { layered.get(&n).copied() } else { at(n) });
                    assert!(same(got, want), "lane {lane} as layers: {got} for {want}");
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 7 * 3 * 3 * 3);
}

/// A float sum and an integer sum of the same two registers are two nodes, each of its kind, and the float reader
/// spells only the float one: the integer sum's bits are a leaf.
#[test]
fn an_operation_is_of_its_kind() {
    use iridia_ir::Scalar as Kind;
    let program =
        iridia_dxbc::ir::lift(include_bytes!("../../iridia/tests/fixtures/blend_kinds.dxbc")).unwrap();
    let values = Values::of_program(&program).unwrap();
    let (float, integer) = (values.output(0, 0).unwrap(), values.output(0, 1).unwrap());
    assert_ne!(float, integer);
    assert_eq!(values.operands(float), values.operands(integer));
    assert_eq!(values.kind(float), Some(Kind::F32));
    // `iadd`, which the lifter computes on u32
    assert_eq!(values.kind(integer), Some(Kind::U32));
    assert_eq!(scalar(&program, &values, float), "(colour.y + colour.x)");
    assert_eq!(
        Scalar::of(&values, integer, &BTreeSet::new()).terms,
        [Term::Leaf(integer)]
    );
    // an input lane is bits each use reads in its own kind
    assert_eq!(values.kind(values.operands(float)[0]), None);
}
