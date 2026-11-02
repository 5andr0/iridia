#![allow(clippy::unwrap_used)]
//! The value evaluator ([`iridia_exec::value`]) on `fixtures/value.hlsl`, compiled to both bytecodes: each output lane
//! it evaluates from the same leaves a run of the program is given is the run's word, bit for bit, across record numbers
//! that round, saturate on conversion and wrap, slots in and past the palette, NaN, and the same values compared as
//! floats and as integers; and a lane that needs a texel the caller does not answer is unknown, naming the texel.
use iridia::flow::{At, Node, Values};
use iridia::{D3dClass, Program};
use iridia_exec::value::{Evaluated, Evaluator};
use iridia_exec::{
    Input, Invocation, Layout, PixelShader, ProgramShader, View, Views, evaluate::Constants, evaluate::Grid,
    evaluate_raw,
};
use std::collections::BTreeMap;

fn programs() -> Vec<(&'static str, Program)> {
    vec![
        (
            "DXBC",
            iridia_dxbc::lift(include_bytes!("fixtures/value.dxbc")).unwrap(),
        ),
        (
            "DXIL",
            iridia_dxil::lift(include_bytes!("fixtures/value.dxil")).unwrap(),
        ),
    ]
}

/// The texel every sample returns.
const TEXEL: [f32; 4] = [0.25, 0.5, 1.5, 1.];

/// One case: the input lanes and the constant buffers' words by slot.
struct Case {
    normal: [f32; 4],
    uv: [f32; 2],
    buffers: BTreeMap<u32, Vec<u32>>,
}

fn case(normal: [f32; 4], slot: f32) -> Case {
    // b4: 26 rows, row r = (r + 0.5, -(r + 0.25), 2r, 1)
    let palette: Vec<u32> = (0..26u32)
        .flat_map(|r| {
            let r = r as f32;
            [r + 0.5, -(r + 0.25), 2. * r, 1.]
        })
        .map(f32::to_bits)
        .collect();
    // b1: slot (.w the slot), tint (rgb toward, a the amount)
    let material: Vec<u32> = [0., 0., 0., slot, 2., -1., 0.5, 0.75]
        .into_iter()
        .map(f32::to_bits)
        .collect();
    Case {
        normal,
        uv: [0.3, 0.6],
        buffers: BTreeMap::from([(4, palette), (1, material)]),
    }
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for w in [0., 0.4, 0.5, 0.6, 1.4, 1.5, 2.5, -1., 1e10, f32::NAN] {
        for slot in [0., 2., 3., 4.] {
            out.push(case([0.25, 0.75, 1.5, w], slot));
        }
    }
    // the comparisons as floats and as unsigned integers disagree on a negative lane
    out.push(case([-0.5, 0.25, -3., 0.], 1.));
    out.push(case([0.5, -0.25, 3., 0.], 1.));
    out
}

/// A run of the program on one pixel: the target's four words.
fn run(program: &Program, c: &Case) -> [u32; 4] {
    let shader = ProgramShader::new(program.clone()).unwrap();
    let layout = Layout::of(&shader).unwrap();
    let mut constants = Constants::new(&layout);
    for (slot, words) in &c.buffers {
        let size = layout.buffers.iter().find(|b| b.0 == *slot).map_or(0, |b| b.1) as usize;
        constants
            .set_words(*slot, 0, &words[..size.min(words.len())])
            .unwrap();
    }
    let textures = Views::refusing(BTreeMap::from([(0, View::Constant(TEXEL))]));
    let inputs = BTreeMap::from([
        (("NORMAL".into(), 0), Input::Constant(c.normal)),
        (
            ("TEXCOORD".into(), 0),
            Input::Constant([c.uv[0], c.uv[1], 0., 0.]),
        ),
    ]);
    let target = shader
        .interface()
        .outputs
        .iter()
        .find(|o| o.label.name == "SV_Target")
        .map(|o| o.provenance.d3d.unwrap().register)
        .unwrap();
    let raw = evaluate_raw(
        &shader,
        &constants,
        &textures,
        &inputs,
        Grid {
            size: 1,
            period: [1., 1.],
        },
        &[(target, 15)],
        Invocation::Scalar,
    )
    .unwrap();
    raw.pixels[0][0].0
}

/// The leaves of `program` in case `c`: the inputs, the constant buffers' words (a component is word 4 row + lane, as
/// both lifts address a buffer's rows), and the texel when `texels` answers them.
fn leaf<'a>(
    program: &'a Program,
    c: &'a Case,
    texels: bool,
) -> impl FnMut(iridia::flow::NodeId, &Node, &[u32]) -> Option<u32> + 'a {
    move |_, node, at| match node {
        Node::Input { input, lane } => {
            let label = &program.interface.inputs[*input].label;
            let v = match (label.name.to_ascii_uppercase().as_str(), label.index) {
                ("NORMAL", 0) => c.normal,
                ("TEXCOORD", 0) => [c.uv[0], c.uv[1], 0., 0.],
                _ => return None,
            };
            Some(v[*lane as usize].to_bits())
        }
        Node::Global { resource, at: place } => {
            let r = program.interface.resource(resource.as_ref()?.clone())?;
            let d3d = r.provenance.d3d.as_ref()?;
            if d3d.class != D3dClass::ConstantBuffer {
                return None;
            }
            let words = c.buffers.get(&d3d.register)?;
            let word = match place {
                At::Component(w) => *w as usize,
                At::Indexed { lane, .. } => match at {
                    [row] => (*row as usize).checked_mul(4)?.checked_add(*lane as usize)?,
                    _ => return None,
                },
            };
            // a word past what the caller binds reads zero, as the run's buffer does
            Some(words.get(word).copied().unwrap_or(0))
        }
        Node::Texel { lane, .. } if texels => Some(TEXEL[*lane as usize].to_bits()),
        _ => None,
    }
}

#[test]
fn each_lane_evaluated_is_the_runs_word_bit_for_bit() {
    for (name, program) in programs() {
        let values = Values::of_program(&program).unwrap();
        let target = program
            .interface
            .outputs
            .iter()
            .position(|o| o.label.name == "SV_Target")
            .unwrap();
        for (i, c) in cases().iter().enumerate() {
            let ran = run(&program, c);
            let mut e = Evaluator::new(&values, program.modes.flush_denormals, leaf(&program, c, true));
            for lane in 0..4u8 {
                let node = values.output(target, lane).unwrap();
                assert_eq!(
                    e.evaluate(node).unwrap(),
                    Evaluated::Word(ran[lane as usize]),
                    "{name} case {i} lane {lane}: normal {:?}",
                    c.normal
                );
            }
        }
    }
}

#[test]
fn a_lane_that_needs_an_unanswered_texel_is_unknown_and_names_it() {
    for (name, program) in programs() {
        let values = Values::of_program(&program).unwrap();
        let target = program
            .interface
            .outputs
            .iter()
            .position(|o| o.label.name == "SV_Target")
            .unwrap();
        let c = case([0.25, 0.75, 1.5, 0.], 0.);
        let ran = run(&program, &c);
        let mut e = Evaluator::new(&values, program.modes.flush_denormals, leaf(&program, &c, false));
        for lane in 0..3u8 {
            match e.evaluate(values.output(target, lane).unwrap()).unwrap() {
                Evaluated::Unknown(id) => {
                    assert!(
                        matches!(values.node(id), Node::Texel { .. }),
                        "{name} lane {lane}"
                    )
                }
                other => panic!("{name} lane {lane}: {other:?}"),
            }
        }
        // the alpha lane reads no texel
        assert_eq!(
            e.evaluate(values.output(target, 3).unwrap()).unwrap(),
            Evaluated::Word(ran[3]),
            "{name}"
        );
    }
}
