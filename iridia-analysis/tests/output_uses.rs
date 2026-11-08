#![allow(clippy::unwrap_used)]
//! What a program's discard and its targets' blend show of its output lanes (`output_uses`), on
//! `fixtures/output_uses.hlsl`, each entry point compiled to both bytecodes and lifted: an alpha-tested
//! program's tested alpha is a lane a discard tests where it writes it, a discard on a value written nowhere tests no
//! lane, and a target blended by the source alpha has its alpha lane as a blend weight.
use iridia_analysis::ProgramGraph;
use iridia_analysis::output_uses::output_uses;
use iridia_ir::interface::{Blend, BlendEquation, BlendFactor, BlendOp, OutputUse, RenderState, TargetState};
use iridia_ir::{Program, interface::Label};

/// Each entry point of the fixture, as fxc and dxc compile it.
const FIXTURES: [(&str, &[u8], &[u8]); 3] = [
    (
        "tested",
        include_bytes!("../../iridia/tests/fixtures/output_uses_tested.dxbc"),
        include_bytes!("../../iridia/tests/fixtures/output_uses_tested.dxil"),
    ),
    (
        "unrelated",
        include_bytes!("../../iridia/tests/fixtures/output_uses_unrelated.dxbc"),
        include_bytes!("../../iridia/tests/fixtures/output_uses_unrelated.dxil"),
    ),
    (
        "opaque",
        include_bytes!("../../iridia/tests/fixtures/output_uses_opaque.dxbc"),
        include_bytes!("../../iridia/tests/fixtures/output_uses_opaque.dxil"),
    ),
];

/// The entry point `entry` lifted from each bytecode.
fn lifted(entry: &str) -> Vec<(&'static str, Program)> {
    let (_, dxbc, dxil) = FIXTURES.iter().find(|f| f.0 == entry).expect("a fixture");
    vec![
        ("dxbc", iridia_dxbc::ir::lift(dxbc).expect("fxc's program lifts")),
        ("dxil", iridia_dxil::ir::lift(dxil).expect("dxc's program lifts")),
    ]
}

/// What the program's code and `state` show of target `n`: its blend weight and the lanes a discard tests, of the lanes
/// `of` (the fixture's lanes each test makes a claim about).
fn shown(uses: &[OutputUse], n: u32, of: u8) -> (u8, u8) {
    let u = uses
        .iter()
        .find(|u| u.target == n)
        .unwrap_or_else(|| panic!("target {n} is observed: {uses:?}"));
    assert_eq!(
        u.label,
        Label {
            name: "SV_Target".into(),
            index: n
        }
    );
    (u.blend_weight & of, u.discard_tested & of)
}

/// The uses on `program` under `state`, with its value graph.
fn observed(program: &Program, state: &RenderState) -> Vec<OutputUse> {
    let graph = ProgramGraph::of_program(program).expect("the value graph");
    let uses = output_uses(&program.interface, Some(&graph), state);
    assert_eq!(uses.len(), 3, "the three targets: {uses:?}");
    uses
}

/// Target 2 blended by `source` over `destination`, colour and alpha alike.
fn blended(source: BlendFactor, destination: BlendFactor) -> RenderState {
    let equation = BlendEquation {
        source,
        destination,
        op: BlendOp::Add,
    };
    let mut state = RenderState::default();
    state.targets[2] = Some(TargetState {
        blend: Some(Blend {
            colour: equation,
            alpha: equation,
        }),
        write_mask: 0xf,
    });
    state
}

/// The lanes each test makes a claim about: target 0's w and target 1's yzw are literals (no evidence), target 2's w the
/// alpha the program writes.
const CLAIMED: [(u32, u8); 3] = [(0, 0b1000), (1, 0b1110), (2, 0b1000)];

/// Each claimed lane's blend weight and discard test.
fn claimed(uses: &[OutputUse]) -> Vec<(u32, u8, u8)> {
    CLAIMED
        .iter()
        .map(|&(n, of)| {
            let (blend, tested) = shown(uses, n, of);
            (n, blend, tested)
        })
        .collect()
}

const NOTHING: [(u32, u8, u8); 3] = [(0, 0, 0), (1, 0, 0), (2, 0, 0)];

#[test]
fn an_alpha_tested_programs_tested_alpha_is_a_tested_lane_where_it_writes_it() {
    for (bytecode, program) in lifted("tested") {
        let uses = observed(&program, &RenderState::default());
        // the lanes it writes as the literal 0 are no evidence
        assert_eq!(
            claimed(&uses),
            [(0, 0, 0), (1, 0, 0), (2, 0, 0b1000)],
            "{bytecode}"
        );
    }
}

#[test]
fn a_discard_on_a_value_written_nowhere_tests_no_lane() {
    for (bytecode, program) in lifted("unrelated") {
        assert!(
            !ProgramGraph::of_program(&program)
                .unwrap()
                .values
                .discards()
                .is_empty(),
            "{bytecode}"
        );
        let uses = observed(&program, &RenderState::default());
        assert_eq!(claimed(&uses), NOTHING, "{bytecode}");
    }
}

#[test]
fn an_opaque_target_shows_nothing_and_one_blended_by_the_source_alpha_weighs_by_its_alpha() {
    for (bytecode, program) in lifted("opaque") {
        assert!(
            ProgramGraph::of_program(&program)
                .unwrap()
                .values
                .discards()
                .is_empty(),
            "{bytecode}"
        );
        let uses = observed(&program, &RenderState::default());
        assert!(
            uses.iter().all(|u| u.blend_weight == 0 && u.discard_tested == 0),
            "{bytecode}: {uses:?}"
        );
        // a blend that does not weigh by the source alpha is no evidence
        let uses = observed(&program, &blended(BlendFactor::One, BlendFactor::One));
        assert_eq!(claimed(&uses), NOTHING, "{bytecode}");
        for (source, destination) in [
            (BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha),
            (BlendFactor::One, BlendFactor::OneMinusSrcAlpha),
        ] {
            let uses = observed(&program, &blended(source, destination));
            assert_eq!(
                claimed(&uses),
                [(0, 0, 0), (1, 0, 0), (2, 0b1000, 0)],
                "{bytecode} {source:?} {destination:?}"
            );
        }
    }
}

/// A lane both tested and blended by the source alpha shows both; which it stands for is the caller's to decide.
#[test]
fn a_tested_alpha_the_target_blends_by_shows_both() {
    for (bytecode, program) in lifted("tested") {
        let state = blended(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha);
        let uses = observed(&program, &state);
        assert_eq!(shown(&uses, 2, 0b1000), (0b1000, 0b1000), "{bytecode}");
    }
}

/// With no value graph no lane is tested; the blend is still shown.
#[test]
fn without_a_value_graph_no_lane_is_tested() {
    for (bytecode, program) in lifted("tested") {
        let state = blended(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha);
        let uses = output_uses(&program.interface, None, &state);
        assert!(uses.iter().all(|u| u.discard_tested == 0), "{bytecode}: {uses:?}");
        assert_eq!(shown(&uses, 2, 0b1111), (0b1000, 0), "{bytecode}");
    }
}

/// A fixture, a bytecode, a state's index and the `(target, blend_weight, discard_tested)` of each target.
type Shown = (&'static str, &'static str, usize, [(u32, u8, u8); 3]);

/// What `output_uses` shows of each fixture and bytecode under each render state: (fixture, bytecode, state, the
/// `(target, blend_weight, discard_tested)` of each target). The four states are, in order, the default, `One`/`One`,
/// `SrcAlpha`/`OneMinusSrcAlpha` and `One`/`OneMinusSrcAlpha` on target 2. Taken from the analysis this one replaced
/// (the rows were equal to its on all 24 cases before it was removed); the tests above check their cases by hand.
const SHOWN: [Shown; 24] = [
    ("tested", "dxbc", 0, [(0, 0, 0), (1, 0, 0), (2, 0, 8)]),
    ("tested", "dxbc", 1, [(0, 0, 0), (1, 0, 0), (2, 0, 8)]),
    ("tested", "dxbc", 2, [(0, 0, 0), (1, 0, 0), (2, 8, 8)]),
    ("tested", "dxbc", 3, [(0, 0, 0), (1, 0, 0), (2, 8, 8)]),
    ("tested", "dxil", 0, [(0, 0, 0), (1, 0, 0), (2, 0, 8)]),
    ("tested", "dxil", 1, [(0, 0, 0), (1, 0, 0), (2, 0, 8)]),
    ("tested", "dxil", 2, [(0, 0, 0), (1, 0, 0), (2, 8, 8)]),
    ("tested", "dxil", 3, [(0, 0, 0), (1, 0, 0), (2, 8, 8)]),
    ("unrelated", "dxbc", 0, [(0, 0, 0), (1, 0, 0), (2, 0, 0)]),
    ("unrelated", "dxbc", 1, [(0, 0, 0), (1, 0, 0), (2, 0, 0)]),
    ("unrelated", "dxbc", 2, [(0, 0, 0), (1, 0, 0), (2, 8, 0)]),
    ("unrelated", "dxbc", 3, [(0, 0, 0), (1, 0, 0), (2, 8, 0)]),
    ("unrelated", "dxil", 0, [(0, 0, 0), (1, 0, 0), (2, 0, 0)]),
    ("unrelated", "dxil", 1, [(0, 0, 0), (1, 0, 0), (2, 0, 0)]),
    ("unrelated", "dxil", 2, [(0, 0, 0), (1, 0, 0), (2, 8, 0)]),
    ("unrelated", "dxil", 3, [(0, 0, 0), (1, 0, 0), (2, 8, 0)]),
    ("opaque", "dxbc", 0, [(0, 0, 0), (1, 0, 0), (2, 0, 0)]),
    ("opaque", "dxbc", 1, [(0, 0, 0), (1, 0, 0), (2, 0, 0)]),
    ("opaque", "dxbc", 2, [(0, 0, 0), (1, 0, 0), (2, 8, 0)]),
    ("opaque", "dxbc", 3, [(0, 0, 0), (1, 0, 0), (2, 8, 0)]),
    ("opaque", "dxil", 0, [(0, 0, 0), (1, 0, 0), (2, 0, 0)]),
    ("opaque", "dxil", 1, [(0, 0, 0), (1, 0, 0), (2, 0, 0)]),
    ("opaque", "dxil", 2, [(0, 0, 0), (1, 0, 0), (2, 8, 0)]),
    ("opaque", "dxil", 3, [(0, 0, 0), (1, 0, 0), (2, 8, 0)]),
];

#[test]
fn output_uses_show_what_they_showed_under_each_blend() {
    let states = [
        RenderState::default(),
        blended(BlendFactor::One, BlendFactor::One),
        blended(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha),
        blended(BlendFactor::One, BlendFactor::OneMinusSrcAlpha),
    ];
    let mut compared = 0;
    for (name, bytecode, k, expected) in SHOWN {
        let (_, program) = lifted(name).into_iter().find(|(b, _)| *b == bytecode).unwrap();
        let graph = ProgramGraph::of_program(&program).unwrap();
        let uses = output_uses(&program.interface, Some(&graph), &states[k]);
        let got: Vec<_> = uses
            .iter()
            .map(|u| (u.target, u.blend_weight, u.discard_tested))
            .collect();
        assert_eq!(got, expected, "{name} {bytecode} state {k}");
        compared += 1;
    }
    assert_eq!(compared, 24);
}
