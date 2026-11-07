#![allow(clippy::unwrap_used)]
//! Properties of the analyses on generated programs: pixel programs of variables, ifs, loops, kills and early
//! returns over an input's lanes, a constant buffer's words and float arithmetic, drawn by proptest from a byte
//! string (so a failure shrinks to a small program).
//!
//! - What [`Flow`] leaves out of a lane's reach cannot change it: each input lane and constant word left out of an
//!   output lane's reach (or the discard's) is changed in runs of the program in `iridia-exec-ir`, and the lane stays.
//! - [`WordsRead`] holds every word that reaches a lane or the discard.
//! - The graph is closed: each operand of a node is a node of it, and each output lane the program has is in it.
#[allow(
    dead_code,
    reason = "the oracle changes texels; the generated programs read none"
)]
mod perturb;

use std::fmt::Write as _;

use iridia_analysis::flow::{Flow, GlobalWord, Lane, Reach, Values};
use iridia_analysis::reads::WordsRead;
use iridia_ir::interface::{
    BufferLayout, D3dBinding, D3dClass, Interface, Label, Resource, ResourceId, ResourceKind,
    ResourceProvenance, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value, Varying,
};
use iridia_ir::{Conventions, FloatEnv, Program, Provenance};
use proptest::prelude::*;

/// The constant buffer's words.
const WORDS: u32 = 16;
/// The variables.
const VARS: usize = 3;

/// A program's text drawn from a byte string: each choice the next byte (zero past its end).
struct Gen<'a> {
    bytes: &'a [u8],
    at: usize,
    next: u32,
    text: String,
    indent: usize,
    labels: u32,
    /// The loops the code is in, innermost last: each one's label.
    loops: Vec<u32>,
}

impl Gen<'_> {
    fn byte(&mut self) -> u8 {
        let b = self.bytes.get(self.at).copied().unwrap_or(0);
        self.at += 1;
        b
    }

    fn pick(&mut self, n: u8) -> u8 {
        self.byte() % n
    }

    fn line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.text.push_str("  ");
        }
        self.text.push_str(s);
        self.text.push('\n');
    }

    /// A new value of type `ty` defined by `def`, its id.
    fn value(&mut self, ty: &str, def: &str) -> u32 {
        let id = self.next;
        self.next += 1;
        self.line(&format!("%{id}: {ty} = {def}"));
        id
    }

    /// An `f32` expression, its value's id.
    fn expr(&mut self, depth: u8) -> u32 {
        let choice = if depth == 0 { self.pick(4) } else { self.pick(8) };
        match choice {
            0 => {
                let lane = ["x", "y", "z", "w"][usize::from(self.pick(4))];
                self.value("f32", &format!("input 0.{lane}"))
            }
            1 => {
                let v = self.pick(VARS as u8);
                self.value("f32", &format!("load v{v}"))
            }
            2 => {
                let x = f32::from(self.pick(9)) / 2.0 - 2.0;
                self.value("f32", &format!("f32({x:?})"))
            }
            3 => {
                let word = u32::from(self.pick(WORDS as u8));
                let row = self.value("vec4<u32>", &format!("buffer r0 + {} oob.zero", 16 * (word / 4)));
                let lane = ["x", "y", "z", "w"][(word % 4) as usize];
                let w = self.value("u32", &format!("%{row}.{lane}"));
                self.value("f32", &format!("bitcast %{w}"))
            }
            4..=6 => {
                let (a, b) = (self.expr(depth - 1), self.expr(depth - 1));
                let op = ["fadd", "fsub", "fmul.ieee"][usize::from(choice - 4)];
                self.value("f32", &format!("{op} %{a}, %{b}"))
            }
            _ => {
                let c = self.condition(depth - 1);
                let (a, b) = (self.expr(depth - 1), self.expr(depth - 1));
                self.value("f32", &format!("select %{c}, %{a}, %{b}"))
            }
        }
    }

    /// A `bool` comparison of two expressions, its id.
    fn condition(&mut self, depth: u8) -> u32 {
        let (a, b) = (self.expr(depth), self.expr(depth));
        self.value("bool", &format!("fcmp.lt %{a}, %{b}"))
    }

    /// Each output lane stored from a variable or an expression.
    fn outputs(&mut self) {
        for lane in ["x", "y", "z", "w"] {
            let v = self.expr(1);
            self.line(&format!("output 0.{lane} = %{v}"));
        }
    }

    /// An `if` on a condition whose arms are `then` and `otherwise`.
    fn branch(&mut self, then: impl FnOnce(&mut Self), otherwise: impl FnOnce(&mut Self)) {
        let c = self.condition(1);
        self.line(&format!("if %{c} {{"));
        self.indent += 1;
        then(self);
        self.indent -= 1;
        self.line("} else {");
        self.indent += 1;
        otherwise(self);
        self.indent -= 1;
        self.line("}");
    }

    /// Statements, at most `n`, at nesting `depth`.
    fn statements(&mut self, n: u8, depth: u8) {
        for _ in 0..n {
            let choice = if depth == 0 { 0 } else { self.pick(10) };
            match choice {
                0..=3 => {
                    let v = self.pick(VARS as u8);
                    let x = self.expr(2);
                    self.line(&format!("v{v} = %{x}"));
                }
                4 | 5 => {
                    let (a, b) = (self.pick(3), self.pick(3));
                    self.branch(
                        |g| {
                            g.statements(a, depth - 1);
                            g.line("yield");
                        },
                        |g| {
                            g.statements(b, depth - 1);
                            g.line("yield");
                        },
                    );
                }
                6 => self.branch(|g| g.line("kill"), |g| g.line("yield")),
                7 => self.branch(
                    |g| {
                        g.outputs();
                        g.line("return");
                    },
                    |g| g.line("yield"),
                ),
                8 if !self.loops.is_empty() => {
                    let label = *self.loops.last().unwrap();
                    self.branch(|g| g.line(&format!("br '{label}")), |g| g.line("yield"));
                }
                _ => self.counted_loop(depth),
            }
        }
    }

    /// A loop of a counter from zero below a bound of one to four, or of a constant word's low two bits.
    fn counted_loop(&mut self, depth: u8) {
        let label = self.labels;
        self.labels += 1;
        let zero = self.value("u32", "u32(0)");
        let counter = self.next;
        self.next += 1;
        self.line(&format!("loop '{label} (%{counter}: u32) = (%{zero}) {{"));
        self.indent += 1;
        self.loops.push(label);
        let n = self.pick(3) + 1;
        self.statements(n, depth - 1);
        self.loops.pop();
        let one = self.value("u32", "u32(1)");
        let next = self.value("u32", &format!("iadd %{counter}, %{one}"));
        let bound = if self.pick(2) == 0 {
            let b = self.pick(4) + 1;
            self.value("u32", &format!("u32({b})"))
        } else {
            let word = u32::from(self.pick(WORDS as u8));
            let row = self.value("vec4<u32>", &format!("buffer r0 + {} oob.zero", 16 * (word / 4)));
            let lane = ["x", "y", "z", "w"][(word % 4) as usize];
            let w = self.value("u32", &format!("%{row}.{lane}"));
            let three = self.value("u32", "u32(3)");
            self.value("u32", &format!("and %{w}, %{three}"))
        };
        let more = self.value("bool", &format!("ucmp.lt %{next}, %{bound}"));
        self.line(&format!("if %{more} {{"));
        self.line(&format!("  continue '{label} %{next}"));
        self.line("} else {");
        self.line("  yield");
        self.line("}");
        self.line("yield");
        self.indent -= 1;
        self.line("}");
    }
}

/// The program a byte string draws.
fn text(bytes: &[u8]) -> String {
    let mut g = Gen {
        bytes,
        at: 0,
        next: 0,
        text: String::new(),
        indent: 1,
        labels: 0,
        loops: Vec::new(),
    };
    for v in 0..VARS {
        g.line(&format!("var v{v}: f32"));
    }
    for v in 0..VARS {
        let x = g.expr(0);
        g.line(&format!("v{v} = %{x}"));
    }
    let n = g.pick(5) + 1;
    g.statements(n, 3);
    g.outputs();
    g.line("return");
    let mut out = String::from("iridia-ir 1\nfn main {\n");
    out.push_str(&g.text);
    out.push_str("}\n");
    out
}

fn varying(name: &str, system: Option<SystemValue>) -> Varying {
    Varying {
        label: Label {
            name: name.into(),
            index: 0,
        },
        system,
        value: Value {
            scalar: ScalarType::Float,
            width: 4,
        },
        interp: None,
        used: true,
        provenance: Default::default(),
    }
}

/// A pixel program of the text, with a `TEXCOORD` input, a `SV_Target` output and a constant buffer `b0`.
fn program(text: &str) -> Program {
    let entry = iridia_ir::text::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    let p = Program {
        stage: Stage::Pixel,
        entry,
        patch: None,
        workgroup: None,
        interface: Interface {
            stage: Stage::Pixel,
            inputs: vec![varying("TEXCOORD", None)],
            outputs: vec![varying("SV_Target", Some(SystemValue::Target(0)))],
            resources: vec![Resource {
                id: ResourceId(0),
                kind: ResourceKind::ConstantBuffer {
                    layout: BufferLayout {
                        size: WORDS * 4,
                        constants: Vec::new(),
                    },
                },
                count: 1,
                provenance: ResourceProvenance {
                    d3d: Some(D3dBinding {
                        class: D3dClass::ConstantBuffer,
                        register: 0,
                        space: 0,
                    }),
                    ..Default::default()
                },
            }],
            facts: StageFacts::default(),
        },
        conventions: Conventions::D3D,
        float: FloatEnv::default(),
        provenance: Provenance {
            format: ShaderFormat::D3d9,
            target: "property".into(),
            lifter: "text".into(),
            lifter_version: 1,
            source_hash: None,
            instruction_count: None,
            policies: Vec::new(),
        },
        notes: Vec::new(),
    };
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    p
}

/// Every source a generated program has.
fn sources() -> Vec<perturb::Source> {
    (0..4)
        .map(|l| perturb::Source::Input(0, l))
        .chain((0..WORDS).map(|w| perturb::Source::Word(ResourceId(0), w)))
        .collect()
}

/// Whether `reach` names `source`.
fn reaches(reach: &Reach, source: perturb::Source) -> bool {
    match source {
        perturb::Source::Input(input, lane) => reach.inputs.contains_key(&Lane { input, lane }),
        perturb::Source::Word(resource, word) => reach
            .words
            .keys()
            .any(|w: &GlobalWord| w.resource == resource && w.word == word),
        perturb::Source::Texel(resource, lane) => reach
            .texels
            .keys()
            .any(|t| t.resource == resource && t.lane == lane),
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn what_flow_leaves_out_cannot_change_a_lane(bytes in prop::collection::vec(any::<u8>(), 0..160)) {
        let text = text(&bytes);
        let program = program(&text);
        let flow = Flow::of_program(&program).unwrap();
        let mut targets: Vec<(perturb::Target, &Reach)> =
            flow.outputs().iter().map(|(k, r)| (perturb::Target::Output(k.0, k.1), r)).collect();
        targets.push((perturb::Target::Discard, flow.discard_reach()));
        for (target, reach) in targets {
            for source in sources().into_iter().filter(|s| !reaches(reach, *s)) {
                if let perturb::Outcome::Changed(what) = perturb::check(&program, target, source, 3) {
                    prop_assert!(false, "{what}\n{text}");
                }
            }
        }
    }

    #[test]
    fn words_read_hold_every_word_that_reaches(bytes in prop::collection::vec(any::<u8>(), 0..160)) {
        let text = text(&bytes);
        let program = program(&text);
        let values = Values::of_program(&program).unwrap();
        let flow = Flow::of_values(&values);
        let reads = WordsRead::of_values(&values, &program.interface);
        let read: Vec<u32> = reads.words(ResourceId(0)).collect();
        for reach in flow.outputs().values().chain([flow.discard_reach()]) {
            for w in reach.words.keys() {
                prop_assert!(read.contains(&w.word), "word {} reaches but is not read\n{text}", w.word);
            }
        }
    }

    #[test]
    fn the_graph_is_closed(bytes in prop::collection::vec(any::<u8>(), 0..160)) {
        let text = text(&bytes);
        let program = program(&text);
        let values = Values::of_program(&program).unwrap();
        for lane in 0..4 {
            prop_assert!(values.output(0, lane).is_some(), "output lane {lane} is not in the graph\n{text}");
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut todo: Vec<_> = values.roots().collect();
        while let Some(n) = todo.pop() {
            if seen.insert(n) {
                for o in values.operands(n) {
                    prop_assert!(values.get(o).is_some(), "an operand of {n:?} is not a node\n{text}");
                    todo.push(o);
                }
            }
        }
    }
}

/// The input generator distinguishes numeric views while leaving casts and unused results out of its inference.
#[test]
fn input_word_views_follow_reads_instead_of_casts_or_dead_results() {
    let cases = [
        (
            "%2: f32 = bitcast %1\n%3: f32 = convert.undef %1\n%4: f32 = fadd %2, %3\noutput 0.x = %4",
            true,
        ),
        (
            "%2: i32 = bitcast %1\n%3: f32 = convert.undef %2\n%4: u32 = iadd %1, %1\n%5: i32 = bitcast %4\n%6: f32 = convert.undef %5\n%7: f32 = fadd %3, %6\noutput 0.x = %7",
            false,
        ),
        (
            "%2: i32 = bitcast %1\n%3: f32 = convert.undef %2\n%4: f32 = convert.undef %1\n%5: f32 = fadd %3, %4\noutput 0.x = %5",
            true,
        ),
        (
            "%2: i32 = bitcast %1\n%3: i32 = smulhigh %2, %2\n%4: f32 = convert.undef %1\noutput 0.x = %4",
            false,
        ),
    ];
    for (body, ambiguous) in cases {
        let p = program(&format!(
            "iridia-ir 1\nfn main {{\n%0: vec4<u32> = buffer r0 + 0 oob.zero\n%1: u32 = %0.x\n{body}\nreturn\n}}"
        ));
        perturb::take_counts();
        perturb::check(
            &p,
            perturb::Target::Output(0, 0),
            perturb::Source::Word(ResourceId(0), 0),
            1,
        );
        assert_eq!(perturb::take_counts().input_ambiguous, ambiguous, "{body}");
    }
}

/// A program of `body` over `%0` = input 0.x and `%1` = input 0.y (and `%2` = 0.0), whose every output lane is `%1`
/// where it gets to its end.
fn over_x_and_y(body: &str) -> Program {
    let mut t = String::from(
        "iridia-ir 1\nfn main {\n  %0: f32 = input 0.x\n  %1: f32 = input 0.y\n  %2: f32 = f32(0.0)\n",
    );
    t.push_str(body);
    for lane in ["x", "y", "z", "w"] {
        let _ = writeln!(t, "  output 0.{lane} = %1");
    }
    t.push_str("  return\n}\n");
    program(&t)
}

/// A way out decides what it skips, and only that: input 0.x decides a way out of a block, an arm or a case, and
/// decides no discard past that construct's end (what the walk once let it decide: the code after the block, the
/// other arm and the other cases).
#[test]
fn a_way_out_decides_only_what_it_skips() {
    let x = perturb::Source::Input(0, 0);
    let early = "      output 0.x = %2
      output 0.y = %2
      output 0.z = %2
      output 0.w = %2
      return
";
    let cases = [
        // a block left early: its end is reached either way
        (
            "the discard after the block",
            "  block '0 {
    %3: bool = fcmp.lt %0, %2
    if %3 {
      br '0
    } else {
      yield
    }
    yield
  }
  %4: bool = fcmp.lt %1, %2
  if %4 {
    kill
  } else {
    yield
  }
"
            .to_owned(),
            false,
        ),
        // a return in one arm: the other arm's discard is decided by the arm's condition alone
        (
            "the other arm's discard",
            format!(
                "  %3: bool = fcmp.lt %1, %2
  if %3 {{
    %4: bool = fcmp.lt %0, %2
    if %4 {{
{early}    }} else {{
      yield
    }}
    yield
  }} else {{
    kill
  }}
"
            ),
            false,
        ),
        // the same in a case
        (
            "the other case's discard",
            format!(
                "  %3: u32 = bitcast %1
  switch '0 %3 {{
  case 0:
    %4: bool = fcmp.lt %0, %2
    if %4 {{
{early}    }} else {{
      yield
    }}
    yield
  default:
    kill
  }}
"
            ),
            false,
        ),
        // and what a way out skips is decided by it
        (
            "the discard after the return",
            format!(
                "  %3: bool = fcmp.lt %0, %2
  if %3 {{
{early}  }} else {{
    yield
  }}
  %4: bool = fcmp.lt %1, %2
  if %4 {{
    kill
  }} else {{
    yield
  }}
"
            ),
            true,
        ),
    ];
    let mut wrong = Vec::new();
    for (place, body, decided) in &cases {
        let flow = Flow::of_program(&over_x_and_y(body)).unwrap();
        if reaches(flow.discard_reach(), x) != *decided {
            wrong.push(format!("{place}: {:?}", flow.discard_reach()));
        }
    }
    assert!(
        wrong.is_empty(),
        "{}",
        wrong.join(
            "
"
        )
    );
}

#[test]
fn a_drawn_program_parses() {
    // the empty string draws the smallest program; a long one exercises each construct
    for bytes in [
        Vec::new(),
        (0..=255).collect::<Vec<u8>>(),
        (0..400).map(|i: u32| (i * 37 % 251) as u8).collect(),
    ] {
        let t = text(&bytes);
        let p = program(&t);
        let mut out = String::new();
        let _ = write!(out, "{}", iridia_ir::text::print(&p.entry));
        assert!(!out.is_empty());
    }
}
