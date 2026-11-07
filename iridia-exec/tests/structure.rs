#![allow(clippy::unwrap_used, clippy::indexing_slicing)]
//! The structurizer's regions, end to end: random control flow graphs are written as a graph body of the IR, structured
//! (`structurize`, then `break_flags` routes the exits innermost-only `break` and `continue` cannot take), validated, and
//! run by the executor. Every node adds to an accumulator, so the output is a hash of the nodes and edges visited in
//! order; each deciding node picks its way from a hash of its visit count. The output must be the one a walk of the
//! graph itself gives, including for graphs with loops, switches and branches that leave several constructs at once,
//! and the reference interpreter's run of the graph before it is structured must give it too.
mod support;

use iridia_exec::{PixelShader, ProgramShader};
use iridia_ir::{
    Binary, Block, Body, Cmp, Function, Node, OnZero, Op, Precision, Program, Target, Terminator, Type,
    ValueDef, ValueId, Var, VarId,
    interface::{Interface, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue},
};

const LIMIT: u32 = 64;

/// A control flow graph: node `n` ends in `terms[n]`; node 0 is the entry.
#[derive(Debug)]
struct Cfg {
    terms: Vec<Term>,
}

/// How a node ends.
#[derive(Debug)]
enum Term {
    /// The graph ends.
    Exit,
    /// On to a node.
    Goto(usize),
    /// On to the first node or the second, as the node decides.
    If(usize, usize),
    /// On to one of the nodes, as the node decides.
    Switch(Vec<usize>),
}

/// The way node `n` goes the `visit`-th time, of `ways`: a 32-bit hash both the program and the walk compute.
fn way(seed: u32, n: usize, visit: u32, ways: u32) -> u32 {
    let mut x = seed.wrapping_mul(0x9e37_79b1)
        ^ (n as u32 + 1).wrapping_mul(0x85eb_ca6b)
        ^ (visit + 1).wrapping_mul(0xc2b2_ae35);
    x ^= x >> 15;
    x = x.wrapping_mul(0x2c1b_3c6d);
    x ^= x >> 12;
    x = x.wrapping_mul(0x297a_2d39);
    x ^= x >> 15;
    x % ways
}

/// The accumulator a walk of the graph ends with.
fn walk(cfg: &Cfg, seed: u32) -> u32 {
    let (mut acc, mut steps, mut node) = (0u32, 0u32, 0usize);
    let mut visits = vec![0u32; cfg.terms.len()];
    loop {
        steps += 1;
        acc = acc.wrapping_mul(31).wrapping_add(node as u32 + 1);
        if steps >= LIMIT {
            return acc;
        }
        let mut decide = |ways: u32| {
            let w = way(seed, node, visits[node], ways);
            visits[node] += 1;
            w as usize
        };
        let next = match &cfg.terms[node] {
            Term::Exit => return acc,
            Term::Goto(t) => *t,
            Term::If(a, b) => [*a, *b][decide(2)],
            Term::Switch(ts) => ts[decide(ts.len() as u32)],
        };
        acc = acc.wrapping_mul(31).wrapping_add(200 + (node * 16 + next) as u32);
        node = next;
    }
}

/// The graph body being written: its values and variables, and the nodes of the block being filled.
struct Toy {
    f: Function,
    acc: VarId,
    steps: VarId,
    visits: Vec<VarId>,
}

impl Toy {
    fn value(&mut self, nodes: &mut Vec<Node>, op: Op) -> ValueId {
        let ty = match op {
            Op::Binary(Binary::UCmp(_), ..) => Type::BOOL,
            _ => Type::U32,
        };
        let id = ValueId(u32::try_from(self.f.values.len()).unwrap());
        self.f.values.push(ValueDef {
            ty,
            op,
            origin: None,
            precision: Precision::Full,
        });
        nodes.push(Node::Value(id));
        id
    }
    fn lit(&mut self, nodes: &mut Vec<Node>, x: u32) -> ValueId {
        self.value(nodes, Op::Literal([u64::from(x), 0, 0, 0]))
    }
    fn binary(&mut self, nodes: &mut Vec<Node>, op: Binary, a: ValueId, b: ValueId) -> ValueId {
        self.value(nodes, Op::Binary(op, a, b))
    }
    /// `acc = acc * 31 + word`.
    fn accumulate(&mut self, nodes: &mut Vec<Node>, word: u32) {
        let acc = self.value(nodes, Op::LoadVar(self.acc));
        let k = self.lit(nodes, 31);
        let times = self.binary(nodes, Binary::IMul, acc, k);
        let w = self.lit(nodes, word);
        let sum = self.binary(nodes, Binary::IAdd, times, w);
        nodes.push(Node::StoreVar(self.acc, sum));
    }
    /// The hash of node `n`'s visit, advancing its count.
    fn pick(&mut self, nodes: &mut Vec<Node>, seed: u32, n: usize, ways: u32) -> ValueId {
        let visit = self.value(nodes, Op::LoadVar(self.visits[n]));
        let one = self.lit(nodes, 1);
        let next = self.binary(nodes, Binary::IAdd, visit, one);
        nodes.push(Node::StoreVar(self.visits[n], next));
        let s = self.lit(nodes, seed);
        let k1 = self.lit(nodes, 0x9e37_79b1);
        let a = self.binary(nodes, Binary::IMul, s, k1);
        let n1 = self.lit(nodes, n as u32 + 1);
        let k2 = self.lit(nodes, 0x85eb_ca6b);
        let c = self.binary(nodes, Binary::IMul, n1, k2);
        let k3 = self.lit(nodes, 0xc2b2_ae35);
        let d = self.binary(nodes, Binary::IMul, next, k3);
        let ac = self.binary(nodes, Binary::Xor, a, c);
        let mut x = self.binary(nodes, Binary::Xor, ac, d);
        for (shift, mul) in [(15, 0x2c1b_3c6d_u32), (12, 0x297a_2d39), (15, 1)] {
            let s = self.lit(nodes, shift);
            let shifted = self.binary(nodes, Binary::ShrU, x, s);
            x = self.binary(nodes, Binary::Xor, x, shifted);
            if mul != 1 {
                let m = self.lit(nodes, mul);
                x = self.binary(nodes, Binary::IMul, x, m);
            }
        }
        let w = self.lit(nodes, ways);
        self.binary(nodes, Binary::URem(OnZero::ByOne), x, w)
    }
}

fn to(block: usize) -> Target {
    Target {
        block: u32::try_from(block).unwrap(),
        args: Vec::new(),
    }
}

/// The program of `cfg` for `seed`, a graph body: a start block, then for each node `n` a block that counts the step,
/// adds the node to the accumulator and leaves for the exit at the step limit, a block that decides the way, and a block
/// per way that adds the edge; and the exit, which outputs the accumulator.
fn program(cfg: &Cfg, seed: u32) -> Program {
    let n = cfg.terms.len();
    let u32_var = |name: String| Var::new(&name, Type::U32);
    let mut vars = vec![u32_var("acc".into()), u32_var("steps".into())];
    vars.extend((0..n).map(|i| u32_var(format!("v{i}"))));
    let mut toy = Toy {
        f: Function {
            queries: vec![],
            name: "main".into(),
            values: Vec::new(),
            vars,
            body: Body::Cfg(Vec::new()),
        },
        acc: VarId(0),
        steps: VarId(1),
        visits: (0..n).map(|i| VarId(u32::try_from(i).unwrap() + 2)).collect(),
    };
    // the blocks: start 0; node i's step block, decision block and edge blocks; the exit last
    let ways = |t: &Term| match t {
        Term::Exit => 0,
        Term::Goto(_) => 1,
        Term::If(..) => 2,
        Term::Switch(ts) => ts.len(),
    };
    let mut first = Vec::with_capacity(n);
    let mut next = 1;
    for t in &cfg.terms {
        first.push(next);
        next += 2 + ways(t);
    }
    let exit = next;
    let mut blocks = Vec::new();
    let mut start = Vec::new();
    let zero = toy.lit(&mut start, 0);
    for v in [toy.acc, toy.steps].into_iter().chain(toy.visits.clone()) {
        start.push(Node::StoreVar(v, zero));
    }
    blocks.push(Block {
        params: Vec::new(),
        nodes: start,
        term: Terminator::Jump(to(first[0])),
    });
    for (i, t) in cfg.terms.iter().enumerate() {
        let (decision, edges) = (first[i] + 1, first[i] + 2);
        // the step
        let mut nodes = Vec::new();
        let steps = toy.value(&mut nodes, Op::LoadVar(toy.steps));
        let one = toy.lit(&mut nodes, 1);
        let steps = toy.binary(&mut nodes, Binary::IAdd, steps, one);
        nodes.push(Node::StoreVar(toy.steps, steps));
        toy.accumulate(&mut nodes, i as u32 + 1);
        let limit = toy.lit(&mut nodes, LIMIT);
        let over = toy.binary(&mut nodes, Binary::UCmp(Cmp::Ge), steps, limit);
        blocks.push(Block {
            params: Vec::new(),
            nodes,
            term: Terminator::Branch {
                cond: over,
                accept: to(exit),
                reject: to(decision),
            },
        });
        // the decision
        let mut nodes = Vec::new();
        let term = match t {
            Term::Exit => Terminator::Jump(to(exit)),
            Term::Goto(_) => Terminator::Jump(to(edges)),
            Term::If(..) => {
                let w = toy.pick(&mut nodes, seed, i, 2);
                let zero = toy.lit(&mut nodes, 0);
                let first_way = toy.binary(&mut nodes, Binary::UCmp(Cmp::Eq), w, zero);
                Terminator::Branch {
                    cond: first_way,
                    accept: to(edges),
                    reject: to(edges + 1),
                }
            }
            Term::Switch(ts) => {
                let w = toy.pick(&mut nodes, seed, i, ts.len() as u32);
                Terminator::Switch {
                    selector: w,
                    cases: (0..ts.len() - 1).map(|k| (k as u32, to(edges + k))).collect(),
                    default: to(edges + ts.len() - 1),
                }
            }
        };
        blocks.push(Block {
            params: Vec::new(),
            nodes,
            term,
        });
        // the edges
        let targets = match t {
            Term::Exit => Vec::new(),
            Term::Goto(a) => vec![*a],
            Term::If(a, b) => vec![*a, *b],
            Term::Switch(ts) => ts.clone(),
        };
        for target in targets {
            let mut nodes = Vec::new();
            toy.accumulate(&mut nodes, 200 + (i * 16 + target) as u32);
            blocks.push(Block {
                params: Vec::new(),
                nodes,
                term: Terminator::Jump(to(first[target])),
            });
        }
    }
    let mut nodes = Vec::new();
    let acc = toy.value(&mut nodes, Op::LoadVar(toy.acc));
    nodes.push(Node::StoreOutput(0, 0b1, acc));
    blocks.push(Block {
        params: Vec::new(),
        nodes,
        term: Terminator::Return,
    });
    assert_eq!(blocks.len(), exit + 1);
    toy.f.body = Body::Cfg(blocks);
    let interface = Interface {
        stage: Stage::Pixel,
        inputs: Vec::new(),
        outputs: vec![support::varying(
            "SV_Target",
            0,
            ScalarType::Uint,
            1,
            Some((0, 0b1)),
            Some(SystemValue::Target(0)),
        )],
        resources: Vec::new(),
        facts: StageFacts::default(),
    };
    support::program(toy.f, interface, ShaderFormat::Dxbc, "toy")
}

/// The graph's program structured as the executor takes it, with the count of exits `break_flags` routed; `None` where
/// the graph is irreducible, which `structurize` refuses.
fn structured(p: &Program) -> Option<(Program, usize)> {
    let mut s = p.clone();
    if let Err(e) = iridia_passes::structurize(&mut s.entry) {
        assert!(e.contains("irreducible"), "{e}");
        return None;
    }
    let routed = iridia_passes::break_flags(&mut s.entry).unwrap();
    iridia_ir::validate(&s).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&s.entry)));
    Some((s, routed))
}

/// The accumulator a run of the structured program ends with in the executor; the reference interpreter's run of the
/// graph and of its structured form must give the same.
fn run(graph: &Program, structured: Program) -> u32 {
    let reference = |p: &Program| {
        let out = support::reference(p, &[], &[], &mut |_| Err("no samples".into()));
        out.outputs[0].unwrap()[0]
    };
    let (of_graph, of_regions) = (reference(graph), reference(&structured));
    let shader = ProgramShader::from_ir(structured).unwrap();
    let execution = shader
        .execute(&[], &[], &mut |_: iridia_exec::Sample| Ok([0.0; 4]))
        .unwrap();
    let acc = execution.outputs[0][0];
    assert_eq!((of_graph, of_regions), (acc, acc), "the reference interpreter");
    acc
}

#[test]
fn lowered_graphs_run_as_their_walk() {
    use Term::*;
    let shapes = vec![
        Cfg {
            terms: vec![If(1, 3), If(2, 3), Goto(4), Goto(4), Exit],
        },
        Cfg {
            terms: vec![Goto(1), If(2, 6), If(3, 4), Goto(1), If(5, 1), Goto(6), Exit],
        },
        // nested loops where the inner one breaks out of both
        Cfg {
            terms: vec![
                Goto(1),
                If(2, 7),
                Goto(3),
                If(4, 6),
                If(5, 7),
                Goto(3),
                Goto(1),
                Exit,
            ],
        },
        // a switch in a loop: one case continues the loop, one leaves it, one joins
        Cfg {
            terms: vec![
                Goto(1),
                Switch(vec![2, 3, 4]),
                Goto(1),
                Goto(5),
                Goto(5),
                If(1, 6),
                Exit,
            ],
        },
        Cfg {
            terms: vec![Switch(vec![1, 2, 2, 3]), Goto(4), Goto(4), Goto(4), Exit],
        },
    ];
    for cfg in &shapes {
        for seed in 0..8 {
            let p = program(cfg, seed);
            let (s, _) = structured(&p).unwrap_or_else(|| panic!("{cfg:?} is irreducible"));
            assert_eq!(run(&p, s), walk(cfg, seed), "{cfg:?} seed {seed}");
        }
    }
}

#[test]
fn random_lowered_graphs_run_as_their_walk() {
    let mut state = 0x0fed_cba9_8765_4321u64;
    let mut next = move |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    let (mut ran, mut routed) = (0, 0);
    for _ in 0..1500 {
        let n = 2 + next(8);
        let terms: Vec<Term> = (0..n)
            .map(|_| match next(10) {
                0 | 1 => Term::Exit,
                2..=4 => Term::Goto(next(n)),
                5..=8 => Term::If(next(n), next(n)),
                _ => Term::Switch((0..2 + next(3)).map(|_| next(n)).collect()),
            })
            .collect();
        let cfg = Cfg { terms };
        for seed in 0..4 {
            let p = program(&cfg, seed);
            let Some((s, flags)) = structured(&p) else { break };
            routed += usize::from(flags > 0);
            assert_eq!(run(&p, s), walk(&cfg, seed), "{cfg:?} seed {seed}");
            ran += 1;
        }
    }
    assert!(
        ran > 1000 && routed > 100,
        "{ran} runs, {routed} with a routed branch"
    );
}
