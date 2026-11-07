# iridia-analysis

Analyses of [iridia's shader IR](../iridia-ir): questions about what a lifted program computes, asked of its code with no game in it.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled shaders into programs other tools can read and write. A reader lifts a program into the IR ([`iridia-dxbc`](../iridia-dxbc), [`iridia-dxil`](../iridia-dxil), [`iridia-d3d9-ir`](../iridia-d3d9-ir)), and this crate answers questions about it: what each output is made of, which constants it reads. This crate is `no_std` with `alloc`, has no `unsafe` and does no I/O.

## Analyses

- `flow::Values`: the program as a value graph, lane by lane: each output lane, each discard's conditions and each write as a graph of input lanes, texels, buffer words and the operations between them. Variables are followed through their loads and stores, branches meet in merges decided by their conditions, and loop headers carry values through one walk of each body. Headers that carry one unchanged value are removed afterwards.
- `flow::Flow`: what may reach each output lane and the discard, and how: through its value, only as what decides which texel is read, or only through a condition (`flow::Via`).
- `flow::VertexDemand`: the vertex input lanes a vertex and pixel program pair needs, the pixel program's reads followed back through the vertex outputs of the same label.
- `output_uses`: which output lanes a discard tests or a stated blend uses as its weight; `discard_compares` and `alpha_tests` retain structural evidence for cutoff references and dither. `AlphaExpr`/`Known`/`Linear` solve supported expressions with at most one varying sample.
- `matrix::lines` and `matrix::lines_at`: whether constant-buffer lines are rows or columns of the matrix applied to column vectors. `lines_at` takes `Result<&ProgramGraph, &GraphError>` and finds the buffer by its D3D binding, including a buffer whose sole array is a palette.
- `reads::WordsRead`: the words of each constant buffer a program's results are made of, and which buffers it reads at an index it computes. `reads::ProgramGraph` is the value graph with its interface and its reads.

## Example

```rust
use iridia_analysis::flow::{Flow, Lane, Via};
use iridia_ir::{
    Conventions, FloatEnv, Program, Provenance,
    interface::{Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value, Varying},
};

let entry = iridia_ir::text::parse(
    "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f32 = input 0.y
  %2: f32 = f32(0.0)
  %3: bool = fcmp.lt %0, %2
  if %3 {
    yield %1
  } else {
    yield %2
  } -> %4: f32
  output 0.x = %4
  return
}
",
)
.unwrap();
let varying = |name: &str, system, width| Varying {
    label: Label { name: name.into(), index: 0 },
    system,
    value: Value { scalar: ScalarType::Float, width },
    interp: None,
    used: true,
    provenance: Default::default(),
};
let program = Program {
    stage: Stage::Pixel,
    entry,
    patch: None,
    workgroup: None,
    interface: Interface {
        stage: Stage::Pixel,
        inputs: vec![varying("TEXCOORD", None, 2)],
        outputs: vec![varying("SV_Target", Some(SystemValue::Target(0)), 1)],
        resources: Vec::new(),
        facts: StageFacts::default(),
    },
    conventions: Conventions::D3D,
    float: FloatEnv::default(),
    provenance: Provenance {
        format: ShaderFormat::D3d9,
        target: "example".into(),
        lifter: "text".into(),
        lifter_version: 1,
        source_hash: None,
        instruction_count: None,
        policies: Vec::new(),
    },
    notes: Vec::new(),
};

let flow = Flow::of_program(&program).unwrap();
let target = Label { name: "SV_Target".into(), index: 0 };
// y is the value the output may take; x only decides whether it does
assert_eq!(
    flow.reaching(&program.interface, &target, 0),
    [(Lane { input: 0, lane: 0 }, Via::Control), (Lane { input: 0, lane: 1 }, Via::Value)]
);
```

## Guarantees and limits

- An analysis reads a program as a lifter gives it: a body that is a control flow graph is structured first (`iridia_passes::structurize`, on a copy), and a program need not be promoted.
- What an analysis cannot follow is an error that says so, never an answer that leaves it out: a geometry program's emitted vertices and a ray tracing program's payload are refused.
- `Flow` is a may-reach over-approximation: a lane it names may not change the output at run time (a value multiplied by zero), but a lane it leaves out cannot change it. The property tests check that last part by running generated programs in [`iridia-exec-ir`](../iridia-exec-ir).
- A hull program's patch constant function reads the control points' outputs as one point: which point it reads is not followed.

## Tests

`cargo test -p iridia-analysis` runs the unit and property tests with no game files. The oracle tests compare each analysis with the one of the `iridia` crate on every program both read: the DXBC and DXIL fixtures, the slim corpus and Elden Ring's programs, which [`primase-testdata`](https://github.com/Kalmarv/primase) finds; with none of them present they skip.

The matrix oracle compares 34 fixture constants and 24 constants in six slim DXBC programs, including integer
matrix products expanded as multiplies and adds or lowered from multiply-add instructions. Integer and floating
arithmetic show the same row/column structure.

## Licence

GPL-3.0-or-later.
