# iridia-exec-ir

Runs a program of [iridia's shader IR](../iridia-ir) once, on the CPU, by the IR's own definition of each operation: the reference the passes and lowerings are checked against.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled shaders into programs other tools can read and write. The design is [`docs/design/ir.md`](../docs/design/ir.md). This crate does no I/O.

## What it runs

- `run(program, bound)` takes the program and a `Bound`: each input slot's lanes, each constant buffer's rows, each pipeline state resource's value, and a sampler the caller answers each `Sample` with (the texture and sampler ids, the coordinate's lanes, the level of detail). It gives a `Run`: each output slot's lanes as last stored (a lane never stored is zero, a slot never stored `None`) and whether the invocation was killed.
- Both body forms: every construct of a `Regions` body, and the blocks of a `Cfg` one (block parameters copied in parallel). A run of more than `STEP_LIMIT` loop iterations or block executions is refused.
- The arithmetic is the reference `iridia-exec` uses: IEEE 754 `f32` for the basic operations, the transcendental functions (`exp2`, `log2`, `sin`, `cos`, `sqrt`, `inverseSqrt`, `pow`) in `f64` rounded once, a dot product summed left to right with each step rounded. Denormals are kept. Each operation's enum decides the rest: a zero-wins multiply, a NaN rule of `min` and `max`, ties away from zero.

## Example

```rust
use iridia_ir::{
    Conventions, FloatEnv, Program, Provenance,
    interface::{Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value, Varying},
};

let entry = iridia_ir::text::parse(
    "iridia-ir 1
fn main {
  %0: vec2<f32> = input 0.xy
  %1: vec2<f32> = vec2<f32>(0x7f800000, 3.0)
  %2: vec2<f32> = fmul.zerowins %0, %1
  output 0.xy = %2
  return
}
",
)
.unwrap();
let varying = |name: &str, system| Varying {
    label: Label { name: name.into(), index: 0 },
    system,
    value: Value { scalar: ScalarType::Float, width: 2 },
    interp: None,
    used: true,
    provenance: Default::default(),
};
let program = Program {
    stage: Stage::Pixel,
    entry,
    interface: Interface {
        stage: Stage::Pixel,
        inputs: vec![varying("TEXCOORD", None)],
        outputs: vec![varying("SV_Target", Some(SystemValue::Target(0)))],
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
iridia_ir::validate(&program).unwrap();

let mut sampler = |_| Err::<[f32; 4], String>("no textures".into());
let inputs = [[0f32, 2.0, 0.0, 0.0].map(f32::to_bits)];
let mut bound = iridia_exec_ir::Bound { inputs: &inputs, buffers: &[], state: &[], sampler: &mut sampler };
let run = iridia_exec_ir::run(&program, &mut bound).unwrap();
// zero wins: 0 * inf is +0, not NaN; the lanes past the output's width are zero
assert_eq!(run.outputs[0], Some([0, 6f32.to_bits(), 0, 0]));
assert!(!run.killed);
```

## Guarantees and limits

- This subset runs 32-bit scalars and vectors, constant buffer reads, pipeline state, samples and kill; anything else (memory other than variables and constant buffers, derivatives, other widths) is refused by name, not approximated.
- One invocation: an implicit level of detail is `Lod::Auto`, for the caller's sampler to decide, since a single invocation has no derivatives.

## Tests

`cargo test -p iridia-exec-ir` needs no game files: `tests/fixtures.rs` runs programs in IR text on inputs whose results are worked by hand from the operations' definitions (the NaN and rounding rules, zero-wins multiplication, saturating conversion, lane masks, a loop, a kill) and checks the text parser's refusals.

## Licence

GPL-3.0-or-later.
