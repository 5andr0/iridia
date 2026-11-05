# iridia-ir

iridia's shader IR: typed SSA values and operations, in a function body that is either a tree of structured regions or a control flow graph with block parameters, beside the interface vocabulary a program exposes (varyings, resources, stage facts, D3D provenance).

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled shaders into programs other tools can read and write. Readers lift into this IR; passes ([`iridia-passes`](../iridia-passes)) rewrite it; writers ([`iridia-naga`](../iridia-naga)) lower it; [`iridia-exec-ir`](../iridia-exec-ir) runs it by its own definitions. The design is [`docs/design/ir.md`](../docs/design/ir.md). This crate is `no_std` with `alloc` and does no I/O.

## What it holds

- A `Program`: its entry `Function`, `Interface`, `Conventions` (clip depth, window and texture origin), `FloatEnv` (denormal controls per float type), `Provenance` (format, target, lifter and version, source hash, the compatibility policies the lifter chose) and notes.
- A `Function` owns its values (`ValueDef`: a `Type`, an `Op`, a source `Origin`, a `Precision` permission such as D3D9's `_pp`) and its variables (source registers until promotion). Its `Body` is `Regions` (`If`, `Block` and `Loop` constructs whose regions end in a yield, a labelled break or continue carrying values, a return, a kill or unreachable) or `Cfg` (blocks with parameters, ending in a jump, a two-way branch, a return, a kill or unreachable).
- Operations whose meaning differs between sources carry the rule as an enum and have no default: `MinMaxNan`, `SaturateNan`, `MulZero` (IEEE or D3D9's zero-wins multiply), `OnZero`, `FloatToInt`, `Rounding` (with ties-away for D3D9 `mova`), `OutOfBounds`, `Level`.
- Interface inputs and outputs are read and written by lane mask (`LoadInput(slot, lanes)`, `StoreOutput(slot, lanes, value)`); a lane never stored is zero. Fixed-function state the code depends on is a declared resource (`ResourceKind::PipelineState`, a `StateParam` such as D3D9's projected-texture flag) read by `LoadState`.
- `validate` checks types, the dominance of every use (regions by scope, graphs by dominator tree; a block nothing reaches is dominated by every block, and its code is still checked), labels and exits, the interface each load and store names, and the stage's rules (a kill and an implicit-level sample are pixel-only). It returns every problem it finds.
- `text::print` writes a function one line per node, and `text::parse` reads it back: printing what it reads gives the same text. The first line names the schema version; the text is for fixtures, diffs and notes, not a storage format.

## Example

```rust
use iridia_ir::{Builder, End, Op, Type};

let mut b = Builder::new("main");
let x = b.value(Type::F32, Op::LoadInput(0, 0b1));
let two = b.f32(2.0);
let y = b.op(Op::Binary(iridia_ir::Binary::FMul(iridia_ir::MulZero::Ieee), x, two)).unwrap();
b.store_output(0, 0b1, y);
let f = b.finish(End::Return).unwrap();

let text = iridia_ir::text::print(&f);
assert_eq!(
    text,
    "iridia-ir 1\nfn main {\n  %0: f32 = input 0.x\n  %1: f32 = f32(2.0)\n  %2: f32 = fmul.ieee %0, %1\n  output 0.x = %2\n  return\n}\n"
);
let back = iridia_ir::text::parse(&text).unwrap();
assert_eq!(iridia_ir::text::print(&back), text);
```

## Guarantees and limits

- A program the validator accepts has one meaning for every input, including NaN, infinities, signed zero and out-of-range values, where its operations' enums say what the source does; the reference for that meaning is `iridia-exec-ir`.
- The text form states everything an operation means but not source origins, and a value the body never places is not written.
- The operation set is what the readers on it need so far (Direct3D 9 shader model 3); it grows with them.

## Licence

GPL-3.0-or-later.
