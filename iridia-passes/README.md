# iridia-passes

Passes over [iridia's shader IR](../iridia-ir): each takes a function and gives one that means the same, in a form closer to what an output needs.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled shaders into programs other tools can read and write. A reader lifts a program into the IR, these passes rewrite it, and a writer ([`iridia-naga`](../iridia-naga)) lowers it. The design is [`docs/design/ir.md`](../docs/design/ir.md). This crate is `no_std` with `alloc` and does no I/O.

## Passes

- `promote`: variables (a source's registers) to SSA values, carried out of `If` and `Block` constructs as results and through loops as parameters. A variable a load reads before any store is undefined there, as the IR says.
- `simplify`: lane bookkeeping folded (a swizzle of a swizzle is one swizzle, the compose of every lane of a value in order is that value, the compose of some of its lanes a swizzle) and unused values, construct results and loop parameters removed together with their transfer arguments. A worklist follows dependencies from effects and control conditions, so dead loop-carried cycles disappear without repeated scans of the function. Atomics whose old value is unused remain effects.
- `specialise_state`: pipeline state the caller knows (`ResourceKind::PipelineState`, such as D3D9's projected-texture flag) made literals, the branches they decide folded and the resources taken out of the interface. A state value that does not fit its parameter is refused.
- `specialise`: a material copy with fully supplied feature decisions folded by `iridia_ir::eval`, the reference executor's numeric definition (one portable `libm` backend, subnormals kept where unspecified). `feature_words` selects only words that decide a branch or select without runtime inputs; colours and scales stay uniforms. Returns `Result<Specialised, SpecialiseError>`, including the selected words, removed resources and fold count. A decision the reference cannot evaluate is a typed refusal.
- `structurize`: a `Cfg` body made `Regions`, by Ramsey's structuring of reducible control flow (`structure`, which works on any graph of nodes and edges); an irreducible graph is refused.
- `break_flags`: an exit to a construct that is not the innermost one, which targets whose `break` and `continue` are innermost-only (naga, WGSL) cannot take, routed through a `jump` variable: the inner constructs break, and a guard after each one passes the exit on. Dead transfer slots are removed before demotion so they need no routing copies; promotion and simplification remove dead slots afterwards too. It returns how many exits it routed.

## Example

```rust
let text = "iridia-ir 1
fn main {
  var r: f32
  %0: f32 = input 0.x
  r = %0
  %1: f32 = f32(0.0)
  %2: bool = fcmp.lt %0, %1
  if %2 {
    %3: f32 = fneg %0
    r = %3
    yield
  } else {
    yield
  }
  %4: f32 = load r
  output 0.x = %4
  return
}
";
let mut f = iridia_ir::text::parse(text).unwrap();
iridia_passes::promote(&mut f).unwrap();
iridia_passes::simplify(&mut f);
assert!(f.vars.is_empty());
// r leaves the if as its result: the negation in one arm, the input in the other
assert_eq!(
    iridia_ir::text::print(&f),
    "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f32 = f32(0.0)
  %2: bool = fcmp.lt %0, %1
  if %2 {
    %3: f32 = fneg %0
    yield %3
  } else {
    yield %0
  } -> %5: f32
  output 0.x = %5
  return
}
"
);
```

## Guarantees and limits

- Each pass keeps the meaning of every program the validator accepts, by the IR's definitions; the tests check that by running programs before and after in [`iridia-exec-ir`](../iridia-exec-ir), on hand-worked fixtures, on 1,500 generated control flow graphs and on the Direct3D 9 programs of the migration gate.
- `structurize` does not split nodes: an irreducible graph is refused, not made reducible.
- `specialise_state` folds state only where the caller states its value; a program that still reads state is for a target that binds it.
- Material selection follows the same source lane through every exit of an `If`, `Block` or `Switch`, even where other result lanes remain dynamic. Loop-carried values are never assumed constant from their initial values.

## Tests

`cargo test -p iridia-passes` needs no game files. `tests/fixtures.rs` checks each pass against the text it should give and by running it; `tests/structurize.rs` has the design's counterexample (a loop entered at two points unless the state is known), a nested loop whose exit needs routing, and the generated graphs.

## Licence

GPL-3.0-or-later.
