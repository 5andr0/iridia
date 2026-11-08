# iridia-naga

Lowers a program of [iridia's shader IR](../iridia-ir) to a stock [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) module, validated, for naga's writers (WGSL).

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled shaders into programs other tools can read and write. Readers lift into the IR, [`iridia-passes`](../iridia-passes) puts a program in the form naga takes, and this crate lowers it. The design is [`docs/design/ir.md`](../docs/design/ir.md). This crate is `no_std` with `alloc` and does no I/O; it depends on naga as published, with no fork.

## What it does

- `module(program, plan)` builds the module of a vertex, pixel or compute program: the entry point `main` with an argument per interface input (a compute program's are its invocation ids, built-ins) and a member of the result structure `Out` per output (`entry_io` states them; a vertex program's clip distances are one `clip_distances` array), the thread group size of a compute program, and each resource a global where the `BindingPlan` binds it. `BindingPlan::d3d` puts Direct3D's register classes in bind groups 0 to 3 (`cbN`, `tN`, `sN`, `uN`), each register its binding; a register of another space than 0 binds at `(space << 16) | register` (`uN_spaceS`), and a structured view's hidden counter in group 4 at the view's binding (`uN_counter`). `BindingPlan::webgpu` keeps inside WebGPU's default limits (four bind groups, bindings below 1000): a register of space 0 at its number, the registers of other spaces after the largest binding of their group, the hidden counters in group 3 after its views; `Profile::BROWSER` binds with it and `Profile::NAGA` with `BindingPlan::d3d` (`BindingPlan::for_profile`).
- Memory is bound as the older lifts bind it: a constant buffer is rows of `vec4<u32>`; a raw or structured buffer a runtime array of 32-bit words (atomic words where an atomic reaches it); a typed buffer an array of `vec4<u32>`, each element four 32-bit lanes the host wrote in the view's type (structures of four atomic words where an atomic reaches it); a storage image `rgba32float`, `rgba32uint` or `rgba32sint` by its texels' type. Every read and write of a buffer or image is checked against its size, as the IR states: outside, a read gives zero and a write does nothing, where WGSL leaves it to the implementation.
- Workgroup variables are `var<workgroup>` (zero at the start, as the IR's); an atomic of the IR is naga's where it has one, a compare-exchange is a loop of naga's weak one until it exchanges or the word differs, and a minimum or maximum of the other signedness than its memory's words is a compare-exchange loop. WGSL's barriers are execution barriers of the workgroup: a memory barrier without execution is lowered as one with it, and a device-wide one as the workgroup's.
- Operations naga has whole map to one expression; the others expand in the lowering, each in one function: a zero-wins multiply, `min` and `max` with their NaN rule, `saturate` of NaN, a float-to-integer conversion that saturates, rounding with ties away from zero, a constant buffer row read outside the buffer.
- A masked input load is a swizzle of the argument; a masked output store is composed with what the output held (zero before the first store).
- `validate(module, profile)` runs naga's validator with the profile's capabilities; `wgsl(program, profile)` (the `wgsl` feature, on by default) preflights the program, lowers it with the D3D plan, validates and writes the WGSL.
- `wgsl_for(program, profile, &Pipeline { plan, feeds })` writes it for one pipeline: the resources where `plan` binds them (two programs of a pipeline share its bind groups), and a vertex program's outputs where the pixel program it `feeds` reads them, matched by semantic as Direct3D links stages. An output read with more lanes is padded with zeros, with fewer its first lanes, as another 32-bit scalar the same bits; each is noted in `Written::notes`, and an input no output carries is refused by name.

## Example

```rust
use iridia_ir::{
    Conventions, FloatEnv, Program, Provenance,
    interface::{Interface, Label, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue, Value, Varying},
};

let entry = iridia_ir::text::parse(
    "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: vec4<f32> = compose %0, %0, %0, %0
  %2: vec4<f32> = saturate.zero %1
  output 0.xyzw = %2
  return
}
",
)
.unwrap();
let varying = |name: &str, width, system| Varying {
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
        inputs: vec![varying("TEXCOORD", 1, None)],
        outputs: vec![varying("SV_Target", 4, Some(SystemValue::Target(0)))],
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
let wgsl = iridia_naga::wgsl(&program, &iridia_naga::Profile::NAGA).unwrap();
assert!(wgsl.contains("@fragment"));
// WGSL leaves what a clamp makes of NaN to the implementation: the NaN-to-zero rule is spelled out
assert!(wgsl.contains("select("), "{wgsl}");
```

## Guarantees and limits

- Standard/browser WGSL filters `derivative_uniformity`, matching the IR's permission to evaluate derivatives in
  divergent control flow. When a program uses subgroups, the `subgroup_uniformity` diagnostic is filtered only if
  `Profile::language_features.subgroup_uniformity` is set. These language features are separate from device
  capabilities: pass the consumer's `GPU.wgslLanguageFeatures` support through `with_language_features`, or use
  `with_subgroup_uniformity`. Built-in profiles start with no optional language features. Naga-dialect text keeps
  its existing directives. Naga 30 accepts both diagnostic filters; its parser still requires the existing bridge
  for the standard `enable subgroups;` directive.

- A program is lowered for a target `Profile` (`Profile::BROWSER` for WebGPU's core WGSL, `Profile::NAGA` for naga's own dialect, either widened with `Profile::with`): `preflight` refuses a program that needs a capability the profile lacks, naming the value, varying or resource that needs it, and naga's validator runs with the profile's capabilities alone. A module naga rejects after the preflight accepted is reported as a lowering bug. The profile also states the WGSL's `enable` directives (`Profile::directives`): `BROWSER` writes `enable subgroups;` before a program that uses a subgroup operation, as a browser requires; `NAGA` leaves it out, as naga's WGSL reader refuses it.
- It lowers the vertex, pixel and compute programs the Direct3D 9 and DXBC readers make, and refuses the rest by name: hull, domain and geometry programs (WGSL has no such stage), a vertex program that feeds a tessellator, an element of an array of resources, an atomic of a storage image (naga's has no result), a multisampled array texture, a sample with a level clamp, a gather at a computed offset, a comparison sample at a level other than the derivatives' or zero, a sample of a 1D texture at a bias or derivatives, an explicit 1D level with an offset or comparison, and a biased sample of a depth texture or one at a computed level.
- naga's `break` and `continue` reach only the innermost construct: run `iridia_passes::break_flags` first. A `Cfg` body must be structured first (`iridia_passes::structurize`).
- Target limitations are checked before allocating the lowering arena, including active operations, resource and
  variable shapes, and structured exit targets and arities. Unused value-table entries are not placed operations.
  Builders retain defensive checks for malformed IR, generated tables, and handle or scope invariants.
- Pipeline state is not lowered: `iridia_passes::specialise_state` it first.
- WGSL text cannot say a varying has no interpolation, and naga's namer adds a `_` to some names; a reader of the WGSL sees those, not the IR's.

## Tests

`cargo test -p iridia-naga` needs no game files: `tests/lower.rs` pins the WGSL of masked loads and stores and checks that unspecialised state is refused, and `tests/compute.rs` lowers compute programs and runs what naga reads back of their WGSL. The Direct3D 9 reader's tests lower every Shaun White Skateboarding program they read, and `iridia-dxbc/tests/ir_gate.rs` runs every lowered program of the Sekiro, Riders Republic and Unity corpora (column C) against the older lift.

## Licence

GPL-3.0-or-later.
