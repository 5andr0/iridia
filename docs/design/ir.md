# iridia's own shader IR

Status: proposed, for approval before any code moves. Written 2026-11-05 against iridia `ecb915c`, dx `04624ef`.

iridia lifts shaders into naga modules today, built with the fork `primase-naga` (naga 30.0.1 plus three commits:
a WGSL ray-query writer, an optional result on `ImageAtomic`, and `Exchange` on images). This document designs the
representation that replaces it: iridia's own IR, with naga as one output target in its stock form. That decision is
made; this document decides how.

The IR is the hub of the whole shader layer. Sources now: Direct3D 9 token streams, DXBC, DXIL, MojoShader GLSL
(MonoGame MGFX). Later: PS3 RSX, SPIR-V, GLSL, console microcode. Targets: a stock naga module, WGSL, GLSL ES,
SPIR-V, a DXBC encoder, HLSL that recompiles, DXIL later, and `iridia-exec`. Passes between: specialisation, fixed
function state to programs, and the analyses (lane provenance, matrix majorness, output uses).

## Contents

0. Summary of decisions
1. Control flow
2. Types and values
3. Resources and interface
4. Source semantics: operations, contracts, policies (with D3D9 `ps_1_x` worked through)
5. Crates, public API, versioning, consumers
6. The naga back end
7. Validation
8. Verification
9. Migration plan
10. Prior art, as checked
11. Decisions for Caleb

## 0. Summary of decisions

| # | Decision | Rejected alternatives |
|---|----------|-----------------------|
| 1 | One vocabulary of typed SSA values and operations, with two validated control phases: `Cfg` (basic blocks with block parameters) and `Regions` (a tree of block, loop, if and switch with labelled exits that carry values). Structured sources build `Regions` directly; graph sources (DXIL, RSX vertex programs, microcode, SPIR-V in) build `Cfg`. A program may stay in `Cfg` through specialisation and execution; a structured target asks for `Regions`, made once by the one shared structurizer and then frozen. Source structure is semantics (it fixes convergence), never a hint. | SSA CFG only, structure rebuilt per output: every CFG rewrite could move a reconvergence point. Regions only: refuses graphs that specialisation would make structurable (section 1.4). SPIR-V merge annotations on a mutable CFG. |
| 2 | Scalars `bool`, `i/u 8/16/32/64`, `f16/f32/f64`; vectors of 2 to 4; matrices where a source states one. Aggregates live in memory and I/O, not in SSA values. Constants are exact bits. Minimum precision is a typed precision policy on an operation (a permission), never a type. Fixed point is not a type: clamp, quantize, pack and unpack are explicit operations. | A `MinF16` type; native f16 for every min-precision annotation; f64 narrowed to f32; a fixed-point type without evidenced arithmetic. |
| 3 | The IR names resources and interface slots by id. Resource kind, element type and layout are stated once, in the shared vocabulary, with per-source provenance as a list of typed source addresses (D3D register, D3D9 bank, RSX unit, Vulkan set and binding, GL name). Target bindings are an output artifact, the `BindingPlan`. Pipeline state that code depends on is declared: state that changes types (a `ps_1_x` texture's dimension) needs a specialisation or distinct typed resources; numeric state (bump matrices, projection flags) may also lower to a state block. | naga globals plus a bindings table (today); one global binding layout; unknown texture types as a uniform. |
| 4 | Behaviour lives in the operation. Where sources' rules differ, the operation is distinct or carries a mandatory semantic enum (`MinNum`, `UDiv { on_zero }`, `FToI { out_of_range }`, `MulZeroWins`, `ClampRange`, `Quantize`), named for the behaviour, never for a source. Each operation states its allowed results (exact, or a stated latitude). Constraints (`precise`, `nonuniform`, convergence) and permissions (minimum precision, contraction) are separate typed fields. Floating-point controls are module defaults with per-operation overrides. Target legalization is a sequence of checked passes driven by a target profile. | Program-wide mode flags whose meaning the body must agree with (today's `D3dModes`); expansions in each lifter or each back end; a source-intrinsic escape hatch. |
| 5 | Small crates: `iridia-ir` (`no_std` + `alloc`, `serde` its only dependency; it owns the interface vocabulary), `iridia-passes`, `iridia-analysis`, one crate per source and per target, `iridia-naga` holding the only naga dependency, `iridia-exec`, the facade `iridia` with pipelines, `iridia-cli`, `iridia-web`. `Program::module` is replaced by `Program::ir`; `iridia_naga::lower` returns a stock `naga::Module`, its `ModuleInfo` and the binding and provenance maps. | One crate; naga behind a feature of `iridia`. |
| 6 | `iridia-naga` pins wgpu's naga (`=30.0.1`), builds every naga node in one adapter module, checks a per-feature preflight table before lowering and validates every result with naga. What stock naga cannot hold (a texture atomic whose old value is read, image exchange) or its writer cannot write (WGSL ray queries) is refused by name and carried by iridia's own SPIR-V writer. naga's next IR break (`As` reworked, `Expression::Bitcast` added) changes the adapter only. | Keeping the fork for three constructs; serde moves between naga versions (they drop fields silently). |
| 7 | Three validation layers: the IR validator (types, scoping, effects, convergence, layouts, stage rules), validation after every pass in tests, and target validation (naga's validator, `spirv-val`, the reference compilers). | Validating through naga only. |
| 8 | Verification is part of the design: a corpus matrix with committed manifests, an evidence ladder from decoding to execution, differential execution against several oracles, per-feature probes with interaction cases, random programs with a resumable journal, external suites run as tools. Every migration step's bar is drawn from it. | |
| 9 | Migration is additive: IR and naga back end, then D3D9 SM3 with a small IR executor (bar: the 7,308 Shaun White Skateboarding programs), SM1/SM2, DXBC, DXIL, the rest of the executor and analyses, consumers, new back ends, then `primase-naga` is dropped. | A big-bang port; deferring all execution to the end. |

## 1. Control flow

### 1.1 What arrives and what is needed

| Source | Control flow as stored |
|--------|------------------------|
| D3D9 | structured: `if`/`ifc`/`else`/`endif`, `loop aL, i#`, `rep i#`, `break`/`breakc`/`breakp`, `call`/`callnz`/`label`/`ret`, predication |
| DXBC | structured: `if`, `loop`, `switch`, `break(c)`, `continue(c)`, `ret(c)`, `call`, subroutines |
| DXIL | an SSA CFG with phis, reducible in dxc's output in practice |
| SPIR-V (later) | a structured CFG with merge and continue declarations |
| RSX fragment programs | structured (`IFE`, `LOOP`, `REP`); vertex programs branch to addresses |
| console microcode | arbitrary branches |

| Target | Needs |
|--------|-------|
| naga, WGSL, GLSL, HLSL, MSL | nested statements; `break` and `continue` leave only the innermost loop or switch |
| SPIR-V | a CFG with selection and loop merges and continue targets obeying SPIR-V 1.6 §2.11 |
| DXBC encoder | nested constructs, innermost break |
| DXIL | any reducible CFG |
| `iridia-exec` | anything |

### 1.2 The two control phases

Every function has one value arena and one operation vocabulary. Its body is one of:

```
Cfg      = [Block]          Block = params, [Op], Terminator
Terminator = Jump(B, args) | Branch(cond, B(args), B(args)) | Switch(sel, [(case, B(args))], B(args))
           | Return(values) | Kill | Unreachable

Regions  = Region           Region = [Node]
Node     = Op
         | Block(L) Region -> results              -- `br L(values)` leaves it
         | Loop(L) params = inits, Region -> results -- `continue L(values)` restarts it, `br L` leaves it
         | If(cond) Region Region -> results
         | Switch(sel) [(case, Region)] Region -> results
Exit     = Br(L, values) | Continue(L, values) | Return(values) | Kill | Unreachable
```

Merging values are block parameters (CFG) or region results and loop parameters (regions); there is no phi
instruction with a predecessor list to keep in step. In `Regions`, a value is visible after its definition in its
region and the regions nested in it, so dominance is nesting.

- Structured sources build `Regions` directly. Their constructs are the program's convergence structure (where
  divergent lanes rejoin, which decides what a derivative, an implicit-LOD sample or a wave operation computes), so
  they are semantics. A pass that changes them must justify the convergence effect; the first passes do not
  (no jump threading, no block merging across constructs).
- Graph sources build `Cfg`. A `Cfg` program is valid, specialisable, analysable and executable as it is.
- `structurize` turns `Cfg` into `Regions`. It is the existing `build::structure` (Ramsey, "Beyond Relooper", ICFP
  2022), which already takes a naga-free graph (`Cfg`/`Term`) and returns the `Block`/`Loop`/`Br`/`Continue` tree
  this phase uses. It moves to `iridia-passes`, is checked against the graph classes it accepts today, and refuses
  an irreducible graph by name. Node splitting can be added there later; a per-invocation program counter loop is
  not a safe default because it changes reconvergence of barriers, derivatives and wave operations.
- An output pipeline asks for the phase it needs. Structured targets run `structurize` once, after specialisation
  and before target legalization, and the result is frozen for that pipeline. A `Regions` program projects to a
  `Cfg` view cheaply where an analysis or the executor prefers edges.

Labelled multi-level exits are allowed in `Regions`, because any reducible graph maps into them without new state.
For targets with innermost-only `break`, the pass `break_flags` adds a jump variable only where an exit crosses a
construct (today's `structure::lower` routing, moved to the IR). It records that it changed control and must keep
the convergence constraint of the exit it rewrote. The SPIR-V writer derives merges and continue targets from the
tree after normalising them: two constructs never share a merge block, each loop has one continue target and one
back edge, and every merge is dominated by its header in the structural sense of §2.11.

### 1.3 Why not a CFG alone

If the canonical form were only a CFG and structure were rebuilt per output, any CFG rewrite (block merging, jump
threading, folding a specialised branch, removing an empty arm) could change the merge the next structurization
chooses, and with it what lanes a derivative inside sees. yuzu, Ryujinx and shadPS4 all structurize once and carry
that structure to their back ends (section 10). Structured sources would also go structured to CFG to structured,
which is a second source of differences against today's WGSL for nothing.

### 1.4 Why not regions alone

A frontend forced to structurize before returning a program refuses graphs that only the caller's specialisation
makes reducible:

```
entry: if state -> A else B        A: if x -> B else exit        B: if y -> A else exit
```

With `state` unknown, the cycle {A, B} has two entries; with `state = true` known, it has one. `Program::specialise`
runs after the lift, so the graph must be a valid program until then. This shape is a synthetic test case, not
something found in a corpus. The `Cfg` phase also lets the executor and DXIL output run a program no structured
target can take.

## 2. Types and values

- Scalars: `bool`, `i8 i16 i32 i64`, `u8 u16 u32 u64`, `f16 f32 f64`. Vectors of 2, 3 or 4. Matrices of float
  columns where a source states a matrix (DXIL's and HLSL's matrix types, a CTAB matrix); never inferred from dot
  products.
- Aggregates (arrays, structs) are types of memory (`Var`, buffers, groupshared), interface slots and constants,
  not of SSA values. A DXIL aggregate result (`extractvalue`/`insertvalue`, a compare-exchange's pair, an add with
  carry, a struct return) is decomposed into scalars and vectors by its frontend, deterministically and with tests.
- Literals hold exact bits (f16 as its 16 bits; no dependency on `half`). Bool constants are bools; a D3D mask
  (`0xffffffff` for true) is a conversion operation, not a bool representation.
- `Convert` (numeric, with a rounding and out-of-range rule) and `Bitcast` (same total bit size, reshaping allowed:
  `vec2<f16>` to `u32`) are distinct operations.
- `Undef` is an explicit value. A frontend states its policy (D3D9 refuses reads of unwritten lanes, as it does
  today; DXIL `undef`/`poison` get a documented policy when DXIL migrates) and the executor reports when an
  `Undef` reaches an output. Nothing is zero-initialised to satisfy a target.
- f64 is a type like the others. Lifters stop refusing doubles; targets state support in their profile (stock
  naga holds f64 and writes it in its own WGSL dialect, `f64` and `1.0lf`, which browsers' WGSL does not have).
- Minimum precision (DXBC `min16float`/`min10float`/`min16int`, D3D9 `_pp`, GLSL `mediump`, RSX precision fields)
  is a typed `Precision` on the operation: the minimum precision and range the source allows. It permits a lower
  precision; it does not state one. A legalization policy that picks one (compute at f32, or at f16 where the
  target has it) is recorded in the output artifact. Exact f16 arithmetic is the `f16` type, a different contract.
- Fixed point (`ps_1_x` registers, RSX `fx12`/`fx9`): no type. A frontend emits `ClampRange` and `Quantize` where
  the source's contract establishes a range or a step (section 4.3).
- RSX register formats: full (R, f32×4) and half (H, f16×4, two H in one R). The RSX frontend keeps one `Var` per R
  and reads and writes H through `PackHalf2x16`/`UnpackHalf2x16` on its lanes, so the overlap is explicit and the
  shared promotion pass handles it. Whether an RSX half operation is IEEE f16 is decided from fixtures and
  documentation when that frontend is written, not from the field's name.
- D3D9 integer (`i#`) and boolean (`b#`) constants are banks like `c#` (section 3); `defi`/`defb`, like `def`, are
  literals that shadow their register.

## 3. Resources and interface

- `Program { ir, interface, provenance, notes }`. `bindings` goes (the IR names resources by id); `conventions`
  move into the interface (clip depth and origins are facts of the interface); `D3dModes` goes (section 4).
- One source of layout truth, in the core. The interface vocabulary (resource kinds, element types, buffer layouts
  with offsets, strides, sizes and matrix majorness, varyings, ids) moves into `iridia-ir`, serializable with `serde`
  (the core's only dependency, `default-features = false` with `alloc`, so `no_std` holds). `iridia-interface`
  stays as a thin crate re-exporting those types, so code that wants interface data without lifting keeps its
  import path. The IR refers to resources and slots by `ResourceId` and `SlotId`. The existing normalisation rule stays: D3D register packing, DX9 constant packing and
  std140 become explicit offsets and strides, the original kept in provenance.
- Provenance is a list of typed source addresses per resource and per varying: `D3d { class, register, space,
  range }`, `D3d9 { bank, register }`, `Rsx { unit }`, `Vulkan { set, binding }`, `Gl { name, location }`, and
  original byte offsets. A list, because linking, merging and aliasing can give one resource several. The D3D
  vocabulary exception in CONTRIBUTING (C6) is no longer needed.
- Target bindings are an output artifact: `BindingPlan` maps each logical resource to a target descriptor and lists
  any helper resource legalization introduced (a state block, a split sampler). `emit::Pipeline`'s chosen bindings
  are a `BindingPlan`. locum can patch a D3D register while primase gets WebGPU bindings from the same program.
- Interface I/O: operations `LoadInput(slot, lanes)`, `StoreOutput(slot, lanes, value)`, so partial lane writes stay
  natural. The entry point's argument and result layout for naga (`entry_io` today) is the naga back end's.
- Pipeline state the code depends on is a declared `StateParam` in the interface (new resource kind
  `PipelineState`): D3D9's texture dimension and `D3DTTFF_PROJECTED` per stage for `ps_1_x`, `D3DTSS_BUMPENV*`,
  alpha test, fog, point sprite, a null sampler, fetch-4, shadow comparison. dxbc-spirv's SM3 frontend reads the
  same set from its `Legacy*` built-ins (`sm3/sm3_resources.cpp`, `sm3/sm3_converter.cpp`). `specialise` folds
  known ones. Unknown numeric state lowers to a state block in the `BindingPlan`. Unknown type-changing state (a
  texture's dimension) cannot be a uniform: the program is either specialised per variant, or declares one typed
  resource per possible dimension and the `BindingPlan` says which the host binds.
- Memory operations state access, scope and ordering; atomics state operation, scope and memory semantics and
  return the old value whatever the resource kind; image and sampler aliasing is explicit.

## 4. Source semantics: operations, contracts, policies

### 4.1 Rules

- Each operation states its allowed results for every input, including NaN, infinities, signed zero, denormals,
  zero divisors, out-of-range conversions and out-of-bounds access: exact, or a stated latitude (an ULP bound, an
  implementation-defined choice from a stated set), or `Undef`. Ordering of effects is part of the contract.
- Where sources' defined rules differ, the operation is distinct or carries a mandatory semantic enum, named for
  the behaviour: `FMin`/`FMax { nan: MinNum | Propagate | Any }`, `UDiv`/`URem { on_zero: AllOnes | Dividend | Undef }`,
  `FToI`/`FToU { out_of_range: Saturate | Undef, nan: Zero | Undef }`, `Saturate { nan: Zero | Undef }`,
  `FMul { zero: Ieee | ZeroWins }` (D3D9's legacy multiply), `FLog2 { neg_inf: Ieee | NegMax }`,
  `ClampRange(lo, hi)`, `Quantize { frac_bits, rounding }`, `Fma` versus separate multiply and add.
  A source whose rule is undefined maps onto any variant (GLSL's `min` with NaN takes `Any`); a source whose rule
  is defined picks exactly its variant.
- The operation keeps its semantic identity until a target. Lifters do not expand quirks into primitive
  operations; the target's legalization passes do, once, with the target's native form where it has one (the DXBC
  encoder writes `udiv`, the naga back end writes the select).
- Typed fields beside the operation, in two kinds: constraints a pass must honour (`precise`, `nonuniform`,
  convergence) and permissions it may use (`Precision`, contraction). Neither kind changes what the operation means.
- Floating-point controls: module defaults per float type (denormals preserve, flush or any; rounding) plus
  per-operation overrides where a source has them (yuzu's IR carries such per-operation controls,
  `frontend/ir/modifiers.h`). D3D11's expected f32 flush is a module default, today's `flush_denormals`.
- Every operation carries its source origin (`Unit::Word`, `Bit` or `Line` and the offset), so refusals, notes,
  `project(after_word)`, locum's patching and the DXBC encoder map back to the bytes.
- A frontend choice between implementation-defined behaviours is a compatibility policy, recorded in provenance
  (part of the cache key) and shown as a note. It is never presented as what the original hardware did unless a
  fixture or capture proves it.

### 4.2 What today's lifters spell out, and what it becomes

| Today (`build::behaviour`, `D3dModes`) | IR |
|----------------------------------------|----|
| `divide_by_zero`: udiv by 0 gives all ones, urem the dividend | `UDiv { on_zero: AllOnes }`, `URem { on_zero: Dividend }` |
| `float_to_int`: truncate, saturate, NaN to 0 | `FToI { out_of_range: Saturate, nan: Zero }` |
| `min_max_nan`: the non-NaN operand | `FMin { nan: MinNum }` |
| `saturate_nan`: NaN to 0 | `Saturate { nan: Zero }` |
| `flush_denormals` (an expectation) | module float control |
| D3D9 `mova`: round half away from zero | `Round { mode: HalfAwayFromZero }` then `FToI` |
| D3D9 `nrm` zero vector: `FLT_MAX` scale | plain operations (a source definition, not a quirk of an operation) |

### 4.3 D3D9 `ps_1_x` worked through

Ranges are Microsoft's "ps_1_X Registers" page, Range table (checked 2026-11-05): `c#` -1 to +1, `r#` within
±`PixelShader1xMaxValue`, `t#` within ±`MaxTextureRepeat`, `v#` 0 to 1, and "early pixel shader hardware represents
data in registers using a fixed-point number ... approximately eight bits for the fractional part".

| Source rule | IR |
|-------------|----|
| `c#` reads are -1..1 | `ClampRange(-1, 1)` on each read; `def` literals clamped at lift (dxbc-spirv does both, `sm3/sm3_resources.cpp` lines 113 and 146) |
| `v#` colours are 0..1 | `ClampRange(0, 1)` on the input read |
| `r#` within ±`PixelShader1xMaxValue`, a device cap | policy `ps1x_max`; DXVK reports `FLT_MAX` (`d3d9_adapter.cpp:708`); finite `m` adds `ClampRange(-m, m)` on each write |
| `t#` as a temporary (1.1 to 1.3) within ±`MaxTextureRepeat` | the same policy shape, `ps1x_texture_max` |
| fixed point, about 8 fractional bits | policy `ps1x_quantize: None or Some(frac_bits, rounding)`; `Some` adds `Quantize` after each write; the documentation says "approximately", so any choice is a compatibility policy |
| source modifiers `_bias`, `_bx2`, `1-x` (`comp`), `_x2` (1.4), negate | explicit arithmetic in the documented order |
| destination `_x2 _x4 _x8 _d2 _d4 _d8`, then `_sat` | multiply by the power of two, the range clamp of the policy, then `Saturate`, in that order (each fixture checks the order) |
| co-issue `+` | the frontend reads every source of the pair before committing either write (dxbc-spirv buffers the stores, `sm3/sm3_converter.cpp` around line 63); nothing in the IR |
| write masks | the next register value is built from the unwritten lanes kept and the written lanes new |
| `cnd` (1.1 to 1.3: `r0.a > 0.5`; 1.4: per lane) | `FCmp(>)` and `Select` |
| `tex t#` (1.1 to 1.3): stage n with texture coordinate n, dimension and projection from texture stage state | `StateParam::TextureDim(n)` (type-changing: specialisation or one typed resource per dimension) and `StateParam::Projected(n)` (numeric) |
| `texbem`, `texbeml` | `StateParam::BumpEnvMat(n)`, `BumpLScale(n)`, `BumpLOffset(n)` and arithmetic |
| `texm3x3pad/tex/spec/vspec`, `texdp3`, `texreg2ar/gb/rgb` | dot products and a typed sample; `texm3x3spec`'s eye vector reads `c#` (clamped) |
| `texkill` | `Kill` under the predicate over the lanes that version tests (xyz in 1.x, all four from 2.0; each in a fixture) |
| `ps_1_4` phases, `texld` and `texcrd` | frontend register and texture-coordinate sequencing at the phase mark; no IR operation |
| `r0` is the colour at the end | `StoreOutput(Target0)` from `r0` at exit |
| `_pp` | a `Precision` permission, not a conversion to f16 |

No type, no mode and no source-specific operation is needed: every behaviour is an operation or a policy-driven
clamp the back ends implement or legalize.

## 5. Crates, public API, versioning, consumers

```
iridia-ir          types, operations, module, both control phases, the interface vocabulary, builder, validator,
                   text printer and parser; no_std + alloc; serde its only dependency
iridia-interface   re-exports the interface vocabulary
iridia-passes      structurize, promote (Var to SSA), dce, cse, fold, specialise, target legalization passes,
                   scalarize, break_flags; no_std + alloc
iridia-analysis    flow (Values, Flow, link, VertexDemand), matrix, output stores and uses, constant reads
iridia-d3d9, iridia-dxbc, iridia-dxil, iridia-mojo; later iridia-rsx, iridia-spirv-in      frontends
iridia-naga        IR to stock naga 30.0.1; the only crate that depends on naga
later: iridia-spirv (on rspirv), iridia-wgsl (naga-free), iridia-glsl (GLSL ES 3.0), iridia-dxbc-asm, iridia-hlsl
iridia-exec        CPU executor over the IR (both phases)
iridia             facade: Program, pipelines (read, lift, passes, emit); emit::wgsl through iridia-naga until
                   iridia-wgsl exists
iridia-cli, iridia-web (the browser page, wasm32)
```

- dx stays bytes to typed records; iridia owns lifting and everything after.
- No I/O and no `unsafe` in iridia's own library source (`#![forbid(unsafe_code)]`); dependencies are audited, not
  required to be free of `unsafe` (`alloc` itself is not). The one planned dependency with `unsafe` is rspirv in
  `iridia-spirv`: it builds for `wasm32-unknown-unknown`, but needs `std` and has `unsafe` in its binary parser
  (`binary/parser.rs`), so the SPIR-V writer is a `std` crate.
- Everything builds for `wasm32-unknown-unknown`; a WGSL-only wasm user stops compiling naga once `iridia-wgsl`
  exists (until then the prototype's WGSL goes through naga, and says so).
- API sketch:
  - `iridia::lift(bytes, Source) -> Result<Program, Refusal>`; each frontend keeps its own `lift`.
  - `program.validate() -> Result<ValidatedProgram<'_>, Vec<Problem>>`: a borrowing checked view, so a mutated
    program cannot keep a stale validation. Lowering and emission take a `ValidatedProgram`.
  - `iridia_naga::lower(&ValidatedProgram, &NagaProfile) -> Result<NagaArtifact, Diagnostic>`, where
    `NagaArtifact { module, info, bindings: BindingPlan, origins }` and `Diagnostic` names the operation, its source
    origin and the target requirement it fails.
  - `iridia::emit::wgsl(&ValidatedProgram, &WgslProfile)` with an explicit browser or naga-dialect profile, and
    `wgsl_for` with a `BindingPlan`.
  - Analyses keep their shape; `flow` re-exports iridia's operation enums instead of naga's.
- Versions: `iridia_ir::VERSION` replaces `PROGRAM_VERSION` (bumped when meaning or the text form changes); each
  frontend's `LIFTER_VERSION` and each target's `EMIT_VERSION` stay; caches key on all three plus the policies. The
  `gate` tables hash the IR text. The text form has a schema version line and is not a promised storage format.
  All crates move from 0.4 to 0.5 at the first cut-over and are versioned together until 1.0.
- Consumers move at migration step 6, from logical ids and the interface rather than naga handles:
  - primase: `core/src/shader` (binding, blend through `iridia::flow` and `iridia_exec::value`, `input_fallback`
    reads `program.module`'s entry result struct, `specialise`); `worlds/*` digest tests.
  - primase-lab: `render/game_shader` (`program.rs` reads `program.module` global bindings, `streams.rs`
    `VertexDemand`, `derived_input.rs` `flow` and `iridia_exec`), exec tests. It drops the fork alias `naga-ir` and
    the WGSL re-parse bridge once its wgpu resolves to the same naga as `iridia-naga` and an integration test
    passes a lowered module straight to wgpu.
  - locum: `cli/src/trace.rs` on `iridia::flow`; later the DXBC encoder for shader patching, without naga.

## 6. The naga back end

- `iridia-naga` depends on `naga = "=30.0.1"`, the version wgpu `=30.0.1` uses, so a lowered module goes straight
  into wgpu.
- Before lowering: the naga profile's legalization passes (quirk operations to naga expressions, minimum precision
  policy, `break_flags`), and `structurize` if the program is in `Cfg`.
- Lowering: `Regions` to naga statements; SSA values to expressions with `Emit` ranges, keeping effect order;
  region results and loop parameters to locals; interface slots to entry arguments and the result struct; the
  `BindingPlan` to globals with group and binding.
- Every naga node is built in one adapter module. naga trunk commit `97474455` (2026-10-31) replaces
  `As { kind, convert }` with `As { target }` and adds `Expression::Bitcast { ty }` (other IR additions since
  30.0.1: `RayQueryFunction::Begin`, `BuiltIn::HitBarycentrics`, `Statement::DebugPrintf`). With iridia's own
  `Convert` and `Bitcast` already distinct, that change lands in the adapter and nowhere else. `iridia-naga`
  follows wgpu's naga, one major at a time; a feature probe for each row of the table below runs on every bump.
- Preflight, before lowering, per feature and per column (stock IR and validator / the requested naga writer /
  portable browser WGSL). naga's own `back::wgsl::supported_capabilities()` returns all capabilities and is not a
  preflight.

| Feature | Stock naga 30.0.1 IR | Its WGSL writer | Browser WGSL | Carried by |
|---------|----------------------|-----------------|--------------|-----------|
| texture atomic whose old value is read | no result field | no | no | `iridia-spirv` |
| image exchange, compare-exchange | rejected by the validator | no | no | `iridia-spirv` |
| texture atomic, result unused | yes (subtract as add of the negation) | yes | needs the feature | naga |
| ray queries | yes, and naga's SPIR-V writer | panics (`writer.rs:1066`) | no | naga to SPIR-V; WGSL refused |
| f16 | yes | yes (`enable f16`) | `shader-f16` | naga |
| f64 | yes | yes, naga's dialect | no | naga (native), refused for browsers |
| subgroup operations | yes | yes | per feature | naga |
| arbitrary CFG | no | no | no | `structurize` first |

- An unused old value may be dropped when that is the only obstacle. An old value is never replaced by a later
  load, and an image is never silently emulated with a buffer (that changes bindings and needs a host contract).
- Every lowered module is validated by `naga::valid::Validator` with the profile's capabilities. A preflight
  refusal is a named target limitation; a module the preflight accepted and naga rejects is a lowering bug,
  reported with the operation's origin.

## 7. Validation

1. The IR validator: ids and ownership; one definition per value; operand and result types; block parameters
   and exit arguments (arity, types); scoping in `Regions` and dominance in `Cfg`; exits target enclosing labels;
   effects and their order; convergence constraints; memory access, scope and ordering; layouts and alignment;
   resource and slot ids agree with the interface (kind, dimension, bounds of constant reads); stage rules (`Kill`
   in pixel programs, derivatives and implicit-LOD sampling where defined, barriers in compute and hull);
   `StateParam`s declared; semantic enums present; unreachable code is checked too. It runs at every trust
   boundary (after a lift, before lowering) and after every pass in tests and debug pipelines.
2. Transformation checks: `structurize`'s output against SPIR-V's structural rules and the convergence it must
   keep; `break_flags` and legalization against their stated obligations.
3. Target validation: naga's validator on every lowered module; our own WGSL re-parsed and validated by naga's
   front end (naga-dialect profile) and checked against WGSL's specification for the browser profile; `spirv-val
   --target-env`; GLSL ES through glslang; HLSL recompiled by fxc or dxc; DXBC by D3D's own compile and
   disassembly round-trip. Acceptance by a validator is not semantic evidence; execution is (section 8).

## 8. Verification

### 8.1 Corpus matrix

Each corpus has a committed manifest (no game bytes): content hash, title and container, stage and model, and per
program the outcome of decode, lift, interface, IR validation, each target's lowering and validation, and
execution where run, with the tool versions and limits. Denominators are measured from the files present; only the
Shaun White Skateboarding count is known in advance.

| Corpus | Source kind | Role |
|--------|-------------|------|
| Shaun White Skateboarding SDB2 | D3D9 `vs_3_0` 1,100, `ps_3_0` 6,208 (7,308 distinct; ecb915c: 7,308 lifted, validated, WGSL) | the SM3 migration gate |
| Sekiro | DXBC SM5 | graphics, resources, layouts; WARP comparisons |
| Riders Republic, Unity games, skate, GTA V | DXBC and DXIL, as each manifest finds | DXBC/DXIL patterns, tessellation, compute |
| Elden Ring, including ray tracing bundles | DXIL | control flow, resources, ray tracing (ray queries and pipelines counted apart) |
| GTA V Enhanced, Frostbite | DXIL | independent modern DXIL workloads |
| MonoGame MGFX | MojoShader GLSL and the D3D9 bytecode inside | both paths counted apart |
| PS3 SHRED | RSX | the RSX frontend, when it exists; decoding counted apart from lifting |

### 8.2 Evidence ladder

1. Decoding and interface: records, reflection and signatures against Microsoft's disassembler (the
   `d3d-disassembler` oracle set; trumank/libd3dcompiler runs `d3dcompiler_47.dll` on Linux, which can move these
   checks off Windows; the DLL is Microsoft's and is obtained separately), vkd3d's disassembler (already
   `iridia-dxbc/tests/vkd3d_oracle.rs`), and captured layouts.
2. IR well-formedness: every lifted and transformed program validates; malformed synthetic IR is refused with a
   diagnostic and no panic (the seeded-mutation census method from the 2026-10-23 crate review, extended to IR).
3. Output acceptance: naga's validator, WGSL re-parse, `spirv-val`, glslang, fxc/dxc.
4. Execution: the same inputs and resources through two implementations, compared exactly for integer, packing and
   bitcast results and within each operation's stated latitude for floats, with NaN, signed zero, infinities and
   denormals as deliberate dimensions, and side effects, discards and helper lanes compared too. Pairs:
   old route against new (migration); `iridia-exec` against WARP (the existing `exec_warp*` suites, Windows);
   against lavapipe running the SPIR-V that naga and `iridia-spirv` write (Linux); against dxbc-spirv on the same
   DXBC (an independent lifter); recompile-equivalence (t7_dxbc's method: the IR as HLSL, through fxc or dxc,
   lifted back and executed). `iridia-exec` implements operations directly from their contracts, not through the
   legalization a back end uses, so one shared lowering bug cannot make both sides agree.
5. Application behaviour: selected captures and renders with their binding data, compared feature by feature as
   the primase render-comparison rule states.

Each oracle has a domain: WARP for D3D11/12 semantics (not `ps_1_x` precision, RSX behaviour, wave width or every
ray tracing feature); libd3dcompiler for compiling and disassembling, not executing; t7_dxbc's simulator
simplifies textures and derivatives, so only its method is borrowed.

### 8.3 Probes, random programs, external suites

- One probe per operation and semantic variant (crosstl's method), plus interaction probes: masked writes in
  loops, co-issue with dependencies, precision with saturation, aliasing with atomics, divergence with
  derivatives, waves and barriers, parallel-copy cycles on loop edges. Each new operation needs a source
  fixture, validator rules, an execution probe with an independently reasoned expected value, and for each target
  profile either an exact lowering or a named refusal.
- Random valid IR (and, once the DXBC encoder exists, random bytecode) through every target, executed and
  compared; journal keys are input hash, IR, lifter and target versions, tool versions, profile, seed and harness
  version, and a run resumes only matching keys (t7_dxbc's resumable journal). Failures keep source, both IRs,
  outputs, inputs and the diagnostic, and are shrunk into a fixture.
- External suites, run as tools or read as oracles at test time and never vendored when (L)GPL:
  vkd3d's `tests/hlsl` and `tests/d3dbc` `.shader_test` files (HLSL or D3D9 assembly with resources and expected
  probes: compiled with fxc or dxc, lifted, executed by `iridia-exec`, compared with the stated probe; LGPL-2.1);
  Wine's d3dcompiler/d3d9/d3d11 tests and vkd3d-proton's D3D12 tests (LGPL-2.1; as conformance for anything that
  runs iridia's output under a runtime, and their shaders as differential inputs); DXC's test tree (compiler and
  FileCheck tests: DXIL inputs and interface facts, not GPU expected values; NCSA); naga's snapshot corpus (input
  WGSL for our WGSL writer's round-trips; MIT or Apache-2.0); dxbc-spirv's tests (MIT; its binaries return 0 even
  when tests fail and tolerate missing references, so the wrapper parses their output); crosstl's coverage tool
  (Apache-2.0) for comparison.
- Published tables per feature and per corpus with `supported`, `refused`, `wrong` and `untested` and
  tested/total beside any ratio. An unavailable oracle is missing evidence; a supported case becoming refused is a
  regression; a declared target limitation is an expected refusal, not coverage. Skips and expected failures of
  external suites are recorded as such.

## 9. Migration plan

Old crates keep working until each replacement passes its bar; nothing is deleted on the way.

| Step | Work | Bar |
|------|------|-----|
| 1 | `iridia-ir`, `iridia-passes` (structurize moved, promote, fold, specialise, `break_flags`), `iridia-naga`, text form; an `iridia-exec` subset over the IR (straight-line arithmetic, sampling, kill, loops) | unit fixtures in IR text with hand-worked values; the synthetic counterexample of 1.4; `cargo build --target wasm32-unknown-unknown` for every new crate except `iridia-naga`'s optional writers |
| 2 | D3D9 `vs_3_0`/`ps_3_0` on the IR, a new crate beside `iridia-d3d9` until cut-over | all 7,308 SWS programs lift, IR-validate, lower, naga-validate and write WGSL; interface and layouts identical to ecb915c; old and new WGSL saved and every difference classified with its reason; three-way execution (ecb915c lift in the current executor, the IR in the new executor, the new WGSL re-parsed and run in the current executor) on the measured runnable subset, which is reported; the Microsoft disassembler test (Windows); `tests/tokens.rs`-style synthetic fixtures re-run |
| 3 | D3D9 SM1.x and SM2.x on the IR (the work put on hold), from the fixture drafts | one-feature fixtures per version and modifier, assembled by Microsoft's assembler; the relevant vkd3d `d3dbc` tests; the SWS gate unchanged; `ps_1_x` policies tested for each choice |
| 4 | DXBC | every DXBC corpus manifest keeps its outcome; existing synthetic, `exec_warp*` and vkd3d tests; image atomics with old values and exchange need a working route (`iridia-spirv` brought forward) before the fork path retires |
| 5 | DXIL, through `Cfg` | Elden Ring including ray tracing, GTA V Enhanced, Frostbite, skate manifests; existing DXIL WARP suites (`exec_warp`, `_draw`, `_pipeline`, `_rq`); ray queries and ray tracing pipelines counted apart |
| 6 | the rest of `iridia-exec` and the analyses on the IR; MojoShader; consumers switch; old lifters deleted | existing executor and analysis tests keep their assertions; primase digest tests (`worlds/*/tests/*digest*.rs`), lab checks, locum trace tests |
| 7 | own targets: `iridia-spirv`, `iridia-wgsl`, `iridia-glsl` (GLSL ES 3.0 for the WebGL2 renderer), the DXBC encoder, HLSL | per-target probes, the target's validator and differential execution; a naga-free wasm WGSL build |
| 8 | drop `primase-naga` from iridia, primase and primase-lab | every fork behaviour has a supported route; no lockfile names the fork |

## 10. Prior art, as checked

All read, none copied. Pins: dxbc-spirv `f528e1e`, DXVK `cf0a0d9`, yuzu `dc94882`, Ryujinx `fec4f4a` (archive
mirror), shadPS4 `98ef621`, RPCS3 `eee3f0f`, wgpu `9e74495` and `v30.0.1`, rspirv `7e6f38e`.

- dxbc-spirv (MIT; DXVK's front end for D3D9 and DXBC, `src/d3d9/d3d9_fixed_function.cpp` uses its `sm3`
  namespace): SSA IR whose instructions carry flags (`Precise`, `NonUniform`, `NoNan` ...) and a per-type
  `SetFpMode`; scoped control flow lowered to SPIR-V-like structured blocks before SSA; `MinF16` as a scalar
  type with a lowering pass; D3D9 quirks as `*Legacy` operations (`FMulLegacy`, `FDotLegacy`, `FLog2Legacy`,
  `FPowLegacy`) that must be lowered; fixed-function state as `Legacy*` built-in blocks. iridia takes the
  operation-level quirks and the state set, and differs on min precision (a permission, not a type) and on
  structure (fixed at the source rather than rebuilt from scoped markers).
- yuzu (GPL-2.0-or-later): builds a structured syntax list by goto elimination before SSA rewriting
  (`frontend/maxwell/translate_program.cpp`), keeps it beside SSA instructions, and emits SPIR-V, GLSL and GLASM from
  it; per-operation float controls in `frontend/ir/modifiers.h`; host-dependent lowering of f64, f16 and int64,
  a silent approximation iridia does not take.
- Ryujinx (MIT): CFG, dominators, SSA, transforms, then de-SSA and goto elimination into a structured AST once
  (`Translation/TranslatorContext.cs`, `StructuredIr/GotoElimination.cs`); half registers packed and unpacked
  explicitly.
- shadPS4 (GPL-2.0-or-later): translates to an unstructured graph, rewrites to SSA and optimises there, then
  lowers phis, builds a structured syntax list once, rebuilds SSA and optimises again (`recompiler.cpp`); one
  SPIR-V back end; f64 narrowed to f32 on some hosts, which iridia does not do.
- RPCS3 (GPL-2.0-only for most files): fragment programs go through a register-level IR with a CFG and passes
  (`Emu/RSX/Program/Assembler/IR.h`, `CFG.h`), including a pass that makes half/full register aliasing explicit,
  then are written as text; fixed12/fixed9 precision is emulated with clamps; vertex programs are still
  decompiled to text directly, branches as a jump-position loop.
- SPIR-V 1.6 §2.11: unique merge ownership, structural dominance of merges by headers, one back edge per loop
  header, continue and merge distinct, construct entry and exit rules. rspirv (Apache-2.0) builds instructions as
  laid out in the binary and checks little else, so `iridia-spirv` validates with `spirv-val`.

## 11. Decisions for Caleb

1. `ps_1_x` default policy: follow what DXVK reports (`PixelShader1xMaxValue = FLT_MAX`, so no range clamp beyond
   `c#` and `v#`, and no quantization), or default to a finite early-hardware range and 8-bit quantization (the
   exact values to be taken from fixtures or documented caps, not assumed). Either is a recorded policy; the other
   stays available as an option.
