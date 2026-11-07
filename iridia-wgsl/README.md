# iridia-wgsl

Writes a program of [iridia's shader IR](../iridia-ir) as WGSL, with no dependency but the IR.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled shaders into programs other tools can read and write. Readers lift into the IR, [`iridia-passes`](../iridia-passes) puts a program in the form a writer takes, and this crate writes the text. The design is [`docs/design/ir.md`](../docs/design/ir.md). This crate is `no_std` with `alloc`, has no unsafe code and does no I/O; it builds for `wasm32-unknown-unknown`.

[`iridia-naga`](../iridia-naga) lowers the same programs to an upstream naga module, for naga's other writers; the two decide alike what each resource binds to ([`iridia_ir::target::BindingPlan`]), how the entry point carries the interface (`entry_io`) and which capabilities a program needs (`requirements`), and each refuses what it cannot write by name.

## What it does

- `wgsl(program, profile)` writes the vertex, pixel or compute program: the entry point `main` with an argument per interface input (a compute program's are its invocation ids, built-ins) and a member of the result structure `Out` per output (a vertex program's clip distances are one `clip_distances` array), the thread group size of a compute program, and each resource a global where the `BindingPlan` binds it. `Profile::BROWSER` binds inside WebGPU's default limits and states the standard's `enable` directives; `Profile::NAGA` binds Direct3D's registers as they are and writes naga's dialect (`f64` and 64-bit integers).
- `wgsl_for(program, profile, &Pipeline { plan, feeds })` writes it for one pipeline: the resources where `plan` binds them, and a vertex program's outputs where the pixel program it `feeds` reads them, matched by semantic as Direct3D links stages; an output read with more lanes is padded with zeros, with fewer its first lanes, as another 32-bit scalar the same bits, each noted in `Written::notes`.
- The text is one `let` per value of the program, in its order, with the structured control flow of its regions. A construct's results and a loop's parameters are variables that each exit assigns and the code after the construct reads once. An output is a local `var` per slot, assigned lane by lane, and returned in `Out`.
- What WGSL does not have whole is written once: `f32tof16`, `f16tof32`, an exact `fmod` and the high half of a product are helper functions at the top of the module (integer arithmetic, so every device gives the same bits); a NaN test, `min` and `max` with their NaN rule, `saturate` of NaN, a float-to-integer conversion that saturates, rounding with ties away from zero and a zero-wins multiply are expressions.
- Memory is bound as the older lifts bind it: a constant buffer is rows of `vec4<u32>`; a raw or structured buffer a runtime array of 32-bit words (atomic words where an atomic reaches it); a typed buffer an array of `vec4<u32>`; a storage image `rgba32float`, `rgba32uint` or `rgba32sint`. Every read and write of a buffer or image is checked against its size, as the IR states: outside, a read gives zero and a write does nothing.
- Floats that are not ordinary (an infinity, a NaN, a denormal, a negative zero, and the largest finite `f32`, whose shortest decimal Chrome's compiler refuses as out of range) are written as the bits of an integer, bitcast.

## Guarantees and limits

- `preflight` refuses a program that needs a capability the profile lacks, naming the value, varying or resource that needs it.
- It writes the vertex, pixel and compute programs the Direct3D 9 and DXBC readers make, and refuses the rest by name, as `iridia-naga` does: hull, domain and geometry programs, mesh and ray tracing stages, an element of an array of resources, an atomic of a storage image, a multisampled array texture, a sample with a level clamp, a gather at a computed offset, a comparison sample at a level other than the derivatives' or zero, a sample of a 1D texture at an explicit level, bias or derivatives, and a biased sample of a depth texture or one at a computed level.
- WGSL's `break` and `continue` reach only the innermost loop or switch: run `iridia_passes::break_flags` first. A `Cfg` body must be structured first (`iridia_passes::structurize`). Pipeline state is not written: `iridia_passes::specialise` it first.

## Tests

`cargo test -p iridia-wgsl` needs no game files: `tests/write.rs` pins what each construct becomes, and `tests/ops.rs` writes each operation and has upstream naga read the text back and validate it. `iridia-dxbc/tests/exec_wgsl.rs` runs the text of every program of the corpora on WARP against the IR executor.

## Licence

GPL-3.0-or-later.
