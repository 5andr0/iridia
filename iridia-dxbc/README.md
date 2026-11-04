# iridia-dxbc

Lifts compiled Direct3D Shader Model 4 and 5 programs (DXBC, what `fxc` writes) into [`iridia::Program`](../iridia), a [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) module with its interface, and reads a container's reflection and signatures.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled Direct3D shaders into naga modules. The container and token stream are read by [`dxbc-reader`](https://github.com/Kalmarv/dx) and `dxcontainer`; this crate maps what they read.

## What it reads

- A DXBC container with a `SHDR` or `SHEX` program of Shader Model 4.0, 4.1, 5.0 or 5.1, of any stage: vertex, pixel, compute, hull, domain and geometry.
- Its `RDEF` and signature chunks (`ISGN`/`ISG1`, `OSGN`/`OSG1`/`OSG5`), for names, constant buffer layouts and signatures.

`lift(bytes)` returns a `Program` whose conventions follow `iridia`'s: every register is four raw 32-bit lanes (`vec4<u32>` locals) and each instruction bitcasts to the type it implies; varyings are unpacked from the signature registers into one entry argument or result member per signature element, with the register and lanes kept in provenance; resources are module globals (`cb<n>` a uniform `array<vec4<u32>, rows>` in group 0, textures and buffer views `t<n>` in group 1, samplers `s<n>` in group 2, typed texture views `u<n>` as storage images of four 32-bit lanes in group 3, binding = slot). Direct3D's results are written out (`D3dModes::ALL`): division by zero, float to integer conversion, `min`/`max`/`saturate` with a NaN, shift counts, and reads out of range of a dynamically indexed buffer (zero, as in D3D). `lift_with` takes the sizes constant buffers are bound at (`iridia::BoundSizes`), for a program that indexes past its declaration.

A pixel input's interpolation comes from the `dcl_input_ps` declarations; a stage that is not an entry point in WGSL (hull, domain, geometry, a vertex program that feeds a tessellator) lifts as functions (`iridia::stage_functions`), which `iridia-exec` runs.

SM5.1 bindings are resolved by resource class and declaration ID to their register range and space. Arrays of textures, samplers, constant buffers and typed/raw/structured SRVs and UAVs use binding arrays, with computed descriptor indices and unbounded ranges. D3D register/space provenance is preserved; naga bindings are numbered by declaration to avoid collisions between spaces. Constant-buffer and storage-buffer array elements are structures holding their rows or words. Non-uniform indices retain their computed selection and are noted on the program; consumers need the corresponding binding-array capabilities.

Reflection without a lift: `reflection` (resources and signatures), `signatures` (for a container whose `RDEF` was stripped, as some engines ship them; the constant layout is then the caller's), `resource_reflection` (resources only; `None` for a program whose `RDEF` is not Shader Model 5.0 or 5.1 pixel, vertex or compute). Reflection preserves register spaces, resource-array counts (including unbounded SM5.1 ranges), and unused constant fields. A stripped program's resources are named by register when lifted.

## Example

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let container = std::fs::read("tests/fixtures/modes.dxbc")?; // `fxc /T ps_5_0` output
    let program = iridia_dxbc::lift(&container)?;

    assert_eq!(program.stage, iridia::Stage::Pixel);
    assert_eq!(program.provenance.target, "ps_5_0");
    // the constant buffer `C : register(b0)` of the HLSL, with the names the container's reflection gives its fields
    let iridia::ResourceKind::ConstantBuffer { layout } = &program.interface.resources[0].kind else {
        unreachable!("the only resource is a constant buffer");
    };
    let fields: Vec<_> = layout.constants.iter().map(|c| (c.name.as_str(), c.offset)).collect();
    assert_eq!(fields, [("f", 0), ("u", 16), ("i", 32)]);

    let wgsl = iridia::emit::wgsl(&program)?;
    assert!(wgsl.contains("@fragment"));

    // a refusal names where and why
    let refusal = iridia_dxbc::lift(&container[..40]).unwrap_err();
    assert_eq!(refusal.bytecode, "DXBC");
    Ok(())
}
```

## Guarantees

- Unknown things are refused, by name and with the instruction word: an opcode the lifter does not carry, an operand of a register type it does not know, a store through a view with a partial write mask, `sample_info`... Nothing is approximated silently; what is approximated is a note on the program.
- A malformed program is an `Err`, not a panic or an allocation its declarations ask for: an instruction's operand count is its opcode's (`dxbc_reader`'s operand formats, checked by its decoder before an operand is read by position), a resource operand is the register class the instruction takes, a declaration is read to its last word, and what a declaration states is within Direct3D 11's limits (4096 temporaries, 4096 elements in an indexable array, 4096 constant buffer rows, 32 KiB of thread group memory, 2048-byte structured elements, 1024 geometry output vertices, 32 geometry instances). `tests/malformed.rs` makes each case from a committed fixture.
- The container's digest is not verified: this reader does not authenticate shaders.

## What it does not carry

Refused by name: double precision (the global flag), mono samplers, append and consume buffer counters, geometry output to a stream other than 0, a store through a view with a partial write mask, a sample of a multisampled texture and any opcode not in its table. The sample count of the rasterizer is the pipeline's, not the program's, and is refused for the same reason. Sampling is not part of lifting: it is the caller's when a lifted program runs (`iridia-exec`).

## Tests

`cargo test -p iridia-dxbc` needs nothing else. What the tests check, and the real-data inputs they read when `GAME_DATA` names them (listed in [`tests/inputs/README.md`](tests/inputs/README.md)):

- `tests/lift.rs` lifts `tests/fixtures/modes.hlsl` (our own HLSL, `fxc /T ps_5_0`) and checks its interface, layout and the Direct3D guards in the WGSL; `tests/reflection.rs`, `tests/bound_size.rs`, `tests/malformed.rs` and `tests/versions.rs` (the version gate against `tests/lifted.hashes`) use the other fixtures.
- `tests/analytic.rs`: programs written as tokens, lifted and run on the executor against values worked out by hand from Direct3D's definition of each instruction (arithmetic and conversion edges, every numeric opcode, flow, derivatives, quads, sample callbacks, relative constant buffer addressing); nothing is read from data.
- `tests/sm51_exec.rs` runs focused SM5.1 fixtures and DXIL builds of the same HLSL on identical inputs, comparing every output word (and SM5.0 where expressible). `tests/sm51_oracle.rs` compares their decoded instructions, operand registers and indices with fxc disassembly. The inputs, expected outputs, build commands and oracle dumps are in `tests/fixtures/sm51`.
- `tests/disassembly.rs` and `tests/reflection.rs` against Microsoft's disassembly of real programs (`GAME_DATA`, `oracles.corpus.general/dxbc`, made by `tests/oracles/dxbc_corpus.py`, and `oracles.d3d-disassembler.riders/shader-pack`): the decoder's opcode sequence and count, and the reflection read from the binary, against the text.
- `tests/lift_corpus.rs` lifts folders of real containers (`oracles.corpus.<game>/shaders`), validates and writes each, and prints the refusals by reason; a program that lifts but fails validation fails.
- `tests/operand_format.rs` reads the Shader Model 4 and 5 programs of the corpora (`oracles.corpus.{sekiro,riders,sledding}/shaders`): every instruction has the operand count its opcode's format gives it (`dxbc_reader::decode` refuses any other), a slot of the format that holds a register with no data of its own (a resource, sampler, view or stream) holds one and no other slot does, a resource slot carries no negate or absolute value, and a destination is selected by a write mask, never a swizzle.
- `tests/container_digest.rs` (here and in `iridia-dxil`) checks the container digest of every container of the corpora and writes each container back with `dxcontainer::serialize`, byte for byte.
- `tests/vkd3d_oracle.rs` runs vkd3d's `vkd3d-dxbc -t` and `vkd3d-compiler` (programs, never linked; `tools.vkd3d_dxbc` and `tools.vkd3d_compiler`) over a spread sample of each corpus (`vkd3d.limit`, all with `all`) and compares part kinds, sizes and offsets, the digest verdict (including a copy with one digest byte changed, refused by both), the signature elements and the instruction mnemonics with `dxcontainer`, the signature readers and `dxbc_reader::decode`.
- `tests/exec_warp.rs` and `tests/exec_warp_tess.rs` render lifted programs on Microsoft's WARP rasterizer and run them on the executor, comparing every lane written; they need `tools.warp` (a host program this repository does not contain) and `tools.fxc`.

## Check the examples

From the repository root, `cargo test -p iridia-dxbc --doc` checks the crate documentation, and
`cargo build -p iridia-dxbc --examples` builds its example programs. Examples that read paths need the caller's files; the repository ships no game assets.

## Licence

GPL-3.0-or-later.

Shader Model 5.0 compute reflection is read through the same reflection path as pixel and vertex programs; a hull/domain patch signature may be `PCSG` or `PSG1`.
