# iridia-d3d9-ir

Reads raw Direct3D 9 `ps_3_0` and `vs_3_0` shader token streams and their CTAB constant table, and lifts them into [iridia's shader IR](../iridia-ir).

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled shaders into programs other tools can read and write. This crate is the shader model 3 reader of [`iridia-d3d9`](../iridia-d3d9) moved onto the IR: the subset it reads and the interface it gives are that crate's, and the program is the IR's instead of a naga module. The design is [`docs/design/ir.md`](../docs/design/ir.md). The input is the token stream a Direct3D 9 compiler writes, version token to `END`, with no container. This crate does no I/O.

## What it reads

- The tokens are read by [`d3d9-reader`](https://github.com/Kalmarv/dx/tree/main/d3d9-reader); `read(bytes)` takes the `vs_3_0` and `ps_3_0` subset this crate lifts and refuses the rest by name and word: straight-line float arithmetic, `dcl` and `def`; vertex `slt`, `sge`, `mova` and float constants relative to `a0`; pixel `cmp`, `dp2add`, `texld` (plain, projected or biased) and `texkill`, with 2D, cube or volume textures.
- `signature` states the interface the declarations and the CTAB give: named varyings with their D3D registers and masks (a varying declared on some lanes of a register has those lanes, in order), the float constant bank as one constant buffer at `cb0` with the CTAB's named constants, and sampler `sN` as a texture at `tN` and a sampler at `sN`.
- `lift(bytes)` builds the program: a variable per register, each instruction's operations in order, the outputs stored at the end. Each operation whose meaning differs between sources states Direct3D 9's rule: `mova` rounds ties away from zero, a relative constant read outside the bank is zero, `texkill` tests all four lanes. A `def` constant is what its register reads as.

## Example

A `ps_3_0` program written as tokens: `def c0, 1, 2, 3, 4` and `mov oC0, c0`.

```rust
let words: [u32; 11] = [
    0xffff_0300,                                    // ps_3_0
    0x0500_0051, 0xa00f_0000, 0x3f80_0000, 0x4000_0000, 0x4040_0000, 0x4080_0000, // def c0, 1, 2, 3, 4
    0x0200_0001, 0x800f_0800, 0xa0e4_0000,          // mov oC0, c0
    0x0000_ffff,                                    // end
];
let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

let mut program = iridia_d3d9_ir::lift(&bytes).unwrap();
assert_eq!(program.stage, iridia_ir::interface::Stage::Pixel);
iridia_passes::promote(&mut program.entry).unwrap();
iridia_passes::simplify(&mut program.entry);
iridia_ir::validate(&program).unwrap();
assert!(iridia_naga::wgsl(&program).unwrap().contains("@fragment"));

// anything outside the subset is refused, with its word offset
let mut other_model = bytes.clone();
other_model[..4].copy_from_slice(&0xffff_0200u32.to_le_bytes()); // ps_2_0
assert!(iridia_d3d9_ir::lift(&other_model).is_err());
```

## Guarantees and limits

- What `iridia-d3d9` refuses this crate refuses: other models, operations and register banks, loop counters, predication, control flow, integer and boolean constants, sampler arrays, uninitialized temporary lanes and undeclared input lanes.
- On the 7,308 distinct Shaun White Skateboarding programs it lifts every program the older reader lifts, with the same interface and bindings, and a seeded run of each agrees with the older reader's in that crate's executor.
- It checks tokens and numbers, not rendered pixels: nothing here is compared against a Direct3D 9 device.

## Tests

`cargo test -p iridia-d3d9-ir` needs no game files: `tests/tokens.rs` runs hand-worked token and arithmetic fixtures through [`iridia-exec-ir`](../iridia-exec-ir) and writes each as WGSL, and `tests/versions.tsv` gates the lifter version on the IR text of its fixtures. With `GAME_DATA` naming a folder of your own extracted programs (see [`tests/inputs/README.md`](tests/inputs/README.md)), `tests/disassembly.rs` and `tests/corpus.rs` compare the decoded instructions against Microsoft's `D3DDisassemble`, then lift, validate and write each program as WGSL.

## Licence

GPL-3.0-or-later.
