# iridia-d3d9-ir

Reads raw Direct3D 9 shader token streams of every model from `vs_1_1` and `ps_1_1` to `vs_3_0` and `ps_3_0`, and their CTAB constant table, and lifts them into [iridia's shader IR](../iridia-ir).

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled shaders into programs other tools can read and write. It began as the shader model 3 reader of [`iridia-d3d9`](../iridia-d3d9) moved onto the IR, and gives the same interface for every program that crate reads; it now reads shader models 1 and 2 and the control flow of 2.x and 3.0 as well. The design is [`docs/design/ir.md`](../docs/design/ir.md). What the lift gives at the inputs Microsoft's instruction reference states a value for (`log` of zero, `texdepth` with `r5.g` zero) or leaves open (`rcp` and `rsq` of zero, `pow` of a zero base, `min` and `max` of a NaN), and why, is in the crate documentation's section on special inputs. The input is the token stream a Direct3D 9 compiler writes, version token to `END`, with no container. This crate does no I/O.

## What it reads

- The tokens are read by [`d3d9-reader`](https://github.com/Kalmarv/dx/tree/main/d3d9-reader); `read(bytes)` checks every register against its version's banks and refuses what it does not lift by name and word (the software versions, `ps_1_0` and `vs_1_0`, which no Direct3D 9 runtime takes, and a relative index other than `a0` into constants or `aL` into inputs and 3.0 outputs).
- Arithmetic of every model, with `def`, `defi` and `defb`; vertex `mova` and constants relative to `a0`; `rep`, `loop` and `aL`, `if`, `if_comp`, `break`, `break_comp`, `breakp`, `call`, `callnz`, `setp` and predicated writes; `dsx`, `dsy`, `texld` (plain, projected, biased), `texldl`, `texldd` and `texkill`, with 2D, cube or volume textures; `vPos`, `vFace`, several render targets and `oDepth`.
- Pixel 1.1 to 1.4: the texture operations (`tex`, `texcoord`, `texbem`, `texbeml`, the `texm3x2` and `texm3x3` families, `texdp3`, `texreg2ar`/`gb`/`rgb`, `texdepth`, 1.4's `texld`, `texcrd`, `bem` and `phase`), co-issue, the source modifiers and the result shifts. A stage's texture dimension and projection are pipeline state the program reads, unless the host states the dimension (`Options::texture_dims`); `iridia_passes::specialise_state` folds the state a host knows.
- `signature` states the interface the declarations and the CTAB give: named varyings with their D3D registers and masks, the float, integer and boolean constant banks as constant buffers with the CTAB's named constants (structs, arrays and matrices as the compiler laid them out), and each sampler as a texture and a sampler.
- `frame(bytes)` finds the first `vs_3_0` or `ps_3_0` program in a buffer whose later bytes are something else (a container's remaining bytes, say) and returns its stage and exact bytes through `END`, without searching for `END` inside operands or comments. `sampler_states` maps `D3DSAMP` state/value pairs to the interface vocabulary. Both are `iridia-d3d9`'s, here beside the lift that replaces it.
- `lift_with(bytes, options)` builds the program: a variable per register, each instruction's operations in order, the outputs stored at the end. Each operation whose meaning differs between sources states Direct3D 9's rule: a multiply's zero times anything is zero (`Options::multiply`, IEEE by choice), `mova` rounds ties away from zero, a relative constant read outside the bank is zero, `min` and `max` of one NaN give the other operand.
- The pixel 1.x register policy (`Options::ps1x`) is that of modern drivers by default: constants read within -1 to 1 and colours within 0 to 1, and no clamp or fixed point after a write; a register range and a fixed point rounding are options.

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
assert!(iridia_naga::wgsl(&program, &iridia_naga::Profile::BROWSER).unwrap().contains("@fragment"));

// anything outside the subset is refused, with its word offset
let mut other_model = bytes.clone();
other_model[..4].copy_from_slice(&0xffff_0100u32.to_le_bytes()); // ps_1_0
assert!(iridia_d3d9_ir::lift(&other_model).is_err());
```

## Guarantees and limits

- On the 7,306 distinct Shaun White Skateboarding programs the older reader lifts it lifts the same programs, with the same interface and bindings, and a seeded run of each agrees with the older reader's in that crate's executor.
- 105 one-feature programs, assembled by Microsoft's `D3DAssemble`, run as their comments state, worked by hand from the instruction definitions; where those leave a choice the rule followed is wined3d's. A program with a derivative runs as a 2 x 2 quad.
- The 1,100 programs of vkd3d's Direct3D 9 shader tests lift, and each probe the tests make of a Direct3D 9 device on Windows that the runner can draw holds, derivatives included (a pixel that takes one runs in its quad).
- `texkill` demotes the pixel (`Node::Demote`): it goes on as a helper whose values its quad's derivatives still read, as on Direct3D 9.

## Tests

`cargo test -p iridia-d3d9-ir` needs no game files: `tests/tokens.rs` runs hand-worked token fixtures, `tests/assembled.rs` the assembled fixtures and `tests/vkd3d.rs` vkd3d's tests, each in the IR, in its control flow graph, in that graph structured again, and as WGSL read back by naga. `tests/versions.tsv` gates the lifter version on what it makes of all of them. With `GAME_DATA` naming a folder of your own extracted programs (see [`tests/inputs/README.md`](tests/inputs/README.md)), `tests/disassembly.rs`, `tests/corpus.rs` and `tests/gate.rs` compare the decoded instructions against Microsoft's `D3DDisassemble`, lift every program and compare it with the older reader's. The assembled fixtures are rebuilt with `tests/oracles/assemble.py` (Windows Python and `d3dcompiler_47.dll`, or Wine).

With `tools.wgsl_run` configured (the WGSL harness, `tools/wgsl-run`, which runs a WGSL entry point on Direct3D 12's WARP adapter through wgpu), `tests/assembled.rs`, `tests/vkd3d.rs` and `tests/gate.rs` also run the WGSL of both writers, `iridia-naga`'s and `iridia-wgsl`'s, on WARP and hold it to the IR's run on the same inputs and constants, within 1e-3 relative (`tests/warp`): every assembled fixture, and one program in 128 of the others by file-name hash (`shader.graphics=1` runs every one). `pow` of a zero base and an exponent of 0 or less is undefined in the IR and in WGSL and a target compiles it as it likes, so a program that differs only there (it agrees in every lane once `pow` gives libm's value there) is counted in a class of its own, per writer.

## Licence

GPL-3.0-or-later.
