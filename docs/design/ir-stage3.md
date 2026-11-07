# IR migration stage 3: Direct3D 9 shader models 1 and 2 on the IR

What `iridia-d3d9-ir` lifts after stage 3: every Direct3D 9 model from `vs_1_1` and `ps_1_1` to `vs_3_0` and
`ps_3_0`, flow control included, checked against Microsoft's assembler, vkd3d's Direct3D 9 tests and the stage 2 gate.

## Results

| Check | Result |
| --- | --- |
| One-feature fixtures assembled by `D3DAssemble` (`tests/assembled.rs`) | 105 of 105 run as their comments state: 18 `ps_1_1`, 6 `ps_1_2`, 2 `ps_1_3`, 8 `ps_1_4`, 20 `ps_2_0`, 7 `ps_2_x`, 11 `ps_3_0`, 14 `vs_1_1`, 10 `vs_2_0`, 5 `vs_2_x`, 4 `vs_3_0`; 4 of them as 2 x 2 quads |
| vkd3d's Direct3D 9 shader tests (`tests/vkd3d.rs`) | 1,100 programs lift; 1,053 run (32 with control flow: 53 ifs, 23 loops), 47 read what the runner does not model; 1,781 probes hold and 3 more with denormals flushed, the 17 that take derivatives among them, each run in its quad |
| Shaun White Skateboarding gate (`tests/gate.rs`, unchanged) | 7,306 programs: 7,229 A = B = C on every run, 77 killed in all three |
| Version table (`tests/versions.tsv`) | 1,209 programs: 4 token fixtures, the 105 assembled and the 1,100 vkd3d programs |

## The assembled fixtures

Each fixture is a few instructions of assembly with comments that state the lift's options and its runs: inputs by
register, varying or pipeline state, and the outputs expected, worked by hand from Microsoft's instruction reference.
`tests/oracles/assemble.py` assembles each with `D3DAssemble` and writes Microsoft's `D3DDisassemble` listing beside the
tokens. The test checks that `d3d9-reader`'s listing is Microsoft's, then runs each stated run in the IR, in its control
flow graph (`flatten`), in that graph structured again (`structurize`), all three bit for bit, and in the WGSL read back
by naga. A program that reads pipeline state also runs specialised by each run's state, and lifted told the stage's
dimension.

Two things in the listings are not compared: the instruction slot estimate, which `d3d9-reader` gives differently from
Microsoft for `frc` (`vs_1_1`, 6 against 8) and `sgn` (`vs_2_0`), and one line Microsoft misprints (`breakp p0.x`, opcode
96, listed as `break p0.`).

## Derivatives: quads

A pixel program that takes a derivative runs as a 2 x 2 quad in both executors, lanes 0 and 1 the top row:
`iridia-exec-ir`'s `run_quad`, and `iridia-exec`'s `execute_quad` for the WGSL. A derivative is a difference
between lanes, coarse (the top row's or the left column's for every lane) or fine (the lane's own row's or column's).
D3D9's `dsx` and `dsy` are `Fineness::Any`, which both executors take as coarse; vkd3d's probes hold with it. An
implicit-LOD sample in a quad is asked with the quad's coarse derivatives of its coordinate, so `texld`'s level
follows its neighbours.

`iridia-exec-ir` runs the lanes one after another, without threads: each runs until it meets a derivative or an
implicit-LOD sample whose four operands are not all known, and once all four have met it, they run again knowing
them. The lanes must meet each one at the same operation, after as many loop iterations, as often before; otherwise
the run is refused as divergent, as `iridia-exec` refuses a derivative its lanes do not all reach.

vkd3d's `discard.shader_test` checks that derivatives are still computed after other pixels of the quad discard. A
`Kill` that ended the invocation could not pass it: the discarded lane had no value to give its neighbours, and in a
control flow graph a kill has no successor to go on to. So the IR has `Node::Demote`, a node in either phase: the
pixel is discarded and goes on as a helper, whose output stores have no effect (D3D9's `texkill`, D3D10's `discard`,
WGSL's `discard`). `texkill` lifts to it; `End::Kill` is a demotion that also ends the invocation. `iridia_ir::VERSION`
is 2 and the lifter's 4; the version gate named exactly the 11 programs with `texkill` as changed. Two killed runs
are alike whatever their outputs, since a discarded pixel writes nothing: the IR keeps the outputs stored before the
demotion, `iridia-exec` a helper's later stores too.

| Fixture | Checks |
| --- | --- |
| `ps_2_x_dsx_dsy` | the sum of the coarse `dsx` and `dsy` of a coordinate, every lane |
| `ps_3_0_dsx_dsy` | `dsx` and `dsy` apart, to two render targets |
| `ps_3_0_texkill_dsx` | a lane `texkill` discards still gives its value to its neighbours' `dsx` |
| `ps_2_0_texld_quad` | an implicit-LOD `texld` is asked with the quad's derivatives of its coordinate |

Each derivative's value is checked by the fixtures above and by vkd3d's derivative probes (`ddxddy.shader_test`
among them), the IR's results against the WGSL's.

## Rules the instruction reference leaves open

Where Microsoft's reference does not settle a result, the lift follows wined3d, whose rules come from tests on
hardware:

- A multiply is zero-wins (`0 * x` is `+0` for every `x`), as Windows drivers do; `Options::multiply` gives IEEE.
- Pixel 1.x reads of `c#` are within -1 to 1 and of `v#` within 0 to 1; a write is not clamped or quantized
  (`Options::ps1x` gives a register range and a fixed point).
- `D3DTTFF_PROJECTED` (pixel 1.1 to 1.3) and `texldp` leave a cube coordinate as it is. Pixel 1.4's `_dz` and `_dw`
  divide whatever the type.
- `texdepth` takes r5.g as 1 where it is past 1. The 0 to 1 clamp of depth is the pipeline's, for every depth output.
- 2.0 `texkill` tests all four lanes; Microsoft's assembler takes no other mask.
- `min` and `max` of one NaN give the other operand; `_sat` of NaN is 0.

## Fixed on the way

- `structurize` dropped a body that was already structured; `specialise` kept a texture only a folded branch sampled,
  so the WGSL bound it twice.
- vs_3_0 `o[aL]` writes and `v[aL]` reads were refused. A write stores into each declared output register under the
  predicate that the index names it; a read selects among the declared inputs, and is zero past them.
- ps_1_4 `texld` asked for the w of an `r#` coordinate that `texcrd r#.rgb` leaves unwritten.
- The version gate now covers every committed program, and names a dozen of each kind of difference before it counts
  the rest.

## Left for later stages

- The Shaun White Skateboarding gate runs one invocation at a time: its implicit-LOD samples leave the level to the
  sampler there.
- vkd3d's textured cases are not modelled. The fixtures sample through the harness's hash of each request.
- `d3d9-reader`'s instruction slot estimates differ from Microsoft's for `frc` and `sgn`.
