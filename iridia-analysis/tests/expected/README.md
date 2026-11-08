# Recorded analysis answers

`oracle.rs` compares each program with the legacy analyses and holds the IR answer and the comparison evidence to
these tables. Resource identities are normalized to the D3D binding, not a lifter's resource numbering. An answer is
the FNV-1a hash of the sorted per-output and discard reach, constant words read and indexed buffers. `demand.tsv`
records the actual vertex and pixel input masks for every fixture pair of the same family.

The remaining columns distinguish an analysis difference, an execution refusal and an exhausted execution budget.
An identical analysis answer does not imply that an executor ran the program: `execution not compared` says why it
did not. A source omitted by the IR that changes its output in a perturbation fails the test. A successful pair of
runs is evidence for that input, not proof of independence. Each ordinary source perturbation takes six seeds.

## Judging the differences

Each differing program is counted once, in the first applicable class below. "More precise" means correcting the
description of the actual computation: distinguishing a supplied builtin from an input, keeping an operand the
legacy graph omitted, or identifying an address as a coordinate. It does not mean that a larger set of leaves is
intrinsically better, or that an executor refusal supplies a runtime witness. Those evidence limits remain on each
row. No class changes the meaning of `Via`, an input lane, a texel-result lane or a buffer word.

| Class | Fixtures | DXBC | DXIL | Elden | Total | Direction | Cause / follow-up |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Typed-buffer contents and their addresses, including discard reads | 3 | 221 | 96 | 2 | 322 | IR more precise | Contents are texel-result leaves and addresses are coordinates; the legacy indexed-global representation loses the contents. |
| Constant-read roots of writes and descriptor selection, with unchanged output reach | 6 | 83 | 3 | 0 | 92 | IR more precise | Reads include observable writes and resource/sampler selection, not just output data. |
| Stage builtins separated from control-point input lanes, including accompanying guard/address corrections | 10 | 68 | 0 | 0 | 78 | IR more precise | Builtins are supplied stage values; actual input lanes and dynamic point selection stay represented. |
| Texel-address input changes from `Value` to `Coordinate`, with unchanged contents | 3 | 0 | 0 | 0 | 3 | IR more precise | The operand chooses a texel rather than supplying its returned value. |
| Descriptor and survival/continuation guards missing from the legacy result/read roots | 9 | 44 | 1 | 0 | 54 | IR more precise | Retains the code's descriptor operands and paths that skip later assignments; runtime witnesses or explicit execution refusals are recorded individually. |
| Fixed raw-buffer contents and atomic-path guard words | 0 | 3 | 4 | 1 | 8 | IR more precise | Literal byte addresses identify words; guards select paths that supply results. Runtime refusals are retained. |
| Nested-switch early-break inputs and wave-gather lane selectors | 0 | 4 | 6 | 0 | 10 | IR more precise | Legacy misses B.y/B.z nested-switch guards or the wave's lane-selector operand. |
| Loop-count word removed from an unrelated switch output | 3 | 0 | 0 | 0 | 3 | IR more precise | The count affects output 1.x, not output 0.x; dropped-source probes keep output 0.x. |
| Equivalent continuation paths retain a syntactic loop guard | 0 | 2 | 0 | 0 | 2 | IR less precise but sound | `021127da…` and `ca1d44de…` keep cb0.w although both paths make the same increment. Known follow-up: coalesce equivalent continuation states before attaching iteration-control dependencies. Both answer hashes match the pre-hang-fix IR baseline. |
| Changed meaning | 0 | 0 | 0 | 0 | 0 | Changed meaning | None. |
| **Total** | **34** | **425** | **110** | **3** | **572** | **570 more precise; 2 less precise but sound** | |

Open follow-up:

- [ ] Coalesce equivalent continuation/backedge states before attaching loop iteration-control dependencies.
  `021127da…` and `ca1d44de…` currently retain cb0.w over paths with the same increment. Keep their soundness
  checks and pin the reduced reach when that precision improvement is implemented.

The following distinctions were checked against the two value graphs and the lifted operations, in addition to the
per-row execution evidence. The tables preserve differences rather than calling a refusal or an unobserved change
an agreement.

| Difference | Judgement |
| --- | --- |
| A typed buffer's texel is added; its address changes from `Value` to `Coordinate` | The IR represents `LoadTexel` as a texel read. The legacy storage-array load records the address as an indexed global and omits its contents from scalar word reach. The address chooses data; it is not itself the loaded data. |
| Fixed raw-buffer words are added | The IR resolves a literal byte address to particular words. The legacy runtime-sized array remains indexed even at a literal element, so its flow omits those content words. This includes `GXFfxtessellateLensFlare.dpo`: words 0 and 1 of t10 decide its outer branch, and words 4 through 11 supply values. |
| Input lanes disappear from hull/domain results | Stage values such as output-control-point ID, primitive ID and domain location are `Supplied` values, not interpolated input lanes. The IR keeps actual control-point inputs and their dynamic point selectors. The scalar executor's stage refusal is retained. |
| Descriptor selection adds constant words | Both image and sampler selection are operands in the IR. The legacy image walk loses sampler selection. The heap/resource-array fixtures expose this separately from texture coordinates. These rows retain the executor's resource-array refusal. |
| Mesh/amplification writes add words to `WordsRead` | Vertex/primitive counts, mesh outputs and launch operands are observable write roots. They do not have to reach a pixel-style output slot to count as reads. |
| Wave gather adds its lane-selector input | `wave.lane` reads both the value and the requested lane. The legacy subgroup gather only records the value operand. The six slim DXIL cases retain the wave-execution refusal. |
| A switch/continue guard adds a word or input | An exit that skips later assignments can choose the returned value. The IR retains that survival guard. In the four DXBC nested-switch cases, B.z selects the 30/31 cases or B.y selects the 11/12 early break; the ordinary small random inputs do not exercise these literals, so their unchanged runs are not a claim that the source is unnecessary. |
| Loop control is retained although perturbations do not change the result | This is a syntactic may-reach analysis. It does not prove that different paths perform equivalent computations or that an iteration's contribution cancels. For example, `021127da…` and `ca1d44de…` retain cb0.w, which chooses two continuation paths with the same eventual increment. Their rows explicitly say no added source changed in the completed runs. |
| A switch fixture loses a word on an unrelated output | `switch*.dxil`'s iteration count affects output 1.x. It does not affect the earlier switch value stored to output 0.x. The dropped-source perturbations keep that lane. |

Two spurious differences were fixed rather than recorded: common discard-survival scopes no longer choose among
returns (`alpha_reference_transformed`), and nested exit/continue guards no longer choose between an outer block's
fallthrough and its explicit exits (`FXAA3HQ`). A switch case's survival guards likewise choose inside that case,
while the switch selector chooses between cases. Unit fixtures pin all three distinctions, including opposite
returns under a common survival guard and a bounded nested loop.

## Additional fixtures after the main rebase

The rebase adds five fixture rows and 46 demand pairs; all existing fixture and demand answers are unchanged.
The four `hull_unwritten` stages come from the hull patch-constant regression: the vertex program reads only a
supplied vertex ID, the hull/domain analyses agree with the legacy graph but scalar execution refuses their stage
builtins, and both graphs refuse geometry emission. `switch_shared.dxil` agrees on reach and completes in three
counted steps. Its `S0` input is absent from every paired vertex interface, so all 14 added DXIL pairs explicitly
refuse that missing semantic. The 32 pairs added for `hull_unwritten.vs` have five successful pairs: three need no
pixel input, and two read position.x, which the vertex program supplies without vertex inputs. The other 27 refuse
absent TEXCOORD0 or COLOR0 outputs.
The demand comparison checks both graphs on all 1,240 pairs: none differ.

One slim row widens from lift refusal to agreement: `3cd9568f…` now lifts after main's patch-function register fix.
The patch function's result registers come from its patch-constant signature, including unwritten results, rather
than the control-point output signature. This is the same cause pinned by `hull_unwritten`; its lifted analysis has
an empty reach and scalar execution completes with zero counted steps. No previously successful answer changes.

The two sampler-heap fixtures add two fixture rows and 83 demand pairs, none recorded before. `sampler_heap.dxil` samples
with `SamplerDescriptorHeap[Indices.sampler_index]`: its row differs from the legacy answer by one added word, the constant
word that selects the sampler, and by the read of it, with nothing dropped. That is the descriptor-selection class above
(IR more precise), with the executor's refusal of an element of an array of resources retained in the row.
`sampler_heap.vs.dxil` agrees in one counted step. None of the 83 demand pairs differs between the graphs.

## Early-depth fixtures

The four `early_depth` and `late_depth` fixtures (one of each in DXBC and DXIL) add four fixture rows and 48 demand
pairs. All existing fixture and demand answers remain unchanged. All four analyses agree with the legacy graph,
with the same answer hash: their output reads position.xy and their UAV store uses position.x as its address.
Early depth changes when the rasterizer permits that store, rather than the shader's dataflow. The analysis
oracle explicitly records `execution not compared` for storage writes; its agreement is not a runtime witness.
The draw executor's separate early-depth test checks the stores and visible pixels under passing, failing and
mixed depth tests, with the late-depth fixtures as controls. All 48 new demand pairs succeed and agree between
the two graphs, bringing the comparison to 1,371 pairs with no differences.

## Execution inputs and work budgets

Constants are filled using reflected scalar declarations. Where reflection is absent, direct word moves are traced
to numeric uses. A cast alone, sign-independent integer arithmetic, or an unused result does not establish a
numeric interpretation. A word with conflicting float/integer or signed/unsigned interpretations keeps the original
mixed raw fill; the row is labelled `input type ambiguous`. Twenty-two DXBC rows have that label. Ten sampled rows
contain live float arithmetic plus integer bit tests, or live signed and unsigned comparisons/conversions/division.

The pilot's largest completed execution took 8,193 counted steps, so the default budget is 16,384. The count is loop
iterations plus executed CFG blocks, not instructions or elapsed time. A straight region can complete with zero
counted steps. Nested compute programs first use that same default. Only a run that actually exhausts it retries
with 1,048,576 steps. Three rows need that retry; each reaches an explicitly unsupported storage write before the
larger ceiling. Their rows record the class and measured lower bound of 16,385 steps. A refusal returns no exact
step count, so no completed count is invented.

Two inputs still exhaust the default and the representative 1,048,576-step probes:

- `41ee8c4d…`: cb4[11].z is converted both as signed and as unsigned. The retained raw input is 1,061,158,912; the
  loop counter advances by four, with adaptive changes to the bound. Its initial nominal bound alone is 265,289,728
  iterations. Exhaustion is recorded with that input and the larger probe, not described as nontermination.
- `f61b7e97…`: Tint's two-word decrement starts at `0xffffffff00000000` (18,446,744,069,414,584,320), requiring about
  2^64 iterations on the nonzero-input path. A separate test supplies input.x=0 and compares the early exit exactly:
  one output lane, one step.

The legacy analysis runs in a separate process capped at 512 MiB and 30 seconds of worker CPU time per request.
The supervisor measures the worker with `GetProcessTimes` on Windows and `/proc/<pid>/stat` on Linux, so time waiting
to be scheduled does not consume its budget. A separate 600-second wall deadline fails a broken protocol; it is not
an accepted budget refusal. Either resource limit produces
`legacy oracle refused (resource budget)`. Diagnostics log which cap won; scheduling can change
that winner without changing the analysis answer. The unbounded SceneBokeh legacy graph previously allocated
21.7 GB. Both SceneBokeh rows retain the explicit resource refusal while their IR analyses complete. The regression
also requests a small fixture after that refusal, checking that the bounded worker restarts successfully.
The IR loop regression counts body walks over nesting depths 1 through 64: exactly one per loop, independent of
timing. The former SceneBokeh stall and both slim DXIL stalls came from repeated discovery of nested loop-carried
values; header phis installed before the single body walk eliminate that exponential work.

## Blend (`blend-*.tsv`)

`blend_oracle.rs` holds the blend recogniser on the IR (`iridia_analysis::blend`) to the older one (`iridia::blend`) on every pixel
program of the fixtures, the slim set and Elden Ring's corpus. Each output lane is spelled as text (its leaves named by what they
are: an input lane, a texel lane, a constant word; a term shared by many paths is spelled once and a spelling past a line is its
hash) and, where every leaf has a name, valued under four assignments of numbers to the named leaves, the same number for the same
name on both sides. A row's verdict:

| Verdict | Meaning |
| --- | --- |
| `same` | Every lane's text is the same. |
| `equivalent` | Texts differ, values agree under every assignment (a regrouping or reordering). |
| `gain` | Texts differ, some leaf is an unnamed `op` on a side so no value is compared, and the IR's lane names every leaf the older lane names. |
| `partial` | As `gain`, but the older lane names a leaf the IR's does not. |
| `differ` | Values differ under the same named leaves. |

| Set | same | equivalent | gain | partial | differ |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixtures | 93 | 6 | 6 | 0 | 0 |
| slim DXBC | 1522 | 0 | 198 | 0 | 16 |
| slim DXIL | 550 | 0 | 80 | 0 | 0 |
| slim D3D9 | 70 | 0 | 21 | 132 | 0 |
| Elden | 14156 | 4 | 11022 | 0 | 0 |

Elden also has one program the older side is not run on (`gxposteffect/SceneBokeh_SimpleBlurFadePS.ppo`: 24 nested loops, and
the older value graph is exponential in memory; a 4-worker pass reached 54.8 GB on it). The IR side runs on it, and the test
holds its lift, graph and blend answers to one second.

The classes, each judged once:

- **`differ`, 16 DXBC rows, all lane `(0, 2)` or `(0, 3)`:** the older graph spells an integer literal that a float operation reads
  by its bits as the integer's value (`literal 1065353216.0`, or the single the sum rounds it to), where the IR's constant is the float
  whose bits it is (`0x3f800000` is `1.0`; `1058642330` is `0.6`; `1061158912` is `0.75`). The IR is right; the older value is wrong.
- **`partial`, 132 D3D9 rows, correct and not a loss:** the IR's lift writes Direct3D 9's `mul` as `FMul(MulZero::Zero)` (0 times anything is 0,
  even infinity), which is not the IEEE multiply a `Term::Multiply` evaluates, so the recogniser leaves it a leaf (`word * 1.0` becomes
  `op`). The older path read it as the IEEE product, which gives NaN where the program gives 0: the older answer is the wrong one. Accepting
  every `FMul` in an experiment turns all 132 rows into `gain` or `same` (85 and 138 of the 223 programs) with no `differ`, so this is the
  whole cause. No consumer computes blend on a Direct3D 9 program: the only consumer of the recogniser in primase is `read_layered_albedo`
  (primase-core), which no world calls and whose test reads a DXBC program; nothing else in primase, primase-lab, locum or iridia uses it.
  A `Term` that spells the zero-wins multiply is built when a Direct3D 9 consumer first wants blend.
- **`gain`:** the older recogniser leaves a node of the lane an unspelled `op` where the IR's graph spells the float operation, and the IR's
  lane names every leaf the older lane named. The IR graph has no bitcast node, which accounts for some of these; the rows are not itemised by cause.
- **`equivalent`:** operands of an addition in the other order, or the same sum regrouped, with the same value.
