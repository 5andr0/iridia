# IR migration stage 2: Direct3D 9 shader model 3 on the IR

What the prototype of [`ir.md`](ir.md) shows on its migration gate: the 7,308 distinct `vs_3_0` and `ps_3_0` programs of
Shaun White Skateboarding's three SDB2 bundles (1,100 vertex, 6,208 pixel by distinct bytes), lifted by `iridia-d3d9-ir`
into the IR, through `iridia-passes` (`promote`, `simplify`), validated, lowered by `iridia-naga` to stock naga and
written as WGSL, against what `iridia-d3d9` at ecb915c makes of the same bytes.

## Results

| Check | Result |
| --- | --- |
| Lift, IR validation, lowering, naga validation, WGSL | 7,308 of 7,308 |
| IR text printed, parsed and printed again | 7,308 identical |
| WGSL declarations (structures, bindings, globals, the entry point's signature) against ecb915c | 7,308 identical |
| Three-way execution, 16 seeded runs each | 7,308 agree: A = B = C |
| Microsoft `D3DDisassemble` gates (`tests/disassembly.rs`) | 26, 25 and 41 checked, all exact |
| Corpus listings against `D3DDisassemble` (`tests/corpus.rs`) | 1,099 vertex and 6,207 pixel listings equal, all lifted: every program of the oracle set, which holds 7,306 (one vertex and one pixel program fewer than the gate: the init bundle's, see below) |
| Hand-worked token fixtures (`tests/tokens.rs`) | 6 of 6, on the IR executor |
| wasm32 builds | `iridia-ir`, `iridia-passes`, `iridia-d3d9-ir`, `iridia-exec-ir`, `iridia-naga` with no features and with `wgsl` |

## The gate in the repository

The checks above first ran as tools outside the repository. They are now one test, `iridia-d3d9-ir/tests/gate.rs`,
which counts every program by outcome class and fails on any outside "A = B = C": each program lifts, IR-validates,
prints and parses back to the same text, and writes WGSL for browsers; its interface and WGSL declarations equal
`iridia-d3d9`'s; it runs three ways, 16 seeded runs each. A second test makes each program's first float addition a
subtraction in B and requires A and B to disagree on at least three quarters of the programs that have one. Without the
oracle set `d3d-disassembler/shaun-white/sdb2` (see `tests/inputs/README.md`) both print SKIP and pass.

The oracle set holds 7,306 programs: the gate above read 7,308, the programs of the four compressed bundles
(`ekshaders*compress.bin`), and the two the set lacks are those of the init bundle (`ekshadersinitpccompress.bin`), which
the set was not made from. On the oracle set, at the shader model 1-3 lifter: 7,229 programs agree on every run and 77
kill on every run (86,205 runs compared, 30,691 killed in all three); 5,893 of the 7,287 with a float addition disagree
once it is a subtraction.

Its first run on the shader model 1-3 lifter failed 7,283 programs: that lifter had renamed the float constant bank,
which neither the IR text nor the WGSL carries, so the comparison of those two alone had not seen it.

## Three-way execution

Each program runs three ways on the same seeded inputs and constants, with samples answered by the same hash of their
request:

- A: the ecb915c lift (a naga module) in `iridia-exec`;
- B: the IR in `iridia-exec-ir`, by the IR's own definitions;
- C: the new WGSL parsed back by naga's front end and run in `iridia-exec`.

C's module is normalised before it runs, for two things WGSL text cannot carry: the entry argument and result member
names and bindings are the old module's (naga's namer adds a `_` to a name ending in a digit, and WGSL has no way to
say a varying has no interpolation, so its front end supplies the default).

Of 116,928 runs, 85,846 had their outputs compared and 31,082 were killed in all three (a kill compares as the kill).
83 programs kill on every run (such as a `texkill` of a negative constant); for those the three agree that they kill,
and nothing more is compared. The other 7,225 agree in every output lane, bit for bit, on every run.

How sensitive the harness is: with the first placed float addition of each program made a subtraction in B, 5,904
programs disagree. Of those that do not, 57 kill on every run, 1,327 take the changed value through data flow that does
not see it on these inputs (an absolute value, a maximum, a comparison into a select; one was traced by hand), and 20
have no float addition.

## Every WGSL difference, by class

The bodies differ; the declarations do not. Each program's old and new body were reduced to the multiset of operations
they compute (calls and operators, after removing declarations, `var` and `let`, loads, stores, swizzles and type
syntax), and every difference in that multiset was assigned to a class. None is left unassigned.

This classification is a description of the differences, not evidence that the programs compute the same thing. The
lane bookkeeping class is a catch-all: before the multisets are compared, every difference in the count of vector
constructors (`vecN(`) and of the result structure (`Out(`) is dropped, so any change that only adds, removes or
rearranges constructors, including one that puts the wrong lane in a constructor, lands there. 6,889 programs do. What
shows the programs agree is the three-way execution above, which compares output bits.

| Class | Programs | What differs |
| --- | --- | --- |
| Storage form only | 63 | The old lowering keeps a `var` per register and stores each instruction's lanes; the new one is SSA, so values are `let`s or written in place. The operations are the same. |
| Lane bookkeeping | 6,889 | The above, and any difference in the number of vector constructors (not checked further here; see above): a destination mask was a store of some lanes to a `var`, and is now a `vecN(...)` of the lanes kept and the lanes computed. Identity swizzles are folded, and a lane never written is zero. |
| Lane bookkeeping and `mova` rounding | 356 | The above, and `mova`'s rounding (ties away from zero): ecb915c writes `sign(x) * floor(abs(x) + 0.5)`, the IR's `fround.nearestaway` lowers to a truncation and a select of the truncation or the step away from zero. Exactly one `floor` and one `*` fewer, and one `trunc`, `-`, `>=` and `select` more, per `mova`. The two agree except where `abs(x) + 0.5` rounds up in f32 (the largest float below 0.5 gives 1 the old way, 0 the new and documented way); no run met such a value. |

Within these classes, two more differences change the text but not the operations: a relative constant read that
falls on a `def` register selects between the `def` value, the bank row and zero in a different nesting (both give the
`def` value at its register, the row inside the bank and zero outside it), and naga's namer gives different names.

One class was found and fixed rather than kept: at first 83 programs computed some values twice in the new WGSL. naga's
writers never give an access (`.x`, `[i]`) a `let` of its own, because an access of a pointer must stay a place; a lane
of an unnamed expression used twice therefore repeated the whole expression at each use (in some vertex programs the
`a0` conversion, three times). `iridia-naga` now names an access referenced more than once whose base would be written
in place, so each IR value is written once. `tests/lower.rs` pins it.

## Beyond the corpus

- The design's counterexample (§1.4: a loop entered at two points unless a pipeline state is known) is a fixture of
  `iridia-passes`: refused as irreducible while the state is unknown, structured once `specialise` makes it known, and
  run on hand-worked values before and after.
- `structurize` and `break_flags` are checked on 1,500 generated control flow graphs: each one that is reducible is
  structured, routed, lowered to WGSL and run equal to its graph over six seeds.
- Unit fixtures in IR text, with values worked by hand, cover each pass, the executor's NaN, rounding, zero-wins,
  conversion and mask rules, and the text parser's refusals.

## Left for later stages

- Both executors compute `exp2`, `log2`, `sin`, `cos`, `sqrt` and `pow` with the platform's maths library, so a
  transcendental lane may differ in its last bits between platforms. Moving one alone would break A = B, so both move
  together.
- Shader models 1 and 2 on the IR are stage 3; `ps_1_x` follows modern drivers.
