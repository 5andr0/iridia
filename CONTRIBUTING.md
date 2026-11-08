# Contributing

## Build and test

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

Rust 1.99 or newer (`rust-toolchain.toml` pins it). The tests need no game files: the fixtures are small compiled shaders and the HLSL they were built from, in each crate's `tests/fixtures`. Tests that read files this repository cannot ship skip with one line naming the input; point them at your own game files with a `GAME_DATA` file in the [primase-testdata](https://github.com/Kalmarv/primase-testdata) layout, and each crate's `tests/inputs/README.md` lists what its tests read.

The examples in the crate READMEs are checked: each crate's README is run as a doc test.

## Before pushing

Run the full check on the gate configuration, then the slim gate below:

```sh
GAME_DATA=<the gate configuration> scripts/check.sh
```

`scripts/check.sh` runs fmt, clippy, every test, the doc tests and the docs, each step timed, and fails when any step
does. With [cargo-nextest](https://nexte.st) installed (`cargo install cargo-nextest`) it runs the tests as
`cargo nextest run --workspace --locked --release` (each test its own process, the test binaries in parallel;
`.config/nextest.toml` names the slow ones) and the doc tests, which nextest does not run, as
`cargo test --workspace --locked --release --doc`: about 9 minutes warm on the gate configuration, where one
`cargo test` took 38. Without it, `cargo test --workspace --locked --release`. Tests in parallel processes are safe: each
takes its own scratch folder (`own_scratch`), and the WARP oracle is held by one test at a time (`warp_exclusive`, a
machine-wide file lock). Without `GAME_DATA` the tests that read game data skip, as on CI.

Run the slim shader set through every lifter and quote its counts in the push's report:

```sh
GAME_DATA=<the gate configuration> cargo test --release --locked -p iridia --test slim -- --nocapture
```

The gate configuration (`[run] require = 'gate'`) needs `oracles.corpus.general/slim`, so a missing set fails instead of
skipping. `iridia/tests/slim.rs` takes each of the set's 8,287 DXBC, DXIL and Direct3D 9 programs through the older lift
and the IR lift of its family, writes WGSL and reads it back with naga, and prints the IR's text and parses it back. It
fails on a panic, or on a program that goes further or less far than `iridia/tests/slim/<family>.tsv` says. When a lift
widens, replace that table with the current one (the failure names its path) in the same commit. GitHub CI has no game
data and skips it.

A change that moves what the coverage table (`docs/coverage`) records, a lifter's, `iridia-exec`'s or the runners' in
`tools/coverage`, makes the table again and commits it:

```sh
cargo run --release --locked -p iridia-coverage -- table docs/coverage <offload-golden-images checkout>
```

It needs no game data: the suites it runs are in `tools/coverage/suites`, programs included, and the golden images are a
checkout of [llvm/offload-golden-images](https://github.com/llvm/offload-golden-images) at the commit
`tools/coverage/suites/offload/README.md` names. It runs each case file through the IR, which it grades, and through the
lift the IR replaces, each row's reference. CI builds the tool once, runs those parts side by side, the longer files in
shards (`iridia-coverage part ... --shard`), makes the table of them (`table --parts`) and fails where it differs from the
committed one.

## No I/O in libraries

The library crates read the bytes and text they are given and never touch files, the environment or processes: `clippy.toml` disallows those calls and every `lib.rs` denies the lint. Examples and tests may.

## Changing what a lifter produces

A cache keyed by a program's bytes trusts `LIFTER_VERSION` (each reader) and `PROGRAM_VERSION` (`iridia`) to know that a lifted program is stale. Each reader's `tests/versions.rs` (or `*versions.tsv` in `iridia-d3d9`, `tests/lift.rs` in `iridia-mojo`) hashes what it makes of its fixtures and compares the table with the committed one (`tests/ir-lifted.hashes` for the IR lifts). If you change what a lifter makes of an input, the test fails and says which version to bump; after the bump it fails once more until you replace the committed table with the current one, whose path the failure names.

## Releasing

Publishing is paused until 0.7.0, the release with no naga fork (`release = false` in `release-plz.toml`): no crate is
published, whatever its version.

1. Raise the version in the workspace and in the crates' dependencies on each other (a patch, 0.x.Y, for fixes and data
   or manifest additions; a minor, 0.X.0, for new API or any breaking change before 1.0; then `cargo check`, so
   `Cargo.lock` follows) and put the change's CHANGELOG lines under a `## <version> (<date>)` heading, in one commit.
2. Push it to main. Once every check has passed, release-plz publishes each crate whose version crates.io does not have
   yet (trusted publishing: no token is stored anywhere), tags the commit `v<version>` and creates the GitHub Release
   with that CHANGELOG section as its notes.

A commit that raises no version only runs CI. Never release by hand.

A new crate's first version can't come from trusted publishing: the token holder publishes it by hand, then adds its
trusted-publishing config (workflow `ci.yml`).

## Reporting a file it cannot read

Open an issue with the "unreadable file" template, for a shader that is refused when it should lift, panics, fails validation, or lifts to something that does not behave as the original. It asks for:

- the version of the crates (or the commit) and your platform;
- which reader you used (`iridia-dxbc`, `iridia-dxil`, `iridia-d3d9`, `iridia-mojo`) and the call (`lift`, `reflection`, `Executable::new`...);
- the exact message: a refusal says the bytecode, the unit (word, bit or line), the offset and the reason; a validation failure lists its problems;
- the shader's target and compiler (`ps_5_0` from `fxc`, `cs_6_6` from `dxc`...) and how you got it: which game or engine, if any, and whether the shader came compiled with it;
- if you can write one, the smallest HLSL (or MojoShader GLSL) that still fails, and the command that compiled it.

Do not attach game files, shader containers extracted from a game, archives or keys: they are not ours to redistribute and the report does not need them. Describe the file instead: its size, its target, the refusal's text, and what Microsoft's disassembler (`fxc /dumpbin`) shows for the instruction at the offset the refusal names. A small HLSL file you wrote and compiled yourself is welcome.

## Changing the code

Keep the public items documented, refusals naming the offset and what was expected, and add a test next to the behaviour you change. Prefer a fixture built from HLSL in `tests/fixtures` (with the compile command in its header) to a binary you cannot reproduce.
