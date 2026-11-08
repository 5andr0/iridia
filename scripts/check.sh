#!/usr/bin/env bash
# CONTRIBUTING.md's full check: fmt, clippy, every test (on the game data GAME_DATA names, where it is set), the doc
# tests and the docs, each step timed; exits non-zero when any step fails, after running them all. The tests run under
# cargo-nextest where it is installed (each test its own process, the test binaries in parallel: about 9 minutes warm on
# the gate configuration, where `cargo test` took 38), otherwise under `cargo test`.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
step() {
  local t0=$SECONDS
  echo "== $*"
  "$@" || { echo "!! failed: $*"; fail=1; }
  echo "== $(( SECONDS - t0 ))s: $*"
}
[ -n "${GAME_DATA:-}" ] || echo "GAME_DATA is not set: the tests that read game data skip"
step cargo fmt --all --check
step cargo clippy --workspace --all-targets --locked -- -D warnings
# tools/wgsl-run is a workspace of its own (wgpu stays out of the lockfile): checked by its manifest
step cargo fmt --manifest-path tools/wgsl-run/Cargo.toml --check
step cargo clippy --manifest-path tools/wgsl-run/Cargo.toml --all-targets --locked -- -D warnings
if cargo nextest --version >/dev/null 2>&1; then
  step cargo nextest run --workspace --locked --release
  step cargo test --workspace --locked --release --doc
else
  echo "cargo-nextest is not installed: the tests run under cargo test, one binary at a time"
  step cargo test --workspace --locked --release --no-fail-fast
fi
RUSTDOCFLAGS="-D warnings" step cargo doc --workspace --no-deps --locked
exit $fail
