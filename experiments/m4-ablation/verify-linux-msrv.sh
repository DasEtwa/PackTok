#!/usr/bin/env bash
set -euo pipefail
# Arguments are an installed toolchain bin directory and isolated Cargo target.
toolchain_dir=${1:?installed Rust 1.85.0 bin directory required}
target_dir=${2:?isolated Linux Cargo target directory required}
export PATH="$toolchain_dir:$PATH"
export CARGO_TARGET_DIR="$target_dir"
cd "$(dirname "$0")/../.."
logdir=experiments/m4-ablation/verification
rustc --version --verbose > "$logdir/linux-environment.txt"
cargo fmt --all --check 2>&1 | tee "$logdir/linux-fmt.txt"
cargo clippy --workspace --all-targets --all-features -- -D warnings 2>&1 | tee "$logdir/linux-clippy.txt"
cargo test --workspace 2>&1 | tee "$logdir/linux-tests.txt"
cargo build --release --workspace 2>&1 | tee "$logdir/linux-build.txt"
