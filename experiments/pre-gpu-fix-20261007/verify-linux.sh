#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/DasEtwa/PackTok
review_dir=experiments/pre-gpu-fix-20261007
toolchain_bin=/var/tmp/packtok-pr1-rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin
export PATH="$toolchain_bin:$PATH"
export CARGO_TARGET_DIR="$PWD/target/linux-pre-gpu-fix-20261007"
logdir="$review_dir/linux-final"
mkdir "$logdir"
{ git rev-parse HEAD; git status --short; rustc --version; cargo --version; uname -sr; } > "$logdir/environment.txt"
cargo fmt --all --check > "$logdir/fmt.txt" 2>&1
cargo clippy --workspace --all-targets --all-features -- -D warnings > "$logdir/clippy.txt" 2>&1
cargo test --workspace > "$logdir/tests-debug.txt" 2>&1
cargo test --workspace --release > "$logdir/tests-release.txt" 2>&1
cargo build --release --workspace > "$logdir/build-release.txt" 2>&1
cargo test -p packtok-model serialization_ > "$logdir/serialization.txt" 2>&1
cargo test -p packtok-model generation_validation_ > "$logdir/generation.txt" 2>&1
cargo test -p packtok-model recurrent_and_both_head_gradients_match_finite_differences > "$logdir/gradients.txt" 2>&1
cargo test -p packtok-model adam_ > "$logdir/optimizer.txt" 2>&1
cargo run --release --manifest-path "$review_dir/Cargo.toml" --target-dir "$PWD/target/linux-pre-gpu-fix-compatibility" > "$logdir/compatibility.txt" 2>&1
printf '%s\n' 'Linux MSRV debug/release checks and compatibility probe passed.'
