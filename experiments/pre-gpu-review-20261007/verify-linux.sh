#!/usr/bin/env bash
set -euo pipefail
cd /mnt/c/Users/DasEtwa/PackTok
review_dir=experiments/pre-gpu-review-20261007
toolchain_bin=/var/tmp/packtok-pr1-rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin
export PATH="$toolchain_bin:$PATH"
export CARGO_TARGET_DIR=/var/tmp/packtok-m4-target
{ git rev-parse HEAD; rustc --version; cargo --version; uname -sr; } > "$review_dir/environment-linux.txt"
cargo fmt --all --check > "$review_dir/fmt-linux.txt" 2>&1
cargo clippy --workspace --all-targets --all-features -- -D warnings > "$review_dir/clippy-linux.txt" 2>&1
cargo test --workspace > "$review_dir/tests-linux.txt" 2>&1
cargo build --release --workspace > "$review_dir/release-build-linux.txt" 2>&1
cargo run --release --manifest-path "$review_dir/Cargo.toml" --target-dir /var/tmp/packtok-pre-gpu-review-probe > "$review_dir/probe-linux.txt" 2>&1
printf '%s\n' 'Linux MSRV verification and review reproductions passed.'
