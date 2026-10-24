# Pre-change baseline verification

This directory records the original repository at the full revision in
`source-commit.txt`, before Semantic Packs changes. Verification ran against a
`git archive HEAD` snapshot at `/tmp/packtok-semantic-baseline`, so concurrent
feature implementation could not affect the baseline. No existing branch was
modified; the active checkout was `feat/semantic-packs-v1` throughout.

`environment.txt` records the Linux host, CPU, OS, Rust and Cargo versions.
The root workspace has no external Rust crate dependencies. Rust 1.85.0,
rustfmt and clippy were installed using the official rustup installer; the
installation transcript is `rust-setup.log`. The installation used task-local
`CARGO_HOME=/workspace/.packtok-cargo` and
`RUSTUP_HOME=/workspace/.packtok-rustup` and retained the inherited proxy and CA
configuration. No GPU work or model training was run.

From the original snapshot, run these commands with Rust 1.85.0 and preserve
stdout/stderr separately as named here:

```sh
cargo +1.85.0 fmt --all --check
cargo +1.85.0 clippy --workspace --all-targets --all-features -- -D warnings
cargo +1.85.0 test --workspace
cargo +1.85.0 test --workspace --release
cargo +1.85.0 build --release --workspace
```

`status.txt` contains the exit status of each command. Debug and release logs
contain individual test names and per-target totals. There are 135 baseline
tests; doc-test targets contain no tests. This baseline is a correctness/static
verification record, not a Semantic Packs speed comparison.
