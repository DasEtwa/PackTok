# PackTok

Rust-first tokenizer research implementation. **Current milestone: M1, a flat
byte-level BPE baseline.** It provides a deterministic trainer, exact byte
round-trips, a versioned artifact, an inspection CLI, and a benchmark against the
M0 byte-only path. It is not yet PackTok's factorized pack architecture.

**Canonical project map:** [DasEtwa/BRAIN/PackTok](https://github.com/DasEtwa/BRAIN/tree/main/PackTok)

## Train and inspect

```text
cargo run --release -p packtok-cli -- train-bpe fixtures/m1_bpe_corpus.txt target/m1.packtok
cargo run --release -p packtok-cli -- inspect target/m1.packtok
cargo run --release -p packtok-cli -- encode --artifact target/m1.packtok "Sämtliche Häuser"
cargo run --release -p packtok-cli -- inspect-token target/m1.packtok 256
cargo run --release -p packtok-cli -- validate target/m1.packtok
cargo run --release -p packtok-bench
```

## Workspace

- `packtok-core` — token and pack contracts.
- `packtok-format` — deterministic version-1 and version-2 artifacts.
- `packtok-tokenizer` — M0 byte fallback and M1 BPE runtime.
- `packtok-train` — deterministic BPE training, corpus handling, and reference oracle.
- `packtok-cli` — training, encoding, decoding, validation, and inspection commands.
- `packtok-bench` — same-input M0/M1 throughput and token-count measurements.

Details: [M1 BPE design](M1_BPE.md) · [artifact format](FORMAT.md) ·
[benchmark results](M1_BPE_BENCHMARK.md) · [milestone history](CHANGELOG.md) ·
[M0 historical benchmark](M0_BENCHMARK_BASELINE.md).

Verify with `cargo fmt --all --check`,
`cargo clippy --workspace --all-targets --all-features -- -D warnings`,
`cargo test --workspace`, and `cargo build --release --workspace`.
