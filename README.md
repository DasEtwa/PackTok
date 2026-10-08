# PackTok

Rust-first tokenizer research implementation. **Current milestone: M4,
factorization ablation and larger-corpus validation (completed).** M1 remains
the flat byte-level BPE control; lexical-v1 and M0–M3 results are frozen.
M4 isolates token sequence from output head on a fixed sourced mixture. It finds
a small tokenizer-sequence benefit at equal updates and an output-head budget
benefit through additional training; it does not establish general superiority.

**Canonical project map:** [DasEtwa/BRAIN/PackTok](https://github.com/DasEtwa/BRAIN/tree/main/PackTok)

## Train and inspect

M3 model architecture and experiment results: [M3_MODEL.md](M3_MODEL.md) and
[M3_MODEL_BENCHMARK.md](M3_MODEL_BENCHMARK.md).
The [current performance and mathematics audit](PERFORMANCE_MATH_AUDIT.md)
records prioritized fixes, numerical gradient checks, and preserved before/after
measurements. Run its focused probes with `cargo run --release -p packtok-bench -- audit`.

The [final pre-GPU correctness pass](PRE_GPU_CODE_REVIEW.md) records the two
model-contract fixes, scoped regressions and CPU baseline freeze verification.
Tokenizer behavior, M3/M4 results and artifact encodings remain frozen.

```text
cargo run --release -p packtok-cli -- train-bpe fixtures/m1_bpe_corpus.txt target/m1.packtok
cargo run --release -p packtok-cli -- inspect target/m1.packtok
cargo run --release -p packtok-cli -- encode --artifact target/m1.packtok "Sämtliche Häuser"
cargo run --release -p packtok-cli -- inspect-token target/m1.packtok 256
cargo run --release -p packtok-cli -- validate target/m1.packtok
cargo run --release -p packtok-bench
cargo run --release -p packtok-bench -- m3 final-run-label

cargo run --release -p packtok-cli -- route "Hello 123!"
cargo run --release -p packtok-cli -- train-packs fixtures/benchmark/train.txt target/m2.packtok
cargo run --release -p packtok-cli -- inspect-pack target/m2.packtok TEXT
cargo run --release -p packtok-cli -- inspect-token target/m2.packtok TEXT:0
cargo run --release -p packtok-cli -- inspect-merges target/m2.packtok TEXT
cargo run --release -p packtok-cli -- stats target/m2.packtok
```

## Workspace

- `packtok-core` — token and pack contracts.
- `packtok-packs` — shared router contract and the experimental `lexical-v1` policy.
- `packtok-format` — deterministic version-1, version-2, and version-3 artifacts.
- `packtok-tokenizer` — M0 byte fallback, frozen M1 BPE, and M2 factorized runtime.
- `packtok-train` — deterministic flat and factorized BPE training, corpus handling, and reference oracles.
- `packtok-model` — small CPU-only causal model with flat and factorized heads for M3.
- `packtok-cli` — training, encoding, decoding, validation, and inspection commands.
- `packtok-bench` — tokenizer benchmarks and the M3 tiny-model comparison harness.

Details: [M2 factorized design](M2_PACKS.md) · [M2 benchmark](M2_PACKS_BENCHMARK.md) ·
[M1 BPE control](M1_BPE.md) · [artifact format](FORMAT.md) ·
[benchmark results](M1_BPE_BENCHMARK.md) · [milestone history](CHANGELOG.md) ·
[M0 historical benchmark](M0_BENCHMARK_BASELINE.md).
The [performance and correctness audit](M1_PERFORMANCE_AUDIT.md) records fixes,
repeated throughput measurements, and the encoder's temporary-memory trade-off.
The [PR #1 review fixes and current results](M1_REVIEW_FIXES.md) cover all seven
Codex findings and the separate synthetic training/evaluation split. Older
benchmark reports retain their original training-like observations.
The [post-fix audit](POST_FIX_AUDIT.md) records three additional fixes, verification
of the previous findings, and extended repetition benchmarks.

Verify with `cargo fmt --all --check`,
`cargo clippy --workspace --all-targets --all-features -- -D warnings`,
`cargo test --workspace`, and `cargo build --release --workspace`.

## M4 factorization ablation

M4 adds the unchanged-tokenizer A/B/C/D CPU experiment and fixed sourced mixed
corpus. Design: [M4_ABLATION.md](M4_ABLATION.md); durable results:
[M4_ABLATION_BENCHMARK.md](M4_ABLATION_BENCHMARK.md). Run preparation with
`cargo run --release -p packtok-bench -- m4-corpus` and comparisons with
`cargo run --release -p packtok-bench -- m4 tiny|large unique-label`.

## M5 GPU Transformer probe (in preparation)

M5 compares only A (M1 flat tokens) and C (canonical flattened M2 tokens) through
the same Transformer and flat head. It preserves the CPU freeze and runs all
preparation in dedicated Ubuntu-24.04 WSL. CUDA/Colab work is isolated and L4
allocation is limited to an explicitly bounded correctness preflight before
separate budget approval. The first runtime preflight failed before Rust started; the L4 was released.
CPU recovery preparation is documented; the user chose to stop at CPU state.
No further GPU work or GPU quality result is claimed.
Design and cost record:
[M5_GPU_TRANSFORMER.md](M5_GPU_TRANSFORMER.md), [M5_GPU_BENCHMARK.md](M5_GPU_BENCHMARK.md).
