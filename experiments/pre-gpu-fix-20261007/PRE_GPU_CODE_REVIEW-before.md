# Pre-GPU code review — 2026-10-07

Reviewed snapshot: `53ef2e8639a1e77c40bb26f2a935bdd321f459c0`, branch
`m4-factorization-ablation`. This is a fresh code review by the same agent that
implemented M4, without delegation. It does not claim an independent reviewer.
The user requested review only: no production implementation, existing tests,
experimental settings, historical reports or artifacts were changed. The new
isolated probe and logs are review evidence, outside the workspace membership.
No GPU experiment or benchmark model retraining was performed.

## Confirmed findings

No P1 issue was confirmed. Both findings below are P2 boundary-validation bugs;
they do not invalidate the retained M4 runs, whose prompts are valid and whose
model artifacts are well below the file cap.

### P2 — A successfully constructed model can exceed the serialization cap

Location: `crates/packtok-model/src/lib.rs:396` (shared construction) and `:975`
(serialization). Construction checks the parameter cap, then allocates weights
and Adam moments, but does not include the artifact header/descriptors in the
file-size check. Serialization subsequently rejects the same model. A caller
can therefore allocate and train an accepted model, only discovering at save
time that its parameters cannot be preserved in the declared model format.

The parameter cap is 16,777,216 f32 values, which alone occupies the entire
67,108,864-byte file cap. A single-pack artifact requires 46 additional bytes.
The read-only probe confirms both paths with seed 19, hidden size 8, context 6:

| Model | Vocabulary rows | Parameters | Required wire bytes | Constructor | Writer |
|---|---:|---:|---:|---|---|
| Flat | 986,888 | 16,777,216 | 67,108,910 | `Ok` | `ModelTooLarge` |
| Factorized, pack ID 7 | 986,887 | 16,777,208 | 67,108,878 | `Ok` | `ModelTooLarge` |

These configurations satisfy the public dimension/vocabulary/parameter limits.
This is a narrow limit mismatch, not evidence that current M4 models fail to
serialize. Suggested future fix: calculate complete serialized size from the
shared layout before allocating model state, and reject incompatible sizes
consistently in construction. Add both-head tests immediately below/above the
effective file-size boundary. No cap increase or format change is needed.

### P2 — Generation can successfully return invalid prompt token IDs

Location: `crates/packtok-model/src/lib.rs:627` (copying the prompt) and `:629`
(truncating context). `greedy_generate` copies the whole prompt without checking
its IDs. Only the suffix used for prediction is validated. An invalid ID earlier
than that suffix is returned unchanged inside an `Ok` result. With zero requested
new tokens, no prompt ID is validated at all. Decoding the returned full sequence
can fail despite successful generation, and invalid/stale model-side IDs are
hidden when they lie outside the active context.

The probe uses hidden/context size 2, seed 19, and vocabulary size 256. A prompt
of three IDs starts with invalid flat `0:256`, followed by two valid `0:97` IDs.
Requesting one new token returns four IDs, including the invalid prefix. The
factorized equivalent uses valid pack 7 and invalid prefix `99:0`, with the same
outcome. Both models reject the invalid ID when passed to `hidden_states` alone.
Both also accept the invalid single-ID prompt when zero new tokens are requested.

Suggested future fix: validate every prompt ID against the model before copying
or predicting. Do not use the existing context-length check on the whole prompt,
because valid long prompts are intentionally supported by sliding the context.
Add tests for invalid prefix IDs, zero-token generation and valid long prompts
for both head kinds. No generation or tokenizer semantic redesign is required.

## Scope and verification

The review traced recurrent forward/backward computation, both cross-entropy
paths, transactional Adam/clipping, input/head parameter layouts, analytical
MACs, model/mapping serialization, M1 synthetic grouping, M2 flattening,
tokenizer encode/decode, corpus preparation, train-only tokenizer provenance,
sampling, held-out scoring and timing boundaries. No additional confirmed
mathematical, mapping, normalization or leakage implementation bug was found.
Existing limitations in M4_ABLATION.md and M4_ABLATION_BENCHMARK.md remain;
this review does not establish GPU suitability or performance.

Fresh verification passed these commands on both platforms:

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo build --release --workspace
```

Windows Rust/Cargo 1.98.1: 125 existing tests passed. Linux WSL on declared MSRV
Rust/Cargo 1.85.0: 126 passed; the additional test is the existing Linux path
fixture. There are no doc tests. The isolated public-API probe reproduced both
findings on both systems. Passing existing tests do not cover these boundary
cases. All 138 entries in the frozen M4 SHA256SUMS.txt matched; no retained file
was updated. Exact environments, command outcomes, counts and raw evidence are
in [the verification summary](experiments/pre-gpu-review-20261007/verification-summary.txt).

Reproduce the review probe from the repository root:

```text
cargo run --release --manifest-path experiments/pre-gpu-review-20261007/Cargo.toml --target-dir target/pre-gpu-review-20261007
```

The separate manifest has no external dependencies; it uses the reviewed local
core/model crates and allocates the two boundary-size models sequentially.
The [probe source](experiments/pre-gpu-review-20261007/probe.rs),
[Windows output](experiments/pre-gpu-review-20261007/probe-windows.txt),
[Linux output](experiments/pre-gpu-review-20261007/probe-linux.txt), and
[Linux verification script](experiments/pre-gpu-review-20261007/verify-linux.sh)
are retained. The script records the existing MSRV/toolchain and target-cache
paths explicitly. Neither finding was patched in this review.
