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

## Final pre-GPU fix pass — both findings resolved

The section above is the original pre-fix review. Its byte-exact before snapshot
is retained as [PRE_GPU_CODE_REVIEW-before.md](experiments/pre-gpu-fix-20261007/PRE_GPU_CODE_REVIEW-before.md),
matching the original review checksum. All original probes, outputs and failed
attempts remain. This resolution was appended only after complete verification.

**Both P2 findings are resolved.** The production change is limited to
`crates/packtok-model/src/lib.rs`; no tokenizer, model arithmetic, initialization,
optimizer, MAC definition, experiment harness, corpus or split changed.

For serialization, `serialized_model_size` is the single checked calculation
used by construction, loading and writing. The existing format's complete size
is `40 + 6*pack_count + 4*parameter_count` bytes, including one declaration for a
flat model. Construction rejects excessive complete size before weights/Adam
allocation; the loader agrees on complete/body sizes. The writer's field order
and bytes are unchanged. No version or cap increased. Multiplication and addition
overflow returns `LengthOverflow`; excessive representable size returns
`ModelTooLarge`. The normative rule is in [FORMAT.md](FORMAT.md).

For generation, the existing context ID-validation loop is factored into
`validate_token_ids`, which still uses `embedding_row` as the canonical pack/local
lookup. Greedy generation validates the whole supplied prompt before copying or
truncating it, including zero-continuation requests. Empty, overflow and excessive
length checks retain their precedence before ID scanning; every successful
result has validated prompt IDs. Long valid prompts still use the trailing
context, and continuation arithmetic/tie-breaking remain unchanged.

### Added regressions and exact fixtures

Nine model tests were added:

- `serialization_size_boundary_includes_flat_and_factorized_headers` — maximum
  complete sizes, the first excessive parameter, exact-cap four-pack case and
  every supported declaration count.
- `serialization_size_checked_arithmetic_rejects_overflow` — multiplication and
  addition overflow in descriptor/body accounting.
- `serialization_constructors_reject_unwritable_boundary_layouts` — both original
  reproductions, a four-pack near-boundary rejection, and nearest smaller layouts
  reaching body validation without large allocations.
- `serialization_constructor_success_obeys_complete_size_rule` — representative
  accepted flat/factorized models fully serialize/load with independent byte counts.
- `serialization_preserves_historical_m3_m4_model_bytes` — embedded canonical M3
  flat/factorized and M4 synthetic/flattened model artifacts retain their exact bytes.
- `generation_validation_rejects_flat_ids_in_entire_prompt` — invalid pack/local
  IDs at every prompt position, outside and inside the context, with zero/one new IDs.
- `generation_validation_rejects_factorized_pack_and_local_ids` — equivalent
  unknown-pack and invalid-local cases using canonical errors and original indices.
- `generation_validation_keeps_valid_long_and_zero_continuation_behavior` — exact
  zero-continuation identity and pre-fix golden generated IDs for both heads.
- `generation_validation_preserves_empty_overflow_and_length_errors` — both heads,
  empty input, checked length overflow, exact returned-length cap and first excess.

Equivalent boundary calculations plus representative constructors are used
instead of allocating several maximum-size model states in concurrent tests.
Single-pack maximum wire-compatible parameter count is 16,777,204 (67,108,862
bytes); four packs allow 16,777,200 and exactly 67,108,864 bytes. Further fixture
configs, seeds, golden IDs and failed probe/launch attempts are documented in
[NOTES.md](experiments/pre-gpu-fix-20261007/NOTES.md).

### Complete verification and historical identity

| Platform/toolchain | Workspace debug tests | Workspace release tests | fmt / strict Clippy / release build |
|---|---:|---:|---|
| Windows, Rust/Cargo 1.98.1 | 134 passed | 134 passed | passed |
| Linux WSL, MSRV Rust/Cargo 1.85.0 | 135 passed | 135 passed | passed |

All failures are zero; doc-test targets contain zero tests. Linux's extra test
is the existing literal-backslash path fixture. Focused serialization, generation,
all-parameter numerical-gradient and Adam atomicity checks also passed on both
systems. Commands, environments, raw logs and totals are retained under
[experiments/pre-gpu-fix-20261007/](experiments/pre-gpu-fix-20261007/).

All 20 entries of the M3 final manifest, all 22 entries of its audit manifest,
all 138 entries of the frozen M4 manifest and all 19 original review entries
match their hashes. The old review root-document entry is checked against its
preserved before snapshot. No historical artifact or result file changed.

The same read-only compatibility program was run before and after the fixes.
It reloaded and reserialized all 72 canonical M3 final/audit and M4 tiny/larger
models and six tokenizer files byte-exactly. The original 12 tiny M4 A/D versus
M3 historical model identity pairs were repeated and are identical. Its 93
structured output lines, including saved-model valid generation and pre-fix
golden generation, match before/after on Windows and on Linux (only path separators
are canonicalized for cross-platform comparison). No research retraining was
needed for these validation-only changes. Evidence:
[historical-hashes.txt](experiments/pre-gpu-fix-20261007/historical-hashes.txt),
[verification-summary.txt](experiments/pre-gpu-fix-20261007/verification-summary.txt)
and the preserved before/after compatibility logs.

### Narrow adversarial self-review and freeze

The final review checked only the affected contracts: every fixed/wire field,
both head kinds, checked products/sums, equality at the cap, next-parameter
rejection, validation before model-state allocation and constructor/loader/writer
agreement. Generation checks covered ignored prefixes, zero new IDs, flat and
pack/local domains, original indices, empty input, checked/excessive lengths,
validation order, valid long prompts and valid returned IDs. No directly related
additional correctness issue was confirmed. This remains a self-review.

Verified model source is identified byte-exactly by
[model-source-sha256.txt](experiments/pre-gpu-fix-20261007/model-source-sha256.txt)
and [model-source.patch](experiments/pre-gpu-fix-20261007/model-source.patch)
against the original review commit. The exact CPU freeze commit is recorded in
[DELIVERY.md](experiments/pre-gpu-fix-20261007/DELIVERY.md). All changed paths are
listed in [changed-files.txt](experiments/pre-gpu-fix-20261007/changed-files.txt).
The CPU baseline is frozen with the existing format and
experimental history; the readiness gate authorizes considering M5, not starting it.

`READY_FOR_M5_GPU`
