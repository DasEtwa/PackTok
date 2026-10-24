# Targeted runner regression review — 2026-10-24

Scope: `src/runner.rs`, following the independent Astra review. No GPU, CUDA
build, or large-model allocation was used.

## Reproduced findings

1. **Resume JSONL corruption (medium).** When an interrupted metrics write left
   a complete JSON object without its final LF, `reconcile_metrics` accepted and
   copied that object, then appended `resume_boundary` on the same line. Parsing
   the resumed metrics failed with `trailing characters`. The new regression test
   failed before the fix. Reconciliation now supplies the missing separator and
   leaves the archived original byte-for-byte unchanged.
2. **Incorrect hardware admission (medium).** The standalone Rust CLI used
   `contains("L4")`, which admitted `NVIDIA L40` and `NVIDIA L40S`. The notebook's
   Python gate already used exact names. The extracted pure Rust parser now
   accepts only `NVIDIA L4` and `NVIDIA L4 GPU`; malformed fields, missing or
   multiple GPU rows, invalid numeric fields and other names are rejected.
   The new regression test failed on `NVIDIA L40` before the fix.
3. **Incomplete resumed training history (medium).** An existing, valid JSONL
   file with only train step 1 was accepted for a saved checkpoint at step 2.
   The regression test confirmed this before the fix. Reconciliation now
   requires train-stage rows 1 through the saved step exactly once and in order,
   with positive integer step values. Numeric train rows beyond the checkpoint
   remain an abandoned tail and are filtered out. Missing rows, gaps,
   duplicates, out-of-order rows and malformed train steps fail before replacing
   the original log; the original archive remains available. Other stages keep
   their existing treatment, and step-zero resume can discard an old train tail.

## Validation

Using Rust 1.90.0 and the committed lockfile:

```sh
cargo +1.90.0 test --locked --manifest-path experiments/semantic-scale/Cargo.toml runner::tests::
cargo +1.90.0 test --locked --manifest-path experiments/semantic-scale/Cargo.toml
cargo +1.90.0 fmt --manifest-path experiments/semantic-scale/Cargo.toml -- --check
cargo +1.90.0 clippy --locked --manifest-path experiments/semantic-scale/Cargo.toml --all-targets -- -D warnings
```

Results: all eight runner tests pass, including four new regressions; all 29
library tests and the CLI integration test pass. Formatting and Clippy pass.
This evidence checks CPU fixtures and admission parsing, not GPU runtime behavior.
