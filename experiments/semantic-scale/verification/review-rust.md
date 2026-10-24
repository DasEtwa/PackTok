# Independent Rust correctness review

Reviewer: separate `scale_review_rust` agent, 2026-10-23. Read-only inspection of
`config.rs`, `runner.rs`, `model.rs`, `optim.rs`, `data.rs` and the locked Candle
source. No production code, branch or commit was changed. The primary agent was
still integrating notebook/storage controls during this review.

## RR-1 — medium: accepted one-step plan cannot pass preflight

Original locations: `TrainPlan::validate` accepted `steps=1`;
`AdamConfig::learning_rate_at` applies cosine decay to zero at the only update;
`preflight` requires a changed model fingerprint.

Reproduction used the real binary, a tiny CPU F32 model, explicit independent
train/validation/test fixtures, context 4, batch/accumulation 1, `--steps 1`:

```text
prepare ...                                               exit 0
plan --preset tiny --context 4 --dtype f32 --steps 1 ...    exit 0
preflight --config ... --output ... --cpu                  exit 1
error: preflight optimizer failed to update model weights
```

Exact commands/output are in `review-rust-one-step-before.txt`. This wastes a
valid-looking run configuration and would cause the same schedule failure on
GPU. Expected: reject the unsupported one-step plan before model allocation, or
provide a defined nonzero one-step learning-rate schedule.

The primary agent changed plan validation to require at least two steps after
this finding. A focused follow-up verifies that the old one-step plan is now
rejected before allocation.

## Static CUDA/BF16 and checkpointing assessment

The locked Candle core 0.9.1/Candle kernels 0.9.2 sources include BF16 GEMM,
GELU-erf/unary/binary and index-select/index-add/scatter-add paths. BF16 kernels
are enabled at SM >= 80, consistent with the notebook's L4 SM 89 target.
No specific unsupported CUDA operation was found in this static inspection.
This is not evidence of successful CUDA compilation/execution or measured VRAM.
A tiny BF16 CPU probe returns `unsupported dtype BF16 for op matmul`, as expected
for this CPU backend; BF16 checkpoint serialization tests do not imply BF16 CPU
training support. The runner explicitly requires F32 for CPU diagnostics.

Manual layer VJP/recompute logic was inspected: detached layer inputs are saved,
head input gradients propagate through reverse layer recomputation, each
parameter gradient is collected once per recomputed stage, and microbatch scale
is applied only to collected parameter gradients. The unscaled upstream VJP is
therefore not repeatedly attenuated by the accumulation factor. Semantic NONE
is masked to zero but still participates in a complete zero-gradient parameter
set. Inference detaches each layer to avoid retaining all activation graphs.

Optimizer checks complete gradient names/layouts/device/finite global norm before
weight updates. FP32 masters preserve BF16 updates, tensor-wise updates bound
transient memory, and F32 masters reuse model storage. Checkpoint writes publish
only a complete generation; first-pass SHA/layout/value/master-rounding
validation precedes model changes, with a second hash check on loading. An
allocation or file-race failure during the second device pass is fatal to that
in-memory run, while the last disk generation remains intact. Dataset records
are bounded and batch windows do not wrap internally.

## Operational observations

- The original preflight checks its soft deadline only at the end. Stage-boundary
  checks and a notebook process timeout were recommended to keep the initial
  preflight bounded despite slow checkpoint I/O. Individual CUDA operations or
  filesystem calls still cannot be synchronously interrupted by Rust alone.
- The original stale-PASS comment was stronger than behavior: validation/device
  failure happened before deletion of a previous marker. Deleting the marker at
  command start was recommended for strict failed-rerun semantics.
- The primary agent already identified and assigned window-local semantic
  recomputation, sampler-counter verification and local checkpoint retention;
  those known integration changes were not counted as new independent findings.

## Verification

`cargo +1.90.0 test --manifest-path experiments/semantic-scale/Cargo.toml --lib`
passed all 21 tests present at the initial pass, in 0.13 seconds. Log:
`review-rust-tests-initial.log`. A separate tiny BF16 CPU probe is retained as
`review-rust-bf16-cpu.log`. No GPU was allocated or executed; no 100M/1B model was
instantiated. No reproduced panic, omitted/doubled parameter gradient or unsafe
index behavior was found in the reviewed paths. Final integration gates and
actual Colab CUDA behavior remain the primary agent's and notebook's checks.

## Focused follow-up — RR-1 closed

The rebuilt binary rejects the original one-step plan with
`invalid versioned scale plan; expected explicit bounded preset/run values`.
It does so before creating the run output directory or allocating a model.
Exact error: `review-rust-one-step-after.txt`. The plan now requires >=2 steps.
No confirmed critical/high Rust defect remains open from this pass.
