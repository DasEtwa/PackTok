# M3 — Tiny-model benchmark and experiment record

This file records the M3 model comparison without changing the historical M0,
M1, or M2 tokenizer reports. Final values are added after the run protocol is
implemented and executed. Raw run output and model artifacts are retained in
`experiments/m3-model/`.

## Planned protocol

- Source branch starts at `m2-factorized-packs` commit
  `bb018b9e01bcae47bdc982533822fd5d86001054`; M3 is a stacked branch and PR
  against `m2-factorized-packs`.
- Dataset: manually authored files in `fixtures/m3-model/`, with separate
  train/validation/test files; no M1/M2 tokenizer evaluation samples are used.
- Preprocessing: raw UTF-8 bytes, no normalization, no separator insertion.
- Tokenizer config: M1 flat BPE and M2 lexical-v1, each target 512 logical
  slots (256 byte slots plus at most 256 learned tokens), minimum pair
  frequency 2; both train only on `train.txt`.
- Model: one tanh recurrent layer, hidden size 16, context length 16; M1 flat
  head and M2 pack-plus-conditional-local heads. Detailed equations and
  accounting are in [M3_MODEL.md](M3_MODEL.md).
- Optimizer: Adam, learning rate 0.01, beta1 0.9, beta2 0.999, epsilon
  1e-8, global gradient clipping 1.0; no weight decay, warmup, dropout, or
  mixed precision. Four sampled windows per update; each window has 16 inputs
  and 16 shifted targets. Regime A uses 120 updates. Regime B reaches the M1
  120-update analytical MAC total and may finish after the target by at most
  one full batch. Seeds are 20,261,007 plus the zero-based repetition number;
  window RNG seed is model seed XOR 0xa341316c9e3779b9.
- The sampler uses xorshift state updates (13/7/17) and selects valid window
  starts by modulo. Same seed values do not imply identical raw byte windows
  across tokenizers because token sequences and valid-start counts differ.
- Data splits are rejected if any exact contiguous 32-byte raw passage is shared
  across train, validation, or test. Validation measurements are observational;
  they do not change configurations.
- Exact tokenizer artifact hashes, file byte counts and SHA-256 values, model
  run values, and final environment are recorded in the tables below and
  preserved raw logs.
- Regime A fixes all update settings; regime B targets an equal analytical
  multiply-accumulate budget. Regime B is an estimate, not exact FLOPs.
- At least three seeds/repetitions are required for final comparison. Results
  are shown per run before any summary; incompatible environments are not
  averaged.
- Training wall time includes model construction/initialization, sample-window
  creation, optimizer work, and scheduled validation checkpoints. Test-split
  evaluation is timed separately. Summary variance is population variance.
- Reproduce a full three-repetition run with
  `cargo run --release -p packtok-bench -- m3 <unique-label>`; the label is
  required so an earlier run directory is never overwritten.

## Preserved preflight

`experiments/m3-model/runs/preflight-20261007-1/` is retained as a working-tree
preflight, not a final benchmark. It ran three seeds per regime before the M3
source was committed. The audit found that its `pack_head_logit_fraction`
reported the mean of per-target fractions instead of the aggregate ratio; the
final implementation now reports aggregate pack logits divided by aggregate
pack-plus-local logits. Its `train_ms` also covered optimizer updates only and
omitted initialization, sampling, and scheduled validation; final runs measure
the full training loop as documented above. Preflight metrics remain intact but
must not be mixed into final summaries. The M3 split is unchanged, and no
hyperparameter or tokenizer setting was adjusted based on held-out outcomes.

## Dataset fingerprints and tokenizer artifacts

Pending final deterministic artifact and fixture hash generation.

## Model runs

Pending three final repetitions per regime and tokenizer.

## M2 factorized-head statistics

Pending final evaluation.

## Verification

Pending M3 Windows and Linux/MSRV verification.
