# M3 — Tiny-model benchmark and experiment record

This is the M3 model comparison. It does not rewrite the historical M0/M1/M2
reports and does not establish general language-model quality. M1 remains the
flat-BPE control; M2 artifacts and routing were consumed without changes.

## Protocol and provenance

- Branch: `m3-tiny-model`, based on M2 commit
  `bb018b9e01bcae47bdc982533822fd5d86001054`. The implementation tested by the
  final run is commit `2603f9613b947e7933c11daba5fa168eeda1905c`.
- Final command: `cargo run --release -p packtok-bench -- m3 final-20261007`.
  The harness runs three seeds in both comparison regimes and refuses to
  overwrite an existing run directory.
- Final raw output, environment, twelve serialized model files, and SHA-256
  manifest are preserved in
  [`experiments/m3-model/runs/final-20261007/`](experiments/m3-model/runs/final-20261007/).
- Machine: Intel Core i7-2600 @ 3.40 GHz, 8 logical CPUs; Windows 10 Pro build
  19045, x86_64 MSVC; release profile; Rust/Cargo 1.98.1. Each run used one
  explicit training/evaluation thread. Full compiler details are in
  `environment.txt`.
- Model: one causal tanh recurrent layer, hidden size 16, context 16, f32
  parameters and gradients, CPU. Adam learning rate 0.01, beta1 0.9, beta2
  0.999, epsilon 1e-8, global gradient clip 1.0; four windows per update,
  sixteen inputs and shifted targets per window. No warmup, weight decay,
  dropout, or mixed precision. Regime A uses 120 updates; regime B runs to the
  flat model's analytical MAC budget, ending only after a full batch. Seeds are
  20,261,007–20,261,009.
- M1 and M2 tokenizers each target 512 logical slots: 256 byte tokens plus at
  most 256 learned tokens. Both train only on the M3 training split with minimum
  pair frequency 2. M2 uses the frozen `lexical-v1` router.
- Both tokenizer artifacts were independently trained twice on the same raw
  training bytes; each pair of serialized outputs was byte-identical. The raw
  log records this check.
- Training wall time includes model construction/initialization, sampling,
  optimizer updates, and scheduled validation checkpoints. Evaluation time
  covers model scoring only; tokenization and construction of evaluation
  examples are outside that timer. The process peak is a process-wide high-water
  mark. The parameter/Adam/gradient storage estimate is 16 bytes per parameter;
  it excludes activations, windows, temporary logits, and allocator overhead.
- Summary standard deviations and variances are population statistics across
  the three seeds. No incompatible runs are averaged. The held-out split and
  model/tokenizer settings were not tuned after viewing its outcomes.

## Dataset, splits, and tokenizer artifacts

The manually authored synthetic corpus is checked in under the repository's
Apache-2.0 license. One raw UTF-8 file belongs to each split; there is no
normalization, separator insertion, downloaded data, or external annotation.
The harness rejects an exact contiguous 32-byte passage shared between splits.
The 32-byte guard is a leakage check, not a statistical guarantee that this
small synthetic test estimates natural language.

| Split | Files | Raw bytes | SHA-256 | M1 tokens | M1 bytes/token | M2 tokens | M2 bytes/token |
|---|---:|---:|---|---:|---:|---:|---:|
| Train | 1 | 2,605 | `010788ec0642e48074798cd26b4b2d16f1f049956afcd23046a6f7ec61efa733` | 872 | 2.9874 | 1,213 | 2.1476 |
| Validation | 1 | 604 | `b0c69ac46dbd289322b0dc79f17eeb24fdc6454cae98ca8f4b35abd24024d795` | 375 | 1.6107 | 418 | 1.4450 |
| Test | 1 | 614 | `ff73816c673075cdfe48e1dba6297952bfdcc129bb174f3d02d3560b9d4e51ec` | 397 | 1.5466 | 425 | 1.4447 |

The M1 format-v2 artifact is 3,949 bytes, SHA-256
`41a3740e63e8fd86a8f338ba0d2cb17dd86076b91d3d4fb9fd858c27996e19eb`.
The M2 format-v3 artifact is 4,464 bytes, SHA-256
`907604b79cd41fa5962b1861893382a112feffa7d09fe5fb4b047f8f445de40f`.
Both realized the full 512-slot budget. M1 has 256 learned merges. M2's 256
learned tokens were allocated as TEXT 253, NUMBER 0, STRUCTURE 3, with 256
shared byte IDs. In the M3 model, zero-merge NUMBER has no local output table;
its digits are represented through the shared byte fallback. The complete
artifact, corpus, model, and raw-log hashes are in
[`SHA256SUMS.txt`](experiments/m3-model/runs/final-20261007/SHA256SUMS.txt).

The differing split token counts are part of the result: on this held-out text,
M2 emitted 425 tokens and M1 emitted 397. This comparison holds the logical
vocabulary budget constant, not the sequence length.

## Results

Each cell below is mean ± population standard deviation across three seeds.
Loss/token is tokenizer-specific. Bits/byte and NLL/byte normalize by the raw
bytes represented by scored targets; the first token in each split has no
preceding context and is excluded. Perplexity is present per seed in the raw
log, but should not be compared across tokenizations.

| Regime | Model | Parameters | Sampled train targets / raw target bytes | Validation loss/token | Validation bits/byte | Test loss/token | Test NLL/byte | Test bits/byte | Train wall time (ms) | Test scoring (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A: same backbone | M1 flat | 17,424 | 7,680 / 23,112.7 ± 124.5 | 6.9019 ± 0.1390 | 6.2068 ± 0.1250 | 7.3649 ± 0.1567 | 4.7655 ± 0.1014 | 6.8752 ± 0.1462 | 318.436 ± 22.628 | 11.416 ± 0.296 |
| A: same backbone | M2 factorized | 17,475 | 7,680 / 16,434.3 ± 116.9 | 6.0388 ± 0.0772 | 6.0449 ± 0.0773 | 6.0106 ± 0.0550 | 4.1642 ± 0.0381 | 6.0077 ± 0.0550 | 200.654 ± 7.970 | 9.007 ± 0.137 |
| B: matched estimated MACs | M1 flat | 17,424 | 7,680 / 23,112.7 ± 124.5 | 6.9019 ± 0.1390 | 6.2068 ± 0.1250 | 7.3649 ± 0.1567 | 4.7655 ± 0.1014 | 6.8752 ± 0.1462 | 308.802 ± 14.989 | 11.309 ± 0.213 |
| B: matched estimated MACs | M2 factorized | 17,475 | 15,936 / 34,199.3 ± 111.7 | 6.0635 ± 0.1670 | 6.0695 ± 0.1672 | 6.0983 ± 0.1878 | 4.2250 ± 0.1301 | 6.0953 ± 0.1877 | 398.220 ± 5.644 | 9.702 ± 1.475 |

Parameter partitions are constant across seeds. M1 has 8,192 embedding,
528 backbone, and 8,704 output-head parameters. M2 has 8,192 embedding,
528 backbone, and 8,755 output-head parameters. The 51-parameter difference
comes from the factorized head; neither model includes optimizer state in its
parameter count.

| Regime | Model | Updates | Estimated MACs per run |
|---|---|---:|---:|
| A | M1 flat | 120 | 194,519,040 each |
| A | M2 factorized | 120 | 93,874,368–93,959,712 |
| B | M1 flat | 120 | 194,519,040 each |
| B | M2 factorized | 249 | 194,691,360–195,058,848 |

The MAC count is an analytical matrix-operation estimate defined in
[M3_MODEL.md](M3_MODEL.md), not measured FLOPs. Regime B's M2 estimate averages
194,836,320 MACs and overshoots the M1 target slightly because updates stop
only after a complete batch. Peak process memory was 6,021,120 bytes for the
full harness. Estimated parameter, Adam-moment, and gradient storage was
278,784 bytes for M1 and 279,600 bytes for M2. Model files were 69,742 bytes
(M1) and 69,958 bytes (M2) per seed.

### Test-seed variance for the primary metric

| Regime/model | Mean test bits/byte | Population SD | Population variance |
|---|---:|---:|---:|
| A / M1 flat | 6.87520175 | 0.14623620 | 0.021385027546 |
| A / M2 factorized | 6.00768927 | 0.05498394 | 0.003023233542 |
| B / M1 flat | 6.87520175 | 0.14623620 | 0.021385027546 |
| B / M2 factorized | 6.09533770 | 0.18774971 | 0.035249953409 |

The exact per-seed loss, accuracy, timing, estimated MAC count, validation
curve, and generated byte sequence are in `run-summary.txt`.

## M2 factorized-head statistics

The test split has 424 scored next-token targets. Their pack counts are
TEXT 136, STRUCTURE 14, and shared BYTE_FALLBACK 274; NUMBER contributes no
targets because it learned no merge. The byte-fallback target share is
274/424 = 64.62%. Encoded test sequences cross pack boundaries 225 times.
The mean active local head contains 246.684 logits. The ratio of aggregate
pack logits to aggregate pack-plus-local logits is 1.2015%; the local head
accounts for 98.7985% of those logits. This is a logit-count proxy, not measured
hardware work.

| Regime | Pack prediction accuracy | Local accuracy given gold pack | TEXT pack accuracy | STRUCTURE pack accuracy | BYTE_FALLBACK pack accuracy |
|---|---:|---:|---:|---:|---:|
| A: same backbone | 0.5165 ± 0.1408 | 0.2382 | 0.4412 ± 0.4148 | 0.0000 | 0.5803 ± 0.4230 |
| B: matched estimated MACs | 0.5904 ± 0.0011 | 0.2634 ± 0.0080 | 0.3750 ± 0.0060 | 0.0000 | 0.7275 ± 0.0034 |

Generation was run greedily on every saved model. Each decoded continuation
preserved the raw prompt `At the harbor` as an exact byte prefix. Generated
content is recorded as hex in the raw log and was not quality-scored.

## Interpretation and limitations

On this fixed synthetic held-out split, M2 has lower mean byte-normalized loss
than M1 in both regimes despite emitting more test tokens. Under the same
backbone, M2 also has lower measured training and test-scoring time, alongside
roughly half the per-update estimated MACs. Under the matched-MAC regime, M2
uses 249 updates versus M1's 120 and takes longer wall time, while its mean
test bits/byte remains lower. This result is only an observation about this
small corpus, tiny model, tokenizer pair, and run protocol. It is not evidence
of general model quality or a claim that PackTok is better.

The pack predictor's same-backbone accuracy varies sharply by seed, and nearly
two-thirds of test targets remain byte fallback. The model uses only three
nonempty pack tables for this learned artifact. There is one benchmark machine,
one synthetic corpus, three seeds, no warm-up benchmark pass, and no CPU
pinning. Process scheduling and timer resolution affect these short runs. The
MAC estimate omits activations, softmax, Adam, memory traffic, and elementwise
work. These constraints prevent broad performance conclusions.

## Preserved preflight

`experiments/m3-model/runs/preflight-20261007-1/` is retained as a working-tree
preflight, not a final benchmark. It ran three seeds per regime before the M3
source was committed. Its head-work fraction averaged per-target ratios instead
of reporting the aggregate ratio, and its training time covered only optimizer
updates. The final implementation fixes both measurement definitions. The
preflight log and model files remain intact and are excluded from all final
tables. No held-out text or model/tokenizer setting changed after observing
the preflight.

## Verification

On Windows 10 Pro build 19045 with Rust/Cargo 1.98.1, the final code passed
`cargo fmt --all --check`, strict workspace Clippy, all workspace tests
(107 unit/integration tests), and the release workspace build. On WSL Ubuntu
26.04.1 using declared MSRV Rust 1.85.0, formatting, strict Clippy, all 108
unit/integration tests, and release build passed. MSRV Clippy initially found
two explicit lifetimes in helpers; they were elided and the complete final
verification then passed. Doc-test targets ran and contained no tests.
