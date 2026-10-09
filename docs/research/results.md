# Research results — authoritative overview

This document synthesizes the current evidence. The linked original reports,
raw logs, per-seed tables and manifests remain authoritative for numerical detail
and reproduction. No historical outcome was rerun or replaced for maintenance.
Lower bits per scored raw byte means lower held-out NLL under the declared scoring
protocol; token perplexities are not comparable between tokenizers.

## M1 — flat BPE control

**Hypothesis.** Deterministic byte BPE can supply a reproducible flat control while
preserving arbitrary-byte fallback.

**Experiment.** The held-out review trains on 1,640 synthetic bytes with 512 IDs
and evaluates separate English, German, Unicode and source-like documents. Three
Windows release repetitions use 25 ms warm-up and 200 ms timed loops on repeated
documents. Rust/Cargo 1.98.1, Intel i7-2600, Windows 10 build 19045; allocations
are included and process-wide memory scope is documented.

**Measured result.** Held-out bytes/token are 1.440, 1.488, 1.062 and 1.349
respectively. The current event encoder is 3.75x, 2.94x, 10.25x and 4.56x the
retained scanner's throughput on those same BPE semantics. Its temporary vector
capacity is larger. Raw-byte M0 remains substantially faster.

**Interpretation.** M1 supplies compression and a tested implementation control;
the scanner speedup is not a PackTok-versus-BPE algorithm advantage.

**Limitations.** Tiny synthetic data and desktop timing do not establish model
quality or large-corpus throughput. Initial training-like observations and later
held-out results are separate records.

**Reproduction.** `cargo run --release -p packtok-bench` runs the current harness;
historical measured fingerprints, commands and raw observations are in
[M1 review](../../M1_REVIEW_FIXES.md). See also [baseline](../../M1_BPE_BENCHMARK.md)
and [post-fix audit](../../POST_FIX_AUDIT.md).

## M2 — factorized pack representation

**Hypothesis.** Independent local vocabularies with a common byte base may change
compression and runtime costs under a fixed learned-token budget.

**Experiment.** Frozen lexical-v1 byte routing into TEXT/NUMBER/STRUCTURE, shared
BYTE_FALLBACK, independent BPE graphs and one global allocation budget. M1/M2 use
the same synthetic train/eval files, 512 logical slots and the same machine/toolchain
as the original report, with three release repetitions.

**Measured result.** Relative to M1, M2 emits 6.92% more English tokens, 11.45%
more German tokens and 7.94% more source-like tokens; Unicode-heavy sequences are
4.29% shorter. The historical M2 artifact is 4,492 bytes versus 3,951 bytes for
the matched M1 artifact. M2 encode/decode throughput is lower on these fixtures.

**Interpretation.** The configurable representation and exact fallback work.
These measurements show trade-offs and mostly unfavorable tokenizer compression
under this heuristic, not a general efficiency win.

**Limitations.** The report precedes a separately documented span-workspace
optimization. The router is context-free, not POS intelligence. No model-quality
conclusion follows from this tokenizer-only comparison.

**Reproduction.** Current tokenizer harness: `cargo run --release -p packtok-bench`.
Original hashes/tables: [M2 benchmark](../../M2_PACKS_BENCHMARK.md); contracts:
[M2 packs](../../M2_PACKS.md); later implementation measurements:
[performance audit](../../PERFORMANCE_MATH_AUDIT.md).

## M3 — tiny CPU model comparison

**Hypothesis.** Packed token sequences and a pack/local head may produce a useful
quality/cost signal under a shared tiny causal backbone.

**Experiment.** Synthetic 2,605/604/614-byte train/validation/test splits; tanh RNN
hidden/context 16, three seeds 20261007–20261009. Same-backbone regime uses 120
updates; the other matches the flat model's analytical MAC budget. Tokenizer and
head both change; total parameters differ (17,424 flat, 17,475 factorized).

| Regime | M1 flat test bits/byte | M2 factorized test bits/byte |
|---|---:|---:|
| Same backbone / schedule | 6.8752 ± 0.1462 | 6.0077 ± 0.0550 |
| Matched analytical MACs | 6.8752 ± 0.1462 | 6.0953 ± 0.1877 |

Values are means ± population SD across three seeds, using exact scored-byte
denominators and excluding the first unscored token. This is the original
Windows release experiment; the later numerical audit retained all twelve
historical model hashes.

**Interpretation.** A narrow synthetic-corpus signal exists, but its mechanism
cannot be assigned to the tokenizer or head from M3 alone.

**Limitations.** Tiny model/data, unequal raw-byte context and exposure, unequal
parameters, and analytical rather than measured compute matching.

**Reproduction.** `cargo run --release -p packtok-bench -- m3 UNIQUE_LABEL`
refuses existing run directories. [M3 report](../../M3_MODEL_BENCHMARK.md),
[protocol](../../M3_MODEL.md) and
[original run](../../experiments/m3-model/runs/final-20261007/) preserve exact evidence.

## M4 — tokenizer/head ablation

**Hypothesis.** M3's signal may come from token sequence, generic output
factorization, or their interaction.

**Experiment.** Frozen A=M1/flat, B=M1/synthetic factorized, C=M2/flat,
D=M2/factorized. Input rows are controlled by bijections/permutations. The same
tiny RNN (hidden/context 16) uses three seeds, a fixed English/German/Rust/synthetic
mixture (1,857,089/82,984/86,816 raw train/validation/test bytes), and either 2,000
updates or A's analytical budget. No held-out outcome changes grouping or schedule.
Measured implementation `8ae3df4968279f978ffff981fa1eb38347a8dcde`; Windows
Rust/Cargo 1.98.1 release on the i7-2600. Corpus/seed/timing scopes are in the report.

| Larger-corpus regime | A | B | C | D |
|---|---:|---:|---:|---:|
| Same schedule, test bits/byte | 2.61247041 ± 0.00154161 | 2.62290377 ± 0.00296306 | 2.58136040 ± 0.00099970 | 2.59248241 ± 0.00554472 |
| Matched analytical MACs, test bits/byte | 2.61247041 ± 0.00154161 | 2.51473152 ± 0.01380958 | 2.58136040 ± 0.00099970 | 2.49148184 ± 0.01241998 |

Means ± population SD, all three seeds retained. Same-schedule paired B−A is
+0.01043336, C−A −0.03111001, D−A −0.01998800 bits/byte. The interaction
`(D−C)−(B−A)` is +0.00068865 ± 0.00628396 and changes sign across seeds.
B receives 7,147 MAC-budget updates; D receives 4,446/4,444/4,451, versus A/C's
2,000. Additional optimizer steps and byte exposure are part of this regime.

**Interpretation.** C supports a small tokenizer-sequence quality difference on
this tested corpus. B loses at equal updates, so generic factorization alone did
not improve same-schedule quality. Reduced analytical output work permits more
updates at the budget and improves B/D there. A strong favorable interaction was
not established. M2 still emits more tokens; compression is not the mechanism.

**Limitations.** One mixture, three seeds and a tiny short-context RNN do not settle
Transformer scaling. Validation/test domain composition differs. MAC matching is
not equal wall time, total FLOPs, parameters or raw exposure. Tiny compatibility
outcomes, including unfavorable longer-budget B behavior, remain in the original
report; every A/D compatibility model matches historical M3.

**Reproduction.** Preparation `cargo run --release -p packtok-bench -- m4-corpus`;
runs `cargo run --release -p packtok-bench -- m4 tiny|large UNIQUE_LABEL` (choose
one of tiny or large). [Protocol](../../M4_ABLATION.md),
[complete report](../../M4_ABLATION_BENCHMARK.md),
[per-seed large tables](../../experiments/m4-ablation/large-complete-tables.md),
[raw summary](../../experiments/m4-ablation/runs/large-final-20261007/run-summary.txt).
Do not regenerate or overwrite historical outputs.

## M5 — GPU Transformer status

**Hypothesis.** M4's small C−A tokenizer-sequence signal may survive a shared flat
causal Transformer.

**Prepared experiment.** A/C use the same 14,681,984-parameter FP32 architecture,
512 global model IDs and frozen corpus-v2 (37,517,499 raw bytes). The isolated Rust
workspace uses Candle 0.9.1. Primary three-seed schedules remain proposals subject
to a successful GPU gate and explicit measured-budget approval.

**Measured result.** The Rust/CUDA gate and 500-step L4 overfit diagnostics passed;
see the preserved [L4 diagnostic](../../experiments/m5-gpu/provenance/l4-recovery-b573d896abe948138f0221283f85af75/M5_OVERFIT_DIAGNOSTIC_L4.md).
One later, explicitly authorized A/C 2,000-update pilot allocation was released
after remote orchestration exited with code 1 before a Rust training update. Its
160-byte result archive contains only exit code 1. There are no A/C training
losses, validation/test scores, checkpoints, pilot step times or VRAM measurements.
The detailed attempt and source/config/package identities are preserved in the
[pilot report](../../experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T032000Z/PILOT_ATTEMPT.md).

**Interpretation.** CUDA correctness is verified, but the A/C Transformer quality
comparison remains incomplete. The likely missing approval environment variable
is an inference from the submitted command and shell contract; remote stderr was
not retained, so the exact failure cause is unverified. The single pilot
allocation authorization was consumed; no retry was made.

**Limitations.** No result from this attempt supports or rejects the tokenizer
hypothesis or readiness for a five-seed study. No pilot TEST score was produced.
The same test examples were included among prepared inputs, so a future revised
experiment must disclose that exposure. Weights-only safetensors cannot resume
optimizer/RNG/cursor state. A successful short Drive test does not prove
multi-day OAuth validity.

**Reproduction reference.** [Frozen Transformer protocol](../../M5_GPU_TRANSFORMER.md),
[failure/accounting report](../../M5_GPU_BENCHMARK.md),
[CPU preparation](../../experiments/m5-gpu/provenance/CPU_PREPARATION.md),
[recovery evidence](../../experiments/m5-gpu/provenance/RECOVERY_STORAGE.md),
[maintenance CPU checks](../maintenance/2026-10-08/REPORT.md).
New GPU work must be a separate explicitly authorized task.


### M5 A/C pilot launch recovery — 2026-10-09

No Transformer comparison was produced. The prior attempt's likely missing approval flag was fixed and locally mocked. The single new L4 attempt then failed in bootstrap because the bridge pre-created the `results` directory before the shell's `mkdir results`; archived stderr confirms this cause. The bridge was corrected and four CPU-only launch regressions pass after the allocation. A/C each have zero verified updates; there are no BPB, checkpoint, throughput or VRAM metrics, and the pilot gives no evidence for the five-seed study. The L4 was released after 386 seconds; displayed CU was 167.24→167.15, with no active assignments. See [attempt evidence](../../experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T115439Z/PILOT_ATTEMPT.md). The earlier failed attempt's stderr remains unavailable, so its precise cause is not asserted.


### M5 A/C first Transformer execution — 2026-10-09 (partial)

The verified single L4 allocation ran A through all 2,000 updates, exposing 4,096,000 target positions and 7,575,564 raw target bytes. Its final validation score was 2.57296485 bits per raw byte; the recorded curve is 4.82559757, 3.16390946, 2.80470130, 2.64190439 and 2.57296485 at steps 1/500/1000/1500/2000. A final-only TEST scoring computation occurred, but its number was not written before the run exited. C started zero updates. The final validation domain aggregate was rejected by a fixed `1e-9` FP64 NLL-sum comparison although independently ordered sums differ from rounding; a deterministic 800,000-token regression reproduced the rejection. A `1e-12` relative NLL tolerance with exact byte/token equality passes the new regression and all five domain tests. Because the code fix followed the immutable package execution and the authorized L4 allocation is consumed, no C run or paired BPB difference is available. The 161,437,114-byte results archive SHA-256 and the final A resume checkpoint hash are in [attempt evidence](../../experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T134013Z/PILOT_ATTEMPT.md) and its checksum manifest. L4 allocation lifetime was 813 seconds; CU changed 167.15 to 166.92; release was verified.


### M5 paired A/C Transformer pilot — 2026-10-09 (complete, exploratory)

Recovered the earlier 2,000-update A checkpoint rather than retraining it, applied
the corrected FP64 domain aggregation, then completed C's fresh 2,000-update run
with the same frozen 14.68M FP32 model, seed, configuration, corpus and target
position budget. Final validation is A 2.572964850 and C 2.597469229 bits per raw
byte; C−A is +0.024504380 BPB, a small measured result favoring A. Each saw
4,096,000 training targets, but raw-byte exposure differed: 7,575,564 for A and
6,946,291 for C. Five validation domains favor A on English, German and Rust;
synthetic JSON and Unicode favor C. This is one paired seed and is not statistical
proof. The frozen final-only runner also scored TEST; those exploratory scores are
not an untouched independent test for any revised study. The L4 was released and
verified; Drive full readback matched SHA-256. Full curves, metrics, checkpoints,
provenance, resource accounting and archive hash are in the
[pilot record](../../experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T154700Z/PILOT_RESULT.md).
The supervisor needed a manual stop because its expected endpoint file was absent;
the remote experiment itself exited 0 and its archive declares completion.
