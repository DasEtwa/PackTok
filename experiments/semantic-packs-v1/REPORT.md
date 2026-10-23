# Semantic Packs v1 experiment and delivery

## Question and architecture

Can multiple English/German surface expressions share stable semantic concept IDs
while original bytes, byte-BPE segmentation and historical artifacts stay intact?
V1 implements a bounded registry, deterministic longest-match byte spans and
conservative contextual abstention as an optional sidecar. See
[architecture and wire contract](../../docs/architecture/semantic-packs-v1.md).
No neural model adapter/training is part of this experiment. The hypothesis of
better generalization remains untested; the sidecar adds information and cost.

The implementation stays inside existing crates: packs owns semantic definitions
and decisions, tokenizer owns the optional wrapper, format owns binary sidecars,
CLI exposes inspection, bench owns measurement and gold parsing. No dependency
or crate was added, no old token ID or artifact version changed, and no frozen
M4/M5 experiment was modified. Registry identity travels with annotated results.

## Baseline and matched protocol

Original repository base `242334241548fee614e104a2fe3b944d7486cefe` (main).
A clean archived snapshot passed 135 debug tests and 135 release tests, rustfmt,
Clippy all-targets/all-features with warnings denied and release workspace build.
See [baseline logs and reproduction](baseline/README.md).

A = existing M1 byte-BPE only. B = the identical tokenizer/artifact plus semantic
annotations. BPE trains once on `fixtures/benchmark/train.txt` (1640 bytes,
FNV-1a `4d0cc0a94cc66e49`), default target 512, max merges 256, min frequency 2.
Result: 512 IDs, 3951-byte artifact, FNV-1a `82e886f6b18cb4c9`. The 33-case
agent-authored development gold corpus is never used for tokenizer training.
Its decoded texts total 472 bytes; its TSV FNV-1a is `e155c795642f05dc`.
It is a policy/conformance fixture, not blind held-out natural-language evidence.

Measurements run with Rust/Cargo 1.85.0 release on Linux x86_64, reported CPU AMD
EPYC 9V74. Same 32320-byte newline-separated/repeated batch, seven trials,
alternating A/B order, 25 ms warmup then at least 100 ms per direction/trial.
Raw trials report actual elapsed time and iterations. Timings exclude training,
artifact load, I/O and sidecar serialization/parsing. Both decode timings call
the identical lexical decoder and ignore annotations. Sidecar load re-runs the
resolver and is not measured as lexical decoding. No GPU or model training ran.

Reproduction from repository root (fresh run label required):

```sh
cargo +1.85.0 run --release -p packtok-bench -- semantic unique-label
cargo +1.85.0 run --release -p packtok-cli -- semantic 'ong idk bro'
```

The harness retains command/toolchain/source/corpus hashes, artifact, case-level
results, gold/actual decisions, raw timing trials and summary; failed run
provenance is preserved. Run directories cannot be overwritten.

## Initial observations (before independent review fixes)

Retained unchanged in [initial-20261023](initial-20261023/report.txt). This was a
working-tree build at the original base; exact source hashes are retained in
its environment record. Attribution wording was corrected afterward; original
output saying “hand-authored” means explicitly specified agent-authored gold.
It must not be read as independently human-labeled data.

| Measurement | A: lexical BPE | B: same BPE + semantics |
|---|---:|---:|
| Median encode MiB/s | 37.967293 | 25.776207 |
| Median decode MiB/s | 437.778013 | 434.773891 |
| Lexical tokens, all 33 cases | 367 | 367 (identical IDs) |
| Lexical output vector capacity, aggregate bytes | 2936 | 2936 |
| Additional binary sidecars, aggregate bytes | 0 | 2224 |
| Additional annotation output capacity, aggregate bytes | 0 | 5348 |

All 37 expected spans and 30 resolved senses matched; seven ambiguous cases
abstained and all six unknown cases produced no annotations. All input roundtrips,
optional-off token differentials, repeated outputs and sidecar roundtrips passed.
The paired process peak RSS was 4136960 bytes; it includes training and both
variants and cannot establish additional semantic peak RSS. Capacity estimates
exclude allocator metadata, registry, temporaries and stack. The initial wrapper
had not yet added owned registry identity; final measurements include this cost.

The initial encode median loses about 32.1% throughput with annotation; equivalently
encoding wall time per byte increases about 47.3%. Decode differences are noise
on the same code path. Sidecars add 2224 bytes to 472 bytes of original text across
33 individually framed cases (1320 header bytes plus 904 annotation record bytes).
There is no compression improvement: lexical sequence lengths are unchanged and
persistent sidecar size is additional. This tiny corpus's perfect conformance
scores do not establish general accuracy or model quality.

## Independent review and fixes

Three independent agents reviewed Rust/correctness, architecture/compatibility and
adversarial semantics. Their original reports and reproducible probes are retained
under [reviews](REVIEWS.md). Deduplicated findings covered truncated context-key
boundaries, vertical whitespace, missing registry identity, case rules incorrectly
coupled to whitespace, quote boundaries and overbroad Unicode-boundary wording.
Regression tests were recorded failing before the resolver fixes, then passing.
The corrected resolver has a distinct policy `semantic-rules-v1.1`; old source
fingerprints/results are preserved. No lexical artifact version changed.

## Limitations and v2 recommendation

This registry has five coarse concepts and fourteen patterns, bounded by explicit
limits. It does not infer negation scope, sarcasm, quotations as mentions, names,
code domains or general sentence meaning; direct expressions may still be false
positives in another domain. Its Unicode boundary heuristic deliberately misses
some punctuation/symbol combinations. Semantic annotation accepts valid UTF-8;
legacy raw-byte tokenization remains lossless for invalid UTF-8. Sidecars are
capped at 16 MiB and use noncryptographic fingerprints, then canonical regeneration
on load. Output-capacity estimates are not full allocator/peak-RSS measurements.

V2 should first freeze an independently labeled challenge set with ambiguous and
negative contexts, measure precision/coverage/abstention, then test a small semantic
input adapter against the exact same tokenizer, model parameter budget, compute
and data split. Batch framing/compact spans or a streaming recognizer should only
follow measured storage/runtime need. Preserve byte spans, stable IDs, versioning
and lossless lexical recovery throughout. No faster-learning claim is justified
until that matched model experiment exists.
