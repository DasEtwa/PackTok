# M1 PR #1 review fixes and held-out benchmark — 2026-10-07

All seven Codex findings on [PR #1](https://github.com/DasEtwa/PackTok/pull/1)
are addressed. The prior correctness/performance audit is included in the same
branch. This remains a flat BPE baseline; no factorized packs or model experiment
are implemented, and these results do not establish PackTok superiority.

## Review findings and regression evidence

| Review comment | Fix | Evidence |
| --- | --- | --- |
| [Training/evaluation overlap](https://github.com/DasEtwa/PackTok/pull/1#discussion_r4202104108) | Train only on `fixtures/benchmark/train.txt`; evaluate four separate documents; reject shared 32-byte substrings. | `benchmark_training_and_evaluation_passages_are_separate`; explicit split and checksums in each run. |
| [Changed decoded stdout](https://github.com/DasEtwa/PackTok/pull/1#discussion_r4202104116) | Write and flush raw bytes without a newline or UTF-8 conversion. | Unit tests plus actual M0/M1 CLI subprocess tests for empty input, Unicode, controls, invalid UTF-8, and encode/decode round-trips. |
| [Missing memory/distribution metrics](https://github.com/DasEtwa/PackTok/pull/1#discussion_r4202104124) | Runtime vector allocation/capacity counters; OS whole-process peak memory; record-count/min/p50/p95/max/mean token lengths. | Instrumented/uninstrumented differential checks, nearest-rank quantile tests, and Windows/Linux memory-query tests. |
| [Unix corpus path collisions](https://github.com/DasEtwa/PackTok/pull/1#discussion_r4202104128) | Join actual path components with `/`, preserving literal backslashes in Unix filenames. Already fixed during the first audit. | Unix regression fixture containing both `a/b` and literal `a\b`; checks distinct provenance, sorted concatenation, and repeatability. |
| [Unbounded artifact reads](https://github.com/DasEtwa/PackTok/pull/1#discussion_r4202104135) | Read at most the 16 MiB format limit plus one byte before parsing. Already fixed during the first audit. | Endless-reader test verifies the exact consumption bound; legacy/truncated artifact tests. |
| [Partial artifact after write failure](https://github.com/DasEtwa/PackTok/pull/1#discussion_r4202104142) | Close and remove the file created by this call after write/sync errors; report cleanup failure; preserve existing files. | Inject a failure after three bytes, check removal, retry successfully, and verify overwrite rejection preserves bytes. |
| [Unbounded token expansion](https://github.com/DasEtwa/PackTok/pull/1#discussion_r4202104151) | Validate a 1 MiB per-token maximum in both model construction and wire parsing. | Accept the exact boundary; reject the next doubling and a forged artifact smaller than 512 bytes before decoding. |

The expansion cap deliberately tightens validation: artifacts containing larger
tokens that previously loaded are now rejected. Wire records, valid artifact bytes,
token IDs, normalization and byte fallback remain unchanged. Total decoded output
still scales with the caller's token sequence; the cap bounds a single token.
CLI decode output intentionally changes to the represented bytes. Escaped display
remains available through `inspect-token`.

The original audit additionally fixes encoder scan overhead, byte-decoder overhead,
reader/writer limit mismatches, excess training/corpus buffers, recursive traversal,
file-count handling and diagnostics. Its full findings and unsuccessful intermediate
optimization are preserved in [M1_PERFORMANCE_AUDIT.md](M1_PERFORMANCE_AUDIT.md).

## Reproducible protocol

- Starting commit: `bfd3224f46f30c1bb61682a94cdea927c2fca079`. Measurements were
  captured before committing the final review fixes. Exact measured source/input
  fingerprints are in [source-sha256.txt](experiments/bpe-baseline/pr1-review-20261007/source-sha256.txt).
- Command: `cargo run --release -p packtok-bench`, repository root; three runs.
  Default release settings; Rust `1.98.1 (48a229cea 2026-09-01)`, LLVM `22.1.8`,
  `x86_64-pc-windows-msvc`; no external Rust crate dependencies.
- Windows 10 Pro `10.0.19045`, Intel Core i7-2600 @ 3.40 GHz, four cores/eight
  logical processors. No CPU affinity/frequency locking or statistical confidence
  intervals. Other desktop activity can affect these short timings.
- Manually authored synthetic fixtures, Apache-2.0 repository license; no downloads,
  external annotations, normalization, or preprocessing. Exact fixture bytes are
  preserved by `.gitattributes`. The split is described in
  [fixtures/benchmark/README.md](fixtures/benchmark/README.md).
- Training: 1,640 bytes; FNV-1a `4d0cc0a94cc66e49`; target 512 IDs, maximum 256
  merges, minimum frequency 2. Result: 256 merges/512 IDs, 3,951-byte artifact.
  Train-once observations: 19.654, 19.802, 19.709 ms, excluding file loading and
  serialization. They do not establish large-corpus training throughput.
- Each complete evaluation document is repeated 32 times for throughput. Each
  operation receives 25 ms warm-up and 200 ms measurement. M0, current BPE and
  retained pre-audit scanner use identical timed inputs; both BPE encoders use
  the same model. Canonical IDs and byte-exact decoding are asserted before timing.
  Allocations are included in timing; instrumented calls run separately.
- Sequence lengths use original newline-delimited records, including newline bytes,
  without timing repetitions. Quantiles use nearest rank; these tiny record counts
  describe fixtures rather than a population estimate.
- Training corpus repetition and synthetic dense/unmatched byte cases are labeled
  stress workloads and excluded from held-out compression measurements.

Raw observations: [run 1](experiments/bpe-baseline/pr1-review-20261007/held-out-1.txt),
[run 2](experiments/bpe-baseline/pr1-review-20261007/held-out-2.txt),
[run 3](experiments/bpe-baseline/pr1-review-20261007/held-out-3.txt).

## Held-out results

Throughput values are medians of the three runs, in MiB/s. Ratios are medians of
the per-run current/scanner ratios. They compare two implementations of the same
flat BPE semantics, not different tokenizer algorithms.

| Evaluation | Original bytes / FNV-1a | Timed bytes / BPE tokens | Bytes/token | Reduction vs M0 | BPE encode | Retained scan encode | Current/scan | BPE decode |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English | 229 / `b00b21654654d1bc` | 7,328 / 5,088 | 1.440 | 30.57% | 19.63 | 5.21 | 3.75x | 240.00 |
| German | 247 / `7767115b6b9b4e70` | 7,904 / 5,312 | 1.488 | 32.79% | 17.75 | 6.04 | 2.94x | 233.86 |
| Unicode-heavy | 223 / `97f4d1f3d4c1a47f` | 7,136 / 6,720 | 1.062 | 5.83% | 57.55 | 5.63 | 10.25x | 300.36 |
| Source-like | 255 / `a3e31d9692c8cd3f` | 8,160 / 6,048 | 1.349 | 25.88% | 23.37 | 5.09 | 4.56x | 251.41 |

M0 encodes one token per byte and remains much faster: median encoding ranges from
1,077.79 to 1,408.04 MiB/s across these inputs. BPE's fewer tokens do not imply
better model loss, training compute or end-to-end inference. The new training
model/split differ from historical reports; cross-report throughput and compression
are not matched comparisons.

| Evaluation | Records | M0 min / p50 / p95 / max / mean | BPE min / p50 / p95 / max / mean |
| --- | ---: | --- | --- |
| English | 3 | 6 / 92 / 131 / 131 / 76.33 | 4 / 57 / 98 / 98 / 53.00 |
| German | 3 | 7 / 91 / 149 / 149 / 82.33 | 5 / 61 / 100 / 100 / 55.33 |
| Unicode-heavy | 3 | 5 / 82 / 136 / 136 / 74.33 | 5 / 81 / 124 / 124 / 70.00 |
| Source-like | 9 | 1 / 28 / 63 / 63 / 28.33 | 1 / 20 / 53 / 53 / 21.44 |

## Memory scope and trade-off

| Evaluation | Encode/decode vector growth requests | Peak encode vector capacity | Peak decode vector capacity |
| --- | ---: | ---: | ---: |
| English | 3 / 2 | 333,824 B | 7,344 B |
| German | 3 / 2 | 358,656 B | 7,920 B |
| Unicode-heavy | 3 / 2 | 339,200 B | 7,152 B |
| Source-like | 3 / 2 | 374,784 B | 8,176 B |

These are successful capacity-growth requests and maximum simultaneous capacity of
runtime-owned vectors, not global allocator calls. They exclude the immutable
model, caller input, allocator bookkeeping and RSS. The event encoder improves
throughput while using roughly four to five times the former scan path's nominal
vector capacity on these fixtures; it is not a memory reduction.

OS high-water working-set measurements were 6,889,472, 6,606,848 and 6,811,648 bytes
(6.30–6.57 MiB). They cover the entire benchmark process, including training,
model construction, M0/M1, scanner and stress cases. They cannot identify a
per-tokenizer peak. Windows reads `PeakWorkingSet64` through a hidden PowerShell
helper after timing; Linux reads `/proc/self/status` `VmHWM`. Neither mechanism
belongs to tokenizer runtime. Unsupported platforms or failed queries report
unavailable rather than zero. No total-training-allocation metric is claimed.

## Verification

- Windows: `cargo fmt --all --check`, Clippy for all workspace targets/features
  with warnings denied, `cargo test --workspace` (59 unit/integration tests), and
  `cargo build --release --workspace` passed. Doc-test suites contain no tests.
- Linux: `cargo +1.85.0 test --workspace` passed all 60 unit/integration tests on
  Ubuntu 26.04.1 LTS, WSL kernel `6.18.40.1-microsoft-standard-WSL2`, using
  `rustc 1.85.0 (4d91de4e4 2025-02-17)`. This verifies the declared minimum Rust
  version, the Unix filename regression, raw CLI output, and the Linux memory
  counter. Compiler and build output were isolated under `/var/tmp/packtok-pr1-*`;
  no project dependency or shell startup configuration was added.
- Original corpus trained and validated twice through the release CLI: 84 merges,
  340 IDs, 1,881 bytes. Both artifacts retain SHA-256
  `CEF973A354422C88E6FEF54D0B6EEE09495980DF4E5BD0B7E5ADC37896E7D907`.
- The audit's independent reference checks, exhaustive/generated byte tests,
  Unicode/German/emoji fixtures, malformed artifacts, boundary limits, corpus
  edge cases and deterministic serialization remain part of the suite.

Historical reports and raw failures have been preserved. No benchmark, model
quality or production-scale capacity result has been silently replaced.
