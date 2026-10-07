# Post-fix branch audit — 2026-10-07

Target: `613f21e48d0337adeea9043ac292391ecfa29807` on `m1-flat-byte-bpe`,
after the PR #1 fixes. Three new confirmed issues were found and fixed. All seven
previous review findings were checked against code and regression tests; none
remains unresolved. Resolved findings are not presented again as new bugs.

## New findings

### 1. P2 / medium — training selects tokens its own validator rejects

- **Exact location at the audited commit:** `crates/packtok-train/src/lib.rs`,
  `train_model`, lines 265–289; same omission in `reference::train_model`,
  lines 602–638. The [selection loop](https://github.com/DasEtwa/PackTok/blob/613f21e48d0337adeea9043ac292391ecfa29807/crates/packtok-train/src/lib.rs#L265-L289)
  checks frequency but not the new per-token expansion bound.
- **Real bug/reproduction:** `train_model(&vec![b'a'; 3 * 1024 * 1024],
  BpeTrainingConfig::default())` fails with
  `InvalidModel(ExpandedTokenTooLarge { token_id: 276 })`. The trainer creates a
  2 MiB merge before validating the complete model. This is a regression caused
  by the previous validation fix, not a recurrence of oversized-token acceptance.
- **Expected vs actual:** a valid raw corpus should train using bounded tokens,
  skipping ineligible candidates and stopping when none remains. Instead, the
  entire operation fails even though three accepted 1 MiB tokens suffice.
- **Concrete fix applied:** track expanded lengths during both training paths and
  exclude pairs exceeding `MAX_BPE_TOKEN_BYTES` before frequency/tie selection.
  Keep the independent counting/replacement implementations and deterministic ties.
- **Regression tests added:**
  `training_skips_merges_above_the_token_expansion_bound` verifies the exact bound,
  20 merges, reference agreement and a 3 MiB round-trip;
  `training_considers_other_pairs_after_a_dominant_pair_reaches_the_bound` checks
  that a rejected dominant pair does not suppress remaining valid merges.

### 2. P2 / medium — dense nonuniform inputs suffer a heap throughput cliff

- **Exact location at the audited commit:** `crates/packtok-tokenizer/src/bpe.rs`,
  `BpeTokenizer::encode_measured`, lines 130–134, especially the
  [path-selection condition](https://github.com/DasEtwa/PackTok/blob/613f21e48d0337adeea9043ac292391ecfa29807/crates/packtok-tokenizer/src/bpe.rs#L130-L134).
- **Real bug/reproduction:** train from 512 `a` bytes, append 64 unused valid
  byte-pair merges, then encode 16,384 `a` bytes with byte 8,191 changed to `b`.
  Current throughput was 5.27 MiB/s versus 371.47 for the matched retained scanner;
  the uniform input was 486.64 MiB/s. Separately, the 9-merge model trained from
  `"ab".repeat(512)` encoded `"ab".repeat(8192)` at 6.92 versus 368.10 MiB/s.
- **Expected vs actual:** dense repetition should retain efficient contiguous
  merging when one byte differs or the period is longer than one byte. Instead,
  either condition forced costly heap bookkeeping for almost every adjacency.
  This is a new nonuniform-path regression; the earlier uniform-input fix works.
- **Concrete fix applied:** recognize dense periods of 1–16 bytes with a bounded
  exception allowance, using aligned samples followed by full-input verification,
  and route those inputs to the existing in-place rank scans. Compare batches
  aligned to every period to avoid per-pattern comparison overhead.
- **Regression tests added:**
  `dense_repetition_detection_handles_periods_and_exception_positions` covers all
  supported periods, six exception positions, sparse inputs and dense-prefix traps;
  `dense_nonuniform_runtime_paths_match_reference_and_round_trip` checks exact
  oracle IDs, raw-byte decoding and avoidance of heap buffers across four models.
  Three permanent benchmark stress cases cover exceptional, alternating and
  three-byte repetitions; timing is not asserted in unit tests.

### 3. P2 / medium — malformed native CLI arguments panic

- **Exact location at the audited commit:** `crates/packtok-cli/src/main.rs`,
  `run`, [lines 36–39](https://github.com/DasEtwa/PackTok/blob/613f21e48d0337adeea9043ac292391ecfa29807/crates/packtok-cli/src/main.rs#L36-L39).
- **Real bug/reproduction:** invoke `packtok validate` with a native argument built
  from Windows UTF-16 `[0xd800]`, or Unix raw bytes `[0xff]`. `env::args()` panics
  while converting the argument before normal command validation can run.
  The Windows reproduction returned exit code 101 and a Rust panic/backtrace hint.
- **Expected vs actual:** unsupported Unicode arguments should produce an ordinary
  CLI error and exit code 1. Actual behavior was an uncontrolled panic. This bug
  predates the prior fixes but was not previously reported.
- **Concrete fix applied:** iterate `args_os()` and fallibly convert arguments,
  returning a safely escaped error on conversion failure. No lossy conversion or
  normalization is introduced; file contents still accept arbitrary bytes.
- **Regression test added:**
  `non_unicode_arguments_return_a_clear_error_instead_of_panicking` invokes the
  real executable with invalid native arguments on both Windows and Unix and
  checks exit code 1, a useful diagnostic and empty stdout.

Observed pre-fix failures are recorded in
[reproductions.txt](experiments/bpe-baseline/post-fix-20261007/reproductions.txt).

## Validation and previous-finding verification

- Windows / Rust 1.98.1: fmt, Clippy for all workspace targets/features with
  warnings denied, all **64 tests**, and the workspace release build passed.
- Linux / Rust 1.85.0, the declared minimum version: **65 tests** passed, including
  the Unix path regression and invalid native-argument subprocess test.
- Previous constructor/reader bounds, expansion rejection, write-error cleanup,
  raw stdout, split checks, quantiles and memory counters remain covered by the
  passing suite. No old finding remains unresolved.
- The original fixture still trains to 84 merges / 340 IDs / 1,881 bytes, retaining
  SHA-256 `CEF973A354422C88E6FEF54D0B6EEE09495980DF4E5BD0B7E5ADC37896E7D907`.
  The held-out fixture retains 256 merges / 512 IDs / 3,951 bytes and identical IDs,
  token counts, record distributions and vector-capacity metrics.

## Benchmark protocol and limits

The prior [held-out protocol](M1_REVIEW_FIXES.md) is unchanged: same synthetic
train/evaluation bytes, configs, Windows hardware, Rust version, 25 ms warm-up and
200 ms measurements. New stress cases are excluded from held-out compression.
The audited implementation was measured once with the extended harness in
[before.txt](experiments/bpe-baseline/post-fix-20261007/before.txt).

The initial detector implementation compared three-byte patterns individually;
that overhead left it slower than the retained scanner on that added stress case.
Its runs `after-1.txt` through `after-4.txt` remain preserved. The final detector
compares batches aligned to each period. No failed or unfavorable result was removed.

Final raw runs: [1](experiments/bpe-baseline/post-fix-20261007/final-1.txt),
[2](experiments/bpe-baseline/post-fix-20261007/final-2.txt),
[3](experiments/bpe-baseline/post-fix-20261007/final-3.txt). Exact measured source/input
hashes are in [source-sha256.txt](experiments/bpe-baseline/post-fix-20261007/source-sha256.txt).
The following values are MiB/s; the audited run is one observation, final values
are medians of three. Small differences are not statistically established gains.

| Encode workload | Audited implementation | Final implementation | Retained scanner, final median |
| --- | ---: | ---: | ---: |
| Dense uniform, 8 merges | 602.47 | 604.73 | 435.69 |
| Dense uniform, 72 merges | 486.64 | 490.30 | 396.01 |
| One exceptional byte, 72 merges | 5.27 | 425.88 | 387.60 |
| Alternating bytes, 9 merges | 6.92 | 533.98 | 433.34 |
| Three-byte period, 10 merges | not measured | 403.36 | 332.46 |
| Unmatched bytes | 252.07 | 251.77 | 6.52 |
| Training-corpus repetition, stress only | 8.25 | 8.45 | 3.27 |

| Held-out evaluation | Audited BPE encode | Final BPE encode | Final BPE decode |
| --- | ---: | ---: | ---: |
| English | 18.55 | 19.69 | 247.71 |
| German | 17.49 | 17.81 | 243.49 |
| Unicode-heavy | 56.47 | 58.55 | 299.04 |
| Source-like | 23.21 | 23.67 | 254.90 |

No BPE throughput regression is visible on these final workloads. M0 English
encoding varied from 609.59 to 715.82 MiB/s in the final runs, below the single
audited observation of 1,156.62; earlier review runs also showed large variation
(656.24–1,101.16). M0 code and input are unchanged, but the cause of this measurement
signal has not been established. It is preserved rather than dismissed as a win.
OS process peaks were 7,053,312–7,163,904 bytes; the expanded stress workload prevents
attributing a whole-process change to a specific tokenizer operation. Same-input
BPE vector-capacity counters remain unchanged. Train-once observations were
20.290, 19.977 and 20.050 ms versus 19.999 ms before the trainer fix.

Test evidence: [Windows](experiments/bpe-baseline/post-fix-20261007/windows-tests.txt),
[Linux/MSRV](experiments/bpe-baseline/post-fix-20261007/linux-tests-final.txt).

Training now retains an additional `usize` length per potential vocabulary ID.
The repetition detector performs extra bounded checks and remains a heuristic:
periods over 16 bytes, multiple damaged sampling positions, and other inputs can
still select a costly path. No universal speed guarantee is claimed. Training
remains memory-resident and total decoded output scales with the supplied sequence;
there is no process-wide memory or output budget. Short fixture measurements have
desktop timing noise and do not establish production-scale or model-quality results.

Final outcomes: **3 new findings fixed; 0 old findings unresolved**. No ID, wire,
normalization or round-trip regression was observed. Final benchmark evidence and
any remaining measured trade-offs are retained alongside this report.
