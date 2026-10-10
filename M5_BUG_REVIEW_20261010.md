# M5 bug review — 2026-10-10

## Scope and provenance

Reviewed the current M5 Rust implementation and the preserved completed A/C pilot,
including byte adapters, finite-context scoring, domain aggregation, resumable
AdamW, checkpoint identities, metrics recovery and transport/ownership handling.
Implementation work and CPU checks use only the native Ubuntu-24.04 checkout at
`/home/dasetwa/projects/PackTok`, branch `m5-gpu-transformer`, starting HEAD
`958f273`. The Windows checkout remains at an older source version with existing
user changes; it was inspected but not synchronized, reset or overwritten.

The user allowed at most one GPU allocation if needed. This round allocated none:
the findings are reproducible with tiny CPU fixtures. No full-model CPU training,
new quality comparison, tokenizer retraining, new corpus or schedule tuning was
performed. Historical source snapshots, run logs, failures and model/tokenizer
artifacts remain preserved. Logs for this review are additive under
`experiments/m5-gpu/provenance/bug-review-20261010/`.

## New findings and corrections

### BR-1 — P1: a missing late gradient partially updates the model and moments

`ResumableAdamW::apply` published each variable and its moments inside its loop.
If a later variable lacked a gradient, it returned an error after earlier rows
had changed, while the optimizer step remained unchanged. Reproduction creates
an objective over every variable except the last variable in canonical order;
the old implementation fails the full model/moment hash invariant.

The update now stages all weights and moments and checks gradient layouts before
publishing them. Predictable gradient/layout/numerical errors leave model,
moments and step unchanged. This is not a claim of transactional recovery from
a device loss or an OS failure during publication.

Regression: `review_missing_late_gradient_does_not_partially_update`.

### BR-2 — P1: nonfinite gradients and candidate state were accepted

The resumable optimizer did not itself reject NaN/Inf gradients or nonfinite
candidate weights/moments. The extended runner checks gradient families only on
selected updates; a finite scalar loss is not a guarantee of finite gradients.
The old optimizer accepted the nonfinite fixture. Finite but sufficiently large
gradients can also overflow FP32 second moments.

Squared norms of every gradient and candidate weight/first/second moment are
now reduced on the device and checked together through one host transfer before
publication. Nonfinite norms, including FP32 norm overflow, fail without changing
state. Explicit zero gradients remain valid. This adds staging memory and
reduction/transfer work; historical throughput/VRAM measurements do not describe
the corrected optimizer. No new GPU performance measurement is claimed.

Regressions: `review_nonfinite_gradient_is_rejected_without_mutation` and
`review_finite_gradient_overflow_is_atomic_and_zero_gradient_is_valid`.
The existing exact CPU comparison against pinned Candle AdamW and uninterrupted
versus resumed state hashes remain required checks for finite valid updates.

### BR-3 — P1: checkpoint schedule was not compared with the expected schedule

Loading verified identity strings and tensor integrity, but accepted any valid
schedule embedded in checkpoint metadata. A checkpoint with the expected identity
and a different valid LR/decay/warmup/length could silently supply its schedule.
The tensor hash does not bind this metadata. This is an input-validation defect;
no evidence says the historical pilot had altered metadata.

Loading now requires the expected `AdamWConfig` supplied by the frozen plan and
compares it before modifying destination weights. The checkpoint format is
unchanged. Valid historical checkpoints remain readable with their expected
schedule; mismatched schedules require an explicit experiment rather than an
implicit resume.

Regression: `review_checkpoint_schedule_mismatch_preserves_destination`, covering
LR, total length, warmup and decay mismatches. Corrupt/truncated image rejection
also checks that destination weights remain unchanged.

### BR-4 — P2: failed metrics recovery changed the log before returning an error

`reconcile_metrics` rewrote/truncated the working log before confirming that a
nonzero checkpoint had its required progress record. A log with an ahead-of-step
row and a partial final row lost evidence even though recovery failed.

Recovery validates required progress before any rewrite. Successful rewinds save
the exact original bytes to a unique adjacent `.before-reconcile-*` file before
replacing the working log. Invalid recovery leaves the original untouched.

Regression: `review_invalid_metrics_reconciliation_preserves_original`; the
existing interrupted-tail/ahead-of-checkpoint test verifies successful recovery.

### BR-5 — P2: elapsed time prevented recovery of a partial final evaluation

`save_report` required byte-identical reports. If the first domain report had
been written but final completion was interrupted, recomputing the same report
produced a new `seconds` value and was rejected. Recovery could not finish.

Existing and candidate report values are now compared with only the top-level
elapsed `seconds` field removed. Every scientific field must still match exactly;
changed losses/counters remain rejected. The original report bytes and original
elapsed time are retained. This is deliberately narrower than accepting arbitrary
floating-point differences or replacing old reports.

Regression: `review_report_retry_ignores_elapsed_time_only`.

## Edge coverage and historical findings

New adapter fixtures cover empty input, all 256 byte values, invalid UTF-8,
German umlauts/eszett, emoji, CJK, repeated class transitions, tabs, NUL and CRLF.
They require deterministic IDs/byte lengths, exact decode and byte coverage for
both existing flat and factorized artifacts. No routing or token IDs changed.

The new scoring matrix covers sequence lengths 2/8/9/16/17/24/25 with context 8
and batches 1/3/8, including full windows, partial final windows, multiple batches
and tokens crossing domain boundaries. Target and byte counters must agree
exactly; global versus stratified BPB must agree within 1e-5 for these CPU fixtures.
Existing empty/single-token/zero-batch rejection, causal mask, finite differences,
malformed sequence and artifact tests remain part of the suites.

Historical domain aggregation tolerance is already corrected and is not a new
finding. The preserved 20261009T154700Z one-shot pilot launcher lacks the endpoint
file expected by its release guard; this was already documented in PILOT_RESULT.md.
Its frozen evidence/consumed launcher was not rewritten. It must not be reused
as a future GPU supervisor. The canonical `l4-recovery.py` records endpoint ownership
and rejects reassigned/unknown aliases; its CPU mocks cover release, endpoint
survival, wrong hardware, transport failure and unrelated-session preservation.
No GPU was allocated to test these mocks and no fresh real lifecycle PASS is claimed.

## Verification and interpretation

Baseline: 35 M5 CPU tests passed. Four new reproduction tests failed on the
uncorrected implementation; raw failures are in `reproductions-before.txt`.
An initial new zero-gradient fixture used multiplication by zero, which Candle
optimized away and therefore yielded missing gradients. That fixture failure is
preserved in `regressions-after.txt`; the corrected fixture inserts explicit
zero gradients and the corrected test output is separate.

Final verification: 44 M5 library tests pass in debug and release profiles;
the root workspace MSRV suite passes 135 tests on Rust 1.85.0. Lifecycle transport
mocks pass 24 tests and the pilot bridge mocks pass four tests. Formatting and
strict all-target Clippy pass. All-target/all-feature Clippy, including CUDA,
also passes on local WSL CPU using the already installed CUDA toolkit and
`target-cuda`; this is compile/static evidence, not CUDA execution.
The release binary build and `git diff --check` pass. The final nine focused
review regressions also pass after the last gradient-layout guard. The pilot
config retains SHA-256
`6536d5194bc7d422bdef6e871b58317497794ed1553b197a927e0434aa65c3d8`.

Reproduction commands (from `experiments/m5-gpu`, unless stated otherwise):

```sh
cargo test --locked --lib
cargo test --release --locked --lib
cargo clippy --locked --all-targets -- -D warnings
cargo fmt --check
env CUDA_ROOT=/home/dasetwa/.local/opt/packtok-cuda-12.4.1 CUDA_COMPUTE_CAP=89 cargo clippy --locked --all-targets --all-features --target-dir target-cuda -- -D warnings
python3 scripts/test-recovery.py
python3 scripts/test-remote-bridge.py
# From the repository root:
cargo +1.85.0 test --locked --workspace
```

Logs: `all-tests-after-2.txt`, `release-tests.txt`, `clippy-final.txt`,
`fmt-final.txt`, `cuda-clippy-cpu.txt`, `root-msrv-tests.txt`,
`lifecycle-tests.txt`, `bridge-tests.txt`, `regressions-final.txt`,
`release-build.txt`, `diff-check.txt`. Earlier intermediate runs and failures
have distinct filenames and are not replaced by the final results.

The reviewed completed pilot still records A test BPB 2.579990970 and C
2.605001505. No finding establishes corruption of those successful finite training
updates or scores. CPU fixes do not demonstrate a PackTok quality gain and do
not replace that negative single-seed result. Corrected CUDA execution and its
new performance remain unverified until a separately prepared bounded run.

## Delivery into the current PR

The current open PR is #7 (`m6-research`), based at
`90c0a98c2b4f9ea24cf5b9b7c8aa91c3ce2f61e1`; PR #5 is already merged.
The corrections and nine added regressions were applied to the existing native
M6 worktree at `/home/dasetwa/projects/PackTok-m6-research` without replacing its
M6 seed/config validation, sampler seed constant, initialization hashes or
replication results. Original review evidence above retains its M5 provenance.

PR-specific CPU checks are written separately as `pr7-*.txt` in this review's
evidence directory. The M6 worktree's ignored `experiments/m5-gpu/data` symlink
points to the unchanged prepared data in the original native M5 checkout for
read-only hash/config tests; no large dataset copy or data regeneration is needed.
No new GPU allocation or experiment was performed during PR delivery.

All-feature CUDA Clippy in this worktree first failed because `nvcc` was not on
its PATH (`pr7-cuda-clippy-cpu.txt`). The retry adds the existing toolkit's
`/home/dasetwa/.local/opt/packtok-cuda-12.4.1/bin` to PATH, retains CUDA_ROOT and
compute capability 89, and passes (`pr7-cuda-clippy-cpu-retry.txt`). No compiler
installation or GPU allocation was required. This check remains CPU-only evidence.

PR #7 revalidation passes 44 debug tests and 44 release tests (including the M6
frozen-config/seed checks), strict all-target Clippy, formatting and the release
binary build. Their complete outputs are `pr7-debug-tests.txt`,
`pr7-release-tests.txt`, `pr7-clippy.txt`, `pr7-fmt.txt` and
`pr7-release-build.txt`. No M6 corpus/config, result file or statistical conclusion
was changed by this delivery.
