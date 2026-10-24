# Validation and independent review record

The scale workspace is experimental. All checks below ran on the available
Linux CPU environment with Rust1.90.0; root compatibility used its declared
Rust1.85.0. No GPU was allocated, and no large model was trained on CPU.
The exact commands/statuses and complete output are preserved in `final/`.

## Final local gates

| Check | Observed result |
|---|---|
| Separate scale workspace rustfmt | pass |
| Separate scale workspace Clippy, locked/all-targets/warnings denied | pass (CPU feature set) |
| Separate scale debug tests | 26 passed, 0 failed/ignored |
| Separate scale release tests | 26 passed, 0 failed/ignored |
| Separate scale release build | pass |
| Original root Rust1.85 fmt +Clippy all-targets/all-features | pass |
| Original root debug/release tests | 161 passed each, 0 failed/ignored |
| Colab/bootstrap/backup standard-library Python tests | 20 passed |
| Notebook JSON/AST/compile | 5 cells, 2 executable code cells |
| Notebook schema, nbformat5.10.4 | pass |

The model tests compare every parameter gradient with full-graph differentiation,
compare two accumulated microbatches with a batch mean, check causal attention,
zero-feature neutrality, deterministic initialization and 1B counts without
allocating its tensors. Optimizer tests compare scalar AdamW math/clipping,
interrupted replay, BF16 unrounded master-state persistence, malformed shards,
publication atomicity and safe retention. Data tests cover exact bytes, bounded
lines, seek/resume, malformed records, prefix causality and visible-window-only
features. CLI integration exercises prepare→plan→preflight→train→resume→evaluate,
rejects one-step plans and verifies stale PASS removal after a failed preflight.
OS run-lock and metrics reconciliation regressions are automated.

`probe-orchestration-cpu.py` additionally executes the actual release Rust CLI
through the notebook helper, with installation/GPU discovery mocked and explicit
tiny FP32 `--cpu` commands. It writes real Safetensors, verifies backup, deletes
all local runtime data, restores the backup and resumes. Two attempts preserve
optimizer step4 and canonical steps[1,2,3,4]. The backup destination is a local
filesystem fixture, **not real Google Drive/FUSE**. Both root and independent
reviewer reran this probe. Logs retain this distinction explicitly.

## Independent review and autonomous fixes

- [Rust review](review-rust.md): index/shape/error paths, full gradient sets,
  checkpoint loading and static BF16 CUDA kernel availability. RR1, accepted
  one-step plans with a zero final cosine LR, is fixed by rejecting plans below
  two updates before model allocation. Actual GPU kernel execution was not inferred.
- [ML review](review-ml.md): gradient math, baseline controls, loss/byte accounting,
  resume history and causal annotation. ML1 duplicates RR1. ML2 is fixed by
  preserving old metrics and reconciling the canonical curve to restored progress.
  ML3 is fixed by recomputing features from the exact current input window rather
  than importing whole-corpus context. Independent cropped-phrase/target-delimiter
  regressions and loss-identical interrupted CPU replay pass.
- [Notebook review](review-notebook.md): first installation, source pinning,
  bounded processes, cleanup and complete checkpoint transport. NB1 fresh
  `--no-checkout` clones incorrectly failed their dirty-index check; NB2 runtime
  disconnection could discard unbacked-up state after setup failure; NB3 Python
  settings bounds disagreed with Rust; NB4 a version-pin unit assertion was wrong;
  NB5 checkpoint publication could outrun copied metrics; NB6 the first prepared
  dataset's parent directory was absent. All are fixed and independently retested.

The checkpoint/log race now freezes a source pointer before copying metadata and
copies only that immutable image; a pruned source retries a later snapshot.
Runtime disconnection requires verified final preservation or explicit disposal
of local output. A real fresh Git clone is tested, not merely a mocked invocation.
The actual Rust orchestration probe found NB6 after earlier mocks passed; its
[original failure](probe-orchestration-cpu-before-fix.log) and
[passing rerun](probe-orchestration-cpu.log) remain available.

Initial scaffold/compiler/test failures are retained alongside the final logs.
They include pending module files, the original small BF16-construction test
guard, and an integration fixture that BPE compressed below its chosen context.
No failed result is presented as a pass or discarded to improve the narrative.

All reproduced review findings are closed. No known reproducible critical/high/
medium defect remains from these reviews. This is bounded test/review evidence,
not proof that no bugs exist.

## Unverified in this environment

- NVCC/CUDA-feature compilation and actual L4 BF16 forward/backward/update.
- Actual 1B peak VRAM, throughput, runtime, initialization and checkpoint costs.
- Real Colab runtime/Drive OAuth, FUSE behavior, available storage quota and
  persistence through a real Colab recycle.
- Long-run model quality, meaningful pretraining data exposure and any improvement
  from semantic features at matched compute/parameters/splits.

The notebook performs the first GPU checks and records actual outcomes. It never
substitutes a CPU pass for a CUDA PASS. Rust's admission formulas and checkpoint
sizes are algebraic estimates until measured. The original frozen M5 experiment
is unchanged; this record makes no new M5 result claim.

## Immutable implementation pin

The released notebook installs and runs source commit `ead9f0a56b2ea382f64626a0a4f2d4c25626e837`.
That commit contains the exact Rust/installer/transport implementation validated
above. A following delivery commit replaces only the notebook pin and records
strict notebook validation. The source snapshot contains an unreleased notebook
placeholder; open the final delivered notebook linked in the scale README.
