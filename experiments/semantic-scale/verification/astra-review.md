# Independent Astra bug review, 2026-10-24

Reviewed implementation: `8c5b0fb00e6eca4c5bdc6e0ace7a0b8437d25b28`, branch
`feat/semantic-packs-v1`. The notebook at that revision pins implementation
`ead9f0a56b2ea382f64626a0a4f2d4c25626e837`. Line references below refer to the
reviewed revision, before the fixes made by the coordinating agent.

Scope: independent source inspection of `src/{model,optim,data,config,runner}.rs`,
`tools/{colab_bootstrap,colab_storage}.py`, the notebook, and relevant pinned
Candle/cuDARC internals. Read AGENTS, IDEA, STRUCTURE and README first. No prior
review report was used to select the findings. The reviewer changed no production
source, branch or commit. Only this report and minimal repro/log files were added.
No GPU, external data download or expensive CPU model experiment was used.

## Confirmed defects

### A1 — P2: watchdog fails to kill descendants after the session leader exits

Location: `tools/colab_bootstrap.py:134-152`, particularly the
`process.poll() is None` guard in `kill()`, together with stdout iteration at 170.

The timeout sends SIGTERM to the subprocess group. If the leader exits but a
child ignores SIGTERM and keeps the inherited stdout pipe open, the reader blocks
waiting for EOF. The forced SIGKILL timer fires after 15 seconds, observes that the
leader has exited, and does not kill the still-live process group. The command
therefore never reaches its TimeoutError or the pipeline's cleanup/backup path.
Build and installation commands can create descendants; the watchdog promises to
bound those commands, not just the leader PID.

Reproduction: `python3 experiments/semantic-scale/verification/astra-repros/reproduce_python.py`.
The `descendant_watchdog` fixture creates a single owned session whose leader
accepts SIGTERM while its child ignores SIGTERM. With `timeout=1`, the call was
still blocked after 18 seconds, beyond the 15-second escalation grace, and the
child remained alive. The fixture unconditionally kills its own process group
and joins the thread afterward; it leaves no intentionally running descendant.
Expected: the process group is terminated and the call returns TimeoutError.
Actual evidence: `astra-repros/python-repros.log`.

Fix direction: track the owned group independently of leader liveness, escalate
against that group even after leader exit, and ensure inherited pipe readers can
finish. Test both timeout and cancellation/exception cleanup.

### A2 — P2: an incomplete first Drive backup rolls back newer local metrics

Location: `tools/colab_storage.py:218-240`, specifically the no-remote-checkpoint
branch at 239-240. `src/runner.rs:407-464` accepts the resulting incomplete metric
history when resuming an otherwise valid checkpoint.

A mirror may already contain metadata through step N before its first checkpoint
copy has completed. If that copy fails and the local runtime survives, the local
run may have a complete newer checkpoint and newer metrics. On retry,
`Mirror.restore()` checks local-vs-remote step ordering only when Drive has a
`latest.json`. With no remote checkpoint pointer it unconditionally copies stale
Drive metadata over the current local metrics, while leaving the local checkpoint
in place. This loses canonical history before Rust can archive it.

Reproductions:

- `reproduce_python.py`: Drive has a step 1 metric and no checkpoint; local has
  step 2 weights and step 1/2 metrics. Restore removes the step 2 metric.
- `python3 experiments/semantic-scale/verification/astra-repros/reproduce_cli.py`:
  prepares a tiny corpus, performs an actual CPU preflight/two-step train, restores
  the stale Drive metadata snapshot, then invokes Rust `train --resume --cpu`.
  The command exits 0; checkpoint and summary both report step 2, but the canonical
  training metric steps are only `[1]`.

Expected: keep the newer complete local run intact, or require an explicit
recovery path that preserves its original metadata. Actual evidence:
`astra-repros/python-repros.log` and `astra-repros/cli-repros.log`.

### A3 — P2: the CUDA installer omits a required cuRAND linker dependency

Location: `tools/colab_bootstrap.py:219-224` and `257-259`.

The fallback package list installs nvcc, cudart/nvrtc development libraries,
cuBLAS and CCCL, but omits `libcurand-dev-12-8`. The toolkit readiness predicate
checks only nvcc and two headers; it can accept a toolkit without any of the
required linkable libraries.

This is a dependency-contract defect with a conditional trigger: a fresh or
partial toolkit that does not already contain cuRAND. It is not a claim that
current Colab images universally lack cuRAND. An actual Colab GPU build was not
run in this review.

Evidence from the locked dependency graph:

- Candle core 0.9.1 and ug-cuda 0.4.0 enable cuDARC 0.16.6 `curand` plus
  `dynamic-linking`; `astra-repros/cudarc-features.log` records the exact
  `cargo tree --locked --features cuda -e features -i cudarc` output.
- cuDARC's build.rs emits dynamic links for `cuda`, `nvrtc`, `curand`, `cublas`
  and `cublasLt` with the enabled features. Candle kernels 0.9.2 additionally
  links `cudart` and the C++ runtime. cuDNN/cuSPARSE/cuSOLVER/NCCL are not enabled.
- Candle's CUDA device construction initializes a cuRAND generator regardless
  of this model's seeded host-side initialization.
- `reproduce_cuda_admission.py` extracts the original readiness function from
  the reviewed commit and calls it on a temporary tree containing only the
  three checked files. It returns true with zero library files. See
  `astra-repros/cuda-admission.log`.

Expected: install all required development libraries and validate a linkable
CUDA toolkit before the expensive Rust CUDA build. Actual: missing cuRAND can
survive setup and fail at `-lcurand` resolution. GPU execution and system package
installation remain untested here.

### A4 — P2: the Rust L4 gate admits NVIDIA L40/L40S

Location: `src/runner.rs:182`: `values[0].contains("L4")`.

The direct Rust CLI gate accepts any GPU name containing L4. Both NVIDIA L40 and
L40S satisfy that predicate, contrary to the explicit L4-only contract. The
notebook's earlier Python gate is stricter and prevents this through the normal
notebook entrypoint; direct CLI use remains affected.

Reproduction: `reproduce_cli.py` supplies a temporary `nvidia-smi` executable
returning `NVIDIA L40, GPU-fixture, 46068, 46000` to the non-CUDA binary and invokes
preflight on a valid tiny plan. Expected: reject the unsupported hardware name.
Actual: execution advances past the hardware filter and reports `CUDA support is
not compiled; build with --features cuda`. No GPU was opened or allocated.
Evidence: `astra-repros/cli-repros.log`.

### A5 — P2: a stale mirror can roll back and delete a newer Drive checkpoint

Location: `tools/colab_storage.py:147-152`, `185-194`, and `242-254`.

Two active runtimes using the same run name/identity can both adopt the Drive run
marker. The original transfer code rejects identity differences but does not
reject an older step or a divergent checkpoint at the same step. A late backup
from the older runtime overwrites `latest.json` and deletes the previously
published newer generation. Metadata is copied even earlier, so adding a guard
only inside `transfer_checkpoint` would still permit stale metric replacement.

Reproduction: `reproduce_stale_mirror.py` creates two independent local sources
for the same Drive run. Runtime A successfully verifies step 2, then runtime B
backs up step 1. Actual result: Drive reports step 1, its step 2 directory has
been deleted, and B reports `verified=True`. See `stale-mirror-before.log`.
Expected: reject stale/divergent publication before replacing either checkpoint
or metadata and preserve the existing newer generation.

Scope: this trigger requires overlapping runtimes or another out-of-date backup
writer. Local Rust file locks do not establish exclusion across Colab machines.
A monotonic-step check and rechecking the expected destination pointer improve
fail-closed behavior; they cannot create an atomic distributed compare-and-swap
on Drive/FUSE. Concurrent writers to the same persistent run must remain an
explicit unsupported operation unless a separate reliable locking mechanism is
introduced.

## Additional review observations

The coordinating reviewer independently found the valid final JSONL row without
LF case in `reconcile_metrics`. Inspection confirms the original code writes that
row unchanged and immediately appends `resume_boundary`, joining two JSON objects
without a line break. It is tracked by the coordinating agent, not claimed here
as a separately discovered finding.

A failed initial notebook fetch after `git clone --no-checkout` can leave an
existing checkout with an empty index; rerunning the notebook then rejects it via
`git diff --cached --quiet`. This is a recovery robustness observation, not one of
the primary findings above. No destructive reset or worktree cleanup is advised.

No additional reproducible mathematical defect was found in the inspected layer
recomputation, direct gradient accumulation, causal window reconstruction,
FP32-master update, or sharded-checkpoint consistency checks. This statement is
limited to inspection and tiny CPU fixtures; it establishes neither BF16 CUDA
execution nor actual 1B fit, throughput or successful large-model training.

## Baseline validation

Before production fixes, the existing suite passed independently:

- Rust 1.90.0 locked debug tests: 25 unit tests plus one tiny CLI integration test;
  raw output in `astra-repros/rust-tests.log`.
- Python standard-library suite: 20 tests;
  raw output in `astra-repros/python-tests.log`.

Commands used the session's isolated CARGO_HOME/RUSTUP_HOME under `/workspace`.
Passing existing tests did not cover A1-A4. Repro logs retain the original failing
behavior and must not be overwritten when validating fixes. The implementation
source commit must be repinned in the published notebook after the fixes land.

## Independent follow-up on fixes

The coordinating agent changed production files; this reviewer inspected the
diffs and reran the original small repros without changing production code.
The first follow-up suite passed 27 Rust unit tests plus the existing tiny CLI
integration test, and 25 Python tests. Logs are `rust-tests-after.log` and
`python-tests-after.log`. These counts describe the first fixes (A1-A4 plus the
coordinator's JSONL correction), before later A5/checkout recovery tests.

`python-repros-after.log` now shows preservation of the local step 2 metric and
TimeoutError after watchdog escalation. Its `child_alive` field checks `/proc`
existence and includes already-killed zombies; it does not imply continued
execution. The additional closed-stdout descendant repro records state `Z`,
confirming that path also stops the descendant (`watchdog-closed-pipe-after.log`).

The added cuRAND dependency and readiness checks match the locked dynamic-link
feature closure. Toolkit checks do not claim a successful actual CUDA build.
The Rust name filter now accepts only `NVIDIA L4` / `NVIDIA L4 GPU`, and the
JSONL recovery path supplies a missing final newline before appending its marker.

### Final counter-review result

The final reviewed working-tree fixes pass **29 Rust unit tests + 1 tiny CLI
integration test**, and **29 Python tests**. See `rust-tests-final.log` and
`python-tests-final.log`. The original repros now show:

- local step 2 weights retain both training rows `[1, 2]` and resume successfully;
- L40 is rejected by the explicit L4 hardware gate (`cli-repros-after.log`);
- the hung-descendant watchdog returns TimeoutError and terminates descendants;
- a stale mirror is rejected with `verified=False`, while remote step 2 and its
  complete generation remain intact (`stale-mirror-after.log`);
- the interrupted empty no-checkout clone resumes successfully
  (`incomplete-checkout-after.log`).

Inspected the additional canonical-history validation: Rust now requires exactly
one ordered `train` record for every saved step before replacing the canonical
metrics log, and preserves the original log when rejecting invalid history.
The newline repair remains correct for a complete final JSON object without LF.

Inspected the A5 defenses before metadata copy, inside checkpoint transfer and
immediately before pointer publication. Existing stale/divergent destinations
are rejected before metadata changes; a detected pointer change during a shard
copy prevents publication and retention. The README explicitly limits a
persistent run to one active runtime. These checks do not make concurrent Drive
writers safe: a remote writer can still race between a check and a write, and
metadata has no distributed atomic transaction. This remains a documented
unsupported case, not an outstanding promise of global writer exclusion.

No additional blocking finding was identified in the final targeted counter-
review. Actual CUDA compilation/execution, BF16 device behavior, 1B memory and
throughput remain unvalidated hardware questions. The notebook must be repinned
to the new immutable implementation commit when the coordinating agent commits
these fixes.
