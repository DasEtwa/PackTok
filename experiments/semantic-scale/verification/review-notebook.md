# Independent notebook/bootstrap review — initial findings

Reviewer: separate `scale_review_notebook` agent, 2026-10-23. Read-only production
review; no GPU, CUDA compilation or large model execution. Root and notebook
implementation agent were still completing local/Drive synchronization while
this review ran; their already assigned storage rewrite is not counted as a new
finding here. Placeholder source SHA is deliberately awaiting a final source
commit and is not counted as a defect.

## NB-1 — high: a fresh clone exits before pinned checkout

Notebook execution cell cloned with `git clone --filter=blob:none --no-checkout`,
then unconditionally ran `git diff --cached --quiet` before any checkout.
A fresh no-checkout clone has an empty index against a populated HEAD tree.
Consequently the staged diff command returns exit 1 and `check=True` aborts every
fresh Colab installation before CUDA build or training.

Independent offline reproduction with a temporary one-file Git repository:

```text
git clone --no-checkout SOURCE DEST
git -C DEST diff --quiet          -> 0
git -C DEST diff --cached --quiet -> 1
```

Expected: skip dirty-checks until a newly created clone has checked out the pinned
commit; keep dirty-worktree protection for existing initialized checkouts. Add
an actual local fresh-clone test rather than mocking Git's exit codes.

## NB-2 — medium: setup/configuration failures still destroy the runtime

The execution cell's `finally` unconditionally called `runtime.unassign()` when
`DISCONNECT_AFTER_RUN=True`. This also happened if the earlier guard rejected
`MOUNT_DRIVE=False`, or if Drive permission/mount failed before any persistent
failure record existed. It can discard local previous results despite the guard
claiming Drive is required for automatic disconnect.

A mocked Colab-module execution of the actual notebook cell reproduced both cases:
[probe evidence](review-notebook-original-probes.json). Both raised their expected
error and still invoked `unassign`. No real Colab/runtime call was made.

Expected: only automatic disconnect after verified persistent final synchronization,
or explicit user-authorized disposal of local output; preserve the runtime when
backup/setup failed. A parameter validation failure before any GPU work should
not destroy local output. Add failure-path tests for the actual cell/helper flow.

## NB-3 — low: notebook and Rust validation bounds differ

Initial `Settings.validate()` allowed microbatch 1..64 and steps 1..10,000,000;
Rust allows microbatch 1..32 and steps 2..1,000,000. Thus accepted notebook form
settings can fail after expensive installation/build instead of at validation.
Use the same bounds and test their boundary values.

## NB-4 — low: the bootstrap unit suite has a false failing pin assertion

`test_rust_install_is_pinned_and_preserves_proxy_environment` required literal
`"1.90.0" in command` list membership, but version-selection arguments use the
correct `+1.90.0`. The initial suite result was 8 passed, 1 failed. This is a test
bug, not evidence that Rust was unpinned; test each command's real argument shape.

## Static installation and integration observations

- CUDA root/SM89 environment names match locked cudarc/bindgen_cuda discovery.
- Official Ubuntu22.04/24.04 NVIDIA dev packages avoid driver replacement.
- Rust1.90.0, --locked, CUDA feature and experiment executable path match source.
- Prepared-data and run-plan identities are checked; no corpus is downloaded.
- GPU path requires a single L4 and never silently substitutes CPU/preset.
- Command watchdog targets only the process's new session, with SIGTERM/SIGKILL.
- Actual CUDA build, BF16 execution, VRAM capacity and Drive FUSE behavior remain
  unverified here and must be established by Colab execution.

Follow-up will independently verify production fixes and the completed storage
monitor before this review is closed. Reports preserve original failed evidence.

## NB-5 — medium: background mirror can publish a newer checkpoint than its logs

In the first local/Drive mirror implementation, `Mirror.sync()` copied metadata
before `transfer_checkpoint()` captured the source latest pointer. Training can
flush train step 2 and publish checkpoint 2 between these operations. The mirror
then publishes a complete, SHA-valid checkpoint 2 paired with metrics copied at
step 1. A runtime interruption before the next sync loses the committed step's
metric record, even though optimizer/cursor state remains resumable.

A deterministic interleaving injected after `copy_metadata()` reproduced a Drive
index at step 2 with a metrics log containing only step 1; see
[reproduction](review-notebook-sync-race.txt). Expected: capture the frozen source
pointer first, copy metadata after that capture, and copy/publish only the captured
generation. If Rust retention removes that image, retry a new complete snapshot
without advancing the destination pointer to a partial image. This preserves the
fact that Rust flushes the train row before publishing its checkpoint pointer.

The notebook implementation agent received this finding for a regression/fix.

## Independent follow-up — 2026-10-24

All five reproduced findings are fixed and independently verified:

| Finding | Implemented fix | Follow-up verification |
|---|---|---|
| NB-1 fresh installation abort | Fresh clone skips pre-checkout dirty-index checks; existing checkout still requires clean tracked/staged state | Extracted the actual notebook `checkout_source` function and exercised real local Git clone, clean rerun and rejected dirty rerun. |
| NB-2 unsafe runtime disposal | Disconnect requires GPU-phase entry plus verified backup, or explicit ephemeral-discard consent. Mount/config/backup failures preserve local runtime. | Actual execution cell mocked mount/setup failure: zero `unassign` calls; durable evidence in `review-notebook-followup-probes.json`. Predicate regressions cover verified backup and explicit disposal. |
| NB-3 settings bounds | Python uses Rust's batch 1..32 and steps 2..1,000,000 bounds | Boundary-validation tests pass. |
| NB-4 false pin test | Pin assertion accepts the actual `+1.90.0` version selector | Bootstrap suite passes. |
| NB-5 mixed checkpoint/log snapshot | Capture source pointer before metadata; copy only that frozen complete generation; missing/pruned image retries without publishing partial state | Original deterministic writer-advance interleaving now backs up step1+metrics1, then next final sync step2+metrics2. See `review-notebook-sync-race-followup.json`. |

The complete Python suite now passes **20 tests** (11 bootstrap, 9 storage),
retained in [follow-up test output](review-notebook-followup-tests.txt). Tests
include actual fresh-clone Git behavior, mocked complete pipeline success/failure,
SHA-verified transport and restore, corrupted payload rejection, unsafe path
rejection, interrupted/pruned-source copy preserving the last complete pointer,
configuration/local-newer restore guards, frozen checkpoint/log publication and
safe disconnect predicates. The original failed reproduction records remain.

The revised notebook writes Rust checkpoints and prepared data locally under
`/content`; Python mirrors only completed training generations and small metadata.
Drive readback and restore verify SHA-256, and destination latest is published
last. A declaration of at least 30 GB Drive working budget is required for the
1B preset; the free 15 GB Drive allocation is not presented as sufficient. Drive
quota errors stop owned subprocess work and preserve local results for recovery.
Preflight diagnostic model images are deliberately not mirrored. No model or
training logic moved into Python.

An actual CPU-only demo preparation/plan check also passed with default settings:
train 915, validation 7603, test 6579 records, so context128 is supported after
BPE. This checked preparation/planning only, not BF16 GPU execution or quality.

### Remaining scope limits

Actual CUDA compilation/BF16 kernels, 1B VRAM fit, Colab package availability and
real Drive FUSE durability cannot be established by these offline tests. The
notebook performs the GPU preflight and leaves failures/logs visible. Subprocess
watchdogs bound GPU phases (including TERM/KILL grace); installation and final
Drive transport are separate work and are not an end-to-end wall-time guarantee.
Use one notebook writer per run directory; this is not distributed training or
a multi-writer checkpoint service. The release must replace the deliberately
pending source-SHA placeholder with the immutable final implementation commit and
run the strict notebook validator before publication.

No reproduced critical/high/medium notebook/bootstrap/transport defect remains
open in this review. Only documentation/evidence was written by this reviewer;
production fixes were made by the implementation agent.

## Additional operational finding and closure — NB-6

Root's later real-Rust orchestration probe found another **high** fresh-run
blocker not covered by the original subprocess mocks: the helper passed
`/content/packtok-prepared/<RUN_NAME>` to Rust `prepare` without first creating
`/content/packtok-prepared`. Rust deliberately creates only the requested output
leaf, so a genuinely fresh runtime failed with ENOENT. Original failure evidence
is preserved in [the pre-fix operational log](probe-orchestration-cpu-before-fix.log).

The notebook helper now creates the parent directory explicitly, and its mock
prepare operation uses `mkdir()` without `parents=True` so the test catches a
regression. The reviewer independently reran
[the actual operational probe](probe-orchestration-cpu.py) after the fix:
[follow-up log](review-notebook-real-pipeline-followup.log). It runs real tiny
Rust FP32 preparation/preflight/training/evaluation with explicit `--cpu`, while
mocking installation/GPU discovery and using a local filesystem as the Drive
transport target. The first attempt reaches optimizer step 4. All local runtime
content is then deleted, and the second attempt restores the complete checkpoint
from the verified backup with canonical train steps `[1,2,3,4]`. Both attempts
passed and verified persistence. No actual CUDA or Drive/FUSE execution is implied.

The helper additionally binds notebook-source.json to the exact implementation
commit, so a run name cannot silently resume under different code even if its
serialized model plan happens to be unchanged. Source identity travels with the
small mirrored metadata.

**Final review disposition:** NB-1 through NB-6 are closed by source inspection
and independent executable checks. Twenty bootstrap/storage tests and the
real-Rust tiny operational/restore probe pass. No known reproduced
critical/high/medium notebook integration defect remains. GPU compilation,
BF16/L4 execution, 1B capacity/stability and real Drive behavior remain explicit
runtime validation work for the delivered notebook; they were not performed here.
