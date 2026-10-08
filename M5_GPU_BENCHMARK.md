# M5 — Provenance, CPU preparation and GPU measurements

Status: `M5_BLOCKED` at the initial runtime preflight. CPU implementation/data
preparation complete; the failed L4 was released and no GPU quality result exists.
Design and limits: [M5_GPU_TRANSFORMER.md](M5_GPU_TRANSFORMER.md).

## Starting state

2026-10-08: Windows checkout clean at 3b304d2 (delivery-only commit after the
required f3b6c3fb41f46c93e701b585d859b0b9071d65a5 CPU freeze). M4 was two commits
ahead of origin; both were pushed and PR #4's freeze summary updated. Native
Ubuntu-24.04 checkout created without changing Ubuntu/MOOS. M5 branch begins
exactly at f3b6c3f. Linux Git reading the Windows checkout with autocrlf disabled
initially reported CRLF differences; with its matching autocrlf=true setting
it was clean. No historical content was edited or stashed. Native checkout uses
Git's Linux line endings; raw -text artifact data remains byte-exact.

Installed environment and no-allocation resource query are preserved in
[initial-environment.txt](experiments/m5-gpu/provenance/initial-environment.txt).
Initial Colab CLI 0.7.4: no active sessions, balance 167.55 compute units,
usage 0.00/hr and zero active assignments. This is a point-in-time observation,
not a future rate/balance promise. Rust 1.99.0, Cargo 1.99.0, Python 3.12.3,
uv 0.12.23 and Git 2.43.0 were already installed. MSRV 1.85.0 and CUDA toolkit
were absent in this dedicated environment at initial inspection.
CLI command help is preserved in provenance/; CLI installed source inspected
for execution, upload/download and stop behavior. No tokens/session metadata
were printed or committed. Existing Windows gh credential helper was used only
by native WSL Git to push M4; its credential output was consumed by Git.

## Preserved preparation failures

- crates.io's metadata API returned HTTP 403. The sparse registry and immutable
  crate downloads worked; pinned dependencies were fetched through Cargo.
- The dedicated checkout initially had no Git author identity. Its local identity
  was copied from the frozen commit's author, without changing global config.
- WSL had no rg installation; native grep/find are used as the available fallback.

Acquisition stdout/stderr and the frozen Cargo dependency resolution are retained;
large source data and weights are ignored, hashes/notices/manifests are tracked.
All corpus outcomes, tokenizer hashes, CPU/GPU tests, times and costs will be
appended when measured. Nothing below is an inferred GPU measurement.

## GPU accounting

CPU-preparation snapshot before the first attempt (current attempt accounting
is below in the preserved failure section):

GPU allocation count so far: 0. Allocated L4 time: 0. No GPU training or evaluation
has been run; no compute-unit cost estimate is claimed yet. The final preflight
accounting will use before/after CLI usage plus timestamped lifecycle logs.
Full final runs require a separate explicit budget approval after preflight.

CPU environment interruption: dedicated Ubuntu-24.04 stopped during preparation;
new launches returned Wsl/Service/CreateInstance/E_FAIL (error 6, step 2).
Cause is unconfirmed. No GPU was allocated. A targeted
`wsl.exe --terminate Ubuntu-24.04` followed by the prescribed launch recovered
the dedicated distribution on 2026-10-08 at 10:13:56 UTC. Ubuntu/MOOS was not
started or modified. Interrupted CUDA/setup/check attempts remain preserved;
CPU preparation resumes only after recovery.

Intermediate runner build failed on a duplicated module import; the library's
12 CPU tests already passed, and the import was removed before later verification.
After WSL recovery, the downloaded libcublas archive no longer matched NVIDIA's
published SHA despite having matched before the interruption. It is quarantined
with its observed hash; no corrupted bytes are used. Recovery redownloads only
that failed component. Cause of the changed cached archive is unconfirmed.

### Local physical-storage failure and recovery (2026-10-08)

C: reached 0 free bytes (119395831808 bytes used). The WSL virtual filesystem
reported ample logical space, but its host VHD could no longer grow. This caused
E_IO, startup failure and interrupted CPU builds/tokenizer preparation. No GPU
was allocated. An attempted deletion of the narrowly identified Windows Cargo
incremental cache was rejected by automatic command approval; it was not deleted.
The user freed space and authorized choosing the larger disk. The dedicated
Ubuntu-24.04 distribution was moved using `wsl --manage Ubuntu-24.04 --move
D:\PackTok-WSL`, after a successful official `wsl --export ... --format vhd`
backup under D:\PackTok-WSL-backup-20261008. A prior direct-copy backup could not
be compared to its locked source and remains explicitly unverified. The other
Ubuntu distribution was not started or modified. Exact host-space observations,
export hash and timings are in provenance/storage-host.json and
provenance/storage-migration.txt. Interrupted verification is not counted as a
pass; integrity checks and CPU verification are repeated after recovery.

### Corpus-v1 split preflight rejected before model outcomes

The original corpus-v1 contained 38447334 retained bytes, including 14608831
TRAIN code bytes, zero VALIDATION code bytes and 8212 TEST code bytes. The
source-local modulo counter restarted for every small Rust source file; most
files had fewer than twenty blocks. This was a preparation error, discovered
from composition metadata, before any held-out model scoring. Corpus-v1 raw
splits, its complete manifest, logs and completed tokenizer artifacts remain
preserved locally. Completed small tokenizer artifacts are archived under
artifacts/corpus-v1-preparation. The interrupted tokenizer log is retained.

Corpus-v2 uses one monotonically increasing block counter over the unchanged
canonical source order. The modulo twenty split and leakage guards are otherwise
unchanged. It writes fresh immutable corpus-v2/prepared-v2 paths and a new
corpus-v2-manifest.json. Tokenizers are freshly trained twice on its TRAIN only.
The corpus-v1 output is not a final M5 dataset/result. No GPU was allocated.

## CPU-ready primary evidence

[CPU_PREPARATION.md](experiments/m5-gpu/provenance/CPU_PREPARATION.md) contains the
37517499-byte retained corpus-v2 composition, all three raw hashes, all tokenizer/
mapping hashes, per-split token counts and encode/roundtrip timings. The complete
per-source/range/rejection evidence is corpus-v2-manifest.json. Both A and C
512-slot tokenizers trained twice byte-identically on TRAIN only. The new adapter
verifies every global mapping row, exact token sequence round-trips and represented
byte coverage; verify-prepared rechecks immutable raw/encoded/artifact hashes.

Root Rust 1.85.0: fmt, strict all-target/all-feature Clippy, 135 debug tests,
135 release tests and release build pass (verification/root-*-3.txt). The isolated
M5 Rust 1.99.0 crate has 18 CPU tests in debug/release, strict Clippy and formatting;
its logs are verification/transformer-*-9.txt / transformer-clippy-8.txt. Ten
CPU-mocked lifecycle cases cover success, upload/execute/download failure, wrong
hardware, unowned/stale sessions, unapproved full training and handled interruption
(verification/lifecycle-cpu-5.txt); the tenth case detects a surviving owned endpoint
even after its alias disappeared. These are simulation tests, not CUDA validation.

72 historical models / 6 tokenizers / 12 M3-versus-M4 identity pairs remain
byte-exact. The three old text-hash differences were investigated and explained
in verification/historical-text-differences-explained.txt; no historical weights,
tokenizer bytes or frozen source changed. Legacy validation-only fixes are intact.

Preserved additional CPU implementation failures: an obsolete fixture tensor name
failed the first finite-difference test and was corrected; current-Clippy API
suggestions were corrected in the isolated crate; one report/check command used
an incorrect relative working directory, and the newly created evidence file was
relocated into the intended repository. None is counted as a successful check.
No failed/preflight evidence was deleted.

Raw upstream licenses/notices, CLI help and verification logs contain original
CRLF/trailing whitespace. The first staged whitespace check is preserved in
verification/precommit-raw-whitespace.txt. Scoped Git attributes preserve these
raw evidence bytes and exempt their whitespace; source/docs retain normal checks.

## Frozen preflight package

CPU implementation/source commit: `a464e12cb27f447f51978cb5855e460859e32542`.
Draft stacked PR: [#5](https://github.com/DasEtwa/PackTok/pull/5), base
`m4-factorization-ablation`; not merged. The CUDA binary was compiled entirely
on local WSL CPU and passed the bundled-loader CPU reference check. Source and
prepared inputs are frozen by bundle-input-SHA256SUMS.txt, bundle-SHA256SUMS.txt
and frozen-source.json inside the archive. Sizes and portable runtime versions/
notices are retained. The elapsed CPU-preparation wall interval (including user
wait/storage recovery, not active CPU time) is cpu-preparation-duration.json.

Planned GPU-only work: CUDA initialization/reference, identical primary A/C
initialization, eighteen updates per variant with sixteen timed, TRAIN-prefix
scoring, fixed tiny overfit, safetensors write and fresh CUDA reload. No held-out
quality run or full training is authorized. Provisioning/upload/unpack/library
resolution/download are unavoidable session overhead; no remote ordinary
installation or Rust/nvcc compilation is planned. The wrapper caps the phase
and cleans up the owned L4 before local analysis.

## Initial L4 preflight failure — preserved, not passed

Attempt 1: provenance/l4-preflight-20261008T121550-498/. Its frozen source is
`a464e12cb27f447f51978cb5855e460859e32542`, packaged before the delivery commit
`c44c94127656bf2338a4b710eff15967f63ceafa`. The downloaded tar and small extracted
logs are retained; download hash/contents and RESOURCE_ACCOUNTING.json are durable.

Actual NVIDIA L4, driver 580.82.07, total device memory 23034 MiB, Ubuntu 24.04.4
LTS/kernel 6.6.122+. The ELF loader exited 127 before the Rust model/CUDA gate
started. Loader stdout is empty and stderr was not captured; the exact missing
remote library cannot be proved from this attempt. A local CPU reproduction
without cuRAND also exits 127 with a missing libcurand.so.10 message
(verification/runtime-missing-libs-reproduction.txt). The initial remote wrapper
also overwrote inherited library paths, which could hide the driver. These are
runtime compatibility hazards, not evidence that the Transformer itself failed.

The CLI execute command exited zero despite a Python RuntimeError (execute.txt).
The original lifecycle exited zero but never produced a CUDA PASS. This wrapper
contract is fixed: after confirmed GPU release, check-preflight.py requires both
remote exit zero and a final Rust gate PASS from the downloaded archive. The
real failed archive is rejected (verification/real-failure-rejected.txt). Eleven
CPU-mocked lifecycle cases pass (lifecycle-cpu-7.txt), including the misleading
CLI-success/remote-failure case and endpoint-survival detection. No allocation
was needed to fix/test this.

Resource boundaries: provisioning 4 s; status 2 s; active usage 1 s; upload 25 s;
remote execution 5 s; pre-transfer usage 1 s; download 1 s; stop 2 s. Allocation
request to stop 41 s, through verified cleanup 45 s. Model initialization,
training and evaluation never started; GPU compilation 0 s (compiled on WSL).
The five-second remote phase includes unpacking/environment/loader failure and
output packaging; CUDA setup itself was not reached/separately measured. No
checkpoint, step throughput, inference throughput or sampled peak VRAM exists.

Displayed balance stayed 167.55 CU; observed active CLI rate was 1.54 CU/hour,
then 0.00/hour with zero assignments after stop. Displayed delta 0.00 CU is not
proof of zero cost because billing delay/rounding is unresolved. Multiplying the
observed rate by 41 s gives a **proxy** 0.017538889 CU, not measured billing.
Full multi-seed time/CU cannot be estimated without a successful timed GPU gate.
The owned endpoint is confirmed absent; no L4 remains allocated.

CPU-only recovery: preserve complete bootstrap/loader stderr, retain inherited
CUDA-driver search paths including /usr/lib64-nvidia, and package the already
acquired pinned cuBLAS/cuBLASLt/cuRAND redistributable libraries/notices. Local
portable-loader checks use only the bundle plus the WSL driver, without reaching
back into toolkit paths. More transfer bytes are a documented tradeoff for
avoiding another remote library hunt/install. No model/tokenizer/data/config
change is made, no failed evidence is overwritten, no automatic GPU retry occurs.
A distinct bundle-v2 may be prepared; retry requires user authorization.

One progress-observation command had a quoting/working-directory failure and
read no provenance path; the corrected file-backed command observed the actual
phase. It did not change the GPU job or delay cleanup. All analysis/edits above
occurred after release verification. M5 is currently blocked at GPU runtime
preflight, not a completed quality comparison.

## CPU-prepared recovery candidate (not executed)

Source commit `c89cefe731e8cda6cdbdc06371f818749c6ddc01`. The distinct
local bundle-v2 resolves CUDA shared libraries from its own runtime, with only
the NVIDIA driver external; portable-loader-runtime-v2-v2.txt and
portable-loader-cpu-v2.txt pass on WSL CPU. No CUDA tensors are executed locally.
The complete input/archive hashes and sizes are bundle-input-SHA256SUMS-v2.txt,
bundle-SHA256SUMS-v2.txt and bundle-sizes-v2.txt; original package/evidence remains
unchanged. Model/tokenizer/split/configuration bytes are unchanged.

Proposed retry is one additional L4 preflight, maximum 1200 s (1080 s supervised
worker plus cleanup reserve); explicit retry authorization is pending. At the
previous observed rate 1.54 CU/hour, twenty minutes is a conditional proxy
0.513333333 CU, not a guaranteed price or billing measurement. retry-plan.json
also records archive bytes and a linear transfer projection using the measured
first 25-second upload; network/provisioning variability is unbounded by that
projection and the hard lifecycle deadline still applies. Full multi-seed time/
cost remain unavailable until a successful timed CUDA gate; full training is
not authorized by retry approval. No second allocation has occurred.

### Recovery transfer preflight on CPU

The v2 candidate is retained but not run. Its 456605835-byte archive
(bundle-sizes-v2.txt; verify exact bytes there) implies about 304 s upload under
a simple linear projection from the first 37548916-byte/25 s transfer. That
would exceed the original 180 s upload timeout. This was detected before any
second allocation. The v3 candidate extends only upload to 420 s while retaining
the global 1080 s worker and 1200 s cleanup-inclusive cap. All v2 checks/hashes
are preserved; no research configuration changed. Exact archive/projection
values in retry-plan.json supersede rounded prose estimates.

V3 CPU candidate (source 70ecf8e29dac240d9a197a5dbcf9dd54fb248358) retained,
never allocated. Inspection of installed official CLI contents.py lines 56–60
confirms full-file reading/base64 encoding. Recovery v4 transfers fixed 32 MiB
CPU-created chunks to reduce CLI memory amplification; no remote upload-size
limit is asserted as a measured fact. Verification/transfer-cpu-1.txt checks
mocked upload ordering and byte identity; lifecycle-cpu-11.txt preserves all
eleven lifecycle cases. The aggregate upload is still capped at 420 s, and the
worker/cleanup-inclusive cap is unchanged. Remote assembly is unavoidable
transport overhead. Source/parts/archive hashes and sizes are recorded per
candidate without overwriting earlier evidence; retry-plan-v2.json archives the
earlier plan. There has still been only one real GPU allocation.

Final retry candidate: v4/source `be47bc53ea6fd860320cc78cffa74e8aa51514b5`.
Local hashes/loader/reference/transfer checks pass; retry-plan.json records its
exact archive size, chunk count and conditional cost projection. Rust source,
primary config, lockfile and CPU-built executable are unchanged from the first
attempt (verification/recovery-model-byte-identity.txt). The first GPU
attempt is finished; retry authorization and
all full-run budget approval are still pending.
