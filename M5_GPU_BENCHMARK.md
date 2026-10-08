# M5 — Provenance, CPU preparation and GPU measurements

Status: CPU implementation and preparation complete; no L4 allocation or GPU quality result.
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
