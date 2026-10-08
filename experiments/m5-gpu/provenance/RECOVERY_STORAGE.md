# M5 v4 recovery and persistent storage — 2026-10-08

This extends the existing M5 draft PR #5 (`m5-gpu-transformer`, base
`m4-factorization-ablation`) from clean delivery commit
`4ed7f0de96cdb90fec23903186838e5a73bb5578`. It does not restart M5. The original
L4 failure, accounting, bundles v1–v4, frozen scientific sources, corpus-v2,
tokenizers, mappings, architecture, seeds and schedule remain preserved.

## Authorization and policy change

The user explicitly authorizes one additional NVIDIA L4 recovery preflight,
with a total 1800-second allocation cap including shutdown. The new supervisor
allows 1620 seconds of provision/transfer/diagnosis/CUDA/download and reserves
180 seconds for bounded stop/inventory/usage. This supersedes the historical
20-minute retry proposal for this invocation only. Full training, new accelerators
and further allocations remain forbidden. The user's follow-up requires a real
WSL→Drive upload/download/SHA-256 pass **before** this allocation.

The earlier prohibition on Drive mounts/cloud weight uploads is superseded
only for scoped PackTok research backup. Primary persistence uses independent
WSL rclone copies into `MyDrive/PackTok/M5/`. Native Colab mounting is optional
and deferred; successful CUDA initialization never depends on it. No live GPU
is used for OAuth/Drive debugging. No unrelated Drive data is changed; no Drive
deletion or destructive sync is implemented. Secrets stay outside the repository
in the WSL user's protected configuration (directory 0700, file 0600).

## Existing bundle reused

Bundle v4 is 456605722 bytes, SHA-256
`05790377fcc9458def42ee083009cdb5fe9c387b3f0cabbd48cb10fc09200040`, fourteen
zero-padded parts with 33554432-byte full chunks. Its source identity is
`be47bc53ea6fd860320cc78cffa74e8aa51514b5`; its scientific model source and
15066728-byte CUDA executable are byte-identical to the earlier attempt. The
executable SHA-256 is
`d343a71a9ad88ce9bb14502ebec3f5273878f8914eb864ade46044149ac8b0b1`.
Pinned CUDA 12.4.1 runtime components and Candle 0.9.1 are unchanged. The
bundle contains no WSL NVIDIA driver; the host supplies `libcuda.so.1`.

The complete original outer/inner checksums, source checksums, exact binary
comparison, bundled-loader dependency resolution and exact regenerated CPU
reference pass in `recovery-20261008/cpu-check-20261008T145301-363/`. The v4
attempt/ownership markers were absent before the current authorization. Prepared
split/tokenizer/mapping hashes and original lifecycle/transfer mocks pass there.
New diagnostic/lifecycle scripts are a small independent transport overlay;
the original archive, remote.sh and historical source manifests are unchanged.
No model rebuild or runtime-stack replacement is required for the preflight.

## Preserved local infrastructure failures

An initial inline PowerShell→WSL shell command lost the intended library path
and failed the local loader check with missing libcurand. A file-backed command
resolved the unchanged library successfully. A fresh CPU-reference check needed
its output parent directory created; it never overwrote the retained reference.
Failed check logs remain in `recovery-20261008/`; the successful fresh checks
have a distinct subdirectory and are not inferred from those failed invocations.

rclone 1.75.1 was downloaded from the official pinned release, its archive
checked against the release SHA256SUMS, and installed in the dedicated WSL
user's `.local/bin`. Initial installer parsing and a config-specific unsupported
flag were corrected; raw OAuth output was kept only in protected local storage,
never printed or committed. Browser OAuth completed with the `drive.file` scope.
The real namespace setup then encountered Google HTTP 403 `rateLimitExceeded`
on the shared rclone OAuth project's API quota. No fixture upload/readback pass
was reached; no GPU was requested. A PackTok/M5 folder skeleton created before
the quota error is preserved. A dedicated user-created Desktop OAuth client is
required for reliable persistence; the user is being guided through its setup.
The shared-client warning is an additional long-term blocker: rclone's official
documentation says it retires during 2026. See [rclone Drive setup](https://rclone.org/drive/#making-your-own-client-id),
[Google Desktop credentials](https://developers.google.com/workspace/guides/create-credentials#desktop-app),
and [OAuth consent](https://developers.google.com/workspace/guides/configure-oauth-consent).

## Durable weight protocol

The active Colab VM writes the existing atomic/fsynced safetensors locally.
After diagnostic/checkpoint transfer, the supervisor stops and verifies L4
absence before any longer hashing, Drive operations, edits or analysis. WSL
syncs the downloaded artifact and directory entry, hashes it, then copies it to
a new random immutable run namespace. Every manifest contains source commit,
variant, seed, primary configuration hash, corpus/tokenizer hashes, byte size,
SHA-256 and completion status. Names are unique; `rclone copyto --immutable`
is used, never sync/delete. The entire artifact is downloaded again and compared
by SHA-256 and size, and the local source is rehashed. Only then is the completion
manifest published; it too is downloaded and verified. Incomplete candidates
retain an INCOMPLETE local journal and must not be treated as known-good.
Retention policy: keep every known-good image and failed candidate for this
bounded experiment; any later pruning requires a separate user instruction.

The CPU-only `recover_weights` example loads the original downloaded image and
the Drive-recovered image into the unchanged Transformer at different initial
seeds, compares all 4096 logits on [1,2,3,4,1,2,3,4], and rejects nonfinite values
or differences beyond 1e-5 + 0.001*abs(original). This complements the existing
on-CUDA fresh-weight reload gate. It changes no training/model behavior and
introduces no CUDA requirement to ordinary CPU/root builds.

## Exact training resume remains locked

Existing safetensors contain model weights only. Neither remote reload nor
Drive round-trip implies interrupted-training resumption. AdamW moments,
optimizer step, sampling RNG and data cursor are absent. Before **any** long
multi-hour M5/M6/M7 training, a bounded independent checkpoint implementation
must: (1) serialize named model/AdamW tensors and scalar steps plus RNG/cursor,
config/dataset/tokenizer/precision/backend version identities; (2) load/validate
them before mutation; (3) compare a continuous N+M run against N/save/load/M on
CPU and authorized CUDA with declared tolerances; (4) prove atomic-write,
partial-upload rejection, immutable retention and persistent recovery. The
current Candle AdamW state exposure must first be checked; if inaccessible,
state serialization requires a narrowly scoped backend adapter rather than
claiming resume support. No such training-backend change is part of this
recovery preflight, and no long-run approval is requested before it passes.

## New bounded lifecycle and diagnostics

`l4-recovery.py` checks original v4 checksums and scientific source identity,
CLI version, session ownership/inventory and CU balance before a single L4
request. The one-attempt marker is exclusive. Every CLI child runs in its own
process group under the remaining monotonic deadline; hung transports are
killed before bounded cleanup. Handled INT/TERM release the owned alias; a stale
owned endpoint is release-only, while unowned aliases remain untouched. Release
requires a successful parsed inventory with both alias and recorded endpoint
absent. Other sessions, including LIV, are retained. Abrupt host/power failure
cannot guarantee cleanup. Manual recovery in Ubuntu-24.04:
`colab status -s packtok-m5; colab stop -s packtok-m5; colab sessions; colab usage`.

`remote-recovery.py` verifies fourteen-part transport identity, records OS/kernel,
actual GPU/driver/VRAM, inherited/effective library paths, library inventory,
file/readelf/ldd and bundled-loader resolution. Loader stderr/status and optional
LD_DEBUG are preserved before entering the unchanged Rust gate. Handled early
loader/bootstrap failures still package a diagnostic archive. Wrong hardware or
loader failure never starts the model. Remote exit zero **and** final Rust gate
PASS are required after verified release; successful CLI transport alone is
insufficient. The existing workload stays eighteen updates per A/C (two warmup,
sixteen synchronized measurements), TRAIN-prefix accounting and fixed auxiliary
overfit/reload. No TEST/VALIDATION quality tuning is performed.

No full-run time/CU estimate is currently measurable without successful CUDA
throughput. The prior 1.54 CU/hour rate gives 0.77 CU for thirty minutes only
conditionally; it is not a guaranteed rate or bill. CPU-only fixtures are not
GPU results. The current pre-allocation storage blocker must be resolved before
this task can consume its one authorized allocation.

## Verified CPU delivery and explicit operational bounds

`recovery-20261008/checks-20261008T150645-347/` preserves 18 successful
orchestration/storage regressions, 18 isolated Rust CPU tests, root Rust 1.85
135 tests, formatting, strict CPU Clippy and both relevant release builds.
`delivery-20261008T150828-357/` adds strict CUDA-feature Clippy performed on
local CPU, the unchanged scientific-source guard, fresh sessions/usage and
separate current-task accounting. Starting and latest CLI observations remain
167.55 CU, 0.00/hour, zero active assignments and no sessions. Windows registry
inspection confirms Ubuntu-24.04's BasePath is `D:\PackTok-WSL`; no other
distribution is used. Current-task GPU allocation count/duration are zero;
cumulative historical count remains one. No new checkpoint or CUDA measurement
exists. The authorized recovery allocation is still unused.

Transport limits are infrastructure only: per-command pre-allocation checksum
90 s and CLI inventory/version/usage 30 s; allocation 180 s, status 30 s,
active usage 20 s. Aggregate v4 chunk upload is capped at 480 s (previous
candidate 420 s), each upload at most 120 s, always limited by the global
remaining 1620 s work deadline. CLI remote timeout is 1050 s and local
transport supervision at most 1070 s; remote unpack is 90 s, ordinary
diagnostics 30 s and the unchanged CUDA gate 900 s (previous remote work
allowance 540 s). The larger allowances consume the user's new total cap;
they add no updates or scientific workload. Download has at most two 60 s
attempts within remaining work time. Cleanup permits two 25 s stop attempts,
two 20 s inventory queries and one 20 s usage query, all inside the 180 s
reserve. Child process groups are killed on timeout/handled interruption so
unresponsive CLI children cannot keep cleanup waiting. Rclone transport uses
one high-level/low-level attempt, 15 s connection timeout, 60 s inactivity
timeout, and 180 s process cap. Its listremotes check is capped at 15 s.
GPU memory monitoring retains the existing 200 ms whole-device sample policy.

`write-recovery-readiness.py` seals the successful CPU log hashes, current
transport/configuration hashes and a real verified Drive completion/readback
into an exclusive PREFLIGHT_READY.json. The supervisor rejects absent/stale
readiness before requesting CUDA. That readiness file has deliberately not
been created while the real Drive fixture is blocked. The authorizer for a
dedicated Desktop client reads only a user-supplied local JSON outside either
checkout, creates a separate remote, and keeps all raw OAuth output private.
No client credentials are embedded in repository scripts.

## Dedicated OAuth and real storage verification, 2026-10-08

The user authorized agent-assisted setup in the signed-in browser. The Drive
API was verified enabled in the existing PackTok Cloud project. A Desktop
client named `PackTok WSL Backup` was created; the user's own account was added
as the sole test user. The consent request was restricted to `drive.file`.
Google initially rejected consent because no test user existed; that rejection
was preserved here and resolved through the normal test-user configuration.
The app remains External/Testing. This bounded verification does not establish
long-term unattended token validity; publication/token-lifetime readiness must
be addressed before future long runs.

The browser JSON download did not produce a completed download event in two
bounded attempts. The visible creation-dialog fields were instead transferred
through a one-shot loopback form to protected WSL storage, without printing
secrets. `receive-drive-client.py` binds only 127.0.0.1, uses an unpredictable
route, validates the local Origin and bounded field lengths, refuses to replace
an existing file, writes with fsync/0600 inside a 0700 directory, and closes
after success or a 300-second limit. Credentials remain outside both checkouts.
The dedicated rclone remote is `packtok-drive-own`; the old shared-client
configuration is retained. No credentials or raw OAuth logs are uploaded.

`cpu-final-20261008T155824-559` was interrupted during a project-context change
before its fixture completed. Its INCOMPLETE journal is retained. A fresh run,
`cpu-final-20261008T160630-344`, completed upload, full download, size/SHA-256
comparison, local source rehash, completion-manifest upload and full manifest
readback. Its fixture run is
`m5-preflight-fixture-ff72854580914442a6b6848b428a1414`, 53 bytes, SHA-256
`5e79279a52956a9e2b16925d6b85bb0870eeaf9a26c34c70d028546a3ec35a83`.
The completion records source commit 74d91332d6612b1e0b8a74734409319a1b542861
and the frozen configuration/dataset/tokenizer hashes. The known old folder
skeleton was preserved as `PackTok/M5/preflight/bootstrap-shared-client-20261008`
through a metadata-only move/rename. No Drive files were deleted.

The repeated CPU verification in this successful run passed, and
`cpu-check-20261008T160850-494` independently reverified the original fourteen
bundle parts, all outer/inner/source hashes, executable identity, driver
exclusion, local loader, exact CPU reference and original transfer/lifecycle
mocks. PREFLIGHT_READY.json now seals the real fixture and successful CPU
evidence. Immediately before dispatch, CLI 0.7.4 reported no active sessions,
167.55 CU, 0.00 CU/hour and zero active assignments. No GPU was allocated by
these preparation commands.

The first detached `nohup` launcher returned a PID, but a subsequent process
inventory found no supervisor and no recovery run/attempt/ownership marker.
Colab still reported no active sessions. This launch consumed no allocation;
the precise local process-termination cause was not established. To preserve
supervision across terminal/context changes, `launch-l4-recovery.sh` now uses a
transient Ubuntu-24.04 user systemd service, with no restart. A synthetic sleep
service was verified active from a separate WSL command after dispatch.
The service sends handled TERM after 1680 seconds and kills remaining
processes after at most another 120 seconds; the Python supervisor retains
its tighter 1620-second work deadline and 180-second cleanup reserve. Abrupt
host failure still cannot guarantee release. The explicit manual recovery
commands above remain required if release cannot be verified.

## Consumed request, verified release and launcher lifetime correction

The recovery request from the preceding readiness snapshot is preserved in
`../provenance/l4-recovery-abafb6d8f2eb4e70a6ff3c3447bf917f/`. Its CLI allocation
log printed `Session READY`; the local user service was stopped at
2026-10-08 16:14:28 UTC and delivered handled SIGTERM while the allocation
child was still running. The service journal confirms the stop and supervisor
exit 1; the user-manager configuration reported Linger=no. The local lifetime
boundary, rather than a Rust/CUDA result, is the blocker. Microsoft's
[WSL systemd documentation](https://learn.microsoft.com/en-us/windows/wsl/systemd)
also states that systemd services do not keep a WSL instance alive. The exact
external shutdown trigger is not independently attributed beyond the recorded
service stop/user-session behavior.

The first short service probe was insufficient. The corrected launcher uses
`systemd-run --user --wait` so the foreground WSL client remains open for the
entire supervised service lifetime. No linger/global WSL setting was changed.
`wsl-lifetime-20261008T161852-470` is a real CPU-only regression: the service
was still active after 20 seconds and completed the full 40-second workload
with exit zero. The 18 existing recovery regressions were rerun successfully.
If the supervising client/host abruptly disappears, guaranteed cleanup remains
impossible; handled service termination invokes the existing bounded release.

The owned session was stopped and its absence verified before analysis/edits.
Resource accounting records one request, no captured endpoint, cleanup=true
and request-to-cleanup verification 8.439486265182495 seconds. CLI balance
167.55 CU before/after is rounded output; actual billed use and exact server
allocation duration are unknown. No active rate was captured. The 1.54 CU/hour
historical conditional proxy gives approximately 0.00361 CU for the measured
local interval, with the same uncertainty as the earlier 0.77 CU window proxy.
No remote command, loader gate or checkpoint ran; no remote archive was
technically available. Local lifecycle evidence is backed up separately using
the verified Drive pipeline after release. The v4 exclusive attempted marker
is retained and the one-request authorization is exhausted. No second request
or full training is permitted by this task.

OAuth remains External/Testing. Google's
[refresh-token expiration documentation](https://developers.google.com/identity/protocols/oauth2)
specifies seven-day refresh tokens for this status with scopes beyond basic
profile/login; `drive.file` is such a scope. Stored files remain durable, but
the current authorization must be renewed normally or the app made production
ready before relying on unattended access beyond that lifetime. No OAuth
bypass or additional scope is introduced. Current experiment state:
**M5_BLOCKED — INFRASTRUCTURE**.
