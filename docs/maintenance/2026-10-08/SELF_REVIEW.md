# Targeted maintenance self-review

Scope: repository/workspace boundaries, preservation, documentation claims/links,
credential hygiene, actual WSL supervision, Drive restore and GPU release
semantics. This is a fresh self-review of the maintenance diff, not a second
model-mathematics review. No delegated reviewer or new GPU was used.

## Confirmed findings fixed in maintenance

| Finding | Severity and reproduction | Expected / prior actual | Fix and regression |
|---|---|---|---|
| MNT-1: successful transport without result archive | High; fake download exits zero without creating the archive | Failure required; prior supervisor could return zero after release | Missing archive explicitly fails. `test_transport_success_without_archive_fails`; remote-exit/gate checks retained. |
| MNT-2: transport descendant survives its leader | Medium; fake successful execute spawns sleep 600 | No owned transport remains; prior successful wait did not reap the group | Reap the owned process group on every command exit. `test_success_reaps_transport_descendants`, actual systemd signal/long cases and empty-cgroup evidence. |
| MNT-3: release relies on earlier alias ownership | High; fake inventory adds a foreign endpoint under packtok-m5 after work, or allocation fails before recording an endpoint | Never stop an unknown/reassigned owner; prior release called stop by alias without a fresh ownership check | Fresh bounded inventory before stop, refuse unknown/changed ownership, retain marker and manual-release evidence. `test_reassigned_alias_is_not_stopped`, `test_unknown_allocation_endpoint_is_not_stopped`. |
| MNT-4: invalid receipt discovered after dispatch | Low; supply an existing receipt or missing parent directory | Refuse before dispatch; prior host launcher could lose its dispatch receipt after starting a job | Validate a new receipt path in an existing directory first; PowerShell rejection probes and final real CPU dispatch. |
| MNT-5: active historical report links missing delivery target | Low; PRE_GPU_CODE_REVIEW.md references absent DELIVERY.md | Active navigation resolves; historical report bytes must stay fixed | Add a dated navigation repair linking existing evidence, without rewriting the old report. |

Locations: [supervisor](../../../experiments/m5-gpu/scripts/l4-recovery.py),
[host launcher](../../../experiments/m5-gpu/scripts/launch-wsl-supervised.ps1),
[recovery regressions](../../../experiments/m5-gpu/scripts/test-recovery.py),
[delivery repair](../../../experiments/pre-gpu-fix-20261007/DELIVERY.md).
Final recovery suite: 22 passing tests. Prior 18/20-test logs are retained.
The six actual-entrypoint systemd cases pass again after the release guard.
The final guarded entrypoint also completes 310.115171 seconds of CPU work,
with verified release, empty cgroup, no descendant and no surviving host client.

## Preservation and scope checks

The initial native-byte inventory covers 785 paths. Only explicit maintenance
documents/guards/supervisor scripts may differ; Rust crates, scientific runner,
corpus, mappings, model/tokenizer artifacts and old raw logs must match exactly.
No file removal or experiment-path migration is accepted. The Windows pending
92-path recovery copy is reconciled only after comparing its Git-filtered blobs
to the preservation commit. Both copies and remote are fast-forwarded to the
delivered branch, with no reset, clean or force push.

Runtime-web is inspected read-only and compared against the protected initial
HEAD/status/file-hash oracle. No unrelated instructions were inherited.
Ubuntu/MOOS remains outside execution. The root Rust CPU and isolated M5 checks
pass; historical compatibility checks pass on Windows and Linux. The M4 review
does not omit the unfavorable same-schedule factorization or inconclusive
interaction. README labels roadmap hypotheses and M5's missing GPU evidence.

The full pre-maintenance reachable-history scan found no credential candidate in
707 blobs; current/staged scans add path/blob checks before commits. No private
OAuth JSON, rclone config, token, CUDA transfer bundle or large checkpoint is
staged. Pattern scans are heuristic, supplemented by inspecting the changed
files. New external receipts contain only approved research identity metadata.
Ignored fixtures/caches are retained locally. No credential rotation is needed
on the evidence observed; actual exposure would require a separate approved
rotation/history-remediation plan.

All active relative Markdown file links resolve at final audit. Twenty-two
pre-existing archival links refer to the original BRAIN namespace or preserved
before-snapshot structure; these are reported without altering historical bytes.
No old numerical measurement is changed. New reports point to preserved raw
checks, including failed setup attempts and the interruption attempt that did
not actually interrupt an active worker.

## Remaining evidence boundaries

- M5's real CUDA loader/Rust GPU gate is unresolved. CPU/mock passes do not prove
  GPU training. Maintenance makes zero real GPU requests and does not merge PRs.
- A five-minute mock is not exhaustive proof for every 30-minute workload or
  host crash. Independent-client/host failure may require manually verified owned
  release. No permanent linger/background setting was installed.
- Alias inventory and stop are separate CLI calls. The new guard rejects observed
  reassignment but cannot provide an atomic endpoint stop against concurrent
  adversarial reassignment. The job's alias must remain exclusive.
- Unknown endpoint or surviving/renamed endpoint retains the ownership marker,
  refuses automatic stop/new allocation and reports release unconfirmed.
- Google OAuth remains External/Testing. Production is selected but needs explicit
  human publishing approval and the console's Branding prerequisite. Only
  drive.file is declared; no broader grant/client replacement was used.
- Current upload/readback/fresh-destination restore proves today's byte integrity,
  not seven-day/multi-month authentication, another account or an OS migration.
  Nine restore regressions cover corrupt/incomplete identity and path/size guards.
  No real GPU model checkpoint exists; weights-only storage is not exact resume.

These are retained limitations, not assumed successes. Final audit and delivery
records alongside [REPORT.md](REPORT.md) provide the machine-readable outcomes.
