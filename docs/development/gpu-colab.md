# Bounded WSL supervision and Colab boundary

Maintenance does not authorize a GPU. M5 remains preflight-incomplete, and its
v4 one-attempt marker is already consumed. A later allocation needs a separate
explicitly authorized task; full runs also need a measured runtime/CU budget.
The frozen model/loader package is not rebuilt or reinterpreted here.

## Startup lifetime

The supported host entrypoint is
[launch-wsl-supervised.ps1](../../experiments/m5-gpu/scripts/launch-wsl-supervised.ps1).
It launches a hidden, independent Windows WSL client for **Ubuntu-24.04**, user
dasetwa. That client runs
[launch-l4-recovery.sh](../../experiments/m5-gpu/scripts/launch-l4-recovery.sh),
which retains systemd-run --user --wait until the bounded Python supervisor
finishes. The original invoking shell/tool may exit; its process lifetime is not
the job owner. A host PID/UTC receipt identifies the independent client.
The receipt means dispatch, not remote success.
A new receipt path in an existing directory is checked before dispatch.

Production bounds remain 1620 seconds of Python work, 180 seconds of release
reserve, systemd RuntimeMaxSec=1680 and TimeoutStopSec=120, KillMode=mixed and
Restart=no. The parent handles TERM and releases ownership; systemd kills all
remaining unit processes at the hard stop. No indefinite keepalive, daemon,
scheduled task, global WSL setting or linger was installed. Closing the
independent host client/host failure is still a failure mode, with bounded
best-effort release and explicit manual recovery if verification is missing.

| Design | Assessment |
|---|---|
| Detached user service alone | Previous failure showed user-session lifetime can terminate it; WSL systemd is not a host keepalive. |
| Foreground transient scope | Couples the workload more directly to the client and does not independently keep WSL alive. |
| User service plus independent bounded WSL client | Selected; combines unit deadlines/cleanup with a client whose lifetime survives the initiating tool. |
| Permanent loginctl linger | Would change host behavior beyond one task; unnecessary for the tested design and requires explicit approval if later needed. |

See [Microsoft WSL systemd](https://learn.microsoft.com/en-us/windows/wsl/systemd),
[systemd-run](https://www.freedesktop.org/software/systemd/man/latest/systemd-run.html)
and [loginctl](https://www.freedesktop.org/software/systemd/man/latest/loginctl.html).
The latter pages returned HTTP 403 to the maintenance web reader; local installed
systemd behavior is tested directly and the links remain documentation references.

## CPU verification and evidence

The actual launcher and l4-recovery.py run with a validated fixture transport
whose PATH excludes the real Colab binary. Fake archive/part/receipt sizes are
tiny. The fixture cannot request a real accelerator or read credentials.

- Two 310-second mock executions cover clean completion, original shell exit,
  and termination of a disposable PAM/logind session. Session identity/leader
  and scope were verified before terminating only session 11.
- A separate disposable initiating PowerShell helper is killed during the active
  mock; the independently launched WSL client remains alive and completes.
- Python deadline, systemd RuntimeMaxSec, handled TERM, task failure, missing
  archive, surviving endpoint, ownership, stale recovery and no second request
  are covered by CPU cases. The unrelated mock LIV session is never stopped.
- Per-stage UTC/PID/process-group/exit/signal records, service state and journals
  are retained. Transport descendants are reaped even after a successful leader
  exit. Unit ControlGroup is empty at completion, and no owned worker remains.

[Long results](../maintenance/2026-10-08/long-supervision-results.json),
[systemd cases](../maintenance/2026-10-08/systemd-cases.json),
[targeted review](../maintenance/2026-10-08/SELF_REVIEW.md) preserve details.
The earlier 40-second sleep test is retained as historical limited evidence.
Five minutes improves that coverage but is not proof of every 30-minute failure,
abrupt power loss or a passing GPU workload.

Reproduce the short actual-entrypoint matrix in dedicated WSL with
bash experiments/m5-gpu/scripts/check-wsl-supervision.sh. The long host/PAM
commands, receipts and scoped helper are recorded in the
[maintenance report](../maintenance/2026-10-08/REPORT.md); fixture preparation is
[supervision-fixture.py](../../experiments/m5-gpu/scripts/supervision-fixture.py).
Invoke PowerShell scripts from the local Windows checkout; the system's
RemoteSigned policy can reject a script on a WSL UNC path. No execution-policy
setting was weakened to run the local copy.

## Release semantics

Only a recorded owned alias/endpoint is eligible for release. Parsed inventory
is checked again immediately before a stop; unknown or reassigned endpoints
retain the marker and require manual ownership verification. A CLI stop and
inventory query are separate calls, so an adversarial concurrent reassignment
between them cannot be made atomic by this client. Do not share the owned alias
with another allocator while a task is active.
Post-stop inventory must show both absent; a vanished alias with a surviving endpoint fails safely.
Transport exit zero never proves remote success. A downloaded archive with remote
exit zero and final Rust gate PASS is mandatory, after release verification.
Missing/failed results remain failures. A leaked mock endpoint preserves its
ownership marker and an urgent manual-release record.

For actual unresolved ownership, run the documented status/stop/inventory/usage
commands in dedicated Ubuntu-24.04 only, against the verified PackTok owner.
Do not stop unrelated aliases or allocate another GPU to debug storage/lifetime.
Original protocol and failure evidence:
[M5](../../M5_GPU_TRANSFORMER.md),
[accounting](../../M5_GPU_BENCHMARK.md),
[recovery](../../experiments/m5-gpu/provenance/RECOVERY_STORAGE.md).
