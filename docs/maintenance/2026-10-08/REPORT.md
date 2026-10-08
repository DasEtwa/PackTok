# PackTok repository maintenance — 2026-10-08

This is maintenance, not M6. M5 remains **GPU preflight incomplete**. No model,
tokenizer, training schedule, corpus, pack allocation or evaluation split changed.
No new GPU was allocated, no stacked PR merged, and no published history rewritten.

## Context and preservation

The authoritative repository is [DasEtwa/PackTok](https://github.com/DasEtwa/PackTok),
branch `m5-gpu-transformer`, existing [draft PR #5](https://github.com/DasEtwa/PackTok/pull/5)
based on `m4-factorization-ablation`. Initial Windows and native WSL HEAD was
`74d91332d6612b1e0b8a74734409319a1b542861`. The native implementation checkout is
`/home/dasetwa/projects/PackTok`, Ubuntu-24.04, user dasetwa; its VHD lives under
`D:\PackTok-WSL`. Windows checkout is `C:\Users\DasEtwa\PackTok`.

The initial inventory found 699 tracked files and 92 pending recovery paths
(six modified, 86 new), totaling 148,996 pending bytes. Both copies had identical
logical contents; 95 byte differences across the complete 785-path inventory
were existing checkout CRLF/LF differences. Pending recovery contents matched
byte-for-byte. No logical conflict was discarded. A protected external snapshot
of the pending paths was retained before edits: SHA-256
`207c1789b6acc15f7a58d547114e6c01c58df1d60dc97a8ab7c7c4f62358cb71`.
The public [initial inventory](initial-inventory.json) records path/size/hashes;
private machine-state details remain outside Git.

Commit `1412541f40f6d11a2fdb674c2d832318473cac0e` preserves all 92 pending M5
recovery paths. It was pushed before restructuring. This retains the previous
loader/lifecycle failures, working Drive receipts and earlier limited lifetime
test. Later maintenance changes are separate logical commits on the same branch.

The saved PackTok project/current task use the correct checkout. The previous
recovery task had an incorrect runtime-web cwd, but a read-only search found no
PackTok recovery work there. Its existing HEAD, status and file hashes are the
unrelated-work oracle. No runtime-web file was modified. Ubuntu/MOOS was not used;
only Ubuntu-24.04 is used for this task's Linux work.

## Documentation and repository hygiene

The [plan](PLAN.md) was written before restructuring. Before: 24 root Markdown
documents and no docs index. After: [docs index](../../README.md), architecture,
M0–M5 milestone navigation, research hypotheses/results/limitations and developer
build/test/contribution/GPU/storage/OAuth guides. Root reports and experiment
paths remain where historical commands/manifests expect them. README now gives
an English introduction, representation versus output-head distinctions, tested
CLI commands, balanced milestones, evidence navigation and an explicitly
hypothetical roadmap. [Research results](../../research/results.md) is the
authoritative synthesis; it links original measurements rather than replacing them.

M4's same-schedule factorization result remains unfavorable, its tokenizer
difference small on the tested corpus, and its interaction inconclusive. Matched
analytical MAC runs obtain more updates; they do not match wall time or token
exposure. M5 still has no passing CUDA/Rust GPU gate or GPU model-quality result.

No historical model, tokenizer, mapping, corpus, raw report or failed result was
moved, deleted, regenerated or normalized. The byte audit uses initial native
SHA-256 values. The inventory includes 84 model files, 16 tokenizer files and
seven mapping files, including preserved copies. Existing edits are restricted
to the documented maintenance allowlist; all scientific sources are unchanged.
The canonical compatibility probe also loads/reserializes 72 models, six
tokenizers and compares 12 historical M3/M4 model pairs on both platforms.

One active pre-GPU link lacked its DELIVERY.md target. A new dated
[navigation repair](../../../experiments/pre-gpu-fix-20261007/DELIVERY.md) links
the actual existing evidence without fabricating a historical log. Original
BRAIN namespace links and archived before-snapshots contain 22 pre-existing
unresolvable relative links; those byte-sensitive historical copies remain
unchanged and are reported separately from active navigation. No invasive path
migration was attempted.

The reachable-history scan checked 707 blobs before new maintenance commits:
no credential candidate and no blob over 5 MiB. Current/staged scans check token,
secret and private-key patterns without printing values. No actual exposed secret
was found, so no credential rotation or history rewrite was initiated. Pattern
scanning is heuristic, not a guarantee against every secret encoding.
Ignore rules now guard rclone/client OAuth files, session data, checkpoints and
disposable transfer/mock caches. Ignored local material was retained, not deleted.
Small versioned evidence remains tracked. No unexpectedly large tracked artifact
requires Git LFS here. [External artifact receipts](external-artifacts.json)
contain scoped namespaces, completion/payload hashes and sizes, without secrets
or authenticated personal Drive links.

## CPU and compatibility verification

| Environment | Commands and result |
|---|---|
| Native Ubuntu-24.04.5, Rust 1.85.0 | Root fmt; strict workspace/all-target/all-feature Clippy; debug tests (135), release tests (135); release workspace build: PASS. |
| Windows 10 build 19045, Rust 1.98.1 | Same five root gates; debug/release tests (134 each): PASS. One platform-specific test accounts for the count difference. |
| Isolated M5, Rust 1.99.0 | fmt; strict CPU Clippy; locked debug/release tests (18 each); release recovery-weight example build: PASS. No CUDA claim. |
| Historical CPU freeze | Direct existing compatibility probe on Linux 1.85 and Windows 1.98: 72 models, six tokenizers, 12 pairs: PASS. |
| Infrastructure | Original 11 lifecycle cases, original transfer checks, new recovery/restore/systemd cases described below: PASS. |

Exact commands, tool versions and exit statuses:
[Linux](linux-results.json), [Windows](windows-results.json),
[test counts](test-counts.json), [Linux compatibility](compatibility-linux.txt),
[Windows compatibility](compatibility-windows.txt).
The compatibility command is:

```sh
cargo +1.85.0 run --locked --release \
  --manifest-path experiments/pre-gpu-fix-20261007/Cargo.toml \
  --target-dir target/maintenance-compatibility
```

Windows uses its installed Cargo without the toolchain selector. Outputs go to
new maintenance logs, never historical verification directories.

The 93 result lines match across platforms after normalizing only printed path
separators in the derived comparison. An initial exact-text comparison failed
on 78 Windows path spellings; both raw logs and that diagnostic are preserved in
[comparison evidence](compatibility-comparison.json). No artifact bytes were
normalized for this comparison.

CLI smoke checks cover actual `packtok` help, M0 raw round-trip, M1 training/validation/inspection
and German round-trip, and pack training inspection. The regenerated temporary
M1 fixture hash matches its known artifact:
`cef973a354422c88e6fef54d0b6eee09495980df4e5bd0b7e5adc37896e7d907`.
See [CLI evidence](cli-examples.json). The unchanged v4 bundle, inner manifest and
frozen source hashes pass [local bundle verification](frozen-local-bundle.txt).

Initial checks failed before execution because a non-login WSL PATH lacked
Cargo; a CLI smoke attempt used the package name instead of the executable.
Both failures are preserved in [setup notes](setup-failure.md), corrected and
followed by passing checks. Native HTTPS push stalled before updating GitHub;
only that task-owned Git process was terminated. Publishing uses Windows'
existing credential manager after fetching the verified native branch into a
separate ref. No credential was copied into Linux or public output.

## Actual WSL supervisor verification

The [bounded design](../../development/gpu-colab.md) uses the checked-in Windows
launcher to start an independent hidden wsl.exe client. It waits on the actual
systemd user service and Python supervisor. Production bounds stay 1620 seconds
work plus release reserve, systemd RuntimeMaxSec=1680/TimeoutStopSec=120,
KillMode=mixed and Restart=no. No permanent daemon, linger, scheduled task or
execution-policy change was installed.

Two original long actual-entrypoint tests measured worker execution
**310.090926 seconds** and **310.121580 seconds**; request-to-verified-release was
312.213461 and 311.844124 seconds. The invoking shells had exited. The second
also terminated only verified disposable PAM/logind session 11. Both ended with
service success, inactive/dead state, empty cgroup and no surviving owned
transport descendant. Journals, UTC timestamps, PIDs/session IDs, signals and
stage exits are retained in [long results](long-supervision-results.json) and
the corresponding wsl-normal/wsl-session evidence directories.

A separate interruption test killed only initiating PowerShell helper PID 8664
while the worker was active, at 2026-10-08T17:17:05.2532147Z. Independent client
PID 4724 remained alive and completed: 20.097423 seconds worker execution,
21.510714 seconds request-to-release, empty cgroup and no descendant. An earlier
20-second attempt completed before helper termination; it is preserved but does
not count as interruption coverage.

After the final ownership/receipt guards, another actual-entrypoint run measured
**310.115171 seconds** worker execution and **311.802287 seconds** request-to-release.
It again ended with verified cleanup, inactive/dead service, empty cgroup, no
transport descendant and no surviving independent Windows client. The invoking
shell had exited. [Final result](wsl-final-review/final-result.json), tested-source
hashes, host receipt and stage/journal evidence are preserved in that directory.
The final scoped-unit listing contains no running supervisor, and user linger
remains **no** in [final lifecycle state](wsl-final-lifecycle.txt).

The short actual-entrypoint matrix tests Python work deadline, systemd timeout,
TERM propagation, task failure, unconfirmed release and success. Each invocation
makes exactly one **fake** request and leaves unrelated fake LIV untouched.
Reinvocation cannot allocate a second session. Missing archive, remote exit 127,
missing gate PASS, incorrect accelerator, upload/download errors and stale/
unowned session cases are covered by the recovery suite. An unconfirmed endpoint
retains ownership and urgent evidence; it never claims release.
Final recovery regressions: 22 PASS in
[review rerun](recovery-after-review.txt); all six actual-entrypoint cases pass
after the ownership guard in [review matrix](suite-review/results.json).

The targeted final review found that an alias reassignment between initial
ownership and release could previously reach a stop call. Release now verifies
the recorded endpoint immediately before stopping; unknown or reassigned
ownership fails safely with its marker retained. Two added regressions cover
reassignment and an allocation transport failure before endpoint capture. CLI
inventory and stop are separate calls, so this is not atomic protection against
an adversarial concurrent reassigner; the owned alias must be exclusive to the
job. Stronger atomic endpoint stop is not assumed without supported CLI evidence.

Reproduce short cases:

```sh
python3 experiments/m5-gpu/scripts/test-recovery.py
python3 experiments/m5-gpu/scripts/test-restore.py
bash experiments/m5-gpu/scripts/check-wsl-supervision.sh
```

For a fresh long CPU fixture, under native WSL:

```sh
python3 experiments/m5-gpu/scripts/supervision-fixture.py create \
  /home/dasetwa/projects/PackTok/docs/maintenance/NEW_RUN/long
```

From the synchronized local Windows checkout, use a new receipt in an existing
directory:

```powershell
.\experiments\m5-gpu\scripts\launch-wsl-supervised.ps1 -Mode cpu-mock `
  -Fixture /home/dasetwa/projects/PackTok/docs/maintenance/NEW_RUN/long `
  -MockMode orphan -MockSeconds 310 -WorkSeconds 340 -RuntimeSeconds 360 `
  -Receipt C:\Users\DasEtwa\PackTok\target\maintenance-host\new-run.json
```

The five-minute test uses the validated fake transport, whose PATH cannot reach
the real Colab binary. The disposable PAM helper is preserved in
[disposable-session.sh](disposable-session.sh). Its original scoped invocation
was a transient root `systemd-run --unit=packtok-maintenance-disposable-session
--property=User=dasetwa --property=PAMName=login --property=RuntimeMaxSec=150`
running that helper. It is not a persistent service. The test inspected
`loginctl show-session 11`, its Leader and the exact helper cgroup before using
`loginctl terminate-session 11`. Session numbers/PIDs must be derived afresh;
never replay that termination against an unverified session. The Codex process,
other user sessions and other distributions were not terminated.

Five minutes is stronger evidence than the retained earlier 40-second sleep,
but not proof of every 30-minute GPU failure or abrupt host power loss. Loss of
the independent client/host can still require manual owned-resource release.
No GPU success was inferred from mock success or a CLI exit zero.

## OAuth and storage

Actual consent branding is **PackTok**; the Desktop OAuth client is **PackTok WSL
Backup**. Audience remains **External / Testing**, one test user. Scope is
`drive.file`; Google Cloud now explicitly declares that same non-sensitive scope.
Client/config permission checks confirm private directories 0700 and files 0600,
matching client identity and a refresh token without exposing any credential.

The supported selected route is External / In production, retaining this client,
account and scope, then one fresh authorization and readback. Publishing requires
the user's explicit approval under this task's instructions. No approval has
been received and publication was not attempted. The console additionally
disables Publish app pending Branding completion, without identifying the exact
missing field; no policy URL, domain or personal contact was invented.

Google's Testing configuration expires this authorization/refresh token after
seven days. Production removes that Testing-specific limit for fresh consent;
revocation, six-month inactivity, token-issuance limits and policy/time-limited
consent still apply. Non-sensitive-only apps do not require mandatory app
verification; name/logo display can require brand verification. No broader Drive
scope or service account is needed. Official sources and the complete migration,
revocation, reauthorization and account-isolation procedure are in
[OAuth](../../development/oauth.md). Long-term unattended authentication is
**pending user publishing approval and Branding readiness**, not proven by today's
short test.

The existing rclone workflow was retained. A new harmless 55-byte file passed
real upload/full-download/size/SHA-256/completion-manifest readback. Payload SHA:
`0fa58bed48e2f8e42c99ce6e374bf9cb616d416d3ce66819757b77a3c0ce9a1c`.
Run ID: `m5-preflight-fixture-e607586553f945d08024abed4d789c1d`.
Nine new restore regressions reject corrupt/incomplete/mismatched manifests,
corrupt payloads, traversal, unrelated namespaces, existing destinations and
oversize entries. The manifest-based restore tool then downloaded and verified
the same file into a new protected external destination: **RESTORED_VERIFIED**.
See [readback](drive-maintenance-result.json), [restore test](restore-live.txt)
and [restore guide](../../development/storage-and-recovery.md).
This tests a fresh destination with the existing account/client, not an OS
reinstallation, another account, multi-month tokens or a real GPU checkpoint.
M5 weights remain recovery weights, not optimizer/RNG/data-cursor exact resume.

Read-only Colab inventory showed no active assignments and usage rate zero;
[session](colab-sessions-readonly.txt) and [usage](colab-usage-readonly.txt) logs
are retained. **Zero new real GPU allocations** were made in maintenance.

## Remaining actions and release boundary

The unresolved CUDA loader/bundle gate remains an M5 infrastructure blocker;
the preserved GPU failure occurred before the Rust CUDA test/training gate.
Fixing it and allocating the next L4 belong to a separately authorized task.
Full multi-seed work additionally requires an approved measured compute budget.
OAuth publication/Branding and fresh consent remain explicit user actions.

Delivery keeps PR #5 draft and its base unchanged. Before each commit, inspect
staged paths/diff, scan credentials and check whitespace. After final push, verify
remote HEAD, both checkout HEADs/tracking refs and clean worktrees; the final
delivery record identifies the exact published commits. See
[targeted self-review](SELF_REVIEW.md) and the final audit JSON beside this report.
