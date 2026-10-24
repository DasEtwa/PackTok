# Astra bug-fix validation — 2026-10-24

Reviewed starting point: `8c5b0fb00e6eca4c5bdc6e0ace7a0b8437d25b28`, on
`feat/semantic-packs-v1`. The independent report and original reproductions are
in `../astra-review.md` and `../astra-repros/`. A separate runner reviewer added
the canonical-history regressions described in `../runner-bugfix-review.md`.

## Changes and regressions

- A1: timeout/backup cancellation targets the owned process group even after its
  leader exits. SIGKILL escalation and final cleanup cover children that retain
  the output pipe and children that close it. Unrelated groups are not signaled.
- A2: absent remote checkpoint metadata cannot overwrite a complete newer local
  run. Local payloads are validated before preserving that run. Rust additionally
  rejects incomplete, duplicate, unordered or malformed canonical training rows
  through the restored step; a rejected log and its archive remain unchanged.
- A3: CUDA admission checks required toolkit headers and all five toolkit linker
  libraries. The official fallback includes cuRAND development files. Existing
  CUDA versions unsupported by locked cuDARC are rejected. These are offline
  dependency/installation-plan checks, not a successful actual CUDA build.
- A4: the Rust CLI admits exact L4 names and rejects L40/L40S.
- A5: stale or same-step divergent backup writers are rejected before metadata
  copy. A destination pointer changing during shard copy prevents publication
  and pruning. Invalid/inconsistent generation steps are rejected. Concurrent
  Drive/FUSE writers remain unsupported: no distributed atomic transaction or
  global writer lock is claimed.
- A complete final JSONL object without LF gains its missing line separator
  before the resume marker. The archived original remains byte-for-byte intact.
- A failed first Git fetch can resume a proven empty no-checkout clone with the
  expected origin; dirty worktrees, untracked notes and a wrong origin remain
  protected. No reset/clean operation was introduced.

`python-before-fixes.log`, `clone-before-fix.log` and
`stale-mirror-before-fix.log` retain the new regression failures before their
production fixes. Subsequent passing results do not overwrite them.

## Final local checks

| Command/check | Result/log |
|---|---|
| Rust 1.90 locked debug tests | 29 unit + 1 CLI; `rust-debug-tests.log` |
| Rust 1.90 locked release tests | 29 unit + 1 CLI; `rust-release-tests.log` |
| Rustfmt | pass; `rust-fmt.log` |
| Clippy locked/all-targets, warnings denied | pass, CPU features; `rust-clippy.log` |
| Locked release build | pass; `rust-release-build.log` |
| Python standard-library tests | 29 passed; `python-tests.log` |
| Notebook JSON/AST/compilation | 5 cells / 2 code cells; strict pin check |
| nbformat 5.10.4 schema | pass; `notebook-schema.log` |
| Actual tiny Rust orchestration + local backup fixture | pass; `orchestration-cpu.log` |

Rust commands use `--manifest-path experiments/semantic-scale/Cargo.toml` and
the committed lockfile. Python tests run with
`python3 -m unittest discover -s experiments/semantic-scale/tools -p 'test_*.py'`.
The standard-library notebook checker is `tools/check_notebook.py`. A missing
temporary schema-validator installation first caused an ImportError; that failure
is retained in `notebook-schema-missing-validator.log`, followed by the pinned
temporary installation and successful validation. The notebook itself does not
require nbformat to run.

The orchestration probe routes only the explicit tiny FP32 fixture through
`--cpu`, mocks installation/GPU discovery, writes real Rust Safetensors, backs
them up to a local directory, deletes the complete simulated local runtime and
restores optimizer step 4 with canonical metric steps `[1,2,3,4]`. It tests neither
Google Drive/FUSE nor CUDA. Original root tokenizer code was not changed by this
bug-fix cycle; its previously recorded 161-test debug/release compatibility
results remain in `../final/`.

No GPU was allocated. Actual L4 CUDA build/execution, BF16 behavior, 1B peak VRAM,
throughput and model-quality comparisons still require the notebook experiment.
