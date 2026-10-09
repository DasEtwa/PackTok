# M5 A/C 2,000-update pilot attempt — blocked before training

Date: 2026-10-09 UTC. Status: `M5_A_C_PILOT_PARTIAL`. No A/C Transformer update was verified.

## Frozen source and package

- Branch: `m5-gpu-transformer`; package source commit: `fb5d0990922b2f50cd8cd2c64e9504fad43bc9fd`.
- Frozen config `experiments/m5-gpu/configs/pilot-2k-v1.json`, SHA-256 `6536d5194bc7d422bdef6e871b58317497794ed1553b197a927e0434aa65c3d8`.
- Unique package `bundle-pilot-2k-v1-l4-20261009T115439Z`; tar SHA-256 `ea7a0f64912d2557a29848b11e24d4bf0d97c700738dda7f24450dc845fa7e2a`. The package contains the current committed transport source and is recorded in `package-evidence/`.
- Fresh CUDA build: `cargo build --locked --release --features cuda --target-dir target-cuda`, Rust 1.85.0, CUDA 12.4.1, compute capability 8.9. Packaged executable SHA-256 `df9b9cc1ff76e89a6c3814b97cf5aa81d7a062172bfb10737826481fad3eb459`. Loader resolution and `plan-extended` passed locally for the frozen A/C/T/seed/config.
- Corpus-v2 manifest SHA-256 `033bf7a3126ad5e39d3d11e5abd257f9cfaaac2d8f001003830215191b9d494f`. The frozen plan predicts 4,096,000 target positions/model; raw-byte projections are A 7,575,564 and C 6,946,291. Actual exposure was zero because training never began.
- The model config and seed are identical for A/C. Existing CUDA diagnostic evidence records the deterministic pre-update hash `155de938a7f469bf3bce65e548e938792bb6ebe0a53ffcda8135d0c65881a0a1`; no pilot checkpoint/model was created in this attempt.

## Launch regression and confirmed failure

The earlier 20261009T032000Z attempt invoked `colab exec` without `--env PACKTOK_M5_GPU_APPROVAL=...`. Its archived stderr is absent, so its exact remote failure remains unconfirmed; the missing approval handoff was a real defect, not proven as that attempt's sole cause. A committed `colab-exec-pilot.sh` now forwards the approval with Colab CLI `--env`.

This attempt used that corrected invocation. The remote archive proves a separate bootstrap defect: `remote-bridge.py` created `packtok-m5/results/` before starting the shell; `remote.sh` then ran `mkdir results` under `set -e` and exited immediately with `mkdir: cannot create directory 'results': File exists`. The archived remote exit code is 1, and `bootstrap-console.txt` preserves this stderr. No Rust command ran and A/C completed zero verified updates. The bridge was corrected to create the result directory after the child returns (or on bootstrap error), and the CPU transport tests now cover both successful handoff and this `mkdir results` contract. Four tests pass. This corrective change was committed after the consumed package; it was not run on GPU because the user authorized only one new L4 allocation.

## Allocation and release

- One L4 session, owned alias `packtok-m5`; Colab CLI 0.7.4. Displayed CU: 167.24 before to 167.15 after; displayed decrease 0.09 CU. Active rate returned to 0.00 CU/h and assignments to zero.
- Session request to verified cleanup: 386 seconds by request/release timestamps; supervisor lifecycle interval: 397 seconds.
- `sessions-after.txt` reports `No active sessions found on server.` The owned alias was stopped, then absence was verified. No unrelated sessions were stopped.
- Training throughput, step time, validation/test BPB, VRAM, checkpoints and checkpoint hashes are unavailable. TEST was not scored or used.

## Result and limitations

The pair did not run, so there is no C−A validation BPB, no quality direction, and no evidence supporting a five-seed study. This is an orchestration failure, not tokenizer/model evidence. A second allocation would exceed the one-allocation authorization and was not requested or attempted.

Raw logs, archived remote output, lifecycle/CU records, readiness, source/config/package hashes and the exact approval-forwarding runner are preserved in this directory. `results.tar.gz` is the verbatim Colab readback from this attempt. The evidence archive was uploaded through the existing rclone Drive remote and fully read back with matching SHA-256; receipt: `DRIVE_READBACK.json`.
