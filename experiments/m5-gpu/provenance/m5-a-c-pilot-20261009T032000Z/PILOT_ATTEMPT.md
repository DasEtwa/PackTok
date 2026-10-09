# M5 A/C 2,000-update pilot attempt — incomplete

Date: 2026-10-09 UTC. Status: `M5_A_C_PILOT_PARTIAL`.

## Frozen identity and preparation

- Branch/source: `m5-gpu-transformer`, commit `5e77c36ff2dd78b076677c45866971beef734f9d`.
- Frozen config: `configs/pilot-2k-v1.json`, SHA-256 `6536d5194bc7d422bdef6e871b58317497794ed1553b197a927e0434aa65c3d8`.
- Package: `bundle-pilot-2k-v1-l4-20261009T032000Z`, SHA-256 `656ae7cc5d6aeabebbadddae6bfcab9f65307ce987c5588efe7fabe7d5f767ac`.
- CUDA executable SHA-256: `c5b9794f50cadca7565c69dc4d7c5fa997b367ad40a43bc9746bc5393d9aac51`. Its bytes equal the verified v5 executable. M5 source, Cargo manifests/lock, scripts and configs are unchanged between v5's `b686eac6645c793a715bbad4b6d9bc3033f3145d` and the current `5e77c36` commit; no CUDA rebuild was needed.
- Corpus-v2 manifest SHA-256: `033bf7a3126ad5e39d3d11e5abd257f9cfaaac2d8f001003830215191b9d494f`.
- CPU `plan-extended` accepted the frozen A/C/T/seed/config and all prepared train/validation/test sequence hashes. It predicted 4,096,000 target positions each and raw-byte exposure of A `7,575,564`, C `6,946,291` for seed `20261008`.
- A previous successful L4 gate, using the same architecture, initialization source and seed, recorded equal pre-update hashes for A and C: `155de938a7f469bf3bce65e548e938792bb6ebe0a53ffcda8135d0c65881a0a1`, with 14,681,984 parameters each. This establishes the initialization identity from the existing CUDA gate; this attempt did not reach the Transformer pilot runner.
- The pilot archive included all frozen splits and artifacts. The runner's initial input hash command covered the test sequence file hash only; no test scoring occurred.

## One authorized L4 session

- User authorized exactly one L4 allocation, 1,800 seconds maximum, with work stopped by 1,500 seconds and 300 seconds reserved for cleanup.
- Colab CLI `0.7.4`; no active sessions before allocation; displayed balance `167.38 CU`, rate `0.00 CU/h`, zero assignments.
- One `NVIDIA L4` session was created under the owned alias `packtok-m5`. The 14 immutable package parts uploaded and the bridge began its remote execution.
- Colab's remote bridge returned exit 1 and raised `RuntimeError: Rust CUDA preflight failed with exit 1`. The phrase "preflight" is a stale bridge error label; it does not mean a pilot run began.
- Downloaded result archive was 160 bytes; its only member was `results/exit-code.txt` with value `1`. No Rust console, training metrics, checkpoint, validation/test score or completion manifest was captured. The local lifecycle evidence shows the runner requested download and the Colab CLI reported a successful transfer of this tiny archive.
- Allocation request to verified cleanup: `386` seconds. After stop, `colab sessions` reported no active sessions. The only owned alias was absent. Displayed CU balance went from `167.38` to `167.24`; displayed decrease `0.14 CU`, active assignments returned to zero and rate to `0.00 CU/h`. These are displayed balances/rates, not a higher-precision billing receipt.

## Failure analysis and limits

- **Observed:** remote exit 1; the downloaded archive contains only exit code 1. No A or C update is evidenced, so completed updates are `0` for both for reporting purposes, and all model metrics are unavailable. The exact remote stderr was not preserved.
- **Likely cause (inference, not direct remote stderr):** the submitted `colab exec` command did not pass the required `PACKTOK_M5_GPU_APPROVAL` using the CLI's `--env` option. The packaged remote shell uses `set -u` and expands that variable before its first Rust command, so an unset value would terminate it before training. The bridge's error wording and the one-file archive do not prove this diagnosis; missing bootstrap-console evidence prevents definitive attribution.
- The single user-authorized allocation is consumed. No retry or second allocation was made. No claim about model quality, throughput, VRAM, final loss or five-seed readiness follows from this attempt.
- The pilot's TEST set was not scored. Because this pilot process included the TEST sequence among its prepared inputs, any future use of these same held-out examples must disclose the exposure and must not describe them as untouched for a revised experiment.

## Preserved evidence

This directory preserves the readiness hashes, frozen plan, package/content checksums, exact local run command, allocation/status/usage records, upload list, Colab execution error, downloaded 160-byte remote result archive, stop/release proof and this report. Two pre-allocation package preparation failures are separately marked in their package directories; neither requested a GPU.

## Drive recovery

- Evidence archive: `pilot-partial-evidence.tar.gz`, 8,660 bytes, SHA-256 `35143d29b46b7bb0b76f656f8ee5effb83fa8c039a411db37d2237f89db08945`.
- Full readback verification: `COMPLETE_VERIFIED`; destination `packtok-drive-own:PackTok/M5/preflight/m5-diagnostic-archive-f99432b0f0e64c51a1785595293a74f`.
- The first local transfer invocation was preserved as `INCOMPLETE` because `/home/dasetwa/.local/bin` was missing from the supervisor PATH. A second invocation with that installed rclone directory in PATH completed upload, full artifact download/SHA-256 comparison and completion-manifest readback. No remote artifact from the incomplete first invocation was created.
