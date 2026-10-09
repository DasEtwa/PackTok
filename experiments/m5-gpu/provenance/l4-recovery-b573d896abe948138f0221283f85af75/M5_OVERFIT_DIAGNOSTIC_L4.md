# M5 L4 overfit diagnostic — 2026-10-09

## Identity and protocol

- Source commit: `b686eac6645c793a715bbad4b6d9bc3033f3145d` on `m5-gpu-transformer`.
- Bundle: `bundle-v5`; archive SHA-256 `5b067860282a192ea9abd81e8e8ee8d37ae5b4b5f0ce9cb01647f3fc2bb56509`.
- Executable SHA-256: `c5b9794f50cadca7565c69dc4d7c5fa997b367ad40a43bc9746bc5393d9aac51`.
- One authorized allocation, owner alias `packtok-m5`, endpoint `gpu-l4-s-kkb-ass1a0-1tu0zxhhqeln5`.
- Ubuntu 24.04.4 LTS, NVIDIA L4, driver 580.82.07, 23,034 MiB device memory. The binary was built against pinned CUDA 12.4.131 / Candle 0.9.1, FP32.
- Bundled loader resolved `libcuda.so.1`, `libcurand.so.10`, `libcublas.so.12`, and `libcublasLt.so.12`. CUDA initialization and the CPU/CUDA logits reference passed; maximum absolute logits delta was `4.3213367462158203e-7`.
- Both runs used the same eight input IDs `[1,2,3,4,1,2,3,4]`, targets `[2,3,4,1,2,3,4,1]`, AdamW LR `0.005`, weight decay 0, no dropout, warmup, or gradient clipping, and 500 updates. A starts fresh at seed `20261008`; B reconstructs the frozen A state after 18 updates (sampler seed `20261008 XOR 0xa341316c9e3779b9`, AdamW LR `0.0003`, weight decay `0.01`) and then creates a fresh overfit optimizer.
- Losses below are cross entropy at the specified synchronized checkpoints. The CPU series is the preserved `overfit-diagnostic-cpu-20261008/diagnostic.jsonl`; the L4 series is `extracted-results/results/gate/preflight.jsonl`.

## CPU and L4 loss curves

| State | Step | CPU loss | L4 loss |
|---|---:|---:|---:|
| Fresh | 0 | 6.246869 | 6.2468691 |
| Fresh | 1 | 6.596113 | 6.596126 |
| Fresh | 10 | 1.3886182 | 1.3886116 |
| Fresh | 25 | 1.3275406 | 1.3275721 |
| Fresh | 50 | 0.2570268 | 0.6630679 |
| Fresh | 100 | 0.2702269 | 0.1996985 |
| Fresh | 200 | 0.0006085 | 0.0001562 |
| Fresh | 350 | 0.0002550 | 0.00008460 |
| Fresh | 500 | 0.0001519 | 0.00005337 |
| Post-18 | 0 | 7.586294 | 7.5862923 |
| Post-18 | 1 | 6.567982 | 6.5678635 |
| Post-18 | 10 | 1.3877027 | 1.3877027 |
| Post-18 | 25 | 1.3849963 | 1.3849589 |
| Post-18 | 50 | 1.3511348 | 1.3424411 |
| Post-18 | 100 | 0.9371630 | 2.5421631 |
| Post-18 | 200 | 0.3428211 | 0.1738532 |
| Post-18 | 350 | 0.2394418 | 0.1733936 |
| Post-18 | 500 | 0.2385193 | 0.1746082 |

## Outcome and discrepancy

- Fresh L4: final loss `0.0000533732`; first recorded checkpoint below `0.25` is step 100. Both original criteria pass: final `<0.25` and `<10%` of initial loss. The CPU final was `0.000151885`.
- Post-18 L4: final loss `0.1746082`; first recorded checkpoint below `0.25` is step 200. Both original criteria pass, including `<10%` of initial loss (`0.758629` limit). The CPU first recorded checkpoint below `0.25` is step 350; CPU final was `0.2385193`.
- All checked gradients and parameters remained finite. No divergence or stagnation was reported. The fresh initial parameter hash matches the preserved CPU diagnostic (`155de938a7f469bf3bce65e548e938792bb6ebe0a53ffcda8135d0c65881a0a1`). Initial losses also match within about `2e-6` for post-18 and about `1e-7` for fresh. The first update losses are close; the selected CPU reference did not preserve per-tensor gradients or AdamW deltas, so exact gradient-level CPU/CUDA parity cannot be asserted.
- The historical 100-update GPU failure is reproduced: post-18 L4 loss at step 100 is `2.542163`, matching the earlier `~2.5422` failure. Extending the same unchanged diagnostic to 500 steps lets both states pass; post-18 crosses the threshold by step 200. Thus the historical gate failed because its 100-step stopping point preceded this L4 trajectory's crossing, not because the 500-step memorization criterion is unreachable.
- CPU and CUDA trajectories are not numerically identical: the post-18 curves separate substantially by step 100 and then cross in relative performance. Initial logits, seeds, data IDs, and configuration are consistent; no nonfinite values or model/runtime errors indicate an implementation defect. The most plausible explanation is FP32/backend perturbations amplified by the high-LR AdamW trajectory, but the exact earliest tensor operation was not localized because historical CPU gradient/update tensors were not retained. This numeric trajectory difference remains an explicit limitation; the task's final overfit gate passes on both backends.

## Timing, memory, checkpoint, and cleanup

- Fresh overfit: 26.6666 s for 500 updates (`53.33 ms/update`); post-18: 26.6335 s (`53.27 ms/update`). The existing synchronized primary-training timing was 91.38 ms/update for A and 92.88 ms/update for C (18-step smoke only).
- Peak sampled device memory over the CUDA preflight: 2,346 MiB. The L4 exposed 23,034 MiB total.
- Safetensors checkpoint after the 18 A updates: 58,738,000 bytes, SHA-256 `d059357b98000dba47071eb3b4b46afd1dbf29e22cd11195c82ffeafea3f5510`. Loading into a newly initialized model reproduced inference logits within the implemented tolerance (1e-5 + 0.001 times absolute reference); checkpoint gate PASS. This is a weights-only roundtrip, not optimizer-state resume validation.
- CUDA gate: PASS. Remote CUDA execution took 95.06 s. Allocation request to confirmed release took 483.83 s. The owned session was stopped, post-stop inventory showed no sessions, cleanup was recorded as verified, the bounded WSL client exited, and no orphaned supervisor remained.
- Displayed CU balance: `167.46` before and `167.38` after (displayed decrease `0.08 CU`). Active assignments after cleanup: 0.
- Full raw result archive was downloaded and independently SHA-256 backed up/read back through the existing Drive workflow: `COMPLETE_VERIFIED`, 54,070,984 bytes, SHA-256 `90019fee9b6842b631166afa41c632643bc64359b9801941e215490024033a27`. The archive includes the safetensors file and sampled GPU telemetry.

## Scope and next step

This verifies the M5 overfit smoke gate on one real L4. It does not run or authorize the paired 2,000-update A/C pilot, make a scientific comparison claim, or validate exact optimizer-resume recovery. The overfit gate is ready for the separately budgeted pilot process; the residual CPU/CUDA trajectory sensitivity should remain reported rather than treated as exact numerical parity.
