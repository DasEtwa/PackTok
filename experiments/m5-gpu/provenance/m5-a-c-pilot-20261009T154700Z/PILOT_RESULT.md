# M5 A/C 2,000-update Transformer pilot

Status: **complete**, exploratory single paired seed. The existing Flat BPE A run was recovered and reevaluated; it was not retrained. PackTok C was trained in the one newly authorized L4 allocation.

## Frozen run and provenance

- Repository/branch: `DasEtwa/PackTok`, `m5-gpu-transformer`.
- Exact launched source commit: `dd0380396e63d926883f378daf9e6d029538f0f6`.
- Fresh CUDA executable SHA-256: `c388a182844f5ade1971b7b0dc3ad4e3fb2b2bcd80d6c20ba6ab6f93c180af98`.
- Bundle: `pilot-2k-v1-l4-20261009T154700Z`; archive SHA-256 `f241410ec33f477a4568079b00c0e860c79afd90f43be5d049899c45a0930f84`.
- Frozen config SHA-256 `6536d5194bc7d422bdef6e871b58317497794ed1553b197a927e0434aa65c3d8`.
- Corpus-v2 manifest SHA-256 `033bf7a3126ad5e39d3d11e5abd257f9cfaaac2d8f001003830215191b9d494f`.
- A/C tokenizer artifacts and all six prepared sequence hashes are recorded in the archived `input-sha256.txt`.
- Architecture and seed: 14,681,984 FP32 parameters; 8 layers, hidden 384, 6 heads, FFN 1536, context 256; batch 8; AdamW 0.0003, weight decay 0.01, constant LR, T regime; seed `20261008`.
- The frozen CUDA readiness record verified A and C share initialization SHA-256 `155de938a7f469bf3bce65e548e938792bb6ebe0a53ffcda8135d0c65881a0a1`. The model/config/training source is unchanged from that verification; the source delta from A's training commit to this evaluation/training package changes only domain aggregation and pilot orchestration. C's run record is a fresh step-0 start with the paired seed.
- Order: A checkpoint resume and final evaluation only; C fresh training for 2,000 updates. A checkpoint SHA-256 `da3b08bfda2bc991992a4487f4659c7b406b2af5f2abe62789bef8d80e50655a`; C checkpoint SHA-256 `e7b45ad2bcaf52434aac6d35db06edab0ba92e86d27ca8c46930a2c1c43fc484`.

## Results

Both variants completed 2,000 updates and 4,096,000 sampled target positions. Mean step time is the runner's synchronized `step_seconds` averaged across all updates, including the first step; post-first-step means are shown for context because A had a cold first-step cost.

| Metric | A — Flat BPE | C — PackTok |
|---|---:|---:|
| Verified updates | 2,000 | 2,000 |
| Final update train loss/token | 3.371998787 | 3.425781488 |
| Final validation BPB | 2.572964850 | 2.597469229 |
| Targets sampled | 4,096,000 | 4,096,000 |
| Actual training raw-byte exposure | 7,575,564 | 6,946,291 |
| Mean synchronized step (all updates) | 106.125 ms | 98.461 ms |
| Mean synchronized step (steps 2–2000) | 98.817 ms | 97.953 ms |
| Recorded tokens/second | 19,298.06 | 20,800.08 |
| Represented raw bytes/second | 35,691.82 | 35,274.27 |
| Training GPU time | 212.249 s | 196.922 s |
| Peak sampled VRAM | 2,442 MiB | 2,410 MiB |
| Final resume checkpoint SHA-256 | `da3b08bf…e50655a` | `e7b45ad2…c43fc484` |

Validation BPB trajectory at steps 1 / 500 / 1,000 / 1,500 / 2,000:

- A: 4.825597572 / 3.163909463 / 2.804701301 / 2.641904390 / 2.572964850.
- C: 5.096670344 / 3.132276250 / 2.862802481 / 2.708066335 / 2.597469229.

**Primary difference:** C − A = **+0.024504379540 BPB**, about **+0.9524%** relative to A's BPB. This measured pilot favors A slightly. One paired seed does not establish statistical significance or a general tokenizer advantage. The target-position budget is equal; C represented 629,273 fewer raw training bytes (8.3066% below A). No claim is made of equal raw-byte exposure or FLOP matching.

| Validation domain | A BPB | C BPB | C − A BPB |
|---|---:|---:|---:|
| English literature | 2.813912684 | 2.850865766 | +0.036953082 |
| German prose | 3.029457902 | 3.089193146 | +0.059735243 |
| Rust source | 2.687568176 | 2.689860304 | +0.002292128 |
| Synthetic JSON | 1.290185308 | 1.256581069 | −0.033604240 |
| Synthetic Unicode | 1.092557356 | 1.078071212 | −0.014486144 |
| Structured / mixed-domain | unavailable | unavailable | unavailable |

The runner also performed its frozen final-only TEST scoring (A 2.579990970 BPB; C 2.605001505 BPB). TEST was not used for tuning or debugging. These scores are exploratory pilot outputs and must not be represented as an untouched independent test for a revised study.

## Evaluation fix, allocation and preservation

- Recovered A's final domain evaluation and C use the same corrected aggregation. The FP64 NLL reconciliation uses a narrow relative tolerance; target-byte and token counters remain exact. All six domain aggregation regression tests passed before allocation, including acceptance of the reproduced 8.38e-9 summation-order difference and rejection of real counter/NLL inconsistencies.
- Remote result archive exit code: `0`; status: `M5_A_C_PILOT_COMPLETE`. It contains full A/C training metrics and console logs, five validation checkpoints per variant, domain validation and final-only test records, final checkpoints, hardware samples, input hashes, execution order and completion records.
- The new allocation was requested at `2026-10-09T16:13:09Z`; the owned Colab session was verified absent at `2026-10-09T16:33:53Z`: **1,244 s (20 min 44 s)** including startup, upload, evaluation, download and cleanup. CU changed **166.92 → 166.50** (−0.42); Colab reported zero active assignments. A had already used a separate earlier allocation (813 s; 167.15 → 166.92 CU) to complete its original training, so both allocation lifetimes sum to 2,057 s (34 min 17 s) and the observed total CU change across the paired work is 167.15 → 166.50 (−0.65).
- The remote run completed normally, but the local supervisor's release guard lacked its expected endpoint file and exited 1 without stopping. The current session listing contained only the alias created by this run (preallocation listing was empty); that owned alias was stopped explicitly. A subsequent Colab sessions/status query reported no active sessions, and usage showed 0.00/hr and zero active assignments. The original `RELEASE NOT VERIFIED` guard record is preserved alongside the later manual-stop verification; the lifecycle issue does not alter the successful remote exit or model results.
- Results archive SHA-256 `d7da7a902f2664f562351ec04d19e9aaed5b42478fb19d0f8d57247b8d1c25b4`, size 322,912,151 bytes. It was uploaded to `packtok-drive-own:PackTok/M5/pilot/m5-a-c-pilot-evidence-20261009T154700Z/results.tar.gz` and downloaded in full; local and Drive readback hashes are identical. `DRIVE_BACKUP.json` and its uploaded/read-back completion record carry the receipt.

Detailed files: this report, `PILOT_RESULT.json`, `results-archive.sha256`, readiness evidence, extracted raw results and rehearsal diagnostics are under this attempt directory. The 308 MiB tar and checkpoints are not committed to Git.
