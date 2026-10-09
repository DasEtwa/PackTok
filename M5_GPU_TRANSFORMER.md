# M5 — GPU Transformer scale probe

Status: CPU preparation complete; L4 CUDA correctness/overfit diagnostics passed.
The first Transformer pilot attempt ran A for 2,000 updates and validation scoring,
then failed in final floating-point domain-score reconciliation before C started.
The session was released; this is partial training evidence, not an A/C comparison.
M5 begins at the exact CPU freeze `f3b6c3fb41f46c93e701b585d859b0b9071d65a5`.
M4's delivery-only commit `3b304d2dfdbae8111da7b6cf714569fff353d030` was pushed
to its existing PR before branching. Branch `m5-gpu-transformer` targets
`m4-factorization-ablation`. M0–M4 contracts, results, failures and encodings remain
frozen.

Current status (2026-10-09): the L4 A run reached all 2,000 updates and recorded
validation BPB from 4.82560 at step 1 to 2.57296 at step 2,000. It exposed TEST
for final-only scoring but did not preserve the TEST score; validation-domain
aggregation then exited with code 1 before C. The strict FP64 NLL equality check
was reproduced as too tight and corrected with a narrow relative tolerance while
retaining exact byte/token equality. A targeted regression and all five domain
tests pass. The correction is post-bundle and needs a fresh package before any
future GPU run. The exact result and limits are in
[`PILOT_ATTEMPT.md`](experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T134013Z/PILOT_ATTEMPT.md).

## Question and controls

Does M4's small C−A byte-normalized gain survive a causal Transformer? A uses
M1 flat IDs; C uses M2 lexical-v1 IDs, concatenated in ascending numeric pack ID,
then local ID through the existing `packtok-model::IdMapping`. This is a
bijection, not a tokenizer change. Both use exactly the same 512-row global
embedding and flat output architecture; there are no pack features or heads.
The M4 corpus is a separate compatibility fixture, never merged with M5 data.

## Architecture and isolation

`experiments/m5-gpu/` is a separate Cargo workspace. Root crates stay CUDA-free,
dependency-free and compatible with Rust 1.85. The experiment uses the existing
core/format/tokenizer/train/model libraries via path dependencies. It owns the
Transformer and experiment orchestration; no root responsibility moves.
Pinned Candle core/nn 0.9.1 provides CPU/CUDA tensors and autograd. CUDA is an
explicit feature, with no automatic CPU or accelerator fallback. The complete
transitive graph is frozen by the experiment's Cargo.lock. Development uses the
already installed dedicated WSL toolchain; its version is in the environment log.

The remote CUDA process sets `NVIDIA_TF32_OVERRIDE=0` before initialization to
disable automatic TF32 substitution ([NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/archive/12.5.0/cublas/index.html)).
BF16 is not enabled.

Primary FP32 model: 8 decoder blocks, hidden 384, 6 attention heads (64 channels
per head), FFN 1536, context 256 and vocabulary 512. Learned absolute positional
embeddings, untied token/output weights, biased Q/K/V/output projections and
biased two-layer GELU FFN. Two pre-norm RMSNorms per block and a final RMSNorm,
scale-only, epsilon 1e-5. Attention scales QK by 1/sqrt(head dimension), masks
strictly future keys with negative infinity, then softmax. No dropout, RoPE,
KV cache, flash attention, pack conditioning or weight tying. RMSNorm uses
ordinary differentiable tensor operations, avoiding inference-only fused ops.

Parameter formula (biases included, optimizer excluded), for V,C,d,L,f:
`V*d + C*d + L*(4*(d*d+d) + 2*d*f+f+d + 2*d) + d + V*d+V`.
The declared primary configuration has **14,681,984 trainable parameters**.
A/C must match both this formula and the actual named tensor counts.
All weights use deterministic host-side xorshift64 (13/7/17), uniform f64 draws
[-0.02,0.02), rounded to FP32; biases zero and RMSNorm scales one, traversing sorted tensor names.
The same seed therefore creates the same A/C weights before any training.
CUDA floating point/autograd is not promised bit-identical to CPU execution.

## Corpus and frozen preparation

A separate 32–64 MiB target corpus uses original public-domain English/German
works from Project Gutenberg, Rust 1.85.0 library sources/config at pinned
`4d91de4e48198da2e33413efdcd9cd2cc0c46688` (MIT OR Apache-2.0), plus explicitly
synthetic structured/Unicode coverage under PackTok's Apache-2.0 license.
Acquisition runs only on local Ubuntu-24.04. Individual PG downloads are capped
at 10 MiB, the pinned Rust archive at 64 MiB. Full downloaded bytes and notices
are retained locally outside Git; their hashes, sizes, extraction ranges,
contributions and immutable split manifest are tracked. PG's catalog declares
public domain in the USA; original authors died long ago and no modern
translation is used. These editions are not relicensed under PackTok's license.
Sources: PG 100 (Shakespeare), 145/7469 (Eliot), 766/883/1023 (Dickens),
2403/2404/2335–2342 (Goethe), 5323 (Fontane), 24782 (Bechstein).
Exact URL template: `https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt`.

Primary corpus-v2 contains 37517499 retained bytes (35.779475 MiB). Raw split
sizes/hashes/composition and individual source provenance are in
[CPU_PREPARATION.md](experiments/m5-gpu/provenance/CPU_PREPARATION.md) and
corpus-v2-manifest.json. Sources follow the fixed PG list above, then sorted Rust
paths, then synthetic JSON/Unicode. Blocks target 8192 bytes, extending to the
next newline, without inserting separators or normalizing bytes. One global
block counter assigns residues 18/19 modulo 20 to VALIDATION/TEST and all others
to TRAIN. Corpus-v1's source-local counter was rejected before model outcomes;
its raw files, manifest, logs and completed artifacts remain archived.

TRAIN is retained first. Proposed held-out blocks are dropped whole if they
share a 128-byte contiguous passage with earlier accepted TRAIN/VALIDATION,
including concatenation boundaries, or have Jaccard similarity at least 0.8
between 32-byte shingle sets sampled every 16 bytes. Rolling hashes use base 257
modulo 2^64; collisions conservatively reject blocks. VALIDATION also rejects
matches to earlier VALIDATION blocks; TEST is compared to TRAIN and VALIDATION.
A final all-offset 128-byte guard checks complete concatenated splits. Every
block range/hash/rejection reason is preserved. This screens lexical copies,
not arbitrary paraphrases; strided near-shingles can miss shifted/edited copies.
Short shared language phrases remain expected. Held-out code has a smaller share
than TRAIN after these conservative guards. Genuine config files have no retained
held-out bytes; held-out structured coverage is explicitly synthetic JSON.
No held-out data may be modified after evaluation. Tokenizers train twice on
TRAIN only, default 512 logical slots/minimum frequency 2; full serialized
byte equality is required. The same original raw splits feed A and C. No
normalization, allocation tuning, extra packs or vocabulary tuning is permitted.

## Training and metrics

Primary proposed fixed schedule, subject to explicit cost approval: paired seeds
20261008, 20261009, 20261010; batch 8; context 256; 2000 AdamW updates.
AdamW lr 0.0003, beta1 0.9, beta2 0.999, epsilon 1e-8, weight decay 0.01.
No warmup, early stopping, gradient clipping, gradient accumulation, dropout,
precision switching or held-out tuning. AdamW decays every trainable variable,
including biases and RMSNorm scales. T starts use xorshift64 with
seed XOR 0xa341316c9e3779b9 and modulo (TRAIN token count minus context). The same
random draws need not correspond to the same raw-byte windows across tokenizers.
Validation curves occur at step 1 and every 500 steps, with final validation/test;
validation is reporting only. If the final step was already scored/checkpointed,
those exact values/image are reused rather than paying for duplicate evaluation
or writes. Final jobs reject seeds outside the frozen list before CUDA or I/O.
Regime T matches token positions/steps and records differing raw-byte exposure.
Regime B, if affordable and approved, matches actual raw training bytes and
records unequal token work and GPU time; it is not compute-matched. B processes
consecutive nonoverlapping target windows from the immutable TRAIN prefix until
at least 8388608 represented target bytes, stopping after a full batch. This
prefix is English drama/prose, a limitation of this secondary regime, not a
balanced sample of all modalities. Its first unscored token and actual byte
overshoot are recorded. Exact B steps/work are in schedule-plan.json. Final
schedule/budget must be frozen before held-out model scoring. A preflight or
incomplete seed set never counts as the final three-paired-seed result.

Loss/token = summed target NLL / target count; NLL/byte = summed target NLL /
exact original bytes represented by those targets; bits/byte = NLL/byte / ln(2).
First tokens without context and all unscored/padded targets are excluded from
both numerator and denominator. Evaluation covers every target exactly once in
nonoverlapping 256-position windows, resetting learned positions/context for each
window, batching up to eight equally sized windows. The final short window has
its exact length and no padding. This is a declared finite-context scoring
protocol, not full-history scoring. Per-token perplexity is not a cross-tokenizer
quality comparison. Paired C−A, arithmetic mean and population SD must preserve
every declared seed and unfavorable result. M5 does not reuse the RNN MAC
formula for Transformer FLOPs.

## GPU lifecycle and approval

All ordinary work runs on local WSL Ubuntu-24.04, user dasetwa, native checkout
`/home/dasetwa/projects/PackTok`. The separate Ubuntu/MOOS environment is unused.
Existing Rust/Cargo/Python/uv/Git/Colab installations are reused. Official CLI:
https://github.com/googlecolab/google-colab-cli . Python is used only by that
external CLI/kernel transport, never for PackTok model/data/training logic.

Only NVIDIA L4 is authorized. Before allocation: version/sessions/usage,
CPU readiness, prepared source/data/config and planned GPU-only operations.
First session hard limit 20 minutes including provisioning, CUDA/library setup,
correctness work and artifact transfer. scripts/l4-session.sh gives its supervised
worker 1080 seconds, then reserves cleanup time; no automatic allocation retry is
allowed for a prepared bundle. Stale owned markers trigger release before CPU
readiness checks. Unowned aliases and other sessions are left alone.

CUDA 12.4.1 redistribution components (nvcc 12.4.131, cudart/nvrtc/CCCL 12.4.127,
cublas 12.4.5.8, curand 10.3.5.147) are hash-verified and installed locally only;
component hashes/sizes/licenses are preserved. CUDA_COMPUTE_CAP=89 and two compile
jobs build the pinned CUDA backend on WSL CPU. A small existing Linux loader/
runtime bundle handles remote glibc compatibility; source/package notices are
retained. The first package relied on remotely exposed cuBLAS/cuRAND sonames and failed
in the loader before Rust could run. Its evidence remains preserved. A recovery
package carries the already CPU-acquired pinned cuBLAS/cuBLASLt/cuRAND shared
libraries plus their original notices; it never carries the WSL CUDA driver.
Inherited library paths and `/usr/lib64-nvidia` are preserved/searched. GPU VM
prerequisites are now an existing NVIDIA driver, L4 hardware and the CLI kernel. No remote ordinary package installation or Rust/
nvcc compilation is planned. Driver PTX/module JIT is unavoidable and included
in setup/warmup, not claimed to be a separately measured compilation interval.
Remote host orchestration, ELF resolution, input loading, CUDA transfers,
checkpoint serialization/reload and output packaging are unavoidable parts of
the CUDA phase and counted while allocated. Dataset/tokenizer construction,
analytical accounting, source/checkpoint hashing and statistical reporting run
locally. In-memory GPU-weight fingerprints are part of the initialization/update
correctness assertions, not final checkpoint hashing. Result verification is also local and after
confirmed release: Colab CLI 0.7.4 can exit zero on a kernel exception, so the
remote exit file and final Rust gate PASS are mandatory. Loader stderr/bootstrap
output are retained. A failed attempt is not converted into a pass.
No full training until measured-throughput/runtime/CU estimate has been shown
and explicitly approved. A missing CU rate is reported unavailable, never guessed.
The local wrapper must trap errors/interruptions and always stop the owned
`packtok-m5` session, then query sessions/usage. It must not stop other sessions.
No --keep, detached jobs, idle allocated GPU, Drive mounts or cloud-weight upload.

After GPU work, download results immediately, stop, then hash locally before editing,
analysis, approval or reporting. Bounded transfer recovery may not exceed the
session cap. Checkpoints stay local and ignored by Git. Manual crash recovery:
`colab status -s packtok-m5`, `colab stop -s packtok-m5`, `colab sessions`,
`colab usage` from Ubuntu-24.04. Abrupt host failure cannot guarantee cleanup.

The initial gate checks actual L4/CUDA execution, finite forward/backward,
nonzero finite gradients in all six parameter families, AdamW weight updates,
causality, tiny overfit, checkpoint identity, mapping/byte metrics and memory.
CPU reference: one layer, hidden 8, heads 2, FFN 16, context 8, vocabulary 512,
seed 19, input [1,2,3,4], targets [2,3,4,1]. All logits/gradients are compared to
CUDA with absolute tolerance 1e-5 plus 0.001 times the CPU absolute value.
Independent CPU finite differences check eleven representative weight tensors
at their largest-gradient coordinate, epsilon 0.001 and tolerance
0.001 + 0.03*abs(analytic gradient). These tolerances are numerical checks,
not a proof of every possible input. Squared-gradient reductions must be finite;
zero individual biases/unused embedding rows are allowed, zero entire families
and zero gradients for any weight tensor (including every attention/FFN/RMSNorm
projection/scale) are rejected. A CPU regression rejects an inactive projection
even when another bias in its family still receives a gradient.

Full primary preflight: seed 20261008, A/C identical initialization, 18 training
updates each at batch 8/context 256; the first two are warmup and sixteen are
timed. GPU-to-host ID transfer is checked. A/C each score a TRAIN-only 4097-token
prefix (4096 targets) for byte coverage/inference timing. No held-out score is
used. One fixed auxiliary A overfit fixture uses [1,2,3,4] repeated twice, rotated
targets, 100 steps, lr 0.005/decay zero, requiring loss below 0.25 and below ten
percent of initial loss. Its hyperparameters do not affect final models. A fresh
model at seed 20261009 must reload the saved weights and reproduce logits.
CPU overfit/reload tests use 100 updates, lr 0.01/decay zero, fixture seed 19 and
fresh seed 100; atomic checkpoint tests additionally use seed 20. The primary
CPU paired-update check uses seed 20261008.
Timings synchronize CUDA; raw logs separate provisioning/setup/init/compile/
training/evaluation/transfer/total allocated time and measurable compute units.
GPU preflight failure stops immediately after bounded diagnostic transfer; no
model-quality interpretation follows. nvidia-smi samples the owned device every
200 ms during CUDA work; reported VRAM is the maximum observed whole-device
sample, not an exact allocator high-water mark. Step timing includes batch tensor
creation/forward/backward/AdamW and synchronization; pipeline timing also includes
sampling/logging. Per-step subprocess telemetry is avoided; training snapshots
occur at step 1/every 100 steps, with the independent periodic monitor retained.

Checkpoints use standard FP32 safetensors with an external frozen model config,
not the historical PTLM encoding. A temporary file is fsynced and atomically
renamed; a partial temporary never replaces the last complete image. Final runs
retain one latest in-progress checkpoint (step 1/every 500), then one final image;
all curves/failed/completed run logs remain. Checkpoints contain model weights,
not AdamW moments/cursor, so exact interrupted-run resumption is unavailable.
Completed seeds are retained and an interrupted seed must restart under a later
approved budget. Checkpoint hashes are computed on WSL after download/release.

Bounds before tensor allocation: layers 1..16, hidden 2..1024, positive heads
dividing hidden, FFN 1..4096, context 1..1024, vocabulary exactly 512, analytical
parameter cap 128000000. Accounting uses u128 after bounds checks. Zero xorshift
seed maps to 0x9e3779b97f4a7c15. GPU atomic gradient reductions may introduce
same-seed training nondeterminism; CPU initialization/references are deterministic.

## Interpretation and limits

A replicated C−A gain would support this tokenizer-sequence signal under this
Transformer/corpus/budget, not general superiority. A disappeared/reversed
signal is a valid result. Three seeds are a small paired probe, not statistical
proof. Equal token context gives different raw-byte context; equal token work
gives different raw-byte exposure. These are reported experimental differences.
No factorization, adaptive tokenizer, routing, production server or M6 work.

Operational final classification is predeclared: three complete paired T seeds
with all C−A test bits/byte differences negative → M5_SIGNAL_REPRODUCED; all
nonnegative → M5_SIGNAL_NOT_REPRODUCED; mixed signs → M5_INCONCLUSIVE. Fewer than
three complete pairs remain incomplete. This is a small empirical sign check,
not significance proof; optional B outcomes are reported separately and may
reveal dependence on exposure/regime. No quality classification is made from
preflight training-prefix or overfit diagnostics.

The first prepared bundle has been attempted once; recovery preparation uses a
distinct `bundle-v2`/`bundle-v3` paths and never deletes the original bundle/failure logs.
No further allocation occurs automatically after the first failed preflight.
A new bounded preflight requires the user to authorize retry; full training still
requires throughput-based budget approval.

Recovery upload is bounded at 420 s, inside the unchanged 1080 s worker /
1200 s lifecycle cap. CPU packaging showed that the pinned runtime payload
would exceed the old 180 s upload allowance at the first observed transfer rate.
The v2 candidate was never allocated; the final candidate is v3 with this
explicit transfer allowance. This changes orchestration only, not experiments.

The recovery archive's CLI upload is split on local CPU into 33554432-byte chunks
(32 MiB). The official CLI base64-encodes complete input files, so chunking bounds
that host-memory amplification instead of creating one large JSON request. The
remote kernel only assembles the transport parts with a 16 MiB copy buffer before
unpacking. This is counted transfer/bootstrap overhead, not corpus preprocessing.
All parts have local hashes, fixed zero-padded ordering and a bounded aggregate
420-second upload inside the same total lifecycle cap. test-transfer.sh checks
upload/order/byte identity with a local mock; no GPU is allocated by it. V3 was
prepared but not allocated; v4 is the final recovery candidate with chunked
transport. Earlier archives/checks/hashes remain preserved.

Current user decision: stop at CPU state. The additional preflight proposal was
declined; no further GPU allocation or full training is authorized. A later
resume must explicitly authorize a bounded new preflight, then separately approve
measured full-run cost if that gate passes. No model-quality conclusion exists.

## Authorized recovery and scoped persistent storage (2026-10-08)

The user's subsequent instruction supersedes the previous stop decision for
exactly one additional L4 correctness preflight, at most 30 minutes including
cleanup. Full training remains locked. The scientific M5 configuration and
bundle-v4 executable/input identity remain frozen. [RECOVERY_STORAGE.md](experiments/m5-gpu/provenance/RECOVERY_STORAGE.md)
records the new independent transport/loader supervisor, 27-minute work deadline
and 3-minute cleanup reserve, unchanged bundle hashes, CPU checks, storage
protocol, failures and full-resume implementation prerequisites.

The same instruction narrowly supersedes the old Drive-mount/cloud-weight-upload
prohibition: user-authorized PackTok artifacts/weights may be backed up under
MyDrive/PackTok, using verified immutable copies. The primary path is local
Ubuntu-24.04 rclone; optional native Colab mounting is deferred. No unrelated
Drive data or Ubuntu/MOOS environment is modified. A follow-up requires a real
WSL upload/full-download/SHA-256 fixture pass before the L4 request. The initial
shared OAuth client hit Google's project quota; dedicated client setup is
pending. No second allocation or GPU model measurement has occurred at this
pre-allocation record. Existing weights-only checkpoints still cannot resume
optimizer/training state; long runs remain prohibited until complete resumable
checkpointing is implemented, tested and a measured budget explicitly approved.

## Recovery outcome after the authorized request (2026-10-08)

The independent WSL Drive fixture now passes with the dedicated Desktop OAuth
client and `drive.file` scope. Exactly one recovery allocation request was
issued from source commit 74d91332d6612b1e0b8a74734409319a1b542861. The CLI
printed `Session READY`, but the local user service received SIGTERM before
hardware inspection, bundle upload or Rust execution. The supervisor stopped
the owned alias and verified its absence; no second allocation is authorized.
The failed request, resource accounting and local supervision correction are
preserved separately from the original loader-127 attempt in
`provenance/l4-recovery-abafb6d8f2eb4e70a6ff3c3447bf917f`.

The corrected launcher retains the foreground WSL client with `systemd-run
--user --wait`, preserving the bounded service while the client remains alive.
A real 40-second CPU lifetime regression passed. The Python work/cleanup bounds
and systemd TERM/KILL bounds remain 1620/180 and 1680/120 seconds respectively.
No global WSL lifetime setting or other distribution was changed. GPU
correctness, throughput, memory and persistent model-weight recovery remain
unmeasured. Full training remains locked. Current state:
**M5_BLOCKED — INFRASTRUCTURE**.

## Dedicated repository maintenance (2026-10-08; no GPU allocation)

The outstanding recovery evidence is preserved in commit
1412541f40f6d11a2fdb674c2d832318473cac0e. Later maintenance keeps this scientific
protocol, original failure records, corpus/mapping/artifact paths and v4 attempt
marker unchanged. The supported host launcher now owns an independent bounded
WSL client; actual-entrypoint CPU tests cover five-minute execution, disposable
PAM-session termination, initiating-client interruption, deadlines, signals and
owned descendant cleanup. Release rechecks endpoint ownership before a stop and
refuses unknown/reassigned aliases. No CPU mock is GPU evidence.

The same drive.file/Desktop-client workflow remains active, with verified
readback and manifest-based fresh-destination restore. OAuth is still
External/Testing; Production publication needs explicit user approval and the
console's Branding prerequisite. No broader scope or permanent WSL setting was
introduced. See the [maintenance report](docs/maintenance/2026-10-08/REPORT.md),
[WSL design](docs/development/gpu-colab.md) and
[OAuth decision](docs/development/oauth.md). M5 GPU preflight remains incomplete;
the next allocation is a separately authorized task, not an automatic retry.


## Launch recovery — 2026-10-09 (no model result)

The first 20261009 A/C launch omitted the required approval environment flag; its archived remote stderr is absent, so the exact failure cause remains unconfirmed. The next one-allocation attempt passed the approval through Colab `--env` but failed before Rust because the bridge pre-created `results/` and the shell then used `mkdir results` under `set -e`. Its downloaded archive records exit 1 and the exact bootstrap stderr. The bridge now defers result-directory creation until the child exits and archives stdout, stderr and the actual child exit code. A CPU-only mock covers missing approval, successful handoff and nonzero status; four cases pass.

That allocation was released after 386 seconds (397 seconds request-to-cleanup), with displayed CU 167.24 to 167.15 and zero active assignments afterward. A and C each completed zero verified updates. No training, validation or final TEST metric, step throughput, VRAM peak or checkpoint exists. The one-allocation authorization is consumed; no retry was made. See the [attempt evidence](experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T115439Z/PILOT_ATTEMPT.md). This launch failure provides no support for or against the tokenizer hypothesis or the longer study.
