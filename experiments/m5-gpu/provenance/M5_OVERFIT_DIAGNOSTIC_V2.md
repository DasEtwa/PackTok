# M5 tiny-overfit diagnostic revision 2 and CPU preparation

Date: 2026-10-08. Verified base Git HEAD: `377fb9e6ab0af0c408f489419b142600f7752329`. No GPU was allocated during this diagnostic or the code preparation recorded here.

## Preserved L4 failure and diagnosis

The previous real L4 run remains a failure at the original 100-update memorization gate: CUDA initialization, the 14,681,984-parameter model, forward/backward, finite nonzero gradient families, AdamW and 18 ordinary A updates passed. On the fixed eight-token fixture, loss was 7.5863 at step 0 and 2.5422 after 100 updates (3.896 seconds); the unchanged requirement was below 0.25 and below 10% of initial loss. Sampled peak memory was 2,250 MiB; ordinary measured step mean was 91.1 ms. The run stopped before C because the old gate returned early. These values are historical GPU observations and are not overwritten by the new CPU result.

The fixture has no dropout, uses the same input/targets at every update, no warmup, no clipping and zero weight decay, with learning rate 0.005. The preflight starts the fixture from A after 18 sampled TRAIN updates. Its initial loss 7.5863 is therefore not an initialization measurement. A fresh CPU model starts at 6.246869, close to `ln(512)=6.238325`; this does not indicate a broken output-head initialization.

The isolated 500-update diagnostic in `src/overfit.rs` uses only the frozen A TRAIN sequence to reconstruct the fresh and post-18 states. It uses the same fixed batch for every optimizer update and reports steps 0, 1, 10, 25, 50, 100, 200, 350 and 500. Every update checks loss and finite gradients; milestone checks inspect every parameter. It records divergence, nonfinite values and 100-step stagnation. The pass predicate remains strictly below 0.25 and below 10% of step-0 loss. An optional explicit learning-rate comparison uses TRAIN only; none was needed for the CPU result.

CPU result details and raw JSONL are in `overfit-diagnostic-cpu-20261008/`. Fresh loss: 6.246869 → 0.270227 at 100 → 0.000608 at 200 → 0.000151885 at 500, 428.928 s. Exact deterministic post-18 replay: 7.586294 → 0.937163 at 100 → 0.342821 at 200 → 0.239442 at 350 → 0.238519 at 500, 416.889 s. Both CPU runs had finite gradients and parameters with no divergence or stagnation and satisfy the original numeric threshold at 500. CPU timings are not GPU timings.

This establishes that 100 updates are insufficient for this fixed-batch gate on CPU, and that the reproduced post-18 weights can meet the unchanged threshold after 500 CPU updates. The CPU trajectory does not reproduce the historical L4 step-100 loss (0.937 versus 2.542), although its step-0 loss matches. Therefore no claim is made that the GPU 500-step gate passes; that GPU-only check remains mandatory. The current preflight attempts the same 500-step post-18 check, reports the original threshold unchanged, saves and reloads the model-only Safetensors checkpoint before the optional memorization gate, and completes the C initialization/training checks before returning gate failure.

## Frozen longer-run preparation

The original M5 preregistration and `configs/primary.json` remain unchanged. Separate frozen proposals are `configs/pilot-2k-v1.json`, `configs/extended-20k-v1.json` and the separately budgeted `configs/extended-30k-extension-v1.json`. All pin the unchanged A/C tokenizers, corpus manifest, encoded train/validation/test sequences, model shape/count, batch 8, context 256 and AdamW parameters.

The pilot is one paired seed and 2,000 T updates per variant at the original constant 0.0003 learning rate. The CPU plan counts 4,096,000 target positions per model and 7,575,564 represented target bytes for A versus 6,946,291 for C at seed 20261008. Its conditional training-only projection uses the historical 91.1 ms GPU step mean: 364.4 s (6.1 min) for 4,000 A+C updates and 0.1559 CU at the previously displayed 1.54 CU/hour. This excludes provisioning, validation, checkpointing and billing delay; it is a proxy, not a quote or approval.

The 20,000-update proposal has five paired seeds (20261008–20261012), linear 100-update warmup and cosine decay at 0.0003. T matches target-token positions/updates and logs raw-byte exposure. B selects an exact shared raw-byte interval whose start/end are boundaries in both encoded TRAIN sequences, bounded by 8 MiB; it logs target counts, updates and GPU time separately. This matches represented bytes, not FLOPs. The optional 30,000-update schedule is a separate proposal and needs separate budget approval. None of these GPU jobs ran.

## Resume and held-out reporting

src/resume.rs adds resumable AdamW with pinned-Candle-equivalent update arithmetic. Safetensors contains model tensors, first/second moments, optimizer/scheduler counters, training step, sampler cursor and RNG state, plus exact config/corpus/tokenizer/train-sequence identity and a SHA-256 over canonical tensor names/shapes/values. It writes and syncs a temporary image, reparses and verifies it, then atomically renames. The step-zero image is saved before metrics logging. On resume, complete JSONL rows beyond the loaded checkpoint are atomically removed, a partial trailing row is discarded, malformed complete rows fail closed, and step/pipeline/wall timing plus sampled peak VRAM are restored from the last matching checkpoint record. The CPU interruption test compares four uninterrupted deterministic updates against two updates plus save/load plus two updates exactly.

`src/domains.rs` maps manifest units without changing held-out data. Evaluation preserves the existing context-256 windows across category boundaries; a target token spanning different categories is assigned in full to `mixed-domain`. English drama/prose aggregate as English literature; code, German prose, synthetic JSON and synthetic Unicode remain separate. Per-domain output includes target bytes, tokens, 8 KiB-block count and deterministic 1,000-resample 95% block-bootstrap interval where at least two blocks exist. Category plus mixed-domain totals are checked against the global byte-normalized score. Structured held-out data has zero retained bytes and is explicitly reported unavailable.

The new extended trainer requires an explicit matching `PACKTOK_M5_GPU_APPROVAL` reference, checks the physical device is NVIDIA L4 before CUDA initialization, and has no CPU or accelerator fallback. No GPU training command was executed during preparation.


## CPU verification

On 2026-10-09, the M5 crate passed 33 tests in debug, release and Rust 1.85 debug. The PackTok workspace passed debug and release tests plus Rust 1.85 debug tests. Strict all-target Clippy passed for both workspace and M5 crate on stable and Rust 1.85; formatting and git diff --check passed. Verification was CPU-only; it does not establish CUDA correctness or a passing GPU memorization gate.
