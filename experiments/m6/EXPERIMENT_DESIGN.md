# M6 experiment design

## Questions and comparability

The primary metric is bits per raw byte from the existing byte-weighted NLL scoring implementation. For each seed, report paired `C − A`; negative favors C. Token perplexity is not a cross-tokenizer quality metric. Report update/target count, sampled raw-byte exposure, wall time, synchronized step time, checkpoints and VRAM separately. No one-seed result is statistical proof.

The frozen M5 test set has already been evaluated and is exposed. It is not eligible as an independent final test for a revised M6 experiment. Use validation for predeclared decisions and reserve a new source-group holdout, acquired only after design freeze.

## T — Equal target-token positions

Reproduce pilot-2k-v1 exactly: A and C, same Transformer/initial parameter tensors, 14,681,984 parameters, FP32, context 256, batch 8, AdamW, LR 0.0003, weight decay 0.01, constant schedule, no warmup, 2,000 updates and 2,048 positions/update. Thus 4,096,000 target positions per variant. Use paired seeds and identical raw corpus/splits/tokenizer identities.

Evaluate validation at updates 1/500/1000/1500/2000 under the same implementation. Record stochastic sampled target-byte exposure from actual emitted sample accounting. These bytes can repeat; do not call them unique coverage or epochs. Equal shape and position count imply matching nominal forward/backward tensor dimensions, not measured equal total FLOPs or equal raw-byte exposure. M5 indicates C received 8.31% fewer sampled training bytes in the single pilot.

## B — Equal raw-byte exposure

Use a fixed training-source byte interval `[start,end)) in raw bytes and select a common boundary-aligned interval that every tokenizer can represent without assigning partial targets. Split/batch by each tokenizer’s own complete target tokens within that same byte interval. Count actual target bytes from the encoded records and require exact equality at the stop criterion (or predeclare a small boundary tolerance and report the exact residual). Reset both models to the same paired initial weights and train until the identical byte budget is consumed.

This necessarily produces different token target counts, updates, padding/context work and likely optimizer-step counts. Freeze and publish the sampler, batch construction and whether the final partial batch is padded/dropped; never hide it. Report positions, updates, synchronized wall time, memory and defensible operation counts per model. B isolates byte exposure more directly but is neither equal-token nor equal-compute. Do not combine B’s ranking with T as if it were the same estimand.

## Compute-aware frontier

Add a quality-versus-resource analysis from actual synchronized GPU training seconds and total session/allocation seconds. Distinguish:
- training-only step time (excluding data transfer/evaluation);
- validation/checkpoint/transfer/startup/cleanup time;
- total allocation wall clock and billed Compute Units;
- target positions, raw bytes, and measured throughput.

Compare paired validation BPB at common wall-clock budgets by using the nearest prespecified checkpoints or interpolation only when the method is predeclared. Equal wall time is not equal FLOPs. If reporting estimated FLOPs, publish the operation-count formula and assumptions and label it analytical; actual hardware counters/profile are preferable. Avoid attributing incidental faster kernels to equal compute.

## Paired seeds and statistical protocol

Use five new paired seeds, for example 20261009–20261013; keep pilot seed 20261008 as historical exploratory evidence, not an extra inferential replicate. Within each seed, initialize one canonical model tensor set and load the exact same tensors into A and C. Record a hash of canonical initial weights and prove tensor equality before training. Keep data order/sampling seeds paired wherever implementation permits, while noting tokenizer-specific token-to-byte mapping causes different raw spans.

Primary estimand: mean of five within-seed final validation BPB differences at the predeclared primary horizon. Report all five paired differences, mean, sample SD, median and a two-sided 95% paired-`t` interval; include a paired bootstrap sensitivity interval over seeds if useful, clearly noting n=5 limits. Also report the raw per-seed A/C values and an effect size in BPB and relative to mean A BPB. Do not infer significance from overlapping marginal intervals or choose a favorable domain after viewing results. Domain outcomes are secondary, predeclared, with per-domain denominators and multiplicity acknowledged; show all domains.

## Horizons and staged decision

A pilot T replication comes first at 2k to quantify paired-seed variance. Then perform B at a byte budget sized from observed M5 exposure, with a small CPU sampler dry-run first. For longer T runs, candidate horizons are 10k, 20k and 30k with five paired seeds only after the schedule/data adequacy review. Existing extended configs use warmup plus cosine and are not the M5 constant-LR continuation; freeze that schedule as a distinct experiment and hash all inputs before launch.

Proposed decision rule (freeze before new results are inspected):
1. Treat absolute paired effects below 0.01 BPB as practically small for this pilot scale; report exact estimates regardless of threshold.
2. Run 2k T for five paired seeds. If the paired 95% interval is wholly above +0.01 or below −0.01 BPB, report the directional result and prioritize B/data diagnosis before extending T. If it overlaps the practical-small band [−0.01,+0.01], the 2k result is inconclusive; run the byte-matched B comparison and consider 10k only if both variants still improve by at least 0.02 BPB over the final 500-update segment.
3. Escalate 10k→20k only if both validation curves improve by at least 0.02 BPB over their final 2,000 updates, there is no clear validation regression, and the projected compute budget has separate approval. Fix this threshold before launching the 10k stage.
4. Consider 30k only after 20k shows at least 0.02 BPB improvement over its final 2,000 updates without validation regression and after a fresh compute-budget authorization. A plateau or regression stops escalation.
These thresholds are operational decision proposals, not known natural constants. Report outcomes under the thresholds and all raw curves; do not tune the thresholds after viewing outcomes.

## Acceptance and limits

A run is usable only with source/config/input hashes, paired init hashes, complete update accounting, raw-byte exposure, validation curve and checkpoint checksums. A partial run remains reportable but cannot answer the full paired estimand. Do not change M5 artifacts or use its exposed TEST outcomes for tuning. This design does not authorize GPU work.
