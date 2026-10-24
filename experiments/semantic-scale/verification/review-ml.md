# Independent ML and experimental architecture review

Reviewer: independent `scale_review_ml` agent. Scope: the new
`experiments/semantic-scale` experiment, not the frozen M5 experiment.
Reviewed registry feature alignment, model gradients/recomputation, AdamW,
streaming data, run plans, preflight/train/evaluate and complete resume.
Production code was not modified by this reviewer. Tiny CPU probes only; no GPU
allocation or 100M/1B tensor allocation occurred.

## Findings and disposition

### ML-1 — medium, fixed: one-update plans have no effective optimizer update

Initial `TrainPlan::validate` accepted `steps=1`. CLI planning chose zero warmup,
while AdamW's cosine schedule gives `learning_rate_at(1)=0` at its sole final
step. Preflight consequently failed `optimizer failed to update model weights`
for an otherwise accepted plan. This was a configuration contract error rather
than divergent model math. The root now rejects plans with fewer than two steps
before any model allocation; the notebook configuration is being aligned to that
bound. The existing multi-update cosine schedule and checkpoint state are not
reinterpreted.

### ML-2 — medium, fixed: metrics retained uncommitted tail across resume

Initially training appended to `metrics.jsonl` without reconciling it with the
loaded checkpoint. A valid interruption after logged update 4 but with latest
checkpoint 2 would replay updates 3/4 and leave duplicate/nonmonotonic training
rows. Model/optimizer resume itself was correct; unqualified learning curves and
aggregate exposure/throughput computed from that history could be misleading.

Root added a streaming reconciliation pass: preserve the complete old log in an
archive, remove canonical rows beyond the loaded checkpoint, publish a new
canonical log atomically and record an explicit resume boundary. Source identity
checks reject unrelated start rows, and interrupted final JSONL writes are kept
in the archive rather than treated as successful updates.

The independent executable probe now passes: prepare three distinct tiny splits,
plan four updates, run preflight and training, point latest back to the complete
step-2 generation, then resume. Canonical training steps are exactly `[1,2,3,4]`;
the old full log exists as `metrics-before-resume-*.jsonl`. Replayed losses at
steps 3/4 equal the initial attempt exactly (`6.071885108947754` and
`6.116955280303955`), with final counters 32 targets and 335 target raw bytes.
This checks reproducibility of the tiny reference, not large-model quality.
Probe source/output: `review-ml-probe.py`, `review-ml-probes.txt`.

### ML-3 — medium, fixed and independently verified: input-window context boundary

Precomputed corpus annotations can depend on expression bytes before a sampled
model window, giving the semantic model context outside the lexical baseline's
input window. Root had already identified this and assigned the data agent to
recompute semantic features from each window's actual input bytes only. The
completed implementation caches bounded exact lexical expansions, reconstructs
only the input token window, and recomputes annotations against those bytes;
stored whole-line annotation IDs do not enter the model. The feature policy was
bumped to `causal-known-window-v1`.

Independent inspection confirms the final target-only record is used only for
next-token targets and byte counters, never appended to the semantic input
window. The new regression proves that cropped `be honest ` has no inherited
HONESTY marker from the excluded `to `, that the complete visible phrase does
emit the marker, and that a delimiter present only as the final target cannot
activate any input feature. All six targeted data tests pass; output retained
in `review-ml-window-followup.txt`. ML-3 is closed.

## Checks with no reproduced defect

- Mean cross-entropy and gradient accumulation are consistent: every parameter's
  mean gradient is scaled once per microbatch. The checkpointed reverse pass uses
  unscaled upstream VJPs while scaling collected parameter gradients, which avoids
  multiplying the scale once per layer.
- Detached layer inputs, per-block recomputation and detached evaluation prevent
  a complete depth-sized autograd graph. FP32 normalization/softmax/loss are
  explicit around BF16 model tensors. Dense within-layer attention remains
  quadratic; no Flash Attention implementation or performance claim is implied.
- Baseline and semantic variants share model tensor layout and initialization,
  including six semantic rows. Baseline features are zero-masked, so parameter
  counts match. Semantic input is auxiliary observed-text information, not a
  replacement for lexical token targets.
- KnownVariant-only events wait for a complete following delimiter scalar;
  ambiguous/context-resolved `fr`/`ka` do not become features. Apostrophe
  delimiters are conservatively excluded. Prefix comparisons must keep the same
  already-observed lexical boundaries; re-tokenizing a shorter string changes
  BPE segmentation and is not that invariant.
- Byte-normalized evaluation sums mean token NLL times scored targets, then
  divides by the exact represented target bytes and ln(2). Context-stride windows
  do not count overlapping input boundary tokens twice as targets. Reporting is
  explicitly a bounded first-eight-window sample per held-out split, not a full
  held-out corpus score.
- Tokenizer training reads only a bounded TRAIN sample. Equal full split hashes
  are rejected; no near-duplicate/subsequence filtering is implemented. The
  user-supplied split provenance and a more representative corpus remain
  experimental responsibilities, so no generalization claim follows from demo
  or conformance data.
- AdamW stores FP32 masters/moments and can restore BF16 master precision. Seed,
  model/config/data identities, schedule and sample cursor are preserved. Root
  added deterministic expected-cursor checks and early context-length checks;
  the probe rejects a 4096-token context when TRAIN has only 825 records before
  GPU allocation.
- Admission memory accounting explicitly includes BF16 model + FP32 master,
  moments and one accumulated FP32 gradient map. It is a rough estimate, not
  measured VRAM or a guarantee of fitting a 1B L4 run. CUDA kernel correctness,
  numeric behavior, actual peak VRAM, checkpoint duration and throughput remain
  Colab-preflight measurements.

## Limits

The 1B preset is real analytical/configuration support, not evidence of a
completed 1B training run. CPU tiny-model probes cannot establish CUDA BF16
behavior or meaningful model quality. No blind model comparison, multi-seed
result, compression benefit or faster-learning result is asserted by this review.

## Final disposition

All three identified/known medium ML issues are now closed by source inspection
and focused executable checks. No reproduced critical/high/medium mathematical,
causality, metric-accounting or experimental-comparability defect remains from
this review. The named CUDA/large-corpus/actual-memory limits remain unmeasured
limitations, not inferred successes. Other Rust/Colab reviewers cover their own
independent scopes.
