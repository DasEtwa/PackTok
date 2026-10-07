# M3 — Tiny autoregressive model comparison

## Research question and boundary

M3 compares the frozen M1 flat byte-level BPE representation with the M2
factorized representation inside a deliberately small next-token neural model.
The comparison asks how tokenization and a genuinely factorized output change
byte-normalized predictive loss, parameter counts, training work, and runtime.
It does not establish general language-model quality or claim that PackTok is
better. M2 routing and artifacts are consumed as-is; this milestone does not
change the router, pack allocation, or tokenizer training policy.

The implementation lives in `packtok-model`, matching the experimental model
boundary already described in [STRUCTURE.md](STRUCTURE.md). Tokenizer training,
format loading, and runtime encoding remain in their existing crates. The model
crate depends only on `packtok-core` for stable `(pack, local)` IDs and does not
depend on tokenizer training, CLI, or benchmark code.

## Neural architecture

Both variants use one CPU-only causal recurrent layer. Hidden size and context
length are explicit configuration fields. For input token `x_t`, position `t`,
and previous state `h_(t-1)`, the cell is:

```text
h_t = tanh(E[x_t] + P[t] + W_h h_(t-1) + b_h)
```

The initial hidden state is zero. Position embeddings are learned and indexed
from zero within each training window. Training uses teacher-forced input
windows and predicts the next token at every aligned position. Evaluation may
provide one target after a context; the loss is charged at the last context
position. The cell is causal because each state depends only on its input,
position, and prior state.

The model is implemented directly in safe Rust using row-major `f32` parameter
arrays, explicit forward/backward loops, stable softmax cross-entropy, and Adam.
No tensor/numerical crate is used: the experiment needs only a single recurrent
cell and small vocabulary heads, and explicit loops keep operation counts,
parameter accounting, CPU behavior, and tests visible. The implementation is
CPU-only and single-threaded. Integer-seeded initialization and example sampling
are deterministic; floating-point results are expected to be repeatable on the
same target/toolchain, but bit-identical updates across different CPUs or Rust
versions are not promised. Softmax normalization accumulates in `f64`; stored
parameters and gradients use `f32`.

The M3 run uses hidden size 16 and context length 16. Adam uses learning rate
0.01, beta1 0.9, beta2 0.999, epsilon 1e-8, and global gradient clipping at
L2 norm 1.0. Weight decay, warmup, dropout, and mixed precision are disabled.
Initialization uses a xorshift integer state transition (xorshift operations
13/7/17; no multiplication step) and the run seed. The recurrent matrix starts
near 0.5 times identity with small seeded noise; embedding, position, and head
parameters use a seeded uniform range scaled by 0.15/sqrt(hidden_size). Biases
start at zero. All updates run on one CPU thread.

M1 input uses the flat token ID in pack `0` to index one embedding table. Its
output is one affine head over every M1 vocabulary ID.

M2 uses a separate local embedding table for each pack. Thus `TEXT:0` and
`NUMBER:0` index different rows even when their local IDs match. The output has
two levels: one affine pack head over all declared pack IDs, then one affine
local-token head for the selected pack. Teacher-forced training adds
`CE(pack | context) + CE(local | gold pack, context)`, the negative log of the
factorized joint target probability. The gold target pack selects the local
head during training, so a wrong pack prediction does not suppress the local
token learning signal. Greedy generation instead selects the highest-scoring
pack first and then the highest-scoring local token in that pack.

All declared local tokens, including the shared 256-byte fallback IDs, are
valid M2 input rows and output targets. No target is flattened into a shared
512-entry M2 output head. There is no normalization or unknown token.

## Parameter and compute accounting

Parameter totals include embedding tables, positional embeddings, recurrent
matrix/bias, and every output-head matrix/bias. The report splits these into
embedding, backbone (position plus recurrent cell), and output-head parameters.
M1 has `V*d` input embedding weights and `V*d + V` output-head parameters. M2
has `sum(pack_local_counts)*d` input embedding weights, a `P*d + P` pack head,
and `sum(pack_local_counts)*d + sum(pack_local_counts)` local-head parameters.
Here `d` is hidden size, `V` is flat vocabulary size, and `P` is the number of
M2 packs. This accounting includes biases and does not include optimizer state.

The benchmark contains two regimes. **A, same backbone**, uses the same hidden
size, context, batch policy, optimizer, learning rate, update count, and seed
for both tokenizers; its raw-byte exposure and output work may differ. **B,
compute budget**, targets the same deterministic estimated multiply-accumulate
budget as the flat run while retaining the same backbone and model dimensions.
The estimate counts the recurrent forward/backward matrix products and the
forward/output-gradient matrix products for the heads actually used. It is an
analytical operation estimate, not hardware instruction or FLOP measurement;
Adam, activation functions, softmax, memory traffic, and head-independent
overheads are excluded. The M2 budget may end after a full batch and therefore
slightly exceed the target. Parameter and estimated-compute differences are
reported rather than presented as exact equality.

Reported training wall time includes model allocation/initialization, window
sampling, optimizer updates, and scheduled validation checkpoints through the
last update. Final validation/test evaluation time is recorded separately.
Repetition summaries report arithmetic mean, population standard deviation,
and population variance; individual runs remain available in the raw report.
Training token/byte counts count every sampled target, including a corpus
position sampled more than once. Evaluation timing covers model scoring only;
tokenization and construction of the sliding-context example list are outside
that timer. The reported parameter/Adam/gradient storage estimate is 16 bytes
per parameter (weights, two Adam moments, one gradient vector); it excludes
activations, sampled windows, temporary logits, allocator overhead, and other
process memory. Process peak memory is a process-wide high-water mark.

In each repetition, the base model seed is 20,261,007 plus the zero-based
repetition index. The window-sampling seed is the model seed XOR
0xa341316c9e3779b9. A xorshift64 transition (13/7/17) chooses each start index
modulo the number of valid windows. Each update samples four windows; each
window has 16 input tokens and 16 shifted next-token targets. The hidden state
resets at every sampled window. Regime A uses 120 updates. Regime B uses the
same seed and data policy and updates through the first full batch whose
estimated MAC total reaches or exceeds the M1 120-update total, with an upper
safety cap of 2,000 updates. Validation loss is recorded after the first update,
then every 20 updates, and after the final budget batch. Validation values do
not alter configs or training decisions.

The MAC estimate counts, per input position, three recurrent matrix products
except that the first position has no previous-state gradient product:
(3*n - 1) * d*d per window of length n. Per supervised target it counts
3 * rows * d, for output logits, output-head weight gradients, and hidden
gradient accumulation. Flat rows=V; factorized rows=P+L_target_pack. The count
excludes Adam, softmax, tanh, bias/elementwise operations, allocations, and
memory traffic. This reproducible MAC estimate is not a hardware instruction
or FLOP counter.

Training windows are deterministic, seeded samples from each tokenizer's
encoding of the same raw training file. Batch windows have the configured
context length; hidden state resets at each window. Validation and test use the
same raw files for both tokenizers and score each next token from at most the
preceding configured context. The first token of each split file has no prior
token target and is excluded from loss and byte denominators.

## Dataset and tokenizer artifacts

The M3 split is manually authored synthetic text checked into
`fixtures/m3-model/` under the repository's Apache-2.0 license. It contains
separate train, validation, and test files. No M1/M2 tokenizer benchmark
evaluation file is used. The files are raw UTF-8 bytes, and there is no
normalization, reformatting, inserted separator, or downloaded data. Exact
file sizes and SHA-256 values, tokenizer artifact hashes, and token counts are
recorded in [M3_MODEL_BENCHMARK.md](M3_MODEL_BENCHMARK.md).

M1 and M2 tokenizer artifacts are trained only from the M3 training file with
the documented default 512-slot configuration and minimum pair frequency 2.
The exact serialized artifact bytes are retained under
`experiments/m3-model/artifacts/`. Artifact hashes identify the frozen inputs to
the model runs; the artifacts themselves define tokenization.

## Metrics and interpretation

For target token `i`, let `nll_i` be its negative log probability in nats and
`b_i` the exact number of raw bytes represented by that target token. The report
uses:

```text
loss/token = sum(nll_i) / number_of_targets
NLL/byte   = sum(nll_i) / sum(b_i)
bits/byte  = NLL/byte / ln(2)
```

Bits/byte and NLL/byte are the primary cross-tokenizer quality measures because
the two systems use different token sequences. Per-token loss and perplexity
are also shown as tokenizer-specific diagnostics; their values are not directly
comparable when vocabularies/token boundaries differ. Bytes per token,
sequence lengths, raw-byte target share, parameter partitions, training and
evaluation time, process peak memory, model artifact size, and estimated
multiply-accumulate counts are reported with the run environment and protocol.

For M2, results also include pack prediction accuracy overall and per target
pack, local accuracy evaluated under the ground-truth target pack, mean active
local head size, target pack frequencies, byte-fallback target frequency, and
pack transitions in encoded evaluation sequences. Output work is the aggregate
count of pack logits divided by pack plus ground-truth-pack local logits over
all scored test targets. It describes teacher-forced target evaluation; it is
not a measured instruction count for greedy generation. These are properties
of this lexical-v1 experiment and are not linguistic quality labels.

## Serialization and generation

The model has its own versioned binary parameter format, separate from tokenizer
artifact format versions 1–3. It stores model kind, dimensions, pack/local
vocabulary sizes, initialization seed, and little-endian `f32` weights. Loading
checks the version, reserved bytes, sorted pack IDs, dimensions, parameter
count, finite weights, exact input length, and documented size caps. Adam
moments are not serialized; model files are for inference/evaluation and do not
provide training resume.

The loader bounds dimensions to hidden size 2–1,024 and context length 1–4,096,
pack count to 1,024, each local vocabulary and total local rows to 1,000,000,
parameters to 16,777,216, and a serialized model to 64 MiB. These safety limits
are enforced before allocation. Greedy generation limits the returned
prompt-plus-continuation sequence to 4,096 tokens.

`greedy_generate` accepts an already tokenized non-empty prompt, predicts a pack
then local token for factorized M2 (or one flat token for M1), and returns the
prompt plus generated IDs. The benchmark smoke test tokenizes a raw prompt,
generates a short continuation, decodes it, and verifies the prompt bytes remain
an exact prefix. Generation quality is not scored; sampling is deferred.

## Known limitations and non-goals

- This is one small recurrent layer, not a production model and not a
  Transformer. A single CPU thread and explicit loops limit throughput.
- The manually authored corpus is synthetic and small. Results do not estimate
  general language modeling, other languages, or large-scale training.
- M2's lexical-v1 router may produce many byte-fallback targets and short
  structure spans; the M2 tokenizer and artifact remain frozen.
- Same update counts do not imply same bytes or compute. The second regime uses
  a documented analytical MAC estimate, which omits non-matrix operations.
- Wall time and process high-water memory include host/runtime effects; memory
  is process-wide and is not attributed to one model allocation.
- Greedy generation is only an end-to-end correctness smoke test.

M3 does not add POS, morphology, learned routing, adaptive vocabulary, model
serving, GPU training, sampling, quantization, or large-scale training.
