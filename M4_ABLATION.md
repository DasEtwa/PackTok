# M4 — Factorization ablation and larger-corpus validation

## Frozen question and factorial design

M3 changed both tokenization and output architecture. M4 tests where its signal
comes from without changing lexical-v1, allocation, token formats, or the RNN.
Negative results are evidence. M0–M3 raw results remain frozen. M4 is stacked on
the audited M3 head; the audit preserved every historical model hash.

| Variant | Token sequence | Input embeddings | Output |
|---|---|---|---|
| A | M1, unchanged | M1 global | global flat |
| B | M1, unchanged | same M1 global rows, storage permutation only | synthetic pack then local |
| C | M2, unchanged | canonical concatenated global | global flat |
| D | M2, unchanged | same canonical concatenated rows | M2 pack then local |

C's canonical order is ascending numeric pack ID, then ascending local ID.
Consequently TEXT precedes NUMBER, STRUCTURE, and BYTE_FALLBACK, when present.
Only artifact-declared nonempty packs participate. All byte rows are included.
Flatten/unflatten is bijective; no string identity, rerouting, or deduplication
occurs. C maps generated global IDs back before decoding.

B sorts **all** frozen M1 token byte expansions lexicographically as unsigned
bytes, breaks equal-expansion ties by original global ID, and assigns sorted
rank modulo P to synthetic pack ID. Locals increment in sorted encounter order.
P equals the number of nonempty M2 model-facing packs. Sizes differ by at most
one. Names are SYNTHETIC_0, SYNTHETIC_1, etc., with no linguistic meaning. No
corpus frequencies, held-out bytes, or model scores are inputs to this algorithm.
The frozen tokenizer artifact supplies expansions; the frozen M2 artifact supplies
P. This couples grouping count to the paired TRAIN-derived artifact, not TEST.
There is no secondary semantic grouping or vocabulary-size tuning in this study.

## Input control and model-side artifacts

M3 local embedding tables already occupy one contiguous, independently trainable
matrix, ordered by pack/local ID. C uses exactly that order, so same-seed C/D
embedding, positional, and recurrent parameters and hidden states are identical
before training. No shared pack vector, extra pack conditioning, or tied row is
present. D-flat-input would therefore duplicate D mathematically and is omitted.

B stores these same independent global M1 input rows in pack-concatenated order:
`E_B[synthetic(id)] = E_A[id]`. The permutation changes only addresses; forward
lookups and embedding gradients use the same vectors, without pack-dependent
input transformations. Initialization first creates the same seeded global rows,
then permutes them. A/B and C/D hidden-state equality is regression-tested.
Head initialization uses the M3 seeded family; differing head shapes consume
different random-stream positions. Heads are not identical parameter objects.

`packtok-model::IdMapping` owns generic model-side bijections; the benchmark
derives them from tokenizer artifacts. Mapping files are necessary because the
M3 parameter format has no tokenizer-ID permutation: encoding it there would
change historical model bytes. A separate bounded mapping format is documented
in FORMAT.md. Tokenizer v1/v2/v3 and model parameter v1 stay unchanged.
Mappings are retained beside models and tokenizers and verified against the
artifact-derived map. Model files are inference-only; Adam state is not resumed.

## Loss, generation, and normalization

B/D use `CE(pack) + CE(local | gold pack)` for training and scoring, equal to
minus log joint target probability. The gold pack selects the conditional head.
Greedy generation chooses the pack first and then its local argmax; this is not
the global joint-probability argmax. B maps the generated synthetic ID back to
the original M1 ID; C maps global M2 IDs back to original PackTok IDs. All four
use raw prompt `At the harbor`, eight greedy tokens, and exact byte-prefix checks.
Generated arbitrary bytes need not form valid UTF-8; decoding is byte-oriented.
Generation content is not a quality metric.

Scoring reuses M3's sliding context: every next token is predicted from at most
the preceding 16 tokens, resetting recurrent state and position at each window.
The first token is unscored; its exact represented bytes are excluded too.
For target losses n_i and byte lengths b_i:

```
loss/token = sum(n_i) / target_count
NLL/byte = sum(n_i) / sum(b_i)
bits/byte = NLL/byte / ln(2)
bytes/token = entire raw split bytes / entire encoded split token count
```

The denominator therefore differs by the tokenizer's first-token byte length;
raw/scored bytes are retained per run. No normalization or UNK is introduced.
Per-token perplexity is not used to compare tokenizers.

## Predeclared corpus and split

Sources are fixed before model outcomes: Austen's *Pride and Prejudice* (English),
Goethe's *Die Wahlverwandtschaften* (German), Rust 1.85.0 vec/btree-map source and
Cargo config, and separately labeled synthetic JSON and Unicode/numeric records.
Exact downloads, full Gutenberg notices, and Rust MIT/Apache licenses are retained.
Sources, license scope, hashes and contributions are in M4_ABLATION_BENCHMARK.md
and `experiments/m4-ablation/sources/`. Historical M3 synthetic splits are reused
unchanged for a separate compatibility run, not merged into this corpus.

Rust preparation strips only Gutenberg header/footer outside the explicit start
and end markers. Other source bytes are unchanged. Split blocks target 4096 bytes,
extending through the next LF; source-local block index modulo 20 sends residues
18/19 to validation/test, all others to TRAIN. Sources retain their listed order;
no bytes are inserted between blocks. Rounding/short sources change the nominal
90/5/5 proportions. Before model/tokenizer runs, entire held-out blocks sharing
any contiguous 128-byte passage with TRAIN are dropped, then TEST blocks sharing
one with validation are dropped. The final guard checks every byte offset across
concatenation boundaries. Composition records all removed bytes. This conservative
content-only filtering is frozen before results; no outcome-based split adjustment
is permitted. Shared shorter language/code patterns remain expected; the guard
does not prove statistical independence or detect paraphrases.

Synthetic sources have 2000 numbered records each, generated by the Rust tool;
the full formula and outputs are retained. They provide JSON, Unicode, numbers,
and punctuation coverage and are not presented as sourced natural language.

## Frozen training and compute protocol

Both tokenizers train twice only on TRAIN, using 512 logical slots, 256 shared
bytes and at most 256 learned tokens, minimum pair frequency 2. Byte-identical
artifact pairs are required. M2 allocation remains its existing global competition.
Fresh TRAIN-derived artifacts are used for the larger corpus. Tiny compatibility
artifacts must equal historical M3 bytes. No 1024-slot experiment is scheduled.

The fixed CPU model retains hidden 16/context 16, tanh RNN, f32 parameters and
gradients, f64 softmax normalization, xorshift initialization 13/7/17, recurrent
0.5 identity plus noise and other weights scaled by 0.15/sqrt(hidden). Adam is
learning rate .01, beta1 .9, beta2 .999, epsilon 1e-8, global clip 1; no dropout,
decay, warmup or mixed precision. Three seeds are 20261007, 20261008, 20261009.
Sampling uses seed XOR 0xa341316c9e3779b9 and the unchanged M3 four-window policy:
16 inputs and 16 shifted targets, uniform token-start index, state reset per
window. Paired A/B and C/D see identical sampled original token windows.

Schedule regime: 120 updates for tiny compatibility, 2000 for the larger corpus.
MAC regime: each seed's A schedule total is the target for all variants. Stop
after the first complete batch reaching that budget, safety cap 20000 updates.
The cap is a failure, never an early-stop policy. Curves use update 1, schedule
quarters/final, or MAC quarters/final. Validation only reports; it changes no
config or stopping decision. Hidden size is not tuned. Three rather than five
seeds bound CPU time; secondary groupings and vocabulary scale are omitted.

M3 analytical MAC v1 is preserved: `(3*n-1)*d*d` per input window plus
`3*d*(V)` per flat target or `3*d*(P+L_gold)` per factorized target. Adam,
softmax, tanh, biases, allocations and traffic are excluded. Budget matching
does not match wall time, raw-byte exposure, effective raw context, or total
parameters. All actual updates/targets/bytes/MACs are reported. Parameters are
`V*d` embeddings, `context*d+d*d+d` backbone, `V*(d+1)` flat head or
`(V+P)*(d+1)` factorized head. Parameter+Adam+gradient storage is 16*parameters
bytes and excludes other buffers/allocator overhead.

Training wall time includes model creation, mapped split copies, sampling,
optimizer, and curve validation; final validation/test are separately timed.
Scoring timers include only model evaluation after constructing examples; model
serialization, reload/differential checks, and generation are outside. Encoder
timing boundaries and process-wide peak scope are explicit in the benchmark.
All variants run in one CPU process, A/B/C/D order within each seed/regime;
high-water memory includes tokenizers, all encoded splits and prior allocations,
and cannot be interpreted as isolated per-model peaks. Timings are one observation
per seed without affinity/frequency locking; no GPU inference is made.

## Interpretation

Report paired B-A, C-A, D-A and `(D-C)-(B-A)` in bits/byte with individual seeds,
mean and population SD. These are empirical factorial contrasts, not independent
causal effects. Nonzero interaction means the combination cannot be explained
by adding the observed single changes. Budget-regime comparisons additionally
vary updates/exposure and must be interpreted separately.

If B improves while C does not, generic output factorization is supported as the
main mechanism; C-only improvement supports the tokenizer sequence. Both gains
support contributions from both mechanisms; D-only gain suggests interaction.
If the larger corpus removes M3's gain, the original signal was dataset-specific
under this model/protocol. Losses everywhere remain valid results. One mixed
corpus, tiny RNN, short token context and three seeds cannot establish general
superiority or settle Transformer scaling. M5 may follow only after this result
is understood; M4 implements no new routing, packs, Transformer or GPU backend.
