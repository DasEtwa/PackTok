# M4 adversarial review checklist and evidence

This is an ongoing record; final completion and any confirmed issues are appended
after all final runs. The reviewer is the implementing agent, not an independent
reviewer. No sub-agent or external service was used.

- Leakage: only TRAIN is passed to token trainers. All split bytes are frozen
  before models run. The Rust 128-byte guard checks all offsets and concatenation
  boundaries; held-out removal is content-only and preserved. Shorter duplicates,
  related prose and synthetic template similarity remain possible. Validation's
  code contribution is zero after guard filtering, prominently documented.
- B bias: sorting uses unsigned artifact token bytes and original IDs only;
  balancing means vocabulary rows, not empirical target frequency. Model config
  P matches the paired artifact's active pack count (primary 4, tiny 3). Given
  the recorded P and M1 artifact, construction needs no M2 bytes or frequency
  table. All assignments are retained in the mapping file; no semantic grouping
  or held-out frequency enters assignment. The source artifact itself was trained
  on TRAIN, as in A; this is not a frequency-neutral tokenizer vocabulary.
- C ambiguity: ascending numeric pack/local IDs gives distinct rows even when
  expansions or local numbers match. All 256 fallback rows are included. Every
  encoded split maps back to identical original token IDs before a model trains.
- Input control: M3's independent pack rows are concatenated, so C/D inputs are
  equivalent. B permutes the same seeded M1 embedding rows, without extra pack
  input features. Hidden-state equality and gradient path use are checked.
- Loss: gold-pack conditional CE plus pack CE is a joint log probability. Uniform
  heads have the independent analytic ln(P)+ln(L) check. Previous all-parameter
  finite differences remain passing. Generation is pack-first greedy, not global
  joint argmax; local accuracy is measured under the gold pack, not generated pack.
- Normalization: represented target bytes exclude first-token bytes. Full raw
  split bytes are used only for bytes/token. Different tokenizers have different
  first-token exclusions and raw context lengths; byte denominators are reported.
- Parameters/MAC: independent test checks (3*n-1)*d²+3*d*head_rows for all four
  variants, including biases in parameter totals. Optimizer storage is excluded
  from parameter count. MACs omit Adam/softmax and therefore do not match elapsed
  CPU work exactly. Complete-batch overshoot and unequal byte exposure are reported.
- Seeds/exposure: exact matching seeds and sampling stream within each tokenizer
  row, no shuffled domain schedule or early stopping. MAC-aware runs have unequal
  update counts; they cannot be interpreted as the schedule regime's factorial
  effects. All seeds are kept.
- Artifacts: unique run directories reject overwrite. Tokenizers train twice;
  mappings regenerate from retained artifacts and serialize round-trip. All
  models reload and reproduce full scoring metrics before generation. Tiny A/D
  match every historical model hash. Canonical provenance spelling fixes the
  failed tokenizer identity check without changing merge/token semantics.
- Timing/memory: model test/validation timers exclude example construction;
  training includes mapped split copies and curve scoring. Reload and extra
  differential scoring are outside recorded test timing. Encoder timing is one
  allocation-inclusive encode; single observations are not stable throughput.
  Process peak includes leakage-check hash sets, tokenizer training and every
  model's data, and is not an isolated variant's memory figure. The vector-state
  estimate counts four f32 arrays per parameter, excluding activations and corpus.
- Documentation: frozen protocol, failed acquisition/preparation/launch and path
  attempts, source notices/hashes, raw models, curves and per-seed outcomes are
  retained. Rust tokenizer/model formats stay unchanged; only the separately
  versioned, bounded model-side bijection format is added.

Known interpretation limits: one corpus mixture, one CPU, three seeds, tiny
RNN, token-based exposure and fixed context. No general advantage, GPU speed,
statistical causal independence, or Transformer outcome can be inferred.

## Final review outcome

All final tiny and larger runs completed with all declared seeds. The post-run
review found no additional numerical, mapping or exposure implementation error.
It added a direct joined-path/full-frozen-artifact regression for the already
corrected provenance failure. Windows/Linux strict verification passes; historical
model and tokenizer hashes match. The domain mismatch, shared peak memory, head
initialization stream offsets, irrelevant B reserved-ID diagnostic and MAC/byte
exposure limitations remain explicit. No outcome-driven changes were made.
