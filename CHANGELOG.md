# Milestone history

## Final pre-GPU correctness pass (2026-10-07)

- Enforce complete model-parameter-v1 artifact size during construction,
  loading and writing through one checked calculation, before allocating model
  state. Keep the existing size limits and exact wire encoding.
- Validate the whole greedy-generation prompt with the canonical model ID
  lookup, including prefixes outside the context and zero-continuation requests.
- Add nine scoped regressions for wire boundaries/overflow, constructors,
  historical wire identity, prompt validation and valid generation behavior.
- Preserve the original pre-fix review, probes and raw logs. Resolution and CPU
  baseline freeze evidence: [PRE_GPU_CODE_REVIEW.md](PRE_GPU_CODE_REVIEW.md).


## Performance and mathematics audit (2026-10-07)

- Addressed seven prioritized findings in model numerical failures, model
  loading/evaluation overhead, M2 workspace sizing, and batch/MAC validation.
- Added transactional Adam failure handling, offset-stable loss, finite inference
  score checks, pre-allocation body validation, direct weight loading,
  scalar evaluation loss, and span-sized M2 symbol buffers.
- Added nine tests, including finite differences across every parameter of small
  flat/factorized models, and additive `packtok-bench audit` probes.
- Preserved three before/after performance repetitions, failed regressions,
  intermediate failures, and a full M3 rerun. All twelve M3 model hashes match
  the historical run; quality conclusions are unchanged. See
  [PERFORMANCE_MATH_AUDIT.md](PERFORMANCE_MATH_AUDIT.md).

## M3 — Tiny autoregressive model comparison (2026-10-07)

- Adds a small CPU-only recurrent model boundary with a flat M1 head and a
  genuinely factorized M2 pack/local head.
- Adds a new manually authored train/validation/test split that is separate from
  the frozen M1/M2 tokenizer evaluation samples.
- Added a small CPU-only causal RNN with an M1 flat head and a teacher-forced
  M2 pack/local factorized head, Adam training, serialization, and greedy
  generation smoke tests.
- Added a fixed synthetic train/validation/test split and two comparison
  regimes, including a documented analytical-MAC budget.
- Ran three seeds per regime on the same Windows host; final byte-normalized
  results, raw logs, model artifacts, hashes, limitations, and Windows/Linux
  MSRV verification are in [M3_MODEL.md](M3_MODEL.md) and
  [M3_MODEL_BENCHMARK.md](M3_MODEL_BENCHMARK.md).
- M2 had lower mean bits/byte on this small held-out synthetic split while
  emitting more tokens. This is a narrow observation, not a general-quality
  claim or evidence that PackTok is better.

## M2 — First factorized pack architecture (2026-10-07)

- Added a generic Rust routing contract and the experimental, versioned lexical-v1
  policy for TEXT, NUMBER, and STRUCTURE spans.
- Added independent pack-local byte-level BPE graphs sharing the one reserved
  byte-fallback namespace, with a deterministic global learned-token budget and
  an independent slow reference implementation.
- Added artifact format version 3, pack-aware runtime encode/decode, inspection
  commands, corpus/allocation provenance, and the M1-matched benchmark harness.
- M0/v1 and M1/v2 artifact paths remain versioned separately. M1 remains the
  frozen control. M2's design, limitations, and measured result are recorded in
  [M2_PACKS.md](M2_PACKS.md) and [M2_PACKS_BENCHMARK.md](M2_PACKS_BENCHMARK.md).

## Post-fix branch audit (2026-10-07)

- Fixed valid repetitive corpora failing training after the per-token size limit:
  both trainers now skip oversized merge candidates and continue with valid pairs.
- Fixed heap-path throughput cliffs on near-uniform and alternating inputs using
  a bounded dense-repetition detector; canonical IDs remain identical to the oracle.
- Replaced panicking Unicode argument iteration with explicit native-argument
  validation and clear CLI errors on Unix and Windows.
- Added adversarial tests and extended stress benchmarks; preserved the slower
  intermediate detector measurements. See [POST_FIX_AUDIT.md](POST_FIX_AUDIT.md).

## PR #1 review fixes (2026-10-07)

- Addressed all seven Codex review findings. The corpus-path and bounded-read fixes
  were already included in the correctness audit.
- Made CLI decoding byte-exact for empty, text and arbitrary binary output;
  removed incomplete artifacts after write/sync errors while preserving existing files.
- Limited individual BPE token expansion to 1 MiB at construction and parsing.
  Oversized tokens that previously loaded are now rejected; IDs and wire records
  of artifacts within the bound are unchanged.
- Separated benchmark training and evaluation fixtures, checked for shared
  32-byte passages, and added record-length distributions and whole-process
  peak memory alongside the runtime vector counters.
- Preserved historical and unsuccessful measurements. Current results and test
  evidence are in [M1_REVIEW_FIXES.md](M1_REVIEW_FIXES.md).

## M1 correctness and performance audit (2026-10-07)

- Replaced full-vocabulary encoding scans with indexed, rank-ordered merge events;
  retained in-place scans for small models and uniform byte runs, and added a
  direct path for unmatched byte inputs. Raw-byte decoding bypasses merge stacks.
- Enforced artifact reader limits at construction/serialization for both format
  versions, and bounded CLI artifact reads before parsing. Wire formats and IDs
  remain unchanged.
- Compacted training symbols in place, removed whole-file corpus copies, made
  directory traversal iterative, and enforced file/path handling consistently.
- Added regression/differential tests, retained the former encoder in the
  benchmark, and added warm-up, stress cases and vector allocation/capacity metrics.
- Results and limitations are preserved in
  [M1_PERFORMANCE_AUDIT.md](M1_PERFORMANCE_AUDIT.md).

## M1 — Flat byte-level BPE baseline (2026-10-07)

- Added deterministic byte-level BPE training, rank-ordered runtime encoding,
  byte-exact decoding, a slow reference trainer/encoder, and corpus provenance.
- Added artifact format version 2 while keeping version-1 artifacts readable and
  writable without changing their serialized representation.
- Added the checked-in M1 fixture, CLI training and inspection commands, and a
  same-input M0/M1 benchmark harness.
- Benchmark observations and exact environment are recorded in
  [M1_BPE_BENCHMARK.md](M1_BPE_BENCHMARK.md).
- This is a flat baseline; no factorized PackTok packs, model integration, or
  claim of superiority is included.

## M0 — Contracts and byte fallback

- Established the Rust workspace and the `packtok-core`, `packtok-format`,
  `packtok-tokenizer`, `packtok-cli`, and `packtok-bench` crates.
- Defined pack/token contracts, deterministic version-1 serialization, and
  exact raw-byte fallback encoding/decoding.
- M0 benchmark results remain in
  [M0_BENCHMARK_BASELINE.md](M0_BENCHMARK_BASELINE.md); M1 does not overwrite them.

## M4 — Factorization ablation (2026-10-07)

Adds artifact-only balanced M1 grouping, bijective M2 flattening, equivalent
input-row control, four-variant CPU harness, sourced corpus preparation and
preserved preflight/final evidence. M0–M3 results remain frozen. Design and
results are in M4_ABLATION.md and M4_ABLATION_BENCHMARK.md.

## M5 preparation — 2026-10-08

Started the isolated Rust/Candle Transformer A/C experiment from exact CPU freeze
f3b6c3f; M4 delivery/provenance was pushed before branching. CPU-first dedicated
WSL preparation and strictly bounded L4 lifecycle are documented in the M5 design
and benchmark record. Historical tokenization, models and results remain frozen.
No final GPU quality result exists at this preparation stage.
