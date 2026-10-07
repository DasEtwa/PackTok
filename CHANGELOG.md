# Milestone history

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
