# Milestone history

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
