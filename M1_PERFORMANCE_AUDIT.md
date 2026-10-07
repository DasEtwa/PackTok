# M1 performance and correctness audit — 2026-10-07

**Historical protocol:** these measurements predate the PR review changes and
use training-like samples with overlapping passages. The model and split differ
from the current held-out harness, so throughput and compression numbers should
not be compared directly across reports. Current results and regression evidence
are in [M1_REVIEW_FIXES.md](M1_REVIEW_FIXES.md). All original runs, including the
unsuccessful intermediate optimization, remain preserved.

This audit reviews the existing flat BPE baseline. It does not add factorized packs
or establish an advantage over another tokenizer or language model. Historical
results in [M1_BPE_BENCHMARK.md](M1_BPE_BENCHMARK.md) remain unchanged.

## Fixed issues

- **Repeated full-input encoding scans:** the previous encoder inspected every
  stored rank and allocated a replacement vector for every applicable merge.
  The runtime now indexes matching pairs and processes only matching adjacencies,
  ordered by rank and original byte position. Invalidated events are discarded.
  Small models and uniform byte runs retain a contiguous in-place scan path;
  unmatched inputs produce byte IDs directly. This preserves canonical IDs.
- **Decoder overhead for byte tokens:** raw byte IDs bypass merge-stack setup and
  expansion. Learned tokens retain iterative decoding, including deep chains.
- **Unreadable serialized artifacts:** construction previously accepted 65,537
  metadata entries or special tokens although the reader allows only 65,536.
  Version-1 construction also accepted artifacts larger than 16 MiB. Regression
  tests reproduced all three failures before the fix. Both constructors and
  writers now enforce the same limits as the reader; valid wire bytes are unchanged.
- **Unbounded CLI reads before validation:** artifact files were read completely
  before the parser checked their size. Reads now stop after 16 MiB plus one byte,
  which also rejects an input that grows while being read.
- **Training and corpus buffers:** merge replacement compacts the training symbol
  vector in place. Zero-merge training avoids allocating corpus-sized symbols.
  Corpus loading appends through a 64 KiB buffer rather than retaining an extra
  complete file alongside the concatenated corpus.
- **Corpus traversal:** directory traversal is iterative, limits file counts
  while collecting them, rejects non-regular entries and non-Unicode paths, and
  constructs portable relative paths from components rather than replacing literal
  backslashes in Unix filenames. Valid file ordering and concatenation are unchanged.
- **Diagnostics and measurement:** pack errors now name the expected pack correctly
  for both tokenizers. The benchmark retains the former scan encoder, adds warm-up
  and stress workloads, and reports runtime vector allocation/capacity counters.
  The copied BRAIN README is explicitly marked as a historical architecture snapshot.

## Reproduction and provenance

- Starting commit: `bfd3224f46f30c1bb61682a94cdea927c2fca079`.
  These observations were captured before committing the audit; fingerprints of
  that measured source state are in
  [source-sha256.txt](experiments/bpe-baseline/audit-20261007/source-sha256.txt).
- Command: `cargo run --release -p packtok-bench` from the repository root.
  Three final executions used the equivalent built `target/release/packtok-bench.exe`.
- Rust: `rustc 1.98.1 (48a229cea 2026-09-01)`, LLVM `22.1.8`,
  `x86_64-pc-windows-msvc`; default Cargo release profile, no external Rust crates.
- Windows 10 Pro `10.0.19045`; Intel Core i7-2600 @ 3.40 GHz,
  four cores/eight logical processors. No CPU affinity or frequency locking.
- Training input: checked-in `fixtures/m1_bpe_corpus.txt`, 712 bytes,
  FNV-1a `3bafa3aa89541485`; target 512 IDs, at most 256 merges, minimum frequency 2.
  Result: 84 merges, 340 IDs, 1,881-byte artifact. No normalization, annotation,
  pack factorization or model experiment; no random benchmark seed.
- The four original samples are unchanged and repeated 32 times. Each operation
  receives 25 ms warm-up and 200 ms measurement. Allocation is included in timing;
  instrumented calls happen separately. Scanner IDs and decoded bytes are checked
  before timing. Samples execute in a fixed order, without interleaving.
- The corpus fixture is also used in one stress workload. This is an implementation
  benchmark, not a held-out compression or model-quality evaluation.

Raw output is preserved in [before.txt](experiments/bpe-baseline/audit-20261007/before.txt),
[after-1.txt](experiments/bpe-baseline/audit-20261007/after-1.txt),
[after-2.txt](experiments/bpe-baseline/audit-20261007/after-2.txt) and
[after-3.txt](experiments/bpe-baseline/audit-20261007/after-3.txt).

## Throughput

Final values are medians of all three runs, in MiB/s. The starting run had no
warm-up and is a single observation, so its before/after ratios are approximate.
The retained scanner uses the current warm-up and identical artifacts/inputs;
its comparison controls those differences. Moving old code into the benchmark
can change compiler layout, so the retained scanner is not the old executable.

| Sample | Starting BPE encode | Current encode | Retained scan encode | Current / scan | Starting decode | Current decode | M1 tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English prose, 4,160 B | 14.17 | 19.29 | 12.51 | 1.54x | 217.47 | 268.57 | 3,264 |
| German prose, 4,736 B | 15.90 | 21.35 | 14.41 | 1.48x | 216.80 | 238.60 | 3,136 |
| Unicode-heavy, 4,320 B | 16.61 | 37.98 | 14.76 | 2.57x | 218.26 | 274.01 | 3,488 |
| Source-like, 5,952 B | 14.34 | 25.45 | 12.70 | 2.00x | 215.29 | 254.84 | 4,288 |

Token counts, bytes/token and sequence reductions remain identical to the original
M1 results. Current BPE remains substantially slower than the same-run M0 byte-only
encoder (roughly 1,320–1,410 MiB/s); fewer tokens do not establish model quality.
Fixture training observations were 4.985, 4.815 and 4.634 ms versus 5.083 ms in the
starting run. These are short single-call observations, not evidence of a general
training-throughput improvement.

## Stress workloads and rejected regressions

| Workload | Input bytes | Merges | Tokens | Current encode | Retained scan | Current / scan |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `a` repeated; trained from 512 `a` bytes | 16,384 | 8 | 64 | 604.74 | 433.93 | 1.39x |
| Same run; 64 unused `(0, right)` byte pairs appended | 16,384 | 72 | 64 | 488.80 | 406.69 | 1.20x |
| Unmatched `Z` repeated; fixture model | 16,384 | 84 | 16,384 | 252.98 | 19.52 | 12.96x |
| Training fixture repeated 32 times | 22,784 | 84 | 13,728 | 15.77 | 7.76 | 2.03x |

An initial heap-only encoder lost badly on dense repeated bytes: approximately
5.32 MiB/s versus 432.10 MiB/s for the scan encoder. This motivated the contiguous
path. An intermediate implementation still measured 408.41 versus 434.62 MiB/s
on that workload (0.94x), preserved in
[intermediate-repetition-regression.txt](experiments/bpe-baseline/audit-20261007/intermediate-repetition-regression.txt).
The final implementation uses compact local-ID scans and checks uniform runs in
32-byte blocks. These measurements are retained rather than treated as wins.

## Allocation and memory trade-off

The optional `encode_bytes_with_stats` / `decode_bytes_with_stats` methods measure
successful vector capacity-growth requests and the maximum simultaneous vector
capacity. Normal encode/decode uses a specialization with counters disabled.
These metrics exclude the model, caller input, allocator overhead and process RSS.
They do not measure every allocation in training or artifact parsing.

| Sample | Encode / decode allocation requests | Peak encode buffers | Peak decode buffers |
| --- | ---: | ---: | ---: |
| English | 3 / 2 | 192,512 B | 4,176 B |
| German | 3 / 2 | 214,528 B | 4,752 B |
| Unicode | 3 / 2 | 200,704 B | 4,336 B |
| Source-like | 3 / 2 | 272,384 B | 5,968 B |

The event encoder exchanges fewer allocations and better throughput for larger
working buffers. On this 64-bit host each linked symbol is 24 bytes and each
queued event is 16 bytes. The former scan path used 4-byte local symbols and
allocated a replacement for each matching rank; its nominal peak capacities on
these samples were about 37–51 KiB. Current peaks are roughly five times that
capacity, not a memory reduction. The pair index adds 3,400 bytes for this model.
Unmatched inputs allocate only the output vector. Training still rebuilds pair
frequencies each rank and retains the full corpus in memory.

## Verification

- `cargo fmt --all --check`: passed after formatting the final test additions.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: passed.
- `cargo test --workspace`: 51 unit tests passed; no failures; doc-test suites empty.
- `cargo build --release --workspace`: passed.
- Added 14 tests covering constructor limits, accepted boundary counts, bounded
  reads, buffer counters, uniform-block boundaries, deep decoder chains, corpus
  chunks/empty inputs and byte fallback.
- Differential runtime checks add 9,840 exhaustive inputs across three models,
  all 713 fixture prefixes, and an input containing every byte value repeated
  16 times. Existing generated trainer/runtime reference checks still pass.
- CLI training twice, validation, token/merge inspection, overwrite protection and
  the text `Sämtliche Häuser äöüß 👩🏽‍💻` round-trip passed. CLI text output keeps its
  existing newline framing; byte-exact runtime tests cover arbitrary bytes.
- Both new CLI artifacts retained the original SHA-256:
  `CEF973A354422C88E6FEF54D0B6EEE09495980DF4E5BD0B7E5ADC37896E7D907`.

This audit ran on Windows only. Unix filename handling is corrected in source,
but no Unix-host test run was performed. Timing remains host/workload dependent;
no large-corpus capacity, process peak-RSS or language-model claim is made.
