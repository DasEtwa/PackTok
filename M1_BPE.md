# M1 — Flat byte-level BPE baseline

M1 is PackTok's first flat tokenizer baseline. It is a control group for later
PackTok experiments, not the factorized pack system and not evidence that PackTok
outperforms another tokenizer.

## Representation and IDs

Input `&str` is read as its existing UTF-8 bytes. M1 performs no normalization,
pre-tokenization, language detection, whitespace rewriting, or special-token
recognition. The 256 byte values are always available as base symbols.

The flat vocabulary uses pack ID `0`. Local IDs `0..=255` are raw byte values;
learned tokens begin at `256`. Merge rank `r` creates local token ID `256 + r`.
Every learned token's byte sequence is defined by its two earlier parent IDs. Its
display text is derived at inspection time and is never authoritative. The M1
artifact also retains the separate M0 byte-fallback descriptor; M1 runtime output
uses the flat pack and can still represent every possible byte sequence.

The runtime API accepts `&str` and also exposes raw-byte encoding. `decode_bytes`
works with invalid UTF-8; the `decode` convenience method returns a UTF-8 error
when those bytes are not valid text. There is no `<unk>` token.

## Training configuration

`BpeTrainingConfig::default()` uses:

| Option | Default | Meaning |
| --- | ---: | --- |
| `target_vocab_size` | 512 | Total IDs including the 256 byte tokens |
| `max_merges` | 256 | Upper bound on learned merges |
| `min_pair_frequency` | 2 | Minimum adjacent-pair count before selection |

The target must be at least 256. `max_merges` cannot exceed the target minus 256.
Minimum pair frequency must be positive. A pair is eligible only if its combined
expanded length is at most 1 MiB. Oversized candidates are skipped while other
eligible pairs remain available; the trainer may stop before reaching the target
if the corpus has no eligible pair. The CLI uses these same defaults and
accepts `--target-vocab`, `--max-merges`, and `--min-frequency` overrides.

The production trainer begins with one local token per corpus byte. At each step it
counts adjacent pairs in the current token sequence, including overlapping pair
occurrences. It selects the most frequent pair that meets the configured minimum.
Ties are resolved by ascending `(left_token_id, right_token_id)`. It then replaces
all non-overlapping matches of that pair in a single left-to-right pass and
appends a new token ID. The assigned token ID and merge rank are both the next
sequence number starting at 256. Training stops at the merge limit, vocabulary
target, or when no pair meets the minimum.

This makes a repeated run over the same concatenated raw bytes and config produce
the same merge table. Corpus path and provenance are descriptive metadata, however,
so moving the same bytes to a different path can change artifact bytes while
leaving the model and encoded IDs identical.

## Reference oracle and runtime encoding

`packtok-train::reference` provides a deliberately slow oracle. It counts pairs by
linear search rather than using the production `BTreeMap` and uses a separate
replacement loop. Deterministic generated small corpora compare both trainers.
Runtime encoder output is compared with the reference encoder on ASCII, German,
Unicode, whitespace, code-like, empty, repeated, and arbitrary-byte fixtures.

The runtime does not retrain. It starts with base byte IDs and visits artifact
merges in increasing rank, replacing every non-overlapping occurrence of each pair
from left to right. This is the same ordering the trainer records. It leaves bytes
as base tokens wherever no learned merge applies. Decoding walks merge parents
iteratively and appends bytes in order; it does not need to allocate a textual
string or recursively call the stack.

The optimized encoder indexes pairs by their left ID and queues matching
adjacencies by `(merge rank, original byte position)`. It checks stale events and
updates only the two neighbors of a merged pair. New adjacencies include the new
result ID and can therefore only reference a later merge rank. This preserves the
reference ordering, including left-to-right replacement of overlapping pairs.
Models with at most eight merges, uniform byte runs and detected dense repetitions
use contiguous in-place rank scans to avoid queue overhead. The repetition detector
checks periods of 1–16 bytes on inputs of at least 96 bytes, using three aligned
samples to select a candidate and checking it across the full input with a bounded
number of exceptional blocks. It compares batches aligned to each period. This
heuristic selects the implementation path, never the encoded IDs.
Inputs with no matching initial byte pair
take a direct byte-token path.

## Corpus inputs and provenance

The checked-in fixture is [`fixtures/m1_bpe_corpus.txt`](fixtures/m1_bpe_corpus.txt).
The CLI accepts one local file or a directory tree; it does not download data.
For a directory, regular files are collected with an iterative tree traversal, then sorted by
root-relative Unicode path using `/` separators. Symlinks and non-Unicode paths
are rejected. Single-file training records the supplied path. Files are read as
raw bytes with no UTF-8 validation and concatenated in recorded order **without
inserted separators**. A file boundary may therefore form a pair across the two
adjacent byte sequences. No corpus transformation is applied.
Non-regular entries are rejected rather than silently omitted. Paths are built
from their components, preserving literal backslashes in Unix filenames. File
contents are appended through a 64 KiB read buffer; the complete concatenated
corpus is still retained in memory. The file-count limit is enforced during
traversal.

Each trained artifact records the config, byte and merge counts, ordering rules,
input path, ordered file list, concatenated byte count, and an FNV-1a 64-bit
checksum. The checksum is deterministic but non-cryptographic. Provenance does
not affect runtime correctness. The loader bounds the recorded input list to
65,000 files so the metadata remains below the format's collection cap.

Example from the repository root:

```text
cargo run --release -p packtok-cli -- train-bpe fixtures/m1_bpe_corpus.txt target/m1.packtok
cargo run --release -p packtok-cli -- validate target/m1.packtok
cargo run --release -p packtok-cli -- inspect target/m1.packtok
cargo run --release -p packtok-cli -- encode --artifact target/m1.packtok "Sämtliche Häuser"
cargo run --release -p packtok-cli -- inspect-token target/m1.packtok 256
cargo run --release -p packtok-cli -- inspect-merges target/m1.packtok 20
```

`decode --artifact <path> <local IDs...>` accepts space-separated flat local IDs.
Both M0 and M1 CLI decoding write the exact raw bytes to stdout, including invalid
UTF-8, with no added newline. Use `inspect-token` for escaped textual inspection.
CLI arguments must be Unicode; malformed native arguments return a clear error
and exit code 1 rather than panicking. Raw bytes inside corpus files remain valid.
`inspect-token` prints decimal bytes, an escaped byte form, UTF-8 display when
valid, and merge rank/parents for learned IDs. `inspect-merges` shows the first
50 ranks by default; pass a limit to change that. The training CLI creates a new
artifact and refuses to overwrite an existing output path. It synchronizes a
completed write and removes its newly created file if writing or synchronization
fails, so a retry is possible. Cleanup failures are reported explicitly.

## Artifact and crate boundaries

M1 uses artifact format version 2; see [FORMAT.md](FORMAT.md) for the exact wire
records and version-1 compatibility. Merge records and the 256-byte base
vocabulary are normative. Provenance metadata is descriptive. Artifact validation
rejects malformed merge ranks, forward or missing parent IDs, duplicate pairs,
invalid expansion lengths, tokens expanding beyond 1 MiB, unexpected pack layouts,
and trailing bytes. The expansion bound also applies to directly constructed models.

- `packtok-core`: existing pack and token contracts; unchanged for M1.
- `packtok-format`: version-2 merge data and validation; no training dependency.
- `packtok-tokenizer`: byte fallback plus immutable BPE runtime; does not depend
  on `packtok-train`.
- `packtok-train`: corpus loader, BPE trainer, provenance, and test oracle.
- `packtok-cli`: thin orchestration and inspection commands.
- `packtok-bench`: compares the unchanged M0 byte-only path with M1 on the same
  four held-out evaluation documents after training only on
  `fixtures/benchmark/train.txt`. It retains the previous scan encoder for matched
  comparisons and reports separate stress cases, vector capacity counters,
  sequence-length distributions, and OS process peak memory.

## Complexity and known limitations

Let `N` be the current corpus token count, `M` the number of accepted merges, and
`U` the number of distinct adjacent pairs at one training step. The reference
trainer is intentionally slow: linear-search pair counting can take
`O(N * U)` per merge. Production training rebuilds a `BTreeMap` of pair counts and
rewrites the current sequence on every merge, so its upper-bound work is
`O(M * (N log U + N))` and pair-count storage is `O(U)`. It retains the current
symbol sequence, compacted in place (`O(N)`), the merge table
(`O(M)`), and per-ID lengths to exclude oversized candidates (`O(V)`). The
implementation has not been optimized for large corpora.

Runtime initialization sorts the pair index in `O(M log M)` time and stores
`O(V + M)` index data. The event encoder does `O(N log V + N log N)` upper-bound
work per input, with `O(N)` linked symbols, pending events and output storage.
There are at most `N - 1` successful merges and two neighbor lookups per merge.
This uses more temporary capacity per input byte than the previous scan encoder;
the measured trade-off is recorded in [M1_PERFORMANCE_AUDIT.md](M1_PERFORMANCE_AUDIT.md).
The contiguous path retains the `O(M * N)` bound but avoids a replacement buffer
per rank. Inputs with no initial match require `O(N log V)` lookup work and one
output allocation. The validated model also stores one expanded
byte length per vocabulary ID for safe decoder reservation (`O(V)`, where `V` is
vocabulary size). Decoding uses an explicit parent stack bounded by merge depth
plus the caller's output buffer.

The trainer keeps the full corpus symbol sequence in memory. The benchmark reports
a small-fixture training observation, not a large-corpus capacity claim. No
incremental training pair-frequency structure, training priority queue, pretokens,
streaming training, BPE dropout, special tokens, normalization, or POS/language/morphology
packs are implemented. Any optimization should first be justified by profiling
and must continue to match the reference oracle.

## Verification and benchmark

The trainer differential test runs 128 cases. It starts a wrapping 32-bit
linear-congruential generator at `0x9e3779b9` (multiplier `1,664,525`, increment
`1,013,904,223`), uses length `(case * 17) % 73`, and emits bytes from `a` through
`e`; its config is target 270, max 14 merges, minimum frequency 1. The runtime
differential test uses a model trained from
`"abababab Sämtliche Häuser äöüß 👩🏽‍💻 e\u{0301} 漢字\n"` with target 260,
max 4 merges, and minimum frequency 2. It then runs 128
generated inputs from seed `0xa341316c`, the same generator recurrence, lengths
`(case * 13) % 97`, and bytes `a` through `g`. Both compare with the independent
reference path and assert byte-exact decoding. Fixed runtime cases cover ASCII,
German characters, emoji, combining Unicode, CJK, source-like code, whitespace,
empty and one-byte input, repeated bytes, NUL, and invalid UTF-8 bytes.

The original M1 verification results are recorded in
[M1_BPE_BENCHMARK.md](M1_BPE_BENCHMARK.md). M0's historical byte-only run remains
unchanged in [M0_BENCHMARK_BASELINE.md](M0_BENCHMARK_BASELINE.md).
The subsequent correctness/performance audit and repeated measurements are in
[M1_PERFORMANCE_AUDIT.md](M1_PERFORMANCE_AUDIT.md). The current harness warms each
operation for 25 ms, then measures it for 200 ms. Optional runtime measurement
methods count vector allocation/reallocation requests and simultaneous vector
capacity; these exclude model/input storage, allocator overhead and process RSS.
Current held-out results and PR review regressions are recorded in
[M1_REVIEW_FIXES.md](M1_REVIEW_FIXES.md). Corpus source, split and record boundaries
are defined in [fixtures/benchmark/README.md](fixtures/benchmark/README.md).
The OS high-water counter covers the whole benchmark process, including training
and all tokenizers, and cannot attribute memory to an individual operation.
Windows uses a hidden PowerShell helper to read `PeakWorkingSet64`; Linux reads
`VmHWM` from `/proc/self/status`. Other platforms report the counter unavailable.
This helper is confined to measurement; tokenizer runtime and training do not use it.
The subsequent audit of that fixed branch, including newly discovered regressions,
is recorded in [POST_FIX_AUDIT.md](POST_FIX_AUDIT.md).

## Next milestone

M2 should add the first deterministic factorized pack representation and compare
it with this flat BPE control on matched inputs, preserving this artifact and
benchmark as the baseline.
