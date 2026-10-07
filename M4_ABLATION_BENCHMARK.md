# M4 — Frozen protocol and benchmark record

The preregistered design is [M4_ABLATION.md](M4_ABLATION.md). This record preserves
sources, failures and all outcomes; results are appended after frozen final runs.

## Provenance and source acquisition

- Stacked base: audit commit `9a31271` on m3-tiny-model; its parent is historical
  M3 documentation commit c6a4667. M0–M3 artifacts/results are preserved.
- Canonical BRAIN read at commit d60a14d651fb25dfbcdd4ce7cd1175b9fb8bbbef.
  All four documents are retained under experiments/m4-ablation/provenance/.
- Raw sources acquired on 2026-10-07 before training/evaluation. No huge dataset
  download, scraping, external annotations or Python pipeline is used.
- Initial `/ebooks/1342.txt.utf-8` and `/ebooks/2403.txt.utf-8` URLs redirected to
  HTTP and PowerShell rejected the insecure redirect. No bytes were accepted.
  HTTPS cache endpoints below supplied the same preselected works; no corpus
  replacement was made. The changed acquisition endpoint is recorded here.

| Retained source | URL/provenance | License/public-domain scope |
|---|---|---|
| austen.txt | https://www.gutenberg.org/cache/epub/1342/pg1342.txt | Original English Austen (died 1817); PG declares public domain USA. Full PG notice/license retained; source updated 2026-09-01. |
| goethe.txt | https://www.gutenberg.org/cache/epub/2403/pg2403.txt | Original German Goethe (died 1832); PG declares public domain USA. Full PG notice/license retained; source updated 2020-04-17. |
| rust-vec.rs | https://raw.githubusercontent.com/rust-lang/rust/4d91de4e48198da2e33413efdcd9cd2cc0c46688/library/alloc/src/vec/mod.rs | Rust 1.85.0, MIT OR Apache-2.0; both license files retained. |
| rust-map.rs | Same pinned revision, library/alloc/src/collections/btree/map.rs | MIT OR Apache-2.0 |
| rust-Cargo.toml | Same pinned revision, Cargo.toml | MIT OR Apache-2.0 |
| synthetic-json.txt | Rust m4-corpus generator, 2000 numbered records | Newly authored PackTok Apache-2.0 synthetic coverage |
| synthetic-unicode.txt | Same generator, 2000 numbered records | Newly authored PackTok Apache-2.0 synthetic coverage |

PG's public-domain declaration is US-specific; these original texts also have
long-deceased authors, with no modern translator used. Retained source notices
apply to redistribution of the downloaded PG editions. They are not relicensed
under the repository's Apache license. Illustrations are not downloaded.
Exact source byte sizes, SHA-256, extracted contributions, split proportions and
removed leakage are preserved in corpus-v2/composition.txt and SHA256SUMS.txt.

## Preflight and limitations

The selected hidden size stays 16 without an outcome-driven architecture search.
The same schedule uses 2000 updates (tiny compatibility 120), three seeds and
512 slots; settings and all metric/timing definitions are in M4_ABLATION.md.
Optional 1024-slot and semantic grouping experiments are omitted to contain CPU
runtime. Windows and Linux verification logs, intermediate failures and exact
final source commits are retained under experiments/m4-ablation/.

Corpus preflight v1 used 8192-byte blocks, yielding no held-out Rust code because
both source files had fewer than twenty blocks. Its full corpus and composition
are retained in `experiments/m4-ablation/corpus/`, excluded from model runs.
Before any model outcome was evaluated, preparation v2 reduced block size to
4096 bytes to cover held-out code. Final inputs are `corpus-v2/`; the source
selection, assignment modulo20, leakage rule and all model settings stayed fixed.
Initial strict Clippy caught a test-only import and indexed loop; both were fixed.
The initial editing script temporarily added field visibility to function
arguments; this was corrected before compilation. All compiled preflight logs
remain under verification/; there was no failed or discarded model seed.
Preparation v2 retained 1,857,089 TRAIN bytes, 82,984 validation bytes and 86,816
TEST bytes; 8,252/4,148 validation/test bytes were dropped by the predeclared
128-byte leakage guard. TEST includes 4,136 Rust source bytes. Validation has no
Rust code after leakage filtering, a known domain-composition mismatch; the
corpus was not further adjusted. Exact per-source contributions and input hashes
are in corpus-v2/composition.txt and input-SHA256SUMS.txt. Primary data total is
2,026,889 bytes before model scoring's first-token exclusion.
Encoder timers include a single full encode and output allocation, exclude byte-
length lookup, differential encode, decoding and mapping; no throughput median
or warm-up claim is made. Timings are one observation per split/tokenizer.
Raw corpus sources retain original CRLF and trailing whitespace deliberately;
Git whitespace checks exclude those data files via -diff, not normalization.
The initial inline WSL launch failed on inherited PATH quoting, before Cargo.
Its cause and the corrected explicit script are preserved under experiments.
