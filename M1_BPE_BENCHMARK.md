# M1 flat byte-level BPE benchmark

Captured 2026-10-07. These are single-machine baseline observations, not a
cross-machine comparison and not a claim that PackTok's future factorized design
is better. M0's historical run remains untouched in
[M0_BENCHMARK_BASELINE.md](M0_BENCHMARK_BASELINE.md).

## Environment and command

- Command: `cargo run --release -p packtok-bench`, run from the repository root.
- Build profile: Cargo `release` (optimized), default workspace settings.
- Rust: `rustc 1.98.1 (48a229cea 2026-09-01)`, host `x86_64-pc-windows-msvc`,
  LLVM `22.1.8`.
- OS: Microsoft Windows 10 Pro x64, version `10.0.19045`, build `19045`.
- CPU: Intel Core i7-2600 @ 3.40 GHz, 4 cores and 8 logical processors.
- No external Rust crate dependencies are used by the workspace.
- The harness repeats each fixed sample 32 times. It measures encode and decode
  separately for 200 ms per tokenizer and direction. It has no warm-up phase;
  sample order is M0 encode/decode followed by M1 encode/decode. Each iteration
  includes the tokenizer's output-buffer allocation. There are no allocation or
  peak-memory counters.
- M1 training configuration: target vocabulary 512, maximum 256 merges, minimum
  pair frequency 2. The fixture stops at 84 merges because no further pair meets
  the minimum. The model has 256 byte tokens and 84 learned tokens, for 340 IDs.
- Training is timed once after corpus loading and before serialization; the
  observed 4.663 ms (0.15 MiB/s) includes in-memory training and artifact
  construction, excludes file loading and artifact serialization. This one-run
  time is an observation, not a stable throughput estimate.
- Corpus: `fixtures/m1_bpe_corpus.txt`, one file, 712 raw UTF-8 bytes, checksum
  `fnv1a64=3bafa3aa89541485`. No normalization or preprocessing is applied.
- The serialized M1 artifact is 1,881 bytes, including registry and provenance
  metadata.

## Same-run results

M0 and M1 use the exact same repeated text for each input class. M0 produces one
token per input byte. “Encode change” and “decode change” compare M1 throughput to
the M0 measurement from this same run. Negative throughput deltas mean the current
M1 implementation is slower for that operation on this machine.

| Input class | Bytes / M0 tokens | M1 tokens | M1 bytes/token | Sequence reduction | M0 encode MiB/s | M1 encode MiB/s | Encode change | M0 decode MiB/s | M1 decode MiB/s | Decode change | Iterations M0 enc/dec | Iterations M1 enc/dec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English prose | 4,160 | 3,264 | 1.275 | 21.54% | 864.80 | 14.08 | -98.37% | 467.71 | 217.91 | -53.41% | 43,597 / 23,579 | 711 / 10,986 |
| German prose | 4,736 | 3,136 | 1.510 | 33.78% | 1,361.21 | 15.85 | -98.84% | 473.41 | 218.47 | -53.85% | 60,276 / 20,963 | 702 / 9,675 |
| Unicode-heavy text | 4,320 | 3,488 | 1.239 | 19.26% | 1,367.85 | 16.66 | -98.78% | 471.49 | 217.99 | -53.77% | 66,403 / 22,889 | 809 / 10,583 |
| Source-like text | 5,952 | 4,288 | 1.388 | 27.96% | 1,426.91 | 14.46 | -98.99% | 469.32 | 218.35 | -53.48% | 50,277 / 16,537 | 510 / 7,694 |

Sequence reduction is `1 - (M1 tokens / M0 tokens)` for each repeated sample.
Bytes/token is `input bytes / M1 token count`. Throughput uses bytes processed by
the direction under measurement divided by elapsed MiB. The current encoder's
large encode-throughput drop is consistent with its deliberately simple ordered
full-sequence scans for each merge rank; this is the baseline to profile before
optimizing. The token-count reduction is not a language-model quality result.
The harness tracks total processed bytes and produced tokens; total bytes per
direction are the row's sample size multiplied by its displayed iteration count.

## Deterministic artifact check

The checked-in fixture was trained twice with the default configuration:

```text
cargo run --release -p packtok-cli -- train-bpe fixtures/m1_bpe_corpus.txt target/m1-final-a.packtok
cargo run --release -p packtok-cli -- train-bpe fixtures/m1_bpe_corpus.txt target/m1-final-b.packtok
```

Both artifacts were 1,881 bytes and a byte-for-byte comparison returned `True`.
Their SHA-256 digest was
`CEF973A354422C88E6FEF54D0B6EEE09495980DF4E5BD0B7E5ADC37896E7D907`.
The SHA-256 value is recorded only as a verification fingerprint; corpus
provenance inside the artifact uses the documented non-cryptographic FNV-1a
checksum.

## Verification

- `cargo fmt --all --check`: passed.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: passed.
- `cargo test --workspace`: 37 unit tests passed, 0 failed; doc-test suites had
  no tests.
- `cargo build --release --workspace`: passed.
- The CLI encoded and decoded `Sämtliche Häuser` through the trained artifact
  byte-for-byte. Artifact validation, token inspection, and merge inspection
  commands also succeeded.

## Scope and limits

The benchmark uses four short, fixed in-code samples, not a representative
production corpus. It compares current Rust implementations on one host. It does
not measure allocations, peak memory, model quality, training on a large corpus,
or PackTok factorization. M1 trades substantial throughput for fewer IDs in this
initial implementation; the results do not establish a quality or end-to-end
system advantage.
