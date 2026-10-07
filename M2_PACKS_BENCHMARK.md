# M2 factorized packs — benchmark record

## Comparison and protocol

This report compares frozen M1 flat byte-level BPE and M2 on the same held-out
files and Windows machine. It measures tokenizer representation/runtime only;
it does not measure model loss or language-model inference. Results do not
establish that PackTok is better than M1.

Both configs request 512 logical slots: 256 byte tokens, at most 256 learned
tokens, and minimum pair frequency 2. Both realized all 256 learned slots.
M1 has one flat learned vocabulary. M2 shares the byte base and allocates those
learned IDs globally among lexical-v1 packs.

- Source commit: `57b9045f3588639dadd25123abf421dc791bb2c3`, branch
  `m2-factorized-packs`. This is the measured implementation commit; this report
  and its final raw logs were added afterward.
- Commands: `cargo build --release --workspace`, then three invocations of
  `target/release/packtok-bench.exe` from the repository root.
- Profile: default optimized Cargo release, no custom RUSTFLAGS.
- Machine: Intel Core i7-2600 at 3.40 GHz, 4 cores / 8 logical processors;
  no CPU affinity or frequency locking.
- OS: Windows 10 Pro 10.0.19045, build 19045.
- Toolchain: rustc 1.98.1 (`48a229cea`, LLVM 22.1.8), Cargo 1.98.1,
  `x86_64-pc-windows-msvc`.
- Protocol: 25 ms warm-up and at least 200 ms timed loop per operation.
  Each held-out file is repeated 32 times for throughput. Allocations are
  included in timing; capacity instrumentation runs separately. Throughput is
  the median of three final repetitions. No confidence intervals were computed.
- Preprocessing: raw UTF-8 bytes, no normalization. The harness checks for
  copied 32-byte passages between training and evaluation files. Directory
  inputs are sorted by path and concatenated without inserted separators.

Training source: `fixtures/benchmark/train.txt`, 1,640 bytes, one file,
FNV-1a 64-bit `4d0cc0a94cc66e49`, SHA-256
`4ec5a71fe266e073f45e705cca42f037f91000919d60331f809031601058d2b3`.
Evaluation source fingerprints:

| Class | File | Bytes | FNV-1a 64-bit | SHA-256 |
| --- | --- | ---: | --- | --- |
| English | `fixtures/benchmark/eval/english.txt` | 229 | `b00b21654654d1bc` | `9683e3407ee8e3fce95be32487c3bf575e0ece4ef02dbd3cdad92128faafd1d2` |
| German | `fixtures/benchmark/eval/german.txt` | 247 | `7767115b6b9b4e70` | `07927e688482c7d5f193e5e28599ce88a81a2a98e592332050a2c0945d16abe4` |
| Unicode-heavy | `fixtures/benchmark/eval/unicode.txt` | 223 | `97f4d1f3d4c1a47f` | `3316f7cb8f969f8d66cdf38ac0a16df3724a1b93026aa5aae83264596cb068be` |
| Source-like | `fixtures/benchmark/eval/code.txt` | 255 | `a3e31d9692c8cd3f` | `2f49a08d1a22f219cbcef707b49b6c37791fc2d469734372358a377b74052f31` |

The fixtures are manually authored and synthetic. Files define the train/eval
split; the passage check is an additional leakage guard. This is not a
representative corpus.

## Artifacts and training allocation

These historical measurements precede the span-sized workspace fix. The
[current audit](PERFORMANCE_MATH_AUDIT.md) records its capacity and timing
effects separately; the tokenizer IDs and artifact formats remain unchanged.

M2 release training was run twice and complete serialized byte arrays were
compared. They were identical, 4,492 bytes, SHA-256
`debc3f36e3a8baff1f7599a3dddad498250620bda7b1230d663f9668e3e6d065`.
Config: v3, lexical-v1, target 512, maximum 256 learned tokens, minimum
frequency 2.

M1 was also trained twice on the same benchmark training file. Artifacts were
identical, 3,951 bytes, SHA-256
`4b3a29b90d468be1f233f8c2af4bd39d5f74c7a679d17ebc5f3ad8b587714ca9`.
The historical M1 fixture still produces 84 merges, 340 IDs, 1,881 bytes, and
SHA-256 `CEF973A354422C88E6FEF54D0B6EEE09495980DF4E5BD0B7E5ADC37896E7D907`,
matching its preserved report. Version 1 and version 2 tests also assert
canonical parse/serialize bytes are unchanged. The historical M1 reports remain
unchanged; the in-run M1 columns below are this matched comparison, not a
replacement of the baseline record.

| Training namespace | Pack ID | Spans | Routed bytes | Learned merges/local IDs |
| --- | ---: | ---: | ---: | ---: |
| Shared BYTE_FALLBACK | 65535 | 0 | 0 | 0 |
| TEXT | 0 | 206 | 1,278 | 243 |
| NUMBER | 1 | 14 | 40 | 3 |
| STRUCTURE | 2 | 212 | 322 | 10 |
| **Total learned** | — | **432** | **1,640** | **256** |

The byte IDs occur once in BYTE_FALLBACK. Realized logical vocabulary size is
512; specialized packs do not duplicate byte IDs.

## Held-out results

Token counts and bytes/token are per original evaluation file. Timed inputs
repeat each file 32 times; raw repeated-input token counts are divided by 32.
Throughput medians are MiB/s. Sequence change is `(M2 / M1 - 1)`; positive
values mean M2 emitted more tokens.

| Evaluation | Bytes | M1 tokens | M2 tokens | M1 B/token | M2 B/token | Sequence change | M1 encode | M2 encode | M1 decode | M2 decode | Artifact M1/M2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| English prose | 229 | 159 | 170 | 1.440 | 1.347 | +6.92% | 19.30 | 2.54 | 251.90 | 148.05 | 3,951 / 4,492 B |
| German prose | 247 | 166 | 185 | 1.488 | 1.335 | +11.45% | 17.34 | 2.45 | 247.22 | 155.00 | 3,951 / 4,492 B |
| Unicode-heavy text | 223 | 210 | 201 | 1.062 | 1.109 | -4.29% | 55.80 | 2.08 | 290.71 | 213.83 | 3,951 / 4,492 B |
| Source-like text | 255 | 189 | 204 | 1.349 | 1.250 | +7.94% | 22.40 | 3.32 | 248.34 | 161.36 | 3,951 / 4,492 B |

M2 emitted more tokens for English, German, and source-like text, and slightly
fewer for the Unicode-heavy fixture. M2 encode and decode throughput were
lower than M1 on all four classes. The M2 artifact was 541 bytes larger at the
same realized vocabulary size. These fixture results are consistent with
lexical-v1 short structure spans and its inability to learn cross-boundary
tokens such as a leading-space word. They do not establish general quality.

## Pack statistics and memory

These instrumented counts use the 32-repeat input. Output tokens are
BYTE_FALLBACK / TEXT / NUMBER / STRUCTURE. Routed bytes, spans, and mean span
bytes are TEXT / NUMBER / STRUCTURE. Learned merges stay TEXT 243, NUMBER 3,
STRUCTURE 10 across evaluation classes. Raw fallback tokens are counted in the
BYTE_FALLBACK output column even though their source bytes came from specialized
spans; the fallback row therefore has zero routed bytes.

| Class | Output tokens B/T/N/S | Routed bytes T/N/S | Spans T/N/S | Mean span bytes T/N/S | Fallback share | Pack transitions | M1 encode/decode capacity | M2 output capacity | M2 temporary span/symbol capacity |
| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| English | 3,872 / 1,440 / 0 / 128 | 6,016 / 32 / 1,280 | 1,088 / 32 / 1,120 | 5.53 / 1.00 / 1.14 | 71.18% | 2,527 | 333,824 / 7,344 B | 58,624 B | 156,928 B |
| German | 4,416 / 1,376 / 0 / 128 | 6,720 / 0 / 1,184 | 1,024 / 0 / 1,024 | 6.56 / — / 1.16 | 74.59% | 2,527 | 358,656 / 7,920 B | 63,232 B | 112,384 B |
| Unicode-heavy | 5,824 / 576 / 0 / 32 | 6,400 / 0 / 736 | 672 / 0 / 672 | 9.52 / — / 1.10 | 90.55% | 1,023 | 339,200 / 7,152 B | 57,088 B | 106,240 B |
| Source-like | 5,088 / 864 / 0 / 576 | 4,672 / 256 / 3,232 | 1,088 / 160 / 1,088 | 4.29 / 1.60 / 2.97 | 77.94% | 1,663 | 374,784 / 8,176 B | 65,280 B | 163,584 B |

M1 capacities are existing M1 encoder/decoder vector observations. M2 output
capacity is the returned token vector. M2 temporary capacity combines routed
span and working-symbol vectors. Counters exclude allocator metadata, model
storage, caller input, and process RSS; M1/M2 definitions are not identical.
M2 allocation counts and isolated per-operation RSS were not measured.

Whole-process peak working-set values were 7,303,168 B, 7,467,008 B, and
7,495,680 B (about 6.96–7.15 MiB). Each includes training, M0, M1, M2,
reference models, and stress workloads; none is a per-tokenizer value.

One-time training observations had medians of 19.857 ms for M1 and 11.553 ms
for M2 on the 1,640-byte fixture, excluding file loading. This tiny corpus and
one training operation do not establish large-corpus throughput or memory
scaling. M2 training peak memory was not separately instrumented.

## Raw observations and noise

Final repetitions on the same source commit and protocol:

- [Final run 1](experiments/factorized-packs/m2-20261007/final-1.txt)
- [Final run 2](experiments/factorized-packs/m2-20261007/final-2.txt)
- [Final run 3](experiments/factorized-packs/m2-20261007/final-3.txt)

An earlier run set on implementation commit 8b324a2, before the v3 writer-limit
regression test commit, is retained and excluded from the final medians:

- [Pre-final run 1](experiments/factorized-packs/m2-20261007/pre-final-8b324a2-1.txt)
- [Pre-final run 2](experiments/factorized-packs/m2-20261007/pre-final-8b324a2-2.txt)
- [Pre-final run 3](experiments/factorized-packs/m2-20261007/pre-final-8b324a2-3.txt)

Preflight runs, made before two semantics-preserving Clippy cleanups, are
retained and excluded from medians:

- [Preflight 1](experiments/factorized-packs/m2-20261007/preflight-1.txt)
- [Preflight 2](experiments/factorized-packs/m2-20261007/preflight-2.txt)
- [Preflight 3](experiments/factorized-packs/m2-20261007/preflight-3.txt)

Throughput varies slightly between runs. Timing windows are short, the machine
was not isolated, and no confidence interval was computed. These medians
describe this hardware and harness; they should not be compared as equivalent
to results from another system.

## Verification

- Windows 10 Pro, rustc/Cargo 1.98.1: `cargo fmt --all --check`, strict
  workspace Clippy with `-D warnings`, `cargo test --workspace` (89
  unit/integration tests passed, 0 failed; doc tests contain no tests), and
  `cargo build --release --workspace` passed.
- Linux Ubuntu 26.04.1 LTS under WSL2, kernel
  `6.18.40.1-microsoft-standard-WSL2`, rustc/Cargo 1.85.0, the declared MSRV:
  the same fmt, strict Clippy, workspace test (90 unit/integration tests
  passed, 0 failed; doc tests contain no tests), and release build passed.
- The Linux target directory was isolated at
  `/var/tmp/packtok-m2-target`; no Linux build artifacts were written into the
  Windows Cargo target directory.

## Limitations and next decision

- The corpus is synthetic and small; it does not establish results on natural
  corpora or other languages.
- TEXT receives 243 of 256 learned tokens under this corpus/configuration;
  this is not a permanent pack allocation.
- Most held-out tokens remain raw bytes. M2 is slower than M1 on every measured
  encode/decode class and was not tuned against held-out data.
- Process high-water memory is shared across benchmark components; capacity
  counters are not allocation totals.
- Training was not evaluated on a larger corpus; the trainer rescans the
  selected pack's spans when recomputing candidates.

M2 demonstrates an exact, deterministic factorized representation and a
fixed-budget comparison, not an improvement over M1. The next logical milestone
is M3: a tiny matched language-model experiment with explicit parameter and
compute budgets and held-out validation.
