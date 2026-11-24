# M7 verification and training observations

The input was six cached literature files from M5, totaling 11,816,998 bytes. Five documents and 6,079,503 extracted UTF-8 bytes survived. The passage detector excluded the 5,618,649-byte extracted Shakespeare anthology in validation after a matching fingerprint in an earlier split. The detected passage content has not been independently classified. Theodor Fontane was the sole remaining validation author. Ingest took 15.51 s and 19,360 KiB peak RSS. Tokenizer fitting, round-trip encoding and stream verification took 14.39 s and 17,088 KiB peak RSS on the same host. These measurements include file integrity verification and overlapped other local verification work. [Input catalog](sources-local-v1.json), [seed manifest](verification/seed-dataset.json), [build log](verification/build.log), [tokenizer log](verification/tokenize.log).

| Split | Text bytes | A tokens | A bytes/token | C tokens | C bytes/token |
|---|---:|---:|---:|---:|---:|
| Train | 2,381,908 | 1,175,355 | 2.0265 | 1,386,014 | 1.7185 |
| Validation | 613,903 | 316,307 | 1.9408 | 353,939 | 1.7345 |
| Test | 3,083,692 | 1,576,080 | 1.9566 | 1,835,812 | 1.6797 |

The shared train-only sample used 262,144 bytes including record separators. The paired 9,360-parameter CPU probe used 96 steps per variant and 768 target positions. A sampled 1,592 represented raw bytes and C 1,310. The validation score uses 508 represented bytes from the same archived Fontane prefix in each variant. The measured compression and loss both favored A on this fixture, with substantial domain and exposure limitations.

| Variant | Train probe BPB at start → end | Validation probe BPB at start → end | Targets/s | Peak process RSS |
|---|---:|---:|---:|---:|
| A Flat BPE | 6.028329 → 5.498302 | 4.888695 → 4.479186 | 2,463 | 11,128 KiB |
| C lexical-v1 | 6.146330 → 5.594362 | 5.474035 → 4.671231 | 2,649 | 11,132 KiB |

Throughput is target positions divided by wall time of the loop, including window reads, all gradient checks, updates, journaling and periodic checkpoint publication. The raw reports record initial and final NLL and exact denominators. Generation was degenerate: A produced 24 letter `e` tokens and C emitted 24 spaces. This fixture verifies byte decoding and a complete generation path. It contributes no evidence of natural text quality. [A log](verification/train-A.log), [C log](verification/train-C.log), [A trajectory](verification/seed-A-trajectory.jsonl), [C trajectory](verification/seed-C-trajectory.jsonl).

A separate process stopped at step 40. An incomplete uncommitted journal line was appended to model an interrupted write. The fresh process verified the committed prefix, archived the longer journal, and resumed to step 96. The full train trace and final checkpoint were identical to uninterrupted A. The final image SHA-256 was `e2fcb89ad7bf9fc4511ca3db3716a9e1330ebe5c2f0a95ba68220419aa3fdc61`. [Step 40](verification/split-A-40.log), [resumed step 96](verification/split-A-96.log), [archived journal](verification/cpu-uncommitted-journal.txt), [gate and hash record](verification/verification.txt).

## Synthetic scale rehearsal

The 100 MB JSONL fixture was generated in Rust with seed 20261124. It produced 1,000 input documents and 99,988,000 retained bytes. Build and verification used 30.12 s. Tokenizer training, both streams and verification used 110.01 s. Peak process RSS including generation was 31,000 KiB. Apparent size was 100,037,139 bytes for the raw data shards and manifests, and 1,056,278,863 bytes for the two token streams and their metadata. This run established bounded memory behavior for synthetic 100 MB input on this host. The 1 GB, 10 GB and 100–150 GB corpus gates still require representative licensed data, storage and measurement. [Scale log](verification/100mb.log), [scale corpus manifest](verification/100mb-dataset.json), [scale tokenizer manifest](verification/100mb-tokenizers.json).

Run the opt-in test from the Transformer workspace with a fresh path and `RAYON_NUM_THREADS=1`:

```sh
PACKTOK_M7_SCALE_OUT=/path/to/new-directory RAYON_NUM_THREADS=1 \
  cargo +1.100.0 test --offline --locked --release --bin packtok-m7 \
  tests::scale_100_mb -- --exact --ignored --nocapture
```

## Independent review and verification

| Agent via agy | Scope and disposition |
|---|---|
| Gemini 4 Neon High | Source research and architecture. Primary legal evidence was checked independently. Selected a small benchmark and on-disk exact/near/passage indexes. Article adapters and external-sort comparisons are later scale work. |
| Gemini 4 Hydrogen Medium | Proposed tests for train-only tokenizers, split grouping, malformed UTF-8, bounded records and recovery. Added train/held-out differential and corruption regressions. The real HTTPS range behavior is pending on an approved source. |
| Claude Sonnet 5.5 Medium | Confirmed gaps in space handling for phone numbers, corrupt .part handling, retry staging, short normalized duplicate coverage, startup recovery and token/sample accounting. Corrected those paths, checked copied catalog data, and verify raw inputs before and after extraction. Tokenization searches backwards for whitespace at chunk boundaries. Prefix-sampling bias and Linux durability are documented. |

Formatting, Clippy with warnings denied, 135 root tests and the isolated workspace tests passed. Isolated results include 38 Transformer unit tests, six new M7 tests and four integration test entries. The 100 MB test was run separately. The first isolated full test run lacked the local prepared-v2 data paths. The historical frozen-config regression passed after those paths were restored. The preliminary window-format rehearsal preceded reviewed improvements and is recorded as exploratory, with A validation BPB 4.566937, C 4.580709 and a bit-identical resumed A. No later parameters were tuned against it. [Verification record](verification/verification.txt).

## Prepared CUDA experiment

The local CUDA-enabled binary and portable runtime verified the frozen dataset and token streams through the bundled loader without CUDA execution. The archived package is 454,300,901 bytes, with SHA-256 and member checksums below. The local path is `~/tmp/packtok-m7-Iyu1X0Wh/m7-gpu-package.tar.gz` in Ubuntu-24.04. It includes the licensed-for-local-research seed text. Before any remote transfer, confirm the archive's intended-use permission for the selected execution environment or substitute independently approved inputs. Keep the original and tokenized payloads outside Git. [Package SHA-256](verification/gpu-package.sha256), [input identities](verification/gpu-inputs.sha256), [loader check](verification/gpu-loader.log), [execution script](gpu-probe.sh).

The proposed separate authorization is one NVIDIA L4 request, up to 20 minutes including upload and verified release, at most 0.60 CU. At the last M6 observed rate of 1.94 CU/h, a 0.60 CU envelope implies at most 1,113 seconds for the allocation lifecycle. Reserve 130 seconds of that bound for release. Runtime and CU figures are planning limits. Measure the current rate after allocation, shorten the budget if needed and release the owned alias on success or failure. Preserve transfer and exit evidence. The remote program runs control A, C, A to step 40 and restored A to 96. Same sampled window indices, RNG and learning rates are mandatory. Report any differing CUDA trajectory, finite values, aggregate loss and actual byte exposure. Do not tune selection or budgets after seeing held-out scores. No M7 GPU allocation has occurred.
