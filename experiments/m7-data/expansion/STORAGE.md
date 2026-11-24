# M7 token stream storage audit

The audit reconciles all six streams per corpus to their exact file lengths. It counts payload, overlap, padding, flags and eight-byte stream headers separately. The two corpus inputs are the archived literature rehearsal and the synthetic 100 MB rehearsal from M7. [Detailed measurements](storage-breakdown.json).

| Contribution | Literature seed, both tokenizers | Synthetic 100 MB, both tokenizers |
|---|---:|---:|
| Logical token IDs, u32 | 26,574,028 B | 462,979,636 B |
| Logical represented-byte lengths, u32 | 26,574,028 B | 462,979,636 B |
| Repeated one-token context | 6,642,480 B | 115,722,664 B |
| Window counts and start flags | 830,502 B | 14,469,333 B |
| Padding for partial windows | 5,608 B | 110,040 B |
| File headers | 48 B | 48 B |
| **Total** | **60,626,694 B** | **1,056,261,357 B** |

The 73-byte `LIVWIN02` record stores a one-byte header and nine pairs of four-byte values. A full record predicts eight targets. Streaming both tokenizers stores two complete representations. The two u32 fields account for about 88% of the bytes across the six streams. One-token overlap accounts for about 11%. Padding is below 0.02% in both samples.

The largest token represented eight bytes in the literature seed and six in the synthetic rehearsal. The vocabulary target is 512 IDs. For these exact samples, u16 IDs plus u8 lengths would use 19,930,521 B of logical payload for the literature seed and 347,234,727 B for the synthetic rehearsal. Deriving lengths from the verified tokenizer/mapping and storing u16 IDs alone would use 13,287,014 B and 231,489,818 B, respectively. Both estimates exclude chunk boundaries, indices and framing. A new indexed layout would need to demonstrate exact round-trip reconstruction, unchanged sampled windows and measured random-access overhead.

A separate `gzip -1` pass on each seed stream produced 13,816,399 B in total, about 77.2% smaller. This is an archive measurement. Streaming gzip changes random-access cost. No on-disk format or training semantics changed in this audit. [Compression measurements](storage-gzip-seed.txt).

Reproduce the field accounting after `packtok-m7 verify-tokens` using `python3 experiments/m7-data/storage-audit.py TOKEN_DIRECTORY`.

## Scratch index pilot

The 67,923,791-byte pilot contains 405 whole admitted documents from the published historical release. The selection is deterministic and caps each language and split cell at 11 MiB. The seed catalogue, input cache and published JSONL all use Ubuntu-24.04's ext4 filesystem backed by D:. Four runs used the same release binary and input in HDD, RAM, RAM, HDD order. The RAM runs used `--temp-dir /dev/shm/<owned-directory>` for the index only. [Selection and catalog hash](index-io-catalog.json).

| Run | Wall seconds | Peak RSS (KiB) | GNU time filesystem inputs | GNU time filesystem outputs |
|---|---:|---:|---:|---:|
| HDD 1 | 41.73 | 44,440 | 155,792 | 285,992 |
| RAM 1 | 32.51 | 44,352 | 0 | 138,216 |
| RAM 2 | 31.51 | 44,764 | 0 | 138,216 |
| HDD 2 | 32.99 | 44,616 | 0 | 212,536 |

The median RAM run was 2.97% faster than the warm HDD run. GNU time's filesystem outputs were about 35% lower. HDD 1 did physical reads while the other runs had zero reported filesystem inputs. The warm comparison avoids attributing the cold read cost to the index location. Directory comparisons and the exact manifest SHA-256 matched across all four runs. This is a small isolation test of filesystem placement. Actual SSD performance and the effect on the full corpus remain unmeasured. [Raw timings](index-io-timings.tsv), [manifest comparisons](index-io-verification.txt).

`packtok-m7 build CATALOG CACHE NEW_DATASET --temp-dir DIR` places `dedup.redb` in a run-specific subdirectory of the existing `DIR`. Both the staging directory and final rename use the output filesystem. Omit the option to keep the database in the staging directory. The build removes its own index on verified success and attempts the same cleanup on handled failure. A process kill or power loss can leave a run-specific directory to inspect. Shrinking the WSL virtual disk is a separate host operation. The existing `import-archives PLAN NEW_IMPORT` output path controls the location of extracted work files.

The only local SSD was below the required 20 GB free before the work began. The pilot used RAM for the index-offload diagnostic and leaves the storage default and tokenstream format as measured. [Disk placement](README.md#local-disk-bottleneck).

## Block-compressed token stream

`compact-stream.rs` implements the standalone `LIVZST01` experiment. It reads the unchanged `LIVWIN02` windows. It uses a 24-byte header and a 12-byte offset and compressed-length index entry per block. Each independent Zstandard level-3 frame carries a Zstandard content checksum. The reader bounds compressed and expanded sizes by the configured limit of 4,096 rows per block. It keeps one decoded block for random reads. The full comparison checks the entire seek table for contiguous blocks, rejects trailing bytes and compares every reconstructed 73-byte row. [Zstandard frame specification](https://www.rfc-editor.org/rfc/rfc8878), [Zstandard seekable format precedent](https://github.com/meta/zstd/blob/dev/contrib/seekable_format/zstd_seekable_compression_format.md).

The experiment compares complete 73-byte records with 28-byte records that store each ID in u16 and each represented-byte length in u8. Both encode the flag and all nine ID/length slots, including overlap and padding, and reconstruct the exact input. Narrow mode rejects a value outside its bounds. The prototype uses Ubuntu-24.04's existing libzstd 1.5.5 through its C ABI. The normal trainer still reads its original streams. Adding a stable compressed training format would require checkpoint identity and cold-cache batch-throughput measurements.

| Input | Stored row bytes | Rows per block | Size including the index | Size reduction | Warm median µs per random draw (raw → compressed) |
|---|---:|---:|---:|---:|---:|
| Literature, all six streams | 73 | 256 | 12,425,555 B | 79.50% | 1.46 → 50.80 |
| Literature, all six streams | 28 | 64 | 12,277,567 B | 79.75% | 1.46 → 14.34 |
| Literature, all six streams | 28 | 256 | 10,045,640 B | 83.43% | 1.49 → 26.85 |
| Literature, all six streams | 28 | 1,024 | 8,676,365 B | 85.69% | 1.50 → 70.33 |
| Synthetic, A and C train streams | 28 | 256 | 140,214,412 B | 82.95% | 1.68 → 29.00 |

The literature input was 60,626,694 bytes and the two synthetic input streams were 822,379,180 bytes. At 256 rows per block, the literature pack step took 0.62 s across its six files and the explicit full check took 0.11 s. The synthetic inputs took 5.70 s to pack and 1.57 s to verify. All 26 configurations passed byte-exact whole-stream verification and 20,000 seeded random window comparisons each. Reported random-access medians are medians across streams of mean per-draw times. Both paths preallocate their 20,000 result slots and compare after timing. Sequential verification warms the page cache, so this measures software overhead for one reader rather than physical disk or GPU batch throughput. PackTok's HDD/SSD and CUDA work overlapped early repetitions on this machine. The reviewed and preliminary runs produced identical compressed files. [Reviewed measurements](zstd-results.tsv), [preliminary measurements](zstd-preliminary.tsv).

Build and run the optional Linux probe from the repository. The standalone source requires the system `libzstd.so.1` runtime and Rust 1.100.0. It creates a new output file, and a failure can leave an experimental partial file in that output path.

```sh
rustc +1.100.0 --edition 2024 -D warnings -O experiments/m7-data/compact-stream.rs -o /tmp/compact-stream
/tmp/compact-stream "$input.windows" "$output.livzst" narrow 256
```

The independent Gemini 4 Hydrogen Medium review confirmed the narrow reconstruction and FFI buffer bounds. Its timing finding led to symmetric preallocation and comparison outside the measured intervals. The benchmark keeps full-stream validation before random access. The focused tests reject narrowing overflow, truncated containers and incorrect block checksums.
