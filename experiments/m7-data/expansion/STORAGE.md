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
