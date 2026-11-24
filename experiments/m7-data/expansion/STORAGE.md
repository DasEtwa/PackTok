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

The 73-byte `LIVWIN02` record stores a one-byte header and nine pairs of four-byte values. A full record predicts eight targets. Streaming both tokenizers stores two complete representations. Two u32 fields account for about 88% of bytes individually aggregated across the six streams. One-token overlap accounts for about 11%. Padding is below 0.02% in both samples.

The largest token represented eight bytes in the literature seed and six in the synthetic rehearsal. The vocabulary target is 512 IDs. For these exact samples, u16 IDs plus u8 lengths would use 19,930,521 B of logical payload for the literature seed and 347,234,727 B for the synthetic rehearsal. Deriving lengths from the verified tokenizer/mapping and storing u16 IDs alone would use 13,287,014 B and 231,489,818 B, respectively. Both estimates exclude chunk boundaries, indices and framing. A new indexed layout would need to demonstrate exact round-trip reconstruction, unchanged sampled windows and measured random-access overhead.

A separate `gzip -1` pass on each seed stream produced 13,816,399 B in total, about 77.2% smaller. This is an archive measurement. Streaming gzip changes random-access cost. No on-disk format or training semantics changed in this audit. [Compression measurements](storage-gzip-seed.txt).

Reproduce the field accounting after `packtok-m7 verify-tokens` using `python3 experiments/m7-data/storage-audit.py TOKEN_DIRECTORY`.
