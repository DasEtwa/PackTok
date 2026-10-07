# M0 byte-fallback baseline

Captured on 2026-10-07 with:

- Command: `cargo run --release -p packtok-bench`
- Rust: `rustc 1.98.1 (48a229cea 2026-09-01)`
- Host: Windows 10 Pro x64, Intel Core i7-2600 @ 3.40 GHz
- Harness: fixed samples repeated 32 times; 200 ms per direction and sample
- Scope: byte-fallback encode/decode only; no model, corpus training, or allocation counter

| Input class | Input bytes / tokens per iteration | Encode MiB/s | Decode MiB/s |
| --- | ---: | ---: | ---: |
| English prose | 4,160 | 1,394.63 | 462.11 |
| German prose | 4,736 | 1,395.36 | 467.03 |
| Unicode-heavy text | 4,320 | 1,370.18 | 470.01 |
| Source-like text | 5,952 | 1,415.35 | 466.41 |

Each input byte produces one token. The benchmark executable also prints the total
bytes, tokens, and iterations measured for each direction. These numbers are a
single-machine M0 baseline, not a cross-platform comparison or a claim that PackTok
is faster than another tokenizer.
