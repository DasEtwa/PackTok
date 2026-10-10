# M6 research findings: what the M5 pilot says

## Scope and sources

This is a reanalysis of the committed exploratory M5 A/C pilot, not a new training run. Primary numeric sources are [PILOT_RESULT.json](../m5-gpu/provenance/m5-a-c-pilot-20261009T154700Z/PILOT_RESULT.json), its human-readable report and hashes. M5’s existing test split was scored at the final step; those test values are already exposed and are not a fresh independent holdout for any revised M6 experiment. The new tokenizer analysis uses only the frozen M5 training raw bytes, training sequences, tokenizer mappings and corpus manifest, identified in the adjacent machine-readable JSON.

A and C each used the same 14,681,984-parameter Transformer, the same initialization hash, seed 20261008, FP32, and 4,096,000 target positions. This is matched token-position/update regime T; raw-byte exposure and measured GPU time were not matched.

## Main result

| Measure | A: Flat BPE | C: PackTok | C − A |
|---|---:|---:|---:|
| Validation BPB | 2.572964850 | 2.597469229 | +0.024504380 |
| Final-only exploratory test BPB | 2.579990970 | 2.605001505 | +0.025010535 |
| Sampled training raw-byte exposure | 7,575,564 | 6,946,291 | −629,273 (−8.31%) |
| Actual synchronized training seconds | 212.249 | 196.922 | −15.327 |
| Tokens/s | 19,298 | 20,800 | +7.78% |
| Represented raw bytes/s | 35,692 | 35,274 | −1.17% |

The validation BPB difference is modest and descriptive of one paired seed. It does not establish a general ranking or statistical effect. The test split was scored once by the frozen runner; because it has been inspected, later work must use a new holdout for independent final evaluation.

## Learning curve and byte exposure

| Update | A BPB | C BPB | C − A |
|---:|---:|---:|---:|
| 1 | 4.825598 | 5.096670 | +0.271073 |
| 500 | 3.163909 | 3.132276 | −0.031633 |
| 1,000 | 2.804701 | 2.862802 | +0.058101 |
| 1,500 | 2.641904 | 2.708066 | +0.066162 |
| 2,000 | 2.572965 | 2.597469 | +0.024504 |

The gap changes sign and is not monotonic. A leads at the first checkpoint, C narrowly leads at 500, and A leads at 1,000–2,000. Between 1,500 and 2,000 the gap contracts by about 63%; that is a description of these checkpoints, not evidence of a stable convergence-rate law. Both curves continue improving at 2,000. Extrapolating the ranking beyond 2,000 is unsupported.

At equal target positions, C saw 8.31% fewer sampled training bytes. Since T samples token windows, the byte totals are sums of sampled target spans with replacement, not unique corpus coverage or epochs. Fewer bytes could contribute to the quality gap, but the pilot does not isolate this mechanism. C also trains faster per target position; its higher token throughput does not translate to higher represented bytes/s.

## Domain behavior

Validation BPB, with C − A:

| Domain | A | C | Difference |
|---|---:|---:|---:|
| English literature | 2.813913 | 2.850866 | +0.036953 |
| German prose | 3.029458 | 3.089193 | +0.059735 |
| Rust source | 2.687568 | 2.689860 | +0.002292 |
| Synthetic JSON | 1.290185 | 1.256581 | −0.033604 |
| Synthetic Unicode | 1.092557 | 1.078071 | −0.014486 |

This pattern is consistent with a possible domain interaction: C is better on the two synthetic structured/Unicode validation domains, near tied on Rust and worse on prose. It is not a confirmed domain effect: one seed, small and uneven domain byte counts, and synthetic domains are especially narrow. A tokenizer-feature association must be measured on the actual training inputs and tested on independent data before assigning mechanism.

## Frozen-input tokenizer analysis

Run the adjacent CPU script against the already present M5 directory to reproduce [the JSON summary](analysis/m5-tokenizer-efficiency.json). It verifies the train-corpus manifest, pilot config, corpus manifest and v5 bundle-input SHA-256 list before parsing. It does not read validation or test sequences and does not copy or edit historical inputs. The outputs describe full training-split tokenization, not the stochastic subset actually sampled in M5.

Report these separately for A and C: tokens/raw byte, raw bytes/token, byte-span length quantiles, pack share and vocabulary utilization, token-ID entropy and frequent IDs, byte-fallback/single-byte share, and fragmentation of Unicode words, numbers, Rust identifiers and JSON keys. A/C comparison of these measures can motivate, but cannot prove, a model-quality mechanism. Pack-local utilization is a deterministic property of the frozen training split and tokenizer, not a causal estimate.

The M5 training sampler did not preserve domain-stratified sampled raw-byte exposure, so no claim about which domains received fewer bytes during training can be recovered from aggregate logs. Full-split domain tokenization is only a proxy.

## Measured training-split tokenization

The CPU analysis parsed 34,632,012 training bytes and verified both complete PTM5SEQ streams and mappings against the historical v5 input checksum manifest. A emitted 18,716,345 tokens (0.5404 tokens/raw byte, 1.8504 raw bytes/token); C emitted 20,414,481 (0.5895 tokens/raw byte, 1.6964 bytes/token). Thus C emits 9.1% more tokens per training byte, consistent with its 8.31% lower byte exposure at equal sampled target positions.

| Training domain | A bytes/token | C bytes/token | C tokenization |
|---|---:|---:|---|
| English literature | 1.918 | 1.624 | More fragmented |
| German prose | 1.711 | 1.517 | More fragmented |
| Rust source | 1.888 | 1.815 | Slightly more fragmented |
| Synthetic JSON | 1.750 | 2.046 | Slightly fewer tokens |
| Synthetic Unicode | 1.595 | 1.593 | Nearly equal |

The distinction between input compression and model score matters: C had fewer JSON tokens per byte and nevertheless lower JSON validation BPB. A routing benefit, generator artifact, or sample noise could explain that; compression alone cannot.

The C global-ID allocation was 256 BYTE_FALLBACK, 196 FLAT, 56 NUMBER and 4 TEXT IDs. On the training split, observed-ID occupancy was 73.4%, 99.5%, 100% and 100%, respectively; emitted-token shares were 56.1%, 33.8%, 9.7% and 0.3%. A used 444/512 global IDs (86.7% observed). C's larger fallback share is observed in the frozen encoding and may indicate that the local capacity allocation/routing constrains reuse, but occupancy by itself does not prove wasted capacity: rare IDs and frequency skew matter.

The byte-fallback/single-byte share was 47.21% for A and 56.12% for C. Because the definition differs by scheme (A base-byte IDs versus C BYTE_FALLBACK pack), compare it as a scheme-level descriptive diagnostic. Token-ID entropy was 7.885 bits/token for A and 7.198 for C; this also reflects the uneven C routing and must not be interpreted as model entropy or quality.

Lexeme-span fragmentation on TRAIN also differs: for Unicode-letter runs in English, 68.0% of A versus 61.6% of C lexemes spanned multiple tokens (2.63 versus 2.42 tokens/lexeme); in German, 90.6% versus 85.2% (3.58 versus 3.34). Rust identifiers were multi-token 73.1% versus 68.7% (3.35 versus 3.17). In the synthetic JSON key field, A split 100% of 88,530 observed keys (2.67 tokens/key) and C split none (one token/key). These are byte-span overlaps for simple reproducible lexeme definitions, not semantic word segmentation. Despite these local reductions in selected lexeme fragmentation, C's overall bytes/token is lower because its fallback and other token emissions increase the total token count.

## Dataset and generalization limits

The frozen corpus-v2 totals 37,517,499 bytes, with 34,632,012 train bytes and roughly 1.46 MB each for validation and test. Training bytes comprise 13,695,842 English literature, 3,804,343 German prose, 13,180,989 Rust source, 1,945,549 synthetic JSON and 1,985,653 synthetic Unicode bytes, plus 19,636 structured bytes. Although the manifest has 1,289 Rust source files, Rust is about 38% of train bytes and English literature about 40%; document counts and byte balance differ. File count is not independent-sample count: files may share project, author, template or source lineage. Repeated stochastic sampling over this small training set makes memorization plausible at longer horizons. Before long runs, quantify exact-duplicate and near-duplicate rates and report domain/document balance.

Keep byte identity, normalization, deduplication, tokenizer artifacts and split boundaries pinned. M5 test scores are already known and cannot validate a revised tokenizer/training design. Design a new holdout from source groups absent from all training and development material; do not acquire or inspect those holdout texts during experiment design.

Candidate acquisition protocol: use individually verified public-domain texts from Project Gutenberg only after per-work and jurisdiction review, or a pinned legally cleared corpus release with documented terms and attribution. Project Gutenberg itself warns that its US copyright status does not guarantee rights elsewhere; see its [terms](https://www.gutenberg.org/policy/terms_of_use.html) and [license](https://www.gutenberg.org/policy/license). FineWeb is a possible broad web-text source under ODC-By 1.0 according to its [official dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb), but its scale is unnecessary here and provenance/attribution duties need review. For source code, prefer a small pinned set of independently licensed repositories with license files and commit hashes; The Stack v2 documents repository-level license metadata and removal requests, so each sampled repository still needs rights/provenance handling ([dataset card](https://huggingface.co/datasets/bigcode/the-stack-v2)). These are candidates, not a legal determination or a download instruction. No external data was downloaded for this phase.

## Identity issue in a later proposal

The historical 20k proposal’s tokenizer SHA-256 values do not match the actual v5 packaged A/C tokenizer artifacts. The M6 analysis binds to the v5 package-input manifest and records actual identities. Do not execute that proposal until its input identity is reconciled in a future authorized preparation; changing only a stale identity field must be documented and validated against the frozen artifacts. This issue does not alter the completed M5 result.

## Reproduction

From the M6 worktree, run:

```sh
python3 -m py_compile experiments/m6/scripts/analyze_m5_tokenizer_efficiency.py
python3 experiments/m6/scripts/analyze_m5_tokenizer_efficiency.py \
  --m5-root /home/dasetwa/projects/PackTok/experiments/m5-gpu \
  --output experiments/m6/analysis/m5-tokenizer-efficiency.json
```

This is local CPU analysis only. The JSON records the pilot/config/corpus/input identities; [SHA256SUMS.txt](analysis/SHA256SUMS.txt) binds the exact analysis script and JSON output.


## Five-seed replication follow-up

The five-pair T-regime replication found lower validation BPB for Flat BPE in all five seeds. Mean paired C-minus-A delta was +0.050281436 BPB (sample SD 0.043590609; paired-t 95% CI -0.003843452 to +0.104406323, n=5). The four new seeds alone averaged +0.056725700 BPB (95% CI -0.018864337 to +0.132315737, n=4). Both intervals include zero, so this small study is descriptive and does not establish statistical significance. PackTok represented 8.4185% fewer raw training bytes at equal target positions. Full data, domain summaries, TEST limitations and provenance are in [MULTI_SEED_RESULTS.md](MULTI_SEED_RESULTS.md).
