# M5 CPU corpus/tokenizer preparation evidence

Generated locally from retained manifests/logs before GPU/model outcomes. The Rust builder and trainer own all corpus/tokenizer operations; this table generation is analysis only.

## Corpus-v2

Retained raw bytes: 37517499 (35.779475 MiB).

| Category | TRAIN bytes | VALIDATION bytes | TEST bytes | Total bytes |
|---|---:|---:|---:|---:|
| code | 13180989 | 274334 | 232128 | 13687451 |
| english-drama | 5059992 | 271163 | 279281 | 5610436 |
| english-prose | 8635850 | 477110 | 477210 | 9590170 |
| german-prose | 3804343 | 213836 | 214064 | 4232243 |
| structured | 19636 | 0 | 0 | 19636 |
| synthetic-json | 1945549 | 107204 | 107110 | 2159863 |
| synthetic-unicode | 1985653 | 116069 | 115978 | 2217700 |

| Split | Bytes | Percentage of retained corpus | SHA-256 |
|---|---:|---:|---|
| train | 34632012 | 92.308957% | d84c640e29b20f2c042d2910d0504efc2d64fb195476bfb3adfc591d523210ce |
| validation | 1459716 | 3.890760% | 72b840ddcb9f471078e377dfeca85667798a27ddbc055bbe4525376e33853135 |
| test | 1425771 | 3.800283% | 62e3397fbc737fadafc236c6d3e1f25dfd73c61d66a9372a48af5450fbffd3ac |

Block outcomes: {'accepted': 5395, 'near-duplicate-jaccard80': 1, 'shared-128-byte': 149}. Every proposed block/range/hash/rejection reason and every individual Rust file hash is retained in corpus-v2-manifest.json.

## Genuine source provenance

| PG source | Title / author | License declaration | Raw bytes | Raw SHA-256 |
|---|---|---|---:|---|
| [PG 100](https://www.gutenberg.org/ebooks/100) | The Complete Works of William Shakespeare / William Shakespeare | Public domain in USA; original edition, notices preserved | 5638480 | 3cf4b3d44ee14cff4e14e78e2ad3318eff76f3f7f2afc3cee6bb925879110a37 |
| [PG 145](https://www.gutenberg.org/ebooks/145) | Middlemarch / George Eliot | Public domain in USA; original edition, notices preserved | 1865684 | dc2e0107e6ae07e8e33da934b2ec4e8e600826a2e0cd48df265119cd02e5fb50 |
| [PG 766](https://www.gutenberg.org/ebooks/766) | David Copperfield / Charles Dickens | Public domain in USA; original edition, notices preserved | 2033321 | 51650d348820a64ca4d4233da035f804819d2231801b59373db4e1279a1985e0 |
| [PG 883](https://www.gutenberg.org/ebooks/883) | Our Mutual Friend / Charles Dickens | Public domain in USA; original edition, notices preserved | 1924202 | 0ef01467f55bdb988048e6dbf3459ce3e9dd1733f84227ceb59036685d80c88e |
| [PG 1023](https://www.gutenberg.org/ebooks/1023) | Bleak House / Charles Dickens | Public domain in USA; original edition, notices preserved | 2044802 | 27e002dee487817b00ba884f3d5ffd8dbb821f82353e914cdc22bf33c413ba16 |
| [PG 7469](https://www.gutenberg.org/ebooks/7469) | Daniel Deronda / George Eliot | Public domain in USA; original edition, notices preserved | 1821169 | 744539ff48aaf2f26c15847bc42cff7d9ed224925b2a910c95c83a01a57a27fa |
| [PG 2403](https://www.gutenberg.org/ebooks/2403) | Die Wahlverwandtschaften / Johann Wolfgang von Goethe | Public domain in USA; original edition, notices preserved | 555695 | 0b7f135824979c81b9d18e36553eb01143d76e2846133c858fe24af3796f829b |
| [PG 2404](https://www.gutenberg.org/ebooks/2404) | Italienische Reise — Band 1 / Johann Wolfgang von Goethe | Public domain in USA; original edition, notices preserved | 742616 | 8d64685ab2307d1427406edff8a008af4fc77b3c2f18d22b3856d71c2b2ba23e |
| [PG 5323](https://www.gutenberg.org/ebooks/5323) | Effi Briest / Theodor Fontane | Public domain in USA; original edition, notices preserved | 633751 | 15a93db448403b07093042b6aa251ab22474ad9c05918e69f7ead5c62c0b2e8e |
| [PG 24782](https://www.gutenberg.org/ebooks/24782) | Der Dunkelgraf / Ludwig Bechstein | Public domain in USA; original edition, notices preserved | 1090067 | 2f9a8337cc70444a76c580e780b4b81a8af69f85a9631fc82fcd261cb9a29cc2 |
| [PG 2335](https://www.gutenberg.org/ebooks/2335) | Wilhelm Meisters Lehrjahre — Band 1 / Johann Wolfgang von Goethe | Public domain in USA; original edition, notices preserved | 162301 | fbb82dde194f034dd6d0c07d1b1f63f14d3e2af396b9bb4c54450967a4832552 |
| [PG 2336](https://www.gutenberg.org/ebooks/2336) | Wilhelm Meisters Lehrjahre — Band 2 / Johann Wolfgang von Goethe | Public domain in USA; original edition, notices preserved | 164682 | b0b9d8f1ff221abfb3ee68c2a02dabbcc37b767442e36c0b7e9d93be916feb5e |
| [PG 2337](https://www.gutenberg.org/ebooks/2337) | Wilhelm Meisters Lehrjahre — Band 3 / Johann Wolfgang von Goethe | Public domain in USA; original edition, notices preserved | 142517 | c36c0c9d59b29d2935ef396f757273937ed7ea01d9608c8826ea5bfd4c44a52e |
| [PG 2338](https://www.gutenberg.org/ebooks/2338) | Wilhelm Meisters Lehrjahre — Band 4 / Johann Wolfgang von Goethe | Public domain in USA; original edition, notices preserved | 190060 | 760d68bdc15550028a11cbe544b50ee035cdd755549566578d479067d163046a |
| [PG 2339](https://www.gutenberg.org/ebooks/2339) | Wilhelm Meisters Lehrjahre — Band 5 / Johann Wolfgang von Goethe | Public domain in USA; original edition, notices preserved | 179928 | 2c3a32a5fb55342f7544c99809ad81be6328794039c90abc57a1a70398501c5a |
| [PG 2340](https://www.gutenberg.org/ebooks/2340) | Wilhelm Meisters Lehrjahre — Band 6 / Johann Wolfgang von Goethe | Public domain in USA; original edition, notices preserved | 158776 | e7df58ff77947ea002ed33b4b97b1647a97d72a6c37b8b6f74b812ab40e909c7 |
| [PG 2341](https://www.gutenberg.org/ebooks/2341) | Wilhelm Meisters Lehrjahre — Band 7 / Johann Wolfgang von Goethe | Public domain in USA; original edition, notices preserved | 186728 | ed3ba096dc756948315e2eb69ee44c2810c81f24dfea8acbcc5959b50e450c39 |
| [PG 2342](https://www.gutenberg.org/ebooks/2342) | Wilhelm Meisters Lehrjahre — Band 8 / Johann Wolfgang von Goethe | Public domain in USA; original edition, notices preserved | 263463 | be1525d5035170d37b86a384dac3d925baf5a822cae28cb6464195b8ed4fe53d |

Raw text endpoints: https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt. Body byte offsets and per-split contributions are in the manifest. Complete removed PG headers/footers remain under provenance/pg-notices/.

Rust source: https://github.com/rust-lang/rust/tree/4d91de4e48198da2e33413efdcd9cd2cc0c46688/library (Rust 1.85.0), MIT OR Apache-2.0. Exact archive URL is in scripts/acquire.sh; its full byte size/checksum is in source-sizes.txt/source-SHA256SUMS.txt. Only UTF-8 .rs/.toml/.json library files plus root Cargo.toml are selected, in canonical path order, whole files up to a 20 MiB cap. Both upstream licenses are retained. Individual file hashes/ranges/contributions are in the corpus manifest.

Synthetic JSON and Unicode: authored by the deterministic Rust builder, PackTok Apache-2.0; 16384 records each. Their complete generated-source sizes/hashes and contributions are in the manifest. They are not represented as sourced prose. Numeric fields and unique markers use the fixed formula in src/corpus.rs; no network or Python generator is required.

## Tokenizer/encoded data

| Artifact | File bytes | SHA-256 |
|---|---:|---|
| A-0.packtok | 3985 | 22d994891a5d0c3697b6bc4eb0259398303cab5cb9375ead9b24dde6804090ec |
| A-1.packtok | 3985 | 22d994891a5d0c3697b6bc4eb0259398303cab5cb9375ead9b24dde6804090ec |
| A.mapping | 3084 | d2801ed4c0a0b29c4480a068cf01f5ee810d891b4a368a92a41cac873fe0a96f |
| C-0.packtok | 4552 | 334b8ce5e8868223e94b7992d2db705f3db09f989319488439fa652fe60cd1cf |
| C-1.packtok | 4552 | 334b8ce5e8868223e94b7992d2db705f3db09f989319488439fa652fe60cd1cf |
| C.mapping | 3084 | 255fc797712640ee38a3cd76bd2c7c0b1d6181a5c2f53c70c4b88b863fd96c4f |

Training configuration: 512 total logical slots, 256 shared/raw bytes plus up to 256 learned tokens, minimum overlapping pair frequency 2, deterministic tie-breaking from frozen M1/M2 implementations. No tokenizer sees VALIDATION or TEST. M1 realized 256 merges. M2 realized TEXT 196, NUMBER 4, STRUCTURE 56, BYTE 256, total 512. This allocation is corpus-dependent metadata, not linguistic truth.

| Variant | Split | Raw bytes | Tokens | Bytes/token | Encode/decode/map wall seconds | Sequence SHA-256 |
|---|---|---:|---:|---:|---:|---|
| A | train | 34632012 | 18716345 | 1.850361916 | 15.775040 | a7fdad9cfd7a4f448d1aa1e41c241bf09050cd73938594487fee35a8a274f82d |
| A | validation | 1459716 | 799653 | 1.825436783 | 0.417757 | 34a99f414c346cffad3fa818c4911f9399384be1fcb1f37b772e0ffa1da651b3 |
| A | test | 1425771 | 783315 | 1.820175791 | 0.417122 | d82168e96a56d59cb85b521da503f3a19570ee2c072d67ebdcdf743417df1ccb |
| C | train | 34632012 | 20414481 | 1.696443422 | 11.666149 | f558ecf674382cd7aa69980560b56718a7be69d917a1396405fa208a3b27939b |
| C | validation | 1459716 | 883730 | 1.651766942 | 0.480401 | 3d995cc1d32e8686615a4b4c400918c8ec36899dbc07ebdea4f37a0a5e753245 |
| C | test | 1425771 | 867165 | 1.644174984 | 0.485925 | 2f2eea0814faec257f0a494deaf795de82ea7c2178c0e52972ae860cb3a90ff7 |

The timing includes tokenizer encoding, exact decoding, mapping/bijection checks and byte-length construction. It is not pure encoding throughput. File writing/hashing follows the timed boundary. Individual repetition timings are in preparation-v2-metrics.json and tokenizer-prepare-3.txt. Both repetitions are byte-identical for each tokenizer. Mapping verification exhaustively checks every global row plus every encoded token.

## Schedule and CPU checks

Frozen proposed primary schedule: configs/primary.json; exact T/B target work and B raw-byte overshoot: provenance/schedule-plan.json. These are GPU work plans, not measured quality outcomes.

Root MSRV debug: 135 passed; raw per-target/doc-test totals retained in verification/root-debug-3.txt.
Root MSRV release: 135 passed; raw per-target/doc-test totals retained in verification/root-release-3.txt.
M5 CPU debug: 18 passed; raw per-target/doc-test totals retained in verification/transformer-debug-9.txt.
M5 CPU release: 18 passed; raw per-target/doc-test totals retained in verification/transformer-release-9.txt.

M3/M4 compatibility: 72 model artifacts and 6 tokenizer artifacts retain byte-exact read/write identity; 12 M4 tiny A/D versus M3 pairs remain byte-identical. Mapping/tokenizer SHA checks pass. The old pre-fix historical-hashes.txt additionally contains two Windows CRLF text hashes and the original PRE_GPU review hash. Those three text entries differ in this Linux/post-fix checkout; CRLF reconstruction matches the two archived text logs, and the archived PRE_GPU_CODE_REVIEW-before.md matches the old review hash. Each current text file matches the required CPU-freeze Git blob. No model/tokenizer hash changed. See verification/historical-text-differences-explained.txt.
