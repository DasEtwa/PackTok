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

Pre-final Windows verification (Rust/Cargo 1.98.1) passed strict Clippy,
workspace build and all 124 tests; Linux WSL Ubuntu 26.04.1 on declared MSRV
Rust/Cargo 1.85.0 passed fmt, strict Clippy, release build and all 125 tests.
The extra Linux test covers literal-backslash corpus paths. Logs are retained.
No model settings changed after these checks. Frozen implementation is commit
9928c15; a233b85 adds only raw-data/verification metadata. Final experiment
environment files record the exact checkout commit used by each run.

The first tiny final attempt at cc7c2eb failed closed before any model trained:
Windows Path::join changed the spelling of TRAIN's provenance path from forward
slashes to backslashes, so descriptive tokenizer metadata differed from the M3
artifact. No tokenization or merge algorithm changed. The M4 harness now supplies
a canonical forward-slash input path, preserving the frozen artifact's complete
bytes on Windows and Linux. The failed directory `runs/tiny-final-20261007/`
and console are retained; the corrected final run uses a new label. No model
outcome was observed in the failed attempt, and no experimental setting changed.
## Tiny compatibility result

Final tiny command: `target/release/packtok-bench.exe m4 tiny final-20261007-v2`,
source commit 8ae3df4. All 24 models (four variants, two regimes, three seeds)
completed, with byte-identical retrained M3 tokenizers. Every A/D model matches
its corresponding historical M3 model SHA-256, including both regimes. Evidence:
[model identity](experiments/m4-ablation/verification/m3-historical-model-identity.txt)
and [complete tiny tables](experiments/m4-ablation/tiny-tables.md).

| Tiny regime | A bits/byte | B | C | D |
|---|---:|---:|---:|---:|
| Same schedule | 6.87520175 ± 0.14623620 | 6.51074555 ± 0.15484687 | 5.97378279 ± 0.00654171 | 6.00768927 ± 0.05498394 |
| MAC-aware | 6.87520175 ± 0.14623620 | 7.64758548 ± 0.07716365 | 5.97378279 ± 0.00654171 | 6.09533770 ± 0.18774971 |

Tiny same-schedule paired contrasts are B-A=-0.36445621, C-A=-0.90141896,
D-A=-0.86751248, interaction=+0.39836269 bits/byte. MAC-aware contrasts are
+0.77238373, -0.90141896, -0.77986405, -0.65082881 respectively. Individual
seeds and population SDs are retained in the linked tables and raw run-summary.
M2's sequence with a flat head already accounts for the tiny-corpus gain.
Synthetic output factorization alone is budget-dependent and loses at the longer
MAC-aware exposure. This is diagnostic of this tiny dataset, not the larger
corpus conclusion. No settings were adjusted in response to these results.

The shared M3 head-statistics helper prints a reserved-ID byte-fallback count
for B too: it is zero because synthetic packs never use ID 65535. This is an
irrelevant namespace diagnostic, not evidence that M1 uses no raw byte tokens.
Only D's field has the M2 byte-fallback interpretation. B's IDs are synthetic.
## Larger fixed corpus: final result

Final command: `target/release/packtok-bench.exe m4 large final-20261007`.
Measured implementation commit: `8ae3df4968279f978ffff981fa1eb38347a8dcde`.
All 24 declared seed/regime/variant runs completed. The final code adds only an
extraction of the already-used canonical path expression plus a frozen-artifact
regression; model, mapping, sampling and scoring semantics did not change after
measurement. No seed or unfavorable outcome was discarded.

Complete individual results, means/population SDs, paired differences, NLL/byte,
loss/token, validation/test times, head statistics and learning curves are in
[large-complete-tables.md](experiments/m4-ablation/large-complete-tables.md),
generated by the preserved PowerShell presentation script from
[runs/large-final-20261007/run-summary.txt](experiments/m4-ablation/runs/large-final-20261007/run-summary.txt).
The earlier compact large-tables.md is retained; differences at the last decimal
between presentation and Rust summaries arise from rounding raw logs, not a
changed measurement. The Rust summary below uses full internal precision.

### Data and artifact identity

The final source contributions (all splits combined) and nominal proportions are
recorded in corpus-v2/composition.txt. Total retained raw data: 2,026,889 bytes.
TRAIN is 1,857,089, validation 82,984, TEST 86,816 bytes. The source mix includes
English/German original prose, Rust code/config and labeled synthetic JSON/Unicode.
No model/test outcome changed these splits. Long-passage guard threshold is
128 bytes; 12,400 held-out bytes were dropped before tokenizer/model evaluation.
Validation has no Rust source after filtering; TEST has 4,136 Rust source bytes.
This limits domain-specific interpretation of aggregate quality.

| Final split | SHA-256 |
|---|---|
| TRAIN | `058ee713454bc189152da5a49839768e0ae221d3bd2bed9b3822a4be6ea4aebd` |
| Validation | `3ccfba064e3bae79eb99b13529f47f13596397975e0a5830cac86990abb9fdc3` |
| TEST | `19f7f19a11d8dbf85ccec7a1bbfedfcd35ce8c8a3381a489b92f1c19937b06c5` |

All raw source hashes/bytes and failed-v1 corpus hashes are preserved in
[input-SHA256SUMS.txt](experiments/m4-ablation/input-SHA256SUMS.txt); all entries
were checked again after final runs. Source URLs/licenses are listed above and
full notices/license files are retained.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| M1 tokenizer v2 | 3,984 | `5591e0355d69268bf4fa47fb7961ba341acdf83816da8aeb8f1ccbfb2755b0b4` |
| M2 tokenizer v3 | 4,543 | `80cb754297f4627bc0d18db34bdfd9049fe5432cc4577f2b34a6bced5746e2dd` |
| B mapping | 3,084 | `72534f345b0c0f37002991172933b6c7f666fcd3adb49e34a5b09f99e2acc7ae` |
| C mapping | 3,084 | `ad9953ea5bc9f07ba8dc42e0b4b9ef17222b872f3c8cd42c1992f9c3d8fdbded` |

Both tokenizer artifacts were trained twice and byte-compared. Both realized
512 logical IDs: 256 byte rows and 256 learned merges. M2 allocation is TEXT
214, NUMBER 9, STRUCTURE 33; these corpus/config observations are not linguistic
truth. M2 model-facing vocabularies are 214/9/33/256 in numeric ID order
0/1/2/65535. B's four synthetic packs have 128 rows each. C's global ranges are
TEXT 0..213, NUMBER 214..222, STRUCTURE 223..255, BYTE 256..511, all inclusive.

| Split | M1 tokens | M1 bytes/token | M2 tokens | M2 bytes/token |
|---|---:|---:|---:|---:|
| TRAIN | 908,767 | 2.04352601 | 1,031,748 | 1.79994437 |
| Validation | 39,979 | 2.07568974 | 46,308 | 1.79200138 |
| TEST | 41,901 | 2.07193146 | 48,255 | 1.79910890 |

M1 scores 41,900 targets representing 86,812 TEST bytes; M2 scores 48,254 targets
representing 86,813 bytes. First-token exclusion is the original M3 rule. M2
emits more tokens on all three splits despite its small byte-normalized model
advantage; fewer tokens are not the mechanism here.

### Primary same-schedule table

Same hidden/context 16, batch four windows, Adam config in the design; 2000
updates and 128000 sampled targets per model. Seeds 20261007/20261008/20261009.
Values are mean ± population SD, except parameter counts and shared process peak.
MACs are the analytical M3 v1 definition; time includes declared boundaries.

| Variant | Test bits/raw byte | Parameters | MACs | Train ms | Test ms | Shared process peak B |
|---|---:|---:|---:|---:|---:|---:|
| A: M1 + flat | 2.61247041 ± 0.00154161 | 17,424 | 3,241,984,000 | 9039.564 ± 131.156 | 1074.113 ± 6.790 | 81,297,408 |
| B: M1 + synthetic factorized | 2.62290377 ± 0.00296306 | 17,492 | 907,264,000 | 4713.490 ± 17.097 | 654.022 ± 11.985 | 81,297,408 |
| C: M2 + flat | 2.58136040 ± 0.00099970 | 17,424 | 3,241,984,000 | 9877.250 ± 109.302 | 1237.723 ± 5.378 | 81,297,408 |
| D: M2 + factorized | 2.59248241 ± 0.00554472 | 17,492 | 1,458,011,328 ± 721,828.9 | 6284.384 ± 60.865 | 860.290 ± 8.108 | 81,297,408 |

All models have 8192 embedding and 528 backbone parameters. A/C heads have
8704 parameters; B/D heads have 8772, including the four pack rows and biases.
The factorized models have 68 extra parameters. Model files are 69742 bytes
(A/C) or 70032 (B/D); parameter/Adam/gradient estimates are 278784/279872 bytes.
The peak in every row belongs to the same full process, including leakage checks
and tokenizer training, and is not an isolated per-variant comparison.

| Same schedule: paired contrast | Mean bits/byte | Population SD |
|---|---:|---:|
| B-A | +0.01043336 | 0.00206394 |
| C-A | -0.03111001 | 0.00239261 |
| D-A | -0.01998800 | 0.00689687 |
| (D-C)-(B-A) | +0.00068865 | 0.00628396 |

C beats A in all corresponding seeds. B loses to A in all seeds. D beats A but
loses to C in all seeds. The interaction changes sign across seeds and its mean
is small relative to its SD; this does not support a strong favorable synergy.
These empirical differences do not establish independent causal effects.

### MAC-aware regime

A's 3,241,984,000 estimated MACs per seed define the target. B gets 7147 updates
per seed; C remains at 2000; D uses 4446/4444/4451. Complete-batch overshoot is
preserved. Same seeds, model dimensions and optimizer; raw exposure is unequal.

| Variant | Test bits/raw byte | Actual MACs mean | Train ms mean ± SD | Test ms mean ± SD |
|---|---:|---:|---:|---:|
| A | 2.61247041 ± 0.00154161 | 3,241,984,000 | 8955.431 ± 46.838 | 1062.902 ± 4.574 |
| B | 2.51473152 ± 0.01380958 | 3,242,107,904 | 9161.516 ± 15.224 | 644.720 ± 5.403 |
| C | 2.58136040 ± 0.00099970 | 3,241,984,000 | 9877.331 ± 43.604 | 1244.129 ± 4.268 |
| D | 2.49148184 ± 0.01241998 | 3,242,441,280 | 9120.821 ± 33.225 | 852.352 ± 7.517 |

| MAC-aware paired contrast | Mean bits/byte | Population SD |
|---|---:|---:|
| B-A | -0.09773889 | 0.01499952 |
| C-A | -0.03111001 | 0.00239261 |
| D-A | -0.12098857 | 0.01388494 |
| (D-C)-(B-A) | +0.00786033 | 0.01261985 |

Every individual seed, train target/raw-byte count, actual MAC count, both sets
of paired contrasts and all checkpoint curves are retained in the complete
linked tables/raw log. Model losses improve through the fixed larger-corpus
curves; validation only reports and never chooses updates/checkpoints/config.
The MAC-aware head gain comes with more optimizer steps and raw-byte exposure;
it is not evidence that a factorized head has inherently better predictions at
an equal schedule. Analytical MAC matching is not wall-time/FLOP matching.

### Structural costs, head and generation observations

A/C evaluate 512 output logits/target. B evaluates four pack logits plus 128
local logits; D evaluates four plus a mean 218.463278 gold-pack local logits.
Total TEST logits are A 21452800, B 5530800, C 24706048, D 10734743. These are
teacher-forced target scoring counts, not greedy joint-output enumeration.

| Regime | Head | Pack accuracy mean ± SD | Local accuracy under gold pack mean ± SD |
|---|---|---:|---:|
| Same schedule | B | 0.360111 ± 0.004806 | 0.301599 ± 0.001423 |
| Same schedule | D | 0.694305 ± 0.002310 | 0.366042 ± 0.003510 |
| MAC-aware | B | 0.355338 ± 0.003613 | 0.333246 ± 0.004261 |
| MAC-aware | D | 0.691977 ± 0.001494 | 0.383809 ± 0.001477 |

B's synthetic target counts are 10641/9315/11438/10506 for IDs 0/1/2/3, despite
balanced vocabulary sizes. D's are TEXT 18783, NUMBER 407, STRUCTURE 4134,
BYTE_FALLBACK 24930. D fallback share is 0.516641. Per-pack accuracies and all
individual observations remain in the raw tables. Generation pack-selection
counts/decoded hex are retained for every model. All 48 final models across
tiny/larger runs reload with identical scoring, generate valid IDs, decode, and
preserve the byte-exact raw prompt. Generation quality is not scored.

Single-encode throughput observations, allocation-inclusive: TRAIN M1 4.036171,
M2 3.341541 MiB/s; TEST M1 7.896893, M2 3.301296 MiB/s. Full split timings are
in the tables. There is no warmup or throughput confidence interval. One-time
large tokenizer training observations were 22460.618 ms M1 and 10158.575 ms M2,
excluding input loading and serialization. The second independent trainings
were for determinism, not used as repeat timing samples. No GPU claim is made.

### Interpretation matrix and next decision

At an equal schedule, the larger-corpus result matches the **C improves, B does
not** case: the surviving small quality advantage comes from M2's token sequence,
not generic output factorization or a strong beneficial interaction. The large
M3 effect shrinks markedly, from tiny D-A=-0.86751248 to larger D-A=-0.01998800
bits/byte. Its sign persists, but its magnitude was dataset-specific under this
RNN/protocol. The tiny flat-M2 control already explains that original gain.

At matched analytical MACs, **both mechanisms contribute**, with the larger
contrast from generic factorization's cheaper active head enabling more training.
D combines the smaller tokenizer-sequence gain and that budget benefit. Neither
interaction is compelling evidence for a special positive PackTok synergy;
interactions are empirical, seed-dependent and not proofs of causal independence.

A modest, fully controlled L4 Transformer ablation is scientifically worth
considering to test whether this CPU head-cost opportunity survives a different
backbone and real compute measurement. It should retain A/B/C/D input controls,
match actual resources, and improve corpus/domain coverage before broad claims.
M4 supplies no GPU performance prediction or justification for tokenizer redesign.
Remaining limits: single corpus/machine, three seeds, small RNN/token context,
unequal raw-byte training/context, held-out filtering/domain mismatch, shared
process peak and analytical compute omissions. No general superiority is claimed.
## Synthetic source algorithm and test fixtures

Corpus generator records i=0..1999 in ascending order, one LF-terminated UTF-8
record each, with no random sampling. JSON fields are record=i, seed=i*7919,
voltage=(i%241).(i%100) with unpadded decimal components, enabled=(i%2==0),
path=/dataset/item/i, limits=[i+17,i*3]. This seed field is synthetic content,
not a model initialization seed. The preserved source includes the exact comma
spacing. Unicode lines use the fixed text "Datensatz i: Grüße aus Köln — Straße,
Häuser, Äpfel; 東京 漢字 🦀 🌍 👩🏽‍💻 é / x ± y = z.", x=i*31, y=i%19,
z=x+y. The é uses U+0065 then U+0301; no normalization is performed.

The four-variant overfit/reproducibility test uses seed 19, hidden/context 16,
80 updates on an authored repeated harbor/numeric fixture and requires final
loss below 40% of initial loss. It compares initial A/B and C/D hidden states
exactly, checks every mapped token round-trip, independent MAC formulas,
parameter totals, reload states and eight-token greedy byte-prefix generation.
Additional fixtures test uniform ln(2)+ln(128) factorized loss, balanced
512/3 grouping sizes 171/171/170, duplicate/missing/out-of-range IDs, represented
byte normalization and long partial-copy leakage. Full test families and
fixtures are in the checked-in Rust tests, with defaults/limits documented above.

## Final verification and adversarial review

Post-run verification passed all four required commands: cargo fmt --all --check,
cargo clippy --workspace --all-targets --all-features -- -D warnings,
cargo test --workspace, cargo build --release --workspace. Windows Rust/Cargo
1.98.1 passed 125 unit/integration tests; Linux WSL Ubuntu 26.04.1 on Rust/Cargo
1.85.0 (declared MSRV) passed 126. Doc-test targets contained zero tests.
The additional post-run regression compares complete retrained M3 tokenizer
bytes under joined/slash path spellings. It extracts the existing path expression
without changing measured behavior. Final verification logs are separate from
all preflight logs under verification/windows-final-20261007 and
verification/linux-final-20261007.

M0/v1 and v2/v3 frozen parse/serialize expectations remain in passing tests.
The historical M1 fixture was retrained twice and matches its historical hash;
the M1/M2 held-out-tokenizer-control artifacts also match their frozen hashes.
Complete hashes/CLI logs are in verification/frozen-artifacts/. The tiny harness
matches all historical M3 A/D model hashes, not only their reported losses.

The implementing agent completed an adversarial self-review after final model
runs. It is not an independent review. Evidence, limitations and each requested
review focus are in [SELF_REVIEW.md](experiments/m4-ablation/SELF_REVIEW.md).
No new numerical or token-sequence bug was found. The earlier path metadata bug
now has a direct full-artifact regression. Failed source endpoints, v1 preparation,
WSL launch and aborted tiny attempt all remain preserved. No experiment or
historical artifact was deleted, overwritten or selectively excluded by quality.
Nominal original cached token/byte-length vector payload lengths on the 64-bit
host are M1 15,850,352 bytes and M2 18,020,976 bytes, combined 33,871,328;
corresponding mapped per-run copies add that tokenizer's nominal payload. These
are length-based TokenId(8 B)+usize(8 B) estimates, not capacities, allocations
or RSS. Exact split percentages and per-source retained contributions are in
[composition-storage-summary.txt](experiments/m4-ablation/composition-storage-summary.txt).
The initial frozen-M1 ad hoc verification failed because its expected hash was
mistyped; extracting the full historical hash from M1_BPE_BENCHMARK.md confirms
both artifacts match. That failed check is retained and did not alter data/code.

Presentation-only summary revisions preserve both initial compact tables and
complete tables. The preserved script recomputes population statistics from
rounded logs and writes new output paths without overwriting raw runs. Root
primary tables use Rust's full-precision summaries; last-digit rounding is
explicit. The Linux verification script now requires a unique labeled directory
and refuses overwrite, retaining original preflight evidence.
The complete SHA256SUMS.txt captures 136 M4 evidence files, including sources,
splits, wire artifacts, scripts, tables and verification logs; every digest was
verified before delivery. Raw verification output preserves CRLF and a final
blank line where emitted. Git whitespace attributes accept those log bytes;
no captured output was trimmed or normalized to satisfy a whitespace check.
