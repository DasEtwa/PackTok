# M6 Slice 2: Five-Paired-Seed Results

**Status:** Complete. The protocol-identical M5 seed 20261008 pair was reused; four new pairs (20261009 through 20261012) completed eight real L4 training runs.

## Frozen protocol

All pairs used the frozen T regime: the 14,681,984-parameter Transformer, A Flat BPE and C PackTok, the same tokenizer/corpus/split identities and 512-ID comparison space, batch 8, context 256, 2,000 AdamW updates, learning rate 0.0003, constant schedule, FP32 and 4,096,000 target positions per variant. Validation ran at steps 1, 500, 1000, 1500 and 2000. Paired A/C initialization hashes match for each seed.

The M5 20261008 pair used the same scientific protocol and is reused as the already-observed first pair. Its split/test outcome had already been seen, so this is not a blinded confirmation.

## Primary validation result

Metric is bits per raw byte. Delta is C minus A: positive favors Flat BPE; negative favors PackTok.

| Seed | A validation BPB | C validation BPB | C minus A BPB |
|---:|---:|---:|---:|
| 20261008 (historical M5) | 2.572964850 | 2.597469229 | +0.024504380 |
| 20261009 | 2.565019717 | 2.571698223 | +0.006678506 |
| 20261010 | 2.518315761 | 2.638156364 | +0.119840603 |
| 20261011 | 2.525880036 | 2.585981443 | +0.060101407 |
| 20261012 | 2.553664034 | 2.593946317 | +0.040282283 |

Across five pairs, mean delta is **+0.050281436 BPB**, median **+0.040282283**, sample SD **0.043590609**. The two-sided paired Student-t 95% interval is **[-0.003843452, +0.104406323] BPB**. All five observed deltas are positive (A lower BPB); this is directional consistency in this small study, not statistical proof or a universal tokenizer result.

Excluding the already-observed M5 seed, the four-new-pair mean is **+0.056725700 BPB**, median **+0.050191845**, sample SD **0.047504364**, and paired-t 95% interval **[-0.018864337, +0.132315737]**. All four new deltas are positive; the interval still includes zero.

## Exposure and measured training cost

T matches target-token positions, not raw bytes, wall-clock allocation time, or FLOPs. C therefore saw fewer training bytes in every pair.

| Seed | A raw training bytes | C raw training bytes | A training seconds | C training seconds | A/C mean synchronized step (ms) |
|---:|---:|---:|---:|---:|---:|
| 20261008 | 7,575,564 | 6,946,291 | 212.249 | 196.922 | 106.125 / 98.461 |
| 20261009 | 7,584,510 | 6,945,183 | 213.267 | 210.341 | 106.633 / 105.170 |
| 20261010 | 7,576,981 | 6,934,928 | 196.987 | 197.084 | 98.493 / 98.542 |
| 20261011 | 7,586,183 | 6,943,689 | 197.210 | 197.107 | 98.605 / 98.553 |
| 20261012 | 7,579,751 | 6,942,028 | 197.209 | 197.221 | 98.605 / 98.611 |
| **Total** | **37,902,989** | **34,712,119** | **1,016.922** | **998.675** |

Across the five pairs, C represents **3,190,870 fewer raw bytes (8.4185% less than A exposure)** at the same token-position budget. Mean measured training duration was 203.384 s for A and 199.735 s for C. These are synchronized training measurements; they are not equal-FLOP claims and exclude allocation provisioning, transfers and cleanup.

Pooled over the five completed pairs, measured throughput was 20,139.20 target positions/s and 37,272.25 raw bytes/s for A, versus 20,507.17 target positions/s and 34,758.17 raw bytes/s for C. Peak sampled VRAM ranged from 2,378 to 2,442 MiB for A and 2,410 to 2,442 MiB for C. Allocation duration is reported separately below.

## Domain validation

The values below average paired C-minus-A BPB over the four seeds with complete paired domain aggregation (20261008 and 20261010 through 20261012). A/20261009 completed training and global validation, but its initial attempt timed out before domain aggregation. These domain means are descriptive on a small mixed corpus; the block-level intervals in the runner are not seed-level statistical intervals.

| Domain | Paired seeds | Mean C-minus-A BPB |
|---|---:|---:|
| English literature | 4 | +0.063019 |
| German prose | 4 | +0.075621 |
| Rust source | 4 | +0.079278 |
| Synthetic JSON | 4 | -0.002022 |
| Synthetic Unicode | 4 | +0.038348 |

The synthetic JSON paired mean is near zero and the seed-level signs vary. Synthetic Unicode also changes direction across pairs. Domain results do not establish specialization; seed-level rows are preserved in [MULTI_SEED_RESULTS.json](MULTI_SEED_RESULTS.json).

## Final-only TEST policy

The existing runner exposed final-only TEST scores for M5 seed 20261008 and the completed new runs. A/20261009 has no TEST score because its run final aggregation timed out after the step-2000 checkpoint. TEST data were not used to tune, debug, select runs, or change the protocol. The split was already exposed by M5 and is not an independent holdout. All exposed values are labeled exploratory in the JSON.

## Execution, recovery and integrity

- Allocation 1 completed A/20261009 through all 2,000 updates, all five global validations and a recoverable final checkpoint. Its per-run timeout was 300 s, so it exited 124 before final-only aggregation. A/20261009 primary validation result remains valid and was reused; it was not retrained.
- The runner was continued at C/20261009 with a 600 s per-run timeout. One subsequent L4 allocation completed the remaining seven variants in the required order: C/20261009, A/C for 20261010, A/C for 20261011, then A/C for 20261012.
- The remote process returned exit 0, its archived exit code is 0, and all seven runs have exactly 2,000 train rows, five validation rows, 4,096,000 targets, matching initialization identity, and checkpoint plus metrics checksums that pass readback verification.
- Two supervisor launches failed before allocating a GPU: exit 209 because the local stdout directory did not yet exist, then exit 127 because the systemd user unit lacked the Colab CLI PATH. The launch wrapper was corrected to create its owned log directory and use the explicit environment PATH; both original failures remain in the attempt provenance. No GPU allocation was consumed by those launches.
- The first local post-download checker exited 1 because it resolved checksum paths from each variant folder, while the archived SHA list correctly names files relative to the archive root. The corrected offline verifier resolved the recorded paths at that root and verified all 14 checkpoint/metrics SHA entries. The original error log is preserved; no remote results were modified.
- A/20261009 checkpoint, metrics and console checksums were independently reverified from the first archive. Its lack of a final row, TEST score and domain aggregation is retained as a limitation, not filled in.
- The statistical CLI documentation and implementation disagreed on accepting an object-wrapped pairs list. The CLI was corrected to accept both the documented wrapper and direct lists; its regression test was added. Three statistical analysis tests pass.

Source and binary identities, initialization hashes, complete validation curves, final-only scores, per-domain metrics, training logs, per-run checkpoint hashes and full run details are preserved in the machine-readable result file. The CUDA binary SHA-256 is `97b25305b455c7b19861b294f8bc5b9a1df2b4ea821a3af3d2a9ec2b4b1859da`; its Rust build source commit is `0af19cb6f009e8959ce72644acb8cc200c1aac31`. The corrected continuation package was built from source commit `802aba1da2df6ce46bab32632ebf48e9579f92da`, with frozen config SHA-256 `50cf36a6c9edf0f4423007b846da28ba3984eb37a8102664b27e165e9ee4954e` and corpus-manifest SHA-256 `033bf7a3126ad5e39d3d11e5abd257f9cfaaac2d8f001003830215191b9d494f`.

Two allocation attempts were used of the three authorized; they were sequential. Allocation lifetimes including provisioning and cleanup were 718 s and 3,387 s (56 min 27 s); total 4,105 s. CU snapshots were 166.50 to 166.24 and 166.24 to 164.92, a measured combined immediate drop of 1.58 CU. A later idle reading was 164.79 after delayed accounting; active assignments were 0. Both session releases are confirmed. No third attempt was used.

## Archive and GitHub

The preserved allocation archives are:

- Allocation 1 (valid partial A/20261009 and exit 124): `c220c11107f5a684d767b9d8f33346ff182c1fbb46d020ea3e4ace5a02431c35`.
- Allocation 2 (seven complete variants, remote exit 0, 1,130,545,701 bytes): `d8e582b852569ebca00c939f95cacd3d64a9a40172df3d86ed9e779ec75e81c8`.

Drive full-readback checksums and destination are recorded in the execution manifest after upload. M6 code and results are committed on `m6-research`; Draft PR #7 remains open and unmerged.
