# M6 resource estimate and staged execution

## Empirical basis

The M5 L4 pilot measured A 212.249 synchronized training seconds and C 196.922 seconds for 2,000 updates each. Their sum is 409.172 seconds per A/C pair, about 6.82 minutes of training-only GPU time. This excludes VM allocation, startup, input transfer, five validation evaluations, checkpoint I/O, final scoring, archive transfer, and cleanup. The historical A and C allocations lasted 813 and 1,244 seconds (2,057 seconds combined) around 409 seconds of synchronized training. They were separate sessions, so this is evidence of substantial overhead, not a reliable multiplier for a future single session; do not treat 6.82 minutes as a session estimate.

Linear training-only projections from this one L4 run:

| Updates per variant | Pair training seconds | Pair training minutes | Five-seed training hours |
|---:|---:|---:|---:|
| 2,000 | 409 | 6.82 | 0.57 |
| 10,000 | 2,046 | 34.1 | 2.84 |
| 20,000 | 4,092 | 68.2 | 5.68 |
| 30,000 | 6,138 | 102.3 | 8.53 |

Five-seed hours are just 5× pair training and exclude all operational/evaluation overhead. The 20k/30k pair projections exceed one 60-minute allocation before overhead; they require multiple explicitly authorized sessions or a revised approved schedule. Do not stretch a bounded allocation.

M5 throughput was 19,298 A tokens/s and 20,800 C tokens/s, but represented raw throughput was 35,692 versus 35,274 bytes/s. Linear 2k exposure repeated across horizons is not guaranteed because windows are sampled with replacement and domain/sampler behavior can vary. For planning only, 10k implies roughly 37.9M A / 34.7M C sampled target bytes; 20k roughly 75.8M / 69.5M; 30k roughly 113.6M / 104.2M. These are projections, not measured unique bytes or epochs.

## Uncertainty and cost controls

Per-step timings came from one run per variant and one L4. Throughput can change with warm-up, validation/checkpoint overhead, thermal/host contention, data-transfer behavior, runtime version and longer-run system effects. Allocate explicit startup/shutdown margin based on observed session history; train only within a preapproved window and release immediately after work. Track actual CU balance and allocation duration separately from synchronized model time. No CU charge estimate here is an authorization or a guarantee.

## Recommended staged plan

1. CPU-only audit the source-group splits, exact duplicates/near duplicates, byte normalization and tokenizer identities. Resolve the stale tokenizer hashes in the old extended proposal before using it.
2. Run five paired 2k T replications (new seeds 20261009–20261013) to learn paired-seed variability at a modest measured training-only cost, after separate GPU budget approval.
3. Run one B sampler dry-run on CPU; then a paired byte-exposure experiment at a frozen budget, explicitly reporting extra positions/updates/time.
4. Only if the pre-registered 2k decision rule supports further training, run a five-seed 10k stage. Review learning curves and data adequacy before authorizing 20k; consider 30k only after 20k demonstrates continued validation improvement and avoids overfit indicators.
5. Keep GPU allocations within approved lifetimes; the 20k and 30k projections require staged sessions. This M6 research-design task has used no GPU and authorizes none.


## Slice 2 — remaining paired-seed cost

Seed 20261008 is already complete, leaving **four new pairs / eight 2,000-update training runs** (seeds 20261009–20261012). The M5 observed training-only timings project 212.249 s for A and 196.922 s for C per pair: 409.171 s (6.82 min) per pair and 1,636.684 s (27.28 min) for all four pairs. This is training-only; it excludes validation, checkpoint I/O, package/input transfer, startup, final-only evaluation, archive work and shutdown.

A single 60-minute allocation for all four pairs is not supported by the historical evidence with enough margin. Plan **two separately authorized L4 sessions**, each completing two paired seeds (four model runs, about 13.64 minutes projected training-only). Allow roughly 30–45 minutes per session as an operational planning range, with substantial uncertainty; begin controlled shutdown at minute 55 in every future authorized session. If measured startup, evaluation, or run progress makes that range unsafe, preserve complete pairs and stop. No GPU work is authorized by this preparation.

For a rough CU planning proxy only, M5 consumed 0.65 CU over two allocations for one pair; linear scaling to four additional pairs is about 2.6 CU. This includes historical overhead and is not a billing guarantee. A two-session plan may reduce repeated startup cost, but no defensible lower bound is available. Read actual CU balance before/after each future session and report it separately from synchronized training time. No allocation has been made for M6 Slice 2.



## Slice 2 — remaining paired-seed cost

Seed 20261008 is already complete, leaving **four new pairs / eight 2,000-update training runs** (seeds 20261009–20261012). The M5 observed training-only timings project 212.249 s for A and 196.922 s for C per pair: 409.171 s (6.82 min) per pair and 1,636.684 s (27.28 min) for all four pairs. This is training-only; it excludes validation, checkpoint I/O, package/input transfer, startup, final-only evaluation, archive work and shutdown.

A single 60-minute allocation for all four pairs is not supported by the historical evidence with enough margin. Plan **two separately authorized L4 sessions**, each completing two paired seeds (four model runs, about 13.64 minutes projected training-only). Allow roughly 30–45 minutes per session as an operational planning range, with substantial uncertainty; begin controlled shutdown at minute 55 in every future authorized session. If measured startup, evaluation, or run progress makes that range unsafe, preserve complete pairs and stop. No GPU work is authorized by this preparation.

For a rough CU planning proxy only, M5 consumed 0.65 CU over two allocations for one pair; linear scaling to four additional pairs is about 2.6 CU. This includes historical overhead and is not a billing guarantee. A two-session plan may reduce repeated startup cost, but no defensible lower bound is available. Read actual CU balance before/after each future session and report it separately from synchronized training time. No allocation has been made for M6 Slice 2.
