# Final CPU contract fix evidence

Base checkout: 53ef2e8639a1e77c40bb26f2a935bdd321f459c0, branch
m4-factorization-ablation. Source changes are limited to model artifact-size
validation and complete greedy-prompt ID validation, plus regression tests.
No M5 implementation, tokenizer behavior change, experiment tuning or benchmark
retraining is performed.

The original review and raw probes under experiments/pre-gpu-review-20261007
are preserved. PRE_GPU_CODE_REVIEW-before.md is the byte-exact original root
report, copied before appending resolution details; its hash matches the old
review manifest's root-document entry. The original manifest remains historical,
so its root entry refers to that before snapshot rather than the updated report.

The compatibility probe loads and reserializes the canonical M3 final/audit
and M4 tiny/larger models, and the associated tokenizer files, without training.
It also repeats every archived tiny M4 A/D versus M3 model equality check and
records fixed valid-prompt generation from each saved model.

Initial compatibility probe failed only in the comparison's filename spelling:
it used b-compute-budget instead of the archived b-matched-mac prefix. The
failed compatibility-before-windows.txt is retained. No model, tokenizer or
experimental setting changed. The corrected compatibility-before-windows-v2.txt
completed: 72 model round-trips, 6 tokenizer round-trips and 12 historical pairs.

Valid generation golden fixtures were recorded before production edits with
hidden/context 2, seed 19, 256 rows and prompt locals 97/98/99. Flat continuation
is 113/15/180/113/15/180/113/15; single-pack factorized continuation is eight
local-112 tokens in pack 7. The tests assert these recorded values directly.

The first scoped post-fix run was cargo fmt --all followed by cargo test
-p packtok-model: all 37 model tests passed. Final platform/debug/release totals
and checks are preserved separately; this initial pass is not the completion gate.

Boundary tests calculate equivalent maximum wire layouts instead of allocating
multiple near-64-MiB weight arrays plus Adam moments. They cover every declared
pack count, multiplication/addition overflow and near-boundary public constructor
rejections. Smaller accepted layouts reach the intentionally mismatched-body
guard without allocating large state. Representative normal models fully load
and serialize; embedded M3/M4 fixtures preserve exact bytes.

Regression fixture details: the byte-size calculator tests pack counts 1..1024;
single-pack maximum parameters 16,777,204 give 67,108,862 bytes, while the next
parameter exceeds the cap. Four packs allow 16,777,200 parameters and exactly
67,108,864 bytes. Overflow cases exercise usize::MAX, usize::MAX/6 descriptors
and usize::MAX/4 parameters. Public rejection fixtures use hidden 8/context 6,
seed 19, flat vocabulary 986888, one pack with 986887 rows, and four packs with
986882/1/1/1 rows. The next smaller flat vocabulary 986887 and four-pack counts
986881/1/1/1 reach the mismatched-body guard without allocating state. Normal
round-trip fixtures use hidden 2/8/16, context 16, seed 19, flat 512 rows or the
documented M4 214/9/33/256 pack counts.

Prompt validation tests use seed 19, hidden/context 2, flat rows 256 or one
256-row pack with ID 7; invalid flat pack 1, flat/local ID 256 and unknown pack
99 are checked in every position of a three-ID prompt, with 0/1 continuations.
Length/error-order fixtures reuse hidden 8/context 4, seed 19 and 256 rows;
empty prompts, usize::MAX length overflow, the 4096-ID returned-sequence cap,
the first excessive length and valid zero continuations are covered.

The first staged whitespace check returned exit 2 on model-source.patch lines
42/131: blank context records in a preserved Git patch contain the required
leading space. The patch bytes were retained. A file-specific whitespace
attribute accepts those patch context records; no source or test changed.
The check stopped before a commit was created.
