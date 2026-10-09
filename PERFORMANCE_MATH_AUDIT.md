# Performance and mathematics audit — 2026-10-07

## Scope and provenance

Reviewed the M0–M3 workspace, prioritizing model loss/backpropagation, optimizer
state, parameter/compute accounting, runtime buffers, artifact loading, and
deterministic tokenizer behavior. Read IDEA.md, STRUCTURE.md, README.md, and
AGENTS.md before edits. The starting checkout was clean at
`c6a4667da35d45a7f41a25b6b31e6e3551171a15`.

Seven findings below were addressed. P1 means a failure can silently invalidate
subsequent model work or leave unusable training state; P2 means a concrete
correctness or avoidable resource problem with narrower triggers; P3 means a
validation/diagnostic inconsistency. No P0 was identified. This is a local
audit, not evidence of general model quality or superiority over a baseline.

Raw reproductions, three before/after microbenchmark repetitions, intermediate
failures, environment details, compatibility checks, and validation logs are in
[experiments/audit-20261007/](experiments/audit-20261007/).
The full M3 rerun lives separately in
[experiments/m3-model/runs/audit-20261007/](experiments/m3-model/runs/audit-20261007/).
Its `source_commit` is the base commit: it ran with these uncommitted fixes.
The companion `source-sha256.txt` and `source.patch` in the audit directory
identify that implementation; the base commit alone does not. Historical
M1/M2/M3 reports, inputs, and result directories are retained.

## Findings and resolutions

| ID | Priority | Finding | Resolution |
|---|---|---|---|
| A01 | P1 | Failed Adam updates partially mutate state; overflowing moments can be accepted | Preflight weights and both moments before committing state or step |
| A02 | P2 | Cross-entropy loses the normalization term at large common offsets | Subtract target from maximum before adding log normalization |
| A03 | P2 | Generation accepts non-finite logits and returns an arbitrary valid ID | Validate flat, pack, and selected local logits before argmax |
| A04 | P2 | Loading allocates before checking dimension/body consistency and generates random weights only to overwrite them | Check the shared layout before parameter allocation and directly load weights |
| A05 | P2 | M2 reserves whole-input symbol storage despite processing one span at a time | Reserve only the longest validated routed span |
| A06 | P2 | Evaluation computes/allocates unused probabilities and gradients | Use an allocation-free scalar loss path |
| A07 | P3 | MAC estimation accepts invalid input IDs; batch shape errors identify example zero | Validate input IDs and preserve the actual example index |

### A01 — transactional Adam state

Previously `apply_adam` incremented the step and changed each weight and its
moments before checking that weight. A later error left earlier changes in
place; the failing weight could remain infinite. A second-moment overflow was
not itself checked: a finite gradient of `1e22` can produce an infinite stored
second moment in the old f32 arithmetic while the weight remains finite and
the method returns success.

Regressions exercise a late failure using two gradients of one, a second weight
of `-f32::MAX`, and learning rate `f32::MAX`; and the overflowing-moment case.
Before the fix the first returned an error after altering state, while the
second returned success. Both now return `NonFiniteComputation` with weights,
moments, and step unchanged. These are adversarial cases, not observed failures
in default M3 training.

The fix keeps successful f32 arithmetic unchanged. It checks all proposed
values in one pass and recomputes them during commit, adding O(parameter count)
work without extra parameter-sized scratch vectors. The MAC metric still
excludes Adam, so wall-time measurements must account for this cost. Combined
patch timings below include the slightly slower flat training median.

### A02 — stable loss

For `[1e20, 1e20]`, the old expression
`maximum + ln(denominator) - target` returned zero: adding `ln(2)` to the large
f64 maximum rounded the term away. The correct loss is `ln(2)`; the old gradient
remained `[-0.5, 0.5]`, making loss and gradient disagree.

Both loss paths now use `(maximum - target) + ln(sum(exp(logit - maximum)))`.
Tests check equal logits at zero, both signs of `1e20`, and both signs of
`f32::MAX`. Probabilities/gradients retain f64 normalization and f32 gradient
storage. There is no text normalization change.

### A03 — finite inference

Finite weights can overflow an affine projection. Previously argmax silently
selected the first row for infinite/NaN scores. The regression saturates hidden
states and uses maximum finite output weights/biases; the old flat predictor
returned `(0, 0)` rather than an error. Tests cover the flat head, factorized
pack head, and selected local head. Non-finite logits are now rejected; finite
ties still select the first row in canonical order. Pack-first greedy
generation remains the documented policy, not global joint-probability argmax.

### A04 — loading order and redundant initialization

The old loader checked declared body length, constructed a seeded model from
dimensions, then compared parameter counts. A 46-byte flat header with hidden
512, context 16, vocabulary 8,192, and declared count zero could construct
8,667,648 parameters before rejecting the mismatch. Weights and two moments
would request `8,667,648 * 3 * 4 = 104,011,776` bytes of vector payload. This
is an analytical allocation count from the old layout, not measured RSS.

The shared layout calculation now checks supplied parameter bytes before
allocating weights/moments. Valid bodies load directly, avoiding seeded values
that would be overwritten. The compact-header regression checks the malformed
error through the loader and its internal construction path. Round-trip tests
and twelve M3 model hashes verify compatibility. Format version, parameter
ordering, IDs, and initialization of newly created models are unchanged.
Valid loaded models still allocate zero Adam moments; removing training state
from inference instances would require a separate API/design change.

### A05 — M2 workspace

The encoder clears and reuses one symbol vector at every routed span; merges
only shrink it. Whole-input reservation was unnecessary for short spans.
It now reserves the longest span after validating coverage and UTF-8 boundaries;
empty input reserves zero. The adversarial fixture repeats `hi 12!` 1,024 times
and checks exact decoding and capacity against router plus longest-span storage.
The focused benchmark also tests many one-byte spans and a single long span.
Routing, IDs, fallback, merge ordering, and serialization remain unchanged.
Output tokens still reserve the input-byte upper bound; materialized router
spans dominate the short-span example's temporary capacity.

### A06 — scalar evaluation

Evaluation previously called training softmax and discarded its gradient,
allocating a `Vec<f64>` for normalized probabilities and `Vec<f32>` for gradients
per scored head, in addition to logits. M2 did this for both pack and gold-local
heads. The scalar path now sums exponentials without those two buffers.
Logit/hidden-state allocations remain. Tests compare scalar and training losses
for one-class heads, widely separated logits, opposite maximum finite logits,
invalid targets, NaNs, and infinities. Teacher-forced evaluation is unchanged.

### A07 — validation/diagnostics

MAC estimation validated targets but not input IDs, despite promising validated
IDs and shapes. It now rejects invalid inputs instead of reporting usable work
for an invalid batch. The training target-count pass preserves the actual
example index for a later empty/inconsistent example. Regressions cover both.
Valid-batch MAC values keep their documented formula.

## Mathematical checks beyond regressions

Finite differences check every parameter of a hidden-size-2/context-3 flat
model (256 rows, 1,292 parameters) and a two-pack factorized model (local counts
2 and 3, 43 parameters), seed 19. Each batch combines a three-input aligned
window with a two-input final-only window, giving four supervised targets.
This checks recurrent carry, embedding/position/bias derivatives, both heads,
and normalization by targets rather than examples. Analytical gradients come
from first-step Adam first moments; clipping threshold 1,000 is checked to be
inactive. Central differences perturb by 1e-3 using actual stored f32
displacement, with tolerance `2e-5 + 0.01 * abs(analytical_gradient)`.

The fixtures also check exact MAC totals against `(3*n - 1)*d*d` per window
plus `3*rows*d` per target. No new error was found in recurrent backpropagation,
target averaging, parameter partitions, or the valid-batch MAC formula. Small
deterministic fixtures do not prove correctness for all model sizes/floats.

## Focused benchmark

Command: `cargo run --release -p packtok-bench -- audit`. The additive `audit-v1`
mode belongs to `packtok-bench`, introducing no crate/dependency. The before
build uses base production code plus this same harness; pending regression
tests are not executed in release measurements. Three sequential before and
three sequential after runs use Windows 10 Pro build 19045, Intel Core i7-2600
@ 3.40 GHz, x86_64 MSVC, Rust/Cargo 1.98.1, and one explicit benchmark thread.
Each workload warms for 25 ms and measures for at least 200 ms, finishing the
last iteration. Returned buffer destruction is included; allocator and host
effects are not isolated.

Models are untrained, seed 20,261,007, hidden/context 16. Flat vocabulary is 512.
Factorized counts are `(1:253, 3:3, 65535:256)`. There are 64 deterministic input
tokens: flat IDs `0..63`; factorized tokens rotate packs with valid modulo local
IDs. Evaluation scores 63 final-only targets with at most 16 preceding inputs.
Training scores four overlapping 16-token aligned windows, cloning the original
model before each default-optimizer update. Loading constructs/destroys a model
from serialized bytes. This characterizes fixed overhead, not model quality.

M2 trains only on literal `hello hello 12 12!!`, default config. Stress inputs
are `a1!` repeated 16,384 times and `a` repeated 49,152 times, each 49,152 bytes.
Both emit 49,152 tokens and round-trip exactly. They probe workspace/routing
overhead rather than compression. No external data is downloaded.

Values are median microseconds/iteration across three runs, with min–max ranges.
Positive change means slower. Raw iterations/elapsed times are retained in
`before-1..3.txt` and `after-1..3.txt`.

| Workload | Before median [range], µs | After median [range], µs | Time change |
|---|---:|---:|---:|
| Flat evaluation, 63 targets | 1,635.772 [1,540.797–1,787.659] | 1,412.287 [1,371.022–1,464.772] | -13.66% |
| Factorized evaluation, 63 targets | 928.905 [886.587–953.477] | 828.502 [813.921–836.603] | -10.81% |
| Flat training including clone, 64 targets | 2,177.212 [2,041.322–2,395.345] | 2,183.793 [2,143.103–2,200.649] | +0.30% |
| Factorized training including clone, 64 targets | 1,044.021 [1,013.999–1,118.597] | 983.054 [977.888–1,021.945] | -5.84% |
| Flat model load | 215.073 [206.868–219.008] | 89.619 [85.077–98.970] | -58.33% |
| Factorized model load | 135.622 [129.457–140.165] | 18.691 [18.497–19.386] | -86.22% |
| M2 encode, short spans | 1,585.926 [1,578.932–1,640.559] | 1,619.777 [1,587.787–1,624.877] | +2.13% |
| M2 encode, long span | 568.010 [561.314–591.802] | 559.487 [555.556–561.364] | -1.50% |

Loading/evaluation reduce measured time on these workloads. Training/encoding
ranges overlap; these samples do not establish their speed changes as general
effects. Adam preflight adds work; short-span M2 encoding has a slower median
despite its memory improvement. No unsuccessful observation was discarded.

Short-span temporary router/symbol capacity decreases from 1,966,080 to
1,572,872 bytes (about 20%): symbols decrease from 393,216 to 8 bytes, with
router capacity unchanged at 1,572,864 bytes. Long-span temporary capacity
stays 393,312 bytes. Output capacity is 393,216 bytes for both cases. These
are vector payload capacities, not allocation counts or per-operation RSS.

## Full M3 reproduction and compatibility

Command: `cargo run --release -p packtok-bench -- m3 audit-20261007`.
The same M3 synthetic splits and tokenizer configs were rerun with three seeds
and both regimes. Both tokenizer artifacts match their existing serialized
bytes. All twelve saved models match the corresponding `final-20261007`
SHA-256 hashes exactly; see `model-compatibility.txt`. Mean test bits/byte still
round to 6.87520175 for M1 in both regimes, 6.00768927 for same-backbone M2,
and 6.09533770 for matched-MAC M2. Historical quality conclusions are preserved
while timing observations are recorded separately.

New mean training/test-scoring milliseconds respectively are 304.8140/10.2972
(A M1), 200.6874/8.3123 (A M2), 300.8816/10.0569 (B M1), and 402.7034/8.1655
(B M2). Population SD/variance and individual seeds are in the raw summary.
Whole-process peak working set is 6,057,984 bytes. Training time still includes
scheduled validation; the matched MAC budget excludes Adam/softmax. The
historical run is not an interleaved performance control, so timing differences
against it are not isolated optimization effects.

## Validation and limits

Formatting, workspace Clippy with all targets/features and `-D warnings`,
`cargo test --workspace` (116 tests), `cargo test --release --workspace` (116
tests), and release workspace build passed on the recorded Windows toolchain.
Nine tests were added to the 107-test baseline.
Before fixes, five new model tests and the tokenizer workspace test failed.
Numerical-gradient, scalar-loss, and compact-loader checks extend that set.
The first test edit had a mutability compile error; an intermediate loader edit
had an RNG-scope compile error. Their logs remain preserved, excluded from
benchmark aggregation; both were repaired before final verification.

This patch does not establish Linux/MSRV behavior, large-corpus/GPU performance,
or broad natural-language quality. Installed Windows toolchains were 1.98.1
and 1.97.1; Rust 1.85 was not installed/downloaded. M2 still scans ordered merges
within spans, both trainers retain repeated corpus scans, and model evaluation
recomputes bounded sliding contexts. Those remain documented design limitations,
not newly demonstrated correctness failures. No new service, runtime dependency,
normalization mode, routing policy, or artifact version is introduced.
