# Semantic Scale — Rust training and Google Colab

This separate, experimental workspace prepares a single-GPU path from a small
correctness fixture through about 100M to about 1B parameters. Rust owns the
model, gradients, optimizer, data preparation, batching, checkpoints and metrics.
The notebook's Python code only installs tools, runs commands and transports files.

**GPU compilation, BF16 CUDA execution, actual 1B memory use and throughput have
not been verified on this development machine.** The notebook builds the CUDA
binary and performs an actual hardware preflight before training. A configuration
that cannot fit fails explicitly; it never silently switches models or hardware.
A short 1B preflight is not a trained, capable language model, and no better
learning/generalization from semantic annotations is established.

## Open the notebook

[Open in Google Colab](https://colab.research.google.com/github/DasEtwa/PackTok/blob/feat/semantic-packs-v1/notebooks/PackTok_Semantic_Scale_Colab.ipynb)

[Notebook file](../../notebooks/PackTok_Semantic_Scale_Colab.ipynb)

Select a **GPU runtime with an NVIDIA L4**, adjust the configuration cell, then
run all cells. Colab's runtime selection/Drive authorization are user actions;
the notebook performs installation and the remaining pipeline itself. It uses a
pinned repository commit, Rust 1.90.0 and the committed Cargo.lock. It checks CUDA
and installs missing development tools from NVIDIA's official repository, builds
the Rust executable and checks the allocated GPU. No GPU is allocated by this
repository or during development here. It targets L4; it does not retry allocations.

The default is a bounded `tiny`/semantic demonstration on explicitly generated
English/German smoke-test data. Choose `100m` or `1b` explicitly. With demonstration
data, `1b` is restricted to preflight: meaningful pretraining requires your own
corpus. The preset label is approximate: at context 128 the exact parameter counts
are tiny 174400, 100m 101940736, and 1b 1021604864. These include the semantic
embedding in both baseline and semantic variants. Other context sizes change
positional embedding counts and are reported by the planner.

## Notebook configuration and persistence

The notebook exposes model preset, baseline/semantic variant, context, microbatch,
accumulation, update count, seed, wall-time limit, data source and run name.
The configuration is bound to the run directory and an OS file lock prevents
concurrent writers; locks release on process exit, including termination. Use a new run name when changing
it. Re-running the same plan can restore its complete optimizer/sampler state.
`train.txt`, `validation.txt`, `test.txt` are required for an external corpus.
No large external corpus or pretrained weights are downloaded automatically.

Checkpoints are written locally, outside Google Drive's FUSE mount. Optional
Drive backup copies completed, hash-verified checkpoint generations and publishes
`latest.json` last. The notebook restores the complete run before resuming.
Only completed images of the same owned run are eligible for retention; logs and
failed/incomplete images are preserved. A local runtime alone is ephemeral.
Disconnect/recycle of Colab loses unbacked-up local state. Runtime disconnection
must occur only after verified preservation or explicit disposal of local output.

Large checkpoints are a material storage requirement. A BF16 1b checkpoint contains
BF16 model weights, FP32 master weights and two FP32 Adam moments: approximately
**14.3 GB per image** at context 128, excluding metadata. New and old backup images
coexist while a new image is verified, so the free 15-GB Drive tier is insufficient
for a reliable rolling 1B backup. The notebook's declared Drive budget prevents
silently assuming more storage. Use sufficient persistent storage, or explicitly
choose an ephemeral local-only preflight. Rust retains two completed training
images and one preflight image, with an extra image temporarily needed during a
write; initial local admission estimates about 55 GiB free for the 1B preset plus
already-prepared data. These are capacity estimates, not measured GPU results.

## Implementation and memory

M5 is frozen at its original configuration and corpus/protocol. Its hard 128M
model limit and 14.7M fixed experiment cannot own this change without invalidating
historical comparisons. Following M5's existing isolation pattern, this workspace
uses pinned Candle core/nn 0.9.1 without adding CUDA dependencies to root crates.
It does not modify the original M0–M5 artifacts, vocabulary IDs or experiments.

- BF16 model weights, FP32 RMSNorm/attention-softmax/loss and FP32 AdamW master
  weights/moments. CPU correctness fixtures use FP32; CPU allocation over 5M
  parameters is rejected. No full-model CPU training is used as a GPU workaround.
- Real layer recomputation: forward saves detached block inputs; backward
  recomputes one block at a time and propagates its input vector-Jacobian product.
  Unit tests compare every gradient to ordinary full-graph autodiff. Evaluation
  also detaches each block rather than retaining a complete autograd graph.
- Microbatch gradients accumulate directly into one FP32 map, avoiding a second
  complete gradient map. Tests compare accumulation with a full-batch mean.
- Causal dense attention remains quadratic inside a layer. This implementation
  does **not** claim Flash Attention, multi-GPU distribution or optimizer sharding.
  Context/microbatch choices remain important even with layer checkpointing.
- Resource planning validates shapes in u128 and caps the model at 2B parameters.
  CPU planning of 1B counts does not allocate its tensors. BF16 state admission
  estimates 18 bytes/parameter (model2 +master4 +moments8 +accumulated gradients4),
  plus layer inputs/current-block work and a 2-GiB reserve. FP32 uses 16 bytes per
  parameter by reusing model storage as master. An estimate is not an OOM guarantee;
  preflight must measure the actual CUDA path on the selected runtime.

The model has separate token, learned position and six-row semantic embeddings,
pre-norm RMSNorm, biased attention/GELU feed-forward blocks and a flat untied head.
Row zero is masked to exactly zero semantic contribution. Baseline and semantic
variants have the same tensor layout, parameter count, initialization, optimizer
schedule and raw token sequence; baseline uses zero semantic inputs. The semantic
variant supplies deterministic annotations, which are additional information whose
benefit must be evaluated. Initialization follows stable sorted names and a seeded
host PRNG. CUDA floating-point reductions are not guaranteed bit-identical across
hardware or executions. No dropout or stochastic sampler state is hidden in resume.

## Causal semantic features and streaming data

Data preparation trains the existing flat BPE only on the first 1 MiB of TRAIN,
with target 512; actual vocabulary may stop at 256–512 IDs. Original lines,
including their newline, are encoded independently, preserving every input byte.
Lines over 1 MiB are rejected before unbounded buffering; there is no silent
normalization, truncation or inserted separator. All lexical IDs remain ordinary
BPE IDs, and all bytes stay representable.

Prepared format v1 has a 32-byte header and fixed 12-byte records: lexical ID,
semantic feature and expanded byte length (`u32` little endian). A SHA-256 manifest
binds all splits, source hashes, tokenizer, registry and feature policy. Dataset
opening validates the data by streaming; batches seek only the required windows,
so host RAM does not grow with the whole corpus. The first token of each epoch
is context only; short split tails are dropped rather than wrapped within a
prediction window. Cursor, seed and counters are checkpointed and verified.

The adapter uses only unambiguous `KnownVariant` results from the built-in
registry. Context-resolved `fr`/`ka` are excluded from this first model adapter.
A feature is delayed until the following delimiter scalar has been completely
observed in lexical input; an EOF-ended phrase emits nothing. Apostrophe
separators are excluded to avoid premature possessive/contraction decisions.
If one lexical token completes several expressions, the latest completion wins.
Malformed/incomplete UTF-8 suppresses semantic detection only in the remaining
portion of its original line, while all lexical bytes are retained.

The feature policy is `causal-known-window-v1`. Lexical expansion caches are
bounded to64MiB and semantic input-window reconstruction to16MiB; exceeding
either bound returns an error rather than truncating text.

Batch features are recomputed from the **actual input window only**, using cached
lexical expansions and the same causal helper. This prevents a feature at a
window's beginning from conveying a phrase outside the model's available context.
It never inspects the next-token target to annotate the current input. Stored
full-line feature records remain preparation provenance, not a source of hidden
context in the model batch. Prefix tests include partial words, emoji, invalid
UTF-8, quotations, phrases and context switches.

You must provide meaningful disjoint source splits. Exact duplicate whole-split
hashes are rejected; near-duplicate passages, document leakage and licensing are
not automatically solved by this small preparer. The generated demo is a smoke
fixture, not independent evidence of multilingual model quality.

## Preflight, training and evaluation

The planner creates an immutable versioned JSON plan and reports exact parameter
counts, dataset identity and a rough VRAM estimate. A successful preflight is
required for the exact plan, executable hash and hardware class before training.
Preflight checks a finite forward/backward/update, changed weights, sharded
checkpoint save/load equality and a further restored backward/update. It writes
PASS only after completing the checks. Initialization, computation and checkpoint
I/O count toward a maximum 20-minute preflight soft bound; the notebook additionally
bounds its own process. No success is inferred from CPU tests or compilation.

Training uses AdamW with warmup/cosine decay, global L2 clipping and configured
gradient accumulation. Wall limits are checked between microbatches; an incomplete
accumulation is discarded and the sampler cursor restored. Checkpoint I/O or an
individual GPU operation can extend the Rust soft deadline. The notebook's process
watchdog handles bounded termination and leaves partial images unpublished.

Complete version-1 checkpoints shard one model variable and its master/m/v states
per Safetensors file (512-MiB per-shard bound). A SHA-256 index binds names, layout,
optimizer schedule, run/data/variant identity, step, cursor, token/byte counts and
seed. Loading validates every shard before changing weights, then transfers one
shard at a time. Host memory scales with the largest shard instead of all 1B
states. The last complete local image is published atomically on POSIX; old image
retention never removes another identity, symlinks, logs or incomplete images.

Resume archives the prior metrics and removes records after the restored
checkpoint from the canonical curve. Failed-attempt history remains available;
resumed weights, optimizer and sampler reproduce the CPU reference trajectory.

Evaluation reports up to the first eight context windows of VALIDATION and TEST,
with exact target-token and expanded-byte counts, mean NLL/token and bits/raw-byte.
It is explicitly sampled, not a full-corpus quality score. Compare baseline and
semantic runs using the same data, context, seeds, update budget and initialization;
use separate run names and record additional annotation/runtime costs. Different
wall-limited update counts are not a matched-compute quality comparison.

## Command line and local verification

Rust 1.90.0 (CPU fixtures here, no CUDA install required):

```sh
cargo +1.90.0 fmt --manifest-path experiments/semantic-scale/Cargo.toml --all --check
cargo +1.90.0 clippy --locked --manifest-path experiments/semantic-scale/Cargo.toml --all-targets -- -D warnings
cargo +1.90.0 test --locked --manifest-path experiments/semantic-scale/Cargo.toml
cargo +1.90.0 test --locked --release --manifest-path experiments/semantic-scale/Cargo.toml
python3 -m unittest discover -s experiments/semantic-scale/tools -p 'test_*.py'
python3 experiments/semantic-scale/tools/check_notebook.py notebooks/PackTok_Semantic_Scale_Colab.ipynb
```

On the Colab L4, the notebook builds with `--features cuda --release --locked`.
CLI commands are `prepare`, `plan`, `preflight`, `train --resume`, `evaluate`;
`help` shows exact options. `--cpu` exists solely for explicit tiny FP32 diagnostics
and cannot run the 100m/1b presets. Dataset output directories are immutable.
Existing plans cannot be silently overwritten with changed configuration.

[Validation and review evidence](verification/README.md) records actual local
checks, original failures/fixes, independent reviews and remaining GPU unknowns.
These results do not establish 1B throughput, completed training or semantic gains.
