# PackTok — Structure

**Canonical BRAIN space:** [`DasEtwa/BRAIN/PackTok`](https://github.com/DasEtwa/BRAIN/tree/main/PackTok)

Related documents: [README](./README.md) · [IDEA](./IDEA.md) · [AGENTS](./AGENTS.md)

This document defines the intended structure of the future standalone PackTok implementation.

The project is **Rust-first**. The structure is designed so experimental training code cannot quietly become a runtime dependency.

## Repository layout

```text
packtok/
├── README.md
├── AGENTS.md
├── STRUCTURE.md
├── IDEA.md
├── LICENSE
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
│
├── crates/
│   ├── packtok-core/
│   ├── packtok-format/
│   ├── packtok-tokenizer/
│   ├── packtok-train/
│   ├── packtok-packs/
│   ├── packtok-runtime/
│   ├── packtok-model/
│   ├── packtok-bench/
│   └── packtok-cli/
│
├── experiments/
│   ├── bpe-baseline/
│   ├── pos-packs/
│   ├── morphology-packs/
│   ├── learned-packs/
│   └── hybrid-packs/
│
├── fixtures/
│   ├── text/
│   ├── corpora/
│   └── expected/
│
├── benches/
├── docs/
└── tools/
```

Not every directory must exist in M0. Create components only when they have a real responsibility.

## Crate boundaries

### `packtok-core`

Small, dependency-light shared types and invariants.

Example concepts:

```rust
pub type PackId = u16;
pub type LocalTokenId = u32;

pub struct TokenId {
    pub pack: PackId,
    pub local: LocalTokenId,
}
```

Responsibilities:

- stable IDs and basic token types
- pack metadata contracts
- special-token definitions
- validation errors
- shared deterministic ordering rules

Must **not** contain corpus training, filesystem-heavy tooling, CLI behavior, or model code.

### `packtok-format`

Stable serialization/deserialization.

Responsibilities:

- tokenizer/vocabulary artifact format
- format versioning
- compatibility checks
- checksums where useful
- deterministic serialization
- migration helpers only when explicitly supported

Runtime loading must never require the training crate.

### `packtok-tokenizer`

Text ↔ PackTok representation.

Responsibilities:

- normalization policy
- pretokens/segmentation contracts
- local vocabulary lookup
- byte/raw fallback
- encode
- decode
- deterministic canonical representation

This crate answers **how existing text is represented**.

It must not own model inference.

### `packtok-train`

Vocabulary and pack training.

Responsibilities:

- corpus statistics
- BPE baseline training
- pack-local vocabulary training
- merge/selection algorithms
- training checkpoints/resume
- deterministic tie-breaking
- training reports

Research-heavy logic belongs here instead of leaking into runtime crates.

### `packtok-packs`

Pack definitions and routing/annotation interfaces.

Responsibilities:

- pack schemas
- pack metadata
- deterministic annotation interfaces
- offline annotation import
- experimental routing strategies

POS, morphology, language, code, numeric, and learned pack schemes should implement common contracts rather than fork the entire tokenizer.

### `packtok-runtime`

Hot-path production runtime.

Responsibilities:

- low-allocation loading
- fast encode/decode path
- packed in-memory representations
- batch APIs
- concurrency-safe immutable tokenizer state
- optional mmap/zero-copy paths when justified by benchmarks

No training dependency.

No hidden Python dependency.

### `packtok-model`

Experimental model-facing integration.

Current M3 responsibilities:

- flat token embeddings and output head for the M1 control
- independent pack-local embeddings and factorized pack/local heads for M2
- tiny CPU causal model and training/evaluation contracts
- parameter accounting, model serialization, and greedy generation smoke tests

The M3 crate depends only on `packtok-core`; tokenization, corpus training,
artifact loading, and benchmark orchestration stay outside it. PackTok remains
usable as a tokenizer/research library without forcing users into this model.

### `packtok-bench`

Shared benchmark/evaluation harness.

It should measure both tokenizer and model-facing costs.

Required metric families:

- encode MB/s
- decode MB/s
- tokens/s
- allocations
- peak memory
- artifact size
- bytes/token
- sequence-length distributions
- training throughput
- inference throughput
- validation loss
- matched-compute comparisons

Benchmark definitions are part of the research result and must be versioned.

### `packtok-cli`

Thin user/operator surface.

Possible commands:

```text
packtok train
packtok encode
packtok decode
packtok inspect
packtok benchmark
packtok compare
packtok validate
```

The CLI orchestrates library crates. Business logic should not live only in command handlers.

## Dependency direction

Preferred dependency flow:

```text
packtok-core
    ↑
packtok-format
    ↑
packtok-tokenizer
    ↑
packtok-runtime

packtok-core
    ↑
packtok-packs
    ↑
packtok-train

packtok-core ─────────────→ packtok-model

packtok-format ───────────┐
packtok-tokenizer ────────┤
packtok-train ────────────┼→ packtok-bench
packtok-packs ────────────┤
packtok-model ────────────┘
library crates ───────────→ packtok-cli
```

Avoid circular dependencies.

In particular:

- runtime must not depend on training
- core must not depend on CLI
- format must not depend on model code
- benchmarks may depend on production crates, production crates must not depend on benchmarks

### Implemented M1 dependency boundary

The current M1 workspace implements only the components needed for a flat BPE
baseline. `packtok-train` depends on `packtok-core` and `packtok-format`; it uses
`packtok-tokenizer` only as a development dependency for differential tests.
`packtok-tokenizer` depends on `packtok-core` and `packtok-format`, and has no
training dependency. The CLI and benchmark are orchestration surfaces that may
depend on those library crates. The M1 benchmark and experiment details are in
[M1_BPE.md](M1_BPE.md) and [M1_BPE_BENCHMARK.md](M1_BPE_BENCHMARK.md).
Current held-out evaluation and review verification are in
[M1_REVIEW_FIXES.md](M1_REVIEW_FIXES.md). OS memory measurement and the retained
scan comparison live inside `packtok-bench`; production crates do not depend on
either measurement helper. Synthetic training/evaluation files live in
`fixtures/benchmark/`, and raw observations are retained in `experiments/bpe-baseline/`.

## Artifact boundary

A trained tokenizer should become a self-contained versioned artifact.

Conceptually:

```text
PackTok artifact
├── format version
├── normalization policy
├── pack definitions
├── local vocabularies
├── merge/segmentation data
├── special tokens
├── byte fallback definition
└── provenance / training metadata
```

The runtime should need only this artifact plus the runtime crate.

## Rust policy

Rust is the implementation language for the project core.

Python is acceptable only for isolated tasks such as:

- one-off dataset acquisition
- plotting
- comparing against an external reference implementation
- producing temporary external annotations when no suitable Rust implementation exists

Python must not become required for:

- normal build
- tokenizer training
- encode/decode
- serialization
- benchmarks
- runtime use
- model-facing PackTok interfaces

If an experiment starts in Python because an external ecosystem tool is unavoidable, its output boundary must be explicit and reproducible.

## Performance policy

Performance is not postponed until "later".

Correctness comes first, but every stable hot path should be benchmarkable from its first real implementation.

Do not optimize by intuition alone.

Useful optimization candidates include:

- compact token representations
- contiguous local vocabularies
- cache-friendly pack tables
- mmap loading
- zero-copy reads
- SIMD
- multithreaded batch processing
- reduced allocations
- specialized lookup structures

Every optimization must preserve deterministic results unless an experiment explicitly studies nondeterminism.

## Milestone shape

### M0 — Contracts

- workspace
- core token/pack types
- artifact skeleton
- round-trip byte fallback
- deterministic tests
- benchmark harness skeleton

### M1 — Flat baseline

- byte-level BPE trainer
- encoder/decoder
- inspector
- serialization
- reproducible baseline metrics

### M2 — PackTok v0.1

- shared byte fallback and independent local BPE vocabularies
- deterministic lexical-v1 routing and one global learned-token budget
- version-3 artifact section with pack-local byte/local merge references
- exact mixed-pack runtime encode/decode and CLI inspection
- held-out comparison against the frozen M1 flat BPE control

### M3 — Tiny model comparison

- matched tiny LM baseline
- factorized pack/local input/output experiment
- validation-loss and compute comparison
- reproducible report

### M4+ — Research

Only after M0–M3 provide trustworthy baselines:

- morphology packs
- learned packs
- soft routing
- multilingual balancing
- pack-specific embeddings
- hierarchical/factorized output heads
- automatic pack discovery

## Documentation rule

If a code change alters a core invariant, artifact format, crate boundary, or experiment definition, update the relevant document in the same change.

## M2 implementation status

The implemented M2 dependency boundary adds `packtok-packs` as the owner of the
generic routing interface and the versioned `lexical-v1` experimental policy.
`packtok-train` uses it to route the training corpus and create independent
pack-local merge graphs; `packtok-tokenizer` uses it to apply the same runtime
policy. Both consume normative model types from `packtok-format`. Neither format
nor runtime depends on training. The M2 CLI and benchmark orchestrate these
library crates. Implementation behavior and measurements are documented in
[M2_PACKS.md](M2_PACKS.md) and [M2_PACKS_BENCHMARK.md](M2_PACKS_BENCHMARK.md).

M2 now implements a shared byte fallback and independent local BPE vocabularies,
deterministic `lexical-v1` routing, one global learned-token budget, a version-3
artifact section, exact mixed-pack encode/decode, inspection commands, and a
held-out comparison against frozen M1. This describes the current code and
does not imply that factorization improves tokenization.

## M3 model boundary

M3 adds `packtok-model` as the CPU-only owner of the causal recurrent reference
model, flat and pack-factorized heads, model parameter serialization, and
deterministic training/evaluation contracts. It depends on `packtok-core` for
token IDs and does not depend on tokenizer training or runtime. The
`packtok-bench` harness uses the model crate and existing tokenizers to train
both variants on the same raw splits and record comparison metrics. This keeps
model implementation separate from tokenizer and experiment orchestration.

The BRAIN space [`DasEtwa/BRAIN/PackTok`](https://github.com/DasEtwa/BRAIN/tree/main/PackTok) remains the high-level project map.
