# PackTok — Experimental Factorized Tokenization in Rust

PackTok investigates whether structured token spaces and pack-based representations
can improve language-model efficiency, quality or extensibility. It provides a
deterministic byte-level BPE control and an experimental tokenizer with independent
pack vocabularies. Every byte remains representable, with exact encode/decode
round-trips and no implicit normalization.

The project is a research implementation. Current results support narrow,
corpus-dependent observations; they do not establish general superiority over
flat BPE, Unigram or byte-level modeling. M0–M4 have reproducible evidence. M5's
first paired Transformer pilot completed 2,000 updates for A and C on one seed;
Flat BPE had validation 2.572965 BPB and PackTok 2.597469 BPB. This exploratory
result does not establish statistical significance. See the [M5 pilot record](experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T154700Z/PILOT_RESULT.md)
and [M6 research design](experiments/m6/EXPERIMENT_DESIGN.md).

## Research motivation

A conventional flat BPE vocabulary assigns every learned token one global ID.
PackTok explores a representation in which a token has a `(pack_id, local_id)`
pair. Specialized vocabularies share a complete raw-byte fallback. Pack names
are metadata; grammatical categories are not built into the core.

The implemented `lexical-v1` experiment routes spans using context-free byte
rules into TEXT, NUMBER and STRUCTURE packs. It is not a contextual POS tagger.
Tokenizer representation and model prediction are separate: pack/local IDs can
be mapped bijectively to global model IDs for a flat output head, or consumed by
a model that predicts a pack and then a local token. A pack representation does
not require a factorized model head.

## Architecture

```text
raw bytes
   ├── M1 flat BPE ──────────────── global token IDs ── flat model output
   └── M2 lexical-v1 + local BPE ── pack/local IDs
                                      ├── bijective global mapping ── flat output
                                      └── pack/local embeddings and output heads
```

The Rust workspace separates contracts (`packtok-core`), serialization
(`packtok-format`), routing (`packtok-packs`), training (`packtok-train`),
encoding/decoding (`packtok-tokenizer`), CPU models (`packtok-model`), CLI
(`packtok-cli`) and evaluation (`packtok-bench`). Training is not a tokenizer
runtime dependency. There is no separate implemented `packtok-runtime` crate.

The M5 Transformer lives in its own Cargo workspace under `experiments/m5-gpu/`.
Its pinned Candle dependency and optional CUDA feature do not enter the root
workspace. Python is confined to external transport and maintenance tools; Rust
owns tokenizer/model training, runtime and benchmark computation. See
[architecture overview](docs/architecture/overview.md) and [STRUCTURE.md](STRUCTURE.md).

## Research milestones

| Milestone | Implemented question and current evidence |
|---|---|
| [M0](docs/milestones/M0.md) | Token/pack contracts, versioned artifacts, deterministic byte fallback and exact round-trips. |
| [M1](docs/milestones/M1.md) | Flat byte-level BPE control with reference paths, held-out tokenizer benchmarks and audits. |
| [M2](docs/milestones/M2.md) | Independent lexical-v1 pack vocabularies with shared bytes; compression and runtime trade-offs against M1. |
| [M3](docs/milestones/M3.md) | Tiny CPU RNN comparison changes both tokenizer and output head; a synthetic-corpus signal motivated ablation. |
| [M4](docs/milestones/M4.md) | A/B/C/D ablation on a fixed sourced mixture. The tokenizer contributed a small tested quality difference. Generic output factorization did not improve same-schedule quality; reduced analytical output work allowed more updates under matched MACs. A strong favorable interaction was not established. |
| [M5](docs/milestones/M5.md) | Same flat Transformer architecture for M1 versus flattened M2 IDs. The exploratory paired A/C pilot completed 2,000 updates each; Flat BPE led by 0.024504 BPB on validation for one seed. See the complete report and limitations. |

[Research results](docs/research/results.md) explains each hypothesis, experiment,
measurement, interpretation and limitation, with links to original reports and
raw records. Negative and inconclusive evidence is retained.

## Getting started

Install Rust using the [official rustup instructions](https://rustup.rs/).
The root workspace declares Rust **1.85** as its minimum version and uses edition
2024. From the repository root:

```sh
rustup toolchain install 1.85.0 --profile minimal --component rustfmt --component clippy
cargo +1.85.0 build --release --workspace
cargo +1.85.0 test --workspace
cargo +1.85.0 run --release -p packtok-cli -- help
```

A byte-fallback example requires no trained artifact:

```sh
cargo +1.85.0 run --release -p packtok-cli -- encode "Hi"
# 65535:72 65535:105
cargo +1.85.0 run --release -p packtok-cli -- decode "65535:72 65535:105"
# Hi (raw decoded bytes, without an added newline)
```

Train and inspect a small BPE artifact (choose a new output path; existing files
are protected from overwrite):

```sh
cargo +1.85.0 run --release -p packtok-cli -- train-bpe fixtures/m1_bpe_corpus.txt target/example.packtok
cargo +1.85.0 run --release -p packtok-cli -- inspect target/example.packtok
cargo +1.85.0 run --release -p packtok-cli -- encode --artifact target/example.packtok "Sämtliche Häuser"
cargo +1.85.0 run --release -p packtok-cli -- validate target/example.packtok
```

Decode the emitted numeric sequence using `decode --artifact target/example.packtok
"<IDs from encode>"`. For pack training and inspection, see the
[tested CLI examples](docs/development/building.md). Example verification is
recorded in [maintenance evidence](docs/maintenance/2026-10-08/REPORT.md).

## Reproducibility

Protocols fix raw corpus/splits, normalization policy, deterministic tie-breaking,
IDs, seeds, model configurations and measurement boundaries. Byte-normalized loss
is used for cross-tokenizer quality; token perplexity is not interchangeable
across different token sequences. Analytical MAC matching is not measured GPU
compute or equal wall time.

- [Artifact formats and compatibility](FORMAT.md)
- [M1 held-out methodology](M1_REVIEW_FIXES.md) and [synthetic fixture provenance](fixtures/benchmark/README.md)
- [M3 protocol](M3_MODEL.md) and [original results](M3_MODEL_BENCHMARK.md)
- [M4 sources, splits and complete results](M4_ABLATION_BENCHMARK.md)
- [M5 CPU provenance](experiments/m5-gpu/provenance/CPU_PREPARATION.md), [frozen protocol](M5_GPU_TRANSFORMER.md) and [failure/accounting record](M5_GPU_BENCHMARK.md)

Historical experiment paths, raw sources, failed runs and numerical measurements
remain intact. Large M5 datasets, CUDA bundles and weights stay outside Git;
[storage and recovery](docs/development/storage-and-recovery.md) describes integrity
manifests, protected credentials and the limits of weights-only checkpoints.

## Documentation

Start with the [documentation index](docs/README.md), [results overview](docs/research/results.md),
[building](docs/development/building.md), [testing](docs/development/testing.md) or
[WSL/Colab operations](docs/development/gpu-colab.md). The [changelog](CHANGELOG.md)
tracks milestones. [IDEA.md](IDEA.md), [STRUCTURE.md](STRUCTURE.md) and
[AGENTS.md](AGENTS.md) retain their canonical roles; the high-level project map is
[DasEtwa/BRAIN/PackTok](https://github.com/DasEtwa/BRAIN/tree/main/PackTok).

## Roadmap

M6 is analyzing tokenizer efficiency and designing controlled follow-up
Transformer experiments; this phase does not authorize GPU training. Exact training resume
also needs optimizer, RNG and cursor state; current safetensors store weights only.

Future hypotheses include improved pack routing, learned pack allocations,
adaptive vocabulary, larger model comparisons, and specialized language or
reasoning packs. These ideas are not implemented or proven by appearing here.
Any new experiment must keep byte fallback, controlled comparisons and preserved
negative results. There is no automatic GPU retry or M6 launch.

## License and contributions

PackTok code is licensed under [Apache-2.0](LICENSE). Retained third-party corpus
sources keep their original notices and licensing scope; see each provenance
record. Contributions should propose a small falsifiable experiment or a measured
correctness/performance improvement, document relevant facts, and pass the
[verification gates](docs/development/testing.md). Read
[contribution guidance](docs/development/contributing.md) and [AGENTS.md](AGENTS.md)
before changing frozen experiments.
