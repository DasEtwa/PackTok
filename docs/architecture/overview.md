# Implemented architecture

The root Rust workspace contains eight crates. `packtok-core` owns identifiers
and contracts; `packtok-format` owns bounded/versioned serialization;
`packtok-packs` owns the router contract and lexical-v1 policy;
`packtok-train` owns deterministic tokenizer training; `packtok-tokenizer` owns
artifact-based encode/decode. `packtok-model` owns the tiny CPU recurrent model
and generic ID bijections. `packtok-cli` and `packtok-bench` orchestrate libraries.
The runtime does not depend on training. No Python dependency enters these crates.

`experiments/m5-gpu/` is an isolated Rust workspace for the causal Transformer,
data adapters, GPU checks and external lifecycle. Candle 0.9.1 and optional CUDA
are confined there. Its configuration is frozen; maintenance changes transport,
supervision and documentation only.

[STRUCTURE.md](../../STRUCTURE.md) remains the crate-boundary authority and also
contains prospective components, such as a dedicated runtime crate, that are not
currently implemented. [IDEA.md](../../IDEA.md) is the research motivation.
See [tokenization](tokenization.md), [packs](packs.md) and [formats](formats.md).
