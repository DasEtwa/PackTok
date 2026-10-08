# Pack representation and model IDs

A canonical PackTok token carries a numeric pack ID and local ID. Pack names are
metadata. The current lexical-v1 router is a context-free byte heuristic for
TEXT, NUMBER and STRUCTURE; BYTE_FALLBACK covers every raw byte. It does not
infer contextual grammatical roles.

M2 trains independent local graphs with a shared byte base. M3 consumes local
embeddings and pack/local outputs; its training loss is pack cross-entropy plus
local cross-entropy under the gold pack. Greedy pack-first decoding is not a
search for the globally most probable joint token.

M4 separates representation from prediction. C flattens M2 IDs bijectively in
ascending numeric pack/local order into a flat model namespace. B constructs
synthetic groups from frozen M1 artifact expansions. No token is rerouted,
renormalized or deduplicated by these model-side maps. M5 compares A and C with
the same flat Transformer head.

Specifications: [M2](../../M2_PACKS.md), [M3](../../M3_MODEL.md),
[M4](../../M4_ABLATION.md), [M5](../../M5_GPU_TRANSFORMER.md).
