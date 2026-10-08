# Artifact formats and compatibility

[FORMAT.md](../../FORMAT.md) is the normative specification for tokenizer formats
v1/v2/v3, CPU model parameters and bounded ID mappings. This navigation page does
not redefine wire formats or change IDs. Canonical parse/serialize bytes and
historical artifacts are regression-tested.

M5 uses FP32 safetensors plus external frozen configuration and identity manifests.
These contain weights only: they omit AdamW moments, sampling RNG and data cursor.
They support inference/reload checks, not exact interrupted-training resumption.
See [M5 protocol](../../M5_GPU_TRANSFORMER.md) and
[storage recovery](../development/storage-and-recovery.md).
