# Artifact formats and compatibility

[FORMAT.md](../../FORMAT.md) is the normative specification for tokenizer formats
v1/v2/v3, CPU model parameters and bounded ID mappings. This navigation page does
not redefine wire formats or change IDs. Canonical parse/serialize bytes and
historical artifacts are regression-tested.

The original M5 FP32 weights-only safetensors support initialization and inference.
The extended M5 runner stores model parameters, AdamW moments and continuation
metadata in v1 resume safetensors. M6 adds checksummed, streamed v2 training
images with immutable generations and journal verification.
See the [v2 format and compatibility table](../milestones/M6.md).
See [M5 protocol](../../M5_GPU_TRANSFORMER.md) and
[storage recovery](../development/storage-and-recovery.md).
