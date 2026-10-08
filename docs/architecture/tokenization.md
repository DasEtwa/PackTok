# Tokenization contracts

M0 uses one raw token per byte. M1 learns flat byte-level BPE merges with
deterministic tie-breaking and serialization. M2 routes spans using frozen
lexical-v1 rules and learns independent local BPE graphs under one global learned
budget. Its byte rows occur once in BYTE_FALLBACK, rather than being duplicated
in every pack.

`decode(encode(bytes)) == bytes` is mandatory, including empty input, malformed
UTF-8 and arbitrary bytes. There is no implicit Unicode normalization or UNK
requirement. Token limits and malformed-artifact rejection are explicit contracts.

Normative details and reference/differential paths: [M1](../../M1_BPE.md),
[M2](../../M2_PACKS.md), [format](../../FORMAT.md). Measurements and validation
must be read with the [research overview](../research/results.md).
