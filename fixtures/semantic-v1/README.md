# Semantic Packs v1 development corpus

`eval.tsv` is a small, original, agent-authored synthetic development corpus with
explicitly specified gold labels,
created for this experiment on 2026-10-23 under this repository's Apache-2.0
license. It includes English/German phrases, slang, ambiguous codes, normal
prose, unknown expressions, emoji, German Unicode, word boundaries, horizontal
whitespace, CRLF, NUL and local context changes.

This is **development gold**, not a statistically representative or blind held-out
benchmark. Its labels instantiate v1's documented conservative policy. The
registry/rules are designed with these examples in view. Accuracy here does not
establish natural-language accuracy, generalization, model quality or learning
speed. The BPE trainer consumes only `fixtures/benchmark/train.txt`; it never
trains on this fixture or its gold labels. No neural model is trained.

Each non-comment row has four tab-separated columns:

1. Unique case ID.
2. Category, used for descriptive reporting; `unknown` explicitly requires no
   recognized expression.
3. Ordered non-overlapping byte spans `start:end:concept_id`, separated by `;`.
   `start:end:?` requires an ambiguous decision and no resolved sense. `-` means
   there are no recognized expressions. Spans refer to the decoded UTF-8 text.
4. Original text. Only `\\`, `\n`, `\r`, `\t`, `\0`, `\v` (U+000B vertical tab), and `\f`
   (U+000C form feed) escapes are supported. Other
   Unicode and spaces, including U+00A0 NBSP, are literal and preserved.

IDs are independent of BPE token IDs: 1 UNCERTAINTY, 2 HONESTY_MARKER,
3 SWEARING_EMPHASIS, 4 AGREEMENT_EMPHASIS, 5 FRENCH_LANGUAGE. The case source
preserves original expressions even when the same concept is shared.

Reproduce from the repository root with Rust 1.85:

```sh
cargo +1.85.0 run --release -p packtok-bench -- semantic unique-run-label
```

Results go to `experiments/semantic-packs-v1/unique-run-label/`, and existing
run directories are never overwritten. The run records toolchain, source hashes,
artifact bytes/hash, corpus hashes, gold/actual decisions, all raw timing trials,
vector-capacity memory estimates and process peak RSS. A failed run retains its
provenance and failure text. A/B reuse the exact same immutable flat BPE artifact
(default target 512 IDs), raw input and lexical decode implementation. A encodes
only lexical tokens; B encodes those same tokens plus semantic annotations.
Sidecar serialization/deserialization is verified and sized but excluded from
encoding timings. Decoding measures lexical byte reconstruction in both variants;
no semantic reconstruction or sidecar validation is needed to recover the text.

The benchmark checks each fixture record separately, then times a batch consisting
of newline-joined original texts repeated 64 times. Seven trials alternate A/B
order. Each direction receives a 25 ms warmup and at least 100 ms measurement.
Medians and minimum/maximum are descriptive observations on the recorded machine,
not confidence intervals. Times exclude file I/O, BPE training and artifact loading.
The whole-process peak RSS includes training and both variants and cannot isolate
semantic overhead. Output capacities count annotation/candidate vectors separately
from the owned registry-identity namespace String. Capacity sizes exclude outer
stack structs, allocator metadata, immutable registry and runtime temporary
allocations; they are not a peak-memory measurement.

Exact-span precision/recall evaluates expression detection; resolved precision/
recall requires an exact byte span and correct concept ID. Ambiguity errors count
an erroneous resolved decision on a gold-ambiguous span. Unknown-case annotations
are counted explicitly. All per-case decisions remain in the raw results, even
when they disagree with gold. The parser and metric-accounting tests cover fixture
validation and ensure that a wrong concept receives no accuracy credit.

`review-regressions.tsv` preserves seven additional gold cases from independent
reviews: bounded-context prefix truncation, vertical tab/form-feed context
barriers, quoted language keys/values, and apostrophe word boundaries. These cases
are validated by an automated annotation/gold test. They are kept separate from
the original 33-case timing corpus so its input bytes and initial timing record
remain comparable. They share the parser and explicit gold format described above.
The timed report records the semantic registry namespace, revision, resolver policy
and fingerprint as well as the independent sidecar format version. Registry
identity is included in pipeline determinism and disabled-mode checks.

The existing BPE `encode_bytes_with_stats` and `decode_bytes_with_stats` paths run
once outside timing, with differential checks against ordinary lexical results.
The report includes their vector allocation/growth requests and peak simultaneous
working/output buffer capacities. These measure the shared lexical operation in
A/B and exclude additional semantic allocations, immutable tokenizer/registry,
caller input and allocator metadata. Semantic output capacity observations are
reported separately; there is no claim to count all annotation allocation calls.
