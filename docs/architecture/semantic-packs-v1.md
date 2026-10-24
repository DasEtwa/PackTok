# Semantic Packs v1 (experimental)

Semantic Packs tests whether different lexical expressions can point to shared
concepts without changing the original text. This is an additive annotation
experiment, not a replacement tokenizer or a demonstrated compression/model gain.
The implementation uses the existing Rust workspace and no new dependency/crate.

## Three independent layers

A. **Lexical representation:** existing M0 byte fallback, M1 deterministic byte
BPE or M2 factorized BPE produces exactly the existing `TokenId` sequence. No
normalization, vocabulary row, pack/local ID or artifact version changes.

B. **Semantic representation:** `packtok-packs::semantic` owns a bounded immutable
registry and byte-span annotations. `SemanticId(u32)` is a separate namespace;
IDs never become lexical tokens. An annotation includes original byte offsets,
ordered candidate IDs, `Resolved`/`Ambiguous`, and an explicit decision reason.
A concept groups related expressions, not necessarily interchangeable synonyms.
Surface text preserves language, register, case, strength and pragmatic nuance.

C. **Context resolution:** a small explicit deterministic rule enum selects a
sense or abstains. Unknown expressions have no annotation. `resolved_id()`
returns `None` for an ambiguous annotation, even with only one known candidate:
unknown alternative senses remain possible. Candidate lists are not exhaustive.
There is no hidden language model, confidence probability or external service.

`packtok-tokenizer::encode_with_semantics(tokenizer, input, registry)` wraps the
existing `Tokenizer` trait. `None` skips recognition entirely and produces
`semantic_annotations=None`; `Some(registry)` produces `Some(annotations)` and an owned `semantic_registry` identity
(namespace, revision, resolver policy, fingerprint).
Decode uses only `lexical_tokens` with the original tokenizer. An enabled empty
annotation list is distinct from a disabled layer. This helper accepts UTF-8,
as does the existing trait; inherent raw-byte M1/M2 APIs remain unchanged and
continue to roundtrip invalid UTF-8. Semantic annotation of invalid UTF-8 is
outside v1. The wrapper can consume any implementation of the existing trait,
including trait objects. Semantic spans need not align with BPE token boundaries.

## Registry and stable IDs

Built-in namespace `packtok.semantic`, revision 1; resolver `semantic-rules-v1.1` (review fixes to the initial v1 policy).
Built-in IDs must not be renumbered or reused. Changes to inventory or rules
require an explicit revision/policy update and a new fingerprint. The built-in
registry has five concepts and fourteen patterns:

| ID | Concept | Surface patterns |
|---|---|---|
| 1 | UNCERTAINTY | idk, dunno, i don't know, don't know, keine ahnung, ka |
| 2 | HONESTY_MARKER | ngl, tbh, not gonna lie, to be honest |
| 3 | SWEARING_EMPHASIS | ong, on god |
| 4 | AGREEMENT_EMPHASIS | for real, fr |
| 5 | FRENCH_LANGUAGE | fr with explicit language metadata |

Custom registries use `SemanticRegistry::new` and must manage their own namespace,
revision and stable ID assignments. Inputs are validated and canonicalized:
unique nonzero IDs, unique uppercase names, unique lowercase ASCII patterns,
known nonduplicate candidates, bounded metadata, valid rule/candidate contracts.
Concepts sort by numeric ID, patterns lexicographically, candidates numerically;
construction order cannot change output or fingerprint. Limits: 256 concepts,
1024 patterns, 128 bytes per pattern, 64 bytes per concept name, 1024 bytes per
description, 128 ASCII bytes for a namespace, nonzero revision. No runtime
learning or dictionary expansion occurs. Custom registries are not persisted as
a new artifact: applications supply the exact registry when loading a sidecar.
Context-specific Fr/Ka rules reserve the documented built-in concept numbers;
other custom concepts can use the context-free rules and multiple candidates.

Fingerprints are length-prefixed FNV-1a over namespace, revision, policy ID and all
sorted definitions (including descriptions/rules/candidates). They detect stale
data; they provide no cryptographic authentication. Built-in initialization uses
`OnceLock`; matching reads immutable data and has no random/global mutable state.

## Recognition and conservative resolution

Recognition is left-to-right, longest non-overlapping whole-phrase matching.
Patterns use ASCII case-insensitive letters. ASCII apostrophes also match U+2019;
spaces in a phrase match one or more horizontal Unicode whitespace characters.
Neither operation rewrites original bytes. VT, FF, CR, LF, NEL, U+2028 and U+2029 cannot
join phrase words or contextual cues. Unicode letters, digits, combining marks
and format controls conservatively attach to neighboring words. ASCII underscore
and intra-word apostrophes attach too; opening/closing single quotes can delimit
expressions while possessives such as `idk’s` cannot match `idk`. Selected ranges
U+2000–206F, U+2E00–2E7F, U+3000–303F, U+2600–27BF and U+1F000–1FAFF
can delimit expressions, with explicit exclusions for format controls. Other
non-ASCII punctuation/symbols conservatively attach: for example `«idk»` and
`idk，` are not currently recognized.
This is a documented small boundary heuristic, not full Unicode segmentation.

- Known direct variants resolve a single concept. Multiple candidates stay
  ambiguous. The oath abbreviation `ong` resolves only in its lowercase form;
  `ONG` is ambiguous. Direct variants such as `IDK` permit ASCII uppercase.
- Bare `fr` offers concepts 4 and 5 and remains ambiguous. An immediate adjacent
  `bro` separated by horizontal whitespace permits lowercase `fr` as concept 4.
  Uppercase/mixed-case variants remain ambiguous without a language label.
- A local `language`, `lang` or `sprache` key followed by `:` or `=` permits `fr`
  as concept 5 in any case; simple surrounding quotes are accepted. A conflicting
  adjacent slang cue yields `ConflictingContext`, not a priority-based guess.
- Lowercase `ka` resolves concept 1 only next to preceding `ich` or following
  `was`, `wie`, `warum`, `ob`, `wer`, `wo`, `wann` across horizontal whitespace.
  Other `ka` occurrences stay ambiguous; uppercase/mixed-case `KA` always abstains.
- Context inspection is bounded to 128 characters per side and never crosses a
  line separator. Distant context and external language hints are not inferred.

These rules are falsifiable heuristics. Quotation/metalinguistic mentions,
negation, code, names and out-of-domain abbreviations can create false positives.
`I don't know` is a non-knowledge expression but its scope/subject is not modeled;
`ngl` and `tbh` share a discourse category without erasing their difference.
No claim is made that this registry covers German/English in general.

## Independent versioned sidecar

Persistence belongs to existing `packtok-format::semantic`, since format already
depends on packs. Existing tokenizer artifact versions 1/2/3 are unchanged.
`SemanticSidecar::annotate`, `to_bytes`, `from_bytes` and `annotations` expose an
optional separate artifact; the CLI shows debug annotations without requiring
persistence. Text recovery requires no sidecar or semantic registry.

All fields are little endian. Header is 40 bytes:

| Field | Bytes | Value |
|---|---|---|
| magic | 8 | `SPACKS\0\0` |
| version | 2 | 1 |
| reserved flags | 2 | 0 |
| registry fingerprint | 8 | definitions and resolver policy |
| original input length | 8 | bytes |
| original input fingerprint | 8 | exact-byte FNV-1a |
| annotation count | 4 | record count |

Each record: start/end `u64` (inclusive/exclusive original byte offsets), decision
`u8`, reason `u8`, candidate count `u16`, followed by `u32` concept IDs.
Decision codes: resolved=1, ambiguous=2. Reason codes: KnownVariant=1,
SlangContext=2, LanguageContext=3, InsufficientContext=4, CaseAmbiguity=5,
ConflictingContext=6. A resolved one-candidate record is 24 bytes, unresolved
`fr` with two candidates 28 bytes. No lexical stream or original text is copied
into the sidecar. Empty sidecars still have a 40-byte header.

Accepted/emitted sidecars are capped at 16 MiB. Parsing rejects unsupported
versions/flags, truncation/trailing bytes, excessive counts, invalid UTF-8 span
boundaries, overlap, unknown/duplicate/unsorted IDs and mismatched input/registry.
It checks record sizes before allocation, then regenerates annotations to verify
canonical contextual decisions. Consequently loading requires original UTF-8 and
the exact registry; it is not a shortcut around recognition. No tokenizer hash is
needed: offsets bind to original bytes independently of token segmentation.
FNV is not suitable for untrusted authenticity; use external cryptographic
integrity if an application needs it. Encoding very large text remains possible
without persistence; serialization over 16 MiB returns an explicit error.

## CLI and example

From repository root:

```sh
cargo +1.85.0 run -p packtok-cli -- semantic 'ong idk bro'
cargo +1.85.0 run -p packtok-cli -- semantic --artifact PATH 'language: fr'
```

The separate debug command prints lexical `pack:local` IDs, registry namespace,
revision, resolver/fingerprint, original quoted spans, candidates and reasons.
Existing `encode`/`decode` CLI output remains unchanged. Pass text as one quoted
argument to preserve whitespace exactly (as with existing CLI argument joining).

For M0 fallback, `ong idk bro` gives lexical byte values
`[111,110,103,32,105,100,107,32,98,114,111]` in pack 65535. Its annotations are
`[0,3) -> 3 SWEARING_EMPHASIS (KnownVariant)` and
`[4,7) -> 1 UNCERTAINTY (KnownVariant)`. `bro` is unchanged/unannotated.
For M1/M2 the lexical segmentation comes exclusively from the supplied artifact.
All three paths decode to exactly `ong idk bro`.

## Experiment, validation and v2

See [development corpus](../../fixtures/semantic-v1/README.md),
[experiment results](../../experiments/semantic-packs-v1/REPORT.md) and
[review/fix log](../../experiments/semantic-packs-v1/REVIEWS.md).
The matched benchmark uses the same frozen-training-input M1 BPE artifact for A
and B; B adds annotation. Gold development cases do not train the tokenizer.
It records raw timing trials, byte/capacity overhead and exact-span/concept
accuracy. It does not train neural models: no model-facing semantic adapter is
part of v1, and an unconsumed sidecar cannot establish a learning improvement.
Alternative internal representations were considered: replacing BPE IDs would
break lexical comparability/losslessness, while dense per-token labels would
couple semantics to segmentation and waste space on unannotated text. Sparse byte
spans keep these responsibilities independent; binary headers dominate tiny cases.

V2 should first freeze a larger independently labeled challenge set, expand
abstention/suppression rules based on measured errors, and evaluate a bounded
model input adapter under matched compute/parameters/splits. Any compressed or
streaming sidecar should be justified by measured overhead, preserve exact
byte offsets and version incompatible formats. Semantic predictions must never
be required for lossless text recovery.
