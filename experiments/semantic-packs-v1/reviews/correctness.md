# Independent correctness/Rust review — Semantic Packs v1

Reviewer: separate `review_correctness` agent, 2026-10-23. Reviewed the first
complete implementation on `feat/semantic-packs-v1` against original base
`2423342`, before root fixes. Did not read other reviewers' findings or modify
production code. Scope: registry validation/matching/context rules, tokenizer
wrapper, semantic sidecar parser/serializer, CLI and differential tests.

## Findings

### C1 — Medium: bounded language-key scan invents a valid key boundary

Location at review: `crates/packtok-packs/src/semantic.rs:478` and `:498`.

Repro:

```rust
let text = format!("xlanguage:{}fr", " ".repeat(119));
let result = SemanticRegistry::builtin().annotate(&text);
assert_eq!(result[0].resolved_id(), Some(SemanticId(5))); // observed, incorrect
```

Expected: unresolved `fr`, because `xlanguage` is not a supported language key.
With 118 spaces the same expression correctly remains unresolved. The reversed
iterator's `.take(128)` ends exactly before the leading `x`; the final boundary
check treats budget exhaustion as the actual beginning of input and accepts the
suffix `language` as a full key. This violates the explicit metadata-key boundary
contract and the conservative resolution requirement. Reject exhausted context,
or inspect the real preceding character without losing the bound.

### C2 — Medium: vertical tab and form feed act as horizontal context

Location at review: `crates/packtok-packs/src/semantic.rs:357`.

`fr\u{000b}bro` and `fr\u{000c}bro` both resolve `fr` to ID 4 with
`SlangContext`. Both controls are vertical separators. The helper named and
specified as horizontal whitespace accepts them through `char::is_whitespace()`.
This can infer slang context across line/page boundaries. The same helper also
allows these controls inside multiword patterns. Exclude VT and FF along with
other vertical separators; add explicit regression cases.

### C3 — Low: advertised single-quoted metadata is unreachable

Location at review: `crates/packtok-packs/src/semantic.rs:363`, `:475`.

`language: 'fr'` produces no annotation; `'language': "fr"` produces ambiguous
`fr`, while `"language": "fr"` correctly resolves ID 5. The context helper
explicitly accepts single quotes, but the whole-word boundary helper classifies
apostrophes as word characters. This blocks the value entirely and also rejects
the key's opening quote. Either support balanced single quotes while preserving
contraction boundaries, or document/narrow quote support to double quotes. This
is a false negative/interface inconsistency, not lexical data loss.

## Checks and non-findings

- 4,000 generated UTF-8 inputs spanning ASCII, separators, NUL, combining marks,
  apostrophes, emoji and joiners passed deterministic ordering, valid/non-overlap
  byte spans, registered candidate IDs and canonical sidecar roundtrips. Probe
  source and observed output are retained alongside this report.
- Reviewed all indexing and `expect` calls in production semantic modules:
  positions advance by UTF-8 scalar lengths, parser reads use checked arithmetic
  and checked slices, and registry validation short-circuits empty patterns before
  indexing. No reproduced panic, invalid index or broken lexical roundtrip.
- Sidecar parsing checks lengths before reservation, rejects malformed ordering,
  unknown IDs/decisions/reasons/version/flags/trailing bytes, and regenerates
  annotations to reject fabricated semantic decisions.
- Lexical decoding uses original tokens only. Optional-off behavior bypasses
  registry work and returns baseline lexical IDs. Arbitrary invalid UTF-8 remains
  supported by pre-existing BPE byte APIs; semantic matching intentionally accepts
  only UTF-8 text, matching the existing `Tokenizer::encode` trait.
- Normal expression ambiguity outside the small documented registry is a known
  heuristic limitation; it is not evidence of probabilistic language understanding.
- Allocation failure in infallible annotation vectors can still abort like normal
  Rust allocation; the 16 MiB serialization limit is not a limit on input length or
  peak annotation memory. No extra allocation guarantee is claimed by these APIs.

Targeted crate-test result is retained in `correctness-tests.txt`. Reproductions
ran with Rust 1.85.0 in a temporary Cargo package depending by path on
`packtok-packs` and `packtok-format`; no production source or branch was changed.
