# Independent architecture and compatibility review

Reviewer: independent Agent 2, 2026-10-23. Reviewed the first complete uncommitted
Semantic Packs v1 implementation on `feat/semantic-packs-v1`, based on original
`2423342`. No other reviewers' reports were consulted. No production code, branch
or commit was changed by this reviewer. Documentation was still being authored
by the implementation agent; an absent final report was not treated as a finding.

## Scope and favorable checks

Read AGENTS.md, IDEA.md, STRUCTURE.md and README.md, the complete additive diff,
new registry/resolver, tokenizer wrapper, sidecar parser/serializer, CLI and its
integration tests, benchmark/corpus and original tokenizer/format boundaries.

- No existing lexical encoder/decoder implementation was changed. The existing
  Tokenizer trait is unchanged and `None` in `encode_with_semantics` performs no
  registry matching. UTF-8 wrapper scope is explicit; raw-byte APIs remain.
- Semantic IDs are a distinct newtype rather than lexical TokenId values. BPE
  vocabulary size stays 512 in the matched experiment and there is no model-ID
  remapping. M0/M1/M2 CLI integration tests exercise actual trained artifacts.
- Existing artifact versions 1/2/3 and their serialization were untouched. The
  new sidecar has independent magic/version, explicit wire codes, bounded counts,
  canonical ordering, input/registry fingerprints and regenerated annotations.
- Registry definitions and annotation belong in packs, serialization in format,
  additive integration in tokenizer, orchestration in CLI/bench. No crate,
  dependency, training-runtime dependency or dependency cycle was introduced.
- Registry sizes are bounded. Definitions are immutable after validation.
  Fingerprints include namespace, revision, policy, all concept metadata and
  sorted variants/candidates; insertion order is irrelevant.
- Benchmark A/B share exact bytes, artifact and lexical IDs. A/B order alternates,
  serialization is excluded explicitly, lexical decode is correctly identical,
  memory capacities and whole-process RSS have honest scope labels. Development
  gold is explicitly not blind held-out evidence. No learning/compression or
  general superiority claim is made. No neural model was run.

## Findings

### ARCH-1 — medium: normal pipeline output loses semantic registry identity

Location at review: `crates/packtok-tokenizer/src/semantic.rs:12-17` and output
construction at 32-37.

`SemanticId` is documented as local to a registry namespace/revision, and the
public wrapper accepts arbitrary validated custom registries. However,
SemanticEncoding stores only lexical IDs and semantic annotations. It drops
namespace, revision, resolver ID and registry fingerprint. Thus equal output
objects can mean different things when composed or retained without their caller's
registry object.

Reproduced with a temporary independent Cargo program depending on packs and
tokenizer by path:

1. Encode `idk` with `SemanticRegistry::builtin()` (id 1 means UNCERTAINTY).
2. Construct registry `custom`, revision 1 with concept id 1 named HONESTY_MARKER,
   a Direct `idk` variant referring to id 1, and encode the same text.
3. `a == b` is true, despite different namespaces and concept meanings.

Observed stdout: `different namespace/concept -> identical pipeline output: true`.
The persisted SemanticSidecar does retain a fingerprint and rejects mismatched
registries, so the problem is confined to the normal wrapper result and clients
which transport/aggregate it. It does not corrupt lexical roundtrip.

Recommended change: attach an optional registry identity/descriptor or fingerprint
to SemanticEncoding whenever semantic annotations are enabled; test differing
namespaces, revisions and definitions. At minimum, make the external identity
requirement explicit if intentionally leaving a context-dependent result type.

### ARCH-2 — medium: LowercaseOnly violates flexible-whitespace variant matching

Location at review: `crates/packtok-packs/src/semantic.rs:512-513`.

The public registry supports multiword patterns with flexible horizontal Unicode
whitespace and apostrophe equivalence. LowercaseOnly then compares the complete
original surface byte string against the pattern, so an orthographic whitespace
variation is reported as CaseAmbiguity even though every letter is lowercase.

Reproduction: custom registry containing a `not gonna lie` variant, candidate
HONESTY_MARKER, rule LowercaseOnly. Results:

```text
"not gonna lie"  -> Resolved, KnownVariant
"not\tgonna lie" -> Ambiguous, CaseAmbiguity
"not  gonna lie" -> Ambiguous, CaseAmbiguity
"Not gonna lie"  -> Ambiguous, CaseAmbiguity
```

Expected: first three resolve; only the uppercase spelling requires case
ambiguity. Built-in LowercaseOnly currently applies only to `ong`, so built-in
results are unaffected. This matters for the promised validated custom-registry
extension surface.

Recommended change: inspect letter case independently of normalized whitespace
and equivalent apostrophes after matching. Add registry regression tests for tabs,
NBSP, repeated spaces and curly apostrophes under this rule.

### ARCH-3 — low: single-quote metadata support is unreachable at lexical boundaries

Locations at review: `word_character` at packs semantic.rs:361-372 and
`language_label` at 475-501.

`language_label` explicitly accepts single as well as double quotes, but the
matcher considers every adjacent ASCII apostrophe part of a word, so the resolver
is never reached for a single-quoted value. Existing debug CLI reproduction:

```text
packtok semantic "language: 'fr'"  -> Semantic annotations: 0
packtok semantic '"language": "fr"' -> fr resolves FRENCH_LANGUAGE
packtok semantic "'idk'"          -> Semantic annotations: 0
packtok semantic '"idk"'          -> idk resolves UNCERTAINTY
```

The conservative contraction/possessive guard is useful (`idk's` should not match),
but it should distinguish a wrapping quote from an apostrophe inside a word if
single-quoted metadata is intended. Either add that distinction with regressions
or explicitly document/exclude single-quoted metadata instead of claiming quote
acceptance generally. This is a recognition coverage issue, not lexical loss.

## Verification and remaining limits

Temporary reproduction project: `/tmp/packtok-architecture-probe`; compiled and
ran with Rust 1.85.0 using the task-local rustup installation. CLI probes used the
available debug binary. `git diff --check` passed. Existing baseline and new
integration tests were inspected; this review did not repeat the entire workspace
suite while the primary agent was coordinating measurements.

No critical/high compatibility defect, old-format change, old-API change,
lexical-ID collision or unconditional semantic matching was found. FNV is
explicitly noncryptographic, sidecar loading intentionally re-annotates, and
natural-language false positives/negatives remain expected heuristic limitations.
These limitations are not presented as proven model improvements.
