# Independent adversarial and semantic review

Reviewed 2026-10-23 by the third independent reviewer against the initial complete
implementation on `feat/semantic-packs-v1` (before root fixes). This reviewer did
not consult the other review reports. No production files or branches were
modified. Temporary reproduction integration tests were removed after execution.

Scope: semantic registry validation, matching/context and byte spans, optional
pipeline semantics, binary sidecar parser; English/German, case, emoji, combining
marks/zero-width controls, Unicode line breaks, hostile whitespace and very large
inputs. Required repository instructions and architecture/idea/readme were read.

## Findings

### ADV-1 — Medium: bounded metadata lookback invents a valid key boundary

Location: `crates/packtok-packs/src/semantic.rs`, `language_label`, initial lines
476–501, especially `.take(128)` and the final `chars.peek()` boundary check.

Reproduction:

```rust
let r = SemanticRegistry::builtin();
let text = format!("xlanguage:{}fr", " ".repeat(119));
let a = r.annotate(&text);
assert_eq!(a[0].resolved_id(), Some(FRENCH_LANGUAGE)); // actual initial output
```

The real key is `xlanguage`, so an explicit `language` key was not present. The
119 spaces, colon and eight letters exhaust the 128-character iterator exactly;
`peek()` reports no predecessor although `x` is still present in the original
input. The shortened `xlanguage: fr` correctly remains ambiguous. Equivalent
examples: `xlang:` + 123 spaces + `fr`; `xsprache:` + 120 spaces + `fr`.

Expected: remain ambiguous whenever the bounded lookback cannot prove the complete
key and its left boundary. Reject a context truncated at that boundary, or peek
the original predecessor without including it in the bounded parse.

Impact: an adversarial but small input changes an explicitly rejected key into a
resolved language meaning, violating conservative context resolution. No lexical
bytes are lost, and the binary sidecar faithfully persists the incorrect semantic
decision, so canonical re-annotation alone cannot detect the underlying rule bug.

### ADV-2 — Medium: vertical whitespace is treated as horizontal context

Location: same module, `horizontal_space`, initial lines 357–359.

Reproductions: `fr\u{000b}bro`, `fr\u{000c}bro` initially resolve to concept 4;
`ka\u{000b}was` initially resolves to concept 1. `lang\u{000c}: fr` initially
resolves to concept 5. U+000B is vertical tab and U+000C is form feed. Both satisfy
Rust `is_whitespace()` and neither is excluded.

Expected: treat VT/FF as vertical context boundaries, like CR/LF/NEL/LS/PS, under
the documented horizontal-only matching/context rule. Phrase patterns likewise
must not match across these separators (e.g. `for\u{000c}real`).

Impact: context leaks between vertical records/pages and can invent a unique
sense. This is a semantic correctness bug, not a byte roundtrip issue.

### ADV-3 — Low: curated punctuation boundaries are narrower than the comment

Location: same module, `word_character`, initial lines 361–372.

Reproductions: `«idk»`, `idk，` (U+FF0C fullwidth comma), and `idk⌨` (U+2328 keyboard)
receive no annotation, whereas `"idk"` and `idk。` do. Single ASCII quotes also
attach to words, so `'idk'` is not recognized. The comment states “Unicode
punctuation and emoji are boundaries”, but only selected ranges are boundaries.

Expected: either cover the intended common punctuation/symbol boundaries with
regressions, or explicitly describe this as a curated conservative boundary
policy with known false negatives. This is not severe: false negatives preserve
bytes and avoid fabricated meanings. A Unicode-property dependency is not required
for v1 if the limited policy is documented accurately.

## Known heuristic limitation, not a new correctness defect

`NGL prices increased` resolves NGL to HONESTY_MARKER although NGL can mean natural
gas liquids in an industry context. Direct aliases intentionally have no domain
understanding, so lexical presence is not proof of intended discourse meaning.
This should be visible in limitations and future held-out adversarial evaluation;
the development conformance corpus does not measure this class of ambiguity.

## Validation performed

The initial CLI was invoked on the reproductions above and on CR/LF/NEL/LS/PS,
combining accents, zero-width joiners, ordinary and quoted metadata, identifier
suffixes, case changes and mixed contexts. CR/LF/NEL/LS/PS, combining-mark and
zero-width guards behaved conservatively on the tried inputs.

Two temporary Rust integration tests passed under Rust 1.85.0: one asserts the
actual initial ADV-1/ADV-2 behavior reproducibly, and one checks determinism and
ordered valid UTF-8 byte spans for 200,000 generated Unicode scalars, 100,000 dense
`idk` annotations, a 2,000,000-byte unknown identifier and a phrase with 2,000,000
spaces. See `adversarial-repro.log`. No panic or invalid span was observed. This is
focused stress testing, not a complete proof or a performance benchmark.

The sidecar parser rejects incompatible fingerprints before semantic parsing,
checks actual remaining byte counts before reserving from hostile lengths,
validates UTF-8 span boundaries, candidate IDs and ordering, and regenerates
canonical annotations. No additional severe parser/roundtrip issue was found in
this pass. FNV is explicitly noncryptographic, as documented.

No critical or high-severity memory safety, byte-loss or determinism issue was
found. ADV-1 and ADV-2 should be fixed and regression-tested before completion.
