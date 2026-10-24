# Independent correctness follow-up — fixes verified

Reviewer: `review_correctness`, 2026-10-23, after the root's regression-first
fixes. This report supplements the preserved original `correctness.md`; no
production code, branches or commits were changed by the reviewer.

Reviewed the fixed matching/context functions, new regression tests, policy
identity `semantic-rules-v1.1`, and the tokenizer wrapper's registry identity.

## Finding dispositions

- **C1 closed.** The bounded context now retains its real start offset and
  rejects a missing boundary caused by budget exhaustion. The original
  `xlanguage:` + 119 spaces + `fr` reproduction is ambiguous. An independent
  sweep checked 11,844 combinations: three recognized key suffixes,
  seven word-like ASCII/Unicode/combining/joiner prefixes, four horizontal
  whitespace types and lengths 0 through 140. No prefixed key falsely resolved.
- **C2 closed.** VT and FF now join CR, LF, NEL, LINE SEPARATOR and PARAGRAPH
  SEPARATOR in the vertical exclusion set. All seven separator families were
  independently tested between `fr`/`bro`, language labels/`fr`, and within
  `keine Ahnung`; none supplied context or phrase whitespace.
- **C3 closed.** Simple ASCII/curly single quotes and double quotes delimit
  standalone expressions and metadata correctly. Independent probes also
  confirmed that letters, Unicode letters and underscores before quoted keys
  cannot manufacture a metadata boundary. Intra-word ASCII/curly apostrophes
  remain protected: possessives and attached words such as `idk's`, `idk’s`,
  `x'idk`, `idk’x` receive no substring annotation.

The revised `LowercaseOnly` check properly examines ASCII letter case without
conflating allowed horizontal spacing and apostrophe variants. Its regression
checks cover repeated spaces, TAB, NBSP and U+2019.

Registry identity now travels with `SemanticEncoding`. Independent custom/built-in
registry probes with equal numeric IDs and equal annotations compare differently
because namespace/revision/policy/fingerprint identify their interpretation;
disabled annotation returns no identity or annotations and the same lexical IDs.

## Validation

- Targeted crate tests: **77 passed**, zero failed (CLI 8 + 4 + 3, format 28,
  packs 12, tokenizer 22). Full raw output: `correctness-followup-tests.txt`.
- Re-ran the original 4,000 generated UTF-8 cases, checking deterministic valid
  non-overlapping spans, registered candidates and canonical sidecar roundtrips.
- Additional boundary, vertical-separator, apostrophe and identity probes passed.
  Source and output: `correctness-followup-probe.rs` and
  `correctness-followup-probe.txt`. Probes use a temporary Cargo package with
  path dependencies on packs, format and tokenizer, under Rust 1.85.0.
- `git diff --check` passed during the follow-up.

No remaining reproduced critical/high correctness issue or open original finding.
The documented heuristic scope and conservative Unicode false negatives remain
limitations; these checks do not establish general natural-language accuracy.
