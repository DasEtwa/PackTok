# Focused independent adversarial re-review

2026-10-23, after the root's semantic-rules-v1.1 fixes. The initial review and
initial reproduction log remain unchanged in this directory. No production edits,
branch changes or commits were made by this reviewer.

## Reproduced fixes

- **ADV-1 closed:** the three original prefixed-key inputs now remain ambiguous.
  Expanded probes cover `xlanguage:`, `xlang:`, `xsprache:` and prefixed single,
  double and curly-apostrophe key quotes, each with 0–200 intervening spaces,
  NBSPs or thin spaces. None fabricates a French-language resolution. The revised
  parser checks whether its bounded context exhausted before proving a left key
  boundary.
- **ADV-2 closed:** VT and FF now block phrase matching and contextual evidence.
  Probes covered all seven excluded separators (VT/FF/CR/LF/NEL/LS/PS), forward and
  backward `bro` and German context, language-key spacing, and `for real` phrase
  spacing. All remain ambiguous or unmatched as appropriate. NBSP remains a valid
  horizontal separator and `fr` + NBSP + `bro` still resolves correctly.
- **ADV-3 addressed by accurate scope documentation:** the code now explicitly
  describes selected punctuation/emoji ranges, conservative attachment of other
  symbols and incomplete Unicode segmentation. Curated punctuation false
  negatives remain an acknowledged v1 limitation, not an unresolved severe bug.

## Quote and stress checks

Ordinary single/double quoted aliases and metadata now recognize the intended
span. The new quote handling does not reopen intra-word recognition for `idk's`,
`idk’s`, `someone'idk`, `someone’idk`, `idk'other` or prefixed quoted `fr` probes.
Quotation handling remains a limited deterministic policy rather than a parser
for every typographic quote pairing. For example, U+2018 opening metadata quotes
are not one of the explicit key-quote forms; this is a conservative false negative.

A further deterministic stress input included 100,000 emoji, 200,000 NBSPs and a
2,000,000-byte unknown identifier around the corrected metadata and quote cases.
All produced ordered valid UTF-8 byte spans without panic; repeated annotations
were equal. This was a focused correctness check, not a timing benchmark.

Four temporary Rust integration tests passed under Rust 1.85.0; see
`adversarial-followup.log`. Their source was removed after execution. The
production code contains permanent regressions for the root fixes.

The policy ID changed from `semantic-rules-v1` to `semantic-rules-v1.1`, and the
registry fingerprint incorporates that ID, so older sidecars cannot silently use
the revised context policy.

**Disposition:** no reproducible critical/high issue and no remaining medium
finding from this independent adversarial pass. Direct-alias domain ambiguity
(e.g. NGL meaning natural gas liquids) and curated Unicode-boundary false
negatives remain explicitly limited heuristic behavior requiring wider v2
held-out evaluation.
