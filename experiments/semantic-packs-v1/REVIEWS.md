# Independent reviews and autonomous fixes

The first complete implementation was reviewed independently by three agents.
Reviewers did not share findings before writing their reports and did not modify
production code. They used separate reproduction probes and the Rust 1.85 toolchain.

| Reviewer | Original report | Scope |
|---|---|---|
| Correctness & Rust | [correctness](reviews/correctness.md) | UTF-8, panics, bounds, determinism, canonical parsing, roundtrip |
| Architecture & compatibility | [architecture](reviews/architecture.md) | old APIs/formats/IDs, registry identity and extension contracts |
| Adversarial semantics | [adversarial](reviews/adversarial.md) | malformed/large/Unicode/context/ambiguity probes |

## Deduplicated triage and changes

No critical/high defect was reproduced. Medium findings were treated as bugs and
fixed autonomously; lower-priority findings were fixed or narrowed to an explicit
coverage limitation.

| Finding | Review IDs | Reproduction | Fix and regression |
|---|---|---|---|
| Medium: context budget fabricated a metadata-key boundary | C1, ADV-1 | `xlanguage:` +119 spaces+`fr` selected French | Track actual context slice start; exhaustion cannot prove a key boundary. Tests also cover xlang/xsprache and adjacent limits. |
| Medium: vertical whitespace supplied horizontal context | C2, ADV-2 | `fr`+VT+`bro`, `ka`+FF+`was` resolved | Exclude VT/FF as well as CR/LF/NEL/LS/PS from phrase/context spaces; test every vertical separator. |
| Medium: annotated output lost registry identity | ARCH-1 | Same numeric ID from unrelated registries produced equal encodings | Add owned namespace/revision/policy/fingerprint to enabled `SemanticEncoding`; custom/builtin collision regression. Disabled layer has no identity or matching. |
| Medium: lowercase extension rule depended on literal whitespace | ARCH-2 | Custom `don't know` rule became case-ambiguous with tabs/doubled spaces | Test ASCII letter case independently after equivalent phrase match; regress tabs/NBSP/spaces/curly apostrophes. |
| Low: single-quoted metadata did not reach resolver | C3, ARCH-3 | `language: 'fr'` unrecognized | Distinguish wrapping quotes from intra-word apostrophes; test quoted keys/values plus idk's/idk’s/someone'idk false matches. |
| Low: Unicode-boundary explanation overpromised coverage | ADV-3 | `«idk»`, `idk，` missed | Document curated ranges and conservative false negatives explicitly; no claim of complete Unicode word segmentation. |

The resolver regressions were added first: all four tests failed, with exact
output retained in [before-fix log](reviews/regressions-before-fix.txt). They pass
after fixes; [targeted after-fix log](reviews/regressions-after-fix.txt) includes
registry identity regression. A separate machine-readable review regression corpus
extends parser/gold checks without changing the original 33-case benchmark fixture.

Resolver changes use policy `semantic-rules-v1.1`, changing registry fingerprint
without renumbering IDs or modifying lexical artifact formats. FNV alone is not
authentication; loader also recomputes canonical decisions. Initial measurements
and failed test evidence remain intact rather than being overwritten.

Correctness probes covered 4000 generated UTF-8 inputs and canonical sidecars.
Adversarial probes covered 200000 generated Unicode scalars, 100000 annotations,
a 2 MB unknown word and a 2 MB horizontal whitespace run. No lexical loss,
reproducible panic or determinism failure was found. These probes supplement the
automated unit/integration suite; exact programs/logs and scope are in reports.

## Independent follow-up results

All three original reviewers performed focused follow-up passes:
[correctness](reviews/correctness-followup.md),
[architecture](reviews/architecture-followup.md),
[adversarial](reviews/adversarial-followup.md).
All reproducible findings are closed. Correctness re-ran 77 relevant tests,
4000 generated UTF-8/sidecar probes and 11844 additional bounded-metadata cases.
Adversarial follow-up tested 0..200-character ASCII/Unicode whitespace boundaries,
all vertical separator families, quote/intra-word apostrophe cases and about 3 MB
of mixed stress input. Architecture confirmed distinct registry identities,
flexible lowercase-rule matching and quoted metadata support. No known
reproducible critical/high/medium defect remains from these passes; the documented
heuristic coverage/domain limitations remain. Scope and logs are linked in each
review, and the automated suite retains the regression cases.
