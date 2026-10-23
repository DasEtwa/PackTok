# Architecture review follow-up

Independent Agent 2, 2026-10-23. This is a focused re-review of the implementation
agent's fixes for ARCH-1, ARCH-2 and ARCH-3 in architecture.md. The initial report
is retained unchanged. No other review reports were consulted and no production
code or commits were changed by this reviewer.

## Outcome

All three findings are fixed and independently verified. No known critical/high
architecture or compatibility defect remains from this review.

- **ARCH-1:** SemanticEncoding now retains
  `Option<SemanticRegistryIdentity>` with owned namespace, revision, policy ID and
  fingerprint. Enabled empty annotation results still carry their interpretation
  domain; disabled results carry neither identity nor annotations. Re-running the
  original different-meaning registry probe changes the observed equality from
  true to false. Lexical tokens and numeric annotations remain equal while
  registry identities and full encoding results differ, as required. The new
  collision regression covers this public API behavior.
- **ARCH-2:** LowercaseOnly now checks ASCII uppercase letters independently of
  spaces and apostrophe spelling after successful matching. The independent
  `not gonna lie` probe now resolves canonical, tabbed and double-space lowercase
  inputs; uppercase `Not gonna lie` remains CaseAmbiguity. A second independent
  regression verifies lowercase `don't know`, tabs, repeated spaces, NBSP and
  curly apostrophe combinations; uppercase remains ambiguous.
- **ARCH-3:** Boundary helpers distinguish wrapping quotes from intra-word
  apostrophes. Independently verified `language: 'fr'`, `'language': 'fr'`,
  `language: ’fr’` and `'language': "fr"` each resolve exactly one French-language
  annotation. `idk's`, `idk’s`, `someone'idk` and `someone’idk` still do not match.

The resolver policy is now explicitly `semantic-rules-v1.1`, which participates
in registry fingerprints; this prevents silent acceptance of old-policy sidecars.
Semantic ID assignments and legacy lexical artifact formats remain unchanged.
The architecture document `docs/architecture/semantic-packs-v1.md`, STRUCTURE.md,
README.md and FORMAT.md now describe the additive ownership/version boundaries,
identity requirements, conservative boundary behavior, custom-registry limits,
UTF-8 annotation scope, persistence and negative/inconclusive experimental scope.
The additional owned namespace allocation must remain part of B's actual timing
and output-memory accounting; the implementation agent is updating that accounting.

## Reproduction and checks

Rust 1.85.0 commands, with the task-local cargo/rustup environment from baseline:

```sh
cargo +1.85.0 run --quiet --manifest-path /tmp/packtok-architecture-probe/Cargo.toml
cargo +1.85.0 test --quiet --manifest-path /tmp/packtok-architecture-probe/Cargo.toml
cargo +1.85.0 test -p packtok-packs -p packtok-tokenizer semantic
git diff --check
```

The temporary external probe was built from the exact steps in architecture.md,
with one assert-based follow-up covering identity fields/collision/disabled mode,
quoted metadata/possessive protection and flexible-whitespace lowercase matching.
Its one test passed. The durable workspace targeted tests passed **13/13**:
9 semantic registry/resolver tests and 4 optional-tokenizer tests, with zero
failures. `git diff --check` passed. The primary agent owns final full-workspace,
release, formatting, static-analysis and benchmark verification.
