# PackTok 📦

> Experimental factorized tokenization and vocabulary architecture for language models.

**Canonical BRAIN space:** [`DasEtwa/BRAIN/PackTok`](https://github.com/DasEtwa/BRAIN/tree/main/PackTok)

PackTok explores a simple question:

> What happens if a language model does not treat its vocabulary as one flat bag of tokens, but as a set of specialized token **packs**?

Instead of only:

```text
2649 -> "beautiful"
3651 -> "house"
7821 -> "running"
```

PackTok can represent tokens as:

```text
ADJ:2649  -> "beautiful"
NOUN:3651 -> "house"
VERB:7821 -> "running"
```

or, more generally:

```text
(pack_id, local_token_id)
```

The important part is not POS tagging itself. The research target is the broader architecture: **factorize the vocabulary, keep packs independently testable, and measure whether that helps compression, model quality, memory, training, or inference.**

PackTok is Rust-first. Python may be used for isolated tooling or external analysis, but the tokenizer, training logic, serialization, runtime, benchmarks, and long-term model-facing interfaces should be implemented in Rust.

## Documents

- [IDEA.md](./IDEA.md) — the idea in one place: motivation, model interaction, open questions, and first experiments.
- [STRUCTURE.md](./STRUCTURE.md) — target repository layout, crate boundaries, data flow, and dependency rules.
- [AGENTS.md](./AGENTS.md) — mandatory rules for coding/research agents working on PackTok.

These documents are intentionally small enough to stay readable. If implementation later moves into a standalone `DasEtwa/PackTok` repository, this BRAIN space remains the durable project map and should link to the production repository.

## Current status

**Idea / architecture stage.**

No claim is made yet that PackTok beats BPE, Unigram, byte-level tokenization, or any other baseline.

The first meaningful result must come from a controlled comparison where PackTok and the baseline use the same corpus, comparable model/parameter budgets, comparable training compute, and identical evaluation data.

## Direction

Early candidate packs:

```text
NOUN
VERB
ADJECTIVE
FUNCTION
NUMBER
PUNCTUATION
CODE
OTHER
BYTE_FALLBACK
```

Later experiments may use morphology, language, code/data domains, automatically learned clusters, or multiple factors at once.

The output model does not have to use hard routing. A token score may be composed from both pack and local-token scores:

```text
score(token) = score(pack) + score(token | pack)
```

That keeps the architecture testable without making one wrong pack decision fatal.

## License direction

If/when PackTok becomes a standalone public implementation, the intended license is **Apache-2.0** unless the project explicitly decides otherwise before publication.
