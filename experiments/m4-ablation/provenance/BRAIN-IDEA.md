# PackTok — The Idea

**Canonical BRAIN space:** [`DasEtwa/BRAIN/PackTok`](https://github.com/DasEtwa/BRAIN/tree/main/PackTok)

Related documents: [README](./README.md) · [STRUCTURE](./STRUCTURE.md) · [AGENTS](./AGENTS.md)

## The short version

Most modern language models ultimately operate on a flat token vocabulary:

```text
context
  ↓
model
  ↓
logits over every token
  ↓
next token
```

PackTok asks whether the vocabulary can instead be factorized into specialized **packs**:

```text
context
  ↓
pack score
  ↓
local token score
  ↓
next token
```

A token becomes conceptually:

```text
(pack_id, local_token_id)
```

For example:

```text
NOUN:3651 -> "Haus"
VERB:7821 -> "laufen"
ADJ:2649  -> "sämtlich"
```

This is not meant to assert that nouns, verbs, and adjectives are the perfect split. They are simply an obvious first experiment.

## Why try it?

A flat vocabulary makes every token compete for the same global vocabulary budget.

English fragments, German morphology, punctuation, source code, numbers, names, URLs, JSON, and rare words all live in one namespace.

PackTok explores whether specialization can improve one or more of these:

- vocabulary efficiency
- sequence length
- reuse of morphology
- multilingual behavior
- unknown-word generalization
- embedding organization
- output-head cost
- cache/memory locality
- training efficiency
- inference efficiency

A useful PackTok result does **not** need to win everywhere. A measurable advantage in a specific workload is still valuable if the trade-off is understood.

## Example: German

German is useful for stress-testing tokenization because productive compounding and morphology create many valid words that may be rare or unseen.

A classical subword tokenizer might produce:

```text
Haustürschlüssel
→ ["Haust", "ür", "schlüssel"]
```

A linguistically nicer segmentation might be:

```text
Haus | tür | schlüssel
```

PackTok does not assume that human-looking boundaries are automatically better. It gives us a framework to test whether specialized vocabularies or morphology-aware packs actually help a model.

Characters such as `ä`, `ö`, `ü`, and `ß` remain ordinary Unicode text. A byte fallback must guarantee that every valid input can always be represented, even if no specialized pack knows it.

## Packs do not have to mean POS

Possible pack schemes include:

```text
NOUN / VERB / ADJECTIVE / FUNCTION
```

```text
ROOT / PREFIX / SUFFIX / INFLECTION
```

```text
GERMAN / ENGLISH / CODE / MATH / STRUCTURED_DATA
```

```text
NUMBER / PUNCTUATION / URL / IDENTIFIER / RAW_BYTE
```

or automatically learned clusters with no human name at all.

Multiple factors may eventually coexist, but the first experiments should remain small enough to explain.

## Hard routing is optional

The simplest hierarchical formulation is:

```text
P(token | context)
=
P(pack | context)
×
P(token | pack, context)
```

But PackTok should also test soft composition:

```text
score(token)
=
score(pack)
+
score(local_token | pack)
```

This matters because linguistic categories are context-dependent.

For example:

```text
"Wir laufen."       -> VERB
"Das Laufen hilft." -> NOUN
```

The architecture must not pretend that a surface string has one permanent grammatical role.

## Encoding and generation are separate problems

PackTok must keep two questions distinct:

1. **How is existing text encoded into canonical pack/local IDs?**
2. **How does a generative model predict the next pack/local ID?**

Training requires a deterministic target representation. Generation may use learned routing.

For early POS-style experiments, corpus annotations may be produced as an offline preprocessing artifact. The PackTok core should consume a documented representation rather than hide a language model or Python tagger inside the runtime.

Long term, PackTok may learn pack structure automatically.

## Byte fallback is non-negotiable

PackTok must always be able to represent arbitrary input.

A reserved byte/raw pack gives the system a complete fallback:

```text
BYTE_FALLBACK:0..255
```

Specialized packs are an optimization and representation choice, never a reason to produce unknown text.

## First experiment

Start small.

### Baseline

```text
byte-level BPE
vocab target: ~16k
```

### PackTok v0.1

Candidate packs:

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

The exact local vocabulary budget is an experimental parameter, not a hard architectural truth.

### Compare

Use:

- the same raw corpus
- the same train/validation/test split
- the same normalization policy
- comparable total model parameters
- comparable training compute
- identical model architecture except where PackTok requires the output/input factorization
- deterministic seeds where practical

Measure at minimum:

- validation loss
- bits or loss per byte/character where appropriate
- bytes per token
- sequence length distribution
- tokenizer encode/decode throughput
- allocations and peak memory
- vocabulary/embedding memory
- training throughput
- inference throughput
- output-head cost
- German morphology/compounds
- code and structured data
- unseen/rare word behavior

## What would count as failure?

PackTok is allowed to lose.

Examples of useful negative results:

- better compression but worse model loss
- cheaper output head but slower routing
- strong German behavior but poor multilingual transfer
- excessive annotation complexity
- pack boundaries that make optimization harder
- no advantage over a flat vocabulary at matched compute

A failed experiment should remain reproducible. Do not massage metrics until the idea looks good.

## Long-term direction

If the factorization is useful, PackTok can become more than a tokenizer.

The same `(pack_id, local_token_id)` representation may flow directly into pack-specific embeddings or model heads:

```text
PackTok
  ↓
packed token representation
  ↓
pack/local embeddings
  ↓
model
  ↓
pack/local output scores
```

At that point the project becomes a vocabulary/model-interface architecture rather than only a text preprocessing library.

That is intentional.
