# PackTok — Agent Rules

**Canonical BRAIN space:** [`DasEtwa/BRAIN/PackTok`](https://github.com/DasEtwa/BRAIN/tree/main/PackTok)

Read before making changes:

1. [IDEA.md](./IDEA.md)
2. [STRUCTURE.md](./STRUCTURE.md)
3. [README.md](./README.md)

These rules apply to coding, research, review, benchmark, and documentation agents working on PackTok.

## 1. Protect the research question

PackTok is an experiment, not a predetermined success story.

Do not write code or reports as if packed vocabularies are already proven better than BPE, Unigram, byte-level tokenization, or flat vocabularies.

Negative results are valid results.

Never hide, reinterpret, or selectively remove a benchmark because PackTok loses.

## 2. Rust-first is a project constraint

The core implementation is Rust.

Rust must own:

- core token/pack types
- tokenizer training
- encode/decode
- artifact serialization
- runtime
- benchmarks
- CLI
- long-term model-facing interfaces

Python may be used only for isolated external tooling or analysis. It must not become an undocumented build/runtime/training dependency.

Do not prototype the real implementation in Python with the assumption that it will be rewritten later.

## 3. Keep the layers separate

Do not collapse tokenizer, pack annotation, model routing, and model prediction into one component.

Always distinguish:

- **representation:** how text maps to canonical IDs
- **pack definition:** what packs exist and what they mean
- **training:** how vocabularies/packs are learned
- **runtime:** fast deterministic encode/decode
- **model integration:** how a model consumes/predicts pack/local IDs
- **evaluation:** how the experiment is measured

A useful experiment may alter one layer without rewriting all others.

## 4. Byte fallback is mandatory

Every input must remain representable.

Specialized packs may fail to recognize a string. The tokenizer must still be able to encode it through a raw/byte fallback.

Do not introduce an `<unk>` requirement merely because a specialized pack has no entry.

Round-trip correctness is a hard invariant:

```text
decode(encode(input)) == input
```

unless an experiment explicitly enables a documented normalization mode.

## 5. Determinism is the default

Training and runtime behavior must be reproducible where practical.

Define deterministic:

- tie-breaking
- vocabulary ordering
- pack ordering
- token ID assignment
- serialization
- corpus traversal where it affects outputs
- random seeds for experiments

If an operation is intentionally nondeterministic, document why and record enough metadata to reproduce or statistically compare it.

## 6. Benchmarks are part of the feature

Do not merge a meaningful tokenizer/runtime optimization without a benchmark that can show whether it helped.

At minimum, keep infrastructure for:

- encode throughput
- decode throughput
- allocations
- peak memory
- artifact/vocabulary size
- bytes per token
- sequence lengths

Model experiments additionally need matched evaluation for:

- validation loss
- training throughput
- inference throughput
- parameter count
- compute budget
- output-head cost

A faster result that changes semantics is not an optimization unless the semantic change is the experiment.

## 7. Compare fairly

When claiming PackTok beats a baseline, control the comparison.

Prefer:

- same raw corpus
- same split
- same normalization
- same model architecture where possible
- comparable parameter budget
- comparable training compute
- identical evaluation inputs
- documented hardware/software versions

If exact matching is impossible, state the mismatch prominently.

Do not compare a tuned PackTok configuration against a deliberately weak or outdated baseline.

## 8. Keep pack schemes configurable

Do not bake `NOUN`, `VERB`, or `ADJECTIVE` into the core architecture.

POS packs are one experiment.

The core must be able to represent other schemes, including:

- morphology
- language
- code/data classes
- numeric/symbol classes
- automatically learned clusters
- hybrid/multi-factor schemes

Pack names are metadata. Stable IDs and contracts are architecture.

## 9. Do not assume one surface form has one role

Natural language is contextual.

Examples:

```text
"Wir laufen."       -> VERB
"Das Laufen hilft." -> NOUN
```

Do not silently assign a permanent grammatical identity to a string just because a first prototype uses POS packs.

Any annotation or routing strategy must state whether it is:

- context-free
- context-sensitive
- externally annotated
- learned
- heuristic

## 10. Prefer small falsifiable milestones

Do not build a giant "complete PackTok system" before the first comparison works.

Preferred order:

1. contracts and byte fallback
2. flat BPE baseline
3. first pack representation
4. controlled tokenizer benchmark
5. tiny matched model experiment
6. only then larger research variants

Each milestone should answer a concrete question.

## 11. Avoid architecture drift

Before adding a crate, service, database, GPU backend, model framework, daemon, or file format, explain why an existing component cannot own the responsibility.

Do not introduce infrastructure merely because it may be useful later.

Do not move responsibilities across crate boundaries without updating [STRUCTURE.md](./STRUCTURE.md).

## 12. Artifact compatibility matters

Tokenizer/model IDs are not casual implementation details.

Once an artifact format is declared stable for an experiment:

- do not reorder IDs silently
- do not reinterpret a pack ID silently
- do not change normalization silently
- do not change special-token behavior silently

Version incompatible changes.

## 13. Performance changes require evidence

Rust-first does not mean "micro-optimize everything".

First prove correctness.

Then profile.

Then optimize the measured bottleneck.

Keep readable reference paths where useful so optimized code can be differential-tested.

## 14. Corpus handling must be explicit

Do not silently download huge datasets.

Scripts must make dataset source, version, licensing assumptions, preprocessing, and expected size visible.

Keep train/validation/test boundaries explicit.

Do not contaminate evaluation sets with training data intentionally.

## 15. External annotations are data, not hidden runtime magic

Early experiments may use an external POS/morphology system to annotate a corpus.

If so:

- store or regenerate annotations reproducibly
- record the annotator/tool/version
- define a clear file/data boundary
- keep PackTok runtime independent of that external tool
- do not present annotation quality as PackTok model intelligence

## 16. Tests before claims

Required test families should include:

- byte round-trip
- Unicode round-trip
- `ä/ö/ü/ß`
- emoji / multi-byte Unicode
- empty input
- whitespace/newline variants
- malformed artifact rejection
- deterministic ID assignment
- pack fallback behavior
- serialization round-trip
- corpus edge cases
- differential checks against the flat baseline where applicable

Add adversarial fixtures when a bug is found.

## 17. Research logs must preserve failures

For significant experiments, keep enough metadata to answer:

- what commit ran?
- what corpus/split?
- what tokenizer artifact?
- what pack schema?
- what model config?
- what seed?
- what hardware?
- what metrics?
- what failed?

Do not overwrite a result directory in a way that destroys provenance.

## 18. Documentation hierarchy

Use these documents for their intended purpose:

- [IDEA.md](./IDEA.md) — why PackTok exists and what it is testing
- [STRUCTURE.md](./STRUCTURE.md) — where code belongs and how components depend on each other
- [AGENTS.md](./AGENTS.md) — how agents must work
- [README.md](./README.md) — human entrypoint and current status

If a change contradicts one of these documents, either change the implementation or deliberately update the relevant document in the same change.

## 19. Do not overclaim novelty

Related work exists in hierarchical softmax, class-based language models, factored language models, morphology-aware tokenization, and byte/token-free modeling.

PackTok may combine ideas differently, but agents must not claim "nobody has ever done this" without a proper literature review.

## 20. Default decision rule

When uncertain between a clever abstraction and a small measurable experiment, choose the small measurable experiment.

PackTok should become sophisticated only when the data gives it a reason to.

## 21. No undocumented facts

From M1 onward, no relevant technical fact may live only in source code, terminal
output, a commit message, an agent conversation, or a final report. Document every
material claim, value, default, assumption, architectural decision, format choice,
benchmark result, experiment configuration, limitation, failure, optimization,
and compatibility decision in a durable repository document or preserved raw log.
Numerical claims must include enough context to reproduce or interpret them.
Preserve unfavorable and failed experiments when they informed a decision; do not
delete or replace them because a later run looks better. Update the relevant
document in the same change as the implementation.

## 22. M4 experimental freeze

M4 is a factorization ablation, not tokenizer redesign. Read M4_ABLATION.md and
M4_ABLATION_BENCHMARK.md before changing its corpus, mappings, metrics or runs.
Preserve lexical-v1, historical artifacts and failed/preflight outputs. Do not
use held-out outcomes to change grouping, allocation, schedule or splits.

## 23. M5 GPU conservation

Read M5_GPU_TRANSFORMER.md and M5_GPU_BENCHMARK.md before M5 changes/runs. Use only
local WSL Ubuntu-24.04 for CPU work; do not use Ubuntu/MOOS. Colab allocation is
only NVIDIA L4, only after CPU readiness. First preflight is capped at 20 minutes.
Always stop and verify the owned session immediately after GPU work/errors,
before analysis/edits/replies. Full multi-seed runs require explicit approval of
the measured runtime/compute-unit budget. Never fall back to another accelerator.
