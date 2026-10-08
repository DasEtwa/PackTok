# Research hypotheses

Structured vocabularies may improve specialization, sequence representation,
output cost or extensibility. These are testable hypotheses, not established
advantages. [IDEA.md](../../IDEA.md) retains the original motivation.

M1 supplies the flat BPE control. M2 tests a deterministic pack tokenizer. M3
changes tokenizer and head together; M4 separates them. M5 asks whether the small
M4 tokenizer-sequence signal survives a shared flat causal Transformer. There is
no completed M5 GPU quality result.

Learned routing/allocations, morphology, adaptive vocabulary and language/reasoning
packs remain future possibilities. They require separate frozen protocols,
matched baselines and preserved negative outcomes. See [results](results.md).
