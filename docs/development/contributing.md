# Contributing

Read [AGENTS.md](../../AGENTS.md), [IDEA.md](../../IDEA.md),
[STRUCTURE.md](../../STRUCTURE.md) and [README.md](../../README.md) first.
Propose a small falsifiable change, preserve byte fallback/deterministic IDs,
and keep tokenizer, routing, model prediction and evaluation separate.

Document relevant claims, defaults, decisions, limitations and failures in the
same change. Performance changes need a reproducible measured comparison and
correctness/reference checks. Do not silently change frozen M0–M5 corpora,
artifacts, mappings or results. Preserve failure evidence and third-party notices.

Run the [applicable gates](testing.md), inspect staged changes for credentials
and large generated files, and preserve the intended PR base. GPU allocations,
full-run budgets, merges, history rewrites and destructive cleanup require their
own explicit authorization. Maintenance does not grant GPU permission.
