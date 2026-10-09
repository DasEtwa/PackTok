# Repository maintenance plan — 2026-10-08

This is repository maintenance, not M6. No model, tokenizer, corpus, mappings,
GPU allocation or PR merge is authorized here. Starting HEAD is
74d91332d6612b1e0b8a74734409319a1b542861; PR #5 stays draft on
m4-factorization-ablation. Native Ubuntu-24.04 is the implementation checkout.

## Inventory before restructuring

The root has 24 Markdown documents and no docs/ navigation directory.
IDEA/STRUCTURE/AGENTS and BRAIN_README hold the project map/instructions;
FORMAT holds version contracts; M0–M5 reports hold protocols and measurements;
M1/POST_FIX/PERFORMANCE_MATH/PRE_GPU reports hold audits. PROJECT_OVERVIEW is an
earlier explanatory entrypoint. CHANGELOG tracks milestones. experiments/ and
fixtures/ contain versioned evidence and original sources. Root README has
accumulated operational updates that obscure the introduction.

The machine-readable initial inventory includes every tracked and pending path,
size and SHA-256, both checkout hashes and read-only runtime-web state. The two
copies have identical logical contents; checkout-only CRLF/LF differences are
recorded, not reconciled cosmetically. Pending M5 recovery files are byte-identical.
An external protected snapshot preserves all pending files before editing.
No tracked/pending file exceeds 5 MiB. No credential candidate was found by the
initial pattern scan; full reachable-history and staged scans follow.

The saved Codex PackTok project and this task point to the Windows PackTok
checkout. The prior recovery task had an incorrect runtime-web cwd. A read-only
filename/content search of that project found no PackTok recovery files. Its
existing unrelated changes must remain untouched. Ubuntu/MOOS stays stopped.

## Ordered delivery

1. Preserve the outstanding M5 recovery evidence/scripts in a focused commit.
2. Add docs/README, architecture, milestone, research and development navigation;
   rewrite README in accessible English, with one central results overview.
3. Harden the actual WSL launcher/supervisor with bounded CPU-only lifetime,
   session-exit, interruption, timeout, ownership and cleanup regressions.
4. Inspect Google OAuth, prepare Production with drive.file and a Desktop client,
   request explicit publishing approval, and document recovery/remaining limits.
5. Run MSRV/root and isolated-runner checks, links, hashes, security and targeted
   self-review; commit and push logical changes, update existing draft PR #5.

## Preservation decisions

No historical report, raw log, source, manifest, model or tokenizer is moved,
regenerated, normalized or removed. New navigation links to existing paths.
The initial inventory is the byte-preservation oracle for this milestone.
The old PROJECT_OVERVIEW and BRAIN_README remain linked historical context, not
duplicate copies of current results. Invasive migration is deferred because
absolute/relative paths and recorded source hashes are reproduction inputs.
No git clean/reset/rebase/force push/history rewrite, no permanent linger setting.
Ignored caches/bundles/weights stay external; no untracked material is deleted.

## Acceptance evidence

Record exact commands/environment/outcomes under this directory, retain failures
and distinguish CPU simulation from GPU evidence. Compare all initial historical
files, scientific sources and the unrelated-project state at delivery. Keep any
OAuth publishing decision pending unless the human explicitly approves it.
