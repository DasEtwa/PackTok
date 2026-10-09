# CPU baseline freeze delivery

Exact CPU freeze commit: **f3b6c3fb41f46c93e701b585d859b0b9071d65a5**,
on `m4-factorization-ablation`, based on
`53ef2e8639a1e77c40bb26f2a935bdd321f459c0`.
This freeze commit contains both fixes, nine regression tests, the resolved
PRE_GPU_CODE_REVIEW.md and complete Windows/Linux verification evidence.
This delivery record is a subsequent documentation-only addition; it changes
no verified source, format, test or historical artifact. The code freeze remains
the commit above rather than a self-referential documentation commit hash.

The tests ran before committing, against the exact model source recorded in
model-source-sha256.txt (SHA-256
`0a96a4feed6a96c4f6034552de2426fe4f0dc91e84b46ed956347ef88a530b6e`) and
model-source.patch. That source is unchanged in the freeze commit. Recorded
environment HEAD values identify the base checkout and dirty patch, not a
different experimental implementation.

Changed implementation: crates/packtok-model/src/lib.rs only.
Changed normative/status documentation: FORMAT.md, README.md, CHANGELOG.md,
PRE_GPU_CODE_REVIEW.md. .gitattributes preserves raw review/fix evidence without
normalizing captured bytes; preserved patch blank-context records have a scoped
whitespace exception. Additional paths are raw review/fix evidence and scripts
under experiments/pre-gpu-review-20261007 and experiments/pre-gpu-fix-20261007.
The complete changed-file list is changed-files.txt.

Windows Rust/Cargo 1.98.1: 134 workspace tests passed in each of debug and release.
Linux/MSRV Rust/Cargo 1.85.0: 135 passed in each profile. Both platforms passed
formatting, strict Clippy, release builds and targeted serialization/generation,
numerical-gradient and optimizer-atomicity checks. No tests failed.

All 72 canonical M3/M4 model files and 6 tokenizer files load/reserialize to
the same bytes; all 12 historical M4 A/D versus M3 pairs remain identical.
Historical M3 final/audit manifests and the frozen M4 manifest match every
expected hash. Valid-prompt generation matches the pre-fix observations on both
platforms. Original review evidence is retained with its before-report snapshot.
No research experiment was retrained, no methodology/tokenizer changed and no
directly related unresolved correctness issue remains. M5 was not started.

`READY_FOR_M5_GPU`

## Maintenance navigation note (2026-10-08)

A later M5 documentation-only follow-up restored a missing DELIVERY.md navigation target referenced by the unchanged root PRE_GPU_CODE_REVIEW.md. This note records that repair; it is not a new audit or reconstructed raw log. Original evidence remains in the files listed above, and the exact CPU freeze remains f3b6c3fb41f46c93e701b585d859b0b9071d65a5.
