M5 compares M1 token sequences and bijectively flattened M2 IDs through the same
14,681,984-parameter flat causal Transformer and frozen corpus-v2. Its isolated
Rust/Candle workspace leaves the historical root CPU workspace CUDA-free. This
draft remains stacked on `m4-factorization-ablation`.

Both historical L4 requests are preserved as failures: the first stopped at ELF
loader exit 127 before Rust; the recovery received local user-service SIGTERM
after READY, before remote diagnostics/Rust execution. Owned-session release was
verified. Outstanding recovery/Drive evidence is now committed. This maintenance
made **zero new GPU allocations**, changed no scientific source/input/artifact,
and merged no PR.

Maintenance adds an English research README, documentation navigation and one
results synthesis retaining M4's unfavorable same-schedule factorization and
inconclusive interaction. The actual WSL supervisor now uses an independent
bounded Windows client, reaps owned transport descendants, rejects missing remote
archives and rechecks endpoint ownership before release. Three CPU mock executions
exceeded five minutes; disposable session termination and active initiating-client
interruption are covered. No linger or permanent background setting was installed.

Validation: Linux root Rust 1.85 fmt/strict Clippy/debug+release tests (135 each)
and release build; equivalent Windows gates (134 tests each); isolated M5 CPU
gates (18 debug/release tests); 22 recovery and nine restore regressions; six
actual-entrypoint systemd cases. Historical compatibility passes on both platforms
for 72 models, six tokenizers and 12 model pairs. Byte audits preserve all historic
artifacts/measurements, active navigation resolves and staged secret scans pass.

Dedicated Desktop OAuth/rclone with `drive.file` now passes real upload/full
download/size/SHA-256/completion-manifest verification and fresh-destination
restore. OAuth remains External/Testing: Production publication requires explicit
user approval and the console's Branding prerequisite. The seven-day Testing
token limit remains; broader Drive access was not introduced.

Remaining blockers: the real CUDA loader/Rust GPU gate is unresolved, no GPU
quality/throughput/VRAM/checkpoint result is claimed, and weights-only safetensors
are not exact optimizer/RNG/cursor resume. The next L4 allocation is a separate
explicitly authorized task; full runs need an approved measured runtime/CU budget.
Operational state remains **M5_BLOCKED — INFRASTRUCTURE**.

Evidence: [maintenance report](https://github.com/DasEtwa/PackTok/blob/m5-gpu-transformer/docs/maintenance/2026-10-08/REPORT.md),
[research results](https://github.com/DasEtwa/PackTok/blob/m5-gpu-transformer/docs/research/results.md),
[M5 accounting](https://github.com/DasEtwa/PackTok/blob/m5-gpu-transformer/M5_GPU_BENCHMARK.md),
[OAuth/recovery](https://github.com/DasEtwa/PackTok/blob/m5-gpu-transformer/docs/development/oauth.md).
