# M6 Slice 2 — multi-seed readiness

Status: CPU-ready; no GPU allocation or neural training was performed. Future GPU work requires explicit approval.

## Frozen five-pair study

The study uses seeds 20261008–20261012. M5 seed 20261008 is protocol-identical and is reused as the historical first pair; its result has already been seen, so all reporting must also show the sensitivity over the four new pairs (20261009–20261012). This is not blinded confirmation. The four new pairs require eight new 2,000-update runs.

Configuration identity is recorded in MULTI_SEED_MANIFEST.json (SHA-256 in the preparation checksum list). It keeps the M5 14,681,984-parameter model, A/C tokenizers, corpus/splits, batch 8, context 256, 2,000 constant-LR AdamW updates, LR 0.0003, weight decay 0.01, FP32, T regime, 4,096,000 targets per variant, existing BPB evaluator and validation points 1/500/1000/1500/2000. The regression test loads the frozen M5 and M6 configs, reverts only revision and seed-list fields, and asserts the complete JSON objects are equal.

CPU-generated initialization identities and RNG states are in experiments/m5-gpu/provenance/m6-multiseed-preparation-20261010/initialization-hashes.json. The five A/C hash pairs match; the 20261008 hash matches the M5 archive. The TSV is the expected hash input checked by the remote launcher before a run is accepted. The sampler uses xorshift64-13-7-17 with seed XOR 0xa341316c9e3779b9.

## Launch and output contract

The launcher is experiments/m5-gpu/scripts/remote-m6-multiseed.sh. The actual existing remote-bridge.py extracts the package and invokes it as remote.sh. It runs only new seeds 20261009–20261012, ascending, A then C for each seed. Output is isolated at results/seed-<seed>/<A|C>; an existing path is a hard refusal. Each run must expose one final row, five validation rows, exactly 2,000 updates, 4,096,000 target positions, and the expected initialization hash. Metrics and checkpoint SHA-256 files are retained. Every run continues to use the existing final-only TEST behavior; those scores remain exploratory because M5 already exposed TEST.

The local end-to-end rehearsal invokes the real bridge and archived launcher boundary with only external nvidia-smi, ELF loader and training executable replaced by harmless doubles. It covered the prior results-directory lifecycle, all eight intended A/C commands and their args, cwd, config, approval forwarding, success archive, deliberate training exit 9 with stderr/status preservation, and missing approval. No GPU was allocated.

## Verification evidence

- Rust: 35 tests passed, 0 failed. This includes config freeze equality and the existing M5 regression suite. Log: experiments/m5-gpu/provenance/m6-multiseed-preparation-20261010/rust-tests.log.
- Remote launch: 3 end-to-end rehearsal cases passed. Log: experiments/m5-gpu/provenance/m6-multiseed-preparation-20261010/remote-rehearsal.log.
- Statistical summary helper: 2 tests passed. Log: experiments/m5-gpu/provenance/m6-multiseed-preparation-20261010/statistics-tests.log.
- cargo fmt --all --check, bash -n on the launcher, Python py_compile, and git diff --check passed.
- CUDA release executable built locally, not executed on a GPU: target-cuda/release/packtok-m5, SHA-256 97b25305b455c7b19861b294f8bc5b9a1df2b4ea821a3af3d2a9ec2b4b1859da. Source code commit: 0af19cb6f009e8959ce72644acb8cc200c1aac31. CUDA toolkit: 12.4.131; compute capability: 8.9.
- ldd resolved all binary dependencies with the CUDA 12.4 runtime path; libcuda is supplied by the eventual host driver. Runtime dependency output is preserved beside the build log.

## Statistics and resource estimate

experiments/m6/scripts/analyze_m6_multiseed.py takes the five final validation A/C BPB values and reports paired C−A deltas, mean, sample SD, median, direction counts, relative effect and a two-sided paired Student-t interval, plus a separate four-new-seed sensitivity interval. It rejects missing seeds and invalid metrics. Five seeds provide preliminary variability evidence only.

Observed M5 training-only time was 212.249 s for A plus 196.922 s for C per pair. Four new pairs project to 27.28 training-only minutes; validation, checkpointing, startup, transfer, final scoring, archive work and cleanup are additional. The plan is two separately authorized sessions, two new seed pairs each, with a 30–45 minute planning range per session and controlled shutdown from minute 55. A rough linear CU proxy is 2.6 CU for four pairs; it is not a billing guarantee. Actual usage must be measured. No GPU allocation has occurred.

## GitHub

Changes are committed on m6-research and update the existing Draft PR #7 against main. PR remains unmerged.
