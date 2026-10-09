# M5 A/C Pilot Attempt — 2026-10-09 (partial)

## Outcome

The authorized single NVIDIA L4 session started and ran Variant A for all 2,000 frozen T updates. The runner then exited with code 1 during final validation domain aggregation, before it could start Variant C. No second allocation was requested. The frozen experiment configuration and prepared corpus were not changed.

The captured error was `domain/category aggregation does not reconcile to global score`. The check required two separately ordered FP64 NLL accumulations to differ by no more than `1e-9`. A deterministic local reproduction with 800,000 otherwise exactly reconciled token contributions produces an 1.68e-8 summation-order difference and fails the old check. Exact per-domain accumulators were not emitted by the failed run, so the precise discrepancy in this execution is unavailable. A local fix now uses a tight relative FP64 tolerance of `1e-12`; byte and target-token totals still must match exactly. The new regression test failed with the old tolerance and passes with the correction; all five domain unit tests pass after the fix. This source change was made after the immutable pilot bundle ran and is not in that bundle.

## Frozen identities

- Branch: `m5-gpu-transformer`
- Checkout HEAD at launch: `b8f96c6a15069654041fc62300393786c774eec4`
- Source commit embedded in binary/package: `b0e964eefcca260de452a7ee81be012ed6a4e032`
- Bundle: `pilot-2k-v1-l4-20261009T130752Z`
- Bundle SHA-256: `04d6e883389d2fdb0479d623df2121109079ba8b4c261099560cda90c03ce227`
- CUDA executable SHA-256: `df9b9cc1ff76e89a6c3814b97cf5aa81d7a062172bfb10737826481fad3eb459`
- Frozen config SHA-256: `6536d5194bc7d422bdef6e871b58317497794ed1553b197a927e0434aa65c3d8`
- Corpus-v2 manifest SHA-256: `033bf7a3126ad5e39d3d11e5abd257f9cfaaac2d8f001003830215191b9d494f`
- Expected paired initialization fingerprint in package manifest: `155de938a7f469bf3bce65e548e938792bb6ebe0a53ffcda8135d0c65881a0a1`. A started with seed 20261008; C did not start, so an actual A/C initialization pair was not observed.
- Remote order: `A 2026-10-09T13:52:13Z`; no C entry.

## Measurements

| Metric | A — Flat BPE | C — PackTok |
|---|---:|---:|
| Verified training updates | 2,000 | 0 (not started) |
| Final training loss | 3.3719987869 per target at update 2,000 | — |
| Validation bits/raw byte at step 1 | 4.8255975720 | — |
| Validation bits/raw byte at step 500 | 3.1639094632 | — |
| Validation bits/raw byte at step 1,000 | 2.8047013010 | — |
| Validation bits/raw byte at step 1,500 | 2.6419043903 | — |
| Validation bits/raw byte at step 2,000 | 2.5729648496 | — |
| Actual training raw-byte exposure | 7,575,564 bytes | 0 measured; frozen plan projection 6,946,291 bytes is not an observation |
| Synchronized mean step time | 0.106125 s (all steps, includes 14.715 s first-step startup/JIT); 0.098816 s for steps 2–2,000 | — |
| Training time | 212.249 s | — |
| Target positions per training second | 19,298.06 | — |
| Represented raw bytes per training second | 35,691.82 | — |
| Peak sampled VRAM | 2,442 MiB | — |
| Final resume checkpoint | `A/latest.resume.safetensors`, SHA-256 in `ARTIFACT_SHA256.txt` | — |

Both frozen variants were planned for 4,096,000 target positions. Only A achieved them. There is no C−A validation BPB comparison. No domain-specific report was emitted.

The frozen runner performed its final-only A TEST scoring computation after the final validation score and before the domain aggregation error, but the final TEST result was not serialized because execution exited before the final result record. The numerical TEST score is unavailable. TEST was therefore accessed for final scoring and must not be described as untouched for a revised experiment. No TEST result was used to tune or alter this pilot. C TEST was not scored.

## Runtime and cleanup

- GPU: NVIDIA L4, driver 580.82.07, 23,034 MiB total.
- Allocation request: 2026-10-09 13:45:33 UTC.
- Release verified: 2026-10-09 13:59:06 UTC; session lifetime 813 s (13 min 33 s), below the 60-minute limit.
- Session alias absent after stop; `Active assignments: 0`.
- Displayed Compute Units: 167.15 before, 166.92 after; observed delta 0.23 CU.
- Remote transport returned 0, but the archived remote exit code was 1; the Colab CLI result was not treated as success.
- Full remote results archive SHA-256 and checkpoint/log hashes are in `ARTIFACT_SHA256.txt`.
- TEST scoring caveat above; no final TEST number was retained.

## Reproduction and follow-up

Run the regression with:

```sh
cargo test --locked --manifest-path experiments/m5-gpu/Cargo.toml --lib domains::tests
```

The regression was also run once with the old `1e-9` comparison to confirm it fails, then after the tolerance correction: five domain tests pass. A new CUDA package/readiness identity is required before any future GPU execution with the fixed source. This attempt consumed the one authorization in this session; no further allocation was made.
