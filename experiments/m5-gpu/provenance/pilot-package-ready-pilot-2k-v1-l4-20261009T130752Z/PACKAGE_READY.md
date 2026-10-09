# M5 pilot package readiness — pilot-2k-v1-l4-20261009T130752Z

Status: ready for a separately authorized single NVIDIA L4 attempt. No GPU was allocated and no consumed attempt marker was created during this preparation.

- Source: `b0e964eefcca260de452a7ee81be012ed6a4e032` on `m5-gpu-transformer`; unrelated dirty recovery artifacts were preserved.
- Bundle SHA-256: `04d6e883389d2fdb0479d623df2121109079ba8b4c261099560cda90c03ce227`.
- CUDA executable SHA-256: `df9b9cc1ff76e89a6c3814b97cf5aa81d7a062172bfb10737826481fad3eb459`. Fresh CUDA build on this source completed successfully; its hash matches the prior binary because Rust/CUDA sources were unchanged.
- Pilot runner SHA-256: `194325eb9dd54007a2f705dc7c8d48a3074fbea9a54dc278849349be0916fbbf`; tracked source `experiments/m5-gpu/scripts/remote-pilot.sh` is packaged as `remote.sh`.
- Frozen config SHA-256: `6536d5194bc7d422bdef6e871b58317497794ed1553b197a927e0434aa65c3d8`. Corpus manifest SHA-256: `033bf7a3126ad5e39d3d11e5abd257f9cfaaac2d8f001003830215191b9d494f`. All six prepared sequence and both tokenizer hashes match the frozen config.
- Runtime loader resolves every binary dependency from the bundle except `libcuda.so.1`, supplied by the future L4 host driver. NVRTC, NVRTC builtins, cuRAND, cuBLAS and cuBLASLt are bundled with notices.
- `plan-extended` passes and confirms 2,000 updates and 4,096,000 target positions for both A and C, seed 20261008.
- Exact bundle E2E rehearsal passed through the packaged bridge and packaged pilot `remote.sh`: success reaches plan/A/C (remote status 0); injected A failure returns status 9, archives stdout/stderr, and does not launch C.
- Four bundled bridge/approval transport regression tests passed. The approval value is passed via Colab `--env` and as the Rust argument; only a harmless fixture approval was used locally.

The full checks, rehearsal output and archived fixture results are stored beside this report. The archives contain no credentials or model checkpoints.
