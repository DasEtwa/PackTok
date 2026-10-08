# Evidence limitations

M1/M2 held-out tokenizer fixtures and M3 data are small, manually authored and
synthetic. M3 conflates tokenizer and output-head changes. M4 uses a larger but
still narrow English/German/code/synthetic mixture, a tiny RNN, context 16 and
three seeds. Validation lacks Rust code after the frozen leakage filter, while
test retains some. No split was adjusted after model outcomes.

Equal token context and token schedules do not equalize raw-byte context or
exposure. Analytical MAC v1 excludes optimizer, nonlinearities, softmax and memory
traffic; matching it is not equal measured FLOPs or wall time. Factorized gold-pack
scoring costs differ from full joint generation. Timing and process peak scopes
must be read in each original report.

M5 has CPU preparation and failed GPU lifecycle evidence, but no passing CUDA
correctness gate, synchronized model timings, measured peak VRAM or completed
quality comparison. Weights-only safetensors do not support exact training resume.
A short Drive test verifies bytes at that moment, not token longevity.

Full qualifications: [M1 review](../../M1_REVIEW_FIXES.md),
[M2 benchmark](../../M2_PACKS_BENCHMARK.md),
[M3 benchmark](../../M3_MODEL_BENCHMARK.md),
[M4 benchmark](../../M4_ABLATION_BENCHMARK.md),
[M5 benchmark](../../M5_GPU_BENCHMARK.md).
