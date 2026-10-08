# M5 CPU preparation adversarial review — 2026-10-08

Scope: new experiment only; no root tokenizer, RNN, artifact encoding or M1–M4
methodology was changed. CUDA correctness remains unverified until the bounded
preflight. This is a CPU/source review, not a GPU pass.

Confirmed preparation issues and corrections, all before held-out model outcomes:

- Corpus-v1 source-local split counter left validation without code. Preserved
  v1 manifest/raw splits/completed tokenizers; global counter writes distinct v2
  immutable paths. Small-file regression covers the modulo split distribution.
- Evaluation admitted no scored target / zero batch. Reject before division;
  regression covers empty, single-token and zero-batch inputs. Full coverage
  test covers short final context and exact represented target bytes.
- An aggregate gradient-family gate could hide an inactive weight projection.
  Every weight tensor now needs finite positive squared gradient norm; individual
  bias/embedding entries may be zero. Regression isolates an inactive FFN
  projection while another family bias is active. Finite differences cover
  eleven weight tensors independently of autograd's loss implementation.
- Fresh checkpoint reload must distinguish initial weights; CPU and planned
  CUDA fixtures reload into a different seed. Layout/names/dtype/finiteness are
  checked before mutation. Atomic temp/fsync/rename retains the last complete
  image if a temporary is incomplete; regression retains and then replaces it.
- Final job must not admit an undeclared seed. It now rejects before CUDA or
  output creation, with a CPU regression. Frozen config equality rejects changed
  architecture/schedule/precision. Paired initialization is deterministic.
- Duplicate final validation/checkpoint writes would waste paid GPU time at an
  already measured final step. Reuse that score/image; no weights, target coverage
  or schedule change. B's non-periodic final step still evaluates/saves once.
- Alias disappearance alone could falsely prove release after CLI state removal.
  Record the assigned endpoint and require it absent from a successful known
  session inventory. The tenth mock case keeps the endpoint under an unknown
  alias; release remains unconfirmed and ownership marker is preserved.
- Source checksum guard prevents running edited scripts against an older CPU
  bundle. One-attempt guard prevents silent repeated allocation. Handled TERM,
  execution/download/upload errors and wrong hardware all trigger owned cleanup;
  unowned sessions remain untouched. Nine earlier plus the new tenth mock cases
  pass. These tests do not allocate a GPU.
- Final checkpoint/source hashing and statistical analysis stay on WSL after
  download/release. Only in-memory CUDA weight fingerprints are part of the
  update/initialization correctness gate. No remote installs, source edits,
  corpus construction, tokenizer training or Rust/nvcc compilation are planned.

Reviewed remaining limitations: strided near-duplicate guard is not semantic
paraphrase detection; held-out code share is smaller and real config coverage is
TRAIN-only. B is an English TRAIN prefix, not the full mixed distribution. CUDA
atomic gradients can vary; no deterministic-GPU promise. Checkpoints omit AdamW
state/cursor, so an interrupted seed must restart under a later approved budget.
Remote loader/CUDA ABI compatibility is a preflight gate; failure stops the L4
rather than starting remote ordinary dependency setup. TF32 substitution is
explicitly disabled for the FP32 preflight. No BF16 or hidden backend fallback.

Verification: root MSRV fmt/strict Clippy/debug/release/build logs root-*-3;
135 debug and 135 release tests. Isolated M5 fmt/strict CPU Clippy/debug/release/
build logs transformer-*-9 / transformer-clippy-8 / transformer-build-cpu-10;
18 debug and 18 release tests. Lifecycle-cpu-5 has ten mock cases. Historical
compatibility logs preserve all canonical model/tokenizer identities. Root
sources/lockfile and historical experiment files match CPU-freeze Git blobs.
No unresolved confirmed CPU correctness issue was found in this narrow pass.
GPU gate is still pending and must not be inferred from these results.

## Post-attempt lifecycle/runtime corrections

The first real VM exposed an additional wrapper issue: CLI transport status zero
can accompany a remote kernel exception. Correctness now requires downloaded
remote exit/gate evidence checked on CPU after verified release; eleven mocked
cases and rejection of the actual failed archive cover this. Loader stderr and
bootstrap output are preserved, inherited driver paths retained, pinned runtime
shared libraries prepared locally for a distinct recovery bundle. The exact
first remote soname failure is unconfirmed because its stderr was missing.
No GPU/model gate has passed and no automatic retry is performed. Models,
schedule/tokenizers/splits are unchanged. CPU resource accounting includes the
failed allocated time rather than hiding it.
