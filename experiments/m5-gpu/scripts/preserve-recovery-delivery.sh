#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/dasetwa/.local/bin:/home/dasetwa/.cargo/bin:$PATH
cd /home/dasetwa/projects/PackTok
root=experiments/m5-gpu
win=/mnt/c/Users/DasEtwa/PackTok
for file in M5_GPU_TRANSFORMER.md M5_GPU_BENCHMARK.md README.md; do
  # PowerShell working-copy CRLF is normalized by native Linux Git.
  tr -d '\r' < "$win/$file" > "$file"
done
tr -d '\r' < "$win/$root/.gitignore" > "$root/.gitignore"
for file in provenance/RECOVERY_STORAGE.md verification/SELF_REVIEW.md; do cp "$win/$root/$file" "$root/$file"; done
for file in authorize-drive-client.py write-recovery-readiness.py preserve-recovery-delivery.sh; do cp "$win/$root/scripts/$file" "$root/scripts/$file"; done
cp -n "$win/$root/provenance/recovery-20261008/shared-client-quota-observation.txt" "$root/provenance/recovery-20261008/"
run="$root/provenance/recovery-20261008/delivery-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir "$run"
{
 date -u
 colab version
 colab sessions
 colab usage
 test ! -f "$root/bundle-v4/preflight-attempted"
 test ! -f "$root/bundle-v4/recovery-owned.json"
 test ! -f "$root/bundle-v4/owned-session"
 git diff 4ed7f0de96cdb90fec23903186838e5a73bb5578 --exit-code -- crates Cargo.toml Cargo.lock rust-toolchain.toml "$root/src" "$root/Cargo.toml" "$root/Cargo.lock" "$root/configs" "$root/artifacts"
} > "$run/frozen-no-allocation-state.txt" 2>&1
export CUDA_ROOT=/home/dasetwa/.local/opt/packtok-cuda-12.4.1
export PATH="$CUDA_ROOT/bin:$PATH" CUDA_COMPUTE_CAP=89 RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2
export LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/lib64:/usr/lib/wsl/lib"
cargo clippy --locked --manifest-path "$root/Cargo.toml" --all-targets --all-features --target-dir "$root/target-cuda" -- -D warnings > "$run/m5-cuda-clippy-cpu.txt" 2>&1
python3 - "$root" "$run" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); run=pathlib.Path(sys.argv[2])
record=dict(status='M5_BLOCKED — INFRASTRUCTURE', reason='user-created Desktop OAuth client pending; required real WSL Drive fixture not verified',
 delivery_start_commit='4ed7f0de96cdb90fec23903186838e5a73bb5578', allocation_count_this_task=0, historical_allocation_count=1,
 allocation_seconds_this_task=0, authorized_unused_allocations=1, cuda_gate=None, gradients=None, optimizer=None,
 overfit=None, timings=None, peak_vram=None, weight_recovery=None, drive_backup='not verified: shared-client project quota',
 full_training_authorized=False, exact_training_resume=False, rate_proxy_30_minutes_cu=0.77,
 rate_proxy_basis='historical 1.54 CU/hour; conditional, not guaranteed billing', current_cli_snapshot='frozen-no-allocation-state.txt')
(run/'CURRENT_RESOURCE_ACCOUNTING.json').write_text(json.dumps(record,indent=2)+'\n')
files=[p for p in root.joinpath('provenance/recovery-20261008').rglob('*') if p.is_file()]
(run/'evidence-SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p)+'\n' for p in sorted(files)))
PY
# Copy only new public evidence and this task's new formatted example back.
cp -a "$root/provenance/recovery-20261008/." "$win/$root/provenance/recovery-20261008/"
cp "$root/examples/recover_weights.rs" "$win/$root/examples/recover_weights.rs"
printf 'Public CPU/resource evidence preserved; no GPU allocation: %s\n' "$run"
