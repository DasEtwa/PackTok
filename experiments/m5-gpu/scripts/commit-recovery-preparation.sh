#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/dasetwa/.local/bin:/home/dasetwa/.cargo/bin:$PATH
cd /home/dasetwa/projects/PackTok
root=experiments/m5-gpu
win=/mnt/c/Users/DasEtwa/PackTok
tr -d '\r' < "$win/$root/.gitignore" > "$root/.gitignore"
for file in prepare-recovery-cpu.sh preserve-recovery-delivery.sh commit-recovery-preparation.sh; do cp "$win/$root/scripts/$file" "$root/scripts/$file"; done
python3 -m py_compile "$root/scripts/authorize-drive-client.py" "$root/scripts/write-recovery-readiness.py"
bash -n "$root/scripts/prepare-recovery-cpu.sh" "$root/scripts/check-recovery-cpu.sh" "$root/scripts/authorize-drive-recovery.sh" "$root/scripts/install-rclone-recovery.sh" "$root/scripts/preserve-recovery-delivery.sh" "$root/scripts/commit-recovery-preparation.sh" "$root/scripts/verify-v4-recovery.sh"
git diff 4ed7f0de96cdb90fec23903186838e5a73bb5578 --exit-code -- crates Cargo.toml Cargo.lock rust-toolchain.toml "$root/src" "$root/Cargo.toml" "$root/Cargo.lock" "$root/configs" "$root/artifacts"
git add M5_GPU_TRANSFORMER.md M5_GPU_BENCHMARK.md README.md "$root/.gitignore" "$root/examples" "$root/scripts" "$root/provenance/RECOVERY_STORAGE.md" "$root/provenance/recovery-20261008" "$root/verification/SELF_REVIEW.md"
git diff --cached --check
git diff --cached --stat
git commit -m 'Prepare immutable M5 L4 recovery and scoped Drive backup workflow'
git rev-parse HEAD
git status --short --branch
git push origin m5-gpu-transformer
