#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/dasetwa/.local/bin:/home/dasetwa/.cargo/bin:$PATH
cd /home/dasetwa/projects/PackTok
root=experiments/m5-gpu
remote=${PACKTOK_M5_DRIVE_REMOTE:-packtok-drive-own}
win=/mnt/c/Users/DasEtwa/PackTok/experiments/m5-gpu
mkdir -p "$root/examples"
for script in install-rclone-recovery.sh authorize-drive-recovery.sh verify-v4-recovery.sh drive-backup.py remote-recovery.py l4-recovery.py test-recovery.py prepare-recovery-cpu.sh; do
  cp "$win/scripts/$script" "$root/scripts/$script"
done
cp "$win/examples/recover_weights.rs" "$root/examples/"
run="$root/provenance/recovery-20261008/cpu-final-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir "$run"
printf '%s\n' "$run" > "$root/provenance/recovery-20261008/cpu-final-path.txt"
{
 date -u
 rclone version
 stat -c 'rclone configuration permissions=%a' /home/dasetwa/.config/rclone/rclone.conf
 stat -c 'rclone directory permissions=%a' /home/dasetwa/.config/rclone
 rclone listremotes --long
 sha256sum /home/dasetwa/.local/share/packtok-rclone/rclone-v1.75.1-linux-amd64.zip /home/dasetwa/.local/bin/rclone
} > "$run/rclone-environment.txt" 2>&1
for dir in manifests logs checkpoints models preflight; do
  timeout 90 rclone mkdir "$remote:PackTok/M5/$dir" --retries 1 --low-level-retries 1 --timeout 30s
done
python3 - "$run" <<'PY'
import hashlib, json, pathlib, subprocess, sys
run=pathlib.Path(sys.argv[1])
root=pathlib.Path('experiments/m5-gpu')
fixture=run/'storage-fixture.txt'
fixture.write_bytes(b'PackTok M5 WSL-to-Drive harmless recovery fixture v1\n')
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
metadata=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              variant='infrastructure-fixture',seed=20261008,kind='preflight-fixture',
              configuration_sha256=sha(root/'configs/primary.json'),
              dataset_sha256={},
              tokenizer_sha256={v:sha(root/f'artifacts/corpus-v2/{v}-0.packtok') for v in ['A','C']})
# Frozen raw corpus identities from the already verified preparation record.
metadata['dataset_sha256']={
 'train':'d84c640e29b20f2c042d2910d0504efc2d64fb195476bfb3adfc591d523210ce',
 'validation':'72b840ddcb9f471078e377dfeca85667798a27ddbc055bbe4525376e33853135',
 'test':'62e3397fbc737fadafc236c6d3e1f25dfd73c61d66a9372a48af5450fbffd3ac'}
(run/'fixture-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
PY
python3 "$root/scripts/drive-backup.py" "$run/storage-fixture.txt" "$run/fixture-metadata.json" "$run/drive-fixture-readback" --remote "$remote" > "$run/drive-fixture-result.txt" 2>&1
python3 "$root/scripts/test-recovery.py" > "$run/recovery-regressions.txt" 2>&1
python3 -m py_compile "$root/scripts/drive-backup.py" "$root/scripts/l4-recovery.py" "$root/scripts/remote-recovery.py"
cargo fmt --manifest-path "$root/Cargo.toml" --all
cargo fmt --manifest-path "$root/Cargo.toml" --all --check > "$run/m5-fmt.txt" 2>&1
cargo clippy --locked --manifest-path "$root/Cargo.toml" --all-targets -- -D warnings > "$run/m5-clippy.txt" 2>&1
cargo test --locked --manifest-path "$root/Cargo.toml" > "$run/m5-tests.txt" 2>&1
cargo build --locked --release --manifest-path "$root/Cargo.toml" --example recover_weights > "$run/recover-weights-build.txt" 2>&1
cargo +1.85.0 fmt --all --check > "$run/root-fmt-msrv.txt" 2>&1
cargo +1.85.0 clippy --workspace --all-targets --all-features -- -D warnings > "$run/root-clippy-msrv.txt" 2>&1
cargo +1.85.0 test --workspace > "$run/root-test-msrv.txt" 2>&1
cargo +1.85.0 build --release --workspace > "$run/root-build-msrv.txt" 2>&1
printf 'CPU checks and real WSL Drive fixture verified: %s\n' "$run"
