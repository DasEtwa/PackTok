#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/dasetwa/.local/bin:/home/dasetwa/.cargo/bin:$PATH
cd /home/dasetwa/projects/PackTok
root=experiments/m5-gpu
win=/mnt/c/Users/DasEtwa/PackTok/experiments/m5-gpu
mkdir -p "$root/examples"
for script in drive-backup.py remote-recovery.py l4-recovery.py test-recovery.py check-recovery-cpu.sh; do cp "$win/scripts/$script" "$root/scripts/$script"; done
cp "$win/examples/recover_weights.rs" "$root/examples/"
run="$root/provenance/recovery-20261008/checks-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir "$run"
printf '%s\n' "$run" > "$root/provenance/recovery-20261008/affected-checks-path.txt"
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
python3 - "$run" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path('experiments/m5-gpu')
run=pathlib.Path(sys.argv[1])
paths=list(run.glob('*.txt'))+[root/'scripts/test-recovery.py',root/'examples/recover_weights.rs']
record=dict(status='CPU_CHECKS_PASS',verified_files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(run/'CPU_CHECKS_PASS.json').write_text(json.dumps(record,indent=2)+'\n')
PY
printf 'CPU suites complete; no GPU allocation: %s\n' "$run"
