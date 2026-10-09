#!/usr/bin/env bash
# CPU-only collection and durable backup, after the owned allocation is absent.
set -euo pipefail
export PATH=/home/dasetwa/.local/bin:/home/dasetwa/.cargo/bin:$PATH
cd /home/dasetwa/projects/PackTok
root=experiments/m5-gpu
attempt="$root/provenance/l4-recovery-abafb6d8f2eb4e70a6ff3c3447bf917f"
run="$root/provenance/recovery-20261008/outcome-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir "$run"
colab sessions > "$run/sessions-final.txt" 2>&1
grep -q 'No active sessions found on server.' "$run/sessions-final.txt"
colab usage > "$run/usage-final.txt" 2>&1
git rev-parse HEAD > "$run/source-commit.txt"
git status --short --branch > "$run/worktree.txt"
python3 "$root/scripts/test-recovery.py" > "$run/regressions.txt" 2>&1
python3 -m py_compile /mnt/c/Users/DasEtwa/PackTok/experiments/m5-gpu/scripts/receive-drive-client.py
bash -n "$root/scripts/launch-l4-recovery.sh"
bash -n /mnt/c/Users/DasEtwa/PackTok/experiments/m5-gpu/scripts/check-wsl-supervision.sh
(
  cd "$root/bundle-v4"
  sha256sum -c bundle.sha256
  sha256sum -c source.sha256
) > "$run/frozen-hashes-after.txt" 2>&1
cp "$root/bundle-v4/preflight-attempted" "$run/one-request-consumed.json"
# Keep public experiment evidence only. No config, OAuth output or user journal.
tar -czf "$run/lifecycle-evidence.tar.gz" -C "$root/provenance" "$(basename "$attempt")"
python3 - "$run" <<'PY'
import json, pathlib, sys
run=pathlib.Path(sys.argv[1])
metadata=json.loads(pathlib.Path('experiments/m5-gpu/provenance/recovery-20261008/cpu-final-20261008T160630-344/fixture-metadata.json').read_text())
metadata['kind']='diagnostic-archive'
metadata['variant']='infrastructure-l4-recovery'
(run/'archive-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
PY
python3 "$root/scripts/drive-backup.py" "$run/lifecycle-evidence.tar.gz" "$run/archive-metadata.json" "$run/drive-archive-readback" --remote packtok-drive-own > "$run/drive-archive-result.txt" 2>&1
sha256sum "$run/lifecycle-evidence.tar.gz" > "$run/archive.sha256"
git diff --check > "$run/native-diff-check.txt" 2>&1
printf 'Local lifecycle evidence and full-content Drive archive readback PASS: %s\n' "$run"
