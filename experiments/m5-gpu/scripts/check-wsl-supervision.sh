#!/usr/bin/env bash
# Real local lifetime regression; never calls Colab or allocates GPU resources.
set -euo pipefail
cd /home/dasetwa/projects/PackTok
run="experiments/m5-gpu/provenance/recovery-20261008/wsl-lifetime-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir "$run"
unit="packtok-wsl-lifetime-$(date -u +%Y%m%dT%H%M%S)-$$"
start=$(date +%s)
systemd-run --user --wait --unit="$unit" --property=RuntimeMaxSec=60 \
  --property=TimeoutStopSec=5 /usr/bin/sleep 40 > "$run/waited-service.txt" 2>&1 &
waiter=$!
# The enclosing WSL client stays open while this waiter runs. The former
# detached service was stopped after the last user session ended (~10 s).
sleep 20
systemctl --user show "$unit.service" -p ActiveState -p MainPID -p Result > "$run/during.txt"
grep -qx 'ActiveState=active' "$run/during.txt"
wait "$waiter"
elapsed=$(($(date +%s)-start))
systemctl --user show "$unit.service" -p ActiveState -p Result -p ExecMainStatus > "$run/after.txt"
grep -qx 'Result=success' "$run/after.txt"
grep -qx 'ExecMainStatus=0' "$run/after.txt"
test "$elapsed" -ge 40
printf 'PASS: foreground WSL lifetime retained for %s seconds; GPU calls=0\n' "$elapsed" > "$run/result.txt"
sha256sum experiments/m5-gpu/scripts/launch-l4-recovery.sh > "$run/launcher-verified.sha256"
cat "$run/result.txt"
printf 'Evidence: %s\n' "$run"
