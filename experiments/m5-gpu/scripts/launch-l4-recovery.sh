#!/usr/bin/env bash
# Keep a WSL client open until the bounded supervisor service has finished.
set -euo pipefail
export PATH=/home/dasetwa/.local/bin:/home/dasetwa/.cargo/bin:$PATH
cd /home/dasetwa/projects/PackTok
root=experiments/m5-gpu
test -f "$root/provenance/recovery-20261008/PREFLIGHT_READY.json"
test ! -e "$root/bundle-v4/preflight-attempted"
run="$root/provenance/recovery-20261008/launcher-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir "$run"
cp "$root/scripts/launch-l4-recovery.sh" "$run/launcher-used.sh"
sha256sum "$root/scripts/launch-l4-recovery.sh" > "$run/launcher.sha256"
# The Python supervisor bounds work to 1620 s and reserves 180 s for cleanup.
# Systemd services alone do not keep WSL/user sessions alive. --wait retains
# the foreground WSL client. TERM is sent at 1680 s, KILL at 1800 s. No restart.
unit="packtok-m5-recovery-$(date -u +%Y%m%dT%H%M%S)-$$"
printf '%s.service\n' "$unit" > "$run/unit.txt"
printf '%s\n' "$run" > "$root/provenance/recovery-20261008/active-launcher-path.txt"
printf 'WSL supervisor starting: unit=%s log=%s/supervisor.txt\n' "$unit" "$run"
systemd-run --user --wait --unit="$unit" \
  --property=RuntimeMaxSec=1680 --property=TimeoutStopSec=120 \
  --property=KillMode=mixed --property=Restart=no \
  --property="StandardOutput=append:$PWD/$run/supervisor.txt" \
  --property=StandardError=inherit --setenv="PATH=$PATH" \
  --working-directory="$PWD" /usr/bin/python3 "$PWD/$root/scripts/l4-recovery.py" --preflight \
  > "$run/service-start.txt" 2>&1
printf 'WSL supervisor finished: unit=%s log=%s/supervisor.txt\n' "$unit" "$run"
