#!/usr/bin/env bash
# This WSL client must be launched independently by launch-wsl-supervised.ps1.
set -euo pipefail
cd /home/dasetwa/projects/PackTok
root=experiments/m5-gpu
if [[ ${1:-} == --cpu-mock ]]; then
  fixture=$(realpath -- "$2")
  # Only the checked-in fake transport may enter this mode. No real Colab PATH.
  python3 "$root/scripts/supervision-fixture.py" validate "$fixture"
  export PATH="$fixture/bin:/usr/bin:/bin"
  export MOCK_ROOT="$fixture" MOCK_MODE="$3" MOCK_EXEC_SECONDS="$4"
  work_seconds="$5" runtime="$6" stop_seconds=5
  run="$fixture/launcher"; mkdir "$run"
  worker_root="$fixture/experiments/m5-gpu"
else
  test "$#" -eq 0
  export PATH=/home/dasetwa/.local/bin:/home/dasetwa/.cargo/bin:$PATH
  test -f "$root/provenance/recovery-20261008/PREFLIGHT_READY.json"
  test ! -e "$root/bundle-v4/preflight-attempted"
  work_seconds=1620; runtime=1680; stop_seconds=120
  run="$root/provenance/recovery-20261008/launcher-$(date -u +%Y%m%dT%H%M%S)-$$"
  mkdir "$run"
  worker_root="$PWD/$root"
fi
run=$(realpath -- "$run")
unit="packtok-supervised-$(date -u +%Y%m%dT%H%M%S)-$$"
printf '%s.service\n' "$unit" > "$run/unit.txt"
cp "$root/scripts/launch-l4-recovery.sh" "$run/launcher-used.sh"
sha256sum "$root/scripts/launch-l4-recovery.sh" > "$run/launcher.sha256"
printf 'pid=%s start=%s\n' "$$" "$(date -u +%FT%TZ)" > "$run/client.txt"
# --wait bounds the independent host WSL client to the service lifetime.
# KillMode=mixed lets the Python parent release ownership on handled TERM;
# after TimeoutStopSec systemd kills every remaining process in this cgroup.
set +e
systemd-run --user --wait --unit="$unit" \
  --property="RuntimeMaxSec=$runtime" --property="TimeoutStopSec=$stop_seconds" \
  --property=KillMode=mixed --property=Restart=no \
  --property="StandardOutput=append:$run/supervisor.txt" \
  --property=StandardError=inherit --setenv="PATH=$PATH" \
  --setenv="MOCK_ROOT=${MOCK_ROOT:-}" --setenv="MOCK_MODE=${MOCK_MODE:-}" \
  --setenv="MOCK_EXEC_SECONDS=${MOCK_EXEC_SECONDS:-0}" \
  --working-directory="$PWD" /usr/bin/python3 "$PWD/$root/scripts/l4-recovery.py" \
  --preflight --root "$worker_root" --work-seconds "$work_seconds" \
  > "$run/service-start.txt" 2>&1
rc=$?
set -e
systemctl --user show "$unit.service" -p ActiveState -p SubState -p Result \
  -p ExecMainStatus -p ExecMainCode -p ControlGroup > "$run/service-after.txt"
journalctl --user -u "$unit.service" --no-pager -o short-iso-precise > "$run/journal.txt"
printf '%s\n' "$rc" > "$run/launcher-exit.txt"
printf 'end=%s exit=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$run/client.txt"
exit "$rc"
