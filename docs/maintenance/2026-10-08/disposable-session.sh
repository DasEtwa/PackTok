#!/usr/bin/env bash
set -euo pipefail
base=/home/dasetwa/projects/PackTok/docs/maintenance/2026-10-08/wsl-session
printf '%s\n' "${XDG_SESSION_ID:?PAM session required}" > "$base/initiating-session.txt"
printf '%s\n' "$$" > "$base/initiating-linux-pid.txt"
/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -File C:\\Users\\DasEtwa\\PackTok\\target\\maintenance-host\\launch-wsl-supervised.ps1 -Mode cpu-mock -Fixture "$base" -MockMode orphan -MockSeconds 310 -WorkSeconds 340 -RuntimeSeconds 360 -Receipt C:\\Users\\DasEtwa\\PackTok\\target\\maintenance-host\\session.json
printf 'launcher-dispatched\n' > "$base/initiating-ready.txt"
exec /usr/bin/sleep 120
