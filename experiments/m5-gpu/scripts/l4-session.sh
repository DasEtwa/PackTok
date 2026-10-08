#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# No final-training mode: explicit approval and an approved budget are prerequisites
# for a future final-run wrapper. This script authorizes only the first preflight.
if test "${1:-}" != preflight || test "$#" -ne 1; then
  printf 'Only preflight is implemented; full training requires user budget approval.\n' >&2
  exit 2
fi
bundle=$(realpath "${PACKTOK_M5_BUNDLE:-bundle}")
logroot=${PACKTOK_M5_LOGROOT:-provenance}
mkdir -p "$logroot"
run="$logroot/l4-preflight-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir "$run"
marker="$bundle/owned-session"
allocated=0
start=0
worker_pid=''
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if test -n "$worker_pid"; then
    kill -TERM "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
  fi
  if test "$allocated" = 1; then
    stop_start=$(date +%s)
    timeout 25 colab stop -s packtok-m5 > "$run/stop.txt" 2>&1 ||
      timeout 25 colab stop -s packtok-m5 >> "$run/stop.txt" 2>&1 || true
    printf '%s
' "$(( $(date +%s) - start ))" > "$run/allocation-request-to-stop-seconds.txt"
    printf '%s
' "$(( $(date +%s) - stop_start ))" > "$run/stop-seconds.txt"
    timeout 20 colab sessions > "$run/sessions-after.txt" 2>&1 || true
    timeout 20 colab usage > "$run/usage-after.txt" 2>&1 || true
    date -u +%FT%TZ > "$run/cleanup-completed-at.txt"
    printf '%s\n' "$(( $(date +%s) - start ))" > "$run/allocation-request-to-cleanup-seconds.txt"
    endpoint=''
    if test -f "$run/endpoint.txt"; then endpoint=$(cat "$run/endpoint.txt"); fi
    verified=0
    if grep -Fq 'No active sessions found on server.' "$run/sessions-after.txt"; then
      verified=1
    elif test -n "$endpoint" &&
         ! grep -Fq "$endpoint" "$run/sessions-after.txt" &&
         grep -Eq '\[.*\].*Hardware:' "$run/sessions-after.txt"; then
      verified=1
    fi
    if test "$verified" = 1; then
      mv "$marker" "$run/ownership-marker.txt"
      cp "$run/cleanup-completed-at.txt" "$run/released-verified-at.txt"
      printf 'Owned session/endpoint absent after stop.\n' > "$run/release-verification.txt"
    else
      printf 'RELEASE NOT CONFIRMED: run colab stop -s packtok-m5; colab sessions; colab usage.\n' > "$run/release-verification.txt"
      rc=1
    fi
  fi
  printf '%s\n' "$rc" > "$run/exit-code.txt"
  printf 'Lifecycle logs: %s\n' "$run"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if test -f "$marker"; then
  # Only this wrapper creates this ownership marker. Recover its stale alias,
  # leave all other assignments alone, and end the invocation before new work.
  allocated=1
  start=$(date +%s)
  printf 'Stale owned-session marker found; releasing it before any new allocation.\n' >&2
  exit 1
fi
# Local preparation checks precede allocation and API work.
test -f "$bundle/CPU_READY"
(cd "$bundle" && sha256sum -c bundle.sha256) > "$run/bundle-integrity.txt"
sha256sum -c "$bundle/source.sha256" > "$run/source-integrity.txt"
timeout 30 colab version > "$run/cli-version.txt" 2>&1
timeout 30 colab sessions > "$run/sessions-before.txt" 2>&1
timeout 30 colab usage > "$run/usage-before.txt" 2>&1
if grep -Fq '[packtok-m5]' "$run/sessions-before.txt"; then
  printf 'Existing alias is not owned by this wrapper; no allocation or stop performed.\n' >&2
  exit 1
fi
# Refuse unknown CLI output: do not interpret an error as an empty inventory.
grep -Eq 'No active sessions found on server|\[.*\].*Hardware:' "$run/sessions-before.txt"
grep -Eq '^Current balance: [0-9]+\.[0-9]+ compute units$' "$run/usage-before.txt"
if test -f "$bundle/preflight-attempted"; then
  printf 'This prepared bundle already requested its preflight; no automatic allocation retry.\n' >&2
  exit 1
fi
start=$(date +%s)
printf '%s\n' "$start" > "$bundle/preflight-attempted"
printf 'session=packtok-m5\nmode=preflight\nstarted_epoch=%s\n' "$start" > "$marker"
allocated=1
export PACKTOK_M5_RUN="$run" PACKTOK_M5_BUNDLE="$bundle"
# 18 minutes for provision/upload/CUDA work/download, leaving 2 minutes for cleanup.
# The deadline starts before requesting the L4; no --keep or detached remote job.
timeout --signal=TERM --kill-after=15 1080 bash scripts/l4-phase.sh &
worker_pid=$!
if wait "$worker_pid"; then
  worker_rc=0
else
  worker_rc=$?
fi
worker_pid=''
exit "$worker_rc"
