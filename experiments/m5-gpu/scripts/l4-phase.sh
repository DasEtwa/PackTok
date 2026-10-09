#!/usr/bin/env bash
set -euo pipefail
: "${PACKTOK_M5_RUN:?parent lifecycle required}"
: "${PACKTOK_M5_BUNDLE:?parent lifecycle required}"
run=$PACKTOK_M5_RUN
bundle=$PACKTOK_M5_BUNDLE
stage() {
  name=$1; limit=$2; shift 2
  a=$(date +%s)
  printf '%s %s START\n' "$(date -u +%FT%TZ)" "$name" >> "$run/stages.txt"
  if timeout --signal=TERM --kill-after=10 "$limit" "$@" > "$run/$name.txt" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  printf '%s %s END seconds=%s exit=%s\n' "$(date -u +%FT%TZ)" "$name" "$(( $(date +%s)-a ))" "$rc" >> "$run/stages.txt"
  return "$rc"
}
stage allocation 180 colab new -s packtok-m5 --gpu L4
stage status 30 colab status -s packtok-m5
sed -n 's/^\[packtok-m5\] \([^ ]*\) |.*/\1/p' "$run/status.txt" > "$run/endpoint.txt"
test -s "$run/endpoint.txt"
grep -Fq 'Hardware: L4' "$run/status.txt"
stage usage-active 30 colab usage
stage upload 420 bash scripts/upload-bundle.sh "$bundle"
# Python here is the official CLI kernel transport only. Rust owns the model.
set +e
stage execute 610 colab exec -s packtok-m5 -f scripts/remote-bridge.py --timeout 600
exec_rc=$?
set -e
stage usage-before-transfer 10 colab usage || true
# One bounded recovery retry. Always return to parent cleanup even if transfer fails.
if ! stage download 60 colab download -s packtok-m5 content/packtok-m5-results.tar.gz "$run/results.tar.gz"; then
  stage download-retry 60 colab download -s packtok-m5 content/packtok-m5-results.tar.gz "$run/results.tar.gz" || exit 1
fi
exit "$exec_rc"
