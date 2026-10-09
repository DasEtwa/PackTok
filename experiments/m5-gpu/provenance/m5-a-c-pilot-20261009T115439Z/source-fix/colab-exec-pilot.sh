#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: colab-exec-pilot.sh SESSION BRIDGE_FILE TIMEOUT_SECONDS" >&2
  exit 2
fi
: "${PACKTOK_M5_GPU_APPROVAL:?required PACKTOK_M5_GPU_APPROVAL is missing}"
exec colab exec -s "$1" -f "$2" --timeout "$3" \
  --env "PACKTOK_M5_GPU_APPROVAL=$PACKTOK_M5_GPU_APPROVAL"
