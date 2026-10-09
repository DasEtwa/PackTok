#!/usr/bin/env bash
# CPU-only real entrypoint tests; no Colab credentials, network or GPU calls.
set -euo pipefail
cd /home/dasetwa/projects/PackTok
output="docs/maintenance/supervision-$(date -u +%Y%m%dT%H%M%S)-$$"
python3 experiments/m5-gpu/scripts/test-supervision.py "$output"
printf 'CPU supervision evidence: %s\n' "$output"
