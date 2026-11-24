#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${PACKTOK_M7_GPU_APPROVAL:?approved M7 L4 reference required}"
export NVIDIA_TF32_OVERRIDE=0 RAYON_NUM_THREADS=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee hardware.csv
test "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = 'NVIDIA L4'
sha256sum --check inputs.sha256
libpath="$PWD/runtime:/usr/local/nvidia/lib64:/usr/lib64-nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
loader=(./runtime/ld-linux-x86-64.so.2 --library-path "$libpath")
"${loader[@]}" --list ./packtok-m7
mkdir results
for row in "control A 96" "packed C 96" "restart A 40" "restart A 96"; do
    read -r run variant stop <<< "$row"
    timeout --signal=TERM --kill-after=5 45 "${loader[@]}" ./packtok-m7 \
        train-l4 dataset tokens "results/$run" "$variant" "$stop" "$PACKTOK_M7_GPU_APPROVAL" \
        > "results/$run-$stop.console.jsonl" 2>&1
done
