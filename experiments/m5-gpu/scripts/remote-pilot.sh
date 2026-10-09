#!/usr/bin/env bash
set -euo pipefail
cd /content/packtok-m5
mkdir results
monitor_pid=''
finish() {
  rc=$?
  trap - EXIT INT TERM
  if test -n "$monitor_pid"; then
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  printf '%s\n' "$rc" > results/exit-code.txt
  if test -f bootstrap-console.txt; then cp bootstrap-console.txt results/; fi
  tar -czf /content/packtok-m5-results.tar.gz results
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%s\n' "$PACKTOK_M5_GPU_APPROVAL" > results/approval-reference.txt
cp frozen-source.json results/
sha256sum configs/pilot-2k-v1.json configs/primary.json provenance/corpus-v2-manifest.json \
  data/prepared-v2/*.seq artifacts/corpus-v2/* > results/input-sha256.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > results/gpu.txt
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv,noheader,nounits --loop-ms=200 \
  > results/gpu-samples.csv 2>&1 &
monitor_pid=$!

libpath="$PWD/runtime:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib64-nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for p in /usr/local/lib/python*/dist-packages/nvidia/*/lib /usr/local/lib/python*/site-packages/nvidia/*/lib; do
  if test -d "$p"; then libpath="$libpath:$p"; fi
done
export LD_LIBRARY_PATH="$libpath" NVIDIA_TF32_OVERRIDE=0
./runtime/ld-linux-x86-64.so.2 --library-path "$libpath" --list ./packtok-m5 \
  > results/runtime-libraries.txt 2>&1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qx 'NVIDIA L4'
./runtime/ld-linux-x86-64.so.2 --library-path "$libpath" ./packtok-m5 \
  plan-extended configs/pilot-2k-v1.json data/prepared-v2 results/pilot-plan.json

for variant in A C; do
  out="results/$variant"
  mkdir "$out"
  printf '%s %s\n' "$variant" "$(date -u +%FT%TZ)" >> results/execution-order.txt
  timeout --signal=TERM --kill-after=10 690 env PACKTOK_M5_GPU_APPROVAL="$PACKTOK_M5_GPU_APPROVAL" \
    ./runtime/ld-linux-x86-64.so.2 --library-path "$libpath" ./packtok-m5 \
    train-extended configs/pilot-2k-v1.json data/prepared-v2 "$out" "$variant" T 20261008 \
    "$PACKTOK_M5_GPU_APPROVAL" > "$out/console.txt" 2>&1
  test "$(grep -c '"stage":"final"' "$out/metrics.jsonl")" = 1
done
printf 'M5_A_C_PILOT_COMPLETE\n' > results/pilot-status.txt
