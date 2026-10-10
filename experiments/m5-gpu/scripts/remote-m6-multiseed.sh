#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir results
monitor_pid=''
finish() {
  rc=$?
  trap - EXIT INT TERM
  if test -n "$monitor_pid"; then kill -TERM "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
  printf '%s\n' "$rc" > results/exit-code.txt
  if test -f bootstrap-console.txt; then cp bootstrap-console.txt results/; fi
  tar -czf "${PACKTOK_M5_RESULTS_ARCHIVE:-/content/packtok-m5-results.tar.gz}" results
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
: "${PACKTOK_M5_GPU_APPROVAL:?required PACKTOK_M5_GPU_APPROVAL is missing}"
printf '%s\n' "$PACKTOK_M5_GPU_APPROVAL" > results/approval-reference.txt
cp frozen-source.json results/
sha256sum configs/m6-five-seed-2k-v1.json configs/primary.json provenance/corpus-v2-manifest.json provenance/m6-five-seed-initialization.tsv data/prepared-v2/*.seq artifacts/corpus-v2/* > results/input-sha256.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > results/gpu.txt
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv,noheader,nounits --loop-ms=200 > results/gpu-samples.csv 2>&1 &
monitor_pid=$!
libpath="$PWD/runtime:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib64-nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for p in /usr/local/lib/python*/dist-packages/nvidia/*/lib /usr/local/lib/python*/site-packages/nvidia/*/lib; do
  if test -d "$p"; then libpath="$libpath:$p"; fi
done
export LD_LIBRARY_PATH="$libpath" NVIDIA_TF32_OVERRIDE=0
./runtime/ld-linux-x86-64.so.2 --library-path "$libpath" --list ./packtok-m5 > results/runtime-libraries.txt 2>&1
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qx 'NVIDIA L4'
./runtime/ld-linux-x86-64.so.2 --library-path "$libpath" ./packtok-m5 plan-extended configs/m6-five-seed-2k-v1.json data/prepared-v2 results/experiment-plan.json
start_at=20261009:A
if printenv PACKTOK_M6_START_AT >/dev/null; then start_at="$(printenv PACKTOK_M6_START_AT)"; fi
case "$start_at" in
  20261009:A|20261009:C|20261010:A|20261010:C|20261011:A|20261011:C|20261012:A|20261012:C) ;;
  *) echo "invalid PACKTOK_M6_START_AT: $start_at" >&2; exit 2 ;;
esac
printf "%s\n" "$start_at" > results/start-at.txt
start_found=0
run_count=0
printf 'seed\tvariant\tstart_utc\toutput\n' > results/execution-order.tsv
for seed in 20261009 20261010 20261011 20261012; do
  for variant in A C; do
    if [[ "$start_found" = 0 ]]; then
      if [[ "$seed:$variant" != "$start_at" ]]; then continue; fi
      start_found=1
    fi
    out="results/seed-$seed/$variant"
    test ! -e "$out" || { echo "refusing to overwrite existing output: $out" >&2; exit 73; }
    mkdir -p "$out"
    printf '%s\t%s\t%s\t%s\n' "$seed" "$variant" "$(date -u +%FT%TZ)" "$out" >> results/execution-order.tsv
    timeout --signal=TERM --kill-after=10 600 env PACKTOK_M5_GPU_APPROVAL="$PACKTOK_M5_GPU_APPROVAL" \
      ./runtime/ld-linux-x86-64.so.2 --library-path "$libpath" ./packtok-m5 \
      train-extended configs/m6-five-seed-2k-v1.json data/prepared-v2 "$out" "$variant" T "$seed" "$PACKTOK_M5_GPU_APPROVAL" \
      > "$out/console.txt" 2>&1
    test "$(grep -c '"stage":"final"' "$out/metrics.jsonl")" = 1
    test "$(grep -c '"stage":"validation"' "$out/metrics.jsonl")" -eq 5
    grep '"stage":"final"' "$out/metrics.jsonl" | grep -q '"updates":2000'
    grep '"stage":"final"' "$out/metrics.jsonl" | grep -q '"targets":4096000'
    expected_init=$(awk -F '\t' -v seed="$seed" '$1 == seed {print $2}' provenance/m6-five-seed-initialization.tsv)
    actual_init=$(grep '"stage":"initialization"' "$out/metrics.jsonl" | sed -n 's/.*"initialization_sha256":"\([a-f0-9]*\)".*/\1/p')
    test -n "$expected_init" && test "$actual_init" = "$expected_init"
    sha256sum "$out/latest.resume.safetensors" "$out/metrics.jsonl" > "$out/SHA256SUMS.txt"
    run_count=$((run_count + 1))
  done
done
test "$start_found" = 1
printf "M6_RUN_SEQUENCE_COMPLETE; start_at=%s; runs=%s; historical seed 20261008 reused from M5\n" "$start_at" "$run_count" > results/pilot-status.txt

