#!/usr/bin/env bash
set -euo pipefail
cd /content/packtok-m5
mkdir results
monitor_pid=''
finish() {
  rc=$?
  trap - EXIT INT TERM
  if test -n "$monitor_pid"; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
  printf '%s\n' "$rc" > results/exit-code.txt
  if test -f bootstrap-console.txt; then cp bootstrap-console.txt results/; fi
  tar -czf /content/packtok-m5-results.tar.gz results
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'Bundle/source/data hashes verified on local WSL before upload; checkpoint hashes are computed on WSL after release.
' > results/integrity-policy.txt
cp frozen-source.json results/
{
 date -u
 uname -a
 nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
 cat /etc/os-release
 printf "NVIDIA_TF32_OVERRIDE=0\ninherited_ld_library_path=%s\n" "${LD_LIBRARY_PATH:-}"
 for library_dir in /usr/local/cuda*/lib64 /usr/local/nvidia/lib64 /usr/lib64-nvidia /usr/lib/x86_64-linux-gnu /usr/local/lib/python*/dist-packages/nvidia/*/lib /usr/local/lib/python*/site-packages/nvidia/*/lib; do
   if test -d "$library_dir"; then
     find "$library_dir" -maxdepth 1 \( -name "libcuda.so*" -o -name "libcublas*.so*" -o -name "libcurand.so*" \) -printf "%p\n"
   fi
 done
} > results/environment.txt
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qx 'NVIDIA L4'
# CPU-built ELF loader/runtime; CUDA driver/libraries must already be present.
# No remote package installation, cloning, corpus preparation or compilation.
libpath="$PWD/runtime:/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/lib64-nvidia:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for p in /usr/local/lib/python*/dist-packages/nvidia/*/lib /usr/local/lib/python*/site-packages/nvidia/*/lib; do
  if test -d "$p"; then libpath="$libpath:$p"; fi
done
export LD_LIBRARY_PATH="$libpath" NVIDIA_TF32_OVERRIDE=0
./runtime/ld-linux-x86-64.so.2 --library-path "$libpath" --list ./packtok-m5 > results/runtime-libraries.txt 2>&1
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu --format=csv,noheader,nounits --loop-ms=200 > results/gpu-samples.csv 2>&1 &
monitor_pid=$!
# Foreground, supervised, hard deadline. CUDA work only.
timeout --signal=TERM --kill-after=10 540 ./runtime/ld-linux-x86-64.so.2 --library-path "$libpath" ./packtok-m5 preflight data cpu-reference.json results/gate > results/rust-console.txt 2>&1
