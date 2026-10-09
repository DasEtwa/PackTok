#!/usr/bin/env bash
set -euo pipefail
cd /home/dasetwa/projects/PackTok
root=experiments/m5-gpu
run="$root/provenance/recovery-20261008/cpu-check-$(date -u +%Y%m%dT%H%M%S)-$$"
mkdir -p "$run"
bundle=$(realpath "$root/bundle-v4")
{
 (cd "$bundle"; sha256sum -c bundle.sha256; sha256sum -c source.sha256)
 test ! -f "$bundle/preflight-attempted"
 test ! -f "$bundle/owned-session"
 (cd "$bundle/packtok-m5"; sha256sum -c SHA256SUMS.txt)
 cmp "$bundle/packtok-m5/packtok-m5" "$root/target-cuda/release/packtok-m5"
 test ! -f "$bundle/packtok-m5/runtime/libcuda.so.1"
 libpath="$bundle/packtok-m5/runtime:/usr/lib/wsl/lib"
 "$bundle/packtok-m5/runtime/ld-linux-x86-64.so.2" --library-path "$libpath" --list "$bundle/packtok-m5/packtok-m5"
 temporary=$(mktemp -d)
 mkdir -p "$temporary/experiments/m5-gpu/provenance"
 # reference creates an immutable output: an isolated fresh directory preserves history.
 (cd "$temporary"; "$bundle/packtok-m5/runtime/ld-linux-x86-64.so.2" --library-path "$libpath" "$bundle/packtok-m5/packtok-m5" reference)
 cmp "$temporary/experiments/m5-gpu/provenance/cpu-reference.json" "$bundle/packtok-m5/cpu-reference.json"
 cat "$bundle/CPU_READY"
} > "$run/bundle-v4-verification-file-backed.txt" 2>&1
"$root/target/release/packtok-m5" verify-prepared > "$run/prepared-verification.txt" 2>&1
bash "$root/scripts/test-transfer.sh" > "$run/transfer-original.txt" 2>&1
bash "$root/scripts/test-lifecycle.sh" > "$run/lifecycle-original.txt" 2>&1
printf 'Bundle, source, executable, driver exclusion, loader, exact CPU reference, prepared inputs and original mocks PASS: %s\n' "$run"
