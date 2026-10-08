#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
root=experiments/m5-gpu
bundle_name=${PACKTOK_M5_BUNDLE_NAME:-bundle}
case "$bundle_name" in bundle) evidence_tag='';; bundle-v2|bundle-v3) evidence_tag="-${bundle_name#bundle-}";; *) echo "Unsupported bundle name" >&2; exit 2;; esac
bundle_root="$root/$bundle_name"
test ! -e "$bundle_root"
# Build only locally, with the already prepared compiler. No CUDA operations.
export CUDA_ROOT=/home/dasetwa/.local/opt/packtok-cuda-12.4.1
export PATH="$CUDA_ROOT/bin:$PATH" CUDA_COMPUTE_CAP=89 RAYON_NUM_THREADS=2 CARGO_BUILD_JOBS=2
export LIBRARY_PATH="$CUDA_ROOT/lib:$CUDA_ROOT/lib64:/usr/lib/wsl/lib"
git diff HEAD --exit-code -- "$root/src" "$root/Cargo.toml" "$root/Cargo.lock" "$root/scripts" "$root/configs"
test "$(git branch --show-current)" = m5-gpu-transformer
test -f "$root/provenance/preparation-v2-metrics.json"
test -f "$root/provenance/schedule-plan.json"
test -f "$root/verification/transformer-debug-9.txt"
test -f "$root/verification/transformer-release-9.txt"
test -f "$root/verification/transformer-clippy-8.txt"
grep -q '18 passed; 0 failed' "$root/verification/transformer-debug-9.txt"
grep -q '18 passed; 0 failed' "$root/verification/transformer-release-9.txt"
# All source/data checks and both CPU suites precede this packaging step.
cargo build --locked --release --features cuda --manifest-path "$root/Cargo.toml" --target-dir "$root/target-cuda" > "$root/verification/cuda-build-bundle${evidence_tag}.txt" 2>&1
"$root/target/release/packtok-m5" verify-prepared > "$root/verification/prepared-integrity-before-bundle${evidence_tag}.txt" 2>&1
# Never replace a failed preflight bundle. Recovery package gets a distinct path.
mkdir "$bundle_root"
bundle="$bundle_root/packtok-m5"
mkdir -p "$bundle/runtime" "$bundle/data" "$bundle/source" "$bundle/artifacts"
cp "$root/target-cuda/release/packtok-m5" "$bundle/packtok-m5"
cp "$root/scripts/remote.sh" "$bundle/remote.sh"
cp "$root/provenance/cpu-reference.json" "$bundle/cpu-reference.json"
cp "$root/data/prepared-v2/A-train.seq" "$root/data/prepared-v2/C-train.seq" "$bundle/data/"
cp "$root/artifacts/corpus-v2/A-0.packtok" "$root/artifacts/corpus-v2/C-0.packtok" "$root/artifacts/corpus-v2/"*.mapping "$bundle/artifacts/"
# Compatibility with a remote older glibc: carry this existing Linux CPU loader
# and ordinary runtime files; never the WSL driver or compiler/static SDK files.
for library in ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1 libgcc_s.so.1 libstdc++.so.6; do
  cp -L "/lib/x86_64-linux-gnu/$library" "$bundle/runtime/$library"
done
# CPU-acquired, pinned CUDA redistributable libraries remove reliance on a
# Colab image exposing the same sonames. Do not carry the WSL driver.
for library in libcurand.so.10 libcublas.so.12 libcublasLt.so.12; do
  cp -L "$CUDA_ROOT/lib/$library" "$bundle/runtime/$library"
done
for package in libc6 libgcc-s1 libstdc++6; do
  cp "/usr/share/doc/$package/copyright" "$bundle/runtime/$package.copyright"
  cp "/usr/share/doc/$package/copyright" "$root/provenance/licenses/$package-runtime-copyright"
done
for package in libcurand libcublas; do
  cp "$root/provenance/licenses/$package-LICENSE" "$bundle/runtime/$package.LICENSE"
done
dpkg-query -W libc6 libgcc-s1 libstdc++6 > "$root/provenance/portable-runtime-versions${evidence_tag}.txt"
commit=$(git rev-parse HEAD)
cat > "$bundle/frozen-source.json" <<EOF
{"source_commit":"$commit","cpu_freeze":"f3b6c3fb41f46c93e701b585d859b0b9071d65a5","backend":"candle-0.9.1","precision":"FP32","gpu":"NVIDIA L4","compiled_cuda":"12.4.131","compute_capability":"8.9","mode":"preflight-only"}
EOF
git archive HEAD -- "$root/src" "$root/Cargo.toml" "$root/Cargo.lock" "$root/configs" "$root/scripts" M5_GPU_TRANSFORMER.md | tar -x -C "$bundle/source"
(cd "$bundle" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt)
cp "$bundle/SHA256SUMS.txt" "$root/provenance/bundle-input-SHA256SUMS${evidence_tag}.txt"
git ls-files -z -- "$root/src" "$root/Cargo.toml" "$root/Cargo.lock" "$root/scripts" "$root/configs" | while IFS= read -r -d '' file; do
  sha256sum "$(realpath "$file")"
done > "$bundle_root/source.sha256"
"$bundle/runtime/ld-linux-x86-64.so.2" --library-path "$(pwd)/$bundle/runtime:/usr/lib/wsl/lib" --list "$bundle/packtok-m5" > "$root/verification/portable-loader-runtime${evidence_tag}.txt" 2>&1
# Verify CPU reference under the actual bundled loader, still on local CPU.
libpath="$(pwd)/$bundle/runtime:/usr/lib/wsl/lib"
"$bundle/runtime/ld-linux-x86-64.so.2" --library-path "$libpath" "$bundle/packtok-m5" reference > "$root/verification/portable-loader-cpu${evidence_tag}.txt" 2>&1
# Deterministic tar metadata; the source commit and file checksums stay explicit.
tar --sort=name --mtime='UTC 2026-10-08' --owner=0 --group=0 --numeric-owner -C "$bundle_root" -cf - packtok-m5 | gzip -n > "$bundle_root/packtok-m5-bundle.tar.gz"
printf 'source_commit=%s\nmode=preflight-only\n' "$commit" > "$bundle_root/CPU_READY"
(cd "$bundle_root" && sha256sum CPU_READY packtok-m5-bundle.tar.gz > bundle.sha256)
cp "$bundle_root/bundle.sha256" "$root/provenance/bundle-SHA256SUMS${evidence_tag}.txt"
wc -c "$bundle_root/packtok-m5-bundle.tar.gz" "$root/target-cuda/release/packtok-m5" > "$root/provenance/bundle-sizes${evidence_tag}.txt"
printf 'CPU bundle ready; no GPU allocated.\n'
