#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# CPU-only compiler/libraries, no driver install or GPU allocation.
toolkit=/home/dasetwa/.local/opt/packtok-cuda-12.4.1
mkdir -p "$toolkit" data/cuda-downloads
while IFS=$'\t' read -r component relative sha bytes; do
    archive="data/cuda-downloads/$(basename "$relative")"
    if ! test -f "$archive"; then
        curl --fail --location --proto '=https' --max-time 300 --max-filesize "$bytes" \
          "https://developer.download.nvidia.com/compute/cuda/redist/$relative" -o "$archive.part"
        mv "$archive.part" "$archive"
    fi
    printf '%s  %s\n' "$sha" "$archive" | sha256sum --check
    if ! test -f "$toolkit/$component.complete"; then
        tar -xJf "$archive" --strip-components=1 -C "$toolkit"
        cp "$toolkit/LICENSE" "provenance/licenses/$component-LICENSE" 2>/dev/null || true
        touch "$toolkit/$component.complete"
    fi
done < provenance/cuda-components.tsv
"$toolkit/bin/nvcc" --version
