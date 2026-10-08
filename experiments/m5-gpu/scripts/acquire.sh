#!/usr/bin/env bash
set -euo pipefail
# Local Ubuntu-24.04 CPU only. Exact downloaded bytes kept outside Git.
cd "$(dirname "$0")/.."
mkdir -p data/sources provenance/licenses
for id in 100 145 766 883 1023 7469 2403 2404 5323 24782 2335 2336 2337 2338 2339 2340 2341 2342; do
  path="data/sources/pg${id}.txt"
  if ! test -f "$path"; then
    curl --fail --location --proto '=https' --max-time 120 --max-filesize 10485760 \
      "https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt" -o "$path.part"
    mv "$path.part" "$path"
  fi
done
path=data/sources/rust-1.85.0.tar.gz
if ! test -f "$path"; then
  curl --fail --location --proto '=https' --max-time 180 --max-filesize 67108864 \
    https://codeload.github.com/rust-lang/rust/tar.gz/4d91de4e48198da2e33413efdcd9cd2cc0c46688 -o "$path.part"
  mv "$path.part" "$path"
fi
if ! test -d data/sources/rust; then
  mkdir data/sources/rust
  tar -xzf "$path" --strip-components=1 -C data/sources/rust --wildcards \
    '*/library/*' '*/LICENSE-APACHE' '*/LICENSE-MIT' '*/Cargo.toml'
fi
cp data/sources/rust/LICENSE-APACHE provenance/licenses/Rust-LICENSE-APACHE
cp data/sources/rust/LICENSE-MIT provenance/licenses/Rust-LICENSE-MIT
if test -f provenance/source-SHA256SUMS.txt; then
  sha256sum -c provenance/source-SHA256SUMS.txt
else
  sha256sum data/sources/pg*.txt data/sources/rust-1.85.0.tar.gz > provenance/source-SHA256SUMS.txt
fi
wc -c data/sources/pg*.txt data/sources/rust-1.85.0.tar.gz > provenance/source-sizes.txt
find data/sources/rust/library -type f \( -name '*.rs' -o -name '*.toml' -o -name '*.json' \) -printf '%s\n' | awk '{n+=$1;c++}END{print "Rust eligible files:",c,"bytes:",n}' >> provenance/source-sizes.txt
