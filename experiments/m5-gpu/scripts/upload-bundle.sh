#!/usr/bin/env bash
set -euo pipefail
test "$#" = 1
shopt -s nullglob
parts=("$1"/packtok-m5-bundle.tar.gz.part[0-9][0-9][0-9])
test "${#parts[@]}" -gt 0
for part in "${parts[@]}"; do
  printf 'Uploading %s\n' "${part##*/}"
  colab upload -s packtok-m5 "$part" "content/${part##*/}"
done
