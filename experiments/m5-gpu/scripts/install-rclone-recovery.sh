#!/usr/bin/env bash
set -euo pipefail
# External backup tooling only; installed in the dedicated user's WSL home.
dest=/home/dasetwa/.local/share/packtok-rclone
mkdir -p "$dest"
curl --fail --location --max-time 30 https://downloads.rclone.org/version.txt -o "$dest/version.txt"
version=$(head -1 "$dest/version.txt" | tr -d '\r\n' | sed 's/^rclone //')
[[ "$version" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]
archive="rclone-$version-linux-amd64.zip"
curl --fail --location --max-time 180 "https://downloads.rclone.org/$version/$archive" -o "$dest/$archive"
curl --fail --location --max-time 30 "https://downloads.rclone.org/$version/SHA256SUMS" -o "$dest/SHA256SUMS-$version"
(cd "$dest"; grep " $archive\$" "SHA256SUMS-$version" | sha256sum -c -)
python3 - "$dest/$archive" "$dest" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(sys.argv[2])
PY
install -m 755 "$dest/rclone-$version-linux-amd64/rclone" /home/dasetwa/.local/bin/rclone
/home/dasetwa/.local/bin/rclone version
sha256sum "$dest/$archive" /home/dasetwa/.local/bin/rclone
printf 'Protected OAuth configuration must be created interactively by the user. No tokens are recorded here.\n'
