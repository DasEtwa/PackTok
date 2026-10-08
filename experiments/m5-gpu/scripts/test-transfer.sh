#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
fixture=$(mktemp -d)
trap 'rm -rf -- "$fixture"' EXIT
mkdir "$fixture/bin" "$fixture/parts" "$fixture/remote"
printf 'first deterministic transport fixture\n' > "$fixture/archive"
dd if=/dev/zero bs=1024 count=90 status=none >> "$fixture/archive"
printf 'last deterministic transport fixture\n' >> "$fixture/archive"
split -b 32768 -d -a 3 "$fixture/archive" "$fixture/parts/packtok-m5-bundle.tar.gz.part"
cat > "$fixture/bin/colab" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
test "$1 $2 $3" = 'upload -s packtok-m5'
cp "$4" "$MOCK_REMOTE/${5##*/}"
MOCK
chmod +x "$fixture/bin/colab"
export MOCK_REMOTE="$fixture/remote" PATH="$fixture/bin:$PATH"
bash scripts/upload-bundle.sh "$fixture/parts"
cat "$fixture/remote/"*.part??? > "$fixture/reassembled"
cmp "$fixture/archive" "$fixture/reassembled"
printf 'PASS CPU-mocked chunk transfer/order/byte identity; no GPU allocated.\n'
