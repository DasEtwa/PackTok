#!/usr/bin/env bash
# Run as root inside the authorized Ubuntu-24.04 instance for transient loop mounts.
set -euo pipefail
if [[ $# != 7 ]]; then
    echo 'usage: ssd-loop-pilot.sh USER BIN BASE EXISTING_SAMPLE NEW_WORK HDD_PARENT SSD_PARENT' >&2
    exit 2
fi
owner=$1
bin=$2
base=$3
sample=$4
work=$5
hdd_parent=$6
ssd_parent=$7
runner="$(dirname -- "$0")/ssd-pilot.py"
image_size=1610612736
minimum=$((10000000000 + image_size))
available=$(df -B1 --output=avail "$ssd_parent" | tail -1)
[[ "$available" -ge "$minimum" ]]

hdd_image_dir=
ssd_image_dir=
mounts=$(mktemp -d /tmp/packtok-m7-loop-pilot-XXXXXXXX)
cleanup() {
    local label target image_dir
    for label in hdd ssd; do
        target="$mounts/$label"
        if [[ "$label" == hdd ]]; then image_dir=$hdd_image_dir; else image_dir=$ssd_image_dir; fi
        if mountpoint -q "$target" && ! umount -- "$target"; then
            echo "Mount cleanup requires attention: $target" >&2
            continue
        fi
        if [[ -n "$image_dir" ]]; then
            rm -f -- "$image_dir/index.ext4"
            rmdir -- "$image_dir"
        fi
        if [[ -d "$target" ]]; then rmdir -- "$target"; fi
    done
    rmdir -- "$mounts"
}
trap cleanup EXIT
chmod 755 "$mounts"
hdd_image_dir=$(mktemp -d "$hdd_parent/packtok-m7-hdd-image-XXXXXXXX")
ssd_image_dir=$(mktemp -d "$ssd_parent/packtok-m7-ssd-image-XXXXXXXX")
for label in hdd ssd; do
    if [[ "$label" == hdd ]]; then parent=$hdd_image_dir; else parent=$ssd_image_dir; fi
    image="$parent/index.ext4"
    target="$mounts/$label"
    mkdir "$target"
    truncate -s "$image_size" "$image"
    mkfs.ext4 -q -F -E nodiscard,lazy_itable_init=0,lazy_journal_init=0 "$image"
    mount -t ext4 -o loop,nodiscard "$image" "$target"
    chown "$owner" "$target"
    printf '%s ' "$label"
    findmnt -rn -o SOURCE,FSTYPE --target "$image"
    findmnt -rn -o SOURCE,FSTYPE --target "$target"
done
args=("$bin" "$base" "$work" "$mounts/hdd" "$mounts/ssd"
      --sample-dir "$sample" --reserve-volume "$ssd_parent")
runuser -u "$owner" -- python3 "$runner" "${args[@]}"
