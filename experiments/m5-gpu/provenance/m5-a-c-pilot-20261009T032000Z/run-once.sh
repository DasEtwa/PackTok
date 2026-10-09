#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/dasetwa/projects/PackTok
bundle=experiments/m5-gpu/bundle-pilot-2k-v1-l4-20261009T032000Z
run=experiments/m5-gpu/provenance/m5-a-c-pilot-20261009T032000Z
approval=M5_A_C_PILOT_20261009_032000Z
marker="$bundle/owned-session"
start=$(date +%s)
cutoff=$((start+1500))
allocated=0
cleaning=0
watchdog=''
remaining() { echo $((cutoff-$(date +%s))); }
logcmd() { local file=$1; shift; local left; left=$(remaining); ((left>0)) || return 124; timeout --signal=TERM --kill-after=10 "$left" "$@" >"$file" 2>&1; }
cleanup() {
 rc=$?
 trap - EXIT INT TERM
 cleaning=1
 if test -n "$watchdog"; then kill "$watchdog" 2>/dev/null || true; fi
 if test "$allocated" = 1; then
   timeout 60 colab stop -s packtok-m5 > "$run/stop.txt" 2>&1 || true
   timeout 30 colab sessions > "$run/sessions-after.txt" 2>&1 || true
   if grep -Fq '[packtok-m5]' "$run/sessions-after.txt" 2>/dev/null; then
     timeout 60 colab stop -s packtok-m5 >> "$run/stop.txt" 2>&1 || true
     timeout 30 colab sessions > "$run/sessions-after.txt" 2>&1 || true
   fi
   timeout 30 colab usage > "$run/usage-after.txt" 2>&1 || true
   if ! grep -Fq '[packtok-m5]' "$run/sessions-after.txt" 2>/dev/null; then
     date -u +%FT%TZ > "$run/released-verified-at.txt"
     printf 'Owned alias packtok-m5 absent after stop.\n' > "$run/release-verification.txt"
   else
     printf 'RELEASE NOT VERIFIED; inspect sessions and release only owned alias.\n' > "$run/release-verification.txt"
     rc=1
   fi
 fi
 printf '%s\n' "$rc" > "$run/lifecycle-exit-code.txt"
 printf '%s\n' "$(( $(date +%s)-start ))" > "$run/allocation-to-cleanup-seconds.txt"
 exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Global 25-minute work cutoff; cleanup trap has the remaining five minutes.
(sleep 1500; kill -TERM $$) & watchdog=$!
test ! -e "$marker"
(cd "$bundle" && sha256sum -c bundle.sha256 && sha256sum -c bundle.parts.sha256)
(cd "$bundle/packtok-m5" && sha256sum -c SHA256SUMS.txt) > "$run/package-content-verification.txt"
colab version > "$run/cli-version.txt" 2>&1
colab sessions > "$run/sessions-before.txt" 2>&1
! grep -Fq '[packtok-m5]' "$run/sessions-before.txt"
colab usage > "$run/usage-before.txt" 2>&1
if grep -Fq 'No active sessions found on server.' "$run/sessions-before.txt"; then :; else grep -Eq '\[.*\].*Hardware:' "$run/sessions-before.txt"; fi
test ! -e "$bundle/allocation-attempted"
date -u +%FT%TZ > "$run/allocation-requested-at.txt"
printf 'authorized_once=%s\nstarted_epoch=%s\n' "$approval" "$start" > "$bundle/allocation-attempted"
: > "$marker"
allocated=1
logcmd "$run/allocation.txt" colab new -s packtok-m5 --gpu L4
logcmd "$run/status.txt" colab status -s packtok-m5
grep -Fq 'Hardware: L4' "$run/status.txt"
logcmd "$run/usage-active.txt" colab usage
shopt -s nullglob
parts=("$bundle"/packtok-m5-bundle.tar.gz.part[0-9][0-9][0-9])
test "${#parts[@]}" -gt 0
: > "$run/upload-list.txt"
for part in "${parts[@]}"; do
 left=$(remaining); ((left>0)) || exit 124
 timeout --signal=TERM --kill-after=10 "$left" colab upload -s packtok-m5 "$part" "content/${part##*/}" >> "$run/upload-list.txt" 2>&1
done
logcmd "$run/execute.txt" colab exec -s packtok-m5 -f "$bundle/packtok-m5/remote-bridge.py" --timeout 1200 || :
if test "$(remaining)" -gt 0; then
 logcmd "$run/usage-before-download.txt" colab usage || true
 logcmd "$run/download.txt" colab download -s packtok-m5 content/packtok-m5-results.tar.gz "$run/results.tar.gz" || true
fi
