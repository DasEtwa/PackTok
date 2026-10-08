#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
fixture=$(mktemp -d)
# Only this script's own temporary fixture; no user/session artifacts.
trap 'rm -rf -- "$fixture"' EXIT
mkdir "$fixture/bin" "$fixture/bundle" "$fixture/logs"
printf 'fixture\n' > "$fixture/bundle/CPU_READY"
printf 'bundle\n' > "$fixture/bundle/packtok-m5-bundle.tar.gz"
sha256sum "$(realpath scripts/l4-session.sh)" > "$fixture/bundle/source.sha256"
(cd "$fixture/bundle" && sha256sum CPU_READY packtok-m5-bundle.tar.gz > bundle.sha256)
cat > "$fixture/bin/colab" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$MOCK_ROOT/calls"
case "$1" in
version) printf 'Version: 0.7.4\n';;
sessions)
 if test -f "$MOCK_ROOT/active"; then
   if test "${MOCK_STOP_LEAK:-0}" = 1; then printf '[?] fixture | Hardware: L4 | Variant: GPU\n';
   else printf '[packtok-m5] fixture | Hardware: L4 | Variant: GPU\n'; fi;
 elif test "${MOCK_EXISTING:-0}" = 1; then printf '[packtok-m5] foreign | Hardware: L4 | Variant: GPU\n';
 else printf '[colab] No active sessions found on server.\n'; fi;;
usage) printf 'Current balance: 100.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n';;
new) test "$*" = 'new -s packtok-m5 --gpu L4'; touch "$MOCK_ROOT/active";;
status) printf '[packtok-m5] fixture | Hardware: %s | Variant: GPU\n' "${MOCK_HARDWARE:-L4}";;
upload) if test "${MOCK_FAIL:-}" = upload; then exit 8; fi;;
exec)
 if test "${MOCK_INTERRUPT:-0}" = 1; then kill -TERM "$MOCK_PARENT"; exit 9; fi
 if test "${MOCK_FAIL:-}" = exec; then exit 7; fi;;
download)
 if test "${MOCK_FAIL:-}" = download; then exit 6; fi
 printf 'results fixture\n' > "$5";;
stop) test "$*" = 'stop -s packtok-m5'; if test "${MOCK_STOP_LEAK:-0}" != 1; then rm -f "$MOCK_ROOT/active"; fi;;
*) exit 99;;
esac
MOCK
chmod +x "$fixture/bin/colab"
export PATH="$fixture/bin:$PATH" MOCK_ROOT="$fixture"
export PACKTOK_M5_BUNDLE="$fixture/bundle" PACKTOK_M5_LOGROOT="$fixture/logs"
for failure in none upload exec download hardware; do
 : > "$fixture/calls"
 rm -f "$fixture/bundle/preflight-attempted"
 export MOCK_FAIL="$failure" MOCK_HARDWARE=L4
 if test "$failure" = hardware; then export MOCK_HARDWARE=T4; fi
 set +e
 bash scripts/l4-session.sh preflight > "$fixture/$failure.txt" 2>&1
 rc=$?
 set -e
 if test "$failure" = none; then test "$rc" = 0; else test "$rc" != 0; fi
 test ! -f "$fixture/active"
 test ! -f "$fixture/bundle/owned-session"
 grep -qx 'new -s packtok-m5 --gpu L4' "$fixture/calls"
 grep -qx 'stop -s packtok-m5' "$fixture/calls"
 if test "$failure" = exec; then grep -q '^download ' "$fixture/calls"; fi
 if test "$failure" = download; then test "$(grep -c '^download ' "$fixture/calls")" = 2; fi
 printf 'PASS cleanup case=%s exit=%s\n' "$failure" "$rc"
done
: > "$fixture/calls"
export MOCK_EXISTING=1 MOCK_HARDWARE=L4 MOCK_FAIL=none
if bash scripts/l4-session.sh preflight > "$fixture/unowned.txt" 2>&1; then exit 1; fi
! grep -Eq '^(new|stop) ' "$fixture/calls"
printf 'PASS unowned alias untouched\n'
export MOCK_EXISTING=0
touch "$fixture/bundle/owned-session" "$fixture/active"
: > "$fixture/calls"
if bash scripts/l4-session.sh preflight > "$fixture/stale.txt" 2>&1; then exit 1; fi
! grep -q '^new ' "$fixture/calls"
grep -qx 'stop -s packtok-m5' "$fixture/calls"
printf 'PASS stale owned session released\n'
: > "$fixture/calls"
if bash scripts/l4-session.sh final > "$fixture/unapproved.txt" 2>&1; then exit 1; fi
test ! -s "$fixture/calls"
printf 'PASS unapproved full training rejected before API call\n'
rm -f "$fixture/bundle/preflight-attempted"
# Handled SIGTERM; no detached remote process is used by the fixture.
: > "$fixture/calls"
export MOCK_INTERRUPT=1
bash -c 'export MOCK_PARENT=$$; exec bash scripts/l4-session.sh preflight' > "$fixture/interrupt.txt" 2>&1 &
child=$!
set +e
wait "$child"
rc=$?
set -e
test "$rc" != 0
grep -qx 'stop -s packtok-m5' "$fixture/calls"
test ! -f "$fixture/active"
printf 'PASS handled interruption cleanup exit=%s\n' "$rc"

export MOCK_INTERRUPT=0 MOCK_STOP_LEAK=1
rm -f "$fixture/bundle/preflight-attempted"
if bash scripts/l4-session.sh preflight > "$fixture/endpoint-leak.txt" 2>&1; then exit 1; fi
test -f "$fixture/bundle/owned-session"
printf 'PASS release not falsely confirmed when the owned endpoint survives under an unknown alias\n'
# Simulated resources only; clean this fake after proving that the marker is retained.
export MOCK_STOP_LEAK=0
colab stop -s packtok-m5 >/dev/null
