#!/usr/bin/env bash
set -euo pipefail
umask 077
config=/home/dasetwa/.config/rclone
private=/home/dasetwa/.local/share/packtok-rclone/private
mkdir -p "$config" "$private"
chmod 700 "$config" "$private"
if /home/dasetwa/.local/bin/rclone listremotes | grep -qx 'packtok-drive:'; then
  printf 'Existing remote left intact; authentication check required.\n'
  exit 0
fi
# Output can include tokens on success: keep ALL raw output private, never in Git/chat.
/home/dasetwa/.local/bin/rclone config create packtok-drive drive scope drive.file config_is_local true config_auth_no_browser true --no-output > "$private/oauth-local.log" 2>&1
chmod 600 "$config/rclone.conf"
printf 'Interactive OAuth completed; protected configuration saved.\n'
