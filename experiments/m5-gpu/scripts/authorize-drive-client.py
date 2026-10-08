"""Import user-created Desktop OAuth credentials locally; never echo secrets.

Run only after the user supplies the local JSON path. No credentials go in Git.
The user performs Google consent in their browser, using the localhost link.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('client_json')
    args = p.parse_args()
    source = Path(args.client_json).resolve()
    repo = Path('/home/dasetwa/projects/PackTok').resolve()
    windows_repo = Path('/mnt/c/Users/DasEtwa/PackTok').resolve()
    if source.is_relative_to(repo) or source.is_relative_to(windows_repo):
        raise SystemExit('OAuth client JSON must stay outside the repository')
    client = json.loads(source.read_text()).get('installed', {})
    if not client.get('client_id') or not client.get('client_secret'):
        raise SystemExit('A Google Desktop-app OAuth JSON file is required')
    os.umask(0o077)
    private = Path('/home/dasetwa/.local/share/packtok-rclone/private')
    private.mkdir(parents=True, exist_ok=True)
    private.chmod(0o700)
    conf = Path('/home/dasetwa/.config/rclone/rclone.conf')
    # New remote preserves the prior shared-client authorization/configuration.
    remote = 'packtok-drive-own'
    names = subprocess.check_output(['/home/dasetwa/.local/bin/rclone', 'listremotes'], text=True)
    if remote + ':' in names.splitlines():
        raise SystemExit('Dedicated remote already exists; inspect authentication without replacing it')
    with (private/'oauth-own-client.log').open('xb') as log:
        result = subprocess.run(['/home/dasetwa/.local/bin/rclone', 'config', 'create', remote, 'drive',
                                 'client_id', client['client_id'], 'client_secret', client['client_secret'],
                                 'scope', 'drive.file', 'config_is_local', 'true',
                                 'config_auth_no_browser', 'true', '--no-output'], stdout=log, stderr=subprocess.STDOUT)
    if conf.exists():
        conf.chmod(0o600)
    if result.returncode:
        raise SystemExit('Dedicated OAuth setup failed; private diagnostics retained, no GPU allocated')
    print('Dedicated OAuth completed; protected configuration saved. No secrets printed.')
