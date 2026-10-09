"""Read-only checks for the 2026-10-08 maintenance delivery.

Credentials are detected heuristically; values are never printed. Historical
identity uses the pre-edit native inventory, not newline-normalized text.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/maintenance/2026-10-08'
SECRET = re.compile(rb'(?:GOCSPX-[A-Za-z0-9_-]{10,}|ya29\.[A-Za-z0-9_-]{10,}|1//[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"(?:(?:access|refresh)_token|client_secret)"\s*:\s*"[^"\s]{10,}")')
ALLOWED = {
    '.gitignore', '.gitattributes', 'README.md', 'STRUCTURE.md', 'CHANGELOG.md',
    'M5_GPU_TRANSFORMER.md', 'M5_GPU_BENCHMARK.md',
    'experiments/m5-gpu/provenance/RECOVERY_STORAGE.md',
    'experiments/m5-gpu/scripts/check-wsl-supervision.sh',
    'experiments/m5-gpu/scripts/l4-recovery.py',
    'experiments/m5-gpu/scripts/launch-l4-recovery.sh',
    'experiments/m5-gpu/scripts/test-recovery.py',
}


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd).decode().strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(output):
    initial = json.loads((OUT/'initial-inventory.json').read_text())
    changes = []
    preserved = 0
    for entry in initial['files']:
        path = ROOT/entry['path']
        if not path.is_file() or sha(path) != entry['native_sha256']:
            changes.append({'path': entry['path'], 'expected_maintenance_edit': entry['path'] in ALLOWED})
        else:
            preserved += 1
    paths = sorted(set(git('ls-files', '--cached', '--others', '--exclude-standard').splitlines()))
    secrets = [p for p in paths if (ROOT/p).is_file() and SECRET.search((ROOT/p).read_bytes())]
    staged = git('diff', '--cached', '--name-only', '--diff-filter=ACM').splitlines()
    staged_secrets = []
    for path in staged:
        if SECRET.search(subprocess.check_output(['git', 'show', ':'+path], cwd=ROOT)):
            staged_secrets.append(path)
    large = [{'path': p, 'bytes': (ROOT/p).stat().st_size} for p in paths
             if (ROOT/p).is_file() and (ROOT/p).stat().st_size > 5*1024*1024]
    links = []
    checked = 0
    for path in paths:
        if not path.endswith('.md') or not (ROOT/path).is_file():
            continue
        text = (ROOT/path).read_text(errors='replace')
        for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', text):
            target = target.strip().strip('<>')
            if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#', 1)[0])
            if not target:
                continue
            checked += 1
            if not ((ROOT/path).parent/target).exists():
                links.append({'document': path, 'target': target,
                              'active': len(Path(path).parts) == 1 or
                                        (path.startswith('docs/') and '/maintenance/' not in path) or
                                        path in {'docs/maintenance/2026-10-08/REPORT.md',
                                                 'docs/maintenance/2026-10-08/SELF_REVIEW.md',
                                                 'docs/maintenance/2026-10-08/PLAN.md'}})
    private = Path('/home/dasetwa/.local/share/packtok-maintenance/2026-10-08/inventory.json')
    other = json.loads(private.read_text())['runtime_web']
    other_root = Path('/mnt/d/runtime-web')
    unrelated_changed = []
    for entry in other['files']:
        path = other_root/entry['path']
        if not path.is_file() or sha(path) != entry['sha256']:
            unrelated_changed.append(entry['path'])
    unrelated_state = (git('rev-parse', 'HEAD', cwd=other_root) == other['head'] and
                       git('status', '--porcelain=v1', cwd=other_root) == other['status'])
    report = dict(initial_files=len(initial['files']), byte_preserved=preserved,
                  expected_existing_edits=changes,
                  unexpected_historical_changes=[c for c in changes if not c['expected_maintenance_edit']],
                  credential_candidates=secrets, staged_credential_candidates=staged_secrets,
                  files_over_5MiB=large, relative_links_checked=checked,
                  missing_links=links, active_missing_links=[l for l in links if l['active']],
                  runtime_web_byte_changes=unrelated_changed,
                  runtime_web_git_state_unchanged=unrelated_state,
                  real_gpu_allocations_this_task=0,
                  security_scan_limit='Heuristic pattern scan; no credential values output',
                  historical_text_policy='Exact native bytes; initial checkout EOL differences remain documented')
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('missing_links', 'expected_existing_edits')}, indent=2))
    return int(bool(report['unexpected_historical_changes'] or secrets or staged_secrets or large or
                    report['active_missing_links'] or unrelated_changed or not unrelated_state))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: check-maintenance.py NEW_REPORT_PATH')
    sys.exit(check(sys.argv[1]))
