"""Seal CPU checks + real independent Drive readback before one allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('cpu_receipt')
p.add_argument('drive_completion')
p.add_argument('--bundle-name', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
root = Path('experiments/m5-gpu').resolve()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

cpu_path = Path(args.cpu_receipt).resolve()
drive_path = Path(args.drive_completion).resolve()
cpu = json.loads(cpu_path.read_text())
stored = json.loads(drive_path.read_text())
if cpu.get('status') != 'CPU_CHECKS_PASS' or stored.get('status') != 'COMPLETE_VERIFIED' or stored.get('kind') != 'preflight-fixture':
    raise SystemExit('CPU and full real WSL Drive verification must both pass')
bundle = root / args.bundle_name
if not bundle.is_dir() or not args.bundle_name.startswith('bundle-v'):
    raise SystemExit('fresh bundle directory is required')
subprocess.run(['sha256sum', '-c', 'bundle.sha256'], cwd=bundle, check=True)
subprocess.run(['sha256sum', '-c', 'SHA256SUMS.txt'], cwd=bundle/'packtok-m5', check=True)
frozen_path = bundle/'packtok-m5/frozen-source.json'
frozen = json.loads(frozen_path.read_text())
expected_sha = (bundle/'packtok-m5-expected-sha256.txt').read_text().strip()
bundle_sha = sha256(bundle/'packtok-m5-bundle.tar.gz')
binary_sha = sha256(bundle/'packtok-m5/packtok-m5')
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if (cpu.get('source_commit') != head or frozen.get('bundle_id') != args.bundle_name
        or frozen.get('source_commit') != head or frozen.get('binary_sha256') != binary_sha
        or expected_sha != bundle_sha):
    raise SystemExit('bundle source, executable or archive identity mismatch')
files = dict(cpu['verified_files'])
for path in [cpu_path, drive_path, drive_path.with_name('completion-readback.json'),
             root/'scripts/l4-recovery.py', root/'scripts/remote-recovery.py',
             root/'scripts/drive-backup.py', root/'configs/primary.json']:
    if not path.is_relative_to(root):
        raise SystemExit('Evidence must stay inside experiment')
    files[str(path.relative_to(root))] = sha256(path)
if hashlib.sha256(drive_path.read_bytes()).digest() != hashlib.sha256(drive_path.with_name('completion-readback.json').read_bytes()).digest():
    raise SystemExit('Drive completion full-content readback mismatch')
bundle_files = [bundle/'bundle.sha256', bundle/'source.sha256', bundle/'CPU_READY',
                bundle/'packtok-m5-expected-sha256.txt', bundle/'packtok-m5-bundle.tar.gz',
                frozen_path, bundle/'packtok-m5/packtok-m5', bundle/'packtok-m5/SHA256SUMS.txt',
                bundle/'packtok-m5/cpu-reference.json', bundle/'packtok-m5/data/A-train.seq',
                bundle/'packtok-m5/data/C-train.seq'] + sorted(bundle.glob('packtok-m5-bundle.tar.gz.part???'))
for path in bundle_files:
    if not path.is_relative_to(root):
        raise SystemExit('Bundle evidence must stay inside experiment')
    files[str(path.relative_to(root))] = sha256(path)
for relative, digest in files.items():
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or sha256(path)!=digest:
        raise SystemExit('CPU/readback evidence changed; rerun checks')
output = (root/args.output).resolve()
if not output.is_relative_to(root) or output.exists():
    raise SystemExit('fresh readiness output must be new and inside experiment')
record=dict(status='CPU_AND_DRIVE_READY', bundle_name=args.bundle_name, source_commit=head,
            bundle_sha256=bundle_sha, binary_sha256=binary_sha, verified_files=files,
            drive_completion=str(drive_path.relative_to(root)),
            frozen_model_unchanged=True, allocation_authorization_count=1,
            allocation_cap_seconds=1800, full_training_authorized=False)
target=output
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('x') as f:
    f.write(json.dumps(record,indent=2)+'\n')
print('CPU/Drive readiness sealed. No GPU allocated by this command.')
