"""Seal CPU checks + real independent Drive readback before one allocation."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('cpu_receipt')
p.add_argument('drive_completion')
args = p.parse_args()
root = Path('experiments/m5-gpu').resolve()
cpu_path = Path(args.cpu_receipt).resolve()
drive_path = Path(args.drive_completion).resolve()
cpu = json.loads(cpu_path.read_text())
stored = json.loads(drive_path.read_text())
if cpu.get('status') != 'CPU_CHECKS_PASS' or stored.get('status') != 'COMPLETE_VERIFIED' or stored.get('kind') != 'preflight-fixture':
    raise SystemExit('CPU and full real WSL Drive verification must both pass')
files = dict(cpu['verified_files'])
for path in [cpu_path, drive_path, drive_path.with_name('completion-readback.json'),
             root/'scripts/l4-recovery.py', root/'scripts/remote-recovery.py',
             root/'scripts/drive-backup.py', root/'configs/primary.json']:
    if not path.is_relative_to(root):
        raise SystemExit('Evidence must stay inside experiment')
    files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
if hashlib.sha256(drive_path.read_bytes()).digest() != hashlib.sha256(drive_path.with_name('completion-readback.json').read_bytes()).digest():
    raise SystemExit('Drive completion full-content readback mismatch')
for relative, digest in files.items():
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise SystemExit('CPU/readback evidence changed; rerun checks')
record=dict(status='CPU_AND_DRIVE_READY', verified_files=files,
            drive_completion=str(drive_path.relative_to(root)),
            frozen_model_unchanged=True, allocation_authorization_count=1,
            allocation_cap_seconds=1800, full_training_authorized=False)
target=root/'provenance/recovery-20261008/PREFLIGHT_READY.json'
with target.open('x') as f:
    f.write(json.dumps(record,indent=2)+'\n')
print('CPU/Drive readiness sealed. No GPU allocated by this command.')
