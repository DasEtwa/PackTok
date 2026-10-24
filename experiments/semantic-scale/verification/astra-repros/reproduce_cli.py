"""Tiny CPU/rejected-GPU CLI probes. Never opens a real GPU or changes source."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'experiments/semantic-scale/tools'))
import colab_storage as storage

with tempfile.TemporaryDirectory(prefix='astra-scale-review-') as tmp:
    root = Path(tmp)
    binary = root / 'scale'
    shutil.copyfile(REPO / 'experiments/semantic-scale/target/debug/packtok-semantic-scale', binary)
    binary.chmod(0o700)
    def run(*args, environment=None, expect_success=True):
        result = subprocess.run([str(binary), *map(str, args)], cwd=REPO, env=environment,
                                capture_output=True, text=True, timeout=60)
        if expect_success and result.returncode:
            raise RuntimeError(result.stderr)
        return result
    raw = root / 'raw'
    raw.mkdir()
    for split in ('train', 'validation', 'test'):
        (raw / f'{split}.txt').write_text(''.join(f'{split} example {i}: idk where the key is.\n' for i in range(16)))
    prepared, local, drive = root / 'prepared', root / 'local', root / 'drive'
    local.mkdir()
    config = local / 'config.json'
    run('prepare', '--input', raw, '--output', prepared)
    run('plan', '--data', prepared, '--out', config, '--context', 2, '--batch', 1,
        '--accum', 1, '--steps', 2, '--max-seconds', 60, '--dtype', 'f32')
    run('preflight', '--config', config, '--output', local, '--cpu')
    run('train', '--config', config, '--output', local, '--cpu')
    mirror = storage.Mirror(local, drive, 'run-a')
    storage.copy_file(config, drive / 'config.json')
    # Snapshot visible on Drive when logs through step 1 were copied but the
    # first completed training checkpoint had not yet been copied/published.
    lines = (local / 'metrics.jsonl').read_text().splitlines(True)
    prior = [line for line in lines if json.loads(line).get('stage') == 'start'
             or (json.loads(line).get('stage') == 'train' and json.loads(line).get('step') == 1)]
    (drive / 'metrics.jsonl').write_text(''.join(prior))
    mirror.restore()
    resumed = run('train', '--config', config, '--output', local, '--cpu', '--resume')
    rows = [json.loads(line) for line in (local / 'metrics.jsonl').read_text().splitlines()]
    print(json.dumps({'case': 'real_checkpoint_with_rolled_back_metrics',
                      'resume_exit_code': resumed.returncode,
                      'checkpoint_step': storage.checkpoint(local / 'checkpoints', verify_payloads=True)[1]['step'],
                      'canonical_train_steps': [r['step'] for r in rows if r.get('stage') == 'train'],
                      'summary_completed_steps': json.loads((local / 'summary.json').read_text())['completed_steps']}), flush=True)
    fake_bin = root / 'fake-bin'
    fake_bin.mkdir()
    fake_smi = fake_bin / 'nvidia-smi'
    fake_smi.write_text("#!/bin/sh\nprintf '%s\\n' 'NVIDIA L40, GPU-fixture, 46068, 46000'\n")
    fake_smi.chmod(0o700)
    environment = dict(os.environ)
    environment['PATH'] = str(fake_bin) + os.pathsep + environment.get('PATH', '')
    rejected = run('preflight', '--config', config, '--output', root / 'gpu-probe',
                   environment=environment, expect_success=False)
    print(json.dumps({'case': 'l40_passes_l4_gate', 'exit_code': rejected.returncode,
                      'stderr': rejected.stderr.strip(),
                      'advanced_past_hardware_gate': 'CUDA support is not compiled' in rejected.stderr}), flush=True)
