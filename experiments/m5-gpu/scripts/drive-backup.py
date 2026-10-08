"""External rclone transport/integrity workflow; no model implementation.

Never sync/delete. Unique remote names, full SHA-256 readback, completion last.
Credentials remain solely in rclone's protected user configuration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    # No config/token introspection. rclone errors are kept local, not uploaded.
    result = subprocess.run(['rclone', *args, '--retries', '1', '--low-level-retries', '1',
                             '--contimeout', '15s', '--timeout', '60s'],
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f'rclone {args[0]} failed (exit {result.returncode}); backup incomplete')
    return result.stdout


def backup(artifact, metadata, local_dir, remote='packtok-drive'):
    artifact = Path(artifact)
    if artifact.is_symlink():
        raise ValueError('regular public research artifact required')
    artifact = artifact.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', remote):
        raise ValueError('remote name')
    required = ['source_commit', 'variant', 'seed', 'configuration_sha256',
                'dataset_sha256', 'tokenizer_sha256', 'kind']
    if any(k not in metadata for k in required):
        raise ValueError('missing identity metadata')
    if set(metadata) != set(required):
        raise ValueError('only explicit research identity fields may be uploaded')
    if not re.fullmatch(r'[0-9a-f]{40}', metadata['source_commit']):
        raise ValueError('source commit')
    for key in ['configuration_sha256', 'dataset_sha256', 'tokenizer_sha256']:
        hashes = metadata[key] if isinstance(metadata[key], dict) else {key: metadata[key]}
        if not hashes or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v)
                             for v in hashes.values()):
            raise ValueError(f'bad identity hash: {key}')
    if metadata['kind'] not in ['preflight-fixture', 'model-weights', 'diagnostic-archive']:
        raise ValueError('unapproved artifact kind')
    # Explicit artifact supplied by operator; no home/config traversal or directories.
    if not artifact.is_file() or artifact.is_symlink() or artifact.name.startswith('.'):
        raise ValueError('regular public research artifact required')
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=False)
    run_id = f"m5-{metadata['kind']}-{uuid.uuid4().hex}"
    destination = f'{remote}:PackTok/M5/preflight/{run_id}'
    # Transport has completed before this entry point. Synchronize local bytes
    # and their directory entry before calculating the durable identity.
    with artifact.open('rb') as f:
        os.fsync(f.fileno())
    directory_fd = os.open(artifact.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    digest = sha(artifact)
    size = artifact.stat().st_size
    record = dict(metadata, run_id=run_id, artifact=artifact.name, bytes=size,
                  sha256=digest, status='INCOMPLETE', exact_training_resume=False,
                  remote_namespace=destination)
    journal = local_dir / 'backup-status.json'
    journal.write_text(json.dumps(record, indent=2) + '\n')
    try:
        remotes = subprocess.run(['rclone', 'listremotes', '--long'], capture_output=True,
                                 text=True, timeout=15, check=True).stdout
        if not any(re.fullmatch(re.escape(remote) + r':\s+drive', row.strip())
                   for row in remotes.splitlines()):
            raise RuntimeError('Drive authentication unavailable: configure the named drive remote interactively')
        run(['mkdir', destination])
        run(['copyto', str(artifact), f'{destination}/{artifact.name}', '--immutable'])
        recovered = local_dir / artifact.name
        run(['copyto', f'{destination}/{artifact.name}', str(recovered), '--immutable'])
        if recovered.stat().st_size != size or sha(recovered) != digest or sha(artifact) != digest:
            raise RuntimeError('Checkpoint integrity mismatch; no completion manifest published')
        record['status'] = 'COMPLETE_VERIFIED'
        record['verification'] = 'full-download SHA-256 and size; local source rehashed'
        manifest = local_dir / 'completion.json'
        with manifest.open('x') as f:
            f.write(json.dumps(record, indent=2) + '\n')
            f.flush()
            os.fsync(f.fileno())
        run(['copyto', str(manifest), f'{destination}/completion.json', '--immutable'])
        readback = local_dir / 'completion-readback.json'
        run(['copyto', f'{destination}/completion.json', str(readback), '--immutable'])
        if sha(readback) != sha(manifest):
            raise RuntimeError('Completion manifest readback mismatch')
        journal.write_text(json.dumps(record, indent=2) + '\n')
        return record
    except Exception as error:
        # Partial remote files remain quarantined by absence of verified completion.
        record['status'] = 'INCOMPLETE'
        record['failure'] = str(error)
        journal.write_text(json.dumps(record, indent=2) + '\n')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('artifact')
    p.add_argument('metadata')
    p.add_argument('local_dir')
    p.add_argument('--remote', default='packtok-drive')
    args = p.parse_args()
    result = backup(args.artifact, json.loads(Path(args.metadata).read_text()), args.local_dir, args.remote)
    print(json.dumps(result, indent=2))
