"""Recover one public PackTok artifact using a trusted repository manifest.

No listing, credential inspection, deletion or training. A new destination is
mandatory. Failure evidence is retained; completion/size/SHA-256 must all match.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('drive_transport', HERE/'drive-backup.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
MAX_BYTES = 2 * 1024 * 1024 * 1024


def restore(manifest, run_id, destination):
    entries = json.loads(Path(manifest).read_text())['artifacts']
    matches = [e for e in entries if e['run_id'] == run_id]
    if len(matches) != 1:
        raise ValueError('One trusted artifact receipt is required')
    entry = matches[0]
    namespace = entry['remote_namespace']
    if not re.fullmatch(r'[A-Za-z0-9_-]+:PackTok/M5/preflight/m5-[a-z-]+-[0-9a-f]{32}', namespace):
        raise ValueError('Invalid scoped remote namespace')
    if namespace.rsplit('/', 1)[1] != run_id:
        raise ValueError('Run identity mismatch')
    name = entry['artifact']
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name) or name in ('.', '..', 'completion.json'):
        raise ValueError('Unsafe artifact filename')
    if type(entry['bytes']) is not int or not 0 <= entry['bytes'] <= MAX_BYTES:
        raise ValueError('Artifact size exceeds the documented 2 GiB restore bound')
    for field in ('sha256', 'completion_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', entry[field]):
            raise ValueError('Invalid trusted hash')
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError('New regular destination required')
    destination.mkdir(parents=True, exist_ok=False)
    receipt = destination/'completion.json'
    transport.run(['copyto', namespace+'/completion.json', str(receipt), '--immutable', '--max-transfer', '1Mi'])
    if transport.sha(receipt) != entry['completion_sha256']:
        raise ValueError('Completion manifest hash mismatch')
    completion = json.loads(receipt.read_text())
    for field in ('run_id', 'remote_namespace', 'artifact', 'bytes', 'sha256'):
        if completion.get(field) != entry[field]:
            raise ValueError('Completion identity mismatch: '+field)
    if completion.get('status') != 'COMPLETE_VERIFIED':
        raise ValueError('Incomplete artifact cannot be restored')
    artifact = destination/name
    transport.run(['copyto', namespace+'/'+name, str(artifact), '--immutable',
                   '--max-transfer', str(entry['bytes']+16384)])
    if artifact.stat().st_size != entry['bytes'] or transport.sha(artifact) != entry['sha256']:
        raise ValueError('Recovered artifact size/SHA-256 mismatch')
    result = dict(run_id=run_id, artifact=name, bytes=entry['bytes'],
                  sha256=entry['sha256'], status='RESTORED_VERIFIED',
                  exact_training_resume=False)
    (destination/'restore-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    parser.add_argument('run_id')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(restore(args.manifest, args.run_id, args.destination), indent=2))
