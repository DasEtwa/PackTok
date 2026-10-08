"""Local CPU result validation after verified L4 release; no model implementation."""
import json
import sys
import tarfile

with tarfile.open(sys.argv[1], 'r:gz') as archive:
    def read(name):
        entry = archive.getmember(name)
        if not entry.isfile() or entry.size > 16 * 1024 * 1024:
            raise ValueError('unexpected result member')
        return archive.extractfile(entry).read().decode('utf-8')
    remote_exit = read('results/exit-code.txt').strip()
    if remote_exit != '0':
        raise SystemExit(f'CUDA preflight failed: remote exit {remote_exit}; CLI transport exit is not a correctness gate')
    rows = [json.loads(line) for line in read('results/gate/preflight.jsonl').splitlines() if line]
    if not rows or rows[-1].get('stage') != 'gate' or rows[-1].get('status') != 'PASS':
        raise SystemExit('CUDA preflight lacks final PASS evidence')
print('Remote exit=0 and final CUDA gate=PASS, verified on CPU after release.')
