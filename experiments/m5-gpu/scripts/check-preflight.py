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
    rows = [json.loads(line) for line in read('results/gate/preflight.jsonl').splitlines() if line]
    if not rows or rows[-1].get('stage') != 'gate':
        if remote_exit != '0':
            raise SystemExit(f'CUDA execution failed before final gate: remote exit {remote_exit}')
        raise SystemExit('CUDA preflight lacks a final gate record')
    if remote_exit == '0' and rows[-1].get('status') == 'PASS':
        print('Remote exit=0 and final CUDA gate=PASS, verified on CPU after release.')
    elif remote_exit == '1' and rows[-1].get('status') == 'FAIL':
        summaries = [row for row in rows if row.get('stage') == 'tiny-overfit-summary']
        states = {row.get('state', 'post-18') for row in summaries}
        if not {'fresh', 'post-18'}.issubset(states):
            raise SystemExit('failed CUDA gate lacks both fresh and post-18 diagnostic summaries')
        raise SystemExit('Valid CUDA overfit failure preserved: final gate=FAIL with fresh and post-18 summaries; not a PASS.')
    else:
        raise SystemExit(f'CUDA execution/gate mismatch: remote exit {remote_exit}, gate={rows[-1].get("status")}')
