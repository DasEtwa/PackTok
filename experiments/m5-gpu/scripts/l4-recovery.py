"""One explicitly authorized L4 recovery, supervised locally in Ubuntu-24.04.

1620 s worker + 180 s cleanup reserve; no full training or accelerator fallback.
All blocking CLI children have bounded deadlines and isolated process groups.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def inventory(text):
    rows = re.findall(r'^\[([^]]+)\] ([^ ]+) \| Hardware: ([^ |]+)', text, re.M)
    if not rows and 'No active sessions found on server.' not in text:
        raise RuntimeError('Unknown session inventory; cannot infer absence')
    return rows


class Supervisor:
    def __init__(self, root, work_seconds=1620):
        self.root = Path(root).resolve()
        self.bundle = self.root / 'bundle-v4'
        self.run = self.root / 'provenance' / ('l4-recovery-' + uuid.uuid4().hex)
        self.run.mkdir()
        self.marker = self.bundle / 'recovery-owned.json'
        self.start = None
        self.deadline = None
        self.work_seconds = work_seconds
        self.endpoint = None
        self.allocated = False
        self.released = False
        self.child = None
        self.requests = 0

    def command(self, stage, args, limit, cleanup=False):
        if not cleanup and self.deadline is not None:
            limit = min(limit, self.deadline - time.monotonic())
        if limit <= 0:
            raise TimeoutError('Supervised work deadline reached')
        start = time.monotonic()
        started_utc = utc()
        rc = None
        pid = None
        termination = None
        with (self.run / (stage + '.txt')).open('xb') as output:
            self.child = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT,
                                          cwd=self.root.parent.parent, start_new_session=True)
            pid = self.child.pid
            try:
                rc = self.child.wait(timeout=limit)
            except BaseException as error:
                termination = type(error).__name__
                raise
            finally:
                # Own process groups only. Reap descendants even when their CLI
                # leader reports success; a surviving transport is never useful.
                try:
                    os.killpg(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.child.wait(timeout=5)
                if rc is None:
                    rc = self.child.returncode
                self.child = None
                with (self.run / 'stages.jsonl').open('a') as f:
                    f.write(json.dumps(dict(stage=stage, start_utc=started_utc, end_utc=utc(),
                                            pid=pid, pgid=pid, termination=termination,
                                            seconds=time.monotonic()-start, exit=rc)) + '\n')
        if rc:
            raise RuntimeError(f'{stage}: CLI/process exit {rc}')
        return (self.run / (stage + '.txt')).read_text()

    def release(self):
        if not self.allocated:
            return
        # User-facing commentary/analysis happens only after these release queries.
        for attempt in range(2):
            try:
                rows = inventory(self.command(f'sessions-before-stop-{attempt}',
                                               ['colab', 'sessions'], 20, cleanup=True))
                aliases = [endpoint for alias, endpoint, hardware in rows if alias == 'packtok-m5']
                if aliases and (self.endpoint is None or aliases != [self.endpoint]):
                    raise RuntimeError('Release refused: alias ownership is unknown or changed')
                if self.endpoint and any(endpoint == self.endpoint and alias != 'packtok-m5'
                                         for alias, endpoint, hardware in rows):
                    raise RuntimeError('Release refused: owned endpoint has another alias')
                if not aliases:
                    break
                self.command(f'stop-{attempt}', ['colab', 'stop', '-s', 'packtok-m5'], 25, cleanup=True)
                break
            except Exception as error:
                (self.run / f'release-refused-{attempt}.txt').write_text(str(error)+'\n')
        for attempt in range(2):
            try:
                rows = inventory(self.command(f'sessions-after-{attempt}', ['colab', 'sessions'], 20, cleanup=True))
                if not any(alias == 'packtok-m5' or (self.endpoint and endpoint == self.endpoint)
                           for alias, endpoint, hardware in rows):
                    self.released = True
                    break
            except Exception:
                pass
        try:
            self.command('usage-after', ['colab', 'usage'], 20, cleanup=True)
        except Exception:
            pass
        accounting = dict(allocation_count_this_invocation=self.requests, endpoint=self.endpoint,
                          request_epoch=self.start, cleanup_verified=self.released,
                          request_to_cleanup_seconds=time.time()-self.start,
                          cleanup_end_utc=utc(), total_cap_seconds=1800,
                          abrupt_host_failure_cleanup_guaranteed=False)
        (self.run / 'lifecycle-accounting.json').write_text(json.dumps(accounting, indent=2)+'\n')
        if self.released:
            self.marker.rename(self.run / 'ownership-released.json')
        else:
            (self.run / 'URGENT-RELEASE-UNCONFIRMED.txt').write_text(
                'Inspect colab sessions/status and the recorded ownership endpoint before any stop.\n'
                'Unknown/reassigned aliases must not be stopped automatically. Manual owner verification required.\n')

    def execute(self):
        # Pre-existing ownership is audited before any fresh request.
        rows = inventory(self.command('sessions-before', ['colab', 'sessions'], 30))
        if self.marker.exists():
            saved = json.loads(self.marker.read_text())
            self.endpoint = saved.get('endpoint')
            aliases = [e for a, e, h in rows if a == 'packtok-m5']
            if aliases and (self.endpoint is None or aliases != [self.endpoint]):
                raise RuntimeError('Unexpected unowned alias; stale ownership cannot authorize takeover')
            if self.endpoint and any(e == self.endpoint and a != 'packtok-m5' for a, e, h in rows):
                raise RuntimeError('Owned endpoint under unexpected alias: manual release required')
            self.allocated = True
            self.start = time.time()
            raise RuntimeError('Stale owned session: release only; no new allocation')
        if any(a == 'packtok-m5' for a, e, h in rows):
            raise RuntimeError('Unowned packtok-m5 alias: untouched')
        if (self.bundle / 'owned-session').exists():
            raise RuntimeError('Historical ownership marker needs audited recovery; no takeover')
        if (self.bundle / 'preflight-attempted').exists():
            raise RuntimeError('v4 already attempted; no second allocation')

    def request(self):
        ready_path = self.root / 'provenance/recovery-20261008/PREFLIGHT_READY.json'
        ready = json.loads(ready_path.read_text())
        if ready.get('status') != 'CPU_AND_DRIVE_READY':
            raise RuntimeError('CPU/Drive readiness is incomplete; no allocation')
        for relative, expected in ready['verified_files'].items():
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise RuntimeError('readiness evidence outside experiment')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('readiness evidence changed; rerun affected checks')
        receipt = json.loads((self.root / ready['drive_completion']).read_text())
        if receipt.get('status') != 'COMPLETE_VERIFIED' or receipt.get('kind') != 'preflight-fixture':
            raise RuntimeError('real WSL Drive fixture verification is required')
        # sha256sum's relative bundle paths require a bundle cwd, handled beforehand
        # by the main entry point; the scientific source guard remains absolute.
        self.command('source-integrity', ['sha256sum', '-c', str(self.bundle / 'source.sha256')], 90)
        parts = sorted(self.bundle.glob('packtok-m5-bundle.tar.gz.part???'))
        if [p.name for p in parts] != [f'packtok-m5-bundle.tar.gz.part{i:03d}' for i in range(14)]:
            raise RuntimeError('invalid fourteen-part ordering')
        self.command('cli-version', ['colab', 'version'], 30)
        usage = self.command('usage-before', ['colab', 'usage'], 30)
        if not re.search(r'^Current balance: [0-9]+\.[0-9]+ compute units$', usage, re.M):
            raise RuntimeError('CU identity/accounting unavailable')
        self.start = time.time()
        self.deadline = time.monotonic() + self.work_seconds
        with (self.bundle / 'preflight-attempted').open('x') as f:
            f.write(json.dumps(dict(request_epoch=self.start, authorization='one recovery, 30 minute total cap'))+'\n')
        self.marker.write_text(json.dumps(dict(run=self.run.name, endpoint=None, request_epoch=self.start))+'\n')
        self.allocated = True
        self.requests = 1
        self.command('allocation', ['colab', 'new', '-s', 'packtok-m5', '--gpu', 'L4'], 180)
        status = self.command('status', ['colab', 'status', '-s', 'packtok-m5'], 30)
        selected = [r for r in inventory(status) if r[0] == 'packtok-m5']
        if len(selected) != 1:
            raise RuntimeError('ownership endpoint unavailable')
        self.endpoint = selected[0][1]
        self.marker.write_text(json.dumps(dict(run=self.run.name, endpoint=self.endpoint, request_epoch=self.start))+'\n')
        if selected[0][2] != 'L4':
            raise RuntimeError('wrong hardware; no fallback')
        self.command('usage-active', ['colab', 'usage'], 20)
        upload_deadline = time.monotonic() + 480
        for i, part in enumerate(parts):
            self.command(f'upload-{i:03d}', ['colab', 'upload', '-s', 'packtok-m5', str(part), 'content/'+part.name],
                         min(120, upload_deadline-time.monotonic()))
        execution_failed = None
        try:
            self.command('execute', ['colab', 'exec', '-s', 'packtok-m5', '-f',
                                    str(self.root / 'scripts/remote-recovery.py'), '--timeout', '1050'], 1070)
        except Exception as error:
            execution_failed = error
        for attempt in range(2):
            try:
                self.command(f'download-{attempt}', ['colab', 'download', '-s', 'packtok-m5',
                             'content/packtok-m5-results.tar.gz', str(self.run/'results.tar.gz')], 60)
                break
            except Exception:
                if attempt == 1:
                    raise
        if execution_failed:
            raise execution_failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true', required=True)
    p.add_argument('--root', default='experiments/m5-gpu')
    p.add_argument('--work-seconds', type=int, default=1620)
    args = p.parse_args()
    if not 1 <= args.work_seconds <= 1620:
        p.error('work deadline must be 1..1620 seconds; cleanup reserve cannot be extended')
    s = Supervisor(args.root, work_seconds=args.work_seconds)
    def interrupted(signum, frame):
        raise InterruptedError(f'handled signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    rc = 0
    try:
        # Inventory/stale-session recovery precedes ordinary CPU verification.
        s.execute()
        # Local hash verification uses the original unchanged v4 manifests.
        with (s.run/'bundle-integrity.txt').open('xb') as out:
            subprocess.run(['sha256sum', '-c', 'bundle.sha256'], cwd=s.bundle, stdout=out,
                           stderr=subprocess.STDOUT, timeout=90, check=True)
        # Inventory/stale session logic, followed by source checks and one request.
        s.request()
    except BaseException as error:
        (s.run/'failure.txt').write_text(str(error)+'\n')
        rc = 1
    finally:
        # Ignore repeated interruption during bounded cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        s.release()
    if s.allocated and not s.released:
        rc = 1
    if s.released and not (s.run/'results.tar.gz').is_file():
        (s.run/'result-missing.txt').write_text('Remote archive missing; transport exit zero is insufficient.\n')
        rc = 1
    if s.released and (s.run/'results.tar.gz').is_file():
        try:
            s.command('result-verification', ['python3', str(s.root/'scripts/check-preflight.py'),
                                            str(s.run/'results.tar.gz')], 30, cleanup=True)
        except Exception:
            rc = 1
    (s.run/'supervisor-process.json').write_text(json.dumps(dict(pid=os.getpid(), pgid=os.getpgrp(), end_utc=utc(), work_seconds=s.work_seconds))+'\n')
    (s.run/'exit-code.txt').write_text(str(rc)+'\n')
    print(f'Lifecycle logs: {s.run}; release_verified={s.released}; exit={rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
