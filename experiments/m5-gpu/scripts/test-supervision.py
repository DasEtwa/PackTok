"""Reproducible CPU cases using the actual WSL launcher/supervisor.
Five-minute independent-client/PAM-session evidence is documented separately.
No real Colab transport is reachable from the validated fixture PATH.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent


def run_cases(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    cases = [('python-timeout', 'pass', 60, 2, 20, None),
             ('systemd-timeout', 'pass', 60, 20, 3, None),
             ('signal', 'orphan', 60, 20, 30, 'TERM'),
             ('failure', 'exec', 0, 20, 30, None),
             ('release-unconfirmed', 'leak', 0, 20, 30, None),
             ('success', 'orphan', 0, 20, 30, None)]
    for name, mode, seconds, work, runtime, trigger in cases:
        base = output/name
        subprocess.run(['python3', str(HERE/'supervision-fixture.py'), 'create', str(base)], check=True)
        start = time.monotonic()
        with (base/'invocation.txt').open('x') as log:
            process = subprocess.Popen(['bash', str(HERE/'launch-l4-recovery.sh'), '--cpu-mock',
                                        str(base), mode, str(seconds), str(work), str(runtime)],
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                if trigger:
                    for _ in range(100):
                        if (base/'exec-pid.json').exists():
                            break
                        if process.poll() is not None:
                            raise RuntimeError('Supervisor ended before signal test')
                        time.sleep(.05)
                    if not (base/'exec-pid.json').exists():
                        raise TimeoutError('Mock worker did not start')
                    unit = (base/'launcher/unit.txt').read_text().strip()
                    subprocess.run(['systemctl', '--user', 'kill', '--kill-who=main',
                                    '--signal='+trigger, unit], check=True)
                code = process.wait(timeout=35)
            except BaseException:
                # Stop only the unit created in this fixture; never other sessions.
                unit_path = base/'launcher/unit.txt'
                if unit_path.exists():
                    subprocess.run(['systemctl', '--user', 'stop', unit_path.read_text().strip()],
                                   timeout=10, check=False)
                process.wait(timeout=10)
                raise
        evidence = next((base/'experiments/m5-gpu/provenance').glob('l4-recovery-*'))
        shutil.copytree(evidence, base/'supervisor-evidence')
        accounting = json.loads((evidence/'lifecycle-accounting.json').read_text())
        if (code == 0) != (name == 'success'):
            raise AssertionError('Unexpected supervisor result')
        if accounting['cleanup_verified'] != (mode != 'leak'):
            raise AssertionError('Unexpected release verification')
        calls = (base/'calls').read_text()
        if calls.count('new -s') != 1 or 'stop -s LIV' in calls:
            raise AssertionError('Ownership or one-request regression')
        rows.append(dict(case=name, exit=code, seconds=time.monotonic()-start,
                         cleanup_verified=accounting['cleanup_verified'], real_gpu_calls=0))
        (output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    print(json.dumps(run_cases(args.output), indent=2))
