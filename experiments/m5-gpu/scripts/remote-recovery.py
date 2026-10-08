"""Official Colab kernel transport and loader diagnostics, using immutable v4.

No package installation, model implementation, corpus processing or compilation.
Every handled failure packages diagnostics before returning the remote status.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time
import traceback

BUNDLE_SHA256 = '05790377fcc9458def42ee083009cdb5fe9c387b3f0cabbd48cb10fc09200040'
BASE = Path('/content')
RESULTS = BASE / 'packtok-m5-recovery-results'


def command(name, args, env=None, timeout=30):
    started = time.monotonic()
    with (RESULTS / f'{name}.txt').open('xb') as output:
        p = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
        except BaseException:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            raise
    with (RESULTS / 'stages.jsonl').open('a') as f:
        f.write(json.dumps(dict(stage=name, seconds=time.monotonic() - started, exit=code)) + '\n')
    return code


def main():
    RESULTS.mkdir(exist_ok=False)
    status = 1
    monitor = None
    try:
        parts = sorted(BASE.glob('packtok-m5-bundle.tar.gz.part[0-9][0-9][0-9]'))
        if len(parts) != 14:
            raise RuntimeError('v4 requires exactly fourteen parts')
        h = hashlib.sha256()
        with (BASE / 'packtok-m5-reassembled.tar.gz').open('xb') as output:
            for i, part in enumerate(parts):
                if part.name != f'packtok-m5-bundle.tar.gz.part{i:03d}':
                    raise RuntimeError('missing/out-of-order v4 part')
                with part.open('rb') as f:
                    for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
                        output.write(block)
                        h.update(block)
        if h.hexdigest() != BUNDLE_SHA256:
            raise RuntimeError('transport archive SHA-256 mismatch')
        (RESULTS / 'transport-sha256.txt').write_text(h.hexdigest() + '\n')
        if command('unpack', ['tar', '-xzf', str(BASE / 'packtok-m5-reassembled.tar.gz'), '-C', str(BASE)], timeout=90):
            raise RuntimeError('unpack failed')
        root = BASE / 'packtok-m5'
        exe = root / 'packtok-m5'
        loader = root / 'runtime/ld-linux-x86-64.so.2'
        inherited = os.environ.get('LD_LIBRARY_PATH', '')
        (RESULTS / 'inherited-library-path.txt').write_text(inherited + '\n')
        env = dict(os.environ, NVIDIA_TF32_OVERRIDE='0')
        paths = [str(root / 'runtime'), '/usr/local/cuda/lib64', '/usr/local/nvidia/lib64',
                 '/usr/lib64-nvidia', '/usr/lib/x86_64-linux-gnu', *inherited.split(':')]
        for pattern in ['/usr/local/lib/python*/dist-packages/nvidia/*/lib',
                        '/usr/local/lib/python*/site-packages/nvidia/*/lib']:
            import glob
            paths.extend(glob.glob(pattern))
        paths = list(dict.fromkeys(p for p in paths if p))
        libpath = ':'.join(paths)
        env['LD_LIBRARY_PATH'] = libpath
        (RESULTS / 'effective-library-path.txt').write_text(libpath + '\n')
        (RESULTS / 'frozen-source.json').write_bytes((root / 'frozen-source.json').read_bytes())
        for name, args in [
            ('kernel', ['uname', '-a']), ('os', ['cat', '/etc/os-release']),
            ('nvidia', ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv']),
            ('file', ['file', str(exe)]), ('elf', ['readelf', '-l', '-d', str(exe)]),
            ('host-ldd', ['ldd', str(exe)]),
        ]:
            try:
                command(name, args, env)
            except FileNotFoundError:
                (RESULTS / f'{name}-unavailable.txt').write_text('diagnostic tool unavailable\n')
        inventory = []
        for directory in paths:
            p = Path(directory)
            if p.is_dir():
                inventory.extend(str(f) for f in p.glob('libcu*.so*'))
        (RESULTS / 'runtime-inventory.txt').write_text('\n'.join(sorted(inventory)) + '\n')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True, timeout=15).strip()
        if gpu != 'NVIDIA L4':
            raise RuntimeError(f'wrong GPU: {gpu}')
        status = command('loader-resolution', [str(loader), '--library-path', libpath, '--list', str(exe)], env)
        if status:
            debug = dict(env, LD_DEBUG='libs')
            command('loader-debug', [str(loader), '--library-path', libpath, '--list', str(exe)], debug)
            raise RuntimeError(f'loader failed: {status}; model not started')
        samples = (RESULTS / 'gpu-samples.csv').open('xb')
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,name,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits', '--loop-ms=200'], stdout=samples, stderr=subprocess.STDOUT)
        status = command('rust-console', [str(loader), '--library-path', libpath, str(exe), 'preflight',
                         str(root / 'data'), str(root / 'cpu-reference.json'), str(RESULTS / 'gate')], env, timeout=900)
        if status:
            raise RuntimeError(f'Rust gate exited {status}')
    except BaseException:
        if status == 0:
            status = 1
        (RESULTS / 'failure.txt').write_text(traceback.format_exc())
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        (RESULTS / 'exit-code.txt').write_text(str(status) + '\n')
        with tarfile.open(BASE / 'packtok-m5-results.tar.gz', 'w:gz') as archive:
            archive.add(RESULTS, arcname='results')
        print(json.dumps({'remote_exit': status, 'diagnostic_archive': '/content/packtok-m5-results.tar.gz'}))
    if status:
        raise RuntimeError(f'M5 remote failure {status}; diagnostic archive preserved')


if __name__ == '__main__':
    main()
