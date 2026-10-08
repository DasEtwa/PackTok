"""CPU-only transport fixture for the actual supervisor and launcher.

This prepares tiny mock archives/parts under an explicit empty directory.
No real Colab CLI, credentials, network calls or GPU operations are used.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('recovery_tests', HERE/'test-recovery.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)


def validate(base):
    base = Path(base).resolve()
    marker = json.loads((base/'MOCK_ONLY.json').read_text())
    actual = hashlib.sha256((base/'bin/colab').read_bytes()).hexdigest()
    expected = hashlib.sha256(tests.FAKE_COLAB.encode()).hexdigest()
    if marker != {'transport_sha256': expected, 'gpu_calls': 0} or actual != expected:
        raise ValueError('Untrusted CPU fixture; launcher refuses all transport')
    return base


def create(base):
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=False)
    class PersistentDirectory:
        name = str(base)
        def cleanup(self):
            pass
    with patch.object(tests.tempfile, 'TemporaryDirectory', PersistentDirectory):
        fixture = tests.Lifecycle()
        fixture.setUp()
        fixture.env.stop()
    (base/'MOCK_ONLY.json').write_text(json.dumps(dict(
        transport_sha256=hashlib.sha256(tests.FAKE_COLAB.encode()).hexdigest(), gpu_calls=0))+'\n')
    validate(base)
    print(base)


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('create', 'validate'):
        raise SystemExit('Usage: supervision-fixture.py create|validate PATH')
    (create if sys.argv[1] == 'create' else validate)(sys.argv[2])
