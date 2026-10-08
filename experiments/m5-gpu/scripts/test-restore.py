"""CPU-only regression tests for manifest-driven Drive restore."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('restore', HERE/'restore-drive-artifact.py')
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


class Restore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.payload = b'harmless fixture\n'
        self.entry = dict(run_id='m5-preflight-fixture-'+'a'*32, artifact='fixture.txt',
                          bytes=len(self.payload), sha256=hashlib.sha256(self.payload).hexdigest(),
                          remote_namespace='packtok-drive-own:PackTok/M5/preflight/m5-preflight-fixture-'+'a'*32)
        self.completion = dict(self.entry, status='COMPLETE_VERIFIED')
        self.mode = 'pass'
        self.calls = []
        def transport(args):
            self.calls.append(args)
            content = self.receipt if args[1].endswith('/completion.json') else self.payload
            if self.mode == 'receipt' and args[1].endswith('/completion.json'):
                content += b'corruption'
            if self.mode == 'payload' and not args[1].endswith('/completion.json'):
                content = b'corruption'
            Path(args[2]).write_bytes(content)
        self.patch = patch.object(restore.transport, 'run', transport)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def invoke(self):
        self.receipt = json.dumps(self.completion).encode()
        self.entry['completion_sha256'] = hashlib.sha256(self.receipt).hexdigest()
        manifest = self.base/'manifest.json'
        manifest.write_text(json.dumps({'artifacts': [self.entry]}))
        return restore.restore(manifest, self.entry['run_id'], self.base/'restored')

    def test_valid_restore(self):
        self.assertEqual(self.invoke()['status'], 'RESTORED_VERIFIED')
        self.assertEqual((self.base/'restored/fixture.txt').read_bytes(), self.payload)
        self.assertTrue(all(c[0] == 'copyto' and '--immutable' in c for c in self.calls))

    def test_corrupt_receipt(self):
        self.mode = 'receipt'
        with self.assertRaisesRegex(ValueError, 'manifest hash'):
            self.invoke()
        self.assertEqual(len(self.calls), 1)

    def test_corrupt_payload(self):
        self.mode = 'payload'
        with self.assertRaisesRegex(ValueError, 'size/SHA-256'):
            self.invoke()

    def test_incomplete_receipt(self):
        self.completion['status'] = 'INCOMPLETE'
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.invoke()
        self.assertEqual(len(self.calls), 1)

    def test_wrong_receipt_identity(self):
        self.completion['sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.invoke()

    def test_path_traversal(self):
        self.entry['artifact'] = '../private'
        with self.assertRaisesRegex(ValueError, 'filename'):
            self.invoke()
        self.assertFalse(self.calls)

    def test_unrelated_namespace(self):
        self.entry['remote_namespace'] = 'packtok-drive-own:Personal/private'
        with self.assertRaisesRegex(ValueError, 'namespace'):
            self.invoke()
        self.assertFalse(self.calls)

    def test_existing_destination_preserved(self):
        (self.base/'restored').mkdir()
        (self.base/'restored/precious').write_bytes(b'keep')
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual((self.base/'restored/precious').read_bytes(), b'keep')

    def test_oversize_rejected_before_transport(self):
        self.entry['bytes'] = restore.MAX_BYTES+1
        with self.assertRaisesRegex(ValueError, 'size'):
            self.invoke()
        self.assertFalse(self.calls)


if __name__ == '__main__':
    unittest.main(verbosity=2)
