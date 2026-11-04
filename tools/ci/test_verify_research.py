import contextlib
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import verify_research as verify


class ResearchVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for path in set(verify.FROZEN) | verify.EXPECTED_PATHS | {verify.MANIFEST}:
            destination = self.root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(verify.ROOT / path, destination)

    def verify_fails(self, path, replacement, message):
        (self.root / path).write_bytes(replacement)
        with patch.object(verify, "ROOT", self.root), patch.object(sys, "argv", ["verify"]):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, message):
                verify.main()

    def test_changed_output_fails_checksum(self):
        self.verify_fails(verify.OUTPUT, b"changed", "SHA-256 mismatch")

    def test_changed_source_fails_frozen_hash(self):
        self.verify_fails(verify.SOURCE, b"changed", "SHA-256 mismatch")

    def test_missing_manifest_entry_fails(self):
        manifest = (self.root / verify.MANIFEST).read_bytes().splitlines(keepends=True)
        self.verify_fails(verify.MANIFEST, b"".join(manifest[:-1]), "unexpected checksum entries")


if __name__ == "__main__":
    unittest.main()
