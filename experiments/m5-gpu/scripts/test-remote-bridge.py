#!/usr/bin/env python3
"""CPU-only regression for the Colab transport -> remote child handoff."""
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

BRIDGE = Path(__file__).with_name("remote-bridge.py")
spec = importlib.util.spec_from_file_location("remote_bridge", BRIDGE)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class RemoteBridgeTests(unittest.TestCase):
    def setup_root(self, root, exit_code):
        package = root / "package" / "packtok-m5"
        package.mkdir(parents=True)
        script = package / "remote.sh"
        script.write_text("#!/usr/bin/env bash\nset -eu\n./mock-binary train-extended frozen-config.json binary-sha256:fixture\n")
        mock = package / "mock-binary"
        mock.write_text(
            "#!/usr/bin/env bash\nset -eu\n"
            "test \"$PWD\" = \"$EXPECTED_CWD\"\n"
            "test \"$PACKTOK_M5_GPU_APPROVAL\" = \"approval-reference-fixture\"\n"
            "test \"$#\" = 3 && test \"$1\" = train-extended && test \"$2\" = frozen-config.json && test \"$3\" = binary-sha256:fixture\n"
            "echo child-stdout\necho child-stderr >&2\n"
            f"mkdir -p results\nprintf '{exit_code}\\n' > results/remote-exit-fixture.txt\n"
            f"exit {exit_code}\n"
        )
        mock.chmod(0o755)
        (package / "frozen-config.json").write_text('{"frozen":true}\n')
        (package / "identity.txt").write_text("binary-sha256:fixture\n")
        part = root / "packtok-m5-bundle.tar.gz.part000"
        with tarfile.open(part, "w:gz") as archive:
            archive.add(package, arcname="packtok-m5")
        return package

    def run_bridge(self, root, env):
        result = bridge.run(root, subprocess.run, env)
        with tarfile.open(root / "packtok-m5-results.tar.gz", "r:gz") as archive:
            files = {member.name: archive.extractfile(member).read() for member in archive.getmembers()
                     if member.isfile()}
        return result, files

    def test_missing_approval_is_archived_and_child_does_not_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_root(root, 0)
            rc, files = self.run_bridge(root, {"EXPECTED_CWD": str(root / "packtok-m5")})
            self.assertEqual(rc, 1)
            self.assertIn(b"required PACKTOK_M5_GPU_APPROVAL is missing", files["results/bootstrap-console.txt"])
            self.assertEqual(files["results/exit-code.txt"], b"1\n")
            self.assertNotIn("results/remote-exit-fixture.txt", files)

    def test_approval_args_cwd_console_and_nonzero_status_survive_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.setup_root(root, 7)
            env = {"PACKTOK_M5_GPU_APPROVAL": "approval-reference-fixture",
                   "EXPECTED_CWD": str(root / "packtok-m5")}
            rc, files = self.run_bridge(root, env)
            self.assertEqual(rc, 7)
            self.assertEqual(files["results/exit-code.txt"], b"7\n")
            self.assertIn(b"child-stdout", files["results/bootstrap-console.txt"])
            self.assertIn(b"child-stderr", files["results/bootstrap-console.txt"])
            self.assertEqual(files["results/remote-exit-fixture.txt"], b"7\n")


if __name__ == "__main__":
    unittest.main()
