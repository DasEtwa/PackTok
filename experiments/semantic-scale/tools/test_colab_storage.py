"""Offline regression tests: no Colab, GPU, network, or model allocation."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import colab_storage as storage


def generation(root, step, identity="run-a"):
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{step:012d}-123-{step + 100}"
    directory = root / name
    directory.mkdir()
    shards = []
    for ordinal in range(2):
        filename = f"tensor-{ordinal:06d}.safetensors"
        path = directory / filename
        path.write_bytes(f"example-shard-step{step}-ordinal{ordinal}".encode())
        shards.append({"name": f"tensor.{ordinal}", "file": filename,
                       "sha256": storage.digest(path), "bytes": path.stat().st_size})
    storage.atomic_json(directory / "index.json", {"format": 1, "step": step,
                                                   "identity": identity, "shards": shards})
    pointer = {"format": 1, "directory": name, "index_sha256": storage.digest(directory / "index.json")}
    storage.atomic_json(root / "latest.json", pointer)
    return directory, pointer


class StorageTests(unittest.TestCase):
    def test_copy_verify_restore_and_keep_only_previous_managed_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, drive, restored = root / "local", root / "drive", root / "restored"
            first, pointer = generation(local / "checkpoints", 1)
            storage.atomic_json(local / "config.json", {"setting": "a"})
            (local / "metrics.jsonl").write_text('{"step":1}\n')
            mirror = storage.Mirror(local, drive, "run-a")
            mirror.sync(final=True)
            self.assertEqual(storage.checkpoint(drive / "checkpoints", verify_payloads=True)[0], pointer)
            second, new_pointer = generation(local / "checkpoints", 2)
            mirror.sync(final=True)
            self.assertFalse((drive / "checkpoints" / first.name).exists())
            self.assertTrue(first.exists())  # Python never prunes local Rust state.
            storage.Mirror(restored, drive, "run-a").restore()
            self.assertEqual(storage.checkpoint(restored / "checkpoints", verify_payloads=True)[0], new_pointer)
            self.assertEqual((restored / "metrics.jsonl").read_text(), '{"step":1}\n')
            self.assertTrue(mirror.verified)

    def test_backup_freezes_checkpoint_before_copying_matching_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, drive = root / "local", root / "drive"
            _, pointer_one = generation(local / "checkpoints", 1)
            (local / "metrics.jsonl").write_text('{"step":1}\n')
            mirror = storage.Mirror(local, drive, "run-a")
            original = storage.copy_metadata
            def writer_advances_after_old_metrics_copy(source, destination):
                original(source, destination)
                (local / "metrics.jsonl").write_text('{"step":1}\n{"step":2}\n')
                generation(local / "checkpoints", 2)
            with patch.object(storage, "copy_metadata", side_effect=writer_advances_after_old_metrics_copy):
                mirror.sync()
            self.assertEqual(storage.read_json(drive / "checkpoints/latest.json"), pointer_one)
            self.assertEqual((drive / "metrics.jsonl").read_text(), '{"step":1}\n')
            mirror.sync(final=True)
            self.assertEqual(storage.checkpoint(drive / "checkpoints", verify_payloads=True)[1]["step"], 2)
            self.assertIn('{"step":2}', (drive / "metrics.jsonl").read_text())

    def test_corrupt_drive_payload_cannot_restore_or_disconnect_as_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, drive = root / "local", root / "drive"
            _, pointer = generation(local / "checkpoints", 1)
            mirror = storage.Mirror(local, drive, "run-a")
            mirror.sync(final=True)
            bad = drive / "checkpoints" / pointer["directory"] / "tensor-000000.safetensors"
            original = bad.read_bytes()
            bad.write_bytes(b"X" + original[1:])
            with self.assertRaises(ValueError):
                storage.Mirror(root / "restore", drive, "run-a").restore()
            with self.assertRaises(ValueError):
                mirror.sync(final=True)
            self.assertFalse((root / "restore/checkpoints/latest.json").exists())

    def test_partial_copy_or_pruned_source_never_replaces_old_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, drive = root / "local", root / "drive"
            _, first_pointer = generation(local, 1)
            storage.transfer_checkpoint(local, drive, force_verify=True)
            generation(local, 2)
            original = storage.copy_file
            calls = []
            def fail_second(source, destination):
                calls.append(source)
                if len(calls) == 2:
                    raise FileNotFoundError("Rust removed old source during slow copy")
                return original(source, destination)
            with patch.object(storage, "copy_file", side_effect=fail_second):
                with self.assertRaises(FileNotFoundError):
                    storage.transfer_checkpoint(local, drive)
            self.assertEqual(storage.read_json(drive / "latest.json"), first_pointer)
            self.assertFalse(any(path.name.endswith(".tmp") for path in drive.iterdir()))
            storage.transfer_checkpoint(local, drive, force_verify=True)
            self.assertEqual(storage.checkpoint(drive, verify_payloads=True)[1]["step"], 2)

    def test_unsafe_paths_symlinks_and_unmanaged_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, drive = root / "local", root / "drive"
            _, pointer = generation(local, 1)
            pointer["directory"] = "../../escape"
            storage.atomic_json(local / "latest.json", pointer)
            with self.assertRaises(ValueError):
                storage.transfer_checkpoint(local, drive)
            drive.mkdir(exist_ok=True)
            (drive / "unrelated.txt").write_text("must remain")
            with self.assertRaises(ValueError):
                storage.Mirror(root / "run", drive, "run-a")
            self.assertEqual((drive / "unrelated.txt").read_text(), "must remain")

    def test_mirror_failure_cancels_owned_process_and_cannot_report_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mirror = storage.Mirror(root / "local", root / "drive", "run-a")
            with patch.object(mirror, "sync", side_effect=OSError("Drive quota exceeded")):
                with self.assertRaises(OSError):
                    mirror.finish()
            self.assertFalse(mirror.verified)

    def test_restore_never_rolls_back_newer_local_checkpoint_or_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, drive = root / "local", root / "drive"
            generation(local / "checkpoints", 1)
            mirror = storage.Mirror(local, drive, "run-a")
            mirror.sync(final=True)
            _, newer = generation(local / "checkpoints", 2)
            mirror.restore()
            self.assertEqual(storage.read_json(local / "checkpoints/latest.json"), newer)
            storage.atomic_json(local / "config.json", {"version": "local"})
            storage.atomic_json(drive / "config.json", {"version": "drive"})
            with self.assertRaises(ValueError):
                mirror.restore()
            self.assertEqual(storage.read_json(local / "config.json"), {"version": "local"})

    def test_restore_without_remote_checkpoint_preserves_newer_local_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, drive = root / "local", root / "drive"
            local.mkdir()
            mirror = storage.Mirror(local, drive, "run-a")
            (local / "metrics.jsonl").write_text('{"step":1}\n')
            mirror.sync()  # Metadata copied, but no first complete checkpoint yet.
            _, pointer = generation(local / "checkpoints", 2)
            metrics = '{"step":1}\n{"step":2}\n'
            (local / "metrics.jsonl").write_text(metrics)
            mirror.restore()
            self.assertEqual(storage.read_json(local / "checkpoints/latest.json"), pointer)
            self.assertEqual((local / "metrics.jsonl").read_text(), metrics)

    def test_restore_validates_local_checkpoint_before_preserving_its_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, drive = root / "local", root / "drive"
            shard_directory, _ = generation(local / "checkpoints", 2)
            mirror = storage.Mirror(local, drive, "run-a")
            (shard_directory / "tensor-000001.safetensors").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                mirror.restore()

    def test_stale_or_divergent_mirror_cannot_replace_remote_weights_or_metadata(self):
        for source_step in (1, 2):
            with self.subTest(source_step=source_step), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                local, drive = root / "local", root / "drive"
                remote_generation, pointer = generation(drive / "checkpoints", 2)
                # Simulate a different runtime: a distinct generation at step 2
                # is divergent even when the plan identity and step are equal.
                if source_step == 2:
                    moved = remote_generation.with_name("step-000000000002-999-777")
                    remote_generation.rename(moved)
                    pointer["directory"] = moved.name
                    storage.atomic_json(drive / "checkpoints/latest.json", pointer)
                storage.atomic_json(drive / storage.MARKER, {
                    "format": 1, "run_name": "run-a", "owner": "packtok-semantic-scale-colab"})
                generation(local / "checkpoints", source_step)
                (local / "metrics.jsonl").write_text("stale-local-metrics\n")
                (drive / "metrics.jsonl").write_text("newer-remote-metrics\n")
                mirror = storage.Mirror(local, drive, "run-a")
                with self.assertRaises(ValueError):
                    mirror.sync(final=True)
                self.assertFalse(mirror.verified)
                self.assertEqual(storage.read_json(drive / "checkpoints/latest.json"), pointer)
                self.assertEqual((drive / "metrics.jsonl").read_text(), "newer-remote-metrics\n")
                self.assertEqual(storage.checkpoint(drive / "checkpoints", verify_payloads=True)[1]["step"], 2)

    def test_metadata_only_mirror_cannot_replace_a_complete_remote_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, drive = root / "local", root / "drive"
            local.mkdir()
            mirror = storage.Mirror(local, drive, "run-a")
            _, pointer = generation(drive / "checkpoints", 2)
            (local / "metrics.jsonl").write_text("stale-local\n")
            (drive / "metrics.jsonl").write_text("newer-remote\n")
            with self.assertRaises(ValueError):
                mirror.sync(final=True)
            self.assertEqual((drive / "metrics.jsonl").read_text(), "newer-remote\n")
            self.assertEqual(storage.read_json(drive / "checkpoints/latest.json"), pointer)

    def test_destination_change_during_copy_never_publishes_or_prunes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, remote = root / "local", root / "remote"
            first, _ = generation(remote, 1)
            generation(local, 2)
            original = storage.copy_file
            concurrent = []
            def publish_newer_while_copying(source, destination):
                original(source, destination)
                if not concurrent:
                    concurrent.append(generation(remote, 3)[1])
            with patch.object(storage, "copy_file", side_effect=publish_newer_while_copying):
                with self.assertRaises(ValueError):
                    storage.transfer_checkpoint(local, remote, remove_previous=True)
            self.assertEqual(storage.read_json(remote / "latest.json"), concurrent[0])
            self.assertEqual(storage.checkpoint(remote, verify_payloads=True)[1]["step"], 3)
            self.assertTrue(first.is_dir(), "a failed publication must not prune its predecessor")

    def test_checkpoint_rejects_malformed_or_inconsistent_step(self):
        for step in (None, True, -1, "2", 1.5, 3):
            with self.subTest(step=step), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint_directory, pointer = generation(root, 2)
                index = storage.read_json(checkpoint_directory / "index.json")
                index["step"] = step
                storage.atomic_json(checkpoint_directory / "index.json", index)
                pointer["index_sha256"] = storage.digest(checkpoint_directory / "index.json")
                storage.atomic_json(root / "latest.json", pointer)
                with self.assertRaises(ValueError):
                    storage.checkpoint(root, verify_payloads=True)

    def notebook_functions(self):
        path = Path(__file__).resolve().parents[3] / "notebooks/PackTok_Semantic_Scale_Colab.ipynb"
        notebook = json.loads(path.read_text())
        functions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                tree = ast.parse("".join(cell["source"]))
                functions.extend(node for node in tree.body if isinstance(node, ast.FunctionDef))
        namespace = {"Path": Path, "re": re, "subprocess": subprocess}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "notebook-functions", "exec"), namespace)
        return namespace

    def test_actual_fresh_no_checkout_clone_and_existing_dirty_checkout(self):
        checkout = self.notebook_functions()["checkout_source"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin, destination = root / "origin", root / "checkout"
            subprocess.run(["git", "init", str(origin)], check=True, capture_output=True)
            (origin / "example.txt").write_text("first")
            subprocess.run(["git", "add", "example.txt"], cwd=origin, check=True)
            subprocess.run(["git", "-c", "user.name=PackTok Test", "-c", "user.email=test@example.invalid",
                            "commit", "-m", "fixture"], cwd=origin, check=True, capture_output=True)
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=origin,
                                    check=True, capture_output=True, text=True).stdout.strip()
            checkout(destination, commit, str(origin))
            self.assertEqual((destination / "example.txt").read_text(), "first")
            checkout(destination, commit, str(origin))
            interrupted = root / "interrupted"
            subprocess.run(["git", "clone", "--no-checkout", str(origin), str(interrupted)],
                           check=True, capture_output=True)
            self.assertFalse((interrupted / ".git/index").exists())
            checkout(interrupted, commit, str(origin))
            self.assertEqual((interrupted / "example.txt").read_text(), "first")
            incomplete_with_notes = root / "incomplete-with-notes"
            subprocess.run(["git", "clone", "--no-checkout", str(origin), str(incomplete_with_notes)],
                           check=True, capture_output=True)
            (incomplete_with_notes / "notes.txt").write_text("user work")
            with self.assertRaises(subprocess.CalledProcessError):
                checkout(incomplete_with_notes, commit, str(origin))
            self.assertEqual((incomplete_with_notes / "notes.txt").read_text(), "user work")
            wrong_origin = root / "wrong-origin"
            subprocess.run(["git", "clone", "--no-checkout", str(origin), str(wrong_origin)],
                           check=True, capture_output=True)
            with self.assertRaises(RuntimeError):
                checkout(wrong_origin, commit, str(root / "different-origin"))
            self.assertFalse((wrong_origin / "example.txt").exists())
            (destination / "example.txt").write_text("user change")
            with self.assertRaises(subprocess.CalledProcessError):
                checkout(destination, commit, str(origin))
            self.assertEqual((destination / "example.txt").read_text(), "user change")

    def test_disconnect_requires_gpu_phase_and_preservation_or_explicit_discard(self):
        disconnect = self.notebook_functions()["should_disconnect"]
        self.assertFalse(disconnect({}, True, False))  # config/mount/setup failure
        self.assertFalse(disconnect({"gpu_work_started": True, "verified": False}, True, True))
        self.assertFalse(disconnect({"gpu_work_started": True, "verified": False}, False, False))
        self.assertTrue(disconnect({"gpu_work_started": True, "verified": False}, False, True))
        self.assertTrue(disconnect({"gpu_work_started": True, "verified": True}, True, False))


if __name__ == "__main__":
    unittest.main()
