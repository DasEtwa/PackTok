"""CPU-only tests for installation plans, provenance, and notebook structure."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name("colab_bootstrap.py")
SPEC = importlib.util.spec_from_file_location("packtok_bootstrap_test", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class BootstrapTests(unittest.TestCase):
    def test_gpu_gate_requires_one_l4_and_compute_89(self):
        self.assertEqual(module.gpu_description("NVIDIA L4, 23034, 8.9\n")["memory_mib"], 23034)
        for output in ("Tesla T4, 15360, 7.5", "NVIDIA A100, 40960, 8.0", "",
                       "NVIDIA L4, 23034, 8.9\nNVIDIA L4, 23034, 8.9",
                       "NVIDIA L4, 8192, 8.9", "NVIDIA L4, 23034, 8.0"):
            with self.subTest(output=output), self.assertRaises((ValueError, RuntimeError)):
                module.gpu_description(output)

    def test_settings_are_bounded_and_demo_1b_is_preflight_only(self):
        module.Settings().validate()
        self.assertTrue(module.Settings(preset="1b").preflight_only)
        self.assertFalse(module.Settings(preset="1b", demo=False).preflight_only)
        for values in ({"train_minutes": 21}, {"train_minutes": 0}, {"run_name": "../escape"},
                       {"preset": "giant"}, {"context": 0}, {"seed": -1}, {"steps": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                module.Settings(**values).validate()

    def test_demo_is_deterministic_distinct_and_never_replaces_modified_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module.create_demo(root)
            first = {name: (root / f"{name}.txt").read_bytes() for name in module.SPLITS}
            self.assertEqual(len(set(first.values())), 3)
            module.create_demo(root)
            self.assertEqual(first, {name: (root / f"{name}.txt").read_bytes() for name in module.SPLITS})
            (root / "train.txt").write_text("changed")
            with self.assertRaises(RuntimeError):
                module.create_demo(root)
            self.assertEqual((root / "train.txt").read_text(), "changed")

    def test_input_provenance_rejects_changes_and_identical_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, prepared = root / "raw", root / "prepared"
            module.create_demo(raw)
            hashes = module.verify_raw_inputs(raw, prepared)
            prepared.mkdir()
            module.write_json(prepared / "raw-input-sha256.json", hashes)
            self.assertEqual(module.verify_raw_inputs(raw, prepared), hashes)
            (raw / "train.txt").write_text("new corpus")
            with self.assertRaises(RuntimeError):
                module.verify_raw_inputs(raw, prepared)
            (raw / "train.txt").write_bytes((raw / "test.txt").read_bytes())
            with self.assertRaises(RuntimeError):
                module.verify_raw_inputs(raw, root / "fresh")

    def test_official_ubuntu_repository_mapping_and_toolkit_completeness(self):
        for version, repository in (("22.04", "ubuntu2204"), ("24.04", "ubuntu2404")):
            self.assertEqual(module.ubuntu_cuda_repository(f'ID=ubuntu\nVERSION_ID="{version}"\n'), repository)
        with self.assertRaises(RuntimeError):
            module.ubuntu_cuda_repository('ID=debian\nVERSION_ID="12"\n')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(module.cuda_toolkit_usable(root))
            (root / "bin").mkdir()
            (root / "bin/nvcc").touch()
            (root / "include").mkdir()
            (root / "include/cuda_runtime.h").touch()
            self.assertFalse(module.cuda_toolkit_usable(root))
            (root / "include/cublas_v2.h").touch()
            self.assertFalse(module.cuda_toolkit_usable(root))
            for header in ("curand.h", "nvrtc.h"):
                (root / "include" / header).touch()
            (root / "lib64").mkdir()
            for library in ("cudart", "cublas", "cublasLt", "nvrtc"):
                (root / "lib64" / f"lib{library}.so").touch()
            self.assertFalse(module.cuda_toolkit_usable(root))
            (root / "lib64/libcurand.so").touch()
            self.assertTrue(module.cuda_toolkit_usable(root))
            (root / "lib64/libcurand.so").unlink()
            (root / "lib64/libcurand.so").symlink_to("missing-curand.so.10")
            self.assertFalse(module.cuda_toolkit_usable(root))

    def test_cuda_install_includes_locked_candle_link_dependencies(self):
        recorded = []
        def run(args, **kwargs):
            recorded.append([str(value) for value in args])
            return "Cuda compilation tools, release 12.8, V12.8.93"
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module, "cuda_toolkit_usable", side_effect=[False, False, True]), \
                 patch.object(module.os, "geteuid", return_value=0), \
                 patch.object(module.urllib.request, "urlretrieve"), \
                 patch.object(module, "ubuntu_cuda_repository", return_value="ubuntu2204"):
                environment = {"PATH": "/usr/bin"}
                module.ensure_cuda(run, environment, Path(directory))
        install = next(args for args in recorded if args[:2] == ["apt-get", "install"])
        for package in ("cuda-nvcc-12-8", "cuda-cudart-dev-12-8", "cuda-nvrtc-dev-12-8",
                        "libcublas-dev-12-8", "libcurand-dev-12-8", "cuda-cccl-12-8"):
            self.assertIn(package, install)
        self.assertNotIn("cuda-drivers", install)

    def test_cuda_rejects_versions_not_supported_by_locked_cudarc(self):
        for version in ("12.7", "13.0"):
            recorded = []
            def run(args, **kwargs):
                recorded.append([str(value) for value in args])
                return f"Cuda compilation tools, release {version}, V{version}.0"
            with patch.object(module, "cuda_toolkit_usable", return_value=True), \
                 patch.object(module.os, "geteuid", return_value=1000), self.assertRaises(RuntimeError):
                module.ensure_cuda(run, {"PATH": "/usr/bin"})
            self.assertEqual(recorded[0][-1], "--version")

    def test_system_install_plan_is_conditional_and_never_installs_a_driver(self):
        recorded = []
        def run(args, **kwargs):
            recorded.append([str(value) for value in args])
            return ""
        with patch.object(module.shutil, "which", return_value="/usr/bin/tool"):
            module.ensure_system_packages(run, {"PATH": "/usr/bin"})
        self.assertEqual(recorded, [])
        with patch.object(module.shutil, "which", return_value=None), patch.object(module.os, "geteuid", return_value=0):
            module.ensure_system_packages(run, {"PATH": "/usr/bin"})
        self.assertEqual(recorded[0], ["apt-get", "update", "-qq"])
        self.assertIn("build-essential", recorded[1])
        self.assertFalse(any("driver" in token for command in recorded for token in command))

    def test_rust_install_is_pinned_and_preserves_proxy_environment(self):
        recorded = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {"PATH": "/usr/bin", "HTTPS_PROXY": "http://example.invalid:8080"}
            def run(args, **kwargs):
                recorded.append([str(value) for value in args])
                return ""
            with patch.object(module.urllib.request, "urlretrieve") as download:
                cargo = module.ensure_rust(run, environment, root)
            self.assertEqual(cargo, root / "packtok-toolchain/cargo/bin/cargo")
            self.assertEqual(environment["HTTPS_PROXY"], "http://example.invalid:8080")
            self.assertTrue(all(any(token in ("1.90.0", "+1.90.0") for token in command) for command in recorded))
            self.assertEqual(download.call_args.args[0], "https://sh.rustup.rs")
            self.assertNotIn("HOME", environment)

    def test_commands_use_argument_vectors_log_success_and_report_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            commands = module.Commands(Path(directory))
            result = commands([sys.executable, "-c", "print('hello')"], timeout=5, label="success")
            self.assertEqual(result.strip(), "hello")
            self.assertIn("hello", (Path(directory) / "01-success.log").read_text())
            with self.assertRaises(RuntimeError):
                commands([sys.executable, "-c", "raise SystemExit(3)"], timeout=5, label="failed")
            self.assertTrue((Path(directory) / "02-failed.log").is_file())

    @unittest.skipUnless(hasattr(os, "fork") and Path("/proc").is_dir(), "Colab regression requires Linux")
    def test_watchdog_kills_descendants_even_after_the_leader_exits(self):
        for leader_exits_first, child_keeps_pipe in ((False, True), (True, True), (False, False)):
            with self.subTest(leader_exits_first=leader_exits_first, child_keeps_pipe=child_keeps_pipe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = root / "group.txt"
                script = root / "leader.py"
                script.write_text("""import os, signal, sys, time
from pathlib import Path
state = Path(sys.argv[1])
if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    state.write_text(str(os.getpgrp()))
    state.with_suffix('.child').write_text(str(os.getpid()))
    print('descendant ready', flush=True)
    if sys.argv[3] == 'False':
        os.close(1)
        os.close(2)
    while True: time.sleep(1)
elif sys.argv[2] == 'True':
    while not state.exists(): time.sleep(.01)
else:
    while True: time.sleep(1)
""")
                results = []
                def run():
                    try:
                        module.Commands(root / "logs")(
                            [sys.executable, script, state, str(leader_exits_first), str(child_keeps_pipe)], timeout=.25)
                    except BaseException as error:
                        results.append(error)
                with patch.object(module, "_TERMINATION_GRACE_SECONDS", .1, create=True):
                    thread = threading.Thread(target=run, daemon=True)
                    thread.start()
                    try:
                        thread.join(timeout=3)
                        self.assertFalse(thread.is_alive(), "watchdog left a descendant holding stdout")
                        self.assertEqual(len(results), 1)
                        self.assertIsInstance(results[0], TimeoutError)
                        child_stat = Path("/proc") / state.with_suffix(".child").read_text() / "stat"
                        for _ in range(100):
                            if not child_stat.exists() or child_stat.read_text().split()[2] == "Z":
                                break
                            threading.Event().wait(.01)
                        else:
                            self.fail("descendant survived timeout")
                    finally:
                        if state.exists():
                            try:
                                os.killpg(int(state.read_text()), signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        thread.join(timeout=2)

    def pipeline_mock(self, content, *, fail_backup=False):
        from contextlib import ExitStack
        import colab_storage
        calls = []
        class FakeCommands:
            def __init__(self, directory):
                directory.mkdir(parents=True, exist_ok=True)
                self.environment = {"PATH": "/usr/bin"}
                self.cancel_event = None
            def __call__(self, arguments, **kwargs):
                args = [str(value) for value in arguments]
                calls.append(args)
                if args[:3] == ["git", "rev-parse", "HEAD"]:
                    return "a" * 40 + "\n"
                command = args[1]
                option = lambda key: Path(args[args.index(key) + 1])
                if command == "prepare":
                    option("--output").mkdir()
                elif command == "plan":
                    module.write_json(option("--out"), {"model": "fake-tiny", "data": str(option("--data"))})
                elif command == "preflight":
                    module.write_json(option("--output") / "preflight.json", {"pass": True, "fake": True})
                elif command == "train":
                    from test_colab_storage import generation
                    generation(option("--output") / "checkpoints", 2)
                elif command == "evaluate":
                    module.write_json(option("--output") / "evaluation.json", {"fake": True})
                return ""
        stack = ExitStack()
        stack.enter_context(patch.object(module, "Commands", FakeCommands))
        stack.enter_context(patch.object(module, "check_gpu", return_value={"name": "fake-unit-test-L4"}))
        stack.enter_context(patch.object(module, "ensure_system_packages"))
        stack.enter_context(patch.object(module, "ensure_cuda"))
        stack.enter_context(patch.object(module, "ensure_rust", return_value=content / "cargo"))
        stack.enter_context(patch.object(module, "build", return_value=content / "fake-binary"))
        if fail_backup:
            stack.enter_context(patch.object(colab_storage.Mirror, "finish", side_effect=OSError("mock quota failure")))
        return stack, calls

    def test_mock_pipeline_local_writer_drive_backup_and_resume_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack, calls = self.pipeline_mock(root)
            preservation = {}
            with stack:
                result = module.execute_pipeline(root / "repository", root / "drive", module.Settings(steps=2),
                                                 preservation=preservation, content=root)
            self.assertTrue(preservation["verified"])
            self.assertTrue(preservation["gpu_work_started"])
            self.assertEqual(result["local_run_directory"], str(root / "packtok-runs/semantic-tiny-42"))
            self.assertNotEqual(result["local_run_directory"], result["run_directory"])
            self.assertTrue((root / "drive/runs/semantic-tiny-42/checkpoints/latest.json").is_file())
            self.assertTrue(any("--resume" in args for args in calls))
            self.assertFalse(any("--cpu" in args for args in calls))

    def test_mock_pipeline_backup_failure_never_sets_preservation_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack, _ = self.pipeline_mock(root, fail_backup=True)
            preservation = {}
            with stack, self.assertRaises(OSError):
                module.execute_pipeline(root / "repository", root / "drive", module.Settings(steps=2),
                                        preservation=preservation, content=root)
            self.assertFalse(preservation["verified"])
            self.assertTrue(preservation["backup_failed"])
            self.assertTrue((root / "packtok-runs/semantic-tiny-42/checkpoints/latest.json").is_file())

    def test_notebook_json_and_all_code_cells_compile_without_colab(self):
        spec = importlib.util.spec_from_file_location("notebook_checker", Path(__file__).with_name("check_notebook.py"))
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        notebook = Path(__file__).resolve().parents[3] / "notebooks/PackTok_Semantic_Scale_Colab.ipynb"
        result = checker.validate(notebook, allow_placeholder=True)
        self.assertEqual(result["compiled_code_cells"], 2)
        document = json.loads(notebook.read_text())
        executable = "".join("".join(cell["source"]) for cell in document["cells"] if cell["cell_type"] == "code")
        self.assertIn("runtime.unassign()", executable)
        self.assertNotIn("--cpu", executable)
        self.assertNotIn("torch", executable)


if __name__ == "__main__":
    unittest.main()
