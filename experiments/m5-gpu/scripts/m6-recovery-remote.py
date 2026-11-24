"""Run the bounded Rust fixture in separate processes. Python owns transport and telemetry."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run_workers(bundle, out, *, cpu_rehearsal=False):
    out.mkdir()
    env = os.environ.copy()
    env.update(NVIDIA_TF32_OVERRIDE="0", PACKTOK_M6_REGIME="T", RAYON_NUM_THREADS="1")
    env.pop("PACKTOK_M6_RESTORE_AUDIT", None)
    env.pop("PACKTOK_M6_PROBE_DEVICE", None)
    runtime = bundle / "runtime"
    libpath = ":".join([str(runtime), "/usr/lib64-nvidia", "/usr/local/nvidia/lib64",
                        "/usr/lib/x86_64-linux-gnu", env.get("LD_LIBRARY_PATH", "")])
    argv = [str(runtime / "ld-linux-x86-64.so.2"), "--library-path", libpath,
            str(bundle / "worker"), "--exact", "training_worker", "--nocapture"]
    monitor = None
    if not cpu_rehearsal:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                       "--format=csv,noheader"], text=True, timeout=10)
        (out / "hardware.txt").write_text(gpu)
        if gpu.split(",")[0].strip() != "NVIDIA L4" or len(gpu.splitlines()) != 1:
            raise RuntimeError("expected exactly one NVIDIA L4")
        env["PACKTOK_M6_PROBE_DEVICE"] = "L4"
        with (out / "vram.csv").open("w") as f:
            monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used",
                                        "--format=csv,noheader,nounits", "--loop-ms=100"], stdout=f)
    try:
        for name in ["full", "split", "audit"]:
            (out / name).mkdir()
        for label, directory, stop in [("control", "full", 4), ("interrupt", "split", 2),
                                       ("resume", "split", 4)]:
            env.update(PACKTOK_M6_FIXTURE_DIR=str(out / directory), PACKTOK_M6_STOP=str(stop))
            if label == "resume":
                env["PACKTOK_M6_RESTORE_AUDIT"] = str(out / "audit")
            log = out / (label + ".log")
            start = time.monotonic()
            peak_kib = 0
            killed = False
            with log.open("wb") as f:
                proc = subprocess.Popen(argv, cwd=bundle, env=env, stdout=f, stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None:
                        if time.monotonic() - start > 65:
                            raise TimeoutError(label)
                        try:
                            status = Path(f"/proc/{proc.pid}/status").read_text()
                            for line in status.splitlines():
                                if line.startswith("VmHWM:"):
                                    peak_kib = max(peak_kib, int(line.split()[1]))
                        except FileNotFoundError:
                            pass
                        if label == "interrupt" and b"KILL_HERE" in log.read_bytes():
                            proc.kill()
                            killed = True
                            break
                        time.sleep(0.02)
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    rc = proc.wait(timeout=5)
            with (out / "processes.jsonl").open("a") as f:
                print(json.dumps(dict(label=label, pid=proc.pid, exit=rc, killed=killed,
                                      seconds=time.monotonic()-start, sampled_vm_hwm_kib=peak_kib)), file=f)
            if rc != (-signal.SIGKILL if killed else 0) or killed != (label == "interrupt"):
                raise RuntimeError(f"{label} did not reach its required exit: {rc}")
    finally:
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=5)


def main():
    root = Path("/content")
    bundle = root / "m6-probe"
    result = root / "m6-result"
    rc = 1
    try:
        archive = root / "m6-probe.tar.gz"
        parts = sorted(root.glob("m6-probe.part[0-9][0-9][0-9]"))
        with archive.open("xb") as target:
            import shutil
            for i, part in enumerate(parts):
                if part.name != f"m6-probe.part{i:03d}":
                    raise ValueError("missing bundle part")
                with part.open("rb") as src:
                    shutil.copyfileobj(src, target, 1024 * 1024)
        if digest(archive) != os.environ["PACKTOK_M6_SHA256"]:
            raise ValueError("uploaded bundle hash mismatch")
        subprocess.run(["tar", "-xzf", str(archive), "-C", str(root)], check=True, timeout=90)
        run_workers(bundle, result)
        rc = 0
    except Exception as error:
        result.mkdir(exist_ok=True)
        (result / "failure.txt").write_text(repr(error) + "\n")
    finally:
        (result / "exit-code.txt").write_text(str(rc) + "\n")
        with tarfile.open(root / "m6-result.tar.gz", "w:gz") as tar:
            tar.add(result, arcname="m6-result")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
