"""Colab transport for immutable bundles; never implements model/data logic."""
import os
import shutil
import subprocess
from pathlib import Path


def run(root=Path("/content"), run_command=subprocess.run, env=None):
    root = Path(root)
    env = os.environ.copy() if env is None else env.copy()
    results = root / "packtok-m5" / "results"
    results.mkdir(parents=True, exist_ok=True)
    console_path = root / "packtok-m5" / "bootstrap-console.txt"
    rc = 1
    try:
        if not env.get("PACKTOK_M5_GPU_APPROVAL", "").strip():
            raise RuntimeError("required PACKTOK_M5_GPU_APPROVAL is missing; remote command not started")
        parts = sorted(root.glob("packtok-m5-bundle.tar.gz.part[0-9][0-9][0-9]"))
        if not parts:
            raise RuntimeError("No uploaded pilot bundle parts")
        with (root / "packtok-m5-bundle.tar.gz").open("wb") as output:
            for i, part in enumerate(parts):
                if part.name != f"packtok-m5-bundle.tar.gz.part{i:03d}":
                    raise RuntimeError("Missing/out-of-order uploaded part")
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
        extracted = run_command(["tar", "-xzf", str(root / "packtok-m5-bundle.tar.gz"), "-C", str(root)],
                                env=env, check=False)
        if extracted.returncode:
            raise RuntimeError(f"bundle extraction failed with exit {extracted.returncode}")
        with console_path.open("wb") as console:
            result = run_command(["bash", str(root / "packtok-m5" / "remote.sh")], cwd=str(root / "packtok-m5"),
                                 env=env, stdout=console, stderr=subprocess.STDOUT, check=False)
        rc = result.returncode
    except Exception as error:
        with console_path.open("ab") as console:
            console.write((f"bootstrap error: {error}\n").encode())
        rc = 1
    (results / "bootstrap-console.txt").write_bytes(console_path.read_bytes() if console_path.exists() else b"")
    (results / "exit-code.txt").write_text(f"{rc}\n")
    archive = root / "packtok-m5-results.tar.gz"
    packed = run_command(["tar", "-czf", str(archive), "results"], cwd=str(root / "packtok-m5"),
                         env=env, check=False)
    if packed.returncode:
        raise RuntimeError(f"result archive failed with exit {packed.returncode}")
    return rc


if __name__ == "__main__":
    raise SystemExit(run())
