"""Colab transport for the M7 training and restore smoke test."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

root = Path("/content")
bundle = root / "m6-probe"
out = root / "m7-diagnostics"
out.mkdir()
rc = 1
try:
    path = root / "m7-input.tar.gz"
    parts = sorted(root.glob("m6-probe.part[0-9][0-9][0-9]"))
    if not parts:
        raise ValueError("missing upload")
    with path.open("xb") as dst:
        for index, part in enumerate(parts):
            if part.name != f"m6-probe.part{index:03d}":
                raise ValueError("nonconsecutive upload")
            with part.open("rb") as src:
                shutil.copyfileobj(src, dst, 1024 * 1024)
    with path.open("rb") as archive:
        if hashlib.file_digest(archive, "sha256").hexdigest() != os.environ["PACKTOK_M6_SHA256"]:
            raise ValueError("upload SHA-256 mismatch")
    subprocess.run(["tar", "-xzf", str(path), "-C", str(root)], check=True, timeout=45)
    # The reference is a run marker carried into the approved remote process.
    env = dict(os.environ, PACKTOK_M7_GPU_APPROVAL="m7-20261124-authored-fixture")
    with (out / "console.txt").open("wb") as log:
        rc = subprocess.run(["bash", "gpu-probe.sh"], cwd=bundle, env=env,
                            stdout=log, stderr=subprocess.STDOUT, timeout=200).returncode
except Exception as error:
    (out / "failure.txt").write_text(repr(error) + "\n")
finally:
    (out / "exit-code.txt").write_text(str(rc) + "\n")
    with tarfile.open(root / "m6-result.tar.gz", "w:gz") as archive:
        archive.add(out, arcname="m7-diagnostics")
        for name in ["results", "hardware.csv"]:
            if (bundle / name).exists():
                archive.add(bundle / name, arcname=name)
raise SystemExit(rc)
