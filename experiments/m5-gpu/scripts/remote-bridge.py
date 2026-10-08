"""Official Colab CLI transport, not a model/corpus/training implementation."""
import subprocess
import shutil
from pathlib import Path

# Transport assembly only, with bounded host memory; no corpus preprocessing.
parts = sorted(Path("/content").glob("packtok-m5-bundle.tar.gz.part[0-9][0-9][0-9]"))
if not parts:
    raise RuntimeError("No uploaded preflight parts")
with open("/content/packtok-m5-bundle.tar.gz", "wb") as output:
    for i, part in enumerate(parts):
        if part.name != f"packtok-m5-bundle.tar.gz.part{i:03d}":
            raise RuntimeError("Missing/out-of-order uploaded part")
        with part.open("rb") as source:
            shutil.copyfileobj(source, output, length=16 * 1024 * 1024)

subprocess.run(
    ["tar", "-xzf", "/content/packtok-m5-bundle.tar.gz", "-C", "/content"],
    check=True,
)
with open("/content/packtok-m5/bootstrap-console.txt", "wb") as console:
    result = subprocess.run(["bash", "/content/packtok-m5/remote.sh"], stdout=console, stderr=subprocess.STDOUT)
if result.returncode:
    raise RuntimeError(f"Rust CUDA preflight failed with exit {result.returncode}")
