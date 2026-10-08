"""Official Colab CLI transport, not a model/corpus/training implementation."""
import subprocess

subprocess.run(
    ["tar", "-xzf", "/content/packtok-m5-bundle.tar.gz", "-C", "/content"],
    check=True,
)
result = subprocess.run(["bash", "/content/packtok-m5/remote.sh"])
if result.returncode:
    raise RuntimeError(f"Rust CUDA preflight failed with exit {result.returncode}")
