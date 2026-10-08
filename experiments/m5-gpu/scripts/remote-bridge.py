"""Official Colab CLI transport, not a model/corpus/training implementation."""
import subprocess

subprocess.run(
    ["tar", "-xzf", "/content/packtok-m5-bundle.tar.gz", "-C", "/content"],
    check=True,
)
with open("/content/packtok-m5/bootstrap-console.txt", "wb") as console:
    result = subprocess.run(["bash", "/content/packtok-m5/remote.sh"], stdout=console, stderr=subprocess.STDOUT)
if result.returncode:
    raise RuntimeError(f"Rust CUDA preflight failed with exit {result.returncode}")
