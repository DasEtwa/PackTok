"""External Colab installation/orchestration only; model and data work stays Rust.

Standard-library-only helper. It never allocates a cloud runtime, switches GPU,
executes training on CPU, or downloads a corpus. The notebook's finally block
releases its own Colab runtime after preserving output on mounted Drive.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import threading
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_storage import Mirror

RUST_VERSION = "1.90.0"
CUDA_VERSION = "12.8"
REPOSITORY = "https://github.com/DasEtwa/PackTok.git"
EXPERIMENT = Path("experiments/semantic-scale")
SPLITS = ("train", "validation", "test")
_TERMINATION_GRACE_SECONDS = 15


@dataclass(frozen=True)
class Settings:
    preset: str = "tiny"
    variant: str = "semantic"
    context: int = 128
    micro_batch: int = 1
    accumulation: int = 4
    steps: int = 100
    train_minutes: int = 10
    seed: int = 42
    dtype: str = "bf16"
    demo: bool = True
    resume: bool = True
    run_name: str = "semantic-tiny-42"

    def validate(self) -> None:
        if self.preset not in ("tiny", "100m", "1b"):
            raise ValueError("PRESET muss tiny, 100m oder 1b sein.")
        if self.variant not in ("baseline", "semantic") or self.dtype not in ("f32", "bf16"):
            raise ValueError("Ungültige Variante oder Präzision.")
        if not 1 <= self.context <= 4096 or not 1 <= self.micro_batch <= 32:
            raise ValueError("Kontext muss 1..4096 und Microbatch 1..32 sein.")
        if not 1 <= self.accumulation <= 1024 or not 2 <= self.steps <= 1_000_000:
            raise ValueError("Ungültige Akkumulation oder Schrittzahl.")
        if not 1 <= self.train_minutes <= 20:
            raise ValueError("Dieser Colab-Pilot ist pro Trainingsphase auf 1..20 Minuten begrenzt.")
        if not 0 <= self.seed < 2**64:
            raise ValueError("SEED muss in den u64-Bereich passen.")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", self.run_name) is None:
            raise ValueError("RUN_NAME: 1..80 Zeichen aus A-Z, a-z, 0-9, _ und -.")

    @property
    def preflight_only(self) -> bool:
        # A billion-parameter run on the tiny demo is a hardware test, not training
        # on a meaningful corpus. Do not silently start a long demo-model job.
        return self.demo and self.preset == "1b"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_description(output: str) -> dict[str, object]:
    rows = [row for row in csv.reader(output.splitlines()) if row]
    if len(rows) != 1 or len(rows[0]) != 3:
        raise RuntimeError("Genau eine NVIDIA L4 wird benötigt; GPU-Abfrage war nicht eindeutig.")
    name, memory, capability = (part.strip() for part in rows[0])
    if name not in ("NVIDIA L4", "NVIDIA L4 GPU") or capability != "8.9":
        raise RuntimeError(
            f"Gefunden: {name} (Compute {capability}). Bitte in Colab eine L4-Laufzeit wählen. "
            "Dieses Notebook wechselt weder GPU noch Modellgröße automatisch."
        )
    memory_mib = int(memory)
    if memory_mib < 20_000:
        raise RuntimeError("Die L4 meldet weniger als 20.000 MiB Gesamtspeicher.")
    return {"name": name, "memory_mib": memory_mib, "compute_capability": capability}


class Commands:
    """Run argument vectors, stream output, retain failures, and bound own jobs."""

    def __init__(self, log_directory: Path, environment: Mapping[str, str] | None = None):
        self.log_directory = log_directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.environment = dict(os.environ if environment is None else environment)
        self.counter = 0
        self.cancel_event: threading.Event | None = None

    def __call__(self, arguments: Sequence[str | Path], *, cwd: Path | None = None,
                 timeout: float | None = None, label: str = "command") -> str:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RuntimeError("Drive-Backup fehlgeschlagen; GPU-Arbeit wird gestoppt und lokale Ergebnisse bleiben erhalten.")
        self.counter += 1
        args = [str(argument) for argument in arguments]
        logfile = self.log_directory / f"{self.counter:02d}-{label}.log"
        print("$ " + " ".join(args), flush=True)
        expired = threading.Event()
        termination_lock = threading.Lock()
        forced: threading.Timer | None = None
        with logfile.open("x", encoding="utf-8") as log:
            log.write(json.dumps({"argv": args, "cwd": str(cwd) if cwd else None,
                                  "timeout_seconds": timeout}) + "\n")
            log.flush()
            process = subprocess.Popen(
                args, cwd=cwd, env=self.environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            def terminate() -> None:
                nonlocal forced
                with termination_lock:
                    if expired.is_set():
                        return
                    expired.set()
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        return
                    # Descendants can outlive their leader and retain its output
                    # pipe. Signal the owned group even after poll() reports exit.
                    forced = threading.Timer(_TERMINATION_GRACE_SECONDS, kill)
                    forced.daemon = True
                    forced.start()

            def kill() -> None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            timer = threading.Timer(timeout, terminate) if timeout else None
            if timer:
                timer.daemon = True
                timer.start()
            cancellation_done = threading.Event()
            def watch_backup() -> None:
                while not cancellation_done.wait(1):
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        terminate()
                        return
            cancellation = threading.Thread(target=watch_backup, daemon=True)
            cancellation.start()
            output: list[str] = []
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    output.append(line)
                returncode = process.wait()
            except BaseException:
                terminate()
                try:
                    process.wait(timeout=_TERMINATION_GRACE_SECONDS + 1)
                except subprocess.TimeoutExpired:
                    kill()
                    process.wait()
                # wait() covers only the leader. An interrupted caller must not
                # cancel the group's watchdog while live descendants remain.
                kill()
                raise
            finally:
                cancellation_done.set()
                cancellation.join()
                if timer:
                    timer.cancel()
                    timer.join()
                if expired.is_set():
                    # The pipe may close as soon as the leader exits, before the
                    # grace timer fires. Still clean up descendants that closed
                    # their own stdout and ignored SIGTERM.
                    kill()
                if forced:
                    forced.cancel()
                    forced.join()
                if process.stdout:
                    process.stdout.close()
            if expired.is_set():
                if self.cancel_event is not None and self.cancel_event.is_set():
                    raise RuntimeError(f"{label}: Drive-Sicherung fehlgeschlagen; GPU-Unterprozess gestoppt. Log: {logfile}")
                raise TimeoutError(f"{label}: Zeitlimit erreicht; Log: {logfile}")
            if returncode:
                raise RuntimeError(f"{label}: Exit {returncode}; Log: {logfile}")
            return "".join(output)


def check_gpu(run: Callable[..., str]) -> dict[str, object]:
    return gpu_description(run([
        "nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"
    ], label="gpu"))


def ubuntu_cuda_repository(os_release: str) -> str:
    fields = {}
    for line in os_release.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value.strip('"')
    if fields.get("ID") != "ubuntu" or fields.get("VERSION_ID") not in ("22.04", "24.04"):
        raise RuntimeError("Automatische CUDA-Installation unterstützt Ubuntu 22.04/24.04.")
    return "ubuntu" + fields["VERSION_ID"].replace(".", "")


def cuda_toolkit_usable(root: Path) -> bool:
    includes = (root / "include", root / "targets/x86_64-linux/include")
    libraries = (root / "lib64", root / "targets/x86_64-linux/lib")
    # The locked Candle/cudarc CUDA features link all five toolkit libraries.
    # Require unversioned linker files, resolving symlinks to catch partial installs.
    return ((root / "bin/nvcc").is_file()
            and all(any((directory / header).is_file() for directory in includes)
                    for header in ("cuda_runtime.h", "cublas_v2.h", "curand.h", "nvrtc.h"))
            and all(any((directory / f"lib{library}.so").is_file() for directory in libraries)
                    for library in ("cudart", "cublas", "cublasLt", "curand", "nvrtc")))


def ensure_system_packages(run: Callable[..., str], environment: Mapping[str, str]) -> None:
    tools = ("cc", "g++", "pkg-config", "curl", "git")
    if all(shutil.which(tool, path=environment.get("PATH")) for tool in tools):
        return
    if os.geteuid() != 0:
        raise RuntimeError("Build-Werkzeuge fehlen; automatische Installation benötigt Colab-Root.")
    run(["apt-get", "update", "-qq"], timeout=300, label="build-apt-update")
    run(["apt-get", "install", "-y", "--no-install-recommends", "build-essential",
         "pkg-config", "curl", "git", "ca-certificates"], timeout=600, label="build-tools")


def ensure_cuda(run: Callable[..., str], environment: dict[str, str],
                content: Path = Path("/content")) -> Path:
    root = next((candidate for candidate in (
        Path("/usr/local/cuda-12.8"), Path("/usr/local/cuda")
    ) if cuda_toolkit_usable(candidate)), None)
    if root is not None:
        version = run([root / "bin/nvcc", "--version"], label="nvcc")
        # cudarc 0.16.6 has no bindings for CUDA 12.7 or CUDA 13.x.
        if re.search(r"release 12\.(?:0|1|2|3|4|5|6|8|9),", version) is None:
            root = None
    if root is None:
        if os.geteuid() != 0:
            raise RuntimeError("CUDA fehlt. Automatische Installation benötigt die normale Colab-Root-Umgebung.")
        repository = ubuntu_cuda_repository(Path("/etc/os-release").read_text())
        package = content / "cuda-keyring_1.1-1_all.deb"
        url = f"https://developer.download.nvidia.com/compute/cuda/repos/{repository}/x86_64/{package.name}"
        print("Installiere CUDA-Entwicklungswerkzeuge aus dem offiziellen NVIDIA-Repository.", flush=True)
        urllib.request.urlretrieve(url, package)
        run(["dpkg", "-i", package], timeout=180, label="cuda-keyring")
        run(["apt-get", "update", "-qq"], timeout=300, label="apt-update")
        run(["apt-get", "install", "-y", "--no-install-recommends",
             "cuda-nvcc-12-8", "cuda-cudart-dev-12-8", "cuda-nvrtc-dev-12-8",
             "libcublas-dev-12-8", "libcurand-dev-12-8", "cuda-cccl-12-8", "build-essential", "pkg-config"],
            timeout=900, label="cuda-toolkit")
        root = Path("/usr/local/cuda-12.8")
        if not cuda_toolkit_usable(root):
            raise RuntimeError("CUDA-Installation unvollständig: nvcc oder Runtime/cuBLAS/cuRAND/NVRTC-Header bzw. Linker-Bibliotheken fehlen.")
    environment["CUDA_ROOT"] = str(root)
    environment["CUDA_PATH"] = str(root)
    environment["CUDA_COMPUTE_CAP"] = "89"
    environment["PATH"] = str(root / "bin") + os.pathsep + environment.get("PATH", "")
    libraries = (root / "lib64", root / "targets/x86_64-linux/lib")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(str(path) for path in libraries) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    environment["NVIDIA_TF32_OVERRIDE"] = "0"
    run([root / "bin/nvcc", "--version"], label="cuda-selected")
    return root


def ensure_rust(run: Callable[..., str], environment: dict[str, str],
                content: Path = Path("/content")) -> Path:
    toolchain = content / "packtok-toolchain"
    environment["CARGO_BUILD_JOBS"] = "2"
    environment["RAYON_NUM_THREADS"] = "2"
    environment["CARGO_HOME"] = str(toolchain / "cargo")
    environment["RUSTUP_HOME"] = str(toolchain / "rustup")
    environment["PATH"] = str(toolchain / "cargo/bin") + os.pathsep + environment.get("PATH", "")
    rustup = toolchain / "cargo/bin/rustup"
    if not rustup.is_file():
        installer = content / "packtok-rustup-init.sh"
        urllib.request.urlretrieve("https://sh.rustup.rs", installer)
        run(["sh", installer, "-y", "--no-modify-path", "--profile", "minimal",
             "--default-toolchain", RUST_VERSION, "--component", "rustfmt", "--component", "clippy"],
            timeout=900, label="rust-install")
    run([rustup, "toolchain", "install", RUST_VERSION, "--profile", "minimal",
         "--component", "rustfmt", "--component", "clippy"], timeout=900, label="rust-toolchain")
    cargo = toolchain / "cargo/bin/cargo"
    run([cargo, "+" + RUST_VERSION, "--version"], label="cargo-version")
    run([toolchain / "cargo/bin/rustc", "+" + RUST_VERSION, "--version", "--verbose"], label="rust-version")
    return cargo


def create_demo(directory: Path) -> None:
    """Original deterministic synthetic smoke data; never download a corpus."""
    directory.mkdir(parents=True, exist_ok=True)
    templates = {
        "train": [
            "Trainingsbeispiel {i}: Ich habe keine Ahnung, wo der kleine Schlüssel liegt.\n",
            "Training example {i}: idk where the blue notebook is; tbh we can look together.\n",
            "Lerntext {i}: ong das Wetter ist schön; ngl, heute lesen wir ein Buch.\n",
            "Example {i}: I don't know the answer. language: fr; fr bro; KA bleibt mehrdeutig.\n",
        ],
        "validation": [
            "Validierungsfall {i}: Wir sehen grüne Häuser. Keine Ahnung, wer zuerst kommt.\n",
            "Validation item {i}: dunno which train arrives next; to be honest I will wait.\n",
        ],
        "test": [
            "Prüffall {i}: Die rote Tür öffnet sich langsam. Ich sage keine Ahnung.\n",
            "Test item {i}: not gonna lie, this is a different story; idk its ending.\n",
        ],
    }
    for split, choices in templates.items():
        expected = "".join(choices[i % len(choices)].format(i=i) for i in range(128)).encode("utf-8")
        path = directory / f"{split}.txt"
        if path.exists():
            if path.read_bytes() != expected:
                raise RuntimeError(f"Vorhandene Demo-Datei weicht ab: {path}. Neuer RUN_NAME erforderlich.")
        else:
            path.write_bytes(expected)
    write_json(directory / "PROVENANCE.json", {
        "kind": "agent-authored synthetic smoke corpus", "license": "Apache-2.0",
        "purpose": "installation, finite training, metrics and checkpoint checks only",
        "limitations": "Shared vocabulary/templates; not independent natural-language generalization evidence.",
        "files": {f"{name}.txt": sha256(directory / f"{name}.txt") for name in SPLITS},
    })


def verify_raw_inputs(input_directory: Path, prepared_directory: Path) -> dict[str, str]:
    paths = [input_directory / f"{name}.txt" for name in SPLITS]
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Nichtleere Datei benötigt: {path}")
    hashes = {path.name: sha256(path) for path in paths}
    if len(set(hashes.values())) != len(SPLITS):
        raise RuntimeError("Train/Validation/Test dürfen nicht dieselbe Datei enthalten.")
    record = prepared_directory / "raw-input-sha256.json"
    if prepared_directory.exists():
        if not record.is_file():
            raise RuntimeError("Vorbereitungsverzeichnis ohne Rohdaten-Provenienz. Neuer RUN_NAME erforderlich.")
        if json.loads(record.read_text()) != hashes:
            raise RuntimeError("Rohdaten wurden verändert. Neuer RUN_NAME erforderlich; vorhandene Ergebnisse bleiben erhalten.")
    return hashes


def build(run: Callable[..., str], repository: Path, cargo: Path) -> Path:
    manifest = repository / EXPERIMENT / "Cargo.toml"
    run([cargo, "+" + RUST_VERSION, "build", "--locked", "--release", "--features", "cuda",
         "--manifest-path", manifest], cwd=repository, timeout=1800, label="build-cuda")
    binary = repository / EXPERIMENT / "target/release/packtok-semantic-scale"
    if not binary.is_file():
        raise RuntimeError(f"Rust-Programm wurde nicht gefunden: {binary}")
    return binary


def execute_pipeline(repository: Path, persistent: Path | None, settings: Settings,
                     raw_input: Path | None = None, *, drive_budget_gb: int = 10,
                     preservation: dict | None = None,
                     content: Path = Path("/content")) -> dict[str, object]:
    settings.validate()
    if persistent is not None and (drive_budget_gb < 1 or (settings.preset == "1b" and drive_budget_gb < 30)):
        raise ValueError("1B + Drive benötigt ein explizites DRIVE_BUDGET_GB von mindestens 30 GB Arbeitsraum; der kostenlose 15-GB-Drive genügt nicht. Alternativ MOUNT_DRIVE=False für lokale, flüchtige Ergebnisse wählen.")
    run_directory = content / "packtok-runs" / settings.run_name
    mirror = Mirror(run_directory, persistent / "runs" / settings.run_name, settings.run_name) if persistent is not None else None
    if mirror:
        mirror.restore()
    attempt = run_directory / "notebook-attempts" / f"{time.time_ns()}"
    commands = Commands(attempt)
    if mirror:
        commands.cancel_event = mirror.cancel
    started = time.monotonic()
    gpu_work_started = False
    if preservation is not None:
        preservation["verified"] = False
    try:
        gpu = check_gpu(commands)
        source_commit = commands(["git", "rev-parse", "HEAD"], cwd=repository, label="source").strip()
        source_record = run_directory / "notebook-source.json"
        expected_source = {"source_commit": source_commit}
        if source_record.exists():
            if json.loads(source_record.read_text()) != expected_source:
                raise RuntimeError("Dieser RUN_NAME gehört zu einem anderen Quellstand. Für neuen Code einen neuen Namen wählen.")
        elif (run_directory / "checkpoints/latest.json").exists():
            raise RuntimeError("Checkpoint ohne Notebook-Quellstand: Wiederaufnahme benötigt den ursprünglichen gepinnten Lauf.")
        else:
            write_json(source_record, expected_source)
        write_json(attempt / "request.json", {"settings": settings.__dict__, "source_commit": source_commit, "gpu": gpu})
        if mirror:
            mirror.start()
        ensure_system_packages(commands, commands.environment)
        ensure_cuda(commands, commands.environment, content)
        cargo = ensure_rust(commands, commands.environment, content)
        binary = build(commands, repository, cargo)
        input_directory = content / "packtok-demo-input" if settings.demo else raw_input
        if input_directory is None:
            raise ValueError("RAW_INPUT_DIR fehlt.")
        if settings.demo:
            create_demo(input_directory)
        prepared = content / "packtok-prepared" / settings.run_name
        hashes = verify_raw_inputs(input_directory, prepared)
        if not prepared.exists():
            prepared.parent.mkdir(parents=True, exist_ok=True)
            commands([binary, "prepare", "--input", input_directory, "--output", prepared],
                     cwd=repository, timeout=900, label="prepare")
            write_json(prepared / "raw-input-sha256.json", hashes)
        candidate = attempt / "config.json"
        commands([binary, "plan", "--preset", settings.preset, "--context", str(settings.context),
                  "--batch", str(settings.micro_batch), "--accum", str(settings.accumulation),
                  "--dtype", settings.dtype, "--variant", settings.variant, "--steps", str(settings.steps),
                  "--data", prepared, "--seed", str(settings.seed), "--max-seconds", str(settings.train_minutes * 60),
                  "--out", candidate], cwd=repository, timeout=60, label="plan")
        config = run_directory / "config.json"
        if config.exists() and json.loads(config.read_text()) != json.loads(candidate.read_text()):
            raise RuntimeError("RUN_NAME existiert mit anderer Konfiguration. Neuen Namen wählen.")
        if not config.exists():
            config.write_bytes(candidate.read_bytes())
        if not settings.resume and (run_directory / "checkpoints/latest.json").exists():
            raise RuntimeError("Checkpoint vorhanden, aber RESUME=False. Neuen RUN_NAME wählen.")
        gpu_work_started = True
        if preservation is not None:
            preservation["gpu_work_started"] = True
        commands([binary, "preflight", "--config", config, "--output", run_directory],
                 cwd=repository, timeout=1185, label="preflight")
        if not settings.preflight_only:
            training = [binary, "train", "--config", config, "--output", run_directory]
            if settings.resume:
                training.append("--resume")
            commands(training, cwd=repository, timeout=settings.train_minutes * 60 + 60, label="train")
            commands([binary, "evaluate", "--config", config, "--output", run_directory],
                     cwd=repository, timeout=600, label="evaluate")
        else:
            print("1B + Demo: Hardware-Preflight beendet. Für Training eigene Daten und USE_DEMO_DATA=False wählen.", flush=True)
        summary: dict[str, object] = {
            "status": "preflight-only" if settings.preflight_only else "bounded-phases-finished",
            "source_commit": source_commit, "local_run_directory": str(run_directory),
            "run_directory": str(mirror.drive if mirror else run_directory),
            "gpu": gpu, "raw_input_sha256": hashes,
            "elapsed_including_setup_seconds": time.monotonic() - started,
            "settings": settings.__dict__,
            "gpu_validation": "executed by this notebook runtime, not established by repository CPU tests",
            "persistence": "verified Drive mirror; diagnostic preflight model images excluded" if mirror else "local ephemeral runtime only",
        }
        write_json(attempt / "summary.json", summary)
        write_json(run_directory / "notebook-summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return summary
    finally:
        if mirror:
            try:
                verified = mirror.finish()
                if preservation is not None:
                    preservation["verified"] = verified and gpu_work_started
            except BaseException as error:
                if preservation is not None:
                    preservation["verified"] = False
                    preservation["backup_failed"] = True
                print("Drive-Sicherung fehlgeschlagen. Lokale Ergebnisse unter " + str(run_directory) + " bleiben erhalten; Laufzeit nicht automatisch trennen. " + str(error), flush=True)
                raise
