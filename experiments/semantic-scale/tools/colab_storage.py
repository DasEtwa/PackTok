"""Verified copying of this experiment's local checkpoints to optional Drive.

Rust writes and fsyncs local POSIX checkpoints. This external transport layer
copies one frozen complete generation, verifies readback, and publishes latest
last. Drive/FUSE is not assumed to provide POSIX durability guarantees. Restore
verifies every byte again. Preflight model images are disposable diagnostics;
only their result JSON/logs are mirrored. No prepared corpus or build cache is
copied. Rust owns local retention; Drive retains the latest verified generation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

MAX_INDEX_BYTES = 4 * 1024 * 1024
MAX_SHARD_BYTES = 512 * 1024 * 1024
GENERATION = re.compile(r"step-[0-9]{12}-[0-9]{1,40}-[0-9]{1,40}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
MARKER = ".packtok-colab-backup.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def regular(path: Path, maximum: int | None = None) -> None:
    if not path.exists() and not path.is_symlink():
        raise FileNotFoundError(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular non-symlink file: {path}")
    if maximum is not None and path.stat().st_size > maximum:
        raise ValueError(f"File exceeds bound: {path}")


def read_json(path: Path, maximum: int = MAX_INDEX_BYTES) -> dict:
    regular(path, maximum)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint(root: Path, *, verify_payloads: bool, frozen_pointer: dict | None = None) -> tuple[dict, dict]:
    pointer = read_json(root / "latest.json") if frozen_pointer is None else frozen_pointer
    if set(pointer) != {"format", "directory", "index_sha256"} or pointer["format"] != 1:
        raise ValueError("Invalid checkpoint latest pointer")
    name = pointer["directory"]
    if not isinstance(name, str) or GENERATION.fullmatch(name) is None:
        raise ValueError("Unsafe checkpoint generation path")
    if not isinstance(pointer["index_sha256"], str) or HASH.fullmatch(pointer["index_sha256"]) is None:
        raise ValueError("Invalid checkpoint index digest")
    directory = root / name
    if directory.is_symlink() or not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint generation missing: {directory}")
    index_path = directory / "index.json"
    regular(index_path, MAX_INDEX_BYTES)
    if digest(index_path) != pointer["index_sha256"]:
        raise ValueError("Checkpoint index hash mismatch")
    index = read_json(index_path)
    if index.get("format") != 1 or not isinstance(index.get("identity"), str) or not index["identity"]:
        raise ValueError("Invalid checkpoint generation identity")
    step = index.get("step")
    if type(step) is not int or step < 0 or step != int(name.split("-")[1]):
        raise ValueError("Invalid checkpoint step or generation/step mismatch")
    shards = index.get("shards")
    if not isinstance(shards, list) or not 1 <= len(shards) <= 4096:
        raise ValueError("Invalid checkpoint shard count")
    expected = {"index.json"}
    for ordinal, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != {"name", "file", "sha256", "bytes"}:
            raise ValueError("Invalid checkpoint shard fields")
        if shard["file"] != f"tensor-{ordinal:06}.safetensors":
            raise ValueError("Unsafe checkpoint shard filename")
        if not isinstance(shard["sha256"], str) or HASH.fullmatch(shard["sha256"]) is None:
            raise ValueError("Invalid checkpoint shard digest")
        if not isinstance(shard["bytes"], int) or not 1 <= shard["bytes"] <= MAX_SHARD_BYTES:
            raise ValueError("Invalid checkpoint shard length")
        path = directory / shard["file"]
        regular(path, MAX_SHARD_BYTES)
        if path.stat().st_size != shard["bytes"]:
            raise ValueError("Checkpoint shard length mismatch")
        if verify_payloads and digest(path) != shard["sha256"]:
            raise ValueError("Checkpoint shard digest mismatch")
        expected.add(shard["file"])
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError("Checkpoint generation contains undeclared files")
    return pointer, index


def copy_file(source: Path, destination: Path) -> None:
    regular(source)
    if destination.is_symlink():
        raise ValueError("Refusing symlink destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
        # FUSE directory fsync may be unsupported. Verify actual copied bytes;
        # restore performs a second full cryptographic verification later.
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def copy_metadata(source: Path, destination: Path) -> None:
    """Only small logs/configuration, never model images, temp evidence or data."""
    if not source.exists():
        return
    for directory, subdirectories, files in os.walk(source, followlinks=False):
        subdirectories[:] = [name for name in subdirectories
                              if name not in ("checkpoints", "preflight-checkpoints")
                              and not name.endswith(".tmp")
                              and not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if name == MARKER or path.suffix not in (".json", ".jsonl", ".log", ".txt") or path.is_symlink():
                continue
            regular(path, 64 * 1024 * 1024)
            target = destination / path.relative_to(source)
            copy_file(path, target)


def destination_pointer(destination: Path, pointer: dict | None, index: dict | None,
                        *, verify_payloads: bool = False) -> dict | None:
    """Reject known stale/divergent writers; this is not a distributed lock."""
    if not (destination / "latest.json").exists():
        return None
    previous, old_index = checkpoint(destination, verify_payloads=verify_payloads)
    if pointer is None or index is None:
        raise ValueError("Remote checkpoint exists but local run has no complete checkpoint")
    if old_index["identity"] != index["identity"]:
        raise ValueError("Destination checkpoint belongs to another run identity")
    if old_index["step"] > index["step"]:
        raise ValueError("Destination checkpoint is newer; refusing stale backup")
    if old_index["step"] == index["step"] and previous != pointer:
        raise ValueError("Destination checkpoint diverges at the same step")
    return previous


def transfer_checkpoint(source: Path, destination: Path, *, force_verify: bool = False,
                        remove_previous: bool = False, frozen_pointer: dict | None = None) -> bool:
    if frozen_pointer is None and not (source / "latest.json").exists():
        return False
    pointer, index = checkpoint(source, verify_payloads=False, frozen_pointer=frozen_pointer)
    destination.mkdir(parents=True, exist_ok=True)
    previous = destination_pointer(destination, pointer, index, verify_payloads=force_verify)
    if previous == pointer:
        return True
    completed = destination / pointer["directory"]
    if not completed.exists():
        temporary = destination / (pointer["directory"] + ".copy-" + uuid.uuid4().hex + ".tmp")
        temporary.mkdir()
        try:
            for shard in index["shards"]:
                copy_file(source / pointer["directory"] / shard["file"], temporary / shard["file"])
                if digest(temporary / shard["file"]) != shard["sha256"]:
                    raise ValueError("Checkpoint copy readback failed")
            copy_file(source / pointer["directory"] / "index.json", temporary / "index.json")
            if digest(temporary / "index.json") != pointer["index_sha256"]:
                raise ValueError("Checkpoint copy index readback failed")
            temporary.replace(completed)
        finally:
            if temporary.exists():
                # Delete only our uniquely named unfinished transport copy.
                shutil.rmtree(temporary)
    else:
        # An earlier complete copy may have survived a pointer publication error.
        original_pointer = destination / "latest.json"
        probe = destination / (".probe-" + uuid.uuid4().hex)
        probe.mkdir()
        try:
            # Validation without repointing the active destination pointer.
            candidate_index = read_json(completed / "index.json")
            if digest(completed / "index.json") != pointer["index_sha256"]:
                raise ValueError("Existing generation has wrong index digest")
            for shard in candidate_index["shards"]:
                if digest(completed / shard["file"]) != shard["sha256"]:
                    raise ValueError("Existing generation is corrupt")
        finally:
            probe.rmdir()
    observed = read_json(destination / "latest.json") if (destination / "latest.json").exists() else None
    if observed != previous:
        # Do not overwrite another runtime's publication or remove its files.
        # Drive/FUSE offers no atomic compare-and-swap: concurrent writers are
        # unsupported even with these defensive change checks.
        raise ValueError("Destination checkpoint changed during copy; refusing publication")
    atomic_json(destination / "latest.json", pointer)
    # Publication is last; no restore can see a pointer to a partial copy.
    verified, verified_index = checkpoint(destination, verify_payloads=force_verify)
    if verified != pointer or verified_index["identity"] != index["identity"]:
        raise ValueError("Published backup verification failed")
    if remove_previous and previous and previous["directory"] != pointer["directory"]:
        old = destination / previous["directory"]
        # The previously validated pointer is the only deletion candidate.
        if old.exists() and not old.is_symlink() and GENERATION.fullmatch(old.name):
            shutil.rmtree(old)
    return True


class Mirror:
    def __init__(self, local: Path, drive: Path, run_name: str):
        self.local, self.drive = local, drive
        self.run_name = run_name
        self.error: BaseException | None = None
        self.cancel = threading.Event()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.verified = False
        self.drive.mkdir(parents=True, exist_ok=True)
        marker = self.drive / MARKER
        expected = {"format": 1, "run_name": run_name, "owner": "packtok-semantic-scale-colab"}
        if marker.exists():
            if read_json(marker) != expected:
                raise ValueError("Drive run directory is owned by another run")
        else:
            if any(self.drive.iterdir()):
                raise ValueError("Refusing to adopt a nonempty unmarked Drive directory")
            atomic_json(marker, expected)

    def restore(self) -> None:
        # Validate all remote payloads before copying into the active local run.
        if (self.local / "config.json").exists() and (self.drive / "config.json").exists():
            if read_json(self.local / "config.json") != read_json(self.drive / "config.json"):
                raise ValueError("Local/Drive configurations differ")
        root = self.drive / "checkpoints"
        if (root / "latest.json").exists():
            pointer, index = checkpoint(root, verify_payloads=True)
            local_root = self.local / "checkpoints"
            if (local_root / "latest.json").exists():
                old_pointer, old_index = checkpoint(local_root, verify_payloads=True)
                if old_index["identity"] != index["identity"]:
                    raise ValueError("Local checkpoint identity differs from Drive")
                if old_index["step"] > index["step"]:
                    # A previous backup failure may have left newer local state.
                    return
                if old_index["step"] == index["step"] and old_pointer != pointer:
                    raise ValueError("Local/Drive checkpoints diverge at the same step")
            # Logs are copied first so checkpoint metric progress can be replayed.
            copy_metadata(self.drive, self.local)
            transfer_checkpoint(root, local_root, force_verify=True, frozen_pointer=pointer)
        else:
            local_root = self.local / "checkpoints"
            if (local_root / "latest.json").exists():
                # A failed first backup may have copied only old metadata. Keep
                # the local checkpoint and its newer metrics together on retry.
                checkpoint(local_root, verify_payloads=True)
                return
            copy_metadata(self.drive, self.local)

    def sync(self, final: bool = False) -> bool:
        if final:
            self.verified = False
        # Freeze the complete generation BEFORE copying its logs. A writer can
        # publish a newer checkpoint while metadata is copied; using that newer
        # pointer afterward would pair checkpoint N+1 with only N's metric log.
        source = self.local / "checkpoints"
        frozen = read_json(source / "latest.json") if (source / "latest.json").exists() else None
        index = None
        if frozen is not None:
            _, index = checkpoint(source, verify_payloads=False, frozen_pointer=frozen)
        # Check before touching logs too: stale metadata can invalidate a newer
        # backup even when its weights/pointer are never replaced.
        destination_pointer(self.drive / "checkpoints", frozen, index)
        copy_metadata(self.local, self.drive)
        copied = transfer_checkpoint(source, self.drive / "checkpoints", force_verify=final,
                                     remove_previous=True, frozen_pointer=frozen) if frozen is not None else False
        if final:
            # Verify copied configuration/results, including preflight-only runs.
            for filename in ("config.json", "preflight.json", "notebook-summary.json"):
                source = self.local / filename
                if source.exists() and digest(source) != digest(self.drive / filename):
                    raise ValueError(f"Final metadata readback mismatch: {filename}")
            self.verified = True
        return copied

    def start(self) -> None:
        def monitor() -> None:
            while not self.stop_event.wait(5):
                try:
                    self.sync()
                except FileNotFoundError:
                    # Rust retention can remove an old generation while a slow
                    # Drive copy is in progress. Its temporary copy is discarded;
                    # retry the newly published latest pointer on the next tick.
                    continue
                except BaseException as error:
                    self.error = error
                    print("Drive-Sicherung fehlgeschlagen; GPU-Arbeit wird gestoppt: " + str(error), flush=True)
                    self.cancel.set()
                    return
        self.thread = threading.Thread(target=monitor, name="packtok-drive-backup", daemon=True)
        self.thread.start()

    def finish(self) -> bool:
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        # Final sync runs with the Rust writer stopped: no retention race remains.
        self.sync(final=True)
        self.error = None
        self.cancel.clear()
        return self.verified
