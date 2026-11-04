#!/usr/bin/env python3
"""Verify frozen research bytes and checkout line endings without rewriting files."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = "experiments/m6/analysis/"
MANIFEST = ANALYSIS + "SHA256SUMS-v2.txt"
SOURCE = ANALYSIS + "m5-tokenizer-efficiency.json"
OUTPUT = ANALYSIS + "m5-tokenizer-efficiency-v2-labels.json"
ORIGINAL = ANALYSIS + "m5-tokenizer-efficiency-v2-labels-original.json"
EXPECTED_PATHS = {
    "experiments/m6/scripts/analyze_m5_tokenizer_efficiency.py",
    "experiments/m6/scripts/relabel_m5_tokenizer_efficiency.py",
    OUTPUT,
    ORIGINAL,
}
FROZEN = {
    SOURCE: "f1e39867b7929e0d14d04cb2a9ac90a9ef3e08edf0202099a90e5347eb5bad41",
    ORIGINAL: "78f341606b5960ead12eaa6e2d6e3d1c3fb753642c65e88e97b89df8411b4e76",
    "experiments/m5-gpu/configs/pilot-2k-v1.json": "6536d5194bc7d422bdef6e871b58317497794ed1553b197a927e0434aa65c3d8",
    "experiments/m5-gpu/configs/m6-five-seed-2k-v1.json": "50cf36a6c9edf0f4423007b846da28ba3984eb37a8102664b27e165e9ee4954e",
    "experiments/m5-gpu/configs/extended-20k-v1.json": "2943cb456a82d76dc1c0f8371bd946897e151dd344b03de239cf98bb3534f621",
    "experiments/m5-gpu/configs/extended-30k-extension-v1.json": "a121011c57269b82361403fd180fc4300470e8839b69303224926b4eb6bd86f0",
    "experiments/m5-gpu/configs/primary.json": "8dde6647e5bdddf00ba99ce0c147d523206d89f29bb16d5615814821aa07b601",
}


def read(path):
    return (ROOT / path).read_bytes()


def check_hash(path, expected):
    actual = hashlib.sha256(read(path)).hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 mismatch: expected {expected}, got {actual}")
    print(f"SHA-256 OK: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-checkout", action="store_true", help="require protected worktree bytes to match the Git index")
    args = parser.parse_args()
    entries = {}
    for line in read(MANIFEST).decode("ascii").splitlines():
        digest, path = line.split("  ", 1)
        if path in entries or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid or duplicate checksum entry: {line}")
        entries[path] = digest
    if entries.keys() != EXPECTED_PATHS:
        raise ValueError(f"unexpected checksum entries: {set(entries) ^ EXPECTED_PATHS}")
    for path, digest in entries.items():
        check_hash(path, digest)
    for path, digest in FROZEN.items():
        check_hash(path, digest)

    source = read(SOURCE)
    new = json.loads(read(OUTPUT))
    old = json.loads(read(ORIGINAL))
    if new["relabeling"]["source_revision1_sha256"] != hashlib.sha256(source).hexdigest():
        raise ValueError("v2 source provenance does not match exact committed v1 bytes")
    old_hash = old["relabeling"]["source_revision1_sha256"]
    if old_hash != hashlib.sha256(source.replace(b"\n", b"\r\n")).hexdigest():
        raise ValueError("historical output does not preserve the recorded CRLF source hash")
    old["relabeling"]["source_revision1_sha256"] = new["relabeling"]["source_revision1_sha256"]
    if old != new:
        raise ValueError("historical and corrected v2 outputs differ beyond source provenance")
    print("M6 historical output and source provenance: OK")

    shell_paths = subprocess.check_output(["git", "ls-files", "-z", "--", "*.sh"], cwd=ROOT).decode().split("\0")
    shell_paths = [p for p in shell_paths if p]
    if not shell_paths:
        raise ValueError("no tracked shell scripts found")
    for path in shell_paths:
        if b"\r" in read(path):
            raise ValueError(f"shell script has non-LF line endings: {path}")
    print(f"Shell scripts use LF: {len(shell_paths)} files")

    if args.check_checkout:
        protected = set(shell_paths) | set(FROZEN) | EXPECTED_PATHS | {
            MANIFEST, ANALYSIS + "SHA256SUMS.txt"
        }
        for path in sorted(protected):
            indexed = subprocess.check_output(["git", "show", f":{path}"], cwd=ROOT)
            if read(path) != indexed:
                raise ValueError(f"checkout bytes differ from Git index: {path}")
        print(f"Protected checkout bytes match Git index: {len(protected)} files")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Research verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
