"""Verify the authored-fixture M7 CUDA recovery result without extracting files."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile


def digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_results(root):
    accounting = json.loads((root / "accounting.json").read_bytes())
    if not accounting["release_verified"] or accounting["transport_exit"] != 0:
        raise ValueError("CUDA probe transport or release verification failed")
    result = root / "results.tar.gz"
    with tarfile.open(result, "r:gz") as archive:
        def member(name):
            entry = archive.getmember(name)
            if not entry.isfile():
                raise ValueError(f"expected regular file: {name}")
            with archive.extractfile(entry) as contents:
                return contents.read()

        if member("m7-diagnostics/exit-code.txt").strip() != b"0":
            raise ValueError("remote worker failed")
        hardware = member("hardware.csv").decode("utf-8").strip()
        if hardware.split(",", 1)[0] != "NVIDIA L4" or "\n" in hardware:
            raise ValueError("expected one NVIDIA L4")
        reports = {}
        for label, variant, step in (("control", "A", 96), ("packed", "C", 96),
                                     ("restart", "A", 40), ("restart", "A", 96)):
            report = json.loads(member(f"results/{label}/report-{step:06d}.json"))
            if (report["backend"], report["variant"], report["step"]) != ("cuda", variant, step):
                raise ValueError(f"incorrect backend, variant or step: {label}-{step}")
            for split in ("train_initial", "train_final", "validation_initial", "validation_final"):
                if not math.isfinite(report[split]["bits_per_byte"]):
                    raise ValueError(f"non-finite bits per byte: {label}-{step}-{split}")
            reports[f"{label}-{step}"] = report
        checkpoint = "training-v2/step-00000000000000000096.ptc"
        control = member(f"results/control/{checkpoint}")
        resumed = member(f"results/restart/{checkpoint}")
        journal = member("results/control/metrics.jsonl")
        resumed_journal = member("results/restart/metrics.jsonl")
        if control != resumed or journal != resumed_journal or reports["control-96"] != reports["restart-96"]:
            raise ValueError("interrupted and uninterrupted CUDA trajectories differ")
        if len(journal.splitlines()) != 96:
            raise ValueError("incomplete CUDA training journal")
    with result.open("rb") as packaged:
        archive_sha256 = digest(packaged)
    return {
        "bitwise_recovery": True,
        "accounting": accounting,
        "hardware": hardware,
        "results_archive_sha256": archive_sha256,
        "final_checkpoint_sha256": digest(io.BytesIO(control)),
        "metrics_sha256": digest(io.BytesIO(journal)),
        "reports": reports,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("allocation", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_results(args.allocation), indent=2))
