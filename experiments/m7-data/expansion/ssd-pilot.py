"""Compare real PackTok dedup indexes on HDD and SSD with data and output on D:."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time


RESERVE = 10_000_000_000
MIB = 1 << 20
CAP_MIB = {("train", "de"): 110, ("train", "en"): 110,
           ("validation", "de"): 22, ("validation", "en"): 12,
           ("test", "de"): 20, ("test", "en"): 9}


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def describe_sample(base, plan):
    with plan.open("rb") as source:
        catalog = json.load(source)
    totals = defaultdict(int)
    for entry in catalog["sources"]:
        totals[(entry["split"], entry["language"])] += entry["bytes"]
    return {
        "base_manifest_sha256": sha(base / "dataset.json"),
        "input_catalog_sha256": sha(plan),
        "sample": "Approved whole sources sorted by ID and greedily bounded per split/language cell.",
        "text_bytes": sum(totals.values()),
        "documents": len(catalog["sources"]),
        "cell_bytes": {"/".join(k): v for k, v in sorted(totals.items())},
    }


def sample(base, work):
    manifest = json.loads((base / "dataset.json").read_bytes())
    catalog = json.loads((base / "sources.json").read_bytes())
    accepted = manifest["sources"]
    totals = defaultdict(int)
    chosen = {}
    for source in sorted(catalog["sources"], key=lambda entry: entry["id"]):
        cell = (source["split"], source["language"])
        size = source["bytes"]
        stats = accepted[source["id"]]
        if stats["admitted_documents"] != 1 or stats["admitted_bytes"] != size:
            continue
        if totals[cell] + size <= CAP_MIB[cell] * MIB:
            totals[cell] += size
            chosen[source["id"]] = source
    cache = work / "cache"
    cache.mkdir()
    found = set()
    for shard in manifest["shards"]:
        with (base / shard["name"]).open("rb") as source:
            for line in source:
                record = json.loads(line)
                identity = record["source"]
                if identity not in chosen:
                    continue
                entry = chosen[identity]
                body = record["text"].encode("utf-8")
                if identity in found or len(body) != entry["bytes"] or hashlib.sha256(body).hexdigest() != entry["sha256"]:
                    raise ValueError(f"source changed: {identity}")
                with (cache / identity).open("xb") as dst:
                    dst.write(body)
                found.add(identity)
    if found != set(chosen) or set(totals) != set(CAP_MIB):
        raise ValueError("incomplete benchmark sample")
    plan = work / "catalog.json"
    plan.write_text(json.dumps({"version": 1, "sources": list(chosen.values())},
                               ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return plan, cache


def run(bin_path, plan, cache, work, placement, root, ssd):
    for path in sorted(cache.iterdir()):
        sha(path)  # Match source page-cache preparation; dedup indexes always start empty.
    with tempfile.TemporaryDirectory(prefix="packtok-m7-index-", dir=root) as directory:
        scratch = Path(directory)
        output = work / placement
        timing = work / f"{placement}.time"
        command = ["/usr/bin/time", "-f", "%e %U %S %M %I %O", "-o", str(timing),
                   str(bin_path), "build", str(plan), str(cache), str(output), "--temp-dir", str(scratch)]
        free_min = shutil.disk_usage(ssd).free
        peak_index = 0
        if free_min < RESERVE:
            raise RuntimeError("SSD reserve unavailable")
        with (work / f"{placement}.log").open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while child.poll() is None:
                    free_min = min(free_min, shutil.disk_usage(ssd).free)
                    if free_min < RESERVE + 100_000_000:
                        raise RuntimeError("SSD reserve guard reached")
                    for index in scratch.glob("*/dedup.redb"):
                        try:
                            peak_index = max(peak_index, index.stat().st_size)
                        except FileNotFoundError:
                            pass
                    time.sleep(0.5)
                if child.returncode:
                    raise RuntimeError(f"{placement} exited {child.returncode}")
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait()
        if any(scratch.iterdir()):
            raise RuntimeError("unexpected scratch residue")
    values = timing.read_text().split()
    metrics = dict(zip(("wall_s", "user_s", "system_s", "max_rss_kib", "fs_inputs", "fs_outputs"),
                       [float(x) for x in values[:3]] + [int(x) for x in values[3:]], strict=True))
    return dict(placement=placement, min_ssd_free_bytes=free_min,
                sampled_peak_index_bytes=peak_index, manifest_sha256=sha(output / "dataset.json"), **metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("bin", "base", "work", "hdd", "ssd"):
        parser.add_argument(flag, type=lambda token: Path(token).resolve())
    parser.add_argument("--sample-dir", type=Path, help="reuse a prepared source catalog and cache")
    parser.add_argument("--prepare-only", action="store_true", help="write the source sample and exit")
    parser.add_argument("--reserve-volume", type=Path, help="physical SSD mount to poll if SSD scratch is a loop mount")
    parser.add_argument("--order", choices=("hdd-ssd", "ssd-hdd-ssd"), default="hdd-ssd")
    a = parser.parse_args()
    a.work.mkdir()
    if a.sample_dir:
        plan = a.sample_dir.resolve() / "catalog.json"
        cache = plan.parent / "cache"
    else:
        plan, cache = sample(a.base, a.work)
    result = describe_sample(a.base, plan)
    receipt = a.work / "receipt.json"
    if a.prepare_only:
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        return
    result["reserve_bytes"] = RESERVE
    result["runs"] = []
    order = [("hdd-1", a.hdd), ("ssd-1", a.ssd)]
    if a.order == "ssd-hdd-ssd":
        order = [order[1], order[0], ("ssd-2", a.ssd)]
    for label, root in order:
        data = run(a.bin, plan, cache, a.work, label, root, a.reserve_volume or a.ssd)
        result["runs"].append(data)
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(data), flush=True)
    if len({data["manifest_sha256"] for data in result["runs"]}) != 1:
        raise ValueError("placement changed a verified corpus manifest")


if __name__ == "__main__":
    main()
