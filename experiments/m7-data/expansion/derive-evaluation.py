"""Create a derived validation view of the verified historical M7 release."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def write_json(path, data):
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, ensure_ascii=False)
        output.write("\n")


def derive(base, overlap_file, staging):
    base_manifest_bytes = (base / "dataset.json").read_bytes()
    manifest = json.loads(base_manifest_bytes)
    if manifest["version"] != 1:
        raise ValueError("expected M7 dataset format version 1")
    overlap_bytes = overlap_file.read_bytes()
    overlaps = json.loads(overlap_bytes)
    excluded = {
        row["source"]: row
        for overlap in overlaps if overlap["split_count_after_filter"] > 1
        for row in overlap["sources"]
        if row["split"] == "validation" and row["admitted_documents"]
    }
    staging.mkdir()
    os.link(base / "sources.json", staging / "sources.json")
    seen = set()
    for shard in manifest["shards"]:
        name = shard["name"]
        if Path(name).name != name or not name.endswith(".jsonl"):
            raise ValueError("invalid shard path")
        if shard["split"] != "validation":
            os.link(base / name, staging / name)
            continue
        rows = size = text_bytes = 0
        digest = hashlib.sha256()
        with (base / name).open("rb") as source, (staging / name).open("xb") as target:
            for line in source:
                record = json.loads(line)
                source_id = record["source"]
                text_size = len(record["text"].encode("utf-8"))
                if source_id in excluded:
                    stats = manifest["sources"][source_id]
                    if (stats["admitted_documents"], stats["admitted_bytes"]) != (1, text_size):
                        raise ValueError(f"expected one pinned validation document for {source_id}")
                    if excluded[source_id]["admitted_bytes"] != text_size:
                        raise ValueError(f"overlap receipt differs for {source_id}")
                    seen.add(source_id)
                    reason = "evaluation-author-name-overlap"
                    stats["admitted_documents"] = stats["admitted_bytes"] = 0
                    stats["rejected"][reason] = 1
                    stats["rejected_bytes"][reason] = text_size
                    continue
                target.write(line)
                digest.update(line)
                rows += 1
                size += len(line)
                text_bytes += text_size
            target.flush()
            os.fsync(target.fileno())
        if rows == 0:
            raise ValueError("derived validation shard would be empty")
        shard.update(rows=rows, bytes=size, text_bytes=text_bytes, sha256=digest.hexdigest())
    if seen != set(excluded):
        raise ValueError("overlap receipt includes missing validation sources")
    write_json(staging / "dataset.json", manifest)
    validation = [s for s in manifest["shards"] if s["split"] == "validation"]
    write_json(staging / "evaluation-receipt.json", {
        "policy_version": 1,
        "rule": "Omit admitted validation records whose normalized author name also occurs in train or test.",
        "base_manifest_sha256": hashlib.sha256(base_manifest_bytes).hexdigest(),
        "derived_manifest_sha256": hashlib.sha256((staging / "dataset.json").read_bytes()).hexdigest(),
        "author_overlap_receipt_sha256": hashlib.sha256(overlap_bytes).hexdigest(),
        "excluded_source_ids": sorted(seen),
        "validation_documents": sum(s["rows"] for s in validation),
        "validation_text_bytes": sum(s["text_bytes"] for s in validation),
        "validation_shards": validation,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="verified historical dataset")
    parser.add_argument("overlaps", type=Path)
    parser.add_argument("staging", type=Path, help="new directory on the base filesystem")
    args = parser.parse_args()
    derive(args.base, args.overlaps, args.staging)
