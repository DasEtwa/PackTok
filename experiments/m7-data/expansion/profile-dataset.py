"""Describe a verified M7 historical dataset without reading its full text."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def profile(dataset, output):
    catalog = json.loads((dataset / "sources.json").read_text())
    manifest = json.loads((dataset / "dataset.json").read_text())
    counts = defaultdict(lambda: {"documents": 0, "bytes": 0, "groups": set()})
    splits = defaultdict(set)
    rows = []
    for source in sorted(catalog["sources"], key=lambda item: item["id"]):
        stats = manifest["sources"][source["id"]]
        documents, size = stats["admitted_documents"], stats["admitted_bytes"]
        if not documents:
            continue
        split, language, category = [source[key] for key in ("split", "language", "category")]
        for key in [(split, "all", "all"), (split, language, "all"), (split, language, category)]:
            count = counts[key]
            count["documents"] += documents
            count["bytes"] += size
            count["groups"].add(source["group"])
        splits[split].add(source["group"])
        if split == "validation":
            rows.append(dict(source=source["id"], group=source["group"], language=language,
                category=category, author=source["author"], archive_edition=source["version"],
                bytes=size, documents=documents, extracted_sha256=source["sha256"]))
    if any(splits[a] & splits[b] for a, b in [
        ("train", "validation"), ("train", "test"), ("validation", "test")
    ]):
        raise ValueError("group occurs in multiple splits")
    report = [{
        "split": split, "language": language, "category": category,
        "documents": count["documents"], "bytes": count["bytes"], "groups": len(count["groups"]),
    } for (split, language, category), count in sorted(counts.items())]
    for split in ("train", "validation", "test"):
        measured = sum(s["text_bytes"] for s in manifest["shards"] if s["split"] == split)
        if counts[(split,"all","all")]["bytes"] != measured:
            raise ValueError(f"{split} source counters do not reconcile to shards")
    with (output / "split-profile.json").open("x") as log:
        json.dump(report, log, indent=2, ensure_ascii=False)
        log.write("\n")
    with (output / "validation-index.jsonl").open("x") as log:
        for row in rows:
            log.write(json.dumps(row, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    profile(args.dataset, args.output)
