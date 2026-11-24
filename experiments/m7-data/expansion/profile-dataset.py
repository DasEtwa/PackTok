"""Describe a verified M7 historical dataset without reading its full text."""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def profile(dataset, output):
    catalog = json.loads((dataset / "sources.json").read_text(encoding="utf-8"))
    manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    counts = defaultdict(lambda: {"documents": 0, "bytes": 0, "groups": set()})
    splits = defaultdict(set)
    author_sources = defaultdict(list)
    rows = []
    for source in sorted(catalog["sources"], key=lambda item: item["id"]):
        stats = manifest["sources"][source["id"]]
        documents, size = stats["admitted_documents"], stats["admitted_bytes"]
        split, language, category = [source[key] for key in ("split", "language", "category")]
        author = source["author"]
        if author != "Unattributed in source":
            key = " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", author).casefold()))
            if key:
                author_sources[key].append({
                    "author": author,
                    "source": source["id"],
                    "split": split,
                    "group": source["group"],
                    "admitted_documents": documents,
                    "admitted_bytes": size,
                    "rejected": stats["rejected"],
                })
        if not documents:
            continue
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
        shards = [s for s in manifest["shards"] if s["split"] == split]
        for metric, field in (("documents", "rows"), ("bytes", "text_bytes")):
            if counts[(split, "all", "all")][metric] != sum(s[field] for s in shards):
                raise ValueError(f"{split} source {metric} do not reconcile to shards")
    author_overlaps = [
        {
            "normalized_name": name,
            "split_count_after_filter": len({s["split"] for s in appearances if s["admitted_documents"]}),
            "sources": appearances,
        }
        for name, appearances in sorted(author_sources.items())
        if len({s["split"] for s in appearances}) > 1
    ]
    with (output / "split-profile.json").open("x", encoding="utf-8") as log:
        json.dump(report, log, indent=2, ensure_ascii=False)
        log.write("\n")
    with (output / "validation-index.jsonl").open("x", encoding="utf-8") as log:
        for row in rows:
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "author-overlap.json").open("x", encoding="utf-8") as log:
        json.dump(author_overlaps, log, indent=2, ensure_ascii=False)
        log.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    profile(args.dataset, args.output)
