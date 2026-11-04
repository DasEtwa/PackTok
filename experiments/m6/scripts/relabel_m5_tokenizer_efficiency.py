#!/usr/bin/env python3
"""Derive corrected pack labels from the committed revision-1 M6 analysis JSON.

Revision 1 of analyze_m5_tokenizer_efficiency.py used one pack-name table for
both variants: {0: FLAT, 1: TEXT, 2: NUMBER, 3: STRUCTURE}. That is correct for
the flat A registry (pack 0) but wrong for C, whose lexical-v1 artifact declares
TEXT=0, NUMBER=1 and STRUCTURE=2 (FORMAT.md; packtok-packs).

The frozen M5 TRAIN sequences are local ignored data, so the analysis cannot be
rerun from this checkout. All counts in revision 1 are keyed by pack ID before
naming, so renaming C's keys is an exact relabeling: no numeric value changes.
The revision-1 JSON is read, never modified; a new file is written.
"""
import argparse, hashlib, json
from pathlib import Path

V1_SCHEMA = "packtok-m6-m5-tokenizer-efficiency-v1"
V2_SCHEMA = "packtok-m6-m5-tokenizer-efficiency-v2"
# Revision-1 label -> pack ID (bijective for the IDs that occur).
V1_LABEL_TO_ID = {"FLAT": 0, "TEXT": 1, "NUMBER": 2, "STRUCTURE": 3, "BYTE_FALLBACK": 65535}
CORRECT = {
    "A": {0: "FLAT", 65535: "BYTE_FALLBACK"},
    "C": {0: "TEXT", 1: "NUMBER", 2: "STRUCTURE", 65535: "BYTE_FALLBACK"},
}
# Pack allocation declared by the frozen C tokenizer artifact
# (experiments/m5-gpu/artifacts/corpus-v2/C-0.packtok, SHA-256 334b8ce5...),
# as printed by `packtok inspect`.
C_ARTIFACT_ALLOCATION = {"TEXT": 196, "NUMBER": 4, "STRUCTURE": 56, "BYTE_FALLBACK": 256}


def rename(variant, label):
    if label not in V1_LABEL_TO_ID:
        raise ValueError(f"unknown revision-1 pack label {label!r}")
    pack = V1_LABEL_TO_ID[label]
    if pack not in CORRECT[variant]:
        raise ValueError(f"variant {variant} has no pack {pack}")
    return CORRECT[variant][pack]


def rename_keys(variant, mapping):
    out = {rename(variant, k): v for k, v in mapping.items()}
    if len(out) != len(mapping):
        raise ValueError("relabeling is not injective")
    return dict(sorted(out.items()))


def relabel(doc):
    if doc.get("schema") != V1_SCHEMA:
        raise ValueError(f"expected schema {V1_SCHEMA}")
    out = json.loads(json.dumps(doc))
    out["schema"] = V2_SCHEMA
    for variant, row in out["variants"].items():
        vocab = row["vocabulary"]
        for label, entry in vocab["pack_utilization"].items():
            if V1_LABEL_TO_ID[label] != entry["pack_id"]:
                raise ValueError("revision-1 label does not match its recorded pack_id")
        vocab["allocated_ids_by_pack"] = rename_keys(variant, vocab["allocated_ids_by_pack"])
        vocab["pack_utilization"] = rename_keys(variant, vocab["pack_utilization"])
        for entry in row["global_token_frequency_top20"]:
            entry["pack"] = rename(variant, entry["pack"])
        for domain in row["domains"].values():
            for key in ("pack_tokens", "pack_token_shares_percent"):
                domain[key] = rename_keys(variant, domain[key])
    if out["variants"]["C"]["vocabulary"]["allocated_ids_by_pack"] != C_ARTIFACT_ALLOCATION:
        raise ValueError("corrected C allocation does not match the frozen tokenizer artifact")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    raw = args.input.read_bytes()
    out = relabel(json.loads(raw.decode("utf-8")))
    out["relabeling"] = {
        "source_revision1_sha256": hashlib.sha256(raw).hexdigest(),
        "method": "exact rename of C pack-name keys using recorded pack IDs; numeric values unchanged",
        "c_pack_names": {str(k): v for k, v in CORRECT["C"].items()},
    }
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output),
                      "c_allocation": out["variants"]["C"]["vocabulary"]["allocated_ids_by_pack"]}))


if __name__ == "__main__":
    main()
