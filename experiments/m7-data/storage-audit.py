"""Offline size accounting for already verified LIVWIN02 files."""
import argparse
import json
from pathlib import Path
import struct


def audit(folder):
    meta = json.loads((folder / "tokens.json").read_text())
    variants = {}
    for key, variant in meta["variants"].items():
        for split, stream in variant["streams"].items():
            path = folder / f"{key}-{split}.windows"
            frames, tokens = stream["frames"], stream["tokens"]
            populated = stream["scored_tokens"] + frames
            parts = {
                "file_magic": 8,
                "frame_count_and_flags": frames,
                "padding": (9 * frames - populated) * 8,
                "token_overlap": (populated - tokens) * 8,
                "logical_ids": tokens * 4,
                "logical_lengths": tokens * 4,
            }
            if sum(parts.values()) != stream["bytes"] or stream["bytes"] != path.stat().st_size:
                raise ValueError(f"inconsistent size for {path}")
            maximum = 0
            with path.open("rb") as f:
                if f.read(8) != b"LIVWIN02":
                    raise ValueError(f"unsupported header in {path}")
                for chunk in iter(lambda: f.read(73 * 131072), b""):
                    if len(chunk) % 73:
                        raise ValueError(f"partial record in {path}")
                    for pos in range(0, len(chunk), 73):
                        count = chunk[pos] & 127
                        if not 1 <= count <= 9:
                            raise ValueError(f"invalid record length in {path}")
                        for i in range(pos + 5, pos + 1 + count * 8, 8):
                            maximum = max(maximum, struct.unpack_from("<I", chunk, i)[0])
            variants[f"{key}-{split}"] = dict(stored_bytes=stream["bytes"],
                tokens=tokens, text_bytes=stream["text_bytes"], breakdown=parts,
                maximum_token_byte_length=maximum,
                u16_u8_payload_bytes=3*tokens if maximum <= 255 else None,
                u16_only_payload_bytes=2*tokens)
    return variants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder), indent=2))
