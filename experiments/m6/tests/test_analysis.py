import importlib.util
import json
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_m5_tokenizer_efficiency.py"
SPEC = importlib.util.spec_from_file_location("m5_analysis", SCRIPT)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


class AnalysisHelpersTest(unittest.TestCase):
    def test_frozen_binary_headers(self):
        self.assertEqual(ANALYSIS.SEQ_HEADER, bytes.fromhex("50544d3553455101"))
        self.assertEqual(ANALYSIS.MAP_HEADER, bytes.fromhex("50544d4150340001"))

    def test_manifest_spans_must_cover_exact_split(self):
        manifest = {
            "sources": [{"name": "en", "category": "english-prose"}],
            "units": [{"proposed": "train", "kept": True, "source": "en",
                       "start": 0, "end": 5}]
        }
        self.assertEqual(ANALYSIS.spans_from_manifest(manifest, "train", 5),
                         [(0, 5, "english-literature")])
        with self.assertRaisesRegex(ValueError, "covers"):
            ANALYSIS.spans_from_manifest(manifest, "train", 6)

    def test_token_byte_intervals_count_boundary_lexemes(self):
        # Token ends align with the three word spans [0,2), [3,5), [6,8).
        ends = [2, 5, 8]
        spans = [(0, 7, "english-literature")]
        raw = b"ab cd ef!"
        result = ANALYSIS.fragmentation(raw, spans, ends, "unicode-words")
        row = result["english-literature"]
        self.assertEqual(row["lexemes"], 3)
        self.assertEqual(row["multi_token_percent"], 0.0)
        self.assertEqual(row["mean_tokens_per_lexeme"], 1.0)

    def test_lexeme_spanning_multiple_tokens_is_counted(self):
        # One six-byte identifier crosses three token intervals.
        ends = [2, 4, 6]
        spans = [(0, 6, "rust-source")]
        result = ANALYSIS.fragmentation(b"abcdef", spans, ends, "rust-identifiers")
        row = result["rust-source"]
        self.assertEqual(row["lexemes"], 1)
        self.assertEqual(row["mean_tokens_per_lexeme"], 3.0)
        self.assertEqual(row["multi_token_percent"], 100.0)

    def test_pack_names_follow_variant_registry(self):
        # A is the flat M1 registry; C follows lexical-v1 (TEXT=0, NUMBER=1, STRUCTURE=2).
        self.assertEqual(ANALYSIS.pack_name("A", 0), "FLAT")
        self.assertEqual(ANALYSIS.pack_name("C", 0), "TEXT")
        self.assertEqual(ANALYSIS.pack_name("C", 1), "NUMBER")
        self.assertEqual(ANALYSIS.pack_name("C", 2), "STRUCTURE")
        self.assertEqual(ANALYSIS.pack_name("C", 65535), "BYTE_FALLBACK")
        self.assertEqual(ANALYSIS.pack_name("C", 7), "PACK_7")

    def test_pack_names_match_packtok_packs_contract(self):
        source = (Path(__file__).parents[3] / "crates" / "packtok-packs" / "src" / "lib.rs").read_text(encoding="utf-8")
        for name, pack in (("TEXT", 0), ("NUMBER", 1), ("STRUCTURE", 2)):
            self.assertIn(f"pub const {name}_PACK_ID: PackId = {pack};", source)
            self.assertEqual(ANALYSIS.pack_name("C", pack), name)


RELABEL_SCRIPT = Path(__file__).parents[1] / "scripts" / "relabel_m5_tokenizer_efficiency.py"
RELABEL_SPEC = importlib.util.spec_from_file_location("m5_relabel", RELABEL_SCRIPT)
RELABEL = importlib.util.module_from_spec(RELABEL_SPEC)
RELABEL_SPEC.loader.exec_module(RELABEL)
V1_JSON = Path(__file__).parents[1] / "analysis" / "m5-tokenizer-efficiency.json"


class RelabelTest(unittest.TestCase):
    def setUp(self):
        self.v1 = json.loads(V1_JSON.read_text(encoding="utf-8"))
        self.v2 = RELABEL.relabel(self.v1)

    def test_c_allocation_matches_frozen_artifact(self):
        self.assertEqual(self.v2["variants"]["C"]["vocabulary"]["allocated_ids_by_pack"],
                         {"BYTE_FALLBACK": 256, "NUMBER": 4, "STRUCTURE": 56, "TEXT": 196})

    def test_pack_ids_and_numbers_are_unchanged(self):
        for variant in ("A", "C"):
            old = self.v1["variants"][variant]["vocabulary"]["pack_utilization"]
            new = self.v2["variants"][variant]["vocabulary"]["pack_utilization"]
            self.assertEqual(sorted(old.values(), key=lambda e: e["pack_id"]),
                             sorted(new.values(), key=lambda e: e["pack_id"]))
            for name, entry in new.items():
                self.assertEqual(RELABEL.CORRECT[variant][entry["pack_id"]], name)

    def test_a_is_unchanged_apart_from_schema(self):
        self.assertEqual(self.v1["variants"]["A"], self.v2["variants"]["A"])

    def test_input_document_is_not_mutated(self):
        self.assertEqual(self.v1, json.loads(V1_JSON.read_text(encoding="utf-8")))

    def test_rejects_wrong_schema(self):
        with self.assertRaises(ValueError):
            RELABEL.relabel(self.v2)


if __name__ == "__main__":
    unittest.main()
