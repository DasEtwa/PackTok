import importlib.util
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


if __name__ == "__main__":
    unittest.main()
