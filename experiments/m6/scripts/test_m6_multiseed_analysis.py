import importlib.util
import unittest
from pathlib import Path
p=Path(__file__).with_name("analyze_m6_multiseed.py")
spec=importlib.util.spec_from_file_location("analysis",p); analysis=importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)
class TestSummary(unittest.TestCase):
 def test_frozen_seed_set_and_deltas(self):
  pairs=[{"seed":s,"A_validation_bpb":2.0,"C_validation_bpb":2.0+d} for s,d in zip(analysis.SEEDS,(-.02,-.01,0,.01,.02))]
  r=analysis.summarize(pairs)
  self.assertAlmostEqual(r["all_pairs"]["mean_delta_bpb"],0.0)
  self.assertEqual(r["all_pairs"]["tie_count"],1)
  self.assertEqual(r["new_seed_sensitivity"]["n"],4)
  self.assertAlmostEqual(r["all_pairs"]["ci95_paired_t"][0],-.01963243087)
  self.assertAlmostEqual(r["all_pairs"]["ci95_paired_t"][1],.01963243087)
 def test_rejects_missing_seed_and_nonfinite_metrics(self):
  with self.assertRaises(ValueError): analysis.summarize([])
  rows=[{"seed":s,"A_validation_bpb":2,"C_validation_bpb":2} for s in analysis.SEEDS]
  rows[-1]["C_validation_bpb"]=float("nan")
  with self.assertRaises(ValueError): analysis.summarize(rows)
if __name__=="__main__":unittest.main(verbosity=2)
