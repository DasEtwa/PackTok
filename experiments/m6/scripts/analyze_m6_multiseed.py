#!/usr/bin/env python3
"""Summarize final validation BPB from the frozen paired-seed M6 T study."""
import argparse
import json
import math
from pathlib import Path

SEEDS = (20261008, 20261009, 20261010, 20261011, 20261012)
T95 = {2: 12.706205, 3: 4.302653, 4: 3.182446, 5: 2.776445}

def summarize(pairs):
    by_seed = {int(row["seed"]): row for row in pairs}
    if tuple(sorted(by_seed)) != SEEDS or len(pairs) != len(SEEDS):
        raise ValueError(f"expected exactly one pair for each frozen seed {SEEDS}")
    rows=[]
    for seed in SEEDS:
        row=by_seed[seed]
        a=float(row["A_validation_bpb"]); c=float(row["C_validation_bpb"])
        if not math.isfinite(a) or not math.isfinite(c) or a <= 0 or c <= 0:
            raise ValueError(f"non-finite or non-positive BPB for seed {seed}")
        delta=c-a
        rows.append({"seed":seed,"A_validation_bpb":a,"C_validation_bpb":c,
                     "delta_C_minus_A_bpb":delta,
                     "relative_delta_percent_of_A":100.0*delta/a})
    def stats(selected):
        ds=[r["delta_C_minus_A_bpb"] for r in selected]
        n=len(ds); mean=sum(ds)/n
        sd=math.sqrt(sum((x-mean)**2 for x in ds)/(n-1)) if n>1 else None
        half=T95[n]*sd/math.sqrt(n) if sd is not None and n in T95 else None
        return {"n":n,"mean_delta_bpb":mean,"sample_sd_bpb":sd,
                "median_delta_bpb":sorted(ds)[n//2] if n%2 else (sorted(ds)[n//2-1]+sorted(ds)[n//2])/2,
                "ci95_paired_t":[mean-half,mean+half] if half is not None else None,
                "C_better_count":sum(x<0 for x in ds),"A_better_count":sum(x>0 for x in ds),
                "tie_count":sum(x==0 for x in ds)}
    return {"metric":"validation bits per raw byte","delta_definition":"C minus A; negative favors C",
            "all_pairs":stats(rows),"new_seed_sensitivity":stats([r for r in rows if r["seed"] != 20261008]),
            "seed_20261008_status":"historical M5 result already observed; not blinded",
            "pairs":rows,"interval_method":"two-sided Student paired-t interval on seed-level deltas; descriptive at n=5"}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input",type=Path,help="JSON object with pairs:[{seed,A_validation_bpb,C_validation_bpb},...]")
    args=ap.parse_args()
    print(json.dumps(summarize(json.loads(args.input.read_text())),indent=2,sort_keys=True))
if __name__=="__main__": main()
