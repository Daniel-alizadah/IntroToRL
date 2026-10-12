"""Pick which seed's .pth to submit for each strategy, using VALIDATION logs only
(never the test-script scores).

Rule (fixed before looking at any test result):
  1. highest best-validation score over the run
  2. ties (many runs hit the validation cap) -> the run whose validation score reached
     the cap in the most validation rounds (more consistent, not a single lucky round)
  3. remaining ties -> lowest seed index

Usage: python select_checkpoint.py curriculum margin strat_curriculum
Reads logs/<condition>_seed<k>_validation.csv and prints the table and selected seed.
The final submitted models are listed in the README under submission/.
"""
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("conditions", nargs="+")
parser.add_argument("--logs", default="logs")
parser.add_argument("--cap", type=float, default=10000.0, help="validation cap (VALIDATION_MAX_STEPS)")
args = parser.parse_args()

for cond in args.conditions:
    rows = []
    for k in range(8):
        f = Path(args.logs) / f"{cond}_seed{k}_validation.csv"
        if not f.exists():
            continue
        v = pd.read_csv(f)["validation_score"]
        rows.append({"seed": k, "best_validation": v.max(),
                     "rounds_at_cap": int((v >= args.cap - 1e-6).sum()), "rounds": len(v)})
    if not rows:
        print(f"{cond}: no validation logs found\n")
        continue
    t = pd.DataFrame(rows).sort_values(["best_validation", "rounds_at_cap", "seed"],
                                       ascending=[False, False, True])
    print(f"== {cond} ==")
    print(t.round(1).to_string(index=False))
    print(f"-> selected {cond} seed {int(t.iloc[0].seed)}\n")
