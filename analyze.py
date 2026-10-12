"""Statistics + figure for the RL assignment 3 results.

Input : results/<condition>_seed<k>_scores.csv  (written by evaluate.py; columns: length, score, ...)
Unit  : one number per training seed (mean score over all 30 lengths x 10 runs), so the
        statistics compare independent training runs, not individual test episodes.

Usage:
  python analyze.py --conditions fixed uniform curriculum reward strat_curriculum \
                    --pairs uniform:fixed curriculum:uniform reward:uniform strat_curriculum:curriculum \
                    --cap 20000

--pairs is the list of PRE-SPECIFIED comparisons (keep it short; Holm correction is applied
over exactly these). Format A:B tests whether A differs from B; Cliff's delta > 0 means A > B.
"""
import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
parser.add_argument("--conditions", nargs="+", required=True)
parser.add_argument("--pairs", nargs="*", default=[])
parser.add_argument("--cap", type=int, default=20000)
parser.add_argument("--out", default="analysis")
parser.add_argument("--boot", type=int, default=10000)
parser.add_argument("--metric", default="mean_score",
                    choices=["mean_score", "short_mean", "mid_mean", "long_mean", "worst5_mean"],
                    help="per-seed metric used for the omnibus/pairwise tests and the figure. "
                         "mean_score = primary; short_mean = mean over the 9 shortest lengths "
                         "(0.40-0.79, i.e. poles shorter than 0.8), pre-specified secondary metric.")
args = parser.parse_args()
rng = np.random.default_rng(0)
out = Path(args.out)
out.mkdir(exist_ok=True)

# ---- load: one row per (condition, seed) ----------------------------------------
rows = []
for cond in args.conditions:
    for f in sorted(Path(args.results).glob(f"{cond}_seed*_scores.csv")):
        m = re.fullmatch(rf"{re.escape(cond)}_seed(\d+)_scores\.csv", f.name)
        if not m:
            continue
        df = pd.read_csv(f)
        per_len = df.groupby("length")["score"].mean()
        rows.append({
            "condition": cond, "seed": int(m.group(1)),
            "mean_score": df["score"].mean(),
            "short_mean": per_len.iloc[:9].mean(),      # the 9 shortest test lengths (0.40-0.79, i.e. < 0.8)
            "mid_mean": per_len.iloc[9:20].mean(),      # lengths 0.83-1.37
            "long_mean": per_len.iloc[20:].mean(),      # the 10 longest lengths (1.41-1.80)
            "frac_over_500": (per_len > 500).mean(),
            "episodes_ge500": (df["score"] >= 500).mean(),     # fraction of EPISODES surviving phase 1
            "episodes_gt1000": (df["score"] > 1000).mean(),    # fraction of EPISODES surviving phase 2
            "frac_over_1000": (per_len > 1000).mean(),
            "frac_episodes_at_cap": (df["score"] >= args.cap).mean(),
            "worst5_mean": per_len.nsmallest(5).mean(),
        })
seeds = pd.DataFrame(rows)
seeds.to_csv(out / "per_seed_metrics.csv", index=False)
missing = [c for c in args.conditions if c not in set(seeds["condition"])]
if missing:
    raise SystemExit(f"No result files found for: {missing}")
print("seeds per condition:\n", seeds.groupby("condition").size().to_string(), "\n")


def boot_ci(x, n=args.boot):
    x = np.asarray(x)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    greater = (a[:, None] > b[None, :]).sum()
    less = (a[:, None] < b[None, :]).sum()
    return (greater - less) / (len(a) * len(b))


# ---- descriptive table -----------------------------------------------------------
table = []
for cond in args.conditions:
    d = seeds[seeds["condition"] == cond]
    lo, hi = boot_ci(d["mean_score"])
    table.append({
        "condition": cond, "n_seeds": len(d),
        "mean": d["mean_score"].mean(), "median": d["mean_score"].median(),
        "short_mean": d["short_mean"].mean(), "mid_mean": d["mid_mean"].mean(),
        "long_mean": d["long_mean"].mean(),
        "ci95_low": lo, "ci95_high": hi,
        "min": d["mean_score"].min(), "max": d["mean_score"].max(),
        "episodes_>=500": d["episodes_ge500"].mean(),
        "episodes_>1000": d["episodes_gt1000"].mean(),
        "frac_lengths_>500": d["frac_over_500"].mean(),
        "frac_lengths_>1000": d["frac_over_1000"].mean(),
        "worst5_mean": d["worst5_mean"].mean(),
        "episodes_at_cap": d["frac_episodes_at_cap"].mean(),
    })
table = pd.DataFrame(table)
table.to_csv(out / "summary_by_condition.csv", index=False)
print(table.round(3).to_string(index=False), "\n")

# ---- omnibus test ----------------------------------------------------------------
M = args.metric
print(f"Metric used for tests and figure: {M}\n")
groups = [seeds[seeds["condition"] == c][M].values for c in args.conditions]
if len(groups) > 2:
    h, p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis across {len(groups)} conditions: H={h:.3f}, p={p:.4g}\n")

# ---- pre-specified pairwise tests with Holm correction ----------------------------
res = []
for pair in args.pairs:
    a, b = pair.split(":")
    xa = seeds[seeds["condition"] == a][M].values
    xb = seeds[seeds["condition"] == b][M].values
    u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided", method="auto")
    res.append({"A": a, "B": b, "median_A": np.median(xa), "median_B": np.median(xb),
                "U": u, "p_raw": p, "cliffs_delta": cliffs_delta(xa, xb)})
if res:
    res = pd.DataFrame(res)
    order = res["p_raw"].argsort().values            # Holm step-down
    m = len(res)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * res.loc[idx, "p_raw"])
        adj[idx] = min(1.0, running)
    res["p_holm"] = adj
    res.to_csv(out / f"pairwise_tests_{M}.csv", index=False)
    print("Pre-specified pairwise comparisons (Mann-Whitney U, Holm-adjusted):")
    print(res.round(4).to_string(index=False), "\n")

# ---- figure: per-seed means, one dot per training seed ----------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for i, cond in enumerate(args.conditions):
    d = seeds[seeds["condition"] == cond][M].values
    ax.scatter(np.full(len(d), i) + rng.uniform(-0.12, 0.12, len(d)), d, alpha=0.8, s=28)
    ax.hlines(np.median(d), i - 0.25, i + 0.25, color="black", lw=2)
ax.set_xticks(range(len(args.conditions)))
ax.set_xticklabels(args.conditions, rotation=20, ha="right")
ax.set_ylabel(f"{M} per seed (cap {args.cap})")
ax.set_title("Per-seed performance (dot = one training seed, bar = median)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out / f"per_seed_{M}.png", dpi=200)
print(f"Wrote tables and figure to {out}/")
