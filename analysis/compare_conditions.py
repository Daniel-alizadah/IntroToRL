"""compare_conditions.py - summary + rank-based tests for the conditions we have locally.
Unit of analysis = one seed's mean score over the 30 test lengths (cap 20000).
Run from the repo root: python analysis/compare_conditions.py"""
import glob, itertools
import numpy as np, pandas as pd
from scipy import stats

CONDS = {"uniform": "results/uniform_rerun/uniform_seed*_scores.csv",
         "margin": "results/margin_seed*_scores.csv",
         "margin_pbrs": "results/margin_pbrs_seed*_scores.csv",
         "margin_linear": "results/margin_linear_seed*_scores.csv",
         "curriculum": "results/curriculum_seed*_scores.csv"}   # Daniel's CSVs


def per_seed(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f); pl = d.groupby("length")["score"].mean()
        short = d.length <= 0.5
        rows.append(dict(mean=pl.mean(), P1=(d.score >= 500).mean(), P2=(d.score > 1000).mean(),
                         cap=(d.score >= 20000).mean(), short=pl[pl.index <= 0.8].mean(),
                         long=pl[pl.index >= 1.4].mean(),
                         onset=((short) & (d.score >= 500) & (d.score <= 525)).sum() / short.sum()))
    return pd.DataFrame(rows)


def cliff(a, b):
    return np.mean([np.sign(x - y) for x in a for y in b])


S = {k: per_seed(p) for k, p in CONDS.items()}
for k, d in S.items():
    print(f"{k:12s} n={len(d)} mean {d['mean'].mean():7.0f} ± {d['mean'].std():6.0f} median {d['mean'].median():7.0f} "
          f"P1 {d.P1.mean():.3f} P2 {d.P2.mean():.3f} cap {d.cap.mean():.2f} short {d.short.mean():6.0f} "
          f"long {d.long.mean():6.0f} died-at-onset(l<=0.5) {d.onset.mean():.2f}")
print("seeds uniform:", S["uniform"]["mean"].round(0).tolist())
print("Kruskal-Wallis (overall mean):", stats.kruskal(*[d["mean"] for d in S.values()]))
pairs = [("margin", "uniform"), ("margin_pbrs", "uniform"), ("margin_linear", "uniform"), ("margin", "margin_pbrs"), ("margin", "curriculum")]
for metric in ["mean", "short", "P2"]:
    ps = [stats.mannwhitneyu(S[a][metric], S[b][metric]).pvalue for a, b in pairs]
    order = np.argsort(ps); holm = np.empty(len(ps)); run = 0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j]); holm[j] = min(1, run)
    for (a, b), p, h in zip(pairs, ps, holm):
        print(f"[{metric:5s}] {a:11s} vs {b:10s}: MWU p={p:.4f} Holm {h:.4f} Cliff {cliff(S[a][metric], S[b][metric]):+.2f}")
