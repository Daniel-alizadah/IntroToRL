"""Figures for the paper (appendix) from results/<condition>_seed<k>_scores.csv.

Output (in --out, default figures/):
  appendix_bars.png   one bar plot per approach: mean score per pole length (bar) with the
                      standard deviation across the 8 training seeds (error bar); overall mean in the title.
  per_length_lines.png  optional line plot of the per-length means for the main strategies.

Usage:  python make_plots.py            (run from the folder that contains results/)
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results")
parser.add_argument("--out", default="figures")
parser.add_argument("--cap", type=int, default=20000)
args = parser.parse_args()
out = Path(args.out)
out.mkdir(exist_ok=True)

# (condition name in results/, label shown in the figure)
PANELS = [
    ("fixed", "Fixed (0.5)"),
    ("uniform", "Uniform"),
    ("curriculum", "S1 Curriculum"),
    ("strat_curriculum", "S2 Stratified + curriculum"),
    ("margin", "S3 Margin reward"),
    ("uniform_oyku", "Uniform (2nd laptop)"),
]


def load(cond):
    """Return a (n_seeds x 30) array of per-length mean scores and the sorted lengths."""
    per_seed = []
    for f in sorted(Path(args.results).glob(f"{cond}_seed*_scores.csv")):
        if not re.fullmatch(rf"{re.escape(cond)}_seed\d+_scores\.csv", f.name):
            continue
        df = pd.read_csv(f)
        per_seed.append(df.groupby("length")["score"].mean().sort_index())
    if not per_seed:
        raise SystemExit(f"No result files for {cond}")
    mat = pd.concat(per_seed, axis=1)
    return mat.values.T, mat.index.values


# ---- appendix: bars with std across seeds ------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.6), sharey=True)
for ax, (cond, label) in zip(axes.ravel(), PANELS):
    data, lengths = load(cond)
    mean, std = data.mean(axis=0), data.std(axis=0, ddof=1)
    ax.bar(range(len(lengths)), mean, yerr=std, capsize=1.5, color="#4C72B0",
           error_kw={"elinewidth": 0.7, "ecolor": "#333333"})
    ax.set_title(f"{label}  (overall mean {data.mean():,.0f})", fontsize=9)
    ax.set_xticks(range(0, len(lengths), 5))
    ax.set_xticklabels([f"{lengths[i]:.2f}" for i in range(0, len(lengths), 5)], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, args.cap * 1.15)
for ax in axes[-1]:
    ax.set_xlabel("pole length", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("test score", fontsize=8)
fig.suptitle(f"Mean test score per pole length (bars: mean over 8 seeds, error bars: std across seeds; cap {args.cap:,})",
             fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out / "appendix_bars.png", dpi=250)
plt.close(fig)

# ---- optional main-text line plot ---------------------------------------------------
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for cond, label in PANELS[:5]:
    data, lengths = load(cond)
    ax.plot(lengths, data.mean(axis=0), label=label.replace(" + curriculum", ""), lw=1.2)
ax.set_xlabel("pole length", fontsize=8)
ax.set_ylabel("mean test score", fontsize=8)
ax.tick_params(labelsize=7)
ax.grid(alpha=0.3)
ax.legend(fontsize=6, frameon=False)
fig.tight_layout()
fig.savefig(out / "per_length_lines.png", dpi=250)
print(f"Wrote {out}/appendix_bars.png and {out}/per_length_lines.png")
