"""stability_check.py - reproducible mechanism check for the stability-margin reward (S3).

For every network it runs SEEDED greedy episodes under NOMINAL dynamics (force 10 N, no test
perturbations) on the 30 test lengths, 3 episodes each, 500 steps (= phase 1 of the test), and
records:
  * mean |x| / x_max      (how far the cart drifts from the centre; lower = more margin)
  * mean |theta| / th_max (how far the pole leans; lower = more margin)
  * mean shaped reward    (the margin reward 1 - 0.5(th/th_max)^2 - 0.25(x/x_max)^2 that
                           these states WOULD receive, i.e. how strong the shaping signal is)
Then it compares groups with two-sided Mann-Whitney U tests (unit = one network/seed).

Run from the repo root:
  python analysis/stability_check.py
Outputs: results/stability_check.csv and a printed summary.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import gym
import numpy as np
import pandas as pd
import torch
from scipy import stats

from test_script import QNetwork

torch.set_num_threads(1)
X_MAX, THETA_MAX = 2.4, 12 * 2 * np.pi / 360
LENGTHS = np.linspace(0.4, 1.8, 30)
EPISODES, STEPS = 3, 500
GROUPS = {
    "uniform": "weights/uniform_rerun/uniform_seed*.pth",
    "margin": "weights/margin_seed*.pth",
    "margin_pbrs": "weights/margin_pbrs_seed*.pth",
    "margin_linear": "weights/margin_linear_seed*.pth",
}


def margin_reward(x, theta):
    a, p = min(abs(theta) / THETA_MAX, 1.0), min(abs(x) / X_MAX, 1.0)
    return 1.0 - 0.5 * a ** 2 - 0.25 * p ** 2


def network_stats(path):
    net = QNetwork(4, 2)
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    xs, ths, rs = [], [], []
    for i, length in enumerate(LENGTHS):
        for e in range(EPISODES):
            env = gym.make("CartPole-v1")
            env.unwrapped.length = length
            s = env.reset(seed=10_000 + 10 * i + e)[0]          # seeded -> reproducible
            for _ in range(STEPS):
                with torch.no_grad():
                    a = net(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).argmax().item()
                s, _, terminated, _, _ = env.step(a)
                xs.append(abs(s[0]) / X_MAX)
                ths.append(abs(s[2]) / THETA_MAX)
                rs.append(margin_reward(s[0], s[2]))
                if terminated:
                    break
    return np.mean(xs), np.mean(ths), np.mean(rs)


rows = []
for group, pattern in GROUPS.items():
    for path in sorted(glob.glob(pattern)):
        x, th, r = network_stats(path)
        rows.append(dict(group=group, network=os.path.basename(path), x=x, theta=th, shaped_reward=r))
        print(f"{group:13s} {os.path.basename(path):24s} |x|/xmax {x:.3f}  |th|/thmax {th:.3f}  shaped r {r:.4f}", flush=True)
df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/stability_check.csv", index=False)

print("\nGroup medians (one value per network):")
print(df.groupby("group")[["x", "theta", "shaped_reward"]].median().round(3))
print("\nMann-Whitney U vs uniform (two-sided, uncorrected, exploratory):")
for g in ["margin", "margin_pbrs", "margin_linear"]:
    for col in ["x", "theta"]:
        a, b = df[df.group == g][col], df[df.group == "uniform"][col]
        if len(a) and len(b):
            print(f"  {g:13s} {col:5s}: p = {stats.mannwhitneyu(a, b).pvalue:.3f}")
