"""
evaluate.py - score a saved Q-network with the course test logic.

Two modes (see --cap):
  * --cap 0      : calls the provided test_pole_length() from test_script.py
                   UNCHANGED. Strong agents can survive for a very long time
                   (the loop only stops when the pole falls), so this can be slow.
  * --cap N (default 5000): line-by-line copy of the same logic with ONE addition,
                   the episode is cut off after N steps. Use the same cap for all
                   conditions and state it in the paper.

Per episode we keep the score, so the statistics script can work on
per-seed means and on the survival fractions.

Usage:
    python evaluate.py weights/model.pth --name uniform_seed0 --seed 0
    python evaluate.py --random --name untrained
Outputs (results/):
    <name>_scores.csv    one row per (length, run)
    <name>_summary.csv   mean/std per length
    <name>_metrics.json  overall mean, frac>500, frac>1000, mean of worst 5 lengths
    <name>_bar.png       bar plot, overall average in the title
"""
import argparse
import json
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Provided by the course - do not modify test_script.py
from test_script import QNetwork, test_pole_length

POLE_LENGTHS = np.linspace(0.4, 1.8, 30)  # fixed by the assignment
RUNS_PER_LENGTH = 10                      # fixed by the assignment


def test_pole_length_capped(env, q_network, cap):
    """Same logic as test_script.test_pole_length, plus an episode cap."""
    wind = 25
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0
    while not done and total_reward < cap:
        action = q_network(state).argmax().item()
        next_state, reward, done, _, __ = env.step(action)  # truncated ignored, as in the test
        state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        total_reward += reward
        if total_reward >= 500 and total_reward <= 1000:
            if total_reward % wind == 0:
                env.unwrapped.force_mag = 75
        if total_reward > 1000:
            env.unwrapped.force_mag = 25 + (0.01 * total_reward)
    return total_reward


def load_network(path, state_dim, action_dim):
    """Load trained weights, or a randomly initialised net if path is None."""
    net = QNetwork(state_dim, action_dim)
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


def evaluate(path, cap, seed=None):
    """Return a DataFrame with one row per episode: length, run, score."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    rows = []
    with torch.no_grad():
        for length in POLE_LENGTHS:
            for run in range(RUNS_PER_LENGTH):
                env = gym.make("CartPole-v1")
                env.unwrapped.length = length  # same call as the test script
                net = load_network(path, env.observation_space.shape[0], env.action_space.n)
                score = test_pole_length(env, net) if cap == 0 else test_pole_length_capped(env, net, cap)
                rows.append({"length": round(float(length), 2), "run": run, "score": score})
                env.close()
            print(f"length {length:.2f} done", flush=True)
    return pd.DataFrame(rows)


def metrics(df):
    per_len = df.groupby("length")["score"].mean()
    return {
        "overall_mean": float(per_len.mean()),                  # one number per seed
        "frac_over_500": float((df["score"] > 500).mean()),
        "frac_over_1000": float((df["score"] > 1000).mean()),
        "worst5_mean": float(per_len.nsmallest(5).mean()),
    }


def bar_plot(df, overall, name, out_png):
    s = df.groupby("length")["score"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(s)), s["mean"], yerr=s["std"], capsize=5, alpha=0.7)
    plt.xticks(range(len(s)), s["length"], rotation=45)
    plt.xlabel("Pole length")
    plt.ylabel("Episode length")
    plt.title(f"{name}: average score over all pole lengths = {round(overall, 0)}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="?", help="path to a .pth file")
    ap.add_argument("--random", action="store_true", help="evaluate an untrained network")
    ap.add_argument("--name", required=True, help="label used in output files")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cap", type=int, default=5000,
                    help="max steps per episode; 0 = use the unmodified course function")
    args = ap.parse_args()
    if not args.random and not args.weights:
        ap.error("give a weights file or use --random")

    os.makedirs("results", exist_ok=True)
    df = evaluate(None if args.random else args.weights, args.cap, args.seed)
    m = metrics(df)
    m["cap"] = args.cap

    df.to_csv(f"results/{args.name}_scores.csv", index=False)
    s = bar_plot(df, m["overall_mean"], args.name, f"results/{args.name}_bar.png")
    s.to_csv(f"results/{args.name}_summary.csv", index=False)
    with open(f"results/{args.name}_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()