"""margin_analysis.py - does a network keep a 'stability margin'?

For each weight file, runs the first 500 steps (nominal phase of the course test) on the
30 test lengths and reports how close the cart and pole stay to their failure limits:
mean |x| / 2.4 and mean |theta| / 12 deg (lower = more margin). Then it continues each
episode with the course test's force schedule and reports where episodes end.

Usage (from the repo root): python analysis/margin_analysis.py weights/margin_seed*.pth weights/uniform_seed*.pth
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for test_script
import gym
import numpy as np
import torch
from test_script import QNetwork

torch.set_num_threads(1)
X_MAX, THETA_MAX = 2.4, 12 * 2 * np.pi / 360
LENGTHS = np.linspace(0.4, 1.8, 30)


def run(net, length, cap=3000):
    env = gym.make("CartPole-v1")
    env.unwrapped.length = length
    s = env.reset()[0]
    xs, ths, total, done = [], [], 0, False
    while not done and total < cap:
        with torch.no_grad():
            a = net(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).argmax().item()
        s, r, done, _, _ = env.step(a)
        total += r
        if total <= 500:
            xs.append(abs(s[0]) / X_MAX); ths.append(abs(s[2]) / THETA_MAX)
        if 500 <= total <= 1000 and total % 25 == 0:
            env.unwrapped.force_mag = 75
        if total > 1000:
            env.unwrapped.force_mag = 25 + 0.01 * total
    return np.mean(xs), np.mean(ths), total


for path in sys.argv[1:]:
    net = QNetwork(4, 2)
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    res = np.array([run(net, L) for L in LENGTHS for _ in range(3)])
    print(f"{path:40s} mean|x|/xmax {res[:,0].mean():.3f}  mean|theta|/thmax {res[:,1].mean():.3f}  "
          f"died at force onset (500-525) {np.mean((res[:,2] >= 500) & (res[:,2] <= 525)):.2f}  "
          f"reached 3000 {np.mean(res[:,2] >= 3000):.2f}", flush=True)
