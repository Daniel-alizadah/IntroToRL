"""Length-stratified replay buffer.

Idea: a plain FIFO replay buffer forgets old pole lengths once the curriculum
(or just chance) moves training on to other lengths. This buffer keeps one small
FIFO queue per pole-length bin and samples the bins evenly, so every length
range seen so far stays represented in each minibatch.

Total capacity is the same as the plain buffer (capacity // n_bins per bin), so
comparisons against the FIFO buffer are not confounded by memory size.

A transition is any tuple whose LAST element is the pole length (this matches
train.py: (state, action, reward, next_state, terminated, pole_length)).
"""
import random
from collections import deque

import numpy as np


class StratifiedReplay:
    def __init__(self, capacity, min_length=0.4, max_length=1.8, n_bins=8):
        self.n_bins = n_bins
        self.edges = np.linspace(min_length, max_length, n_bins + 1)
        per_bin = max(1, capacity // n_bins)
        self.bins = [deque(maxlen=per_bin) for _ in range(n_bins)]

    def bin_index(self, length):
        # inner edges only, so lengths at/beyond the ends fall in the first/last bin
        return int(np.clip(np.searchsorted(self.edges[1:-1], length, side="right"),
                           0, self.n_bins - 1))

    def append(self, transition):
        self.bins[self.bin_index(transition[-1])].append(transition)

    def __len__(self):
        return sum(len(b) for b in self.bins)

    def sample(self, batch_size):
        """Pick a bin uniformly among non-empty bins for every sample, then a
        transition uniformly inside that bin (sampling with replacement across
        draws is fine for DQN minibatches)."""
        filled = [b for b in self.bins if len(b) > 0]
        batch = []
        for _ in range(batch_size):
            bucket = random.choice(filled)
            batch.append(bucket[random.randrange(len(bucket))])
        return batch

    def counts(self):
        return [len(b) for b in self.bins]
