import random
from collections import deque
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TOTAL_STEPS = 300000
SEED = 0
CPU_THREADS = 1
CONDITION = "uniform"

MIN_LENGTH = 0.4
MAX_LENGTH = 1.8
FIXED_LENGTH = 0.5

LEARNING_RATE = 0.0001
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 20000
LEARNING_STARTS = 1000
TRAIN_EVERY = 1
TARGET_UPDATE_EVERY = 1000

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_FRACTION = 0.5

VALIDATE_EVERY = 10000
VALIDATION_LENGTHS = [0.45, 0.75, 1.05, 1.35, 1.75]
VALIDATION_EPISODES = 3
VALIDATION_MAX_STEPS = 10000


# Match the network in test_script.py.
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        values = torch.relu(self.fc1(state))
        values = torch.relu(self.fc2(values))
        return self.fc3(values)


def choose_pole_length(condition, step=0, total_steps=TOTAL_STEPS):
    if condition == "fixed":
        return FIXED_LENGTH
    if condition == "curriculum":
        progress = step / max(1, total_steps)
        progress = min(progress, 1.0)
        return MIN_LENGTH + progress * (MAX_LENGTH - MIN_LENGTH)
    return float(np.random.uniform(MIN_LENGTH, MAX_LENGTH))


def calculate_epsilon(step, total_steps):
    decay_steps = max(1, int(total_steps * EPSILON_DECAY_FRACTION))
    progress = min(step / decay_steps, 1.0)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def choose_action(state, model, epsilon):
    if random.random() < epsilon:
        return random.randrange(2)

    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0)
        action_values = model(state_tensor)
        return action_values.argmax().item()


def learn_from_memory(model, target_model, optimizer, memory):
    batch = random.sample(list(memory), BATCH_SIZE)

    states = []
    actions = []
    rewards = []
    next_states = []
    terminated_flags = []

    for transition in batch:
        state, action, reward, next_state, terminated, pole_length = transition
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        terminated_flags.append(terminated)

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    terminated_flags = torch.tensor(terminated_flags, dtype=torch.float32)

    all_action_values = model(states)
    predicted_values = all_action_values.gather(1, actions.unsqueeze(1))
    predicted_values = predicted_values.squeeze(1)

    with torch.no_grad():
        next_action_values = target_model(next_states)
        best_next_values = next_action_values.max(dim=1).values

        # Bootstrap through time limits, but not actual termination.
        can_continue = 1 - terminated_flags
        target_values = rewards + GAMMA * best_next_values * can_continue

    loss = F.mse_loss(predicted_values, target_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def validate_model(model):
    env = gym.make("CartPole-v1", max_episode_steps=VALIDATION_MAX_STEPS)
    total_score = 0
    model.eval()

    for length_index, length in enumerate(VALIDATION_LENGTHS):
        env.unwrapped.length = length
        for episode in range(VALIDATION_EPISODES):
            validation_seed = 900000 + length_index * VALIDATION_EPISODES + episode
            state, _ = env.reset(seed=validation_seed)

            for step in range(VALIDATION_MAX_STEPS):
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = model(state_tensor).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
                total_score += reward
                if terminated or truncated:
                    break

    env.close()
    model.train()
    number_of_episodes = len(VALIDATION_LENGTHS) * VALIDATION_EPISODES
    return total_score / number_of_episodes


def train(total_steps=TOTAL_STEPS, seed=SEED, condition=CONDITION, output_folder="weights"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(CPU_THREADS)

    env = gym.make("CartPole-v1")
    model = QNetwork(4, 2)
    target_model = QNetwork(4, 2)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=MEMORY_SIZE)

    length = choose_pole_length(condition, 0, total_steps)
    env.unwrapped.length = length
    state, _ = env.reset(seed=seed)
    print(f"Training for {total_steps} steps...")

    Path("logs").mkdir(exist_ok=True)
    log = open(f"logs/{condition}_seed{seed}.csv", "w")
    log.write("step,length,episode_return\n")
    episode_return = 0

    weight_file = Path(output_folder) / f"{condition}_seed{seed}.pth"
    weight_file.parent.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_step = 0
    validation_log = open(f"logs/{condition}_seed{seed}_validation.csv", "w")
    validation_log.write("step,validation_score,saved_best\n")

    for step in range(1, total_steps + 1):
        epsilon = calculate_epsilon(step - 1, total_steps)
        action = choose_action(state, model, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward

        # Store pole length for stratified replay.
        transition = (state.copy(), action, reward,
                      next_state.copy(), terminated, length)
        memory.append(transition)
        state = next_state

        warmup_finished = step >= LEARNING_STARTS
        enough_samples = len(memory) >= BATCH_SIZE
        time_to_learn = step % TRAIN_EVERY == 0
        if warmup_finished and enough_samples and time_to_learn:
            learn_from_memory(model, target_model, optimizer, memory)

        if step % TARGET_UPDATE_EVERY == 0:
            target_model.load_state_dict(model.state_dict())

        if terminated or truncated:
            log.write(f"{step},{length},{episode_return}\n")
            episode_return = 0

        if step % VALIDATE_EVERY == 0 or step == total_steps:
            score = validate_model(model)
            improved = score > best_score
            if improved:
                best_score = score
                best_step = step
                torch.save(model.state_dict(), weight_file)
            validation_log.write(f"{step},{score},{improved}\n")
            validation_log.flush()
            print(f"Step {step}: validation {score:.1f}, best {best_score:.1f}")

        if (terminated or truncated) and step < total_steps:
            length = choose_pole_length(condition, step, total_steps)
            env.unwrapped.length = length
            state, _ = env.reset()

    log.close()
    validation_log.close()
    env.close()

    print(f"Best model: step {best_step}, validation {best_score:.1f}, saved to {weight_file}")
    model.load_state_dict(torch.load(weight_file, weights_only=True))
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["uniform", "fixed", "curriculum"], default=CONDITION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()
    train(args.steps, args.seed, args.condition)
