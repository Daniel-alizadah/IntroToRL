# IntroToRL

Team 6: Daniel, Zelal and Öykü. Assignment 3 for Introduction to Reinforcement Learning at VU Amsterdam.

We train a DQN agent to balance CartPole across pole lengths from 0.4 to 1.8. We compare a curriculum, stratified replay and reward shaping with fixed-length and uniform-random training.

## Setup

Use Python 3.10–3.12. Create a virtual environment:

```bash
python -m venv venv
```

Activate it with `source venv/bin/activate` on macOS/Linux, or `.\venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the packages:

```bash
pip install -r requirements.txt
```

## Train and evaluate

Run commands from the repository folder. For example:

```bash
python train.py --condition margin --seed 0
python evaluate.py weights/margin_seed0.pth --name margin_seed0 --cap 20000
```

Training saves the model in `weights/` and logs in `logs/`. Evaluation saves episode scores, a summary, metrics and a bar plot in `results/`.

The available conditions are:

| Condition | Description |
|---|---|
| `fixed` | Baseline: pole length 0.5 |
| `uniform` | Control: random length from 0.4 to 1.8 each episode |
| `curriculum` | S1: length increases linearly from 0.4 to 1.8 |
| `strat_curriculum` | S2: the same curriculum with length-stratified replay |
| `margin` | S3: squared angle and position penalties in the training reward |
| `strat_uniform` | Replay-buffer diagnostic with uniform lengths |
| `margin_pbrs` | Potential-based reward-shaping ablation |
| `margin_linear` | Reward-shaping ablation with linear penalties |

We use seeds 0–7 for each condition. The Bash helpers run three jobs at a time:

```bash
bash run_cond.sh margin
bash eval_missing.sh margin
```

The evaluation uses 30 pole lengths and ten episodes per length. We cap episodes at 20,000 steps for every condition. This cap is our addition to the course test. To use the original course function without a cap:

```bash
python evaluate.py submission/S3_margin_seed5.pth --name S3_official --cap 0
```

`test_script.py` is the original course file. We do not use its scores for training or checkpoint selection.

## Analysis

Put the score CSVs in `results/` with names such as `curriculum_seed0_scores.csv`. Once all the requested conditions are available:

```bash
python analyze.py --conditions fixed uniform curriculum strat_curriculum margin --pairs uniform:fixed curriculum:uniform strat_curriculum:curriculum margin:uniform --cap 20000
python make_plots.py
```

`analyze.py` compares training seeds using Kruskal-Wallis and Mann-Whitney tests, Holm correction, Cliff's delta and bootstrap confidence intervals. `make_plots.py` also expects the second laptop's uniform results named `uniform_oyku_seed<k>_scores.csv`.

The S3 checks use the margin models and the uniform control rerun:

```bash
python analysis/compare_conditions.py
python analysis/stability_check.py
```

The first command also needs the curriculum CSVs. The stability check seeds the environment resets and measures cart position, pole angle and shaped reward during the first 500 nominal steps. Its tests are exploratory and uncorrected; measurements stop if an episode fails.

## Submitted models

The three selected models are in `submission/`:

- S1: `S1_curriculum_seed6.pth`
- S2: `S2_stratified_seed4.pth`
- S3: `S3_margin_seed5.pth`

We select models from validation logs: highest best-validation score, then most validation rounds at the 10,000-step cap, then lowest seed.

```bash
python select_checkpoint.py curriculum strat_curriculum margin
```

This command needs the validation logs for each condition in `logs/`.

## Shared training settings

All conditions use the course network (4 → 128 → 128 → 2) and 300,000 environment steps. The settings are in `train.py`: Adam learning rate 1e-4, gamma 0.99, batch size 64, MSE loss, replay capacity 20,000, and a target-network update every 1,000 steps. Learning starts after 1,000 steps and runs every step. Epsilon decreases from 1.0 to 0.05 during the first half of training. Time-limit truncations allow bootstrapping; actual failures do not.

Validation runs every 10,000 steps on five lengths, with three seeded episodes per length and a 10,000-step cap. The best validation checkpoint is saved. Only pole length changes during training; reward-shaping conditions change the reward stored in replay.

Experiments ran on different laptops, so scores can vary across machines and library versions. For S3, we use a uniform control rerun on the same laptop, stored under `weights/uniform_rerun/` and `results/uniform_rerun/`.
