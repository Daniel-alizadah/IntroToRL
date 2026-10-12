# IntroToRL — Team 6, Assignment 3

Deep Q-learning generalist for **CartPole-v1 with varying pole lengths** (Introduction to
Reinforcement Learning, VU Amsterdam, 2026). All agents use the prescribed network from
`test_script.py` (4 → 128 → 128 → 2). During training **only the pole length is changed**
(`env.unwrapped.length`). The course test (`test_script.py`) is never modified and never used
for training or for choosing checkpoints.

## Setup

```bash
python -m venv venv && source venv/bin/activate      # Python 3.10–3.12
pip install -r requirements.txt
```

## Conditions (`train.py --condition ...`)

| Condition | Role | Pole length per episode | Training change |
|---|---|---|---|
| `fixed` | baseline | always 0.5 | – |
| `uniform` | control | uniform in [0.4, 1.8] | – |
| `curriculum` | **S1** (Daniel) | rises linearly 0.4 → 1.8 over training | – |
| `strat_curriculum` | **S2** (Zelal) | as S1 | length-stratified replay buffer (`stratified_buffer.py`) |
| `strat_uniform` | S2 diagnostic | as `uniform` | length-stratified replay buffer |
| `margin` | **S3** (Öykü) | as `uniform` | stability-margin reward stored in replay: `1 − 0.5(θ/θmax)² − 0.25(x/xmax)²` |
| `margin_pbrs` | S3 ablation | as `uniform` | potential-based version of the same penalty (Ng et al., 1999) |
| `margin_linear` | S3 ablation | as `uniform` | linear penalties, weights 0.5 / 0.5 |

Shared settings (constants at the top of `train.py`): 300k environment steps, one update per
step after 1k warm-up steps, batch 64, Adam lr 1e-4, γ = 0.99, MSE TD loss, replay memory 20k,
target network copied every 1k steps, ε from 1.0 to 0.05 over the first 50% of steps,
bootstrapping through time-limit truncation (not through real failures), validation every 10k
steps under nominal dynamics (5 lengths × 3 seeded episodes, 10k-step cap), and the best
validation checkpoint is saved. Seeds 0–7 for every condition.

## Reproduce

```bash
# 1. train one condition/seed  -> weights/<condition>_seed<k>.pth, logs/<condition>_seed<k>*.csv
python train.py --condition margin --seed 0

# 2. evaluate with the course test protocol (30 lengths x 10 episodes, episodes capped at 20000 steps)
python evaluate.py weights/margin_seed0.pth --name margin_seed0 --cap 20000   # -> results/

#    or train + evaluate seeds 0-7 of one or more conditions (3 in parallel)
bash run_cond.sh margin

# 3. choose the submitted seed per strategy from the VALIDATION logs only
python select_checkpoint.py curriculum strat_curriculum margin

# 4. statistics (Kruskal-Wallis, Mann-Whitney U with Holm correction, Cliff's delta, bootstrap CIs) and figures
python analyze.py --conditions fixed uniform curriculum strat_curriculum margin --pairs ... --cap 20000
python make_plots.py

# 5. S3 mechanism checks (run from the repo root)
python analysis/compare_conditions.py      # margin variants vs uniform re-run vs curriculum
python analysis/stability_check.py         # seeded: cart drift, pole angle, size of the shaped reward
```

`evaluate.py --cap 0` runs the unmodified `test_pole_length()` from `test_script.py` without a cap.

## Submitted networks (`submission/`)

| Strategy | File | Chosen by |
|---|---|---|
| S1 curriculum | `S1_curriculum_seed6.pth` | `select_checkpoint.py` |
| S2 stratified replay + curriculum | `S2_stratified_seed4.pth` | `select_checkpoint.py` |
| S3 stability-margin reward | `S3_margin_seed5.pth` | `select_checkpoint.py` |

Selection rule (fixed before looking at test scores): highest best-validation score; ties → the
most validation rounds at the 10k cap; then the lowest seed.

To run the official test script on a submitted network, set `trained_nn` in `test_script.py`
to e.g. `"../submission/S3_margin_seed5.pth"` (it loads from `weights/`), or use
`python evaluate.py submission/S3_margin_seed5.pth --name S3_official --cap 0`.

## Repository layout

```
train.py               DQN training, all conditions (--condition)
stratified_buffer.py   length-stratified replay buffer (S2)
evaluate.py            course test protocol, saves per-episode scores (results/)
select_checkpoint.py   picks the submitted seed from validation logs
analyze.py, make_plots.py   statistics and figures for the paper
run_cond.sh, eval_missing.sh   batch helpers (train/evaluate seeds 0-7)
analysis/              S3 analyses (compare_conditions.py, margin_analysis.py, stability_check.py)
test_script.py         provided by the course, unchanged
weights/               trained networks (<condition>_seed<k>.pth); weights/uniform_rerun/ = uniform control re-run used for S3
results/               test scores (<condition>_seed<k>_scores.csv, cap 20000); results/uniform_rerun/
logs/                  per-episode training logs and validation logs
submission/            the three submitted networks
```

## Notes

* Runs were spread over the three team members' laptops (same code and seeds). Results can
  differ between machines/library versions: the uniform control gave 10.4k on one laptop and
  12.6k on another, within seed noise. S3 is therefore compared with a uniform re-run on the same
  machine (`results/uniform_rerun/`).
* The 20k evaluation cap is ours, not the course's; it is the same for every condition.
