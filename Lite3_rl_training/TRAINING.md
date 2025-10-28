# Lite3 Two-Leg Stand Training Cheat Sheet

This repo already exposes convenient CLI flags for fast curriculum iteration.  
Use the commands below as a quick reference while you and other contributors tune phases or resume from checkpoints.

## Fresh Training Run

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task lite3_two_leg_stand \
  --run_name baseline_run \
  --experiment_name two_leg_stand_exp
```

* `--run_name` is the folder name under `legged_gym/logs/two_leg_stand`.  
* `--experiment_name` lets you group multiple runs (optional).

## Resume From A Checkpoint

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task lite3_two_leg_stand \
  --resume \
  --load_run two_leg_stand/baseline_run \
  --checkpoint model_0500.pt \
  --run_name phase1_tuning \
  --max_iterations 2000
```

Key flags:

- `--resume` – tells the runner to load model + optimizer + iteration counter.
- `--load_run <path-or--1>` – pick the run folder (`-1` means “latest”).
- `--checkpoint model_XXXX.pt` – exact file inside the run folder (`-1` for most recent).
- `--run_name` – new log directory for this tuning session.
- `--max_iterations` – optional cap if you only want to run through the next phase.

Curriculum progress is stored in the checkpoint, so resuming at iteration N drops you straight into the next phase using your updated reward scales.

## Other Useful Flags

- `--num_envs <N>` – override the environment count.
- `--seed <int>` – fix the RNG seed.
- `--save_rewards` – export per-term reward CSVs to the run directory.
- `--headless` – disable rendering (default is headless already).

## Workflow Tip

1. Train until the run reaches the phase you care about and a checkpoint is saved (e.g. `model_0500.pt`).  
2. Adjust the reward scales or curriculum thresholds in `two_leg_stand_config.py`.  
3. Resume from that checkpoint with a new `--run_name` to validate the change quickly.  
4. Rinse and repeat for the next phase.

That’s it—feel free to expand this sheet as you add more automation to the training loop.***
