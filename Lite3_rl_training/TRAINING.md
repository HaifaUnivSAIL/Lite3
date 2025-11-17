# Lite3 Two-Leg Stand Training Cheat Sheet

This repo already exposes convenient CLI flags for fast curriculum iteration.  
Use the commands below as a quick reference while you and other contributors tune phases or resume from checkpoints.

## Fresh Training Run

Most experiments start from scratch with the two-leg-stand curriculum. The command below keeps things headless and logs into `legged_gym/logs/<experiment>/<run_name>`.

```bash
cd Lite3_rl_training
python legged_gym/legged_gym/scripts/train.py \
  --task lite3_two_leg_stand \
  --experiment_name two_leg_stand_exp \
  --run_name baseline_run \
  --save_rewards
```

* `--experiment_name` – folder grouping for multiple runs (optional but keeps logs tidy).
* `--run_name` – unique run directory inside the experiment.
* `--save_rewards` – dumps per-term reward CSVs for later debugging (safe to drop if you do not need them).

### Injecting Near-Goal Initializations From The Start

If you want to warm-start a locomotion experiment (e.g., `--task lite3`) with near-goal resets before the curriculum flips them on, pass a probability explicitly:

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task lite3 \
  --run_name probe_v4 \
  --near_goal_init_prob 0.2
```

The CLI flag clamps to `[0,1]` and overrides whatever is encoded in `lite3_config.py`. When omitted, the curriculum-defined probabilities (e.g., in `two_leg_stand_config.py`) take over automatically.

## Resume From A Checkpoint

Curriculum progress lives inside checkpoints, so resuming keeps you in the correct phase. Point to the run folder and checkpoint you care about, then give the resumed run a new name.

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task lite3_two_leg_stand \
  --resume \
  --load_run two_leg_stand_exp/baseline_run \
  --checkpoint model_0500.pt \
  --run_name phase1_tuning \
  --max_iterations 2000
```

Flag refresher:

| Flag | Purpose |
| --- | --- |
| `--resume` | Load network, optimizer, and iteration counters. |
| `--load_run <path or -1>` | Pick an existing run directory (`-1` = most recent). |
| `--checkpoint <file or -1>` | Exact checkpoint inside the run (`-1` = latest). |
| `--max_iterations` | Optional cap for short validation bursts. |
| `--num_envs <N>` | Override GPU load when debugging. |
| `--seed <int>` | Force deterministic replay. |
| `--headless` | Force viewer off (already the default on headless machines). |

## Playing / Visualizing A Policy

`play.py` spins up a viewer, loads your checkpoint, and rolls out trajectories. Always match the `--task` with the training run.

```bash
python legged_gym/legged_gym/scripts/play.py \
  --task lite3_two_leg_stand \
  --load_run two_leg_stand_exp/baseline_run \
  --checkpoint model_0500.pt \
  --num_envs 1 \
  --headless   # drop this flag if you want the viewer
```

Notes:

- `--num_envs` defaults to 10; set it lower if you only need a single actor in the viewer.
- `--headless` keeps the render off for remote debugging; omit it locally to see the viewer.
- `play.py` reuses the same `--near_goal_init_prob` flag if you want to force near-goal resets during eval (`--near_goal_init_prob 1` is handy for inspection).

## Workflow Tip

1. Train until the run reaches the phase you care about and a checkpoint is saved (e.g., `model_0500.pt`).  
2. Edit curriculum settings in `legged_gym/envs/base/two_leg_stand_config.py` (or `lite3_config.py` for locomotion).  
3. Resume from that checkpoint with a new `--run_name` to validate the tweak quickly.  
4. Use `play.py` with the same checkpoint to visually sanity-check the behavior or to force continuous near-goal spawning for qualitative review.

Feel free to append more “known-good” command snippets here whenever you introduce new experiments or automation helpers.***

## One-Command Replay / Resume (Generated Scripts)

Every training run now drops two helper scripts into its log directory. The “run folder” here means the path `legged_gym/logs/<experiment_name>/<run_name>` that `train.py` created (the same place `env_cfg.json` lives). Change into that folder and run:

- `./run_play.sh [--headless] [--checkpoint model_XXXX.pt]`  
  Replays the run with `play.py` using the correct `--task` and `--load_run`. Defaults to the latest checkpoint. Add `--headless` to suppress the viewer.

- `./run_resume.sh [--checkpoint model_XXXX.pt]`  
  Resumes training headlessly from the specified (or latest) checkpoint for that run. Task and run path are prefilled, so you only choose the checkpoint.

Both scripts are made executable automatically during training startup and use repo-relative paths, so they won’t hit “file not found” as long as you launch them from inside the run’s log directory.***
