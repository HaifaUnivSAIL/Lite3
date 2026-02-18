# Run and Play Guide (rl_training_new)

This guide shows how to run a training experiment and play a trained policy. It covers both stacks in this repo:

- Isaac Lab manager-based stack (scripts/reinforcement_learning/rsl_rl). Recommended.
- Legacy legged_gym stack (legged_gym/scripts).

Use the stack that matches your environment (Isaac Lab vs legacy Isaac Gym).

## Two-leg stand task IDs

Isaac Lab stack (ManagerBasedRLEnv, registered in rl_training tasks):
- TwoLegStand-Deeprobotics-Lite3-v0
- TwoLegStandStill-Deeprobotics-Lite3-v0
- TwoLegStandStillV2-Deeprobotics-Lite3-v0
- TwoLegStandSafe-Deeprobotics-Lite3-v0
- TwoLegStandRobust-Deeprobotics-Lite3-v0
- TwoLegStandDeployAligned-Deeprobotics-Lite3-v0
- TwoLegStandDeployR1-Deeprobotics-Lite3-v0
- Legacy aliases: lite3_two_leg_stand, lite3_two_leg_stand_still, lite3_two_leg_stand_still_v2, lite3_two_leg_stand_still_safe, lite3_two_leg_stand_robust, lite3_two_leg_stand_deploy_aligned
- Legacy alias (deploy/r1): lite3_two_leg_stand_deploy_r1
- Compatibility alias: two_leg_stand_robust

Legacy legged_gym stack (registered in legged_gym helpers):
- lite3_two_leg_stand
- lite3_two_leg_stand_still
- lite3_two_leg_stand_still_v2
- lite3_two_leg_stand_still_safe
- lite3_two_leg_stand_deploy_aligned

## Stack A: Isaac Lab manager-based (recommended)

All commands below assume:

```bash
cd /home/sail/Lite3/rl_training_new
```

### Train (example)

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStand-Deeprobotics-Lite3-v0 \
  --headless
```

### Train (deploy/r1 curriculum, matches Lite3_rl_training logs/deploy/r1)

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStandDeployR1-Deeprobotics-Lite3-v0 \
  --headless
```

### Train (robust randomization + perturbation curriculum)

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStandRobust-Deeprobotics-Lite3-v0 \
  --run_name robust_rand_v1 \
  --headless
```

### Resume (deploy/r1 from a specific checkpoint)

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStandDeployR1-Deeprobotics-Lite3-v0 \
  --headless \
  --resume \
  --load_run r1 \
  --checkpoint model_14500.pt
```

### Naming and resume flags (important)

These are the flags that control experiment/run naming and loading:

- --run_name <name>  
  Appends a suffix to the timestamped run directory.
- --resume  
  Resume training from an existing run.
- --load_run <run_folder>  
  Which run folder to load when resuming or playing. This is relative to
  logs/rsl_rl/<experiment_name>/ unless you pass a full checkpoint path.
- --checkpoint <model_XXXX.pt>  
  Which checkpoint file to load inside the run directory.
- --agent <entry_point>  
  Which agent config to use (defaults to rsl_rl_cfg_entry_point).

Notes on experiment_name:

- The experiment folder name is taken from the agent config (for example, two_leg_stand).
- The CLI flag --experiment_name exists in the parser but is not applied in this script.
  To change experiment_name, edit the agent config or use a Hydra override if needed.

### Where logs and checkpoints go

- logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
  - model_*.pt checkpoints
  - params/env.yaml and params/agent.yaml
  - videos/ (if --video was used)

### Play a trained policy (example)

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=TwoLegStand-Deeprobotics-Lite3-v0 \
  --load_run <run_folder> \
  --checkpoint <model_XXXX.pt>
```

### Play (deploy/r1 example)

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=TwoLegStandDeployR1-Deeprobotics-Lite3-v0 \
  --load_run r1 \
  --checkpoint model_best.pt
```

### Play (robust task example)

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=TwoLegStandRobust-Deeprobotics-Lite3-v0 \
  --load_run <timestamp>_robust_rand_v1 \
  --checkpoint model_best.pt
```

Helpful play flags:

- --video --video_length 400  
  Record a short clip into logs/rsl_rl/<experiment_name>/<run>/videos/play/
- --real-time  
  Try to keep real-time pace.
- --keyboard  
  Control a single robot with the keyboard (forces num_envs = 1).
- --num_envs <N>  
  Number of environments during playback.

Tip: If you want to load by absolute checkpoint path, pass:

```
--checkpoint /abs/path/to/model_XXXX.pt
```

### Quick flag discovery

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --help
python scripts/reinforcement_learning/rsl_rl/play.py --help
```

### Debug dumps (training)

Enable per-iteration debug dumps for rewards and key signals:

- LITE3_DEBUG_TRAIN_DUMPS: number of iterations to dump (0 disables)
- LITE3_DEBUG_DUMP_EVERY: dump every N iterations (default 1)
- LITE3_DEBUG_DUMP_DIR: override output directory (default: logs/.../debug_dumps)
- LITE3_DEBUG_DUMP_FULL: if set, also writes a .npz with tensors from the last step

Example (dump 5 iterations, every 10 iters, plus full tensors):

```bash
LITE3_DEBUG_TRAIN_DUMPS=5 \
LITE3_DEBUG_DUMP_EVERY=10 \
LITE3_DEBUG_DUMP_FULL=1 \
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStandDeployR1-Deeprobotics-Lite3-v0 \
  --headless
```

Each dump writes:

- `iter_XXXXXX.json`: scalar summary (losses, phase, reward means, weights)
- `iter_XXXXXX.npz` (optional): obs, privileged_obs, obs_history, actions, rewards, dones

### History Semantics (Default and Debug)

By default, observation history is reset on episode resets/done events. This is the required production behavior for train/play parity with deploy.

- Default (production): keep `LITE3_UNREALISTIC_HISTORY_FEED` unset (or `0`).
- Debug-only legacy mode: set `LITE3_UNREALISTIC_HISTORY_FEED=1` to preserve history across done resets.

Example (debug only):

```bash
LITE3_UNREALISTIC_HISTORY_FEED=1 \
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=TwoLegStand-Deeprobotics-Lite3-v0 \
  --load_run <run_folder> \
  --checkpoint <model_XXXX.pt>
```

## Stack B: Legacy legged_gym

All commands below assume:

```bash
cd /home/sail/Lite3/rl_training_new
```

### Train (example)

```bash
python legged_gym/scripts/train.py \
  --task=lite3_two_leg_stand \
  --headless
```

### Naming and resume flags (important)

- --experiment_name <name>  
  Overrides the experiment folder name.
- --run_name <name>  
  Overrides the run folder name.
- --resume  
  Resume training from an existing run.
- --load_run <run_folder>  
  Which run folder to load when resuming or playing.
- --checkpoint <model_XXXX.pt>  
  Which checkpoint file to load inside the run directory.

Legacy logs are stored at:

- legged_gym/logs/<experiment_name>/<run_name>/

### Play a trained policy (example)

```bash
python legged_gym/scripts/play.py \
  --task=lite3_two_leg_stand \
  --experiment_name two_leg_stand \
  --load_run <run_name> \
  --checkpoint <model_XXXX.pt>
```

Note: Each legacy run folder includes a helper script:

- legged_gym/logs/<experiment_name>/<run_name>/run_play.sh

### Quick flag discovery

```bash
python legged_gym/scripts/train.py --help
python legged_gym/scripts/play.py --help
```

## Viewing results

TensorBoard (both stacks):

```bash
tensorboard --logdir=logs
```

For legacy logs, you can also point at:

```
tensorboard --logdir=legged_gym/logs
```

## Current experiment focused command

Use this exact pair to train and then play the same run (deploy/r1 task).

Train:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStandDeployR1-Deeprobotics-Lite3-v0 \
  --run_name demo_r1 \
  --headless
```

Play (same run_name):

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=TwoLegStandDeployR1-Deeprobotics-Lite3-v0 \
  --load_run <timestamp>_demo_r1 \
  --checkpoint model_best.pt
```

Notes:
- `<timestamp>_demo_r1` must match the run folder created during training, e.g. `2026-01-28_12-34-56_demo_r1`.
- The experiment folder is `logs/rsl_rl/deploy/` for this task (from the agent config).
