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
- TwoLegStandDeployAligned-Deeprobotics-Lite3-v0
- Legacy aliases: lite3_two_leg_stand, lite3_two_leg_stand_still, lite3_two_leg_stand_still_v2, lite3_two_leg_stand_still_safe, lite3_two_leg_stand_deploy_aligned

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
