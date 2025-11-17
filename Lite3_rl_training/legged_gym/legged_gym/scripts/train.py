import os
currentdir = os.path.dirname(os.path.abspath(__file__))
legged_gym_dir = os.path.dirname(os.path.dirname(currentdir))
isaacgym_dir = os.path.join(os.path.dirname(legged_gym_dir), "isaacgym/python")
rsl_rl_dir = os.path.join(os.path.dirname(legged_gym_dir), "rsl_rl")
os.sys.path.insert(0, legged_gym_dir)
os.sys.path.insert(0, isaacgym_dir)
os.sys.path.insert(0, rsl_rl_dir)
import numpy as np
import json
from datetime import datetime
import isaacgym
import shutil
from legged_gym.envs import *
from legged_gym.utils import get_args, Logger, register
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import class_to_dict


def train(args):
    register(args.task, task_registry)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for training
    env_cfg.commands.fixed_commands = None
    if args.near_goal_init_prob is not None:
        env_cfg.init_state.near_goal_init_prob = min(
            max(args.near_goal_init_prob, 0.0), 1.0)

    # prepare environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load model
    if args.load_run:
        train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg,
        enable_summary_writer=True)

    # if this is a fresh run and a previous run folder exists, wipe it
    if not train_cfg.runner.resume and os.path.isdir(ppo_runner.log_dir):
        shutil.rmtree(ppo_runner.log_dir)

    # record configs as log files
    os.makedirs(ppo_runner.log_dir, exist_ok=True)
    # drop a helper script to replay this run easily
    run_dir_abs = os.path.abspath(ppo_runner.log_dir)
    run_play_path = os.path.join(ppo_runner.log_dir, "run_play.sh")
    run_resume_path = os.path.join(ppo_runner.log_dir, "run_resume.sh")
    exp_name = os.path.basename(os.path.dirname(run_dir_abs))
    run_name = os.path.basename(run_dir_abs)

    play_cmd = f"""#!/usr/bin/env bash
set -euo pipefail
# Resolve repo/log roots relative to this run directory
THIS_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
LOGS_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LOGS_ROOT/.." && pwd)"

HEADLESS_FLAG=""
CHECKPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless) HEADLESS_FLAG="--headless"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
CKPT_FLAG=""
if [[ -n "$CHECKPOINT" ]]; then
  CKPT_FLAG="--checkpoint $CHECKPOINT"
fi

python "$REPO_ROOT/legged_gym/scripts/play.py" \\
  --task {args.task} \\
  --experiment_name "{exp_name}" \\
  --run_name "{run_name}" \\
  --load_run "{run_name}" \\
  $CKPT_FLAG \\
  $HEADLESS_FLAG
"""
    resume_cmd = f"""#!/usr/bin/env bash
set -euo pipefail
# Resolve repo/log roots relative to this run directory
THIS_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
LOGS_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LOGS_ROOT/.." && pwd)"

CHECKPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
CKPT_FLAG=""
if [[ -n "$CHECKPOINT" ]]; then
  CKPT_FLAG="--checkpoint $CHECKPOINT"
fi

# Always headless for resume scripts
python "$REPO_ROOT/legged_gym/scripts/train.py" \\
  --task {args.task} \\
  --resume \\
  --experiment_name "{exp_name}" \\
  --run_name "{run_name}" \\
  --load_run "{run_name}" \\
  $CKPT_FLAG \\
  --headless
"""
    for path, content in [(run_play_path, play_cmd), (run_resume_path, resume_cmd)]:
        with open(path, "w") as fp:
            fp.write(content)
        os.chmod(path, 0o755)
    with open(os.path.join(ppo_runner.log_dir, 'env_cfg.json'), 'w') as fp:
        json.dump(class_to_dict(env_cfg), fp)
    with open(os.path.join(ppo_runner.log_dir, 'train_cfg.json'), 'w') as fp:
        json.dump(class_to_dict(train_cfg), fp)

    # train ppo policy
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    args.save_rewards = True
    train(args)
