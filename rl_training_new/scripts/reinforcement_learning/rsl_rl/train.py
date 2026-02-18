# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import copy
import json
import statistics

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
import textwrap

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401
from rl_training.utils.env_wrappers import RslRlCompatWrapper


def _make_world_writable(path: str, is_dir: bool = True) -> None:
    """Best-effort chmod to allow collaborative access to logs."""
    mode = 0o777 if is_dir else 0o666
    try:
        if not is_dir:
            current = os.stat(path).st_mode
            if current & 0o111:
                mode = 0o777
        os.chmod(path, mode)
    except OSError:
        pass


def _parse_int_env(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _history_mode_label() -> str:
    if _parse_bool_env("LITE3_UNREALISTIC_HISTORY_FEED", default=False):
        return "unrealistic_history_feed"
    return "default_reset_on_done"


def _configure_eval_env_cfg(eval_cfg, force_deploy_reset: bool) -> None:
    """Apply play-style deterministic settings to an eval env config."""
    eval_cfg.observations.policy.enable_corruption = False
    eval_cfg.curriculum.command_levels = None

    if hasattr(eval_cfg.events, "randomize_apply_external_force_torque"):
        eval_cfg.events.randomize_apply_external_force_torque = None
    if hasattr(eval_cfg.events, "randomize_push_robot"):
        eval_cfg.events.randomize_push_robot = None

    if force_deploy_reset:
        if hasattr(eval_cfg.events, "reset_to_deploy"):
            eval_cfg.events.reset_to_deploy.params["deploy_prob"] = 1.0
            eval_cfg.events.reset_to_deploy.params["add_noise"] = False
        if hasattr(eval_cfg.events, "reset_to_near_goal"):
            eval_cfg.events.reset_to_near_goal.params["near_goal_prob"] = 0.0
        if hasattr(eval_cfg.events, "randomize_reset_base"):
            eval_cfg.events.randomize_reset_base.params["pose_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
            }
            eval_cfg.events.randomize_reset_base.params["velocity_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            }
        if hasattr(eval_cfg.events, "randomize_reset_joints"):
            eval_cfg.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        if hasattr(eval_cfg.events, "randomize_actuator_gains"):
            eval_cfg.events.randomize_actuator_gains = None
        if hasattr(eval_cfg.events, "randomize_motor_strength"):
            eval_cfg.events.randomize_motor_strength = None

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # Ensure logs are world-accessible by default.
    os.umask(0o000)
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create log directory early so helper scripts can be generated
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    _make_world_writable(os.path.join("logs", "rsl_rl"))
    _make_world_writable(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    _make_world_writable(log_dir)
    _make_world_writable(os.path.join(log_dir, "params"))
    _write_run_scripts(
        log_dir=log_dir,
        task=args_cli.task,
        experiment_name=agent_cfg.experiment_name,
        agent_entry_point=args_cli.agent,
    )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # prepare optional reward clipping and observation history
    only_positive_rewards = getattr(env_cfg, "only_positive_rewards", None)
    if only_positive_rewards is None:
        only_positive_rewards = getattr(getattr(env_cfg, "rewards", None), "only_positive_rewards", False)
    term_weight = None
    if only_positive_rewards:
        if getattr(env_cfg, "rewards", None) is not None and getattr(env_cfg.rewards, "is_terminated", None) is not None:
            term_weight = env_cfg.rewards.is_terminated.weight
    obs_history_len = int(getattr(env_cfg, "num_observation_history", 0) or 0)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env = RslRlCompatWrapper(
        env,
        obs_history_length=obs_history_len,
        only_positive_rewards=only_positive_rewards,
        termination_reward_weight=term_weight,
    )
    history_mode = _history_mode_label()
    if history_mode == "unrealistic_history_feed":
        print("[WARN] History mode: unrealistic_history_feed (debug-only legacy behavior).")
    else:
        print("[INFO] History mode: default_reset_on_done")

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # Optional train-time evaluation pass (deploy-aligned).
    eval_env = None
    eval_inplace_env = None
    eval_every = _parse_int_env("LITE3_TRAIN_EVAL_EVERY", default=200)
    if eval_every > 0:
        eval_steps = _parse_int_env("LITE3_TRAIN_EVAL_STEPS", default=200)
        eval_num_envs = _parse_int_env("LITE3_TRAIN_EVAL_NUM_ENVS", default=1)
        force_deploy = _parse_bool_env("LITE3_TRAIN_EVAL_FORCE_DEPLOY_RESET", default=True)
        eval_mode = os.getenv("LITE3_TRAIN_EVAL_MODE", "inplace").strip().lower()
        if eval_mode == "separate" and not _parse_bool_env("LITE3_TRAIN_EVAL_ALLOW_SEPARATE", default=False):
            print("[INFO] LITE3_TRAIN_EVAL_MODE=separate ignored (set "
                  "LITE3_TRAIN_EVAL_ALLOW_SEPARATE=1 to enable). Falling back to inplace eval.")
            eval_mode = "inplace"

        if eval_mode == "separate":
            try:
                eval_cfg = copy.deepcopy(env_cfg)
                eval_cfg.scene.num_envs = eval_num_envs
                _configure_eval_env_cfg(eval_cfg, force_deploy_reset=force_deploy)

                eval_env = gym.make(args_cli.task, cfg=eval_cfg, render_mode=None)
                if isinstance(eval_env.unwrapped, DirectMARLEnv):
                    eval_env = multi_agent_to_single_agent(eval_env)
                eval_env = RslRlVecEnvWrapper(eval_env, clip_actions=agent_cfg.clip_actions)
                eval_env = RslRlCompatWrapper(
                    eval_env,
                    obs_history_length=obs_history_len,
                    only_positive_rewards=only_positive_rewards,
                    termination_reward_weight=term_weight,
                )
            except RuntimeError as exc:
                if "Simulation context already exists" in str(exc):
                    print("[WARN] Eval env creation failed (simulation context already exists). "
                          "Falling back to in-place eval on training env.")
                    eval_env = None
                    eval_inplace_env = env
                    eval_mode = "inplace"
                else:
                    raise
        else:
            eval_inplace_env = env
            eval_mode = "inplace"

        eval_log_path = os.path.join(log_dir, "eval_history.jsonl")
        eval_metric_name = os.getenv("LITE3_TRAIN_EVAL_METRIC", "two_leg_stand_metric")
        eval_metric_threshold = float(os.getenv("LITE3_TRAIN_EVAL_SUCCESS_THRESHOLD", "0.75"))

        metric_env = None
        metric_fn = None
        metric_params = None
        metric_source = eval_env if eval_env is not None else eval_inplace_env
        for candidate in (metric_source, getattr(metric_source, "env", None), getattr(metric_source, "unwrapped", None)):
            if candidate is None:
                continue
            rm = getattr(candidate, "reward_manager", None)
            if rm is None or not hasattr(rm, "get_term_cfg"):
                continue
            try:
                term_cfg = rm.get_term_cfg(eval_metric_name)
            except Exception:
                term_cfg = None
            if term_cfg is not None:
                metric_env = candidate
                metric_fn = term_cfg.func
                metric_params = term_cfg.params
                break

        def _eval_callback(it: int) -> None:
            if eval_env is None:
                # In-place eval: use current training env state (no reset/steps).
                mean_rew = float("nan")
                mean_len = float("nan")
                if metric_fn is not None and metric_env is not None:
                    metric_val = metric_fn(metric_env, **metric_params)
                    mean_metric = float(metric_val.detach().mean().item())
                    success_rate = float((metric_val >= eval_metric_threshold).float().mean().item())
                else:
                    mean_metric = None
                    success_rate = None
                eval_mode = "inplace"
            else:
                eval_mode = "rollout"
                policy = runner.get_inference_policy(device=runner.device)
                obs_dict = eval_env.reset()
                obs = obs_dict.get("obs")
                obs_history = obs_dict.get("obs_history")
                if obs is None:
                    return
                obs = obs.to(runner.device)
                if obs_history is not None:
                    obs_history = obs_history.to(runner.device)

                rewbuffer = []
                lenbuffer = []
                cur_reward_sum = torch.zeros(eval_env.num_envs, dtype=torch.float, device=runner.device)
                cur_episode_length = torch.zeros(eval_env.num_envs, dtype=torch.float, device=runner.device)
                metric_sum = None
                metric_hits = None
                metric_count = 0

                with torch.inference_mode():
                    for _ in range(eval_steps):
                        if obs_history is None:
                            actions = policy(obs)
                        else:
                            actions = policy(obs, obs_history)
                        obs_dict, rewards, dones, infos = eval_env.step(actions)
                        obs = obs_dict.get("obs")
                        obs_history = obs_dict.get("obs_history")
                        if obs is None:
                            break
                        obs = obs.to(runner.device)
                        if obs_history is not None:
                            obs_history = obs_history.to(runner.device)
                        rewards = rewards.to(runner.device)
                        dones = dones.to(runner.device)

                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        if metric_fn is not None and metric_env is not None:
                            metric_val = metric_fn(metric_env, **metric_params)
                            if metric_sum is None:
                                metric_sum = metric_val.detach().clone()
                                metric_hits = (metric_val >= eval_metric_threshold).float()
                            else:
                                metric_sum += metric_val.detach()
                                metric_hits += (metric_val >= eval_metric_threshold).float()
                            metric_count += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if new_ids.numel() > 0:
                            rewbuffer.extend(cur_reward_sum[new_ids][:, 0].detach().cpu().numpy().tolist())
                            lenbuffer.extend(cur_episode_length[new_ids][:, 0].detach().cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                mean_rew = statistics.mean(rewbuffer) if rewbuffer else float(cur_reward_sum.mean().item())
                mean_len = statistics.mean(lenbuffer) if lenbuffer else float(cur_episode_length.mean().item())
                if metric_fn is not None and metric_sum is not None and metric_count > 0:
                    mean_metric = float((metric_sum / metric_count).mean().item())
                    success_rate = float((metric_hits / metric_count).mean().item())
                else:
                    mean_metric = None
                    success_rate = None

            payload = {
                "iteration": int(it),
                "eval_steps": int(eval_steps),
                "num_envs": int(eval_num_envs),
                "force_deploy_reset": bool(force_deploy),
                "history_mode": history_mode,
                "eval_mode": eval_mode,
                "mean_reward": float(mean_rew),
                "mean_episode_length": float(mean_len),
                "success_metric_name": eval_metric_name if metric_fn is not None else None,
                "success_metric_threshold": eval_metric_threshold if metric_fn is not None else None,
                "mean_success_metric": mean_metric,
                "success_rate": success_rate,
            }
            print(
                f"[EVAL] iter {it}: mean_reward={payload['mean_reward']:.4f}, "
                f"mean_len={payload['mean_episode_length']:.2f}, "
                f"success_rate={payload['success_rate'] if payload['success_rate'] is not None else 'N/A'}"
            )
            try:
                with open(eval_log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload) + "\n")
            except Exception:
                pass

        runner.eval_every = eval_every
        runner.eval_callback = _eval_callback

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    _make_world_writable(os.path.join(log_dir, "params", "env.yaml"), is_dir=False)
    _make_world_writable(os.path.join(log_dir, "params", "agent.yaml"), is_dir=False)
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    if eval_env is not None:
        eval_env.close()
    env.close()


def _write_run_scripts(log_dir: str, task: str, experiment_name: str, agent_entry_point: str) -> None:
    """Create run_play.sh, run_resume.sh, run_evolution.sh in the run directory."""
    run_dir_abs = os.path.abspath(log_dir)
    exp_name = experiment_name
    run_name = os.path.basename(run_dir_abs)

    run_play_path = os.path.join(run_dir_abs, "run_play.sh")
    run_resume_path = os.path.join(run_dir_abs, "run_resume.sh")
    run_evolution_path = os.path.join(run_dir_abs, "run_evolution.sh")

    play_cmd = f"""#!/usr/bin/env bash
set -euo pipefail
# Resolve repo/log roots relative to this run directory
THIS_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
LOGS_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LOGS_ROOT/../.." && pwd)"

PYTHON_BIN="${{PYTHON_BIN:-python}}"
if [[ -x "/isaac-sim/python.sh" ]]; then
  PYTHON_BIN="/isaac-sim/python.sh"
fi

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

# Optional debug dumps for parity checks (enabled by default for quick inspection).
# Always enable debug dumps for parity workflow (no env needed).
export LITE3_DEBUG_PLAY_DUMPS="5"
export LITE3_DEBUG_PLAY_EVERY="1"
# Default to a stable, shared debug root for parity workflows.
export LITE3_DEBUG_PLAY_DIR="${{LITE3_DEBUG_PLAY_DIR:-/workspace/rl_training_new/lite3_debug/train/$(basename \"$THIS_DIR\")}}"
# Parity helper (opt-in): force deploy-style reset only when explicitly requested.
export LITE3_PLAY_FORCE_DEPLOY_RESET="${{LITE3_PLAY_FORCE_DEPLOY_RESET:-0}}"
export LITE3_PLAY_NUM_ENVS="${{LITE3_PLAY_NUM_ENVS:-1}}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/reinforcement_learning/rsl_rl/play.py" \\
  --task "{task}" \\
  --agent "{agent_entry_point}" \\
  --load_run "{run_name}" \\
  $CKPT_FLAG \\
  $HEADLESS_FLAG
"""

    resume_cmd = f"""#!/usr/bin/env bash
set -euo pipefail
# Resolve repo/log roots relative to this run directory
THIS_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
LOGS_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LOGS_ROOT/../.." && pwd)"

PYTHON_BIN="${{PYTHON_BIN:-python}}"
if [[ -x "/isaac-sim/python.sh" ]]; then
  PYTHON_BIN="/isaac-sim/python.sh"
fi

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
"$PYTHON_BIN" "$REPO_ROOT/scripts/reinforcement_learning/rsl_rl/train.py" \\
  --task "{task}" \\
  --agent "{agent_entry_point}" \\
  --resume \\
  --load_run "{run_name}" \\
  $CKPT_FLAG \\
  --headless
"""

    evolution_cmd = f"""#!/usr/bin/env bash
set -euo pipefail

# Resolve repo/log roots relative to this run directory
THIS_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
LOGS_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LOGS_ROOT/../.." && pwd)"

PYTHON_BIN="${{PYTHON_BIN:-python}}"
if [[ -x "/isaac-sim/python.sh" ]]; then
  PYTHON_BIN="/isaac-sim/python.sh"
fi

EXP_NAME="$(basename "$(dirname "$THIS_DIR")")"
RUN_NAME="$(basename "$THIS_DIR")"

# Default checkpoints to showcase policy evolution
DEFAULT_CKPTS=("model_1000.pt" "model_2000.pt" "model_3000.pt" "model_4000.pt" "model_5000.pt" "model_6000.pt" "model_7000.pt" "model_8000.pt" "model_9000.pt" "model_10000.pt" "model_11000.pt" "model_12000.pt" "model_13000.pt" "model_14000.pt" "model_15000.pt" "model_16000.pt" "model_17000.pt" "model_18000.pt" "model_19000.pt" "model_20000.pt")
CHECKPOINTS=("${{DEFAULT_CKPTS[@]}}")
FPS=30
SECONDS_PER_CKPT=5
FRAMES_PER_CKPT=""
NUM_ENVS=20
OUTPUT_NAME="policy_evolution.mp4"
KEEP_FRAMES="${{KEEP_FRAMES:-1}}"
SIM_DEVICE="${{SIM_DEVICE:-cuda:0}}"
RL_DEVICE="${{RL_DEVICE:-cuda:0}}"
HEADLESS=0

usage() {{
  cat <<'USAGE'
Usage: ./run_evolution.sh [options]
  --checkpoints <ckpt1 ckpt2 ...>   Space-separated checkpoint names
  --fps <int>                       Frames per second for the output video (default: 30)
  --seconds <float>                 Seconds to record per checkpoint (default: 5)
  --frames <int>                    Override total frames per checkpoint (overrides --seconds)
  --num-envs <int>                  Number of envs to roll out (default: 20)
  --output <filename.mp4>           Final combined video name (default: policy_evolution.mp4)
  --headless                        Run Isaac Sim headless
  --keep-frames                     Keep intermediate MP4s (default: kept)
  --sim-device <device>             Simulation device (default: cuda:0)
  --rl-device <device>              RL device (default: cuda:0)
USAGE
}}

# Parse CLI flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoints)
      CHECKPOINTS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CHECKPOINTS+=("$1")
        shift
      done
      ;;
    --fps) FPS="$2"; shift 2 ;;
    --seconds) SECONDS_PER_CKPT="$2"; shift 2 ;;
    --frames) FRAMES_PER_CKPT="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --output) OUTPUT_NAME="$2"; shift 2 ;;
    --headless) HEADLESS=1; shift ;;
    --keep-frames) KEEP_FRAMES=1; shift ;;
    --sim-device) SIM_DEVICE="$2"; shift 2 ;;
    --rl-device) RL_DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

TASK="{task}"

export EVOLVE_RUN_DIR="$THIS_DIR"
export EVOLVE_REPO_ROOT="$REPO_ROOT"
export EVOLVE_TASK="$TASK"
export EVOLVE_EXP_NAME="$EXP_NAME"
export EVOLVE_RUN_NAME="$RUN_NAME"
export EVOLVE_AGENT="{agent_entry_point}"
export EVOLVE_CHECKPOINTS="${{CHECKPOINTS[*]}}"
export EVOLVE_FPS="$FPS"
export EVOLVE_SECONDS="$SECONDS_PER_CKPT"
export EVOLVE_FRAMES="$FRAMES_PER_CKPT"
export EVOLVE_NUM_ENVS="$NUM_ENVS"
export EVOLVE_OUTPUT_NAME="$OUTPUT_NAME"
export EVOLVE_KEEP_FRAMES="$KEEP_FRAMES"
export EVOLVE_SIM_DEVICE="$SIM_DEVICE"
export EVOLVE_RL_DEVICE="$RL_DEVICE"
export EVOLVE_HEADLESS="$HEADLESS"

"$PYTHON_BIN" - <<'PY'
import os
import shutil
import subprocess
import sys
from pathlib import Path

ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
if shutil.which(ffmpeg_bin) is None:
    sys.stderr.write(
        f"[error] ffmpeg not found (looked for '{{ffmpeg_bin}}'). "
        "Install it (e.g., `apt-get update && apt-get install -y ffmpeg`) "
        "or set FFMPEG_BIN=/path/to/ffmpeg.\\n"
    )
    sys.exit(1)
ffmpeg_crf = os.environ.get("FFMPEG_CRF", "18")
ffmpeg_preset = os.environ.get("FFMPEG_PRESET", "medium")

run_dir = Path(os.environ["EVOLVE_RUN_DIR"]).resolve()
repo_root = Path(os.environ["EVOLVE_REPO_ROOT"]).resolve()
task = os.environ.get("EVOLVE_TASK", "lite3")
run_name = os.environ["EVOLVE_RUN_NAME"]
agent_entry = os.environ.get("EVOLVE_AGENT", "rsl_rl_cfg_entry_point")
ckpt_list = os.environ.get("EVOLVE_CHECKPOINTS", "").split()
if not ckpt_list:
    print("No checkpoints provided. Nothing to record.", file=sys.stderr)
    sys.exit(1)

fps = int(os.environ.get("EVOLVE_FPS", "30"))
frames_env = os.environ.get("EVOLVE_FRAMES", "").strip()
if frames_env:
    frames_per_ckpt = int(float(frames_env))
else:
    seconds = float(os.environ.get("EVOLVE_SECONDS", "5"))
    frames_per_ckpt = max(1, int(fps * seconds))
num_envs = int(os.environ.get("EVOLVE_NUM_ENVS", "1"))
output_name = os.environ.get("EVOLVE_OUTPUT_NAME", "policy_evolution.mp4")
keep_frames = os.environ.get("EVOLVE_KEEP_FRAMES", "1") == "1"
headless = os.environ.get("EVOLVE_HEADLESS", "0") == "1"

videos_root = run_dir / "evolution_videos"
videos_root.mkdir(exist_ok=True)

def _run_play(checkpoint: str) -> Path | None:
    ckpt_path = run_dir / checkpoint
    if not ckpt_path.exists():
        print(f"[skip] checkpoint {{checkpoint}} not found in {{run_dir}}", file=sys.stderr)
        return None
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"),
        "--task",
        task,
        "--agent",
        agent_entry,
        "--load_run",
        run_name,
        "--checkpoint",
        str(ckpt_path),
        "--video",
        "--video_length",
        str(frames_per_ckpt),
        "--num_envs",
        str(num_envs),
    ]
    if headless:
        cmd.append("--headless")
    print("[record]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    play_dir = run_dir / "videos" / "play"
    if not play_dir.exists():
        print(f"[error] play video folder not found: {{play_dir}}", file=sys.stderr)
        return None
    mp4s = sorted(play_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4s:
        print(f"[error] no mp4 produced in {{play_dir}}", file=sys.stderr)
        return None
    latest = mp4s[-1]
    target = videos_root / f"{{checkpoint.replace('.pt','')}}.mp4"
    latest.rename(target)
    return target

clips = []
for ckpt in ckpt_list:
    path = _run_play(ckpt)
    if path:
        clips.append(path)

if not clips:
    print("No clips were produced. Ensure checkpoints exist and Isaac Sim can render.", file=sys.stderr)
    sys.exit(1)

concat_list = videos_root / "concat_list.txt"
with open(concat_list, "w") as f:
    for clip in clips:
        f.write(f"file '{{clip.as_posix()}}'\\n")

final_path = run_dir / output_name
ffmpeg_cmd = [
    ffmpeg_bin,
    "-y",
    "-f",
    "concat",
    "-safe",
    "0",
    "-i",
    str(concat_list),
    "-c:v",
    "libx264",
    "-crf",
    str(ffmpeg_crf),
    "-preset",
    str(ffmpeg_preset),
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    str(final_path),
]
subprocess.run(ffmpeg_cmd, check=True)
print(f"[done] Video saved to: {{final_path}}")

if not keep_frames:
    for clip in clips:
        clip.unlink(missing_ok=True)
    concat_list.unlink(missing_ok=True)
PY
"""

    script_pairs = [
        (run_play_path, play_cmd),
        (run_resume_path, resume_cmd),
        (run_evolution_path, evolution_cmd),
    ]
    for path, content in script_pairs:
        with open(path, "w") as f:
            f.write(textwrap.dedent(content))
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
        _make_world_writable(path, is_dir=False)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
