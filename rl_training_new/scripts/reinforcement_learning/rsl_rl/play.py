# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Omniverse/IsaacSim-dependent utilities after SimulationApp is ready.
from rl_utils import camera_follow

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import numpy as np

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401
from rl_training.utils.env_wrappers import RslRlCompatWrapper


OBS_CONTRACT = [
    ("cmd", 3),
    ("base_rpy", 3),
    ("body_omega", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("joint_pos_history", 36),
    ("joint_vel_history", 24),
    ("action_history", 24),
]
OBS_DIM = sum(dim for _, dim in OBS_CONTRACT)

POLICY_JOINT_ORDER = [
    "FL_HipX_joint",
    "FR_HipX_joint",
    "HL_HipX_joint",
    "HR_HipX_joint",
    "FL_HipY_joint",
    "FR_HipY_joint",
    "HL_HipY_joint",
    "HR_HipY_joint",
    "FL_Knee_joint",
    "FR_Knee_joint",
    "HL_Knee_joint",
    "HR_Knee_joint",
]
# Mapping used by deploy runner: robot_idx -> policy_idx.
POLICY_IDX_FOR_ROBOT = np.asarray([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=np.int64)
# Mapping used by deploy runner: policy_idx -> robot_idx.
POLICY_FROM_ROBOT_IDX = np.asarray([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=np.int64)
DEFAULT_JOINT_POS_POLICY = np.asarray(
    [
        -0.0154048,
        0.0159887,
        -0.0221317,
        0.0224431,
        -0.76697,
        -0.768286,
        -0.765865,
        -0.767203,
        1.53761,
        1.53636,
        1.54788,
        1.54679,
    ],
    dtype=np.float32,
)
DEFAULT_PD_KP = np.full((12,), 20.0, dtype=np.float32)
DEFAULT_PD_KD = np.full((12,), 0.7, dtype=np.float32)
DEFAULT_EFFORT_LIMITS = np.asarray(
    [24.0, 24.0, 36.0, 24.0, 24.0, 36.0, 24.0, 24.0, 36.0, 24.0, 24.0, 36.0],
    dtype=np.float32,
)


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


def _to_cpu_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _first_env_numpy(value):
    arr = _to_cpu_numpy(value)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim >= 2 and arr.shape[0] > 0:
        arr = arr[0]
    return arr


def _as_float_vec(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray | list[float] | tuple[float, ...] | None) -> np.ndarray | None:
    if quat_wxyz is None:
        return None
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if q.size != 4:
        return None
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1.0e-8:
        return None
    w, x, y, z = q / norm
    rot = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rot


def _extract_action_scale(env_cfg) -> float:
    try:
        action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
        if action_cfg is None:
            return 1.0
        scale = getattr(action_cfg, "scale", 1.0)
        if isinstance(scale, (tuple, list)):
            return float(scale[0]) if len(scale) > 0 else 1.0
        if isinstance(scale, dict):
            if not scale:
                return 1.0
            return float(next(iter(scale.values())))
        return float(scale)
    except Exception:
        return 1.0


def _resolve_effective_gain(raw_gain: np.ndarray | None, fallback: np.ndarray, act_dim: int) -> np.ndarray:
    """Use fallback gains when simulator-exported gain tensors are missing/placeholder zeros."""
    gain = fallback[:act_dim].astype(np.float32, copy=True)
    if raw_gain is None or raw_gain.size < act_dim:
        return gain
    cand = raw_gain[:act_dim].astype(np.float32, copy=False)
    if not np.all(np.isfinite(cand)):
        return gain
    if float(np.max(np.abs(cand))) <= 1.0e-6:
        return gain
    return cand.astype(np.float32, copy=True)


def _resolve_effective_effort_limits(raw_limits: np.ndarray | None, act_dim: int) -> np.ndarray:
    """Use training actuator limits when sim reports unbounded placeholder limits."""
    limits = DEFAULT_EFFORT_LIMITS[:act_dim].astype(np.float32, copy=True)
    if raw_limits is None or raw_limits.size < act_dim:
        return limits
    cand = raw_limits[:act_dim].astype(np.float32, copy=False)
    if not np.all(np.isfinite(cand)):
        return limits
    max_abs = float(np.max(np.abs(cand)))
    if max_abs <= 1.0e-6 or max_abs >= 1.0e6:
        return limits
    if float(np.min(cand)) <= 0.0:
        return limits
    return cand.astype(np.float32, copy=True)


def _ensure_world_writable(path: str) -> None:
    """Best-effort make files/dirs world-writable so host users can delete logs."""
    if not path or not os.path.exists(path):
        return
    try:
        for root, dirs, files in os.walk(path):
            for name in dirs:
                dpath = os.path.join(root, name)
                try:
                    mode = os.stat(dpath).st_mode
                    os.chmod(dpath, mode | 0o777)
                except OSError:
                    pass
            for name in files:
                fpath = os.path.join(root, name)
                try:
                    mode = os.stat(fpath).st_mode
                    os.chmod(fpath, mode | 0o666)
                except OSError:
                    pass
        mode = os.stat(path).st_mode
        os.chmod(path, mode | 0o777)
    except OSError:
        pass


def _clear_two_leg_history_caches(env) -> int:
    """Best-effort clear cached two-leg history state on wrapper chain."""
    cleared = 0
    seen = set()
    candidates = [env]
    for _ in range(6):
        nxt = []
        for obj in candidates:
            if obj is None:
                continue
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            for attr in ("env", "unwrapped"):
                if hasattr(obj, attr):
                    try:
                        nxt.append(getattr(obj, attr))
                    except Exception:
                        pass
        candidates.extend(nxt)

    for obj in list(candidates):
        if obj is None:
            continue
        # Clear compatibility wrapper history buffers.
        if hasattr(obj, "obs_history"):
            try:
                hist = getattr(obj, "obs_history")
                if hist is not None:
                    if torch.is_tensor(hist):
                        hist.zero_()
                    else:
                        hist[:] = 0
                    cleared += 1
            except Exception:
                pass
        # Drop cached two-leg tensors; observation helpers will lazily rebuild.
        for name in dir(obj):
            if not name.startswith("_two_leg_"):
                continue
            try:
                delattr(obj, name)
                cleared += 1
            except Exception:
                try:
                    setattr(obj, name, None)
                    cleared += 1
                except Exception:
                    pass
    return cleared


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_num_override = _parse_int_env("LITE3_PLAY_NUM_ENVS", default=0)
    if env_num_override > 0:
        env_cfg.scene.num_envs = env_num_override
    else:
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    if hasattr(env_cfg.events, "randomize_apply_external_force_torque"):
        env_cfg.events.randomize_apply_external_force_torque = None
    if hasattr(env_cfg.events, "randomize_push_robot"):
        env_cfg.events.randomize_push_robot = None
    env_cfg.curriculum.command_levels = None

    # Optional: force deploy-style reset to align play observations with deployment parity checks.
    if _parse_bool_env("LITE3_PLAY_FORCE_DEPLOY_RESET", default=False):
        print("[INFO] Forcing deploy-style reset for play parity checks.")
        if hasattr(env_cfg.events, "reset_to_deploy"):
            env_cfg.events.reset_to_deploy.params["deploy_prob"] = 1.0
            env_cfg.events.reset_to_deploy.params["add_noise"] = False
        if hasattr(env_cfg.events, "reset_to_near_goal"):
            env_cfg.events.reset_to_near_goal.params["near_goal_prob"] = 0.0
        if hasattr(env_cfg.events, "randomize_reset_base"):
            env_cfg.events.randomize_reset_base.params["pose_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
            }
            env_cfg.events.randomize_reset_base.params["velocity_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            }
        if hasattr(env_cfg.events, "randomize_reset_joints"):
            env_cfg.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        if hasattr(env_cfg.events, "randomize_actuator_gains"):
            env_cfg.events.randomize_actuator_gains = None
        if hasattr(env_cfg.events, "randomize_motor_strength"):
            env_cfg.events.randomize_motor_strength = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1]/2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments (anchor to repo root, not CWD)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    log_root_path = os.path.join(repo_root, "logs", "rsl_rl", agent_cfg.experiment_name)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

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

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.onnx",
    )
    export_policy_as_jit(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.pt",
    )
    _ensure_world_writable(export_model_dir)

    dt = env.unwrapped.step_dt
    # print(dt, "dt")
    # reset environment
    
    # Use an explicit reset so observation/history caches are deterministic for parity dumps.
    obs = env.reset()
    if _parse_bool_env("LITE3_PLAY_FORCE_DEPLOY_RESET", default=False):
        cleared = _clear_two_leg_history_caches(env)
        if cleared > 0:
            print(f"[INFO] Cleared {cleared} cached history fields for deploy-parity reset.")
        # Refresh observations after cache cleanup.
        obs = env.get_observations()
    if isinstance(obs, tuple) and len(obs) == 2:
        obs = obs[0]
    obs_history = None
    if isinstance(obs, dict):
        obs_history = obs.get("obs_history", None)
        obs = obs.get("obs", obs)

    # Debug dump config for parity checks with deploy.
    debug_quota = _parse_int_env("LITE3_DEBUG_PLAY_DUMPS", default=5)
    debug_every = max(1, _parse_int_env("LITE3_DEBUG_PLAY_EVERY", default=1))
    debug_dump_full = _parse_bool_env("LITE3_DEBUG_PLAY_FULL", default=True)
    debug_counter = 0
    debug_dir = None
    policy_action_scale = _extract_action_scale(env_cfg)
    if debug_quota > 0:
        run_id = os.path.basename(os.path.normpath(log_dir)) if log_dir else "unknown_run"
        default_root = "/workspace/rl_training_new/lite3_debug/train"
        debug_dir = os.getenv("LITE3_DEBUG_PLAY_DIR") or os.path.join(default_root, run_id)
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[DEBUG] Play dumps enabled: quota={debug_quota} every={debug_every} dir={debug_dir}")

    def _maybe_dump_play_debug(step_idx, obs, obs_history, actions):
        nonlocal debug_counter
        if debug_quota <= 0 or debug_dir is None:
            return
        if step_idx % debug_every != 0:
            return
        if debug_counter >= debug_quota:
            return
        debug_counter += 1

        obs_np = _to_cpu_numpy(obs)
        hist_np = _to_cpu_numpy(obs_history)
        act_np = _to_cpu_numpy(actions)

        if obs_np is None:
            return
        # Use first env for parity checks.
        obs0 = obs_np[0] if obs_np.ndim >= 2 else obs_np
        hist0 = None
        if hist_np is not None:
            hist0 = hist_np[0] if hist_np.ndim >= 2 else hist_np

        # Build flattened input that matches deploy: [obs, obs_history].
        if hist0 is None:
            obs_flat = obs0
        else:
            obs_flat = np.concatenate([obs0, hist0], axis=-1)

        # Extract key slices (matches deploy observation builder / contract).
        cmd = obs0[0:3]
        base_rpy = obs0[3:6]
        body_omega = obs0[6:9]
        joint_pos = obs0[9:21]
        joint_vel = obs0[21:33]
        joint_pos_history = obs0[33:69]
        joint_vel_history = obs0[69:93]
        action_history = obs0[93:117]

        actions0 = act_np[0] if act_np is not None and getattr(act_np, "ndim", 0) >= 2 else act_np
        actions0 = _as_float_vec(actions0)

        # Optional extended parity debug fields (state + control path).
        extended_payload = {}
        if debug_dump_full:
            robot = None
            robot_data = None
            try:
                robot = env.unwrapped.scene["robot"]
                robot_data = getattr(robot, "data", None)
            except Exception:
                robot = None
                robot_data = None

            base_quat_wxyz = _as_float_vec(
                _first_env_numpy(getattr(robot_data, "root_quat_w", None)) if robot_data is not None else None
            )
            projected_gravity = _as_float_vec(
                _first_env_numpy(getattr(robot_data, "projected_gravity_b", None)) if robot_data is not None else None
            )
            omega_world = _as_float_vec(
                _first_env_numpy(getattr(robot_data, "root_ang_vel_w", None)) if robot_data is not None else None
            )
            base_rot_mat = _quat_wxyz_to_rotmat(base_quat_wxyz)

            # Isaac tensors are in policy joint order; deploy control diagnostics are logged in robot order.
            joint_pos_policy_state = _as_float_vec(
                _first_env_numpy(getattr(robot_data, "joint_pos", None)) if robot_data is not None else None
            )
            joint_vel_policy_state = _as_float_vec(
                _first_env_numpy(getattr(robot_data, "joint_vel", None)) if robot_data is not None else None
            )
            joint_stiffness_policy = _as_float_vec(
                _first_env_numpy(getattr(robot_data, "joint_stiffness", None)) if robot_data is not None else None
            )
            joint_damping_policy = _as_float_vec(
                _first_env_numpy(getattr(robot_data, "joint_damping", None)) if robot_data is not None else None
            )
            default_joint_pos_policy_state = _as_float_vec(
                _first_env_numpy(getattr(robot_data, "default_joint_pos", None)) if robot_data is not None else None
            )

            soft_limits = _first_env_numpy(getattr(robot_data, "soft_joint_pos_limits", None)) if robot_data is not None else None
            joint_limits_lower_policy = None
            joint_limits_upper_policy = None
            if soft_limits is not None:
                soft_limits = np.asarray(soft_limits, dtype=np.float32)
                if soft_limits.ndim == 2 and soft_limits.shape[1] >= 2:
                    joint_limits_lower_policy = soft_limits[:, 0].reshape(-1)
                    joint_limits_upper_policy = soft_limits[:, 1].reshape(-1)

            effort_limits_policy = None
            if robot_data is not None:
                for attr_name in (
                    "joint_effort_limits",
                    "joint_effort_limit",
                    "actuator_effort_limits",
                    "joint_torque_limits",
                    "joint_torque_limit",
                ):
                    raw = getattr(robot_data, attr_name, None)
                    if raw is not None:
                        effort_limits_policy = _as_float_vec(_first_env_numpy(raw))
                        if effort_limits_policy is not None:
                            break

            act_dim = 12
            joint_pos_robot = None
            joint_vel_robot = None
            joint_limits_lower = None
            joint_limits_upper = None
            kp = None
            kd = None
            effort_lim_eff = None
            if actions0 is not None and actions0.size >= act_dim:
                action_raw = actions0[:act_dim].astype(np.float32, copy=False)
                action_offset = action_raw * np.float32(policy_action_scale)
                target_joint_pos_policy = DEFAULT_JOINT_POS_POLICY + action_offset
                target_joint_pos_robot = target_joint_pos_policy[POLICY_IDX_FOR_ROBOT]

                if joint_pos_policy_state is not None and joint_pos_policy_state.size >= act_dim:
                    joint_pos_robot = joint_pos_policy_state[:act_dim][POLICY_IDX_FOR_ROBOT]
                if joint_vel_policy_state is not None and joint_vel_policy_state.size >= act_dim:
                    joint_vel_robot = joint_vel_policy_state[:act_dim][POLICY_IDX_FOR_ROBOT]
                if joint_limits_lower_policy is not None and joint_limits_lower_policy.size >= act_dim:
                    joint_limits_lower = joint_limits_lower_policy[:act_dim][POLICY_IDX_FOR_ROBOT]
                if joint_limits_upper_policy is not None and joint_limits_upper_policy.size >= act_dim:
                    joint_limits_upper = joint_limits_upper_policy[:act_dim][POLICY_IDX_FOR_ROBOT]

                target_joint_pos_clipped = target_joint_pos_robot.copy()
                if joint_limits_lower is not None and joint_limits_upper is not None:
                    target_joint_pos_clipped = np.clip(
                        target_joint_pos_clipped,
                        joint_limits_lower[:act_dim],
                        joint_limits_upper[:act_dim],
                    )

                pd_tau_raw_est = None
                pd_tau_clipped_est = None
                kp_policy = _resolve_effective_gain(joint_stiffness_policy, DEFAULT_PD_KP, act_dim)
                kd_policy = _resolve_effective_gain(joint_damping_policy, DEFAULT_PD_KD, act_dim)
                effort_lim_eff_policy = _resolve_effective_effort_limits(effort_limits_policy, act_dim)
                kp = kp_policy[POLICY_IDX_FOR_ROBOT]
                kd = kd_policy[POLICY_IDX_FOR_ROBOT]
                effort_lim_eff = effort_lim_eff_policy[POLICY_IDX_FOR_ROBOT]
                if joint_pos_robot is not None and joint_vel_robot is not None:
                    if joint_pos_robot.size >= act_dim and joint_vel_robot.size >= act_dim:
                        pd_tau_raw_est = kp * (target_joint_pos_robot - joint_pos_robot[:act_dim]) + kd * (
                            -joint_vel_robot[:act_dim]
                        )
                        pd_tau_clipped_est = pd_tau_raw_est.copy()
                        pd_tau_clipped_est = np.clip(pd_tau_clipped_est, -effort_lim_eff, effort_lim_eff)

                extended_payload.update(
                    {
                        "action_offset": action_offset.astype(np.float32, copy=False),
                        "target_joint_pos_policy": target_joint_pos_policy.astype(np.float32, copy=False),
                        "target_joint_pos_robot": target_joint_pos_robot.astype(np.float32, copy=False),
                        "target_joint_pos_clipped": target_joint_pos_clipped.astype(np.float32, copy=False),
                    }
                )
                extended_payload["joint_stiffness"] = kp.astype(np.float32, copy=False)
                extended_payload["joint_damping"] = kd.astype(np.float32, copy=False)
                extended_payload["effort_limits"] = effort_lim_eff.astype(np.float32, copy=False)
                if pd_tau_raw_est is not None:
                    extended_payload["pd_tau_raw_est"] = pd_tau_raw_est.astype(np.float32, copy=False)
                if pd_tau_clipped_est is not None:
                    extended_payload["pd_tau_clipped_est"] = pd_tau_clipped_est.astype(np.float32, copy=False)
                if default_joint_pos_policy_state is not None and default_joint_pos_policy_state.size >= act_dim:
                    extended_payload["default_joint_pos_policy"] = default_joint_pos_policy_state[:act_dim].astype(
                        np.float32, copy=False
                    )

            if base_quat_wxyz is not None and base_quat_wxyz.size >= 4:
                extended_payload["base_quat_wxyz"] = base_quat_wxyz[:4].astype(np.float32, copy=False)
            if base_rot_mat is not None:
                extended_payload["base_rot_mat"] = base_rot_mat.reshape(-1).astype(np.float32, copy=False)
            if projected_gravity is not None and projected_gravity.size >= 3:
                extended_payload["projected_gravity"] = projected_gravity[:3].astype(np.float32, copy=False)
            if omega_world is not None and omega_world.size >= 3:
                extended_payload["omega_world"] = omega_world[:3].astype(np.float32, copy=False)
            if joint_pos_robot is not None and joint_pos_robot.size >= act_dim:
                extended_payload["joint_pos_robot"] = joint_pos_robot[:act_dim].astype(np.float32, copy=False)
            if joint_vel_robot is not None and joint_vel_robot.size >= act_dim:
                extended_payload["joint_vel_robot"] = joint_vel_robot[:act_dim].astype(np.float32, copy=False)
            if joint_limits_lower is not None and joint_limits_lower.size >= act_dim:
                extended_payload["joint_limits_lower"] = joint_limits_lower[:act_dim].astype(np.float32, copy=False)
            if joint_limits_upper is not None and joint_limits_upper.size >= act_dim:
                extended_payload["joint_limits_upper"] = joint_limits_upper[:act_dim].astype(np.float32, copy=False)
            if "effort_limits" not in extended_payload:
                effort_lim_fallback = _resolve_effective_effort_limits(effort_limits_policy, act_dim)
                extended_payload["effort_limits"] = effort_lim_fallback[POLICY_IDX_FOR_ROBOT]
            if "joint_stiffness" not in extended_payload:
                kp_fallback = _resolve_effective_gain(joint_stiffness_policy, DEFAULT_PD_KP, act_dim)
                extended_payload["joint_stiffness"] = kp_fallback[POLICY_IDX_FOR_ROBOT]
            if "joint_damping" not in extended_payload:
                kd_fallback = _resolve_effective_gain(joint_damping_policy, DEFAULT_PD_KD, act_dim)
                extended_payload["joint_damping"] = kd_fallback[POLICY_IDX_FOR_ROBOT]

        out_path = os.path.join(debug_dir, f"debug_play_step{step_idx}.npz")
        payload = {
            "obs_contract_names": np.asarray([name for name, _ in OBS_CONTRACT]),
            "obs_contract_dims": np.asarray([dim for _, dim in OBS_CONTRACT], dtype=np.int32),
            "obs": obs0,
            "obs_history": hist0,
            "obs_flat": obs_flat,
            "actions": actions0,
            "cmd": cmd,
            "base_rpy": base_rpy,
            "body_omega": body_omega,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_pos_history": joint_pos_history,
            "joint_vel_history": joint_vel_history,
            "action_history": action_history,
            "policy_action_scale": np.asarray([policy_action_scale], dtype=np.float32),
            "policy_joint_order": np.asarray(POLICY_JOINT_ORDER),
        }
        payload.update(extended_payload)
        np.savez(out_path, **payload)

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if obs_history is None:
                actions = policy(obs)
            else:
                actions = policy(obs, obs_history)

            _maybe_dump_play_debug(timestep, obs, obs_history, actions)

            # env stepping
            obs, _, _, _ = env.step(actions)
            if isinstance(obs, dict):
                obs_history = obs.get("obs_history", obs_history)
                obs = obs.get("obs", obs)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
