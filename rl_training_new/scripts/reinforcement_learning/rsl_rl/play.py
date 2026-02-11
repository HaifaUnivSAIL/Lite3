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
    
    obs = env.get_observations()
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

        # Extract key slices (matches deploy observation builder).
        cmd = obs0[0:3]
        base_rpy = obs0[3:6]
        body_omega = obs0[6:9]
        joint_pos = obs0[9:21]
        joint_vel = obs0[21:33]

        out_path = os.path.join(debug_dir, f"debug_play_step{step_idx}.npz")
        np.savez(
            out_path,
            obs=obs0,
            obs_history=hist0,
            obs_flat=obs_flat,
            actions=act_np[0] if act_np is not None and act_np.ndim >= 2 else act_np,
            cmd=cmd,
            base_rpy=base_rpy,
            body_omega=body_omega,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

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
