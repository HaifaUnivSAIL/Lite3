# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Two-leg standing reward functions for Lite3 quadruped robot.
# Ported from Lite3_rl_training legged_gym implementation.

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Helper Functions
# =============================================================================

def _get_rpy_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to roll-pitch-yaw angles.

    Args:
        quat: Quaternion tensor of shape (num_envs, 4) in (w, x, y, z) format.

    Returns:
        RPY tensor of shape (num_envs, 3) in radians.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * math.pi / 2,
        torch.asin(sinp.clamp(-1, 1))
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=1)


def _orientation_gate(
    env: ManagerBasedRLEnv,
    pitch_width: float,
    roll_width: float,
    pitch_target: float = -1.22,  # -70 degrees default
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute a 2D Gaussian gate based on pitch/roll error.

    Args:
        env: The RL environment instance.
        pitch_width: Denominator for pitch error term.
        roll_width: Denominator for roll error term.
        pitch_target: Target pitch angle in radians (default: -70 deg for lean-back).
        asset_cfg: Asset configuration.

    Returns:
        Gate values in [0, 1] for each environment.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    rpy = _get_rpy_from_quat(asset.data.root_quat_w)
    roll = rpy[:, 0]
    pitch = rpy[:, 1]

    pitch_error = pitch - pitch_target
    pitch_den = max(pitch_width, 1e-4)
    roll_den = max(roll_width, 1e-4)

    return torch.exp(-(torch.square(pitch_error) / pitch_den + torch.square(roll) / roll_den))


def _get_front_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Check if front feet are in contact.

    Returns:
        Boolean tensor of shape (num_envs,) - True if any front foot in contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    # Use the most recent contact sample for parity with legged_gym contact_filt.
    current_forces = net_forces[:, -1, sensor_cfg.body_ids]
    is_contact = torch.norm(current_forces, dim=-1) > threshold
    return torch.any(is_contact, dim=1)


def _get_hind_feet_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Check if both hind feet are in contact.

    Returns:
        Boolean tensor of shape (num_envs,) - True if both hind feet in contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    current_forces = net_forces[:, -1, sensor_cfg.body_ids]
    is_contact = torch.norm(current_forces, dim=-1) > threshold
    return torch.all(is_contact, dim=1)


def _get_action_history(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (current, prev, prev_prev) actions with step-safe caching."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    step = getattr(env, "common_step_counter", None)
    last_step = getattr(env, "_two_leg_action_hist_step", None)
    reset_mask = None
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        reset_mask = env.episode_length_buf == 0
    if reset_mask is not None and reset_mask.any():
        asset: Articulation = env.scene["robot"]
        reset_actions = asset.data.joint_pos - asset.data.default_joint_pos
        prev_action = prev_action.clone()
        prev_action[reset_mask] = reset_actions[reset_mask]
    if last_step != step:
        prev_prev = getattr(env, "_two_leg_prev_action_hist", prev_action.clone())
        if reset_mask is not None and reset_mask.any():
            prev_prev = prev_prev.clone()
            prev_prev[reset_mask] = prev_action[reset_mask]
        env._two_leg_prev_prev_action = prev_prev
        env._two_leg_prev_action_hist = prev_action.clone()
        env._two_leg_action_hist_step = step
    prev_prev_action = getattr(env, "_two_leg_prev_prev_action", prev_action.clone())
    if reset_mask is not None and reset_mask.any():
        prev_prev_action = prev_prev_action.clone()
        prev_prev_action[reset_mask] = prev_action[reset_mask]
    return action, prev_action, prev_prev_action


def _get_joint_vel_history(env: ManagerBasedRLEnv, joint_vel: torch.Tensor) -> torch.Tensor:
    """Return previous-step joint velocities with step-safe caching."""
    step = getattr(env, "common_step_counter", None)
    last_step = getattr(env, "_two_leg_joint_vel_hist_step", None)
    reset_mask = None
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        reset_mask = env.episode_length_buf == 0
    if last_step != step:
        prev_vel = getattr(env, "_two_leg_prev_joint_vel_hist", joint_vel.clone())
        prev_prev_vel = getattr(env, "_two_leg_prev_joint_vel", prev_vel.clone())
        if reset_mask is not None and reset_mask.any():
            prev_vel = prev_vel.clone()
            prev_prev_vel = prev_prev_vel.clone()
            prev_vel[reset_mask] = joint_vel[reset_mask]
            prev_prev_vel[reset_mask] = joint_vel[reset_mask]
        env._two_leg_prev_prev_joint_vel = prev_prev_vel
        env._two_leg_prev_joint_vel = prev_vel
        env._two_leg_prev_joint_vel_hist = joint_vel.clone()
        env._two_leg_joint_vel_hist_step = step
    elif reset_mask is not None and reset_mask.any():
        prev_vel = getattr(env, "_two_leg_prev_joint_vel", joint_vel.clone())
        prev_prev_vel = getattr(env, "_two_leg_prev_prev_joint_vel", prev_vel.clone())
        prev_vel = prev_vel.clone()
        prev_prev_vel = prev_prev_vel.clone()
        prev_vel[reset_mask] = joint_vel[reset_mask]
        prev_prev_vel[reset_mask] = joint_vel[reset_mask]
        env._two_leg_prev_joint_vel = prev_vel
        env._two_leg_prev_prev_joint_vel = prev_prev_vel
        if hasattr(env, "_two_leg_prev_joint_vel_hist"):
            curr_vel = env._two_leg_prev_joint_vel_hist.clone()
            curr_vel[reset_mask] = joint_vel[reset_mask]
            env._two_leg_prev_joint_vel_hist = curr_vel
    return getattr(env, "_two_leg_prev_joint_vel", joint_vel.clone())


# =============================================================================
# Torso Upright Rewards
# =============================================================================

def torso_upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for aligning torso Z-axis with world Z-axis (upward).

    Uses projected gravity to measure alignment - when upright,
    projected_gravity[:, 2] should be close to -1.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    alignment = -asset.data.projected_gravity_b[:, 2]  # +1 when torso z-axis points up
    return alignment.clamp(min=0.0)


def torso_upright_soften(
    env: ManagerBasedRLEnv,
    pitch_tolerance: float = 0.35,
    pitch_target: float = -1.22,
    min_height: float = 0.32,
    height_range: float = 0.15,
    front_feet_sensor_cfg: SceneEntityCfg = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Smooth upright reward considering roll, pitch, and height.

    Args:
        env: Environment instance.
        pitch_tolerance: Tolerance for pitch error (squared width).
        pitch_target: Target pitch angle in radians.
        min_height: Minimum height for reward.
        height_range: Height range for scaling reward.
        front_feet_sensor_cfg: Contact sensor config for front feet.
        asset_cfg: Robot asset config.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Orientation gate
    orientation_score = torch.pow(
        _orientation_gate(env, pitch_tolerance ** 2, 0.2, pitch_target, asset_cfg) + 1e-5,
        0.5
    )
    torso_upright_reward = torch.sigmoid(4.0 * (orientation_score - 0.3))

    # Height gate
    base_height = asset.data.root_pos_w[:, 2]
    height_gate = torch.clamp((base_height - min_height) / height_range, min=0.0, max=1.0)

    # Front contact gate (penalize front foot contact)
    if front_feet_sensor_cfg is not None:
        front_contact = _get_front_feet_contact(env, front_feet_sensor_cfg)
        contact_gate = 1.0 - 0.5 * front_contact.float()
    else:
        contact_gate = torch.ones(env.num_envs, device=env.device)

    return torso_upright_reward * height_gate * contact_gate


def torso_upright_warmup(
    env: ManagerBasedRLEnv,
    pitch_tolerance: float = 0.35,
    pitch_target: float = -1.22,
    min_height: float = 0.25,
    height_range: float = 0.25,
    front_feet_sensor_cfg: SceneEntityCfg = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Early-stage upright reward with relaxed constraints.

    Uses larger tolerances than torso_upright_soften for warmup phase.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Relaxed orientation gate (1.5x pitch tolerance, wider roll)
    pitch_width = (pitch_tolerance * 1.5) ** 2
    orientation_score = torch.pow(
        _orientation_gate(env, pitch_width, 0.35, pitch_target, asset_cfg) + 1e-5,
        0.5
    )

    # Height gate (lower threshold)
    base_height = asset.data.root_pos_w[:, 2]
    height_gate = torch.clamp((base_height - min_height) / height_range, min=0.0, max=1.0)

    # Front contact gate (lighter penalty)
    if front_feet_sensor_cfg is not None:
        front_contact = _get_front_feet_contact(env, front_feet_sensor_cfg)
        contact_gate = 1.0 - 0.25 * front_contact.float()
    else:
        contact_gate = torch.ones(env.num_envs, device=env.device)

    return orientation_score * height_gate * contact_gate


def torso_upright_continuous(
    env: ManagerBasedRLEnv,
    pitch_tolerance: float = 0.35,
    pitch_target: float = -1.22,
    min_height: float = 0.32,
    height_range: float = 0.15,
    front_feet_sensor_cfg: SceneEntityCfg = None,
    hind_feet_sensor_cfg: SceneEntityCfg = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stricter upright reward requiring stability while tall.

    Adds hind-foot support requirement, rotation gates, and sway penalties.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Base upright reward
    base_reward = torso_upright_soften(
        env, pitch_tolerance, pitch_target, min_height, height_range,
        front_feet_sensor_cfg, asset_cfg
    )

    # Hind support gate
    if hind_feet_sensor_cfg is not None:
        hind_support = _get_hind_feet_contact(env, hind_feet_sensor_cfg)
        hind_support_gate = hind_support.float()
    else:
        hind_support_gate = torch.ones(env.num_envs, device=env.device)

    # Rotation gate (penalize pitch rate and yaw rate)
    ang_vel = asset.data.root_ang_vel_b
    pitch_rate = torch.abs(ang_vel[:, 0])
    yaw_rate = torch.abs(ang_vel[:, 2])
    rotation_gate = torch.exp(-1.5 * yaw_rate - 0.8 * pitch_rate)

    # Sway gate (penalize lateral motion)
    lin_vel = asset.data.root_lin_vel_b
    body_lin_motion = torch.norm(lin_vel[:, :2], dim=1)
    sway_gate = torch.exp(-1.0 * body_lin_motion)

    return base_reward * hind_support_gate * rotation_gate * sway_gate


# =============================================================================
# Front Legs Up Rewards
# =============================================================================

def front_legs_up(
    env: ManagerBasedRLEnv,
    front_feet_sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary reward: 1 if both front feet are airborne, 0 otherwise."""
    front_contact = _get_front_feet_contact(env, front_feet_sensor_cfg, threshold)
    return (~front_contact).float()


def front_legs_up_warmup(
    env: ManagerBasedRLEnv,
    front_feet_sensor_cfg: SceneEntityCfg,
    hind_feet_sensor_cfg: SceneEntityCfg = None,
    front_feet_body_cfg: SceneEntityCfg = None,
    pitch_tolerance: float = 0.35,
    pitch_target: float = -1.22,
    min_height: float = 0.30,
    height_range: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Transition reward to guide the robot toward stable front-leg lifting.

    Includes duration tracking, velocity gate, orientation gate, and height gate.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Basic clearance
    both_up = front_legs_up(env, front_feet_sensor_cfg)

    # Hind support gate
    if hind_feet_sensor_cfg is not None:
        hind_support = _get_hind_feet_contact(env, hind_feet_sensor_cfg)
        hind_support_gate = hind_support.float()
    else:
        hind_support_gate = torch.ones(env.num_envs, device=env.device)

    # Air-time gate (encourage both front feet to stay airborne briefly)
    contact_sensor: ContactSensor = env.scene.sensors[front_feet_sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor.track_air_time for front_legs_up_warmup.")
    front_air_time = contact_sensor.data.current_air_time[:, front_feet_sensor_cfg.body_ids]
    min_air = torch.min(front_air_time, dim=1).values
    short_target = 0.3
    duration_reward = torch.clamp(min_air / short_target, 0.0, 1.0)

    # Vertical speed gate (discourage tapping)
    if front_feet_body_cfg is not None:
        front_vel = asset.data.body_lin_vel_w[:, front_feet_body_cfg.body_ids, 2]
        vertical_speed = torch.abs(front_vel).mean(dim=1)
    else:
        vertical_speed = torch.zeros(env.num_envs, device=env.device)
    velocity_gate = torch.exp(-2.0 * vertical_speed)

    # Orientation gate (relaxed)
    pitch_width = (pitch_tolerance * 1.7) ** 2
    orientation_gate = _orientation_gate(env, pitch_width, 0.3, pitch_target, asset_cfg)

    # Height gate
    base_height = asset.data.root_pos_w[:, 2]
    height_gate = torch.clamp((base_height - min_height) / height_range, min=0.0, max=1.0)

    # Rotation gate
    ang_vel = asset.data.root_ang_vel_b
    yaw_rate = torch.abs(ang_vel[:, 2])
    rotation_gate = torch.exp(-1.5 * torch.square(yaw_rate))

    base_score = 0.4 + 0.6 * duration_reward
    return both_up * base_score * velocity_gate * orientation_gate * height_gate * hind_support_gate * rotation_gate


def front_legs_up_continuous(
    env: ManagerBasedRLEnv,
    front_feet_sensor_cfg: SceneEntityCfg,
    hind_feet_sensor_cfg: SceneEntityCfg = None,
    front_feet_body_cfg: SceneEntityCfg = None,
    pitch_tolerance: float = 0.35,
    pitch_target: float = -1.22,
    min_height: float = 0.35,
    height_range: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward sustained front-leg clearance with smooth motion.

    Stricter than warmup variant with tighter gates.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Basic clearance
    both_up = front_legs_up(env, front_feet_sensor_cfg)

    # Hind support gate (required)
    if hind_feet_sensor_cfg is not None:
        hind_support = _get_hind_feet_contact(env, hind_feet_sensor_cfg)
        hind_support_gate = hind_support.float()
    else:
        hind_support_gate = torch.ones(env.num_envs, device=env.device)

    # Air-time gate (longer durations)
    contact_sensor: ContactSensor = env.scene.sensors[front_feet_sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor.track_air_time for front_legs_up_continuous.")
    front_air_time = contact_sensor.data.current_air_time[:, front_feet_sensor_cfg.body_ids]
    min_air = torch.min(front_air_time, dim=1).values
    min_duration = 0.2
    target_duration = 0.6
    duration_reward = torch.clamp(
        (min_air - min_duration) / (target_duration - min_duration), min=0.0, max=1.0
    )

    # Vertical speed gate (discourage tapping)
    if front_feet_body_cfg is not None:
        front_vel = asset.data.body_lin_vel_w[:, front_feet_body_cfg.body_ids, 2]
        vertical_speed = torch.abs(front_vel).mean(dim=1)
    else:
        vertical_speed = torch.zeros(env.num_envs, device=env.device)
    velocity_gate = torch.exp(-5.0 * vertical_speed)

    # Orientation gate (strict)
    pitch_width = (pitch_tolerance * 0.7) ** 2
    orientation_gate = _orientation_gate(env, pitch_width, 0.1, pitch_target, asset_cfg)

    # Height gate
    base_height = asset.data.root_pos_w[:, 2]
    height_gate = torch.clamp((base_height - min_height) / height_range, min=0.0, max=1.0)

    # Rotation gate (stricter)
    ang_vel = asset.data.root_ang_vel_b
    yaw_rate = torch.abs(ang_vel[:, 2])
    rotation_gate = torch.exp(-2.0 * torch.square(yaw_rate))

    return both_up * duration_reward * velocity_gate * orientation_gate * height_gate * hind_support_gate * rotation_gate


def front_legs_up_warmup_safe(
    env: ManagerBasedRLEnv,
    front_feet_sensor_cfg: SceneEntityCfg,
    hind_feet_sensor_cfg: SceneEntityCfg = None,
    front_feet_body_cfg: SceneEntityCfg = None,
    pitch_tolerance: float = 0.35,
    pitch_target: float = -1.22,
    min_height: float = 0.30,
    height_range: float = 0.15,
    safe_gate_torque_limits_weight: float = 0.0,
    safe_gate_dof_vel_limits_weight: float = 0.0,
    safe_gate_power_weight: float = 0.0,
    safe_gate_action_weight: float = 0.0,
    torque_soft_limit: float = 0.9,
    dof_vel_soft_limit: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Safe version of front-legs-up warmup reward."""
    base = front_legs_up_warmup(
        env,
        front_feet_sensor_cfg,
        hind_feet_sensor_cfg,
        front_feet_body_cfg,
        pitch_tolerance,
        pitch_target,
        min_height,
        height_range,
        asset_cfg,
    )
    gate = _safety_effort_gate(
        env,
        safe_gate_torque_limits_weight=safe_gate_torque_limits_weight,
        safe_gate_dof_vel_limits_weight=safe_gate_dof_vel_limits_weight,
        safe_gate_power_weight=safe_gate_power_weight,
        safe_gate_action_weight=safe_gate_action_weight,
        torque_soft_limit=torque_soft_limit,
        dof_vel_soft_limit=dof_vel_soft_limit,
        asset_cfg=asset_cfg,
    )
    return base * gate


def front_legs_up_continuous_safe(
    env: ManagerBasedRLEnv,
    front_feet_sensor_cfg: SceneEntityCfg,
    hind_feet_sensor_cfg: SceneEntityCfg = None,
    front_feet_body_cfg: SceneEntityCfg = None,
    pitch_tolerance: float = 0.35,
    pitch_target: float = -1.22,
    min_height: float = 0.35,
    height_range: float = 0.2,
    safe_gate_torque_limits_weight: float = 0.0,
    safe_gate_dof_vel_limits_weight: float = 0.0,
    safe_gate_power_weight: float = 0.0,
    safe_gate_action_weight: float = 0.0,
    torque_soft_limit: float = 0.9,
    dof_vel_soft_limit: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Safe version of front-legs-up continuous reward."""
    base = front_legs_up_continuous(
        env,
        front_feet_sensor_cfg,
        hind_feet_sensor_cfg,
        front_feet_body_cfg,
        pitch_tolerance,
        pitch_target,
        min_height,
        height_range,
        asset_cfg,
    )
    gate = _safety_effort_gate(
        env,
        safe_gate_torque_limits_weight=safe_gate_torque_limits_weight,
        safe_gate_dof_vel_limits_weight=safe_gate_dof_vel_limits_weight,
        safe_gate_power_weight=safe_gate_power_weight,
        safe_gate_action_weight=safe_gate_action_weight,
        torque_soft_limit=torque_soft_limit,
        dof_vel_soft_limit=dof_vel_soft_limit,
        asset_cfg=asset_cfg,
    )
    return base * gate


def front_tap_penalty(
    env: ManagerBasedRLEnv,
    front_feet_sensor_cfg: SceneEntityCfg,
    front_feet_body_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize front feet contact rate and vertical speed."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Contact rate
    contact_sensor: ContactSensor = env.scene.sensors[front_feet_sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    current_forces = net_forces[:, -1, front_feet_sensor_cfg.body_ids]
    is_contact = torch.norm(current_forces, dim=-1) > threshold
    contact_rate = is_contact.float().mean(dim=1)

    # Tap speed (vertical velocity of front feet)
    front_vel = asset.data.body_lin_vel_w[:, front_feet_body_cfg.body_ids, 2]
    tap_speed = torch.abs(front_vel).mean(dim=1)

    return contact_rate + 0.25 * tap_speed


# =============================================================================
# Human Posture Rewards
# =============================================================================

def human_posture(
    env: ManagerBasedRLEnv,
    hind_knee_joint_ids: list[int],
    hind_hip_joint_ids: list[int],
    hind_hip_body_ids: list[int],
    hind_foot_body_ids: list[int],
    hip_targets: list[float] = [-0.2, -0.2],
    hind_feet_sensor_cfg: SceneEntityCfg = None,
    front_feet_sensor_cfg: SceneEntityCfg = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Encourage a human-like upright stance driven by hind-leg posture.

    Combines knee extension, hip target alignment, and geometric extension.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Knee extension reward
    knees = asset.data.joint_pos[:, hind_knee_joint_ids]
    knee_lower = asset.data.soft_joint_pos_limits[:, hind_knee_joint_ids, 0]
    knee_upper = asset.data.soft_joint_pos_limits[:, hind_knee_joint_ids, 1]
    knee_range = torch.clamp(knee_upper - knee_lower, min=1e-6)
    knee_extension = 1.0 - torch.clamp((knees - knee_lower) / knee_range, 0.0, 1.0)

    # Hip alignment reward
    hip_targets_t = torch.tensor(hip_targets, device=env.device, dtype=asset.data.joint_pos.dtype)
    hips = asset.data.joint_pos[:, hind_hip_joint_ids]
    hip_alignment = torch.exp(-torch.square(hips - hip_targets_t) / 0.1)

    # Joint posture combined
    joint_posture_reward = knee_extension.mean(dim=1) * hip_alignment.mean(dim=1)

    # Geometric extension (hind leg length/vertical alignment)
    geom_reward = hind_leg_extension_geom(
        env, hind_hip_body_ids, hind_foot_body_ids,
        hind_feet_sensor_cfg, asset_cfg
    )

    combined_reward = 0.6 * joint_posture_reward + 0.4 * geom_reward

    # Cheat guard (penalize front leg support or large torso lean)
    cheat_guard = _human_posture_guard(env, front_feet_sensor_cfg, asset_cfg)

    # Hind support gate
    if hind_feet_sensor_cfg is not None:
        hind_support = _get_hind_feet_contact(env, hind_feet_sensor_cfg)
        hind_support_gate = hind_support.float()
    else:
        hind_support_gate = torch.ones(env.num_envs, device=env.device)

    # Rotation gate
    ang_vel = asset.data.root_ang_vel_b
    yaw_rate = torch.abs(ang_vel[:, 2])
    rotation_gate = torch.exp(-1.5 * torch.square(yaw_rate))

    return combined_reward * cheat_guard * hind_support_gate * rotation_gate


def _human_posture_guard(
    env: ManagerBasedRLEnv,
    front_feet_sensor_cfg: SceneEntityCfg = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gate posture rewards to prevent exploits like front-leg support."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Front clear gate
    if front_feet_sensor_cfg is not None:
        front_contact = _get_front_feet_contact(env, front_feet_sensor_cfg)
        front_clear_gate = (~front_contact).float()
    else:
        front_clear_gate = torch.ones(env.num_envs, device=env.device)

    # Height gate
    base_height = asset.data.root_pos_w[:, 2]
    height_gate = torch.clamp((base_height - 0.35) / 0.2, min=0.0, max=1.0)

    return front_clear_gate * height_gate


def human_posture_warmup(
    env: ManagerBasedRLEnv,
    hind_knee_joint_ids: list[int],
    hind_hip_body_ids: list[int],
    hind_feet_sensor_cfg: SceneEntityCfg = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Coarse reward to encourage hind-leg extension and torso elevation."""
    asset: Articulation = env.scene[asset_cfg.name]

    # Knee extension
    knees = asset.data.joint_pos[:, hind_knee_joint_ids]
    knee_lower = asset.data.soft_joint_pos_limits[:, hind_knee_joint_ids, 0]
    knee_upper = asset.data.soft_joint_pos_limits[:, hind_knee_joint_ids, 1]
    knee_range = torch.clamp(knee_upper - knee_lower, min=1e-6)
    knee_extension = 1.0 - torch.clamp((knees - knee_lower) / knee_range, 0.0, 1.0)
    knee_score = torch.clamp((knee_extension.mean(dim=1) - 0.2) / 0.6, 0.0, 1.0)

    # Hip height
    if len(hind_hip_body_ids) > 0:
        hind_hip_heights = asset.data.body_pos_w[:, hind_hip_body_ids, 2].mean(dim=1)
    else:
        hind_hip_heights = asset.data.root_pos_w[:, 2]
    height_score = torch.clamp((hind_hip_heights - 0.30) / 0.12, min=0.0, max=1.0)

    # Hind support gate
    if hind_feet_sensor_cfg is not None:
        hind_support = _get_hind_feet_contact(env, hind_feet_sensor_cfg)
        hind_support_gate = hind_support.float()
    else:
        hind_support_gate = torch.ones(env.num_envs, device=env.device)

    base_reward = 0.3 + 0.7 * knee_score * height_score
    return base_reward * hind_support_gate


# =============================================================================
# Hind Leg Rewards
# =============================================================================

def hind_leg_extension_geom(
    env: ManagerBasedRLEnv,
    hind_hip_body_ids: list[int],
    hind_foot_body_ids: list[int],
    hind_feet_sensor_cfg: SceneEntityCfg = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward straight, weight-bearing hind legs with vertical alignment.

    Combines leg length with vertical alignment and soft penalties on
    horizontal splay and angular velocity.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get positions
    hip_pos = asset.data.body_pos_w[:, hind_hip_body_ids, :3]  # (N, 2, 3)
    foot_pos = asset.data.body_pos_w[:, hind_foot_body_ids, :3]  # (N, 2, 3)

    # Hip->foot vectors
    v = foot_pos - hip_pos  # (N, 2, 3)

    # Lengths and normalization
    L = torch.norm(v, dim=2)  # (N, 2)
    length_norm = torch.clamp(L.mean(dim=1) / 0.4, 0.0, 1.0)

    # Vertical alignment (match legged_gym sign convention)
    L_safe = L + 1e-6
    vz = v[:, :, 2]  # (N, 2)
    align = torch.clamp(vz / L_safe, 0.0, 1.0).mean(dim=1)

    # Horizontal splay penalty
    horiz = torch.norm(v[:, :, :2], dim=2).mean(dim=1)
    horiz_gate = torch.exp(-2.0 * torch.square(horiz))

    # Hind support gate
    if hind_feet_sensor_cfg is not None:
        hind_support = _get_hind_feet_contact(env, hind_feet_sensor_cfg)
        hind_support_gate = hind_support.float()
    else:
        hind_support_gate = torch.ones(env.num_envs, device=env.device)

    # Angular velocity gate
    ang_vel = asset.data.root_ang_vel_b
    roll_yaw_speed = torch.norm(ang_vel[:, [0, 2]], dim=1)
    ang_gate = torch.exp(-torch.square(roll_yaw_speed / 1.0))

    # Height gate
    base_height = asset.data.root_pos_w[:, 2]
    height_gate = torch.clamp((base_height - 0.45) / 0.2, min=0.0, max=1.0)

    return length_norm * align * horiz_gate * hind_support_gate * ang_gate * height_gate


def hind_knee_extension(
    env: ManagerBasedRLEnv,
    hind_knee_joint_ids: list[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward robot for extending hind knees (promoting standing posture)."""
    asset: Articulation = env.scene[asset_cfg.name]
    angles = asset.data.joint_pos[:, hind_knee_joint_ids]
    return angles.sum(dim=1)  # Larger angles = more extended


def hind_legs_calmness(
    env: ManagerBasedRLEnv,
    hind_joint_ids: list[int],
    hind_feet_sensor_cfg: SceneEntityCfg = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward hind legs for staying calm: low joint velocities and torques while in contact."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_vel = asset.data.joint_vel[:, hind_joint_ids]
    joint_torque = asset.data.applied_torque[:, hind_joint_ids]

    vel_gate = torch.exp(-0.5 * torch.norm(joint_vel, dim=1))
    torque_gate = torch.exp(-0.2 * torch.norm(joint_torque, dim=1))

    if hind_feet_sensor_cfg is not None:
        hind_support = _get_hind_feet_contact(env, hind_feet_sensor_cfg)
        contact_gate = hind_support.float()
    else:
        contact_gate = torch.ones(env.num_envs, device=env.device)

    return vel_gate * torque_gate * contact_gate


# =============================================================================
# Stand Still Rewards
# =============================================================================

def stand_still(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping the robot still (low joint deviation and body motion)."""
    asset: Articulation = env.scene[asset_cfg.name]

    joint_deviation = torch.mean(
        torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
    )
    body_lin_motion = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_motion = torch.norm(asset.data.root_ang_vel_b[:, [0, 2]], dim=1)

    motion_metric = 1.5 * joint_deviation + body_lin_motion + 0.5 * body_ang_motion
    return torch.exp(-motion_metric)


def stand_still_roll_only(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping roll rate low."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll_rate = torch.abs(asset.data.root_ang_vel_b[:, 0])
    return torch.exp(-1.0 * roll_rate)


def stand_still_yaw_only(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping yaw rate low."""
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_rate = torch.abs(asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-0.5 * yaw_rate)


def stand_still_lin_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping base velocity along x low."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_x = torch.abs(asset.data.root_lin_vel_b[:, 0])
    return torch.exp(-1.0 * lin_x)


def stand_still_lin_y(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping base velocity along y low."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_y = torch.abs(asset.data.root_lin_vel_b[:, 1])
    return torch.exp(-1.0 * lin_y)


def stand_still_lin_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping vertical velocity low (prevents hopping)."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_z = torch.abs(asset.data.root_lin_vel_b[:, 2])
    return torch.exp(-0.8 * lin_z)


# =============================================================================
# Base Height Rewards
# =============================================================================

def base_height_bonus(
    env: ManagerBasedRLEnv,
    min_height: float = 0.55,
    max_height: float = 0.8,
    hind_feet_sensor_cfg: SceneEntityCfg = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bonus reward for reaching target base height."""
    asset: Articulation = env.scene[asset_cfg.name]

    base_height = asset.data.root_pos_w[:, 2]
    height_range = max(max_height - min_height, 1e-4)
    bonus = torch.clamp((base_height - min_height) / height_range, min=0.0, max=1.0)

    if hind_feet_sensor_cfg is not None:
        hind_support = _get_hind_feet_contact(env, hind_feet_sensor_cfg)
        contact_gate = hind_support.float()
    else:
        contact_gate = torch.ones(env.num_envs, device=env.device)

    return bonus * contact_gate


# =============================================================================
# Safety Rewards
# =============================================================================

def deploy_posture_gate(
    env: ManagerBasedRLEnv,
    roll_limit_rad: float = 0.7,
    pitch_limit_rad: float = 1.57,
    margin_deg: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for exceeding deployment posture limits (roll/pitch)."""
    asset: Articulation = env.scene[asset_cfg.name]
    rpy = _get_rpy_from_quat(asset.data.root_quat_w)
    roll = rpy[:, 0]
    pitch = rpy[:, 1]

    if margin_deg:
        margin_rad = math.radians(float(margin_deg))
        roll_limit_rad += margin_rad
        pitch_limit_rad += margin_rad

    roll_excess = (torch.abs(roll) - roll_limit_rad).clamp(min=0.0)
    pitch_excess = (torch.abs(pitch) - pitch_limit_rad).clamp(min=0.0)

    return roll_excess + pitch_excess


def torque_limits(
    env: ManagerBasedRLEnv,
    soft_limit: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize torques exceeding soft limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque
    torque_limits = getattr(asset.data, "joint_effort_limits", None)
    if torque_limits is None:
        torque_limits = getattr(asset.data, "joint_torque_limits", None)
    if torque_limits is None:
        torque_limits = getattr(asset.data, "actuator_effort_limits", None)
    if torque_limits is None:
        return torch.sum(torch.square(torques), dim=1)
    if torque_limits.dim() == 1:
        torque_limits = torque_limits.unsqueeze(0).repeat(torques.shape[0], 1)
    excess = (torch.abs(torques) - torque_limits * soft_limit).clamp(min=0.0)
    return torch.sum(excess, dim=1)


def dof_vel_limits(
    env: ManagerBasedRLEnv,
    soft_limit: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities exceeding soft limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel
    vel_limits = getattr(asset.data, "soft_joint_vel_limits", None)
    if vel_limits is None:
        vel_limits = getattr(asset.data, "joint_vel_limits", None)
    if vel_limits is None:
        return torch.sum(torch.square(vel), dim=1)
    if vel_limits.dim() == 1:
        vel_limits = vel_limits.unsqueeze(0).repeat(vel.shape[0], 1)
    excess = (torch.abs(vel) - vel_limits * soft_limit).clamp(min=0.0, max=1.0)
    return torch.sum(excess, dim=1)


def power(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize instantaneous power consumption."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.applied_torque) * torch.abs(asset.data.joint_vel),
        dim=1
    )


def action_magnitude(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize large action magnitudes."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def action_rate(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize changes in actions."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    return torch.sum(torch.square(action - prev_action), dim=1)


def target_smoothness(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize action jerk (changes in action and second derivative)."""
    action, prev_action, prev_prev_action = _get_action_history(env)
    return torch.sum(
        torch.square(prev_action - action)
        + torch.square(action - 2 * prev_action + prev_prev_action),
        dim=1,
    )


def torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def dof_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def dof_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared joint accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(asset.data, "joint_acc"):
        acc = asset.data.joint_acc
    else:
        prev_vel = _get_joint_vel_history(env, asset.data.joint_vel)
        dt = float(getattr(env, "step_dt", 0.005) or 0.005)
        acc = (prev_vel - asset.data.joint_vel) / max(dt, 1e-6)
    return torch.sum(torch.square(acc), dim=1)


def lin_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize z-axis base linear velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def feet_velocity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot slip velocity while in contact."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    current_forces = net_forces[:, -1, sensor_cfg.body_ids]
    in_contact = torch.norm(current_forces, dim=-1) > 1.0
    foot_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :3]
    slip_speed = torch.norm(foot_vel, dim=-1)
    return torch.sum(slip_speed * in_contact.float(), dim=1)


def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_force: float = 100.0,
) -> torch.Tensor:
    """Penalize high foot contact forces."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    current_forces = net_forces[:, -1, sensor_cfg.body_ids]
    contact_mag = torch.norm(current_forces, dim=-1)
    return torch.sum((contact_mag - max_contact_force).clamp(min=0.0), dim=1)


def collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize collisions on selected bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    current_forces = net_forces[:, -1, sensor_cfg.body_ids]
    in_contact = torch.norm(current_forces, dim=-1) > threshold
    return torch.sum(in_contact.float(), dim=1)


# =============================================================================
# Two-Leg Stability Metric
# =============================================================================

def _safety_effort_gate(
    env: ManagerBasedRLEnv,
    safe_gate_torque_limits_weight: float = 0.0,
    safe_gate_dof_vel_limits_weight: float = 0.0,
    safe_gate_power_weight: float = 0.0,
    safe_gate_action_weight: float = 0.0,
    torque_soft_limit: float = 0.9,
    dof_vel_soft_limit: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gate rewards based on effort/safety metrics (returns [0, 1])."""
    penalty = torch.zeros(env.num_envs, device=env.device)
    if safe_gate_torque_limits_weight > 0.0:
        penalty = penalty + safe_gate_torque_limits_weight * torque_limits(
            env, soft_limit=torque_soft_limit, asset_cfg=asset_cfg
        )
    if safe_gate_dof_vel_limits_weight > 0.0:
        penalty = penalty + safe_gate_dof_vel_limits_weight * dof_vel_limits(
            env, soft_limit=dof_vel_soft_limit, asset_cfg=asset_cfg
        )
    if safe_gate_power_weight > 0.0:
        penalty = penalty + safe_gate_power_weight * power(env, asset_cfg=asset_cfg)
    if safe_gate_action_weight > 0.0:
        penalty = penalty + safe_gate_action_weight * action_magnitude(env)
    return (1.0 / (1.0 + penalty)).clamp(min=0.0, max=1.0)


def two_leg_stand_metric(
    env: ManagerBasedRLEnv,
    front_feet_sensor_cfg: SceneEntityCfg,
    hind_feet_sensor_cfg: SceneEntityCfg,
    front_feet_body_cfg: SceneEntityCfg = None,
    pitch_tolerance: float = 0.35,
    pitch_target: float = -1.22,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute a normalized stability metric for upright two-leg standing.

    Combines: front clear, hind support, orientation, height, linear/angular velocity gates.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Front clear
    front_contact = _get_front_feet_contact(env, front_feet_sensor_cfg)
    front_clear = (~front_contact).float()

    # Hind support
    hind_support = _get_hind_feet_contact(env, hind_feet_sensor_cfg).float()

    # Orientation gate
    pitch_width = (pitch_tolerance * 0.6) ** 2
    orientation_gate = torch.clamp(
        _orientation_gate(env, pitch_width, 0.15, pitch_target, asset_cfg), 0.0, 1.0
    )

    # Height gate
    base_height = asset.data.root_pos_w[:, 2]
    height_gate = torch.clamp((base_height - 0.45) / 0.25, min=0.0, max=1.0)

    # Linear velocity gate
    lin_speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    lin_gate = torch.exp(-torch.square(lin_speed / 0.25))

    # Angular velocity gate
    roll_yaw_speed = torch.norm(asset.data.root_ang_vel_b[:, [0, 2]], dim=1)
    ang_gate = torch.exp(-torch.square(roll_yaw_speed / 0.8))

    # Front-foot tapping gate
    if front_feet_body_cfg is not None:
        front_vel = asset.data.body_lin_vel_w[:, front_feet_body_cfg.body_ids, 2]
        vertical_speed = torch.abs(front_vel).mean(dim=1)
        tapping_gate = torch.exp(-6.0 * vertical_speed)
    else:
        tapping_gate = torch.ones(env.num_envs, device=env.device)

    # Combine
    metric = front_clear * hind_support
    metric = metric * orientation_gate * height_gate
    metric = metric * torch.clamp(lin_gate, 0.0, 1.0) * torch.clamp(ang_gate, 0.0, 1.0)
    metric = metric * torch.clamp(tapping_gate, 0.0, 1.0)

    return torch.clamp(metric, 0.0, 1.0)


def two_leg_stability_safe(
    env: ManagerBasedRLEnv,
    front_feet_sensor_cfg: SceneEntityCfg,
    hind_feet_sensor_cfg: SceneEntityCfg,
    front_feet_body_cfg: SceneEntityCfg = None,
    pitch_tolerance: float = 0.35,
    pitch_target: float = -1.22,
    safe_gate_torque_limits_weight: float = 0.0,
    safe_gate_dof_vel_limits_weight: float = 0.0,
    safe_gate_power_weight: float = 0.0,
    safe_gate_action_weight: float = 0.0,
    torque_soft_limit: float = 0.9,
    dof_vel_soft_limit: float = 0.9,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward two-leg stability only when effort stays inside a safe envelope.

    Multiplies the stability metric by a safety effort gate.
    """
    metric = two_leg_stand_metric(
        env, front_feet_sensor_cfg, hind_feet_sensor_cfg, front_feet_body_cfg,
        pitch_tolerance, pitch_target, asset_cfg
    )

    safety_gate = _safety_effort_gate(
        env,
        safe_gate_torque_limits_weight=safe_gate_torque_limits_weight,
        safe_gate_dof_vel_limits_weight=safe_gate_dof_vel_limits_weight,
        safe_gate_power_weight=safe_gate_power_weight,
        safe_gate_action_weight=safe_gate_action_weight,
        torque_soft_limit=torque_soft_limit,
        dof_vel_soft_limit=dof_vel_soft_limit,
        asset_cfg=asset_cfg,
    )

    return metric * safety_gate
