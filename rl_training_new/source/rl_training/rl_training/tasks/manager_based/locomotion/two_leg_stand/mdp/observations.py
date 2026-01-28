# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Observation helpers for two-leg standing task.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_rpy_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to roll, pitch, yaw (radians)."""
    w = quat[:, 3]
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)


def base_rpy(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base roll, pitch, yaw in radians."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _get_rpy_from_quat(asset.data.root_quat_w)


def joint_pos(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint positions (absolute)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_vel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def _reset_mask(env: "ManagerBasedRLEnv") -> torch.Tensor | None:
    if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
        return env.episode_length_buf == 0
    return None


def _get_joint_pos_history(
    env: "ManagerBasedRLEnv", joint_pos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (t-3, t-2, t-1) joint positions with step-safe caching."""
    step = getattr(env, "common_step_counter", None)
    last_step = getattr(env, "_two_leg_joint_pos_hist_step", None)
    reset_mask = _reset_mask(env)

    last_pos = getattr(env, "_two_leg_last_pos", joint_pos.clone())
    last_last_pos = getattr(env, "_two_leg_last_last_pos", last_pos.clone())
    last_last_last_pos = getattr(env, "_two_leg_last_last_last_pos", last_last_pos.clone())

    if reset_mask is not None and reset_mask.any():
        last_pos = last_pos.clone()
        last_last_pos = last_last_pos.clone()
        last_last_last_pos = last_last_last_pos.clone()
        last_pos[reset_mask] = joint_pos[reset_mask]
        last_last_pos[reset_mask] = joint_pos[reset_mask]
        last_last_last_pos[reset_mask] = joint_pos[reset_mask]

    if last_step != step:
        env._two_leg_joint_pos_hist_snapshot = (
            last_last_last_pos.clone(),
            last_last_pos.clone(),
            last_pos.clone(),
        )
        env._two_leg_last_last_last_pos = last_last_pos.clone()
        env._two_leg_last_last_pos = last_pos.clone()
        env._two_leg_last_pos = joint_pos.clone()
        env._two_leg_joint_pos_hist_step = step

    snapshot = getattr(env, "_two_leg_joint_pos_hist_snapshot", None)
    if snapshot is None or getattr(env, "_two_leg_joint_pos_hist_snapshot_step", None) != step:
        env._two_leg_joint_pos_hist_snapshot = (
            last_last_last_pos.clone(),
            last_last_pos.clone(),
            last_pos.clone(),
        )
        env._two_leg_joint_pos_hist_snapshot_step = step
        snapshot = env._two_leg_joint_pos_hist_snapshot

    return snapshot


def _get_joint_vel_history(
    env: "ManagerBasedRLEnv", joint_vel: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (t-2, t-1) joint velocities with step-safe caching."""
    step = getattr(env, "common_step_counter", None)
    last_step = getattr(env, "_two_leg_joint_vel_hist_step", None)
    reset_mask = _reset_mask(env)

    if last_step != step:
        prev_vel = getattr(env, "_two_leg_prev_joint_vel_hist", joint_vel.clone())
        prev_prev_vel = getattr(env, "_two_leg_prev_joint_vel", prev_vel.clone())

        if reset_mask is not None and reset_mask.any():
            prev_vel = prev_vel.clone()
            prev_prev_vel = prev_prev_vel.clone()
            prev_vel[reset_mask] = joint_vel[reset_mask]
            prev_prev_vel[reset_mask] = joint_vel[reset_mask]

        env._two_leg_joint_vel_hist_snapshot = (prev_prev_vel.clone(), prev_vel.clone())
        env._two_leg_prev_prev_joint_vel = prev_prev_vel.clone()
        env._two_leg_prev_joint_vel = prev_vel.clone()
        env._two_leg_prev_joint_vel_hist = joint_vel.clone()
        env._two_leg_joint_vel_hist_step = step

    snapshot = getattr(env, "_two_leg_joint_vel_hist_snapshot", None)
    if snapshot is None or getattr(env, "_two_leg_joint_vel_hist_snapshot_step", None) != step:
        prev_vel = getattr(env, "_two_leg_prev_joint_vel", joint_vel.clone())
        prev_prev_vel = getattr(env, "_two_leg_prev_prev_joint_vel", prev_vel.clone())
        env._two_leg_joint_vel_hist_snapshot = (prev_prev_vel.clone(), prev_vel.clone())
        env._two_leg_joint_vel_hist_snapshot_step = step
        snapshot = env._two_leg_joint_vel_hist_snapshot

    return snapshot


def _get_action_history(env: "ManagerBasedRLEnv") -> tuple[torch.Tensor, torch.Tensor]:
    """Return (t-2, t-1) actions with step-safe caching."""
    step = getattr(env, "common_step_counter", None)
    last_step = getattr(env, "_two_leg_action_hist_step", None)
    reset_mask = _reset_mask(env)

    prev_action = env.action_manager.prev_action
    prev_prev_action = getattr(env, "_two_leg_prev_prev_action", prev_action.clone())

    if reset_mask is not None and reset_mask.any():
        asset: Articulation = env.scene["robot"]
        reset_actions = asset.data.joint_pos - asset.data.default_joint_pos
        prev_action = prev_action.clone()
        prev_prev_action = prev_prev_action.clone()
        prev_action[reset_mask] = reset_actions[reset_mask]
        prev_prev_action[reset_mask] = reset_actions[reset_mask]

    if last_step != step:
        prev_prev_action = getattr(env, "_two_leg_prev_action_hist", prev_action.clone())
        env._two_leg_prev_prev_action = prev_prev_action
        env._two_leg_prev_action_hist = prev_action.clone()
        env._two_leg_action_hist_step = step

    snapshot = getattr(env, "_two_leg_action_hist_snapshot", None)
    if snapshot is None or getattr(env, "_two_leg_action_hist_snapshot_step", None) != step:
        env._two_leg_action_hist_snapshot = (prev_prev_action.clone(), prev_action.clone())
        env._two_leg_action_hist_snapshot_step = step
        snapshot = env._two_leg_action_hist_snapshot

    return snapshot


def joint_pos_history(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint position history (t-3, t-2, t-1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_val = asset.data.joint_pos[:, asset_cfg.joint_ids]
    last_last_last_pos, last_last_pos, last_pos = _get_joint_pos_history(env, joint_pos_val)
    return torch.cat((last_last_last_pos, last_last_pos, last_pos), dim=-1)


def joint_vel_history(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint velocity history (t-2, t-1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_val = asset.data.joint_vel[:, asset_cfg.joint_ids]
    prev_prev_vel, prev_vel = _get_joint_vel_history(env, joint_vel_val)
    return torch.cat((prev_prev_vel, prev_vel), dim=-1)


def action_history(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Action history (t-2, t-1)."""
    prev_prev_action, prev_action = _get_action_history(env)
    return torch.cat((prev_prev_action, prev_action), dim=-1)


def contact_states(
    env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """Binary contact states per body (1 if in contact)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    if hasattr(contact_sensor.data, "net_forces_w_history"):
        net_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]
    else:
        net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return (torch.linalg.norm(net_forces, dim=-1) > 1.0).to(net_forces.dtype)


def friction_coeffs(
    env: "ManagerBasedRLEnv", repeat: int = 4
) -> torch.Tensor:
    """Friction coefficients per foot (repeated scalar)."""
    friction = getattr(env, "_two_leg_friction_coeffs", None)
    if friction is None:
        asset = None
        try:
            asset = env.scene["robot"]
        except Exception:
            asset = None
        friction = None
        if asset is not None and hasattr(asset, "data"):
            for attr in ("friction_coeffs", "friction", "material_friction"):
                friction = getattr(asset.data, attr, None)
                if friction is not None:
                    break
        if friction is None:
            static_friction = getattr(getattr(env.scene, "terrain", None), "physics_material", None)
            default_value = 1.0 if static_friction is None else static_friction.static_friction
            friction = torch.full((env.scene.num_envs, 1), float(default_value), device=env.device)
    if friction.dim() == 1:
        friction = friction[:, None]
    if friction.shape[1] != 1:
        friction = friction[:, :1]
    return friction.repeat(1, repeat)


def external_wrench(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """External forces and torques applied on the base (6D)."""
    forces = getattr(env, "push_forces", None)
    torques = getattr(env, "push_torques", None)
    if forces is None or torques is None:
        return torch.zeros((env.scene.num_envs, 6), device=env.device)
    if forces.dim() == 3:
        forces = forces[:, 0, :]
    if torques.dim() == 3:
        torques = torques[:, 0, :]
    return torch.cat((forces, torques), dim=-1)


def mass_payload(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base")
) -> torch.Tensor:
    """Payload mass offset (base mass minus nominal)."""
    asset: Articulation = env.scene[asset_cfg.name]
    masses = None
    if hasattr(asset, "data") and hasattr(asset.data, "body_mass"):
        masses = asset.data.body_mass
    elif hasattr(asset, "root_physx_view") and hasattr(asset.root_physx_view, "get_masses"):
        masses = asset.root_physx_view.get_masses()
    if masses is None:
        return torch.zeros((env.scene.num_envs, 1), device=env.device)
    if masses.dim() == 2:
        masses = masses[:, asset_cfg.body_ids]
    else:
        masses = masses[:, asset_cfg.body_ids, 0]
    base_mass = masses[:, :1]
    default_mass = 6.0
    if hasattr(asset, "data") and hasattr(asset.data, "default_body_mass"):
        default_mass = float(asset.data.default_body_mass[0, asset_cfg.body_ids][0].item())
    return base_mass - default_mass


def com_displacement(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base")
) -> torch.Tensor:
    """Center-of-mass displacement for the base body (3D)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if hasattr(asset, "root_physx_view") and hasattr(asset.root_physx_view, "get_coms"):
        coms = asset.root_physx_view.get_coms()
        coms = coms[:, asset_cfg.body_ids, :]
    elif hasattr(asset, "data") and hasattr(asset.data, "body_com"):
        coms = asset.data.body_com[:, asset_cfg.body_ids, :]
    else:
        return torch.zeros((env.scene.num_envs, 3), device=env.device)
    coms = coms.squeeze(1)
    default_com = getattr(asset.data, "default_body_com", None) if hasattr(asset, "data") else None
    if default_com is not None:
        coms = coms - default_com[:, asset_cfg.body_ids, :].squeeze(1)
    return coms


def motor_strength_factors(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*")
) -> torch.Tensor:
    """Motor strength factors (current / default torque limits)."""
    cached = getattr(env, "_motor_strength_factors", None)
    if cached is not None:
        if isinstance(asset_cfg.joint_ids, slice):
            return cached - 1.0
        return cached[:, asset_cfg.joint_ids] - 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    effort = None
    default_effort = None
    if hasattr(asset, "data"):
        for attr in ("joint_effort_limits", "joint_effort_limit", "joint_torque_limits", "joint_torque_limit"):
            effort = getattr(asset.data, attr, None)
            if effort is not None:
                break
        for attr in ("default_joint_effort_limits", "default_joint_effort_limit", "default_joint_torque_limits", "default_joint_torque_limit"):
            default_effort = getattr(asset.data, attr, None)
            if default_effort is not None:
                break
    if effort is None or default_effort is None:
        return torch.zeros((env.scene.num_envs, len(asset_cfg.joint_ids)), device=env.device)
    effort = effort[:, asset_cfg.joint_ids]
    default_effort = default_effort[:, asset_cfg.joint_ids]
    ratio = effort / default_effort.clamp(min=1.0e-6)
    return ratio - 1.0


def kp_factors(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*")
) -> torch.Tensor:
    """Kp factors (current / default stiffness)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(asset, "data"):
        return torch.zeros((env.scene.num_envs, len(asset_cfg.joint_ids)), device=env.device)
    stiffness = getattr(asset.data, "joint_stiffness", None)
    default_stiffness = getattr(asset.data, "default_joint_stiffness", None)
    if stiffness is None or default_stiffness is None:
        return torch.zeros((env.scene.num_envs, len(asset_cfg.joint_ids)), device=env.device)
    stiffness = stiffness[:, asset_cfg.joint_ids]
    default_stiffness = default_stiffness[:, asset_cfg.joint_ids]
    ratio = stiffness / default_stiffness.clamp(min=1.0e-6)
    if getattr(env, "_motor_strength_applied_to_gains", False):
        factors = getattr(env, "_motor_strength_factors", None)
        if factors is not None:
            if isinstance(asset_cfg.joint_ids, slice):
                factor_slice = factors
            else:
                factor_slice = factors[:, asset_cfg.joint_ids]
            ratio = ratio / factor_slice.clamp(min=1.0e-6)
    return ratio - 1.0


def kd_factors(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*")
) -> torch.Tensor:
    """Kd factors (current / default damping)."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(asset, "data"):
        return torch.zeros((env.scene.num_envs, len(asset_cfg.joint_ids)), device=env.device)
    damping = getattr(asset.data, "joint_damping", None)
    default_damping = getattr(asset.data, "default_joint_damping", None)
    if damping is None or default_damping is None:
        return torch.zeros((env.scene.num_envs, len(asset_cfg.joint_ids)), device=env.device)
    damping = damping[:, asset_cfg.joint_ids]
    default_damping = default_damping[:, asset_cfg.joint_ids]
    ratio = damping / default_damping.clamp(min=1.0e-6)
    if getattr(env, "_motor_strength_applied_to_gains", False):
        factors = getattr(env, "_motor_strength_factors", None)
        if factors is not None:
            if isinstance(asset_cfg.joint_ids, slice):
                factor_slice = factors
            else:
                factor_slice = factors[:, asset_cfg.joint_ids]
            ratio = ratio / factor_slice.clamp(min=1.0e-6)
    return ratio - 1.0
