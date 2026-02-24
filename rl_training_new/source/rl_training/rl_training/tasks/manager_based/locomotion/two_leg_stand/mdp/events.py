# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Custom reset events for two-leg standing task.

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _resolve_env_ids(env: "ManagerBasedEnv", env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.scene.num_envs, device=env.device)
    return env_ids.to(device=env.device)


def _resolve_ids(ids, length: int, device: torch.device) -> torch.Tensor:
    if isinstance(ids, slice):
        return torch.arange(length, device=device)
    if torch.is_tensor(ids):
        return ids.to(device=device)
    return torch.tensor(list(ids), device=device, dtype=torch.long)


def _get_attr_first(obj, names: tuple[str, ...]):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _expand_env_dim(value: torch.Tensor | None, num_envs: int) -> torch.Tensor | None:
    if value is None:
        return None
    if value.dim() == 1:
        return value.unsqueeze(0).repeat(num_envs, 1)
    return value


def _get_or_cache_default_tensor(
    env: "ManagerBasedEnv",
    cache_name: str,
    current: torch.Tensor,
    explicit_default: torch.Tensor | None,
) -> torch.Tensor:
    if explicit_default is not None:
        return explicit_default
    cached = getattr(env, cache_name, None)
    if cached is None or cached.shape != current.shape or cached.device != current.device:
        cached = current.clone()
        setattr(env, cache_name, cached)
    return cached


def _clear_push_buffers(env: "ManagerBasedEnv", env_ids: torch.Tensor | None = None) -> None:
    """Zero cached push buffers (all envs or selected env_ids)."""
    push_forces = getattr(env, "push_forces", None)
    push_torques = getattr(env, "push_torques", None)
    if push_forces is None or push_torques is None:
        return
    if env_ids is None:
        push_forces.zero_()
        push_torques.zero_()
        return
    if env_ids.numel() == 0:
        return
    push_forces[env_ids] = 0.0
    push_torques[env_ids] = 0.0


def randomize_gravity(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    gravity_z_range: tuple[float, float] = (-9.81, -9.81),
    gravity_xy_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Randomize world gravity vector.

    Notes:
    - Gravity is a scene-global property (not per-env), so ``env_ids`` is used only
      as a trigger guard for event manager compatibility.
    - Function is defensive across IsaacLab/PhysX API variants.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    gx = float(torch.empty(1, device=env.device).uniform_(float(gravity_xy_range[0]), float(gravity_xy_range[1])).item())
    gy = float(torch.empty(1, device=env.device).uniform_(float(gravity_xy_range[0]), float(gravity_xy_range[1])).item())
    gz = float(torch.empty(1, device=env.device).uniform_(float(gravity_z_range[0]), float(gravity_z_range[1])).item())
    gravity_vec = torch.tensor([gx, gy, gz], device=env.device, dtype=torch.float32)

    sim = getattr(env, "sim", None)
    applied = False
    if sim is not None:
        for fn_name in ("set_gravity", "set_physics_scene_gravity"):
            fn = getattr(sim, fn_name, None)
            if fn is None:
                continue
            try:
                fn((gx, gy, gz))
                applied = True
                break
            except TypeError:
                try:
                    fn(gx, gy, gz)
                    applied = True
                    break
                except Exception:
                    continue
            except Exception:
                continue

        if not applied:
            physics_view = getattr(sim, "physics_sim_view", None)
            if physics_view is None:
                physics_view = getattr(sim, "_physics_sim_view", None)
            if physics_view is not None:
                fn = getattr(physics_view, "set_gravity", None)
                if fn is not None:
                    try:
                        fn((gx, gy, gz))
                        applied = True
                    except TypeError:
                        try:
                            fn(gx, gy, gz)
                            applied = True
                        except Exception:
                            pass
                    except Exception:
                        pass

        cfg = getattr(sim, "cfg", None)
        if cfg is not None and hasattr(cfg, "gravity"):
            try:
                gravity_cfg = getattr(cfg, "gravity")
                if hasattr(gravity_cfg, "x") and hasattr(gravity_cfg, "y") and hasattr(gravity_cfg, "z"):
                    gravity_cfg.x = gx
                    gravity_cfg.y = gy
                    gravity_cfg.z = gz
                else:
                    cfg.gravity = (gx, gy, gz)
            except Exception:
                pass

    env._two_leg_gravity_w = gravity_vec


def randomize_motor_strength(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    strength_range: tuple[float, float] = (0.8, 1.2),
    apply_to_gains: bool = True,
) -> None:
    """Randomize motor strength by scaling effort limits (and optionally gains).

    This mirrors the legacy motor strength scaling used in legged_gym, which
    multiplies actuator output torques by a per-joint factor.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    joint_ids = _resolve_ids(asset_cfg.joint_ids, asset.num_joints, env.device)
    if joint_ids.numel() == 0:
        return

    low, high = float(strength_range[0]), float(strength_range[1])
    factors = torch.empty((env_ids.numel(), joint_ids.numel()), device=env.device)
    factors.uniform_(low, high)

    if not hasattr(env, "_motor_strength_factors") or env._motor_strength_factors.shape != (
        env.scene.num_envs,
        asset.num_joints,
    ):
        env._motor_strength_factors = torch.ones(
            (env.scene.num_envs, asset.num_joints), device=env.device
        )
    env._motor_strength_factors[env_ids[:, None], joint_ids] = factors
    env._motor_strength_applied_to_gains = bool(apply_to_gains)

    effort = _get_attr_first(
        asset.data,
        (
            "joint_effort_limits",
            "joint_effort_limit",
            "joint_torque_limits",
            "joint_torque_limit",
            "actuator_effort_limits",
        ),
    )
    default_effort = _get_attr_first(
        asset.data,
        (
            "default_joint_effort_limits",
            "default_joint_effort_limit",
            "default_joint_torque_limits",
            "default_joint_torque_limit",
        ),
    )

    if effort is not None:
        effort = _expand_env_dim(effort, env.scene.num_envs)
        default_effort = _expand_env_dim(default_effort, env.scene.num_envs)
        default_effort = _get_or_cache_default_tensor(
            env,
            "_motor_default_effort_limits",
            effort,
            default_effort,
        )
        scaled = default_effort[env_ids[:, None], joint_ids] * factors.to(default_effort.dtype)
        effort[env_ids[:, None], joint_ids] = scaled
        for name in (
            "write_joint_effort_limits_to_sim",
            "write_joint_torque_limits_to_sim",
            "write_actuator_effort_limits_to_sim",
        ):
            if hasattr(asset, name):
                getattr(asset, name)(effort, env_ids=env_ids)
                break

    if not apply_to_gains:
        return

    stiffness = getattr(asset.data, "joint_stiffness", None)
    damping = getattr(asset.data, "joint_damping", None)

    if stiffness is not None:
        stiffness = _expand_env_dim(stiffness, env.scene.num_envs)
        default_stiffness = _expand_env_dim(
            getattr(asset.data, "default_joint_stiffness", None),
            env.scene.num_envs,
        )
        base_stiffness = _get_or_cache_default_tensor(
            env,
            "_motor_default_joint_stiffness",
            stiffness,
            default_stiffness,
        )
        stiffness[env_ids[:, None], joint_ids] = (
            base_stiffness[env_ids[:, None], joint_ids] * factors.to(base_stiffness.dtype)
        )

    if damping is not None:
        damping = _expand_env_dim(damping, env.scene.num_envs)
        default_damping = _expand_env_dim(
            getattr(asset.data, "default_joint_damping", None),
            env.scene.num_envs,
        )
        base_damping = _get_or_cache_default_tensor(
            env,
            "_motor_default_joint_damping",
            damping,
            default_damping,
        )
        damping[env_ids[:, None], joint_ids] = (
            base_damping[env_ids[:, None], joint_ids] * factors.to(base_damping.dtype)
        )

    if stiffness is not None and damping is not None and hasattr(asset, "write_joint_gains_to_sim"):
        asset.write_joint_gains_to_sim(stiffness, damping, env_ids=env_ids)


def push_robots(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    max_force: float = 10.0,
    max_torque: float = 10.0,
    max_vel_xy: float = 0.5,
) -> None:
    """Apply random pushes to the robot base (force/torque impulse)."""
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    body_ids = _resolve_ids(asset_cfg.body_ids, asset.num_bodies, env.device)
    num_envs = env.scene.num_envs
    num_bodies = asset.num_bodies

    if not hasattr(env, "push_forces") or env.push_forces.shape != (num_envs, num_bodies, 3):
        env.push_forces = torch.zeros((num_envs, num_bodies, 3), device=env.device)
        env.push_torques = torch.zeros((num_envs, num_bodies, 3), device=env.device)

    env.push_forces.zero_()
    env.push_torques.zero_()

    forces = torch.empty((env_ids.numel(), body_ids.numel(), 3), device=env.device)
    torques = torch.empty((env_ids.numel(), body_ids.numel(), 3), device=env.device)
    forces.uniform_(-max_force, max_force)
    torques.uniform_(-max_torque, max_torque)

    env.push_forces[env_ids[:, None], body_ids] = forces
    env.push_torques[env_ids[:, None], body_ids] = torques
    env._two_leg_last_push_step = getattr(env, "common_step_counter", None)

    applied = False
    view = getattr(asset, "root_physx_view", None)
    if view is not None and hasattr(view, "apply_forces_and_torques"):
        try:
            view.apply_forces_and_torques(env.push_forces, env.push_torques, env_ids)
            applied = True
        except TypeError:
            try:
                view.apply_forces_and_torques(env.push_forces, env.push_torques)
                applied = True
            except Exception:
                applied = False
        except Exception:
            applied = False

    if not applied:
        # Fallback: emulate impulse by nudging base linear velocity.
        vel_xy = torch.empty((env_ids.numel(), 2), device=env.device).uniform_(-max_vel_xy, max_vel_xy)
        asset.data.root_lin_vel_w[env_ids, 0:2] = vel_xy
        asset.write_root_velocity_to_sim(asset.data.root_state_w[:, 7:], env_ids=env_ids)


def reset_to_near_goal_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    near_goal_prob: float = 0.0,
    goal_pos: list[float] = [0.0, 0.0, 0.62],
    goal_rpy_deg: list[float] = [0.0, 65.0, 0.0],
    goal_joint_angles: dict[str, float] = None,
    pos_noise: float = 0.02,
    rot_noise: float = 0.08,
    vel_noise: float = 0.08,
    joint_noise: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset selected environments to a near-goal standing state.

    This is used for curriculum learning where the robot starts closer to the
    goal standing posture to learn fine-tuning behaviors.

    Args:
        env: Environment instance.
        env_ids: Environment indices to reset.
        near_goal_prob: Probability of using near-goal reset (vs default).
        goal_pos: Target base position [x, y, z].
        goal_rpy_deg: Target base orientation [roll, pitch, yaw] in degrees.
        goal_joint_angles: Dict mapping joint names to target angles.
        pos_noise: Position noise magnitude.
        rot_noise: Rotation noise magnitude (radians).
        vel_noise: Velocity noise magnitude.
        joint_noise: Joint angle noise magnitude.
        asset_cfg: Asset configuration.
    """
    if len(env_ids) == 0:
        return
    _clear_push_buffers(env, env_ids)

    asset: Articulation = env.scene[asset_cfg.name]

    # Determine which envs get near-goal reset
    use_near_goal = torch.rand(len(env_ids), device=env.device) < near_goal_prob
    near_goal_ids = env_ids[use_near_goal]

    if len(near_goal_ids) == 0:
        return

    # Convert RPY to quaternion
    rpy_rad = torch.tensor(
        [math.radians(r) for r in goal_rpy_deg],
        device=env.device, dtype=torch.float32
    )
    # Apply position with noise
    pos = torch.tensor(goal_pos, device=env.device, dtype=torch.float32)
    pos_with_noise = pos.unsqueeze(0).repeat(len(near_goal_ids), 1)
    pos_with_noise += torch.randn_like(pos_with_noise) * pos_noise

    # Apply rotation with noise (perturb euler angles then convert)
    rpy_noise = torch.randn(len(near_goal_ids), 3, device=env.device) * rot_noise
    rpy_with_noise = rpy_rad.unsqueeze(0).repeat(len(near_goal_ids), 1) + rpy_noise
    quat_with_noise = quat_from_euler_xyz(
        rpy_with_noise[:, 0],
        rpy_with_noise[:, 1],
        rpy_with_noise[:, 2]
    )

    # Set root state
    asset.data.root_pos_w[near_goal_ids] = pos_with_noise + env.scene.env_origins[near_goal_ids]
    asset.data.root_quat_w[near_goal_ids] = quat_with_noise

    # Apply velocity with noise
    asset.data.root_lin_vel_w[near_goal_ids] = torch.randn(len(near_goal_ids), 3, device=env.device) * vel_noise
    asset.data.root_ang_vel_w[near_goal_ids] = torch.randn(len(near_goal_ids), 3, device=env.device) * vel_noise

    # Apply joint angles if provided
    if goal_joint_angles is not None:
        joint_ids = []
        joint_values = []
        for joint_name, angle in goal_joint_angles.items():
            # Find joint index
            joint_idx = asset.find_joints(joint_name)
            if len(joint_idx[0]) > 0:
                joint_ids.append(joint_idx[0][0])
                joint_values.append(angle)

        if joint_ids:
            joint_ids_t = torch.tensor(joint_ids, device=env.device)
            joint_values_t = torch.tensor(joint_values, device=env.device, dtype=torch.float32)

            # Set joint positions with noise
            for i, jid in enumerate(joint_ids_t):
                noise = torch.randn(len(near_goal_ids), device=env.device) * joint_noise
                asset.data.joint_pos[near_goal_ids, jid] = joint_values_t[i] + noise
                asset.data.joint_vel[near_goal_ids, jid] = torch.randn(len(near_goal_ids), device=env.device) * vel_noise

    # Write to simulation
    asset.write_root_pose_to_sim(asset.data.root_state_w[near_goal_ids, :7], env_ids=near_goal_ids)
    asset.write_root_velocity_to_sim(asset.data.root_state_w[near_goal_ids, 7:], env_ids=near_goal_ids)
    joint_pos = asset.data.joint_pos.clone()
    joint_vel = asset.data.joint_vel.clone()
    asset.write_joint_state_to_sim(
        joint_pos[near_goal_ids],
        joint_vel[near_goal_ids],
        env_ids=near_goal_ids
    )


def reset_to_deploy_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    deploy_prob: float = 1.0,
    deploy_height: float = 0.30,
    deploy_quat_w: list[float] | None = None,
    deploy_joint_angles: dict[str, float] = None,
    add_noise: bool = False,
    pos_noise: float = 0.0,
    joint_noise: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset environments to deployment-aligned state.

    This matches the state used in the deployment C++ stack to ensure
    consistent behavior between training and deployment.

    Args:
        env: Environment instance.
        env_ids: Environment indices to reset.
        deploy_prob: Probability of using deploy reset.
        deploy_height: Base height for deployment.
        deploy_quat_w: Optional root orientation quaternion in (w, x, y, z).
        deploy_joint_angles: Dict mapping joint names to target angles.
        add_noise: Whether to add noise.
        pos_noise: Position noise magnitude.
        joint_noise: Joint angle noise magnitude.
        asset_cfg: Asset configuration.
    """
    if len(env_ids) == 0:
        return
    _clear_push_buffers(env, env_ids)

    asset: Articulation = env.scene[asset_cfg.name]

    # Determine which envs get deploy reset
    use_deploy = torch.rand(len(env_ids), device=env.device) < deploy_prob
    deploy_ids = env_ids[use_deploy]

    if len(deploy_ids) == 0:
        return

    # Set position (flat on ground, identity orientation)
    pos = torch.zeros(len(deploy_ids), 3, device=env.device)
    pos[:, 2] = deploy_height
    if add_noise:
        pos += torch.randn_like(pos) * pos_noise

    # Root orientation (w, x, y, z)
    if deploy_quat_w is not None:
        quat = torch.tensor(deploy_quat_w, device=env.device, dtype=asset.data.root_quat_w.dtype)
        quat = quat.unsqueeze(0).repeat(len(deploy_ids), 1)
    else:
        quat = torch.zeros(len(deploy_ids), 4, device=env.device, dtype=asset.data.root_quat_w.dtype)
        quat[:, 0] = 1.0  # w = 1

    # Set root state
    asset.data.root_pos_w[deploy_ids] = pos + env.scene.env_origins[deploy_ids]
    asset.data.root_quat_w[deploy_ids] = quat
    asset.data.root_lin_vel_w[deploy_ids] = torch.zeros(len(deploy_ids), 3, device=env.device)
    asset.data.root_ang_vel_w[deploy_ids] = torch.zeros(len(deploy_ids), 3, device=env.device)

    # Apply joint angles if provided
    if deploy_joint_angles is not None:
        for joint_name, angle in deploy_joint_angles.items():
            joint_idx = asset.find_joints(joint_name)
            if len(joint_idx[0]) > 0:
                jid = joint_idx[0][0]
                if add_noise:
                    noise = torch.randn(len(deploy_ids), device=env.device) * joint_noise
                    asset.data.joint_pos[deploy_ids, jid] = angle + noise
                else:
                    asset.data.joint_pos[deploy_ids, jid] = angle
                asset.data.joint_vel[deploy_ids, jid] = 0.0

    # Write to simulation (subset only to avoid shape mismatch)
    asset.write_root_pose_to_sim(asset.data.root_state_w[deploy_ids, :7], env_ids=deploy_ids)
    asset.write_root_velocity_to_sim(asset.data.root_state_w[deploy_ids, 7:], env_ids=deploy_ids)
    # Clone to avoid overlapping memory writes when selecting env_ids
    joint_pos = asset.data.joint_pos.clone()
    joint_vel = asset.data.joint_vel.clone()
    # Write only the subset for these envs
    asset.write_joint_state_to_sim(
        joint_pos[deploy_ids],
        joint_vel[deploy_ids],
        env_ids=deploy_ids
    )


def bad_orientation_two_leg(
    env: ManagerBasedEnv,
    roll_limit_rad: float = 0.7,  # ~40 degrees
    pitch_limit_rad: float = 1.57,  # ~90 degrees
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Termination condition: check if robot orientation exceeds limits.

    For two-leg standing, we allow large pitch (lean back) but limit roll.

    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get RPY from quaternion
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * math.pi / 2,
        torch.asin(sinp.clamp(-1, 1))
    )

    # Check limits
    roll_exceeded = torch.abs(roll) > roll_limit_rad
    pitch_exceeded = torch.abs(pitch) > pitch_limit_rad

    return roll_exceeded | pitch_exceeded


def hind_feet_off(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    min_steps_after_reset: int = 2,
) -> torch.Tensor:
    """Termination condition: hind feet lose contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    current_forces = net_forces[:, -1, sensor_cfg.body_ids]
    is_contact = torch.norm(current_forces, dim=-1) > threshold
    lost_contact = ~torch.all(is_contact, dim=1)
    # Contact sensors can report transient zeros right after reset.
    # Add a short grace window to avoid immediate length-1 episodes.
    if min_steps_after_reset > 0 and hasattr(env, "episode_length_buf"):
        try:
            grace = env.episode_length_buf <= int(min_steps_after_reset)
            lost_contact = lost_contact & (~grace)
        except Exception:
            pass
    return lost_contact


def front_touch_termination(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Termination condition when front feet touch and termination is enabled."""
    if not getattr(env, "front_touch_termination_active", False):
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    current_forces = net_forces[:, -1, sensor_cfg.body_ids]
    is_contact = torch.norm(current_forces, dim=-1) > threshold
    return torch.any(is_contact, dim=1)
