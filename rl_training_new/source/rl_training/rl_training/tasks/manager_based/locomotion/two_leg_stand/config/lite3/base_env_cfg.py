# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Lite3 two-leg standing environment configuration.

from __future__ import annotations

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from rl_training.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG
from rl_training.tasks.manager_based.locomotion.two_leg_stand import (
    TwoLegStandEnvCfg,
    TwoLegStandRewardsCfg,
    TwoLegStandEventCfg,
    mdp,
)


@configclass
class Lite3TwoLegStandEnvCfg(TwoLegStandEnvCfg):
    """Lite3-specific two-leg standing configuration."""

    def __post_init__(self):
        # Call parent post_init first
        super().__post_init__()

        # Set robot asset
        self.scene.robot = DEEPROBOTICS_LITE3_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=DEEPROBOTICS_LITE3_CFG.init_state.replace(
                pos=(0.0, 0.0, 0.32),
                rot=(
                    0.9999929146412841,
                    -0.00023085526184233324,
                    -0.0032073138974974646,
                    -0.0019571690372445424,
                ),
                joint_pos={
                    "FL_HipX_joint": -0.015,
                    "FR_HipX_joint": 0.016,
                    "HL_HipX_joint": -0.022,
                    "HR_HipX_joint": 0.022,
                    "FL_HipY_joint": -0.77,
                    "FR_HipY_joint": -0.77,
                    "HL_HipY_joint": -0.77,
                    "HR_HipY_joint": -0.77,
                    "FL_Knee_joint": 1.54,
                    "FR_Knee_joint": 1.54,
                    "HL_Knee_joint": 1.55,
                    "HR_Knee_joint": 1.55,
                },
            ),
            actuators={
                "Hip": DEEPROBOTICS_LITE3_CFG.actuators["Hip"].replace(
                    stiffness=20.0,
                    damping=0.7,
                ),
                "Knee": DEEPROBOTICS_LITE3_CFG.actuators["Knee"].replace(
                    stiffness=20.0,
                    damping=0.7,
                ),
            },
        )

        # Update action scale for standing (smaller for precision)
        self.actions.joint_pos.scale = 0.25

        # Episode length for standing task
        self.episode_length_s = 5.0

        # === Configure reward weights for basic two-leg standing ===
        # Torso upright rewards
        self.rewards.torso_upright.weight = 4.5
        self.rewards.torso_upright_soften.weight = 1.8
        self.rewards.torso_upright_warmup.weight = 1.8
        self.rewards.torso_upright_continuous.weight = 7.0

        # Front legs up rewards
        self.rewards.front_legs_up.weight = 2.5
        self.rewards.front_legs_up_warmup.weight = 3.0
        self.rewards.front_legs_up_continuous.weight = 5.5
        self.rewards.front_tap_penalty.weight = -3.0

        # Posture rewards
        self.rewards.human_posture.weight = 5.0
        self.rewards.human_posture_warmup.weight = 1.8
        self.rewards.hind_leg_extension_geom.weight = 1.0
        self.rewards.hind_knee_extension.weight = 0.5

        # Stand still rewards
        self.rewards.stand_still.weight = 0.15

        # Base height
        self.rewards.base_height_bonus.weight = 1.5
        self.rewards.base_height_bonus.params["min_height"] = 0.45
        self.rewards.base_height_bonus.params["max_height"] = 0.75

        # Collision penalty on thigh/shank (matches legged_gym penalize_contacts_on)
        self.rewards.collision.weight = -1.0
        self.rewards.collision.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[".*THIGH.*", ".*SHANK.*"]
        )

        # Termination
        self.rewards.is_terminated.weight = -10.0

        # Update termination body names for Lite3
        self.terminations.illegal_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["TORSO"]
        )
        # Disable hind-contact termination unless explicitly enabled (safe config).
        self.terminations.hind_contact = None

        # Update foot names in reward params for Lite3
        self._update_foot_sensor_configs()

    def _update_foot_sensor_configs(self):
        """Update all foot-related sensor configurations for Lite3."""
        front_feet_cfg = SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"])
        hind_feet_cfg = SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"])
        front_feet_body_cfg = SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"])

        # Update torso upright rewards
        self.rewards.torso_upright_soften.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.torso_upright_warmup.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.torso_upright_continuous.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.torso_upright_continuous.params["hind_feet_sensor_cfg"] = hind_feet_cfg

        # Update front legs up rewards
        self.rewards.front_legs_up.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.front_legs_up_warmup.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.front_legs_up_warmup.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.front_legs_up_warmup.params["front_feet_body_cfg"] = front_feet_body_cfg
        self.rewards.front_legs_up_continuous.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.front_legs_up_continuous.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.front_legs_up_continuous.params["front_feet_body_cfg"] = front_feet_body_cfg
        self.rewards.front_legs_up_warmup_safe.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.front_legs_up_warmup_safe.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.front_legs_up_warmup_safe.params["front_feet_body_cfg"] = front_feet_body_cfg
        self.rewards.front_legs_up_continuous_safe.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.front_legs_up_continuous_safe.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.front_legs_up_continuous_safe.params["front_feet_body_cfg"] = front_feet_body_cfg
        self.rewards.front_tap_penalty.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.front_tap_penalty.params["front_feet_body_cfg"] = front_feet_body_cfg

        # Update human posture rewards
        self.rewards.human_posture.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.human_posture.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.human_posture_warmup.params["hind_feet_sensor_cfg"] = hind_feet_cfg

        # Update hind leg rewards
        self.rewards.hind_leg_extension_geom.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.hind_legs_calmness.params["hind_feet_sensor_cfg"] = hind_feet_cfg

        # Update base height bonus
        self.rewards.base_height_bonus.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.base_height_bonus.params["front_feet_sensor_cfg"] = front_feet_cfg

        # Update stability metric
        self.rewards.two_leg_stand_metric.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.two_leg_stand_metric.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.two_leg_stand_metric.params["front_feet_body_cfg"] = front_feet_body_cfg
        self.rewards.two_leg_stability_safe.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.two_leg_stability_safe.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.two_leg_stability_safe.params["front_feet_body_cfg"] = front_feet_body_cfg
        self.rewards.two_leg_state_hold_bonus.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.two_leg_state_hold_bonus.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.two_leg_state_hold_bonus.params["front_feet_body_cfg"] = front_feet_body_cfg
        self.rewards.transition_dynamics_penalty.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.transition_dynamics_penalty.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.transition_dynamics_penalty.params["front_feet_body_cfg"] = front_feet_body_cfg
        self.rewards.fall_after_stand_penalty.params["front_feet_sensor_cfg"] = front_feet_cfg
        self.rewards.fall_after_stand_penalty.params["hind_feet_sensor_cfg"] = hind_feet_cfg
        self.rewards.fall_after_stand_penalty.params["front_feet_body_cfg"] = front_feet_body_cfg

        # Update feet penalties
        self.rewards.feet_velocity.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        )
        self.rewards.feet_contact_forces.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
        )


@configclass
class Lite3TwoLegStandStillEnvCfg(Lite3TwoLegStandEnvCfg):
    """Lite3 two-leg standing with stillness focus."""

    def __post_init__(self):
        super().__post_init__()

        # Use stillness-focused curriculum phases
        self.curriculum.phases.params["phases"] = mdp.get_two_leg_stand_still_phases()

        # Additional stillness rewards
        self.rewards.stand_still_roll_only.weight = 0.05
        self.rewards.stand_still_yaw_only.weight = 0.05
        self.rewards.hind_legs_calmness.weight = 0.2
        self.rewards.feet_velocity.weight = -0.05

        # Safety gate
        self.rewards.deploy_posture_gate.weight = -5.0
        self.rewards.deploy_posture_gate.params["margin_deg"] = 0.5

        # Penalty terms
        self.rewards.action_rate.weight = -0.01


@configclass
class Lite3TwoLegStandStillV2EnvCfg(Lite3TwoLegStandStillEnvCfg):
    """Lite3 two-leg standing with extended stillness curriculum."""

    def __post_init__(self):
        super().__post_init__()

        # Use still-v2 curriculum phases
        self.curriculum.phases.params["phases"] = mdp.get_two_leg_stand_still_v2_phases()
        self.curriculum.phases.params["front_touch_termination"] = {
            "enabled": False,
            "metrics": {"two_leg_stability": 0.75},
            "log_enable": True,
        }

        # Ensure velocity-only stillness terms exist for curriculum use
        self.rewards.stand_still_lin_x.weight = 0.05
        self.rewards.stand_still_lin_y.weight = 0.05
        self.rewards.stand_still_lin_z.weight = 0.05
        self.rewards.dof_vel.weight = -1e-6


@configclass
class Lite3TwoLegStandSafeEnvCfg(Lite3TwoLegStandStillV2EnvCfg):
    """Lite3 two-leg standing with safety constraints."""

    def __post_init__(self):
        super().__post_init__()

        # Longer episode for safe training
        self.episode_length_s = 8.0

        # Base safety scales (match legged_gym defaults so curriculum can override)
        self.rewards.torques.weight = -1e-5
        self.rewards.dof_vel.weight = -1e-6
        self.rewards.dof_acc.weight = -2.5e-7
        self.rewards.action_rate.weight = -0.01
        self.rewards.target_smoothness.weight = -1e-6
        self.rewards.action_magnitude.weight = -1e-6
        self.rewards.ang_vel_xy.weight = -1e-6
        self.rewards.lin_vel_z.weight = -1e-6
        self.rewards.front_legs_up_warmup_safe.weight = 1e-6
        self.rewards.front_legs_up_continuous_safe.weight = 1e-6
        self.rewards.two_leg_stability_safe.weight = 1e-6
        self.rewards.torque_limits.weight = -1e-3
        self.rewards.dof_vel_limits.weight = -1e-3
        self.rewards.power.weight = -1e-3
        self.rewards.feet_contact_forces.weight = -1e-3
        self.rewards.feet_velocity.weight = -0.05

        # Safety gate weights + soft limits
        self.rewards.two_leg_stability_safe.params.update(
            {
                "safe_gate_torque_limits_weight": 0.05,
                "safe_gate_dof_vel_limits_weight": 0.05,
                "safe_gate_power_weight": 5e-4,
                "safe_gate_action_weight": 0.02,
                "torque_soft_limit": 0.7,
                "dof_vel_soft_limit": 0.7,
            }
        )
        self.rewards.front_legs_up_warmup_safe.params.update(
            {
                "safe_gate_torque_limits_weight": 0.05,
                "safe_gate_dof_vel_limits_weight": 0.05,
                "safe_gate_power_weight": 5e-4,
                "safe_gate_action_weight": 0.02,
                "torque_soft_limit": 0.7,
                "dof_vel_soft_limit": 0.7,
            }
        )
        self.rewards.front_legs_up_continuous_safe.params.update(
            {
                "safe_gate_torque_limits_weight": 0.05,
                "safe_gate_dof_vel_limits_weight": 0.05,
                "safe_gate_power_weight": 5e-4,
                "safe_gate_action_weight": 0.02,
                "torque_soft_limit": 0.7,
                "dof_vel_soft_limit": 0.7,
            }
        )
        self.rewards.torque_limits.params["soft_limit"] = 0.7
        self.rewards.dof_vel_limits.params["soft_limit"] = 0.7

        # Use safety-focused curriculum phases
        self.curriculum.phases.params["phases"] = mdp.get_two_leg_stand_safe_phases()
        self.curriculum.phases.params["front_touch_termination"] = {
            "enabled": False,
            "metrics": {"two_leg_stability": 0.75},
            "log_enable": True,
        }

        # Require hind contact for termination
        self.terminations.hind_contact = DoneTerm(
            func=mdp.hind_feet_off,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
                "threshold": 1.0,
                "min_steps_after_reset": 2,
            },
        )
        self.rewards.feet_contact_forces.params["max_contact_force"] = 80.0

        # Update termination limits to be stricter
        self.terminations.bad_orientation.params["roll_limit_rad"] = math.radians(40.0)
        self.terminations.bad_orientation.params["pitch_limit_rad"] = math.radians(90.0)


@configclass
class Lite3TwoLegStandSafeSlowLowPowerEnvCfg(Lite3TwoLegStandSafeEnvCfg):
    """Isolated task scaffold for safe/slow/low-power reward development.

    Keeps the existing safe curriculum behavior while disabling positive-only
    reward clipping so negative penalties fully affect optimization.
    """

    def __post_init__(self):
        super().__post_init__()
        # Ensure penalty terms are not clipped at zero.
        self.only_positive_rewards = False
        # Use isolated exploration-first -> strict-late curriculum.
        self.curriculum.phases.params["phases"] = mdp.get_two_leg_stand_safe_slow_low_power_phases()
        self.curriculum.phases.params["steps_per_env"] = 24
        self.curriculum.phases.params["front_touch_termination"] = {
            "enabled": False,
            "metrics": {"two_leg_stability": 0.75},
            "log_enable": True,
        }


@configclass
class Lite3TwoLegStandRobustEnvCfg(Lite3TwoLegStandSafeEnvCfg):
    """Lite3 two-leg standing with strong randomization and perturbations."""

    def __post_init__(self):
        super().__post_init__()

        # Longer horizons improve disturbance recovery behavior.
        self.episode_length_s = 10.0

        # Use robust curriculum with lower near-goal dependence.
        self.curriculum.phases.params["phases"] = mdp.get_two_leg_stand_robust_phases()
        self.curriculum.phases.params["steps_per_env"] = 24
        self.curriculum.phases.params["front_touch_termination"] = {
            "enabled": False,
            "metrics": {"two_leg_stability": 0.8},
            "log_enable": True,
        }

        # Slightly smaller action scale helps survive stronger pushes.
        self.actions.joint_pos.scale = 0.2

        # Stronger observation corruption for sim-to-real robustness.
        self.observations.policy.enable_corruption = True
        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.35, n_max=0.35)
        self.observations.policy.joint_pos.noise = Unoise(n_min=-0.02, n_max=0.02)
        self.observations.policy.joint_vel.noise = Unoise(n_min=-2.0, n_max=2.0)

        # Wider dynamics randomization to cover coefficient mismatch on hardware.
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.05, 1.6)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.05, 1.4)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.6)
        self.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (-2.0, 4.0)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.08, 0.04),
            "y": (-0.05, 0.05),
            "z": (-0.05, 0.05),
        }

        # Re-randomize gains and motor strength every reset.
        self.events.randomize_actuator_gains.mode = "reset"
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.6, 1.45)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.6, 1.45)
        self.events.randomize_motor_strength.mode = "reset"
        self.events.randomize_motor_strength.params["strength_range"] = (0.55, 1.45)
        self.events.randomize_motor_strength.params["apply_to_gains"] = True

        # Harder reset randomization for recovery policy learning.
        self.events.randomize_reset_joints.params["position_range"] = (0.35, 1.75)
        self.events.randomize_reset_joints.params["velocity_range"] = (-1.0, 1.0)
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.08, 0.08),
            "y": (-0.08, 0.08),
            "z": (-0.02, 0.02),
            "roll": (-0.25, 0.25),
            "pitch": (-0.35, 0.25),
            "yaw": (-0.6, 0.6),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.9, 0.9),
            "y": (-0.9, 0.9),
            "z": (-0.8, 0.8),
            "roll": (-1.0, 1.0),
            "pitch": (-1.0, 1.0),
            "yaw": (-1.2, 1.2),
        }

        # Frequent and stronger interval pushes.
        self.events.randomize_push_robot.interval_range_s = (2.0, 5.0)
        self.events.randomize_push_robot.params["max_force"] = 26.0
        self.events.randomize_push_robot.params["max_torque"] = 15.0
        self.events.randomize_push_robot.params["max_vel_xy"] = 1.2

        # Keep near-goal reset but with larger reset noise for robustness.
        self.events.reset_to_near_goal.params["near_goal_prob"] = 0.0
        self.events.reset_to_near_goal.params["pos_noise"] = 0.05
        self.events.reset_to_near_goal.params["rot_noise"] = 0.15
        self.events.reset_to_near_goal.params["vel_noise"] = 0.2
        self.events.reset_to_near_goal.params["joint_noise"] = 0.15


@configclass
class Lite3TwoLegStandDeployAlignedEnvCfg(Lite3TwoLegStandSafeEnvCfg):
    """Lite3 two-leg standing aligned with deployment reset states."""

    def __post_init__(self):
        super().__post_init__()

        # Calculate deploy-aligned joint angles using inverse kinematics
        deploy_height = 0.30
        thigh_len = 0.20
        shank_len = 0.21

        # Inverse kinematics for standing height
        cos_hipy = (thigh_len**2 + deploy_height**2 - shank_len**2) / (2.0 * deploy_height * thigh_len)
        cos_hipy = max(-1.0, min(1.0, cos_hipy))
        hipy = -math.acos(cos_hipy)

        cos_knee = (thigh_len**2 + shank_len**2 - deploy_height**2) / (2.0 * thigh_len * shank_len)
        cos_knee = max(-1.0, min(1.0, cos_knee))
        knee = math.pi - math.acos(cos_knee)

        # Update robot initial state to deployment position
        self.scene.robot = self.scene.robot.replace(
            init_state=self.scene.robot.init_state.replace(
                pos=(0.0, 0.0, deploy_height),
                rot=(
                    0.9999929146412841,
                    -0.00023085526184233324,
                    -0.0032073138974974646,
                    -0.0019571690372445424,
                ),
                joint_pos={
                    "FL_HipX_joint": 0.0,
                    "FR_HipX_joint": 0.0,
                    "HL_HipX_joint": 0.0,
                    "HR_HipX_joint": 0.0,
                    "FL_HipY_joint": hipy,
                    "FR_HipY_joint": hipy,
                    "HL_HipY_joint": hipy,
                    "HR_HipY_joint": hipy,
                    "FL_Knee_joint": knee,
                    "FR_Knee_joint": knee,
                    "HL_Knee_joint": knee,
                    "HR_Knee_joint": knee,
                },
            ),
        )

        # Disable domain randomization for deployment consistency
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_motor_strength = None
        self.events.randomize_push_robot = None
        self.events.reset_to_near_goal.params["near_goal_prob"] = 0.0
        self.observations.policy.enable_corruption = False

        # Minimal reset noise
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)

        self.events.reset_to_deploy.params["deploy_prob"] = 1.0
        self.events.reset_to_deploy.params["deploy_height"] = deploy_height
        self.events.reset_to_deploy.params["deploy_quat_w"] = (
            0.9999929146412841,
            -0.00023085526184233324,
            -0.0032073138974974646,
            -0.0019571690372445424,
        )
        self.events.reset_to_deploy.params["deploy_joint_angles"] = {
            "FL_HipX_joint": 0.0,
            "FR_HipX_joint": 0.0,
            "HL_HipX_joint": 0.0,
            "HR_HipX_joint": 0.0,
            "FL_HipY_joint": hipy,
            "FR_HipY_joint": hipy,
            "HL_HipY_joint": hipy,
            "HR_HipY_joint": hipy,
            "FL_Knee_joint": knee,
            "FR_Knee_joint": knee,
            "HL_Knee_joint": knee,
            "HR_Knee_joint": knee,
        }
        self.events.reset_to_deploy.params["add_noise"] = False


@configclass
class Lite3TwoLegStandDeployR1EnvCfg(Lite3TwoLegStandDeployAlignedEnvCfg):
    """Lite3 two-leg standing curriculum matching Lite3_rl_training deploy/r1."""

    def __post_init__(self):
        super().__post_init__()

        # Match deploy/r1 curriculum phases
        self.curriculum.phases.params["phases"] = mdp.get_two_leg_stand_deploy_r1_phases()
        self.curriculum.phases.params["steps_per_env"] = 24
        self.curriculum.phases.params["front_touch_termination"] = {
            "enabled": False,
            "metrics": {"two_leg_stability": 0.75},
            "log_enable": True,
        }

        # Scene / sim parameters
        self.scene.env_spacing = 3.0
        self.episode_length_s = 8.0
        self.sim.dt = 0.003
        self.sim.substeps = 1
        self.sim.physx.max_velocity_iteration_count = 0
        self.scene.terrain.physics_material.restitution = 0.0
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Explicit init state to match r1
        self.scene.robot = self.scene.robot.replace(
            init_state=self.scene.robot.init_state.replace(
                pos=(0.0, 0.0, 0.32),
                rot=(
                    0.9999929146412841,
                    -0.00023085526184233324,
                    -0.0032073138974974646,
                    -0.0019571690372445424,
                ),
                joint_pos={
                    "FL_HipX_joint": -0.0154048,
                    "FR_HipX_joint": 0.0159887,
                    "HL_HipX_joint": -0.0221317,
                    "HR_HipX_joint": 0.0224431,
                    "FL_HipY_joint": -0.76697,
                    "FR_HipY_joint": -0.768286,
                    "HL_HipY_joint": -0.765865,
                    "HR_HipY_joint": -0.767203,
                    "FL_Knee_joint": 1.53761,
                    "FR_Knee_joint": 1.53636,
                    "HL_Knee_joint": 1.54788,
                    "HR_Knee_joint": 1.54679,
                },
            ),
        )

        # Deployment reset configuration (match r1)
        self.events.reset_to_deploy.params["deploy_prob"] = 1.0
        self.events.reset_to_deploy.params["deploy_height"] = 0.30
        self.events.reset_to_deploy.params["deploy_quat_w"] = (
            0.9999929146412841,
            -0.00023085526184233324,
            -0.0032073138974974646,
            -0.0019571690372445424,
        )
        self.events.reset_to_deploy.params["deploy_joint_angles"] = {
            "FL_HipX_joint": 0.0,
            "FR_HipX_joint": 0.0,
            "HL_HipX_joint": 0.0,
            "HR_HipX_joint": 0.0,
            "FL_HipY_joint": -0.7729795255029084,
            "FR_HipY_joint": -0.7729795255029084,
            "HL_HipY_joint": -0.7729795255029084,
            "HR_HipY_joint": -0.7729795255029084,
            "FL_Knee_joint": 1.5005003509817765,
            "FR_Knee_joint": 1.5005003509817765,
            "HL_Knee_joint": 1.5005003509817765,
            "HR_Knee_joint": 1.5005003509817765,
        }
        self.events.reset_to_deploy.params["add_noise"] = False
        self.events.reset_to_near_goal.params["near_goal_prob"] = 0.0

        # Disable domain randomization for deploy/r1
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_motor_strength = None
        self.events.randomize_push_robot = None
        self.observations.policy.enable_corruption = False

        # Minimal reset noise
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)

        # Reward parameter tuning to match r1 config
        pitch_target = -1.2217304763960306
        pitch_tolerance = 0.3839724354387525
        for term in (
            self.rewards.torso_upright_soften,
            self.rewards.torso_upright_warmup,
            self.rewards.torso_upright_continuous,
            self.rewards.front_legs_up_warmup,
            self.rewards.front_legs_up_continuous,
            self.rewards.front_legs_up_warmup_safe,
            self.rewards.front_legs_up_continuous_safe,
        ):
            term.params["pitch_target"] = pitch_target
            term.params["pitch_tolerance"] = pitch_tolerance

        self.rewards.base_height_bonus.params["min_height"] = 0.45
        self.rewards.base_height_bonus.params["max_height"] = 0.75
        self.rewards.feet_contact_forces.params["max_contact_force"] = 80.0
        self.rewards.deploy_posture_gate.params["margin_deg"] = 0.5
        self.rewards.deploy_posture_gate.params["roll_limit_rad"] = math.radians(40.0)
        self.rewards.deploy_posture_gate.params["pitch_limit_rad"] = math.radians(90.0)

        # Base reward scales from deploy/r1 config (curriculum overrides per phase)
        self.rewards.torso_upright.weight = 4.5
        self.rewards.torso_upright_soften.weight = 1.8
        self.rewards.torso_upright_warmup.weight = 1.8
        self.rewards.torso_upright_continuous.weight = 7.0
        self.rewards.front_legs_up.weight = 2.5
        self.rewards.front_legs_up_warmup.weight = 3.0
        self.rewards.front_legs_up_continuous.weight = 5.5
        self.rewards.front_legs_up_warmup_safe.weight = 1.0e-6
        self.rewards.front_legs_up_continuous_safe.weight = 1.0e-6
        self.rewards.front_tap_penalty.weight = -3.0
        self.rewards.human_posture.weight = 5.0
        self.rewards.human_posture_warmup.weight = 1.8
        self.rewards.hind_leg_extension_geom.weight = 1.0
        self.rewards.hind_knee_extension.weight = 0.5
        self.rewards.hind_legs_calmness.weight = 0.2
        self.rewards.stand_still.weight = 0.15
        self.rewards.stand_still_roll_only.weight = 0.05
        self.rewards.stand_still_yaw_only.weight = 0.05
        self.rewards.stand_still_lin_x.weight = 0.05
        self.rewards.stand_still_lin_y.weight = 0.05
        self.rewards.stand_still_lin_z.weight = 0.05
        self.rewards.base_height_bonus.weight = 1.5
        self.rewards.lin_vel_z.weight = -1.0e-6
        self.rewards.ang_vel_xy.weight = -1.0e-6
        self.rewards.action_rate.weight = -1.0e-2
        self.rewards.action_magnitude.weight = -1.0e-6
        self.rewards.target_smoothness.weight = -1.0e-6
        self.rewards.dof_vel.weight = -1.0e-6
        self.rewards.dof_acc.weight = -2.5e-7
        self.rewards.power.weight = -1.0e-3
        self.rewards.torques.weight = -1.0e-5
        self.rewards.torque_limits.weight = -1.0e-3
        self.rewards.dof_vel_limits.weight = -1.0e-3
        self.rewards.feet_velocity.weight = -5.0e-2
        self.rewards.feet_contact_forces.weight = -1.0e-3
        self.rewards.collision.weight = -1.0
        self.rewards.two_leg_stability_safe.weight = 1.0e-6
        self.rewards.deploy_posture_gate.weight = -5.0
        self.rewards.is_terminated.weight = -10.0


@configclass
class Lite3TwoLegStandDeployR12MimicEnvCfg(Lite3TwoLegStandDeployR1EnvCfg):
    """Isolated baseline task that starts from the legacy deploy/r1-r2 strategy.

    Notes:
    - Intentionally keeps the current rl_training_new reward semantics (including
      base_height_bonus behavior) as requested.
    - Keeps default realistic history semantics via global wrappers
      (history cleared on done/reset unless debug override is enabled).
    """

    def __post_init__(self):
        super().__post_init__()

        # Isolated baseline: deploy/r1 phases with non-safe exploration phase 0.
        self.curriculum.phases.params["phases"] = mdp.get_two_leg_stand_deploy_r12_mimic_phases()
        self.curriculum.phases.params["steps_per_env"] = 24
        self.curriculum.phases.params["front_touch_termination"] = {
            "enabled": False,
            "metrics": {"two_leg_stability": 0.75},
            "log_enable": True,
        }


@configclass
class Lite3TwoLegStandDeploySafeV2EnvCfg(Lite3TwoLegStandDeployR1EnvCfg):
    """Lite3 two-leg standing with deploy-safe v2 curriculum progression."""

    def __post_init__(self):
        super().__post_init__()

        # Use improved safety progression curriculum.
        self.curriculum.phases.params["phases"] = mdp.get_two_leg_stand_deploy_safe_v2_phases()
        self.curriculum.phases.params["steps_per_env"] = 24
        self.curriculum.phases.params["front_touch_termination"] = {
            "enabled": False,
            "metrics": {"two_leg_stability": 0.75},
            "log_enable": True,
        }
