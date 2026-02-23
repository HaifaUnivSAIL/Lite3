"""Pinned deploy-r1/r2 training strategy for the legacy Lite3 stack.

This config intentionally mirrors the strategy captured in:
- legged_gym/logs/deploy/r1/env_cfg.json
- legged_gym/logs/deploy/r2/env_cfg.json

Use this when you want to reproduce that exact curriculum/reward shaping profile.
"""

from legged_gym.envs.base.two_leg_stand_deploy_aligned_config import (
    TwoLegStandDeployAlignedCfg,
    TwoLegStandDeployAlignedCfgPPO,
)


class TwoLegStandDeployR12MimicCfg(TwoLegStandDeployAlignedCfg):
    """Replay the deploy/r1-r2 curriculum strategy exactly."""

    class env(TwoLegStandDeployAlignedCfg.env):
        episode_length_s = 8.0

    class rewards(TwoLegStandDeployAlignedCfg.rewards):
        # Keep hard safety and posture guard identical to deploy logs.
        deploy_roll_limit_deg = 40.0
        deploy_pitch_limit_deg = 90.0
        deploy_posture_margin_deg = 0.5
        termination_roll_deg = 40.0
        termination_pitch_deg = 90.0
        require_hind_contact = True
        max_contact_force = 80.0

        # Safety gate mixing (from deploy/r1-r2).
        safe_gate_torque_limits_weight = 0.05
        safe_gate_dof_vel_limits_weight = 0.05
        safe_gate_power_weight = 5e-4
        safe_gate_action_weight = 0.02

        class scales(TwoLegStandDeployAlignedCfg.rewards.scales):
            # Keep these active so curriculum can schedule them.
            torques = -1e-5
            dof_vel = -1e-6
            dof_acc = -2.5e-7
            action_rate = -0.01
            target_smoothness = -1e-6
            action_magnitude = -1e-6
            ang_vel_xy = -1e-6
            lin_vel_z = -1e-6
            front_legs_up_warmup_safe = 1e-6
            front_legs_up_continuous_safe = 1e-6
            two_leg_stability_safe = 1e-6
            torque_limits = -1e-3
            dof_vel_limits = -1e-3
            power = -1e-3
            feet_contact_forces = -1e-3
            feet_velocity = -0.05
            deploy_posture_gate = -5.0

        class curriculum(TwoLegStandDeployAlignedCfg.rewards.curriculum):
            phases = [
                {
                    "name": "phase_0_legs_up_safe_warmup",
                    "trigger_thresh": 500,
                    "near_goal_init_prob": 0.0,
                    "reward_scales": {
                        "front_legs_up_warmup_safe": 18.0,
                        "torso_upright_warmup": 8.0,
                        "base_height_bonus": 6.0,
                        "hind_leg_extension_geom": 1.0,
                        "hind_legs_calmness": 0.5,
                        "stand_still_yaw_only": 0.3,
                        "stand_still_roll_only": 0.2,
                        "stand_still_lin_x": 0.2,
                        "stand_still_lin_y": 0.2,
                        "front_tap_penalty": -0.3,
                        "deploy_posture_gate": -5.0,
                        "lin_vel_z": -0.1,
                        "ang_vel_xy": -0.05,
                        "feet_velocity": -0.15,
                        "action_rate": -0.02,
                        "target_smoothness": -0.002,
                        "dof_acc": -2e-6,
                        "power": -2e-5,
                        "torque_limits": -0.05,
                        "dof_vel_limits": -0.05,
                        "termination": -10.0,
                    },
                },
                {
                    "name": "phase_1_explore_stable_two_leg_safe",
                    "trigger_thresh": 3000,
                    "near_goal_init_prob": 0.15,
                    "reward_scales": {
                        "front_legs_up_warmup_safe": 12.0,
                        "front_legs_up_continuous_safe": 6.0,
                        "torso_upright_soften": 8.0,
                        "torso_upright_continuous": 5.0,
                        "base_height_bonus": 8.0,
                        "hind_leg_extension_geom": 3.0,
                        "hind_legs_calmness": 1.5,
                        "stand_still_yaw_only": 0.8,
                        "stand_still_roll_only": 0.4,
                        "stand_still_lin_x": 0.4,
                        "stand_still_lin_y": 0.4,
                        "stand_still_lin_z": 0.2,
                        "two_leg_stability_safe": 1.5,
                        "lin_vel_z": -0.12,
                        "ang_vel_xy": -0.06,
                        "feet_velocity": -0.22,
                        "front_tap_penalty": -1.0,
                        "deploy_posture_gate": -5.0,
                        "action_rate": -0.03,
                        "target_smoothness": -0.004,
                        "dof_acc": -3e-6,
                        "power": -4e-5,
                        "torque_limits": -0.1,
                        "dof_vel_limits": -0.1,
                        "termination": -10.0,
                    },
                },
                {
                    "name": "phase_2_transition_reduce_spin_safe",
                    "trigger_thresh": 10000,
                    "near_goal_init_prob": 0.25,
                    "reward_scales": {
                        "front_legs_up_warmup_safe": 8.0,
                        "front_legs_up_continuous_safe": 7.0,
                        "torso_upright_soften": 6.0,
                        "torso_upright_continuous": 5.0,
                        "base_height_bonus": 7.0,
                        "hind_leg_extension_geom": 4.0,
                        "human_posture_warmup": 2.5,
                        "hind_legs_calmness": 2.0,
                        "stand_still_yaw_only": 1.5,
                        "stand_still_roll_only": 0.6,
                        "stand_still_lin_x": 0.7,
                        "stand_still_lin_y": 0.7,
                        "stand_still_lin_z": 0.4,
                        "two_leg_stability_safe": 3.0,
                        "lin_vel_z": -0.15,
                        "ang_vel_xy": -0.08,
                        "feet_velocity": -0.3,
                        "front_tap_penalty": -1.8,
                        "deploy_posture_gate": -5.0,
                        "action_rate": -0.04,
                        "target_smoothness": -0.01,
                        "dof_acc": -4e-6,
                        "power": -6e-5,
                        "torque_limits": -0.15,
                        "dof_vel_limits": -0.15,
                        "termination": -10.0,
                    },
                },
                {
                    "name": "phase_3_refine_still_stand_safe",
                    "trigger_thresh": 15000,
                    "near_goal_init_prob": 0.3,
                    "reward_scales": {
                        "front_legs_up_continuous_safe": 8.0,
                        "torso_upright_continuous": 9.0,
                        "human_posture": 5.0,
                        "hind_leg_extension_geom": 5.0,
                        "hind_legs_calmness": 3.0,
                        "stand_still_yaw_only": 3.5,
                        "stand_still_roll_only": 1.0,
                        "stand_still_lin_x": 1.2,
                        "stand_still_lin_y": 1.2,
                        "stand_still_lin_z": 0.8,
                        "two_leg_stability_safe": 5.0,
                        "lin_vel_z": -0.18,
                        "ang_vel_xy": -0.12,
                        "feet_velocity": -1.5,
                        "front_tap_penalty": -4.0,
                        "base_height_bonus": 6.0,
                        "deploy_posture_gate": -5.0,
                        "action_rate": -0.08,
                        "target_smoothness": -0.04,
                        "dof_vel": -0.002,
                        "dof_acc": -2e-5,
                        "power": -0.0002,
                        "torque_limits": -0.35,
                        "dof_vel_limits": -0.5,
                        "feet_contact_forces": -0.05,
                        "termination": -10.0,
                    },
                },
                {
                    "name": "phase_4_final_polish_safe",
                    "trigger_thresh": 999999,
                    "near_goal_init_prob": 0.35,
                    "reward_scales": {
                        "front_legs_up_continuous_safe": 8.0,
                        "torso_upright_continuous": 10.0,
                        "human_posture": 6.0,
                        "hind_leg_extension_geom": 6.0,
                        "hind_legs_calmness": 3.5,
                        "stand_still_yaw_only": 4.0,
                        "stand_still_roll_only": 1.2,
                        "stand_still_lin_x": 1.4,
                        "stand_still_lin_y": 1.4,
                        "stand_still_lin_z": 1.0,
                        "two_leg_stability_safe": 6.0,
                        "lin_vel_z": -0.2,
                        "ang_vel_xy": -0.15,
                        "feet_velocity": -1.8,
                        "front_tap_penalty": -5.0,
                        "base_height_bonus": 6.0,
                        "deploy_posture_gate": -5.0,
                        "action_rate": -0.1,
                        "target_smoothness": -0.06,
                        "dof_vel": -0.003,
                        "dof_acc": -3e-5,
                        "power": -0.0003,
                        "torque_limits": -0.45,
                        "dof_vel_limits": -0.6,
                        "feet_contact_forces": -0.08,
                        "termination": -10.0,
                    },
                },
            ]


class TwoLegStandDeployR12MimicCfgPPO(TwoLegStandDeployAlignedCfgPPO):
    class runner(TwoLegStandDeployAlignedCfgPPO.runner):
        experiment_name = "deploy_r12_mimic"

