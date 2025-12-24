from legged_gym.envs.base.two_leg_stand_still_v2_config import (
    TwoLegStandStillV2Cfg,
    TwoLegStandStillV2CfgPPO,
)


class TwoLegStandStillSafeCfg(TwoLegStandStillV2Cfg):
    """Two-leg stand curriculum with stricter effort/velocity safety shaping.

    Goal: keep the two-leg stand behavior, but reduce "violent" torques / joint speeds
    so the policy transfers better to the MuJoCo deploy stack.
    """

    class env(TwoLegStandStillV2Cfg.env):
        # Give the policy enough time to transition slowly and safely.
        episode_length_s = 8.0

    class rewards(TwoLegStandStillV2Cfg.rewards):
        # ---------------- Deployment safety gates ----------------
        # Mirror the deploy posture guard (roll/pitch) but keep pitch high enough for the task.
        deploy_roll_limit_deg = 40.0
        deploy_pitch_limit_deg = 90.0
        deploy_posture_margin_deg = 0.5

        termination_roll_deg = deploy_roll_limit_deg
        termination_pitch_deg = deploy_pitch_limit_deg

        # ---------------- Effort/velocity "soft" safety gates ----------------
        # These thresholds affect the *_limits reward terms (penalize only when exceeding).
        # Lower values = stricter safety envelope.
        soft_torque_limit = 0.7
        soft_dof_vel_limit = 0.7

        # Require both hind feet support to discourage hopping during stand-up.
        require_hind_contact = True

        # Optional: penalize high contact forces (used by `feet_contact_forces`).
        max_contact_force = 80.0

        # ---------------- Safety-gated rewards ----------------
        # `*_safe` rewards multiply the task progress metric by a gate that drops toward 0
        # when torques/velocities/power/action magnitude are large.
        safe_gate_torque_limits_weight = 0.05
        safe_gate_dof_vel_limits_weight = 0.05
        safe_gate_power_weight = 5e-4
        safe_gate_action_weight = 0.02

        class scales(TwoLegStandStillV2Cfg.rewards.scales):
            # Ensure these terms exist so curriculum can activate them.
            torques = -1e-5
            dof_vel = -1e-6
            dof_acc = -2.5e-7
            action_rate = -0.01
            # Must be non-zero so curriculum can activate it.
            target_smoothness = -1e-6
            action_magnitude = -1e-6
            ang_vel_xy = -1e-6
            lin_vel_z = -1e-6

            # Safety-gated progress/stability rewards (bounded in [0, 1]).
            front_legs_up_warmup_safe = 1e-6
            front_legs_up_continuous_safe = 1e-6
            two_leg_stability_safe = 1e-6

            # Safety-focused terms (thresholded, energy, contact/feet slip)
            torque_limits = -1e-3
            dof_vel_limits = -1e-3
            power = -1e-3
            feet_contact_forces = -1e-3
            feet_velocity = -0.05

        class curriculum(TwoLegStandStillV2Cfg.rewards.curriculum):
            phases = [
                {  # Warmup: learn legs-up + posture, but already discourage spikes.
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
                        # Safety shaping (mild): smooth, slow transition.
                        "lin_vel_z": -0.1,
                        "ang_vel_xy": -0.05,
                        "feet_velocity": -0.15,
                        "action_rate": -0.02,
                        "target_smoothness": -0.002,
                        "dof_acc": -2.0e-6,
                        "power": -0.00002,
                        "torque_limits": -0.05,
                        "dof_vel_limits": -0.05,
                        "termination": -10.0,
                    },
                },
                {  # Explore: extend stable two-leg, increase smoothness + effort limits.
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
                        # Safety shaping (medium)
                        "action_rate": -0.03,
                        "target_smoothness": -0.004,
                        "dof_acc": -3.0e-6,
                        "power": -0.00004,
                        "torque_limits": -0.1,
                        "dof_vel_limits": -0.1,
                        "termination": -10.0,
                    },
                },
                {  # Transition: tighten safety while keeping exploration.
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
                        # Safety shaping (strong)
                        "action_rate": -0.04,
                        "target_smoothness": -0.01,
                        "dof_acc": -4.0e-6,
                        "power": -0.00006,
                        "torque_limits": -0.15,
                        "dof_vel_limits": -0.15,
                        "termination": -10.0,
                    },
                },
                {  # Refinement: prioritize quiet, low-effort stability.
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
                        "feet_velocity": -0.65,
                        "front_tap_penalty": -2.8,
                        "base_height_bonus": 6.0,
                        "deploy_posture_gate": -5.0,
                        # Safety shaping (very strong)
                        "action_rate": -0.04,
                        "target_smoothness": -0.01,
                        "dof_acc": -5.0e-6,
                        "power": -0.00008,
                        "torque_limits": -0.2,
                        "dof_vel_limits": -0.2,
                        "feet_contact_forces": -0.01,
                        "termination": -10.0,
                    },
                },
                {  # Final polish: keep tightening safety without sacrificing success.
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
                        "feet_velocity": -0.75,
                        "front_tap_penalty": -3.2,
                        "base_height_bonus": 6.0,
                        "deploy_posture_gate": -5.0,
                        # Safety shaping (max)
                        "action_rate": -0.05,
                        "target_smoothness": -0.02,
                        "dof_acc": -6.0e-6,
                        "power": -0.0001,
                        "torque_limits": -0.25,
                        "dof_vel_limits": -0.25,
                        "feet_contact_forces": -0.02,
                        "termination": -10.0,
                    },
                },
            ]


class TwoLegStandStillSafeCfgPPO(TwoLegStandStillV2CfgPPO):
    class runner(TwoLegStandStillV2CfgPPO.runner):
        experiment_name = "two_leg_stand_still_safe"
