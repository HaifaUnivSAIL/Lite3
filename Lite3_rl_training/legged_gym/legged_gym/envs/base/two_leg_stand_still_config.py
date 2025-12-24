from legged_gym.envs.base.two_leg_stand_config import TwoLegStandCfg, TwoLegStandCfgPPO


class TwoLegStandStillCfg(TwoLegStandCfg):
    """Two-leg stand variant that prioritizes stillness and low yaw-rate stabilization.

    Kept in a separate module so `two_leg_stand_config.py` (current best) stays
    unchanged and fully reproducible.
    """

    class rewards(TwoLegStandCfg.rewards):
        # ---------------- Deployment safety gates ----------------
        # Mirror the deploy posture guard (roll/pitch) so training learns to stay inside
        # the same envelope that would otherwise trigger JointDamping on the robot.
        deploy_roll_limit_deg = 40.0
        # NOTE: two-leg stand intentionally uses a large lean-back pitch; keep this high enough
        # to not block the intended solution (tighten only if your real-robot safety requires it).
        deploy_pitch_limit_deg = 90.0
        deploy_posture_margin_deg = 0.5  # extra slack (deg) to avoid chattering at the boundary

        # Also align training terminations with the same limits (optional but recommended).
        termination_roll_deg = deploy_roll_limit_deg
        termination_pitch_deg = deploy_pitch_limit_deg

        # Ensure stillness terms exist in base scales so curriculum can activate them.
        class scales(TwoLegStandCfg.rewards.scales):
            stand_still_roll_only = 0.05
            stand_still_yaw_only = 0.05
            hind_legs_calmness = 0.2
            feet_velocity = -0.05
            action_rate = -0.01
            # Penalty for exceeding deploy posture limits (radians over limit).
            deploy_posture_gate = -5.0

        class curriculum:
            enabled = True
            log_curriculum = True

            class front_touch_termination:
                enabled = False
                metrics = {
                    "human_posture": 0.0,
                    "front_legs_up_continuous": 0.0,
                }
                log_enable = True

            phases = [
                {  # Get both front legs airborne quickly, maintain basic posture.
                    "name": "phase_0_legs_up_warmup",
                    "trigger_thresh": 500,
                    "near_goal_init_prob": 0.0,
                    "reward_scales": {
                        "front_legs_up_warmup": 18.0,
                        "torso_upright_warmup": 8.0,
                        "base_height_bonus": 6.0,
                        "front_tap_penalty": -0.5,
                        "deploy_posture_gate": -5.0,
                        "termination": -10.0,
                    },
                },
                {  # Start learning to settle (reduce yaw/roll rates) while staying upright.
                    "name": "phase_1_basic_stability",
                    "trigger_thresh": 3000,
                    "near_goal_init_prob": 0.25,
                    "reward_scales": {
                        "front_legs_up_warmup": 14.0,
                        "torso_upright_soften": 8.0,
                        "base_height_bonus": 8.0,
                        "hind_leg_extension_geom": 2.0,
                        "hind_legs_calmness": 2.0,
                        "stand_still": 2.0,
                        "stand_still_yaw_only": 1.5,
                        "stand_still_roll_only": 0.5,
                        "feet_velocity": -0.2,
                        "action_rate": -0.01,
                        "front_tap_penalty": -1.5,
                        "deploy_posture_gate": -5.0,
                        "termination": -10.0,
                    },
                },
                {  # Switch to continuous stability rewards (penalizes yaw/spin implicitly).
                    "name": "phase_2_reduce_spin",
                    "trigger_thresh": 5000,
                    "near_goal_init_prob": 0.6,
                    "reward_scales": {
                        "front_legs_up_continuous": 6.0,
                        "torso_upright_continuous": 8.0,
                        "human_posture_warmup": 4.0,
                        "hind_leg_extension_geom": 3.0,
                        "hind_legs_calmness": 3.0,
                        "stand_still": 4.0,
                        "stand_still_yaw_only": 3.0,
                        "feet_velocity": -0.4,
                        "action_rate": -0.02,
                        "front_tap_penalty": -2.0,
                        "base_height_bonus": 6.0,
                        "deploy_posture_gate": -5.0,
                        "termination": -10.0,
                    },
                },
                {  # Final: stand tall, quiet, and still; spinning only when necessary.
                    "name": "phase_3_fine_still_stand",
                    "trigger_thresh": 7500,
                    "near_goal_init_prob": 0.75,
                    "reward_scales": {
                        "front_legs_up_continuous": 8.0,
                        "torso_upright_continuous": 10.0,
                        "human_posture": 6.0,
                        "hind_leg_extension_geom": 4.0,
                        "hind_legs_calmness": 4.0,
                        "stand_still": 6.0,
                        "stand_still_yaw_only": 5.0,
                        "stand_still_roll_only": 1.0,
                        "feet_velocity": -0.6,
                        "action_rate": -0.03,
                        "front_tap_penalty": -3.0,
                        "base_height_bonus": 6.0,
                        "deploy_posture_gate": -5.0,
                        "termination": -10.0,
                    },
                },
            ]


class TwoLegStandStillCfgPPO(TwoLegStandCfgPPO):
    class runner(TwoLegStandCfgPPO.runner):
        experiment_name = "two_leg_stand_still"
