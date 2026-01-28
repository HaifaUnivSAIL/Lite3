from legged_gym.envs.base.two_leg_stand_still_config import (
    TwoLegStandStillCfg,
    TwoLegStandStillCfgPPO,
)


class TwoLegStandStillV2Cfg(TwoLegStandStillCfg):
    """Two-leg stand curriculum tuned for longer exploration before tightening gates.

    Observed failure mode: switching too early into the "reduce spin" phase can
    collapse the policy into a 4-leg freeze. This variant:
      - extends the exploration window substantially (phase transitions later),
      - replaces posture-biased stillness (`stand_still`, which prefers the
        default 4-leg pose) with velocity-only stillness rewards,
      - ramps spin/tap penalties more gradually.
    """

    class rewards(TwoLegStandStillCfg.rewards):
        # Make velocity-only stillness rewards available to curriculum.
        class scales(TwoLegStandStillCfg.rewards.scales):
            stand_still_lin_x = 0.05
            stand_still_lin_y = 0.05
            stand_still_lin_z = 0.05
            # Enable joint-velocity penalties for later curriculum phases.
            dof_vel = -1e-6

        class curriculum:
            enabled = True
            log_curriculum = True

            class front_touch_termination:
                enabled = False
                metrics = {
                    "two_leg_stability": 0.75,
                }
                log_enable = True

            phases = [
                {  # Establish front-leg clearance + base height early.
                    "name": "phase_0_legs_up_warmup",
                    "trigger_thresh": 500,
                    "near_goal_init_prob": 0.0,
                    "reward_scales": {
                        "front_legs_up_warmup": 18.0,
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
                        "termination": -10.0,
                    },
                },
                {  # Long exploration window: keep successful shaping longer.
                    "name": "phase_1_explore_stable_two_leg",
                    "trigger_thresh": 3000,
                    "near_goal_init_prob": 0.2,
                    "reward_scales": {
                        "front_legs_up_warmup": 14.0,
                        "torso_upright_soften": 8.0,
                        "base_height_bonus": 8.0,
                        "hind_leg_extension_geom": 3.0,
                        "hind_legs_calmness": 1.5,
                        "stand_still_yaw_only": 0.8,
                        "stand_still_roll_only": 0.4,
                        "stand_still_lin_x": 0.4,
                        "stand_still_lin_y": 0.4,
                        "stand_still_lin_z": 0.2,
                        "feet_velocity": -0.15,
                        "action_rate": -0.01,
                        "front_tap_penalty": -1.0,
                        "deploy_posture_gate": -5.0,
                        "termination": -10.0,
                    },
                },
                {  # Gentle transition: mix warmup + continuous terms, light penalties.
                    "name": "phase_2_transition_reduce_spin",
                    # Keep this phase longer; model_5000.pt tends to be best here.
                    "trigger_thresh": 10000,
                    "near_goal_init_prob": 0.45,
                    "reward_scales": {
                        "front_legs_up_warmup": 10.0,
                        "front_legs_up_continuous": 4.0,
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
                        "feet_velocity": -0.25,
                        "action_rate": -0.02,
                        "front_tap_penalty": -1.8,
                        "deploy_posture_gate": -5.0,
                        "termination": -10.0,
                    },
                },
                {  # Refinement: emphasize quiet, no-spin stability.
                    "name": "phase_3_refine_still_stand",
                    # Run the remainder of a 15k-iter training budget in this phase.
                    "trigger_thresh": 15000,
                    "near_goal_init_prob": 0.7,
                    "reward_scales": {
                        "front_legs_up_continuous": 7.0,
                        "torso_upright_continuous": 9.0,
                        "human_posture": 5.0,
                        "hind_leg_extension_geom": 5.0,
                        "hind_legs_calmness": 3.0,
                        "stand_still_yaw_only": 3.5,
                        "stand_still_roll_only": 1.0,
                        "stand_still_lin_x": 1.2,
                        "stand_still_lin_y": 1.2,
                        "stand_still_lin_z": 0.8,
                        # Stronger gating on effort/energy + leg motion.
                        "torques": -0.0002,
                        "dof_vel": -0.0002,
                        "feet_velocity": -0.65,
                        "action_rate": -0.04,
                        "front_tap_penalty": -2.8,
                        "base_height_bonus": 6.0,
                        "deploy_posture_gate": -5.0,
                        "termination": -10.0,
                    },
                },
                {  # Final polish: allow long training runs to keep improving quietly.
                    "name": "phase_4_final_polish",
                    "trigger_thresh": 999999,
                    "near_goal_init_prob": 0.75,
                    "reward_scales": {
                        "front_legs_up_continuous": 8.0,
                        "torso_upright_continuous": 10.0,
                        "human_posture": 6.0,
                        "hind_leg_extension_geom": 6.0,
                        "hind_legs_calmness": 3.5,
                        "stand_still_yaw_only": 4.0,
                        "stand_still_roll_only": 1.2,
                        "stand_still_lin_x": 1.4,
                        "stand_still_lin_y": 1.4,
                        "stand_still_lin_z": 1.0,
                        "torques": -0.0003,
                        "dof_vel": -0.0003,
                        "feet_velocity": -0.75,
                        "action_rate": -0.05,
                        "front_tap_penalty": -3.2,
                        "base_height_bonus": 6.0,
                        "deploy_posture_gate": -5.0,
                        "termination": -10.0,
                    },
                },
            ]

    class commands(TwoLegStandStillCfg.commands):
        # Train on zero commands (stand task) even though train.py clears fixed_commands.
        class ranges(TwoLegStandStillCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]


class TwoLegStandStillV2CfgPPO(TwoLegStandStillCfgPPO):
    class runner(TwoLegStandStillCfgPPO.runner):
        experiment_name = "two_leg_stand_still_v2"
