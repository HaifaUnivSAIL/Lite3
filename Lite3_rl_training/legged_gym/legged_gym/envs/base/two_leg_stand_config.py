from copy import deepcopy

import numpy as np
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TwoLegStandCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.32]
        rot = [-0.00023085526184233324, -0.0032073138974974646, -0.0019571690372445424, 0.9999929146412841]
        default_joint_angles = {
            'FL_HipX_joint': -0.0154048,
            'FR_HipX_joint': 0.0159887,
            'HL_HipX_joint': -0.0221317,
            'HR_HipX_joint': 0.0224431,
            'FL_HipY_joint': -0.76697,
            'FR_HipY_joint': -0.768286,
            'FL_Knee_joint': 1.53761,
            'FR_Knee_joint': 1.53636,
            'HL_HipY_joint': -0.765865,
            'HR_HipY_joint': -0.767203,
            'HL_Knee_joint': 1.54788,
            'HR_Knee_joint': 1.54679,
        }
        near_goal_init_prob = 0.0
        near_goal_state = {
            'pos': [0.0, 0.0, 0.62],
            'rot': [0.0, 0.5372996083468239, 0.0, 0.8433914458128857],  # roll=0, pitch=65deg, yaw=0
            'lin_vel': [0.0, 0.0, 0.0],
            'ang_vel': [0.0, 0.0, 0.0],
            'default_joint_angles': {
                'FL_HipX_joint': -0.02,
                'FR_HipX_joint': 0.02,
                'HL_HipX_joint': -0.03,
                'HR_HipX_joint': 0.03,
                'FL_HipY_joint': 0.2,
                'FR_HipY_joint': 0.2,
                'HL_HipY_joint': -1.25,
                'HR_HipY_joint': -1.25,
                'FL_Knee_joint': 2.35,
                'FR_Knee_joint': 2.35,
                'HL_Knee_joint': 0.65,
                'HR_Knee_joint': 0.65,
            },
        }
        near_goal_noise = {
            'pos': 0.02,
            'rot': 0.08,
            'lin_vel': 0.08,
            'ang_vel': 0.08,
            'joint': 0.08,
        }

    class env(LeggedRobotCfg.env):
        num_envs = 2048
        num_observations = 117
        num_privileged_obs = 54
        num_observation_history = 40
        episode_length_s = 5.0
        curriculum_factor = 0.8

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 20.0}
        damping = {'joint': 0.7}
        action_scale = 0.25
        decimation = 4
        use_torch_vel_estimator = False
        use_actuator_network = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lite3/urdf/Lite3.urdf'
        name = "Lite3"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]
        self_collisions = 1
        restitution_mean = 0.5
        restitution_offset_range = [-0.1, 0.1]
        compliance = 0.5

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 1
        still_all = True
        only_positive_rewards = True
        pitch_roll_factor = [0.2, 1.0]  # keep roll tight, relax pitch

        # Custom posture shaping params
        front_legs = ["FL", "FR"]
        rear_legs = ["HL", "HR"]
        torso_upright_pitch_target = float(np.deg2rad(-70.0))  # preferred lean-back
        torso_upright_pitch_tolerance = float(np.deg2rad(25.0))
        reward_upright_tolerance = float(np.deg2rad(22.0))
        front_foot_contact_penalty = -2.0
        foot_stillness_reward_weight = 1.0
        base_height_bonus_threshold = 0.6
        base_height_bonus_ceiling = 0.9

        class scales(LeggedRobotCfg.rewards.scales):
            torso_upright = 4.5
            front_legs_up_warmup = 3.0
            human_posture_warmup = 1.8
            torso_upright_soften = 1.8
            torso_upright_warmup = 1.8
            torso_upright_continuous = 7.0
            human_posture = 5.0
            front_legs_up = 2.5
            termination = -10.0
            stand_still = 0.15
            front_legs_up_continuous = 5.5
            base_height = -0.1
            base_height_bonus = 1.5
            hind_leg_extension_geom = 1.0
            hind_knee_extension = 0.5
            front_tap_penalty = -3.0

            # Disabled locomotion terms
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            stumble = 0.0
            feet_velocity = 0.0
            dof_pos_limits = 0.0
            episode_length = 0.0
                # Curriculum toggles
        #  --------------------------- Curriculum cfg part ---------------------------------------------#
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
                {  # Emphasize immediate front-leg lift-off
                    "name": "phase_0_legs_up_warmup",
                    "trigger_thresh": 500,
                    "near_goal_init_prob": 0.0,
                    "reward_scales": {
                        "front_legs_up_warmup": 18.0,
                        "torso_upright_warmup": 8.0,
                        "base_height_bonus": 6.0,
                        "front_tap_penalty": 0.0,
                        "termination": -10.0,
                    }
                },
                {
                    "name": "phase_0_basic",
                    "trigger_thresh": 1000,
                    "near_goal_init_prob": 0.0,
                    "reward_scales": {
                        "front_legs_up_warmup": 14.0,
                        "torso_upright_warmup": 10.0,
                        "base_height_bonus": 8.0,
                        "stand_still": 0.0,
                        "front_tap_penalty": -1.0,
                        "termination": -10.0,
                    }
                },
                {
                    "name": "phase_1_posture_alignment",
                    "trigger_thresh": 2500,
                    "near_goal_init_prob": 0.45,
                    "reward_scales": {
                        "front_legs_up_warmup": 14.0,
                        "torso_upright_warmup": 10.0,
                        "base_height_bonus": 8.0,
                        "stand_still_roll_only": 1.0,
                        "front_tap_penalty": -1.0,
                        "termination": -10.0,
                    }
                },
                {
                    "name": "phase_2_fine_standing_roll_supression",
                    "trigger_thresh": 4000,
                    "near_goal_init_prob": 0.7,
                    "reward_scales": {
                        "front_legs_up_warmup": 14.0,
                        "torso_upright_warmup": 10.0,
                        "base_height_bonus": 8.0,
                        "stand_still_roll_only": 10.0,
                        "front_tap_penalty": -1.0,
                        "termination": -10.0,
                    }
                },
                {
                    "name": "phase_2_fine_standing_locally",
                    "trigger_thresh": 7000,
                    "near_goal_init_prob": 0.7,
                    "reward_scales": {
                        "front_legs_up_warmup": 14.0,
                        "torso_upright_warmup": 10.0,
                        "base_height_bonus": 8.0,
                        "stand_still_roll_only": 10.0,
                        "stand_still_lin_x": 10.0,
                        "stand_still_lin_y": 10.0,
                        "front_tap_penalty": -1.0,
                        "termination": -10.0,
                    }
                },
                {
                    "name": "phase_2_fine_standing_chill",
                    "trigger_thresh": 7000,
                    "near_goal_init_prob": 0.7,
                    "reward_scales": {
                        "front_legs_up_warmup": 14.0,
                        "torso_upright_warmup": 10.0,
                        "base_height_bonus": 8.0,
                        "stand_still_roll_only": 10.0,
                        "stand_still_lin_x": 10.0,
                        "stand_still_lin_y": 10.0,
                        "_reward_torque_energy": 10.0,
                        "front_tap_penalty": -1.0,
                        "termination": -10.0,
                    }
                }
            ]

        # --------------------------------------------------------------------------------------------- #


    class normalization(LeggedRobotCfg.normalization):
        dof_history_interval = 1
        clip_angles = [[-0.523, 0.523], [-0.314, 3.6], [-2.792, -0.524]]

        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            height_measurements = 0.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        heights_uniform_noise = False
        heights_gaussian_mean_mutable = True
        heights_downgrade_frequency = False

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            height_measurements = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        fixed_commands = [0.0, 0.0, 0.0]
        resampling_time = 6

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        dummy_normal = True
        random_reset = False
        curriculum = False
        terrain_proportions = [0.0] * 7
        measure_heights = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_com_offset = True
        com_offset_range = [[-0.05, 0.01], [-0.03, 0.03], [-0.03, 0.03]]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]
        randomize_Kd_factor = True
        Kd_factor_range = [0.8, 1.2]


class TwoLegStandCfgPPO(LeggedRobotCfgPPO):

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'two_leg_stand'
        max_iterations = 15000
        resume = False
        resume_path = 'legged_gym/logs/two_leg_stand'
        load_run = ''
        checkpoint = -1
