import numpy as np
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TwoLegStandCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.32]
        default_joint_angles = {
            'FL_HipX_joint': 0.0, 'FR_HipX_joint': 0.0,
            'HL_HipX_joint': 0.0, 'HR_HipX_joint': 0.0,
            'FL_HipY_joint': -0.2, 'FR_HipY_joint': -0.2,  # lifted front
            'FL_Knee_joint': 0.1, 'FR_Knee_joint': 0.1,
            'HL_HipY_joint': -1.0, 'HR_HipY_joint': -1.0,
            'HL_Knee_joint': 1.8, 'HR_Knee_joint': 1.8,
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
        base_height_target = 0.36
        still_all = True
        only_positive_rewards = True
        pitch_roll_factor = [1.0, 1.0]  # still used by orientation reward

        # Custom posture shaping params
        front_legs = ["FL", "FR"]
        rear_legs = ["HL", "HR"]
        torso_upright_pitch_target = 0.0  # radians
        torso_upright_pitch_tolerance = 0.3  # allow some lean
        reward_upright_tolerance = 0.2
        front_foot_contact_penalty = -2.0
        foot_stillness_reward_weight = 1.0

        class scales(LeggedRobotCfg.rewards.scales):
            torso_upright = 3.0
            # hind_knee_extension = 7.0
            hind_leg_extension_geom = 4.0
            front_legs_up = 1.0
            base_height = -0.5
            termination = -10.0
            # orientation = -1.0
            # base_height = -0.5
            stand_still = 1.0
            # torques = -0.0001
            # action_rate = -0.01
            # collision = -1.0
            # termination = -10.0

            # # NEW: encourage torso uprightness
            # torso_upright = 3.0

            # # NEW: reward front legs off ground
            front_legs_up = 1.0
            # # NEW: reward for hind knee extension
            # hind_knee_extension = 2.0
            # # NEW: reward for stable feet (discourage thrashing)
            # foot_stillness = 1.0

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

            phases = [
                {
                    "name": "phase_0_basic_balance",
                    "trigger_thresh": 1000,
                    "reward_scales": {
                        "base_height": -0.5,
                        "hind_leg_extension_geom": 4.0,
                        "front_legs_up": 1.0,
                        "torso_upright": 3.0,
                        "termination": -10.0,
                    }
                },
                {
                    "name": "phase_1_posture_alignment",
                    "trigger_thresh": 3000,
                    "reward_scales": {
                        "base_height": -0.5,
                        "hind_leg_extension_geom": 4.0,
                        "front_legs_up": 1.0,
                        "torso_upright": 3.0,
                        "termination": -10.0,
                        "stand_still": 1.0
                    }
                },
                {
                    "name": "phase_2_fine_standing",
                    "trigger_thresh": np.inf,
                    "reward_scales": {
                        "hind_leg_stretch": 4.0,
                        "front_legs_up": 1.0,
                        "torso_upright": 3.0,
                        "termination": -10.0,
                        "stand_still": 1.0
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

        class ranges:
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
