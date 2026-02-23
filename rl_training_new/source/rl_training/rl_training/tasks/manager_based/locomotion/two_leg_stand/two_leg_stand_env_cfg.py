# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Two-leg standing environment configuration for Lite3 quadruped robot.

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# Import base MDP functions
from isaaclab.envs import mdp as base_mdp

# Import two-leg standing specific MDP
from rl_training.tasks.manager_based.locomotion.two_leg_stand import mdp


##
# Scene definition
##


@configclass
class TwoLegStandSceneCfg(InteractiveSceneCfg):
    """Configuration for the two-leg standing scene."""

    # Ground terrain - flat plane for standing task
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.5,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # Robot articulation (set by subclass)
    robot: ArticulationCfg = MISSING

    # Contact sensor for all bodies
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    # Lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class TwoLegStandCommandsCfg:
    """Command specifications for two-leg standing."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class TwoLegStandActionsCfg:
    """Action specifications for two-leg standing."""

    joint_pos = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,  # Smaller scale for precise standing control
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )


@configclass
class TwoLegStandObservationsCfg:
    """Observation specifications for two-leg standing."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Commands (3D)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Base orientation (RPY) (3D)
        base_rpy = ObsTerm(
            func=mdp.base_rpy,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Angular velocity (3D)
        base_ang_vel = ObsTerm(
            func=base_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Joint positions (12D)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Joint velocities (12D)
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.1,
        )

        # Joint position history (36D)
        joint_pos_history = ObsTerm(
            func=mdp.joint_pos_history,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # Joint velocity history (24D)
        joint_vel_history = ObsTerm(
            func=mdp.joint_vel_history,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.1,
        )

        # Action history (24D)
        action_history = ObsTerm(
            func=mdp.action_history,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group (no noise for value estimation)."""

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        base_rpy = ObsTerm(
            func=mdp.base_rpy,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        base_ang_vel = ObsTerm(
            func=base_mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.1,
        )

        joint_pos_history = ObsTerm(
            func=mdp.joint_pos_history,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        joint_vel_history = ObsTerm(
            func=mdp.joint_vel_history,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.1,
        )

        action_history = ObsTerm(
            func=mdp.action_history,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Privileged observations for asymmetric training."""

        contact_states = ObsTerm(
            func=mdp.contact_states,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
                )
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        friction_coeffs = ObsTerm(
            func=mdp.friction_coeffs,
            params={"repeat": 4},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        external_wrench = ObsTerm(
            func=mdp.external_wrench,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        mass_payload = ObsTerm(
            func=mdp.mass_payload,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["TORSO"])},
            clip=(-100.0, 100.0),
            scale=0.5,
        )

        com_displacement = ObsTerm(
            func=mdp.com_displacement,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["TORSO"])},
            clip=(-100.0, 100.0),
            scale=20.0,
        )

        motor_strength = ObsTerm(
            func=mdp.motor_strength_factors,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=5.0,
        )

        kp_factor = ObsTerm(
            func=mdp.kp_factors,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=4.0,
        )

        kd_factor = ObsTerm(
            func=mdp.kd_factors,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=2.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class TwoLegStandEventCfg:
    """Configuration for domain randomization events."""

    # Startup events
    randomize_rigid_body_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.25),
            "dynamic_friction_range": (0.1, 1.25),
            "restitution_range": (0.4, 0.6),
            "num_buckets": 1024,
        },
    )

    randomize_rigid_body_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["TORSO"]),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    randomize_com_positions = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.01), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
        },
    )

    # Reset events
    randomize_reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "strength_range": (0.8, 1.2),
            "apply_to_gains": True,
        },
    )

    randomize_reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # Interval events
    randomize_push_robot = EventTerm(
        func=mdp.push_robots,
        mode="interval",
        interval_range_s=(15.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["TORSO"]),
            "max_force": 10.0,
            "max_torque": 10.0,
            "max_vel_xy": 0.5,
        },
    )

    reset_to_near_goal = EventTerm(
        func=mdp.reset_to_near_goal_state,
        mode="reset",
        params={
            "near_goal_prob": 0.0,
            "goal_pos": [0.0, 0.0, 0.62],
            "goal_rpy_deg": [0.0, 65.0, 0.0],
            "goal_joint_angles": {
                "FL_HipX_joint": -0.02,
                "FR_HipX_joint": 0.02,
                "HL_HipX_joint": -0.03,
                "HR_HipX_joint": 0.03,
                "FL_HipY_joint": 0.2,
                "FR_HipY_joint": 0.2,
                "HL_HipY_joint": -1.25,
                "HR_HipY_joint": -1.25,
                "FL_Knee_joint": 2.35,
                "FR_Knee_joint": 2.35,
                "HL_Knee_joint": 0.65,
                "HR_Knee_joint": 0.65,
            },
            "pos_noise": 0.02,
            "rot_noise": 0.08,
            "vel_noise": 0.08,
            "joint_noise": 0.08,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_to_deploy = EventTerm(
        func=mdp.reset_to_deploy_state,
        mode="reset",
        params={
            "deploy_prob": 0.0,
            "deploy_height": 0.30,
            "deploy_quat_w": None,
            "deploy_joint_angles": None,
            "add_noise": False,
            "pos_noise": 0.0,
            "joint_noise": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class TwoLegStandRewardsCfg:
    """Reward terms for two-leg standing."""

    # Termination penalty
    is_terminated = RewTerm(func=base_mdp.is_terminated, weight=-10.0)

    # === Torso Upright Rewards ===
    torso_upright = RewTerm(
        func=mdp.torso_upright,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    torso_upright_soften = RewTerm(
        func=mdp.torso_upright_soften,
        weight=0.0,
        params={
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,  # -70 degrees
            "min_height": 0.32,
            "height_range": 0.15,
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    torso_upright_warmup = RewTerm(
        func=mdp.torso_upright_warmup,
        weight=0.0,
        params={
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "min_height": 0.25,
            "height_range": 0.25,
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    torso_upright_continuous = RewTerm(
        func=mdp.torso_upright_continuous,
        weight=0.0,
        params={
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "min_height": 0.32,
            "height_range": 0.15,
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === Front Legs Up Rewards ===
    front_legs_up = RewTerm(
        func=mdp.front_legs_up,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
        },
    )

    front_legs_up_warmup = RewTerm(
        func=mdp.front_legs_up_warmup,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_body_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"]),
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "min_height": 0.30,
            "height_range": 0.15,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    front_legs_up_continuous = RewTerm(
        func=mdp.front_legs_up_continuous,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_body_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"]),
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "min_height": 0.35,
            "height_range": 0.2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    front_legs_up_warmup_safe = RewTerm(
        func=mdp.front_legs_up_warmup_safe,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_body_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"]),
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "min_height": 0.30,
            "height_range": 0.15,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    front_legs_up_continuous_safe = RewTerm(
        func=mdp.front_legs_up_continuous_safe,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_body_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"]),
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "min_height": 0.35,
            "height_range": 0.2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    front_tap_penalty = RewTerm(
        func=mdp.front_tap_penalty,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "front_feet_body_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === Human Posture Rewards ===
    human_posture = RewTerm(
        func=mdp.human_posture,
        weight=0.0,
        params={
            "hind_knee_joint_ids": [8, 11],  # HL_Knee, HR_Knee
            "hind_hip_joint_ids": [7, 10],   # HL_HipY, HR_HipY
            "hind_hip_body_ids": [9, 13],    # HL hip body indices
            "hind_foot_body_ids": [12, 16],  # HL/HR foot body indices
            "hip_targets": [-0.2, -0.2],
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    human_posture_warmup = RewTerm(
        func=mdp.human_posture_warmup,
        weight=0.0,
        params={
            "hind_knee_joint_ids": [8, 11],
            "hind_hip_body_ids": [9, 13],
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === Hind Leg Rewards ===
    hind_leg_extension_geom = RewTerm(
        func=mdp.hind_leg_extension_geom,
        weight=0.0,
        params={
            "hind_hip_body_ids": [9, 13],
            "hind_foot_body_ids": [12, 16],
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    hind_knee_extension = RewTerm(
        func=mdp.hind_knee_extension,
        weight=0.0,
        params={
            "hind_knee_joint_ids": [8, 11],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    hind_legs_calmness = RewTerm(
        func=mdp.hind_legs_calmness,
        weight=0.0,
        params={
            "hind_joint_ids": [7, 10, 8, 11],  # HL/HR hip and knee
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === Stand Still Rewards ===
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    stand_still_roll_only = RewTerm(
        func=mdp.stand_still_roll_only,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    stand_still_yaw_only = RewTerm(
        func=mdp.stand_still_yaw_only,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    stand_still_lin_x = RewTerm(
        func=mdp.stand_still_lin_x,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    stand_still_lin_y = RewTerm(
        func=mdp.stand_still_lin_y,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    stand_still_lin_z = RewTerm(
        func=mdp.stand_still_lin_z,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # === Base Height Rewards ===
    base_height_bonus = RewTerm(
        func=mdp.base_height_bonus,
        weight=0.0,
        params={
            "min_height": 0.45,
            "max_height": 0.75,
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === Safety Rewards ===
    deploy_posture_gate = RewTerm(
        func=mdp.deploy_posture_gate,
        weight=0.0,
        params={
            "roll_limit_rad": 0.7,  # ~40 degrees
            "pitch_limit_rad": 1.57,  # ~90 degrees
            "margin_deg": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    power = RewTerm(
        func=mdp.power,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    action_rate = RewTerm(
        func=mdp.action_rate,
        weight=0.0,
    )

    action_magnitude = RewTerm(
        func=mdp.action_magnitude,
        weight=0.0,
    )

    target_smoothness = RewTerm(
        func=mdp.target_smoothness,
        weight=0.0,
    )

    torques = RewTerm(
        func=mdp.torques,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_vel = RewTerm(
        func=mdp.dof_vel,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    dof_acc = RewTerm(
        func=mdp.dof_acc,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    feet_velocity = RewTerm(
        func=mdp.feet_velocity,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"])},
    )

    feet_contact_forces = RewTerm(
        func=mdp.feet_contact_forces,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]),
            "max_contact_force": 100.0,
        },
    )

    collision = RewTerm(
        func=mdp.collision,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH.*", ".*SHANK.*"]),
            "threshold": 0.1,
        },
    )

    torque_limits = RewTerm(
        func=mdp.torque_limits,
        weight=0.0,
        params={"soft_limit": 0.9, "asset_cfg": SceneEntityCfg("robot")},
    )

    dof_vel_limits = RewTerm(
        func=mdp.dof_vel_limits,
        weight=0.0,
        params={"soft_limit": 0.9, "asset_cfg": SceneEntityCfg("robot")},
    )

    # === Stability Metric ===
    two_leg_stand_metric = RewTerm(
        func=mdp.two_leg_stand_metric,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_body_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"]),
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    two_leg_stability_safe = RewTerm(
        func=mdp.two_leg_stability_safe,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_body_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"]),
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === Safe/Slow/Low-Power Curriculum Rewards ===
    two_leg_state_hold_bonus = RewTerm(
        func=mdp.two_leg_state_hold_bonus,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_body_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"]),
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "enter_threshold": 0.80,
            "exit_threshold": 0.70,
            "tau_hold": 60.0,
            "hold_cap": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    transition_dynamics_penalty = RewTerm(
        func=mdp.transition_dynamics_penalty,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_body_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"]),
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "enter_threshold": 0.80,
            "exit_threshold": 0.70,
            "hold_grace_steps": 20,
            "activation_metric_threshold": 0.45,
            "lin_vel_z_ref": 0.35,
            "ang_vel_xy_ref": 2.0,
            "dof_acc_ref": 80.0,
            "action_rate_ref": 0.35,
            "dyn_cap": 2.5,
            "w_lin_z": 1.0,
            "w_ang_xy": 1.0,
            "w_dof_acc": 0.1,
            "w_action_rate": 0.2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    effort_bundle_penalty = RewTerm(
        func=mdp.effort_bundle_penalty,
        weight=0.0,
        params={
            "torque_soft_limit": 0.9,
            "dof_vel_soft_limit": 0.9,
            "w_torque_limits": 1.0,
            "w_dof_vel_limits": 1.0,
            "w_power": 0.01,
            "w_action_magnitude": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    fall_after_stand_penalty = RewTerm(
        func=mdp.fall_after_stand_penalty,
        weight=0.0,
        params={
            "front_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "hind_feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "front_feet_body_cfg": SceneEntityCfg("robot", body_names=["FL_FOOT", "FR_FOOT"]),
            "pitch_tolerance": 0.35,
            "pitch_target": -1.22,
            "enter_threshold": 0.80,
            "exit_threshold": 0.70,
            "base_fall_penalty": 1.0,
            "hold_scale": 1.0,
            "hold_ref_steps": 120.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # === Standard Penalties (from base) ===
    action_rate_l2 = RewTerm(func=base_mdp.action_rate_l2, weight=0.0)

    joint_torques_l2 = RewTerm(
        func=base_mdp.joint_torques_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    joint_acc_l2 = RewTerm(
        func=base_mdp.joint_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    joint_vel_l2 = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )


@configclass
class TwoLegStandTerminationsCfg:
    """Termination terms for two-leg standing."""

    # Time out
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    # Illegal contact (torso)
    illegal_contact = DoneTerm(
        func=base_mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["TORSO"]),
            "threshold": 1.0,
        },
    )

    # Bad orientation (custom for two-leg standing)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation_two_leg,
        params={
            "roll_limit_rad": 0.7,  # ~40 degrees
            "pitch_limit_rad": 1.57,  # ~90 degrees
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    hind_contact = DoneTerm(
        func=mdp.hind_feet_off,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["HL_FOOT", "HR_FOOT"]),
            "threshold": 1.0,
            "min_steps_after_reset": 2,
        },
    )

    front_touch = DoneTerm(
        func=mdp.front_touch_termination,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_FOOT", "FR_FOOT"]),
            "threshold": 1.0,
        },
    )


@configclass
class TwoLegStandCurriculumCfg:
    """Curriculum terms for two-leg standing."""

    phases = CurrTerm(
        func=mdp.two_leg_stand_curriculum,
        params={
            "phases": mdp.get_two_leg_stand_phases(),
            "steps_per_env": 24,
            "log_curriculum": True,
            "front_touch_termination": {
                "enabled": False,
                "metrics": {
                    "human_posture": 0.0,
                    "front_legs_up_continuous": 0.0,
                },
                "log_enable": True,
            },
        },
    )


##
# Environment configuration
##


@configclass
class TwoLegStandEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the two-leg standing environment."""

    only_positive_rewards: bool = True
    num_privileged_obs: int = 54
    num_observation_history: int = 40

    # Scene settings
    scene: TwoLegStandSceneCfg = TwoLegStandSceneCfg(num_envs=2048, env_spacing=2.5)

    # Basic settings
    observations: TwoLegStandObservationsCfg = TwoLegStandObservationsCfg()
    actions: TwoLegStandActionsCfg = TwoLegStandActionsCfg()
    commands: TwoLegStandCommandsCfg = TwoLegStandCommandsCfg()

    # MDP settings
    rewards: TwoLegStandRewardsCfg = TwoLegStandRewardsCfg()
    terminations: TwoLegStandTerminationsCfg = TwoLegStandTerminationsCfg()
    events: TwoLegStandEventCfg = TwoLegStandEventCfg()
    curriculum: TwoLegStandCurriculumCfg = TwoLegStandCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 4
        self.episode_length_s = 5.0

        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # PhysX settings
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 1

        # Update sensor periods
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

    def disable_zero_weight_rewards(self):
        """Disable rewards with zero weight to save computation."""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr, None)
                if reward_attr is not None and hasattr(reward_attr, "weight"):
                    if reward_attr.weight == 0:
                        setattr(self.rewards, attr, None)
