# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# MDP components for two-leg standing task.

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .rewards import (
    # Torso upright rewards
    torso_upright,
    torso_upright_soften,
    torso_upright_warmup,
    torso_upright_continuous,
    # Front legs up rewards
    front_legs_up,
    front_legs_up_warmup,
    front_legs_up_continuous,
    front_legs_up_warmup_safe,
    front_legs_up_continuous_safe,
    front_tap_penalty,
    # Human posture rewards
    human_posture,
    human_posture_warmup,
    # Hind leg rewards
    hind_leg_extension_geom,
    hind_knee_extension,
    hind_legs_calmness,
    # Stand still rewards
    stand_still,
    stand_still_roll_only,
    stand_still_yaw_only,
    stand_still_lin_x,
    stand_still_lin_y,
    stand_still_lin_z,
    # Base height rewards
    base_height_bonus,
    # Safety rewards
    deploy_posture_gate,
    torque_limits,
    dof_vel_limits,
    power,
    action_rate,
    action_magnitude,
    target_smoothness,
    torques,
    dof_vel,
    dof_acc,
    lin_vel_z,
    ang_vel_xy,
    feet_velocity,
    feet_contact_forces,
    collision,
    # Stability metric
    two_leg_stand_metric,
    two_leg_stability_safe,
)

from .events import (
    reset_to_near_goal_state,
    reset_to_deploy_state,
    randomize_motor_strength,
    push_robots,
    bad_orientation_two_leg,
    hind_feet_off,
    front_touch_termination,
)

from .curriculums import (
    TwoLegStandCurriculumManager,
    CurriculumPhase,
    get_two_leg_stand_phases,
    get_two_leg_stand_still_phases,
    get_two_leg_stand_still_v2_phases,
    get_two_leg_stand_safe_phases,
    get_two_leg_stand_deploy_r1_phases,
    get_two_leg_stand_robust_phases,
    two_leg_stand_curriculum,
)

from .observations import (
    base_rpy,
    joint_pos,
    joint_vel,
    joint_pos_history,
    joint_vel_history,
    action_history,
    contact_states,
    friction_coeffs,
    external_wrench,
    mass_payload,
    com_displacement,
    motor_strength_factors,
    kp_factors,
    kd_factors,
)

# Reuse the velocity command with thresholding used in locomotion tasks.
from rl_training.tasks.manager_based.locomotion.velocity.mdp.commands import (  # noqa: E402
    UniformThresholdVelocityCommand,
    UniformThresholdVelocityCommandCfg,
)
