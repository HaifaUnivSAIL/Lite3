import math

from legged_gym.envs.base.two_leg_stand_still_safe_config import (
    TwoLegStandStillSafeCfg,
    TwoLegStandStillSafeCfgPPO,
)


def _standup_hy_knee(height: float, thigh_len: float = 0.20, shank_len: float = 0.21):
    """Match deploy standup_state.hpp geometry for a given base height."""
    hipy = -math.acos((thigh_len * thigh_len + height * height - shank_len * shank_len) / (2.0 * height * thigh_len))
    knee = math.pi - math.acos((thigh_len * thigh_len + shank_len * shank_len - height * height) / (2.0 * thigh_len * shank_len))
    return hipy, knee


_DEPLOY_STAND_HEIGHT = 0.30
_HIPY_STAND, _KNEE_STAND = _standup_hy_knee(_DEPLOY_STAND_HEIGHT)


class TwoLegStandDeployAlignedCfg(TwoLegStandStillSafeCfg):
    """Two-leg stand curriculum aligned to the deploy stand-up phase."""

    class init_state(TwoLegStandStillSafeCfg.init_state):
        # Use a deploy-aligned reset pose without repurposing near-goal init.
        deploy_reset_prob = 1.0
        deploy_reset_state = {
            "pos": [0.0, 0.0, _DEPLOY_STAND_HEIGHT],
            "rot": [
                -0.00023085526184233324,
                -0.0032073138974974646,
                -0.0019571690372445424,
                0.9999929146412841,
            ],
            "lin_vel": [0.0, 0.0, 0.0],
            "ang_vel": [0.0, 0.0, 0.0],
            "default_joint_angles": {
                "FL_HipX_joint": 0.0,
                "FR_HipX_joint": 0.0,
                "HL_HipX_joint": 0.0,
                "HR_HipX_joint": 0.0,
                "FL_HipY_joint": _HIPY_STAND,
                "FR_HipY_joint": _HIPY_STAND,
                "HL_HipY_joint": _HIPY_STAND,
                "HR_HipY_joint": _HIPY_STAND,
                "FL_Knee_joint": _KNEE_STAND,
                "FR_Knee_joint": _KNEE_STAND,
                "HL_Knee_joint": _KNEE_STAND,
                "HR_Knee_joint": _KNEE_STAND,
            },
        }
        deploy_reset_noise = {
            "pos": 0.0,
            "rot": 0.0,
            "lin_vel": 0.0,
            "ang_vel": 0.0,
            "joint": 0.0,
        }

    class noise(TwoLegStandStillSafeCfg.noise):
        add_noise = False

    class domain_rand(TwoLegStandStillSafeCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        randomize_com_offset = False
        randomize_motor_strength = False
        randomize_Kp_factor = False
        randomize_Kd_factor = False
        push_robots = False

class TwoLegStandDeployAlignedCfgPPO(TwoLegStandStillSafeCfgPPO):
    class runner(TwoLegStandStillSafeCfgPPO.runner):
        experiment_name = "two_leg_stand_deploy_aligned"
