import math
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from legged_gym.envs.base.two_leg_stand_config import TwoLegStandCfg


def quat_to_euler_xyz(q):
    """Convert [x, y, z, w] quaternion to roll, pitch, yaw (xyz convention)."""
    x, y, z, w = q
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def test_near_goal_pose_orientation():
    """Near-goal pose should pitch back ~65deg, not inverted."""
    q = TwoLegStandCfg.init_state.near_goal_state["rot"]
    roll, pitch, yaw = quat_to_euler_xyz(q)
    assert abs(roll) < math.radians(5), "Roll should be near zero"
    assert abs(yaw) < math.radians(5), "Yaw should be near zero"
    assert math.isclose(pitch, math.radians(65), abs_tol=math.radians(5)), "Pitch should lean back ~65deg"


def test_near_goal_pose_position():
    """Near-goal pose should start elevated."""
    z = TwoLegStandCfg.init_state.near_goal_state["pos"][2]
    assert z > 0.5, "Base height should be lifted for rear-stand spawn"


def test_near_goal_joint_targets():
    joints = TwoLegStandCfg.init_state.near_goal_state["default_joint_angles"]
    # Hind hips/knees extended back/down; front hips forward and knees tucked
    assert joints["HL_HipY_joint"] < -1.0 and joints["HR_HipY_joint"] < -1.0
    assert joints["HL_Knee_joint"] < 1.0 and joints["HR_Knee_joint"] < 1.0
    assert joints["FL_HipY_joint"] > 0.0 and joints["FR_HipY_joint"] > 0.0
    assert joints["FL_Knee_joint"] > 2.0 and joints["FR_Knee_joint"] > 2.0
