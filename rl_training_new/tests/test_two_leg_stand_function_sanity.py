from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS_PATH = (
    REPO_ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "two_leg_stand"
    / "mdp"
    / "observations.py"
)
EVENTS_PATH = (
    REPO_ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "two_leg_stand"
    / "mdp"
    / "events.py"
)


def test_observation_quaternion_order_matches_wxyz():
    text = OBSERVATIONS_PATH.read_text(encoding="utf-8")
    assert "w = quat[:, 0]" in text
    assert "x = quat[:, 1]" in text
    assert "y = quat[:, 2]" in text
    assert "z = quat[:, 3]" in text


def test_observation_joint_count_fallback_handles_slices():
    text = OBSERVATIONS_PATH.read_text(encoding="utf-8")
    assert "def _joint_count(" in text
    assert "if isinstance(joint_ids, slice):" in text
    assert "joint_dim = _joint_count(asset_cfg.joint_ids, asset.num_joints)" in text


def test_motor_strength_randomization_uses_default_or_cached_baseline():
    text = EVENTS_PATH.read_text(encoding="utf-8")
    assert "def _get_or_cache_default_tensor(" in text
    assert '"_motor_default_effort_limits"' in text
    assert '"_motor_default_joint_stiffness"' in text
    assert '"_motor_default_joint_damping"' in text
