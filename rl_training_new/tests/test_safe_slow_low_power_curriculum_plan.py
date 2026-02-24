from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CURRICULUMS_PATH = (
    REPO_ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "two_leg_stand"
    / "mdp"
    / "curriculums.py"
)
BASE_CFG_PATH = (
    REPO_ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "two_leg_stand"
    / "config"
    / "lite3"
    / "base_env_cfg.py"
)
REWARDS_PATH = (
    REPO_ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "two_leg_stand"
    / "mdp"
    / "rewards.py"
)


def test_safe_slow_low_power_phase_factory_exists():
    text = CURRICULUMS_PATH.read_text(encoding="utf-8")
    assert "def get_two_leg_stand_safe_slow_low_power_phases()" in text


def test_safe_slow_low_power_phase_terms_present():
    text = CURRICULUMS_PATH.read_text(encoding="utf-8")
    assert '"two_leg_state_hold_bonus": 0.5' in text
    assert '"lin_vel_z": 0.0' in text
    assert '"torque_limits": 0.0' in text
    assert '"transition_dynamics_penalty": -1.0' in text
    assert '"effort_bundle_penalty": -1.3' in text
    assert '"fall_after_stand_penalty": -1.8' in text
    assert '"termination": -14.0' in text


def test_new_env_cfg_uses_safe_slow_low_power_phase_factory():
    text = BASE_CFG_PATH.read_text(encoding="utf-8")
    assert "class Lite3TwoLegStandSafeSlowLowPowerEnvCfg" in text
    assert 'self.curriculum.phases.params["phases"] = mdp.get_two_leg_stand_safe_slow_low_power_phases()' in text


def test_transition_dynamics_penalty_is_bounded_and_gated():
    text = REWARDS_PATH.read_text(encoding="utf-8")
    assert "activation_metric_threshold" in text
    assert 'posture_progress = components["hind_support"] * components["orientation_gate"] * components["height_gate"]' in text
    assert "proximity_gate = torch.clamp((posture_progress - start_threshold) / denom, min=0.0, max=1.0)" in text
    assert "dof_count = max(int(asset.data.joint_vel.shape[1]), 1)" in text
    assert "action_count = max(int(env.action_manager.action.shape[1]), 1)" in text
    assert "raw_dyn = torch.clamp(raw_dyn, max=float(dyn_cap))" in text
