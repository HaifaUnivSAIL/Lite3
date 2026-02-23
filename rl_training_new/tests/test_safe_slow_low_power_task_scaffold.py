from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_REG_PATH = (
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
    / "__init__.py"
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
RUNNER_CFG_PATH = (
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
    / "agents"
    / "rsl_rl_ppo_cfg.py"
)


def test_new_task_registration_present():
    text = TASK_REG_PATH.read_text(encoding="utf-8")
    assert 'id="TwoLegStandSafeSlowLowPower-Deeprobotics-Lite3-v0"' in text
    assert "Lite3TwoLegStandSafeSlowLowPowerEnvCfg" in text
    assert "Lite3TwoLegStandSafeSlowLowPowerPPORunnerCfg" in text


def test_new_env_cfg_disables_positive_only_clipping():
    text = BASE_CFG_PATH.read_text(encoding="utf-8")
    assert "class Lite3TwoLegStandSafeSlowLowPowerEnvCfg" in text
    assert "self.only_positive_rewards = False" in text


def test_new_runner_cfg_present():
    text = RUNNER_CFG_PATH.read_text(encoding="utf-8")
    assert "class Lite3TwoLegStandSafeSlowLowPowerPPORunnerCfg" in text
    assert 'self.experiment_name = "two_leg_stand_safe_slow_low_power"' in text
