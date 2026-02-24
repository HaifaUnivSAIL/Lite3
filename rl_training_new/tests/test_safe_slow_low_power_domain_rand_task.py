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
TWO_LEG_ENV_CFG_PATH = (
    REPO_ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "two_leg_stand"
    / "two_leg_stand_env_cfg.py"
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


def test_domain_rand_task_registration_present():
    text = TASK_REG_PATH.read_text(encoding="utf-8")
    assert 'id="TwoLegStandSafeSlowLowPowerDomainRand-Deeprobotics-Lite3-v0"' in text
    assert "Lite3TwoLegStandSafeSlowLowPowerDomainRandEnvCfg" in text
    assert "Lite3TwoLegStandSafeSlowLowPowerDomainRandPPORunnerCfg" in text


def test_domain_rand_env_cfg_split_randomization_contract():
    text = BASE_CFG_PATH.read_text(encoding="utf-8")
    assert "class Lite3TwoLegStandSafeSlowLowPowerDomainRandEnvCfg" in text
    assert "enable_environment_randomization: bool = False" in text
    assert '"LITE3_ENABLE_ENV_DOMAIN_RANDOMIZATION"' in text
    assert 'self.events.randomize_actuator_gains.mode = "reset"' in text
    assert 'self.events.randomize_motor_strength.mode = "reset"' in text
    assert "self.events.randomize_rigid_body_material = None" in text
    assert "self.events.randomize_gravity = None" in text
    assert "self.events.randomize_push_robot = None" in text


def test_domain_rand_runner_cfg_present():
    text = RUNNER_CFG_PATH.read_text(encoding="utf-8")
    assert "class Lite3TwoLegStandSafeSlowLowPowerDomainRandPPORunnerCfg" in text
    assert 'self.experiment_name = "two_leg_stand_safe_slow_low_power_domain_rand"' in text


def test_gravity_randomization_event_hook_present():
    env_cfg_text = TWO_LEG_ENV_CFG_PATH.read_text(encoding="utf-8")
    events_text = EVENTS_PATH.read_text(encoding="utf-8")
    assert "randomize_gravity = EventTerm(" in env_cfg_text
    assert "func=mdp.randomize_gravity" in env_cfg_text
    assert "def randomize_gravity(" in events_text
