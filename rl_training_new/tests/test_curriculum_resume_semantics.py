import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


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
_spec = importlib.util.spec_from_file_location("lite3_curriculums_test_import", CURRICULUMS_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_two_leg_stand_phases = _mod.get_two_leg_stand_phases
two_leg_stand_curriculum = _mod.two_leg_stand_curriculum


class _DummyTermCfg:
    def __init__(self):
        self.weight = 0.0


class _DummyRewardManager:
    def __init__(self):
        self._episode_sums = {}
        self._cfg = {}

    def get_term_cfg(self, name):
        if name not in self._cfg:
            self._cfg[name] = _DummyTermCfg()
        return self._cfg[name]


class _DummyEnv:
    def __init__(self):
        self.common_step_counter = 0
        self.reward_manager = _DummyRewardManager()
        self.max_episode_length_s = 1.0


def test_curriculum_uses_common_step_counter_without_progress_override():
    env = _DummyEnv()
    env.common_step_counter = 24 * 1100

    two_leg_stand_curriculum(
        env=env,
        env_ids=torch.tensor([0], dtype=torch.long),
        phases=get_two_leg_stand_phases(),
        steps_per_env=24,
    )

    assert env._two_leg_curriculum.current_phase_idx == 2
    assert env.curriculum_controller.current_phase == 2


def test_curriculum_resume_uses_runner_progress_buf_when_present():
    env = _DummyEnv()
    phases = get_two_leg_stand_phases()

    two_leg_stand_curriculum(
        env=env,
        env_ids=torch.tensor([0], dtype=torch.long),
        phases=phases,
        steps_per_env=24,
    )
    assert env._two_leg_curriculum.current_phase_idx == 0

    # Simulate OnPolicyRunner stamping resumed training iteration.
    env.curriculum_controller.get_progress_buf(3098)
    env.common_step_counter = 24

    two_leg_stand_curriculum(
        env=env,
        env_ids=torch.tensor([0], dtype=torch.long),
        phases=phases,
        steps_per_env=24,
    )

    assert env._two_leg_curriculum.current_phase_idx == 3
    assert env.curriculum_controller.current_phase == 3
