import sys
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

    sys.modules.setdefault("gymnasium", gym)


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_WRAPPERS_PATH = REPO_ROOT / "source" / "rl_training" / "rl_training" / "utils" / "env_wrappers.py"
_spec = importlib.util.spec_from_file_location("lite3_env_wrappers_test_import", ENV_WRAPPERS_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ObservationHistoryWrapper = _mod.ObservationHistoryWrapper
RslRlCompatWrapper = _mod.RslRlCompatWrapper


class _DummyCompatEnv:
    def __init__(self, num_envs=2, num_obs=3, num_actions=2):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.num_policy_outputs = num_actions
        self.num_privileged_obs = 0
        self.max_episode_length = 100
        self.device = torch.device("cpu")
        self._step = 0
        self._done_by_step = {}

    def set_done_for_step(self, step: int, done_mask: torch.Tensor):
        self._done_by_step[step] = done_mask.to(dtype=torch.bool).clone()

    def _obs(self) -> torch.Tensor:
        env_id = torch.arange(self.num_envs, dtype=torch.float32).unsqueeze(1)
        obs_id = torch.arange(self.num_obs, dtype=torch.float32).unsqueeze(0)
        return env_id * 100.0 + obs_id + float(self._step)

    def get_observations(self):
        obs = self._obs()
        priv = torch.zeros((self.num_envs, 0), dtype=obs.dtype)
        return {"obs": obs, "privileged_obs": priv}

    def reset(self, **kwargs):
        self._step = 0
        return self.get_observations()

    def step(self, actions):
        self._step += 1
        obs = self.get_observations()
        reward = torch.zeros(self.num_envs, dtype=torch.float32)
        dones = self._done_by_step.get(self._step, torch.zeros(self.num_envs, dtype=torch.bool))
        info = {}
        return obs, reward, dones, info


class _DummyObsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, num_envs=2, num_obs=3):
        super().__init__()
        self.num_envs = num_envs
        self.num_obs = num_obs
        self._step = 0
        self._done_by_step = {}
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {"obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs,), dtype=np.float32)}
        )

    def set_done_for_step(self, step: int, done_mask: torch.Tensor):
        self._done_by_step[step] = done_mask.to(dtype=torch.bool).clone()

    def _obs(self) -> torch.Tensor:
        env_id = torch.arange(self.num_envs, dtype=torch.float32).unsqueeze(1)
        obs_id = torch.arange(self.num_obs, dtype=torch.float32).unsqueeze(0)
        return env_id * 100.0 + obs_id + float(self._step)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return {"obs": self._obs()}, {}

    def step(self, action):
        self._step += 1
        obs = {"obs": self._obs()}
        reward = torch.zeros(self.num_envs, dtype=torch.float32)
        terminated = self._done_by_step.get(self._step, torch.zeros(self.num_envs, dtype=torch.bool))
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        info = {}
        return obs, reward, terminated, truncated, info


def _hist_two_frames(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cat((a, b), dim=-1)


def _hist_zero_then_obs(obs: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros_like(obs), obs), dim=-1)


def test_rsl_compat_default_clears_history_on_done(monkeypatch):
    monkeypatch.delenv("LITE3_UNREALISTIC_HISTORY_FEED", raising=False)

    env = _DummyCompatEnv()
    env.set_done_for_step(1, torch.tensor([False, True]))
    wrapper = RslRlCompatWrapper(env, obs_history_length=2)

    reset_obs = wrapper.reset()["obs"].clone()
    out, _, dones, _ = wrapper.step(torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32))
    step_obs = out["obs"].clone()
    hist = out["obs_history"]

    assert torch.equal(dones, torch.tensor([False, True]))
    assert torch.allclose(hist[0], _hist_two_frames(reset_obs[0], step_obs[0]))
    assert torch.count_nonzero(hist[1]).item() == 0


def test_rsl_compat_legacy_mode_preserves_history_on_done(monkeypatch):
    monkeypatch.setenv("LITE3_UNREALISTIC_HISTORY_FEED", "1")

    env = _DummyCompatEnv()
    env.set_done_for_step(1, torch.tensor([False, True]))
    wrapper = RslRlCompatWrapper(env, obs_history_length=2)

    reset_obs = wrapper.reset()["obs"].clone()
    out, _, _, _ = wrapper.step(torch.zeros((env.num_envs, env.num_actions), dtype=torch.float32))
    step_obs = out["obs"].clone()
    hist = out["obs_history"]

    assert torch.allclose(hist[1], _hist_two_frames(reset_obs[1], step_obs[1]))


def test_observation_wrapper_default_clears_history_on_done(monkeypatch):
    monkeypatch.delenv("LITE3_UNREALISTIC_HISTORY_FEED", raising=False)

    env = _DummyObsEnv()
    env.set_done_for_step(1, torch.tensor([False, True]))
    wrapper = ObservationHistoryWrapper(env, obs_history_length=2, obs_key="obs")

    wrapper.reset()
    obs1, _, terminated, _, _ = wrapper.step(torch.zeros((env.num_envs, 1), dtype=torch.float32))
    step_obs = obs1["obs"].clone()
    hist = obs1["obs_history"]

    assert torch.equal(terminated, torch.tensor([False, True]))
    assert torch.allclose(hist[0], _hist_zero_then_obs(step_obs[0]))
    assert torch.count_nonzero(hist[1]).item() == 0


def test_observation_wrapper_legacy_mode_preserves_history_on_done(monkeypatch):
    monkeypatch.setenv("LITE3_UNREALISTIC_HISTORY_FEED", "1")

    env = _DummyObsEnv()
    env.set_done_for_step(1, torch.tensor([False, True]))
    wrapper = ObservationHistoryWrapper(env, obs_history_length=2, obs_key="obs")

    wrapper.reset()
    obs1, _, _, _, _ = wrapper.step(torch.zeros((env.num_envs, 1), dtype=torch.float32))
    step_obs = obs1["obs"].clone()
    hist = obs1["obs_history"]

    assert torch.allclose(hist[1], _hist_zero_then_obs(step_obs[1]))
