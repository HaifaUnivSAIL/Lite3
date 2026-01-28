# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

from typing import Any

from collections.abc import Mapping

import numpy as np
import torch
import gymnasium as gym


class OnlyPositiveRewardsWrapper(gym.Wrapper):
    """Clip total rewards at zero before adding termination penalty."""

    def __init__(self, env: gym.Env, termination_reward_weight: float | None = None) -> None:
        super().__init__(env)
        self.termination_reward_weight = termination_reward_weight

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._clip_reward(reward, terminated)
        return obs, reward, terminated, truncated, info

    def _clip_reward(self, reward: Any, terminated: Any):
        if torch.is_tensor(reward):
            term_mask = _as_torch(terminated, reward)
            if self.termination_reward_weight is None:
                return torch.clamp(reward, min=0.0)
            term_reward = term_mask * float(self.termination_reward_weight)
            reward = torch.clamp(reward - term_reward, min=0.0)
            return reward + term_reward
        reward_np = np.asarray(reward)
        term_mask = _as_numpy(terminated, reward_np)
        if self.termination_reward_weight is None:
            return np.maximum(reward_np, 0.0)
        term_reward = term_mask * float(self.termination_reward_weight)
        reward_np = np.maximum(reward_np - term_reward, 0.0)
        return reward_np + term_reward


class ObservationHistoryWrapper(gym.Wrapper):
    """Track a fixed-length observation history for policy observations."""

    def __init__(self, env: gym.Env, obs_history_length: int, obs_key: str = "policy") -> None:
        super().__init__(env)
        self.obs_history_length = int(obs_history_length)
        self.obs_key = obs_key
        self.num_obs: int | None = None
        self.num_obs_history: int | None = None
        self.obs_history = None

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        policy_obs = self._extract_policy_obs(obs)
        self._ensure_buffers(policy_obs)
        self._clear_history()
        obs = self._attach_history(obs)
        return (obs, info) if isinstance(out, tuple) else obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        policy_obs = self._extract_policy_obs(obs)
        self._ensure_buffers(policy_obs)
        self._append_history(policy_obs)
        reset_mask = self._reset_mask(terminated, truncated)
        if reset_mask is not None:
            self._clear_history(reset_mask)
        obs = self._attach_history(obs)
        return obs, reward, terminated, truncated, info

    def _extract_policy_obs(self, obs):
        if isinstance(obs, dict):
            if self.obs_key in obs:
                return obs[self.obs_key]
            if "obs" in obs:
                return obs["obs"]
            return next(iter(obs.values()))
        return obs

    def _ensure_buffers(self, policy_obs):
        if self.obs_history is not None:
            return
        self.num_obs = int(policy_obs.shape[-1])
        self.num_obs_history = self.num_obs * self.obs_history_length
        if torch.is_tensor(policy_obs):
            self.obs_history = torch.zeros(
                (policy_obs.shape[0], self.num_obs_history),
                device=policy_obs.device,
                dtype=policy_obs.dtype,
            )
        else:
            self.obs_history = np.zeros((policy_obs.shape[0], self.num_obs_history), dtype=policy_obs.dtype)

    def _append_history(self, policy_obs):
        if torch.is_tensor(policy_obs):
            self.obs_history = torch.cat((self.obs_history[:, self.num_obs :], policy_obs), dim=-1)
        else:
            self.obs_history = np.concatenate((self.obs_history[:, self.num_obs :], policy_obs), axis=-1)

    def _clear_history(self, mask=None):
        if mask is None:
            self.obs_history[:] = 0
            return
        if torch.is_tensor(self.obs_history):
            self.obs_history[mask] = 0
        else:
            self.obs_history[np.asarray(mask, dtype=bool)] = 0

    def _attach_history(self, obs):
        if isinstance(obs, dict):
            obs_out = dict(obs)
            obs_out["obs_history"] = self.obs_history
            return obs_out
        return obs

    def _reset_mask(self, terminated, truncated):
        env = self.env.unwrapped
        if hasattr(env, "episode_length_buf") and env.episode_length_buf is not None:
            return env.episode_length_buf == 0
        if terminated is None and truncated is None:
            return None
        if torch.is_tensor(terminated) or torch.is_tensor(truncated):
            term = terminated if torch.is_tensor(terminated) else torch.zeros_like(truncated)
            trunc = truncated if torch.is_tensor(truncated) else torch.zeros_like(terminated)
            return term | trunc
        term = np.asarray(terminated, dtype=bool) if terminated is not None else None
        trunc = np.asarray(truncated, dtype=bool) if truncated is not None else None
        if term is None:
            return trunc
        if trunc is None:
            return term
        return np.logical_or(term, trunc)


def _as_torch(value, ref):
    if torch.is_tensor(value):
        return value.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


def _as_numpy(value, ref):
    return np.asarray(value, dtype=ref.dtype)


class RslRlCompatWrapper:
    """Compatibility wrapper for IsaacLab -> legacy rsl_rl interface."""

    def __init__(
        self,
        env,
        obs_history_length: int = 0,
        only_positive_rewards: bool = False,
        termination_reward_weight: float | None = None,
    ) -> None:
        self.env = env
        # Forward curriculum controller onto the immediate wrapper if any inner env provides it.
        self._curriculum_controller = _find_attr(self.env, "curriculum_controller")
        if self._curriculum_controller is not None:
            try:
                setattr(self.env, "curriculum_controller", self._curriculum_controller)
            except Exception:
                pass
        self.obs_history_length = int(obs_history_length or 0)
        self.only_positive_rewards = bool(only_positive_rewards)
        self.termination_reward_weight = termination_reward_weight

        # Cache common attributes from underlying env if present.
        self.num_envs = getattr(env, "num_envs", None)
        self.num_actions = getattr(env, "num_actions", None)
        self.num_policy_outputs = getattr(env, "num_policy_outputs", self.num_actions)
        self.max_episode_length = getattr(env, "max_episode_length", None)
        self.device = getattr(env, "device", None)

        env_num_priv = getattr(env, "num_privileged_obs", None)
        self.num_privileged_obs = int(env_num_priv) if env_num_priv is not None else 0

        self.obs_history = None
        self._last_obs_dict = None

        # Prime buffers to infer dimensions.
        obs_dict = self._ensure_obs_dict(self._get_initial_obs())
        self._init_shapes(obs_dict)
        self._last_obs_dict = obs_dict

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def curriculum_controller(self):
        return self._curriculum_controller

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs_dict = self._ensure_obs_dict(obs)
        self._clear_history()
        obs_dict = self._attach_history(obs_dict)
        obs_dict = self._ensure_tensor_obs_dict(obs_dict)
        self._last_obs_dict = obs_dict
        return obs_dict

    def step(self, actions):
        out = self.env.step(actions)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            dones = _merge_terminated_truncated(terminated, truncated)
        elif isinstance(out, tuple) and len(out) == 4:
            obs, reward, dones, info = out
        else:
            raise RuntimeError(f"Unexpected env.step() output format: {type(out)}")

        obs_dict = self._ensure_obs_dict(obs)
        obs_dict = self._attach_history(obs_dict)
        obs_dict = self._ensure_tensor_obs_dict(obs_dict)
        reward = self._clip_reward(reward, dones)
        self._last_obs_dict = obs_dict
        return obs_dict, reward, dones, info

    def get_observations(self):
        if hasattr(self.env, "get_observations"):
            obs = self.env.get_observations()
        elif self._last_obs_dict is not None:
            return self._last_obs_dict
        else:
            obs = self.reset()
        obs_dict = self._ensure_obs_dict(obs)
        obs_dict = self._attach_history(obs_dict)
        obs_dict = self._ensure_tensor_obs_dict(obs_dict)
        self._last_obs_dict = obs_dict
        return obs_dict

    def get_privileged_observations(self):
        obs_dict = self._last_obs_dict or self.get_observations()
        return obs_dict.get("privileged_obs")

    def _get_initial_obs(self):
        if hasattr(self.env, "get_observations"):
            return self.env.get_observations()
        return self.env.reset()

    def _ensure_obs_dict(self, obs):
        # Handle gymnasium reset signature.
        if isinstance(obs, tuple) and len(obs) == 2:
            obs = obs[0]

        if isinstance(obs, Mapping):
            if "obs" in obs:
                policy_obs = obs["obs"]
                privileged_obs = obs.get("privileged_obs", obs.get("critic", obs.get("privileged")))
                obs_history = obs.get("obs_history")
            elif "policy" in obs:
                policy_obs = obs["policy"]
                privileged_obs = obs.get("critic", obs.get("privileged"))
                obs_history = obs.get("obs_history")
            else:
                # Fallback: take the first value as policy obs.
                policy_obs = next(iter(obs.values()))
                privileged_obs = obs.get("privileged_obs", obs.get("critic", obs.get("privileged")))
                obs_history = obs.get("obs_history")
        else:
            policy_obs = obs
            privileged_obs = None
            obs_history = None

        if privileged_obs is None:
            privileged_obs = _zeros_like_obs(policy_obs, int(self.num_privileged_obs or 0))

        return {
            "obs": _to_tensor_if_numpy(policy_obs),
            "privileged_obs": _to_tensor_if_numpy(privileged_obs),
            "obs_history": _to_tensor_if_numpy(obs_history),
        }

    def _init_shapes(self, obs_dict):
        obs = obs_dict["obs"]
        self.num_obs = int(obs.shape[-1])
        priv = obs_dict.get("privileged_obs")
        if priv is not None:
            self.num_privileged_obs = int(priv.shape[-1])
        else:
            env_priv = getattr(self.env, "num_privileged_obs", None)
            self.num_privileged_obs = int(env_priv) if env_priv is not None else 0
        if obs_dict.get("obs_history") is not None:
            self.num_obs_history = int(obs_dict["obs_history"].shape[-1])
        else:
            self.num_obs_history = self.num_obs * self.obs_history_length

        if self.obs_history_length > 0 and obs_dict.get("obs_history") is None:
            self._init_history_buffer(obs)

        if obs_dict.get("privileged_obs") is None:
            obs_dict["privileged_obs"] = _zeros_like_obs(obs, self.num_privileged_obs)
        if obs_dict.get("obs_history") is None:
            if self.obs_history_length > 0:
                if self.obs_history is None:
                    self._init_history_buffer(obs)
                obs_dict["obs_history"] = self.obs_history
            else:
                obs_dict["obs_history"] = _zeros_like_obs(obs, 0)
                self.obs_history = obs_dict["obs_history"]

        if self.num_envs is None:
            self.num_envs = int(obs.shape[0])
        if self.device is None and torch.is_tensor(obs):
            self.device = obs.device
        if self.num_actions is None:
            action_space = getattr(self.env, "action_space", None)
            if action_space is not None and hasattr(action_space, "shape") and action_space.shape:
                self.num_actions = int(action_space.shape[-1])
        if self.num_policy_outputs is None:
            self.num_policy_outputs = self.num_actions

    def _init_history_buffer(self, obs):
        if torch.is_tensor(obs):
            self.obs_history = torch.zeros(
                (obs.shape[0], self.num_obs_history),
                device=obs.device,
                dtype=obs.dtype,
            )
        else:
            self.obs_history = np.zeros((obs.shape[0], self.num_obs_history), dtype=obs.dtype)

    def _clear_history(self):
        if self.obs_history is None:
            return
        if torch.is_tensor(self.obs_history):
            self.obs_history.zero_()
        else:
            self.obs_history[:] = 0

    def _append_history(self, obs):
        if self.obs_history is None:
            self._init_history_buffer(obs)
        if torch.is_tensor(obs):
            self.obs_history = torch.cat((self.obs_history[:, self.num_obs :], obs), dim=-1)
        else:
            self.obs_history = np.concatenate((self.obs_history[:, self.num_obs :], obs), axis=-1)

    def _attach_history(self, obs_dict):
        if obs_dict.get("obs_history") is not None:
            return obs_dict
        if self.obs_history_length <= 0:
            obs_dict["obs_history"] = _zeros_like_obs(obs_dict["obs"], 0)
            return obs_dict
        self._append_history(obs_dict["obs"])
        obs_dict["obs_history"] = self.obs_history
        return obs_dict

    def _ensure_tensor_obs_dict(self, obs_dict):
        obs_dict["obs"] = _to_tensor_if_numpy(obs_dict.get("obs"))
        obs_dict["privileged_obs"] = _to_tensor_if_numpy(obs_dict.get("privileged_obs"))
        obs_dict["obs_history"] = _to_tensor_if_numpy(obs_dict.get("obs_history"))
        return obs_dict

    def _clip_reward(self, reward, terminated):
        if not self.only_positive_rewards:
            return reward
        if torch.is_tensor(reward):
            term_mask = _as_torch(terminated, reward)
            if self.termination_reward_weight is None:
                return torch.clamp(reward, min=0.0)
            term_reward = term_mask * float(self.termination_reward_weight)
            reward = torch.clamp(reward - term_reward, min=0.0)
            return reward + term_reward
        reward_np = np.asarray(reward)
        term_mask = _as_numpy(terminated, reward_np)
        if self.termination_reward_weight is None:
            return np.maximum(reward_np, 0.0)
        term_reward = term_mask * float(self.termination_reward_weight)
        reward_np = np.maximum(reward_np - term_reward, 0.0)
        return reward_np + term_reward


def _merge_terminated_truncated(terminated, truncated):
    if torch.is_tensor(terminated) or torch.is_tensor(truncated):
        term = terminated if torch.is_tensor(terminated) else torch.zeros_like(truncated)
        trunc = truncated if torch.is_tensor(truncated) else torch.zeros_like(terminated)
        return term | trunc
    term = np.asarray(terminated, dtype=bool) if terminated is not None else None
    trunc = np.asarray(truncated, dtype=bool) if truncated is not None else None
    if term is None:
        return trunc
    if trunc is None:
        return term
    return np.logical_or(term, trunc)


def _zeros_like_obs(obs, dim: int):
    if torch.is_tensor(obs):
        return torch.zeros((obs.shape[0], dim), device=obs.device, dtype=obs.dtype)
    dtype = obs.dtype
    if isinstance(dtype, torch.dtype):
        dtype = torch.zeros((), dtype=dtype).numpy().dtype
    return np.zeros((obs.shape[0], dim), dtype=dtype)


def _to_tensor_if_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value


def _find_attr(obj, name: str, max_depth: int = 6):
    cur = obj
    for _ in range(max_depth):
        if cur is None:
            break
        if hasattr(cur, name):
            return getattr(cur, name)
        unwrapped = getattr(cur, "unwrapped", None)
        if unwrapped is not None and unwrapped is not cur and hasattr(unwrapped, name):
            return getattr(unwrapped, name)
        cur = getattr(cur, "env", None)
    return None
