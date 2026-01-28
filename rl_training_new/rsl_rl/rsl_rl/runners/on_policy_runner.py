# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2023, HUAWEI TECHNOLOGIES

import time
import os
from collections import deque
import statistics
import csv
from legged_gym.utils.helpers import get_load_path
from legged_gym import LEGGED_GYM_ROOT_DIR

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv, HistoryWrapper


class OnPolicyRunner:

    def __init__(self,
                 env: HistoryWrapper,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 save_rewards=False,
                 enable_summary_writer=False):

        # Accept both legacy (runner/algorithm/policy) and flat IsaacLab-style configs.
        if "runner" not in train_cfg:
            runner_cfg = {k: v for k, v in train_cfg.items() if k not in ("algorithm", "policy")}
            train_cfg = {
                "runner": runner_cfg,
                "algorithm": train_cfg.get("algorithm", {}),
                "policy": train_cfg.get("policy", {}),
            }

        self.cfg = _as_dict(train_cfg.get("runner", {}))
        self.alg_cfg = _as_dict(train_cfg.get("algorithm", {}))
        self.policy_cfg = _as_dict(train_cfg.get("policy", {}))

        # Fill required defaults expected by legacy runner config.
        policy_class_name = self.cfg.get("policy_class_name") or self.policy_cfg.pop("class_name", None)
        alg_class_name = self.cfg.get("algorithm_class_name") or self.alg_cfg.pop("class_name", None)
        if policy_class_name is not None:
            self.cfg["policy_class_name"] = policy_class_name
        if alg_class_name is not None:
            self.cfg["algorithm_class_name"] = alg_class_name
        self.cfg.setdefault("policy_class_name", "ActorCritic")
        self.cfg.setdefault("algorithm_class_name", "PPO")
        if "checkpoint" not in self.cfg and "load_checkpoint" in self.cfg:
            self.cfg["checkpoint"] = self.cfg["load_checkpoint"]
        self.cfg.setdefault("num_steps_per_env", 24)
        self.cfg.setdefault("save_interval", 500)
        self.cfg.setdefault("resume", False)
        self.cfg.setdefault("load_run", "")
        self.device = device
        self.env = env
        self.save_rewards = save_rewards
        self.csv_header = None

        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        policy_kwargs = _filter_kwargs(self.policy_cfg, actor_critic_class)
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs,
            self.env.num_privileged_obs,
            self.env.num_obs_history,
            self.env.num_policy_outputs,
            **policy_kwargs,
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        alg_kwargs = _filter_kwargs(self.alg_cfg, alg_class)
        self.alg: PPO = alg_class(actor_critic, device=self.device, **alg_kwargs)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_policy_outputs])

        # Log
        self.log_dir = log_dir
        if self.log_dir:
            self.exported_path = os.path.join(self.log_dir, "exported")
            os.makedirs(self.exported_path, exist_ok=True)
        else:
            self.exported_path = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        if self.cfg['resume']:
            # load previously trained model
            load_root = os.path.dirname(log_dir) if log_dir else None
            load_experiment = self.cfg.get("load_experiment")
            if load_experiment:
                load_root = load_experiment if os.path.isabs(load_experiment) else os.path.join(
                    LEGGED_GYM_ROOT_DIR, "logs", load_experiment
                )
            if not load_root:
                raise ValueError("Cannot resume without a log_dir or runner.load_experiment")
            resume_path = get_load_path(load_root,
                                        load_run=self.cfg['load_run'],
                                        checkpoint=self.cfg['checkpoint'])  # last one
            print(f"Loading model from: {resume_path}")
            print(resume_path)
            self.load(resume_path)
            
        if enable_summary_writer:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            self.writer = None
        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs_dict = self.env.get_observations()
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        best_reward = 0
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        curriculum = None
        for candidate in (self.env, getattr(self.env, "env", None), getattr(self.env, "unwrapped", None)):
            if candidate is not None and hasattr(candidate, "curriculum_controller"):
                curriculum = candidate.curriculum_controller
                break

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, obs_history)
                    if curriculum is not None:
                        curriculum.get_progress_buf(buf_element=it)
                    obs_dict, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]
                    obs, privileged_obs, obs_history, rewards, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs, privileged_obs)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if self.save_interval != -1 and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), iteration=it)
                # Save TorchScript version for ONNX export
                scripted = torch.jit.script(self.alg.actor_critic.export_policy())
                
                scripted.save(os.path.join(self.exported_path, 'model_{}.pt'.format(it)))
            if rewbuffer and statistics.mean(rewbuffer) > best_reward:
                best_reward = statistics.mean(rewbuffer)
                self.save(os.path.join(self.log_dir, 'model_best.pt'.format(it)), iteration=it)
                # Save TorchScript version for ONNX export
                scripted = torch.jit.script(self.alg.actor_critic.export_policy())
                scripted.save(os.path.join(self.exported_path, 'model_best.pt'))
            ep_infos.clear()

            env_for_curriculum = None
            for candidate in (self.env, getattr(self.env, "env", None), getattr(self.env, "unwrapped", None)):
                if candidate is not None and hasattr(candidate, "curriculum_factor"):
                    env_for_curriculum = candidate
                    break
            if env_for_curriculum is not None and hasattr(env_for_curriculum, "cfg"):
                try:
                    convergence_rate = env_for_curriculum.cfg.env.convergence_rate
                    env_for_curriculum.curriculum_factor = pow(env_for_curriculum.curriculum_factor, convergence_rate)
                    if self.writer is not None:
                        self.writer.add_scalar('Episode/' + 'curriculum_factor', env_for_curriculum.curriculum_factor, it)
                except Exception:
                    pass

                try:
                    if env_for_curriculum.cfg.noise.heights_gaussian_mean_mutable:
                        env_for_curriculum.height_noise_mean = torch.distributions.uniform.Uniform(-0.03, 0.03).sample()
                except Exception:
                    pass

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)),
                  iteration=self.current_learning_iteration)
        # Save TorchScript version for ONNX export
        scripted = torch.jit.script(self.alg.actor_critic.export_policy())
        scripted.save(os.path.join(self.exported_path, 'model_{}.pt'.format(self.current_learning_iteration)))

        if self.save_rewards is True:
            from legged_gym.scripts.plot_reward import save_reward_csv
            save_reward_csv(os.path.join(self.log_dir, 'rewards.csv'), dt=self.env.dt)

    def _get_active_reward_names(self):
        env = self.env.env if hasattr(self.env, "env") else self.env
        if env is None:
            return None
        reward_scales = getattr(env, "reward_scales", None)
        base_active = None
        if reward_scales is not None:
            base_active = {name for name, scale in reward_scales.items() if scale != 0}
        curriculum = getattr(env, "curriculum_controller", None)
        if curriculum is not None and getattr(curriculum, "enabled", False):
            current_scales = getattr(curriculum, "current_scales", None)
            if current_scales:
                active = {name for name, scale in current_scales.items() if scale != 0}
                if reward_scales is not None and reward_scales.get("termination", 0) != 0:
                    active.add("termination")
                return active
        return base_active

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        # Get curriculum phase name
        near_goal_prob = None
        try:
            cc = self.env.env.curriculum_controller
            phase_idx = cc.current_phase
            phase_name = cc.phases[phase_idx]["name"]
            near_goal_prob = getattr(self.env.env, "goal_state_prob", None)
        except Exception:
            phase_name = "N/A"
            try:
                near_goal_prob = getattr(self.env.env, "goal_state_prob", None)
            except Exception:
                near_goal_prob = None

        if near_goal_prob is None:
            goal_prob_line = f"""{'Near goal init prob:':>{pad}} N/A\n"""
        else:
            goal_prob_line = f"""{'Near goal init prob:':>{pad}} {near_goal_prob:.2f}\n"""

        ep_string = f''
        if locs['ep_infos']:
            active_reward_names = self._get_active_reward_names()
            if self.save_rewards is True and self.csv_header is None:
                self.csv_header = [key for key in locs['ep_infos'][0]] + ['total_reward']
                with open(os.path.join(self.log_dir, 'rewards.csv'), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_header)
                    self.exported_path = os.path.join(self.log_dir, 'exported')
                    os.makedirs(self.exported_path, exist_ok=True)
            reward_row = []
            for key in locs['ep_infos'][0]:  # each reward terms
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:  # num_steps_per_env
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)  # total sub-reward for num_steps_per_env steps
                if self.writer is not None:
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                reward_row.append(value.cpu().numpy())  # record rewards
                should_print = True
                if key.startswith("rew_") and active_reward_names is not None:
                    reward_name = key[4:]
                    should_print = reward_name in active_reward_names
                if should_print:
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

            # write each reward item into a csv file
            reward_row.append(statistics.mean(locs['rewbuffer']) if len(locs['rewbuffer']) != 0 else 0.)
            if self.save_rewards is True:
                assert len(reward_row) == len(self.csv_header)
                with open(os.path.join(self.log_dir, 'rewards.csv'), 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(reward_row)

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        if self.writer is not None:
            self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
            self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
            self.writer.add_scalar('Loss/adaptation', locs['mean_adaptation_loss'], locs['it'])
            self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
            self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
            self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
            self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
            self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
            if len(locs['rewbuffer']) > 0:
                self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
                self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
                self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
                self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']),
                                       self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Curriculum phase:':>{pad}} {phase_name}\n"""  # <-- Now prints phase name
                          f"""{goal_prob_line}"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Curriculum phase:':>{pad}} {phase_name}\n"""  # <-- Now prints phase name
                          f"""{goal_prob_line}"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] - self.current_learning_iteration + 1) * (
                               self.current_learning_iteration + locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
                      )
        print(log_string)

    def save(self, path, infos=None, iteration=None):
        if iteration is None:
            iteration = self.current_learning_iteration
        torch.save(
            {
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'iter': iteration,
                'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def add_git_repo_to_log(self, file_path: str) -> None:
        """Optionally log git state for reproducibility."""
        if self.log_dir is None:
            return
        try:
            import os
            import subprocess

            repo_dir = os.path.dirname(os.path.abspath(file_path))
            top_level = subprocess.check_output(
                ["git", "-C", repo_dir, "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            if not top_level:
                return

            head = subprocess.check_output(
                ["git", "-C", top_level, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip()
            status = subprocess.check_output(
                ["git", "-C", top_level, "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
            )

            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, "git_state.txt"), "w", encoding="utf-8") as handle:
                handle.write(f"repo: {top_level}\n")
                handle.write(f"head: {head}\n")
                handle.write("status:\n")
                handle.write(status)
        except Exception:
            # Best-effort logging only.
            return


def _as_dict(cfg):
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    return {}


def _filter_kwargs(kwargs, cls):
    """Drop unsupported kwargs for legacy classes."""
    if not isinstance(kwargs, dict):
        return {}
    try:
        import inspect

        sig = inspect.signature(cls.__init__)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
            return dict(kwargs)
        allowed = {name for name in sig.parameters if name not in ("self",)}
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)
