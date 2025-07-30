import os
currentdir = os.path.dirname(os.path.abspath(__file__))
legged_gym_dir = os.path.dirname(os.path.dirname(currentdir))
isaacgym_dir = os.path.join(os.path.dirname(legged_gym_dir), "isaacgym/python")
rsl_rl_dir = os.path.join(os.path.dirname(legged_gym_dir), "rsl_rl")
os.sys.path.insert(0, legged_gym_dir)
os.sys.path.insert(0, isaacgym_dir)
os.sys.path.insert(0, rsl_rl_dir)

import isaacgym
import numpy as np
import torch
from legged_gym.utils import get_args, register

def quat_to_euler_xyz(quat):
    """Convert quaternion to euler angles [roll, pitch, yaw]"""
    x, y, z, w = quat.unbind(dim=-1)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack([roll, pitch, yaw], dim=-1)

def play(args):
    from legged_gym.utils.task_registry import task_registry
    register(args.task, task_registry)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 10
    env_cfg.viewer.real_time_step = True
    env_cfg.pmtg.train_mode = False
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.fixed_commands = [0.0, 0.0, 0.0]
    env_cfg.env.episode_length_s = 100

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs_dict = env.get_observations()
    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]

    num_envs = env.num_envs
    total_steps = 0
    standing_counts = torch.zeros(num_envs, device=env.device)
    upright_counts = torch.zeros(num_envs, device=env.device)
    front_contact_counts = torch.zeros(num_envs, device=env.device)
    longest_streaks = torch.zeros(num_envs, device=env.device)
    current_streaks = torch.zeros(num_envs, device=env.device)

    lin_vel_acc = torch.zeros(num_envs, device=env.device)
    ang_vel_acc = torch.zeros(num_envs, device=env.device)

    IDENTITY_QUAT = torch.tensor([0, 0, 0, 1], device=env.device)

    for _ in range(10 * int(env.max_episode_length)):
        with torch.no_grad():
            actions = policy(obs, obs_history)
        obs_dict, rews, dones, infos = env.step(actions)
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]

        base_quat = env.base_quat  # [N, 4]
        quat_dot = torch.sum(base_quat * IDENTITY_QUAT, dim=1).abs()
        upright = quat_dot > 0.95  # equivalent to angle < ~36°
        upright_counts += upright.int()

        contacts = env.contact_filt.int()
        rear_contacts = contacts[:, 2] + contacts[:, 3]
        front_contacts = contacts[:, 0] + contacts[:, 1]
        standing = (rear_contacts == 2) & (front_contacts == 0)
        standing_counts += standing.int()
        front_contact_counts += (front_contacts > 0).int()

        current_streaks = torch.where(standing, current_streaks + 1, torch.zeros_like(current_streaks))
        longest_streaks = torch.max(longest_streaks, current_streaks)

        root_states = env.root_states
        lin_vel = torch.norm(root_states[:, 7:10], dim=1)
        ang_vel = torch.norm(root_states[:, 10:13], dim=1)
        lin_vel_acc += lin_vel
        ang_vel_acc += ang_vel

        total_steps += 1

    def avg(tensor): return tensor.cpu().numpy() / total_steps
    def stat(tensor): return f"{tensor.mean():.2f} ± {tensor.std():.2f}"

    print("\n===== Standing Evaluation Summary =====")
    print(f"✅ % Time Standing        : {stat(avg(standing_counts * 100))}")
    print(f"🧍 % Time Upright         : {stat(avg(upright_counts * 100))}")
    print(f"👣 # Front Foot Contacts   : {stat(front_contact_counts)}")
    print(f"🕒 Longest Standing Streak: {stat(longest_streaks)}")
    print(f"📉 Mean Linear Velocity   : {stat(lin_vel_acc)}")
    print(f"🌀 Mean Angular Velocity  : {stat(ang_vel_acc)}")
    print("========================================\n")

if __name__ == '__main__':
    args = get_args()
    play(args)
