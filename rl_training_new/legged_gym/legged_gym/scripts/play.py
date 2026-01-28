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
import csv
from legged_gym.utils import get_args, Logger, register


def play(args):
    from legged_gym.utils.task_registry import task_registry
    from legged_gym import LEGGED_GYM_ROOT_DIR
    import torch
    record_policy_output = False
    register(args.task, task_registry)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = args.num_envs or 10

    env_cfg.viewer.real_time_step = True
    env_cfg.pmtg.train_mode = False
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # env_cfg.terrain.evaluation_mode = True

    # customized terrain mode
    # env_cfg.terrain.selected = True
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.commands.fixed_commands = [0.8, 0.0, 0.0]
    fixed_cmd_env = os.getenv("LITE3_FIXED_CMD")
    if fixed_cmd_env:
        token = fixed_cmd_env.strip().lower()
        if token in ("none", "off", "disable"):
            env_cfg.commands.fixed_commands = None
        else:
            parts = fixed_cmd_env.replace(",", " ").split()
            if len(parts) >= 3:
                try:
                    env_cfg.commands.fixed_commands = [float(parts[0]), float(parts[1]), float(parts[2])]
                except ValueError:
                    print(f"[play.py] Invalid LITE3_FIXED_CMD='{fixed_cmd_env}', keeping default.")

    if os.getenv("LITE3_DISABLE_NEAR_GOAL", "0") not in ("0", "", "false", "False"):
        env_cfg.init_state.near_goal_init_prob = 0.0

    if os.getenv("LITE3_DISABLE_DOMAIN_RAND", "0") not in ("0", "", "false", "False"):
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.randomize_com_offset = False
        env_cfg.domain_rand.randomize_motor_strength = False
        env_cfg.domain_rand.randomize_Kp_factor = False
        env_cfg.domain_rand.randomize_Kd_factor = False
        env_cfg.domain_rand.push_robots = False
    # env_cfg.viewer.debug_viz = True
    # env_cfg.terrain.terrain_length = 8
    # env_cfg.terrain.terrain_width = 8
    # env_cfg.terrain.num_rows = 6
    # env_cfg.terrain.num_cols = 2
    env_cfg.env.episode_length_s = 100
    # env_cfg.terrain.slope_treshold = 0.5  # for stair generation
    # env_cfg.terrain.terrain_kwargs = {'type': 'sloped_terrain', 'slope': 0.26}
    # env_cfg.terrain.terrain_kwargs = [{'type': 'slope_platform_stairs_terrain', 'slope': 0.36, 'step_width': 0.2, 'step_height': 0.1, 'num_steps': 5}]
    # env_cfg.terrain.terrain_kwargs = [{'type': 'slope_platform_stairs_terrain', 'slope': 0.36, 'step_width': 0.2, 'step_height': 0.1, 'num_steps': 5},
    #                                   {'type': 'stairs_platform_slope_terrain', 'step_width': 0.2, 'step_height': 0.1, 'num_steps': 5, 'slope': 0.36}]
    # env_cfg.terrain.terrain_kwargs = [{
    #     'type': 'pyramid_stairs_terrain',
    #     'step_width': 0.3,
    #     'step_height': -0.1,
    #     'platform_size': 3.
    # }, {
    #     'type': 'pyramid_stairs_terrain',
    #     'step_width': 0.3,
    #     'step_height': 0.1,
    #     'platform_size': 3.
    # }, {
    #     'type': 'pyramid_sloped_terrain',
    #     'slope': 0.26
    # }, {
    #     'type': 'discrete_obstacles_terrain',
    #     'max_height': 0.10,
    #     'min_size': 0.1,
    #     'max_size': 0.5,
    #     'num_rects': 200
    # }, {
    #     'type': 'wave_terrain',
    #     'num_waves': 4,
    #     'amplitude': 0.15
    # }, {
    #     'type': 'stepping_stones_terrain',
    #     'stone_size': 0.1,
    #     'stone_distance': 0.,
    #     'max_height': 0.03
    # }]
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if os.getenv("LITE3_DEBUG_DEFAULT_RESET", "0") not in ("0", "", "false", "False"):
        try:
            from isaacgym import gymtorch
        except Exception as exc:
            print(f"[play.py] Failed to import gymtorch for debug reset: {exc}")
        else:
            base_env = getattr(env, "env", env)
            if hasattr(base_env, "gym") and hasattr(base_env, "dof_state"):
                env_ids = torch.arange(base_env.num_envs, device=base_env.device)
                base_env.dof_pos[env_ids] = base_env.default_dof_pos
                base_env.dof_vel[env_ids] = 0.0
                base_env.root_states[env_ids] = base_env.base_init_state
                base_env.root_states[env_ids, :3] += base_env.env_origins[env_ids]
                base_env.root_states[env_ids, 7:13] = 0.0
                env_ids_int32 = env_ids.to(dtype=torch.int32)
                base_env.gym.set_dof_state_tensor_indexed(
                    base_env.sim,
                    gymtorch.unwrap_tensor(base_env.dof_state),
                    gymtorch.unwrap_tensor(env_ids_int32),
                    len(env_ids_int32),
                )
                base_env.gym.set_actor_root_state_tensor_indexed(
                    base_env.sim,
                    gymtorch.unwrap_tensor(base_env.root_states),
                    gymtorch.unwrap_tensor(env_ids_int32),
                    len(env_ids_int32),
                )
                base_env.gym.refresh_dof_state_tensor(base_env.sim)
                base_env.gym.refresh_actor_root_state_tensor(base_env.sim)
                if hasattr(env, "obs_history"):
                    env.obs_history[:] = 0

    obs_dict = env.get_observations()
    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
    debug_dump_quota = 0
    debug_env = os.getenv("LITE3_DEBUG_DUMPS")
    if debug_env:
        try:
            debug_dump_quota = max(0, int(debug_env))
        except ValueError:
            print(f"[play.py] Invalid LITE3_DEBUG_DUMPS='{debug_env}', disabling dumps.")
    dump_root = None
    if debug_dump_quota > 0:
        dump_root = os.getenv("LITE3_DEBUG_DUMP_DIR")
        if not dump_root:
            dump_root = os.path.join(os.path.dirname(LEGGED_GYM_ROOT_DIR), "debug_training_obs")
        os.makedirs(dump_root, exist_ok=True)
    if record_policy_output:
        csv_header = [str(i) for i in range(env.num_policy_outputs)]
        with open(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run,
                                'policy_outputs.csv'),
                    'w',
                    newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
    for i in range(10 * int(env.max_episode_length)):
        with torch.no_grad():
            actions = policy(obs, obs_history)
            # --- Debug export of obs/action for deployment comparison ---
            if debug_dump_quota > 0 and i < debug_dump_quota:
                np.savez(
                    os.path.join(dump_root, f"step_{i:02d}.npz"),
                    obs=obs.cpu().numpy(),
                    obs_history=obs_history.cpu().numpy(),
                    action=actions.cpu().numpy(),
                )
                if i == 0:
                    # Flatten ONNX-style input: [current_obs, history_frames (oldest->newest)]
                    hist = obs_history[0].cpu().numpy().reshape(-1, obs.shape[1])
                    flat_input = np.concatenate([obs[0].cpu().numpy(), hist.flatten()])
                    with open(os.path.join(dump_root, "flat_step_00.txt"), "w") as f:
                        f.write(" ".join(map(str, flat_input)))
            # --- End debug export ---
            # print(actions[0])
        obs_dict, rews, dones, infos = env.step(actions)
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]


if __name__ == '__main__':
    args = get_args()
    play(args)
