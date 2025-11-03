# Lite3 Two-Leg Stand Training Stack Guide

This document walks collaborators through the end-to-end workflow for training, sweeping and deploying Lite3’s two-leg stand behaviour. It expands on the existing README by detailing code layout, configuration structure, hyper-parameter surfaces, logging conventions, and Weights & Biases (W&B) integration.

---

## 1. Project Structure (training subset)

```
Lite3_rl_training/
├─ README.md                     # Original top-level readme
├─ TRAINING_STACK_GUIDE.md       # This guide
├─ setup_Lite3_rl_training.sh    # Environment bootstrap script
├─ legged_gym/
│  ├─ legged_gym/
│  │  ├─ envs/
│  │  │  ├─ base/
│  │  │  │  ├─ legged_robot.py
│  │  │  │  ├─ legged_robot_config.py
│  │  │  │  ├─ two_leg_stand_config.py
│  │  │  │  └─ original_init.json
│  │  │  └─ ... (other env variants)
│  │  ├─ scripts/
│  │  │  ├─ train.py
│  │  │  ├─ play.py
│  │  │  └─ ...
│  │  ├─ utils/
│  │  │  ├─ task_registry.py
│  │  │  ├─ helpers.py
│  │  │  └─ ...
│  │  └─ ...
│  └─ logs/                      # Default log root (tensorboard, checkpoints)
├─ wandb/
│  ├─ run_agent.py               # Sweep agent wrapper
│  ├─ sweep_init.py              # Sweep creation helper
│  ├─ utils.py                   # Shared sweep utilities
│  └─ templates/
│     ├─ two_leg_stand_shallow.json
│     └─ two_leg_stand_exhaustive.json
└─ debug_training_obs/           # Diagnostics captures
```

---

## 2. Environment & Dependencies

Training relies on NVIDIA Isaac Gym, the `legged_gym` environment library, and Huawei’s fork of `rsl_rl`. Ensure the following:

1. **Isaac Gym installed** under `Lite3_rl_training/isaacgym`. The Python bindings must be importable (`isaacgym/python` path on `PYTHONPATH`).
2. **rsl_rl** shipped in-tree (`Lite3_rl_training/rsl_rl`). No wheel install needed; simply keep the folder on `PYTHONPATH`.
3. **Python environment** via the provided setup script:
   ```bash
   bash setup_Lite3_rl_training.sh
   source venv/bin/activate
   ```
   This script pins PyTorch, Isaac Gym dependencies, and utility libraries (NumPy, Matplotlib, etc.).
4. **Weights & Biases** (optional but recommended):
   ```bash
   pip install wandb>=0.22
   wandb login   # provides API key once per machine
   ```

> **Note**  
> Docker images typically mount the repository at `/workspace/Lite3_rl_training`. Paths in this guide assume that layout.

---

## 3. Core Training Workflow

### 3.1 Entry Script

`legged_gym/legged_gym/scripts/train.py` bootstraps the environment and PPO runner:

1. Extends `sys.path` with:
   - `<repo>/legged_gym/legged_gym`
   - `<repo>/isaacgym/python`
   - `<repo>/rsl_rl`
2. Imports and registers the requested task (default `lite3` but we override to `lite3_two_leg_stand`).
3. Builds environment + algorithm via `task_registry.make_env` and `task_registry.make_alg_runner`.
4. Persists resolved configs (`env_cfg.json`, `train_cfg.json`) and runs PPO with logging.

Run manually with:
```bash
python legged_gym/legged_gym/scripts/train.py \
  --task lite3_two_leg_stand \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --physics_engine physicsX \
  --num_envs 2048 \
  --headless
```

### 3.2 Configuration Objects

Configs derive from `legged_robot_config.py` and specialise in `two_leg_stand_config.py`:

| Config block                          | Purpose |
| ------------------------------------ | ------- |
| `TwoLegStandCfg.env`                 | Env size, observation history, curriculum factor |
| `TwoLegStandCfg.control`             | PD gains, action scaling, decimation |
| `TwoLegStandCfg.asset`               | URDF path, contact settings |
| `TwoLegStandCfg.rewards` & `.scales` | Reward shaping weights for torso, leg posture, stand still, etc. |
| `TwoLegStandCfg.curriculum`          | Phase definitions and triggers for staged learning |
| `TwoLegStandCfg.normalization`       | Observation scaling |
| `TwoLegStandCfg.noise`               | Observation noise toggles |
| `TwoLegStandCfg.commands`            | Velocity command sampling |
| `TwoLegStandCfg.domain_rand`         | Physical parameter randomization |
| `TwoLegStandCfgPPO.algorithm`        | PPO hyper-parameters |
| `TwoLegStandCfgPPO.runner`           | Loop duration, checkpointing, experiment naming |

> Use `wandb/utils.py::class_to_dict` to flatten these objects for logging or sweeps.

---

## 4. Logging & Outputs

During training each run logs into `legged_gym/logs/<experiment>/<timestamp_run>`:

- `env_cfg.json`, `train_cfg.json` – fully resolved configurations.
- `model_<iter>.pt`, `model_best.pt` – PyTorch checkpoints. TorchScript exports are placed under `exported/`.
- `rewards.csv` – per-iteration reward breakdown (used by the W&B wrapper).
- TensorBoard summaries (`events.out.tfevents...`) if `--save_rewards` or W&B agent enables summary writer.

Episode rewards include:
- `torso_upright`, `front_legs_up`, `human_posture`, `stand_still`, etc.
- Aggregations such as `total_reward`.

---

## 5. W&B Sweep Integration

### 5.1 Utilities

- `wandb/utils.py` provides:
  - `load_config_module` (load by dotted path or file path).
  - `instantiate_cfgs` (instantiate `LeggedRobotCfg` + `LeggedRobotCfgPPO`).
  - `build_sweep_parameters` (flatten configurations into dotted keys).
  - `apply_overrides` / `dotted_to_nested` (apply sweep configs onto class instances).

These functions convert nested class attributes into a W&B-friendly parameter dictionary, enabling sweeps to override any hyper-parameter exposed by the training stack.

### 5.2 Sweep Templates

Located in `wandb/templates/`:

1. **`two_leg_stand_shallow.json`**
   - Quick Bayesian scan over core PPO and reward scales (learning rate, entropy coefficient, rollout length, control gains).
   - Use for smoke tests or resource-constrained sweeps.

2. **`two_leg_stand_exhaustive.json`**
   - Broader search covering PPO stability knobs (γ, λ, clip, KL), policy activation, curriculum factor, action scaling, reward tolerances, domain randomization toggles, and noise settings.
   - Revised to include binary toggles and tolerance sweeps for comprehensive exploration.

Both templates omit `entity`/`project` to allow overrides via CLI.

### 5.3 Creating a Sweep

```bash
python wandb/sweep_init.py \
  --template wandb/templates/two_leg_stand_shallow.json \
  --config legged_gym.envs.base.two_leg_stand_config \
  --entity HaifaUnivSAIL \
  --project Lite3
```

Flags:
- `--env-class` / `--train-class` allow overriding class names if a module exports multiple specialisations.
- `--output <file>` writes the resolved sweep JSON (with defaults) without creating it on W&B.
- `--dry-run` prints the JSON, useful for verifying parameter surfaces.

### 5.4 Running a Sweep Agent

Once `sweep_init.py` returns a sweep ID (`HaifaUnivSAIL/Lite3/abc123`), start runners:

```bash
python wandb/run_agent.py \
  --sweep-id HaifaUnivSAIL/Lite3/abc123 \
  --config legged_gym.envs.base.two_leg_stand_config \
  --task lite3_two_leg_stand \
  --rl-device cuda:0 \
  --sim-device cuda:0 \
  --physics-engine physicsX \
  --num-runs 1 \
  --headless
```

Multiple agents (on different GPUs/nodes) can be launched to accelerate sweeps. Each agent:
1. Imports Isaac Gym before PyTorch to satisfy dependency requirements.
2. Registers the task, instantiates configs, and applies W&B overrides.
3. Runs PPO training via `task_registry.make_env` / `make_alg_runner`.
4. Updates `wandb.summary` with:
   - `log_dir` of the run.
   - `episode_reward_mean`, `episode_reward_max`, `episode_reward_last` (derived from `rewards.csv`).
5. Syncs the flattened final config to W&B for provenance.

> Ensure `wandb login` has been executed once in the environment and that API keys are available.

---

## 6. Hyper-Parameter Surfaces (Key Levers)

| Block | Parameters | Notes |
| --- | --- | --- |
| **PPO Algorithm** | `learning_rate`, `entropy_coef`, `clip_param`, `desired_kl`, `gamma`, `lam`, `num_learning_epochs`, `num_mini_batches`, `max_grad_norm` | Drive stability vs. convergence speed. Higher entropy aids exploration; clip/desired_kl bound policy updates. |
| **Policy** | `init_noise_std`, `actor_hidden_dims`, `critic_hidden_dims`, `activation` | Control expressivity and initial exploration amplitude. Activation choices (`elu`, `relu`, `tanh`) adjust smoothness. |
| **Runner** | `num_steps_per_env`, `max_iterations`, `save_interval` | Trade-off between on-policy data reuse and GPU memory. |
| **Env Control** | `action_scale`, `stiffness.joint`, `damping.joint`, `decimation` | Fine-tune actuator responsiveness; large scales increase saturation risk. |
| **Rewards** | `torso_upright*`, `front_legs_up*`, `human_posture`, `stand_still`, `base_height`, `termination`, `front_foot_contact_penalty`, `reward_upright_tolerance`, `torso_upright_pitch_tolerance`, `only_positive_rewards` | Determine shaping priorities. Lower termination penalty allows riskier exploration; tolerance controls width of acceptance bands. |
| **Curriculum** | `phases[].trigger_thresh`, `reward_scales` | Stage progression; higher thresholds demand longer warm-up before tightening. |
| **Domain Randomisation / Noise** | Binary toggles controlling friction, base mass, motor strength randomization, sensor noise. Sweeps can evaluate transfer robustness vs. training difficulty. |

---

## 7. Debugging & Diagnostics

### 7.1 Common Issues

- **`ImportError: PyTorch was imported before isaacgym`**  
  Ensure Isaac Gym paths are prepended **and** `import isaacgym` happens before any torch import (handled inside `run_agent.py`).

- **Missing `transformations` module**  
  Install via `pip install transformations` if not bundled in the environment.

- **No W&B metrics appearing**  
  - Confirm `wandb run` shows `episode_reward_*` metrics.  
  - Verify `rewards.csv` generated by the runner (requires `--save_rewards` or W&B agent default).
  - Check API key: `wandb login`.

- **Sweep overrides not applied**  
  - Inspect agent logs; `wandb.config` is printed at start.  
  - Ensure template parameter keys match flattened dotted names (`env_cfg.rewards.scales...`).  
  - Use `--dry-run` on `sweep_init.py` to inspect final JSON.

### 7.2 Debug Utilities

- `debug_training_obs/` stores captured observation/command states for offline comparison (`compare_blocks.py` helper).
- `legged_gym/utils/logger.py` supports plotting states and rewards (`plot_reward.py` script).
- `legged_gym/logs/<run>/rewards.csv` can be analysed to inspect reward component trends.

---

## 8. Extending or Forking Tasks

1. **Clone config** – derive from `TwoLegStandCfg` and adjust new reward terms or command ranges.
2. **Register task** – extend `legged_gym/utils/helpers.py::register` with a new case.
3. **Update templates** – add parameters under new dotted keys (e.g. `env_cfg.rewards.scales.<new_term>`).
4. **Sweep** – supply `--config path.to.new_config` to both `sweep_init.py` and `run_agent.py`.

---

## 9. Quick Reference Commands

| Task | Command |
| --- | --- |
| Activate environment | `source venv/bin/activate` |
| Manual training run | `python legged_gym/legged_gym/scripts/train.py --task lite3_two_leg_stand ...` |
| Create shallow sweep | `python wandb/sweep_init.py --template wandb/templates/two_leg_stand_shallow.json --config legged_gym.envs.base.two_leg_stand_config --entity HaifaUnivSAIL --project Lite3` |
| Launch agent | `python wandb/run_agent.py --sweep-id HaifaUnivSAIL/Lite3/abc123 --config legged_gym.envs.base.two_leg_stand_config --task lite3_two_leg_stand --rl-device cuda:0 --sim-device cuda:0 --num-runs 1 --headless` |
| List sweep parameters | `python wandb/sweep_init.py --template ... --config ... --dry-run` |
| Inspect reward logs | `python legged_gym/legged_gym/scripts/plot_reward.py --logdir legged_gym/logs/<experiment>/<run>` |

---

## 10. Checklist for New Collaborators

- [ ] Clone repo and run `setup_Lite3_rl_training.sh`.
- [ ] Validate Isaac Gym and GPU availability.
- [ ] `wandb login` (if using sweeps).
- [ ] Launch a basic training run to confirm rendering/headless modes.
- [ ] Use `sweep_init.py` with appropriate template to seed sweeps.
- [ ] Spin up `run_agent.py` instances (one per GPU).
- [ ] Monitor runs via W&B (charts, parameter importance, tables).
- [ ] Post-process logs in `legged_gym/logs` as needed.

---

This guide should give contributors the context required to operate and extend the Lite3 two-leg stand training stack quickly. For further questions, reach out via the project’s Slack channel or create issues in the repository. Happy experimenting! 🎉

