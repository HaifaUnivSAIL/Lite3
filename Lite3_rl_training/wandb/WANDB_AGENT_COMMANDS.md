# Lite3 W&B Sweep Agent Cheatsheet

All commands assume they are executed from `Lite3_rl_training/` with the virtual
environment activated (e.g. `source venv/bin/activate`). Replace placeholder
values (e.g. `<SWEEP_ID>`) before running.

## 1. Configure authentication (one‑time per machine)

```bash
# Paste your key into the helper file (or edit in your editor)
printf "YOUR_WANDB_API_KEY\n" > wandb/wandb-api-key-file

# Optional: login immediately without running an agent
WANDB_API_KEY="$(cat wandb/wandb-api-key-file)" python -m wandb login --relogin "${WANDB_API_KEY}"
```

The launcher automatically reads the key file when
`--wandb-api-key-file wandb/wandb-api-key-file` is supplied.

## 2. Basic sweep agent (single GPU)

```bash
nohup ./wandb/run_agent.sh \
  --sweep-id <SWEEP_ID> \
  --num-runs 1 \
  --rl-device cuda:0 \
  --sim-device cuda:0 \
  --physics-engine physicsX \
  --headless \
  --wandb-api-key-file wandb/wandb-api-key-file \
  > agent_gpu0.log 2>&1 &
```

## 3. Multi-run agent on GPU1

```bash
nohup ./wandb/run_agent.sh \
  --sweep-id <SWEEP_ID> \
  --num-runs 100 \
  --rl-device cuda:1 \
  --sim-device cuda:1 \
  --headless \
  --wandb-api-key-file wandb/wandb-api-key-file \
  > agent_gpu1.log 2>&1 &
```

## 4. Overriding config/task arguments

Append `--` followed by `run_agent.py` overrides. Example:

```bash
nohup ./wandb/run_agent.sh \
  --sweep-id <SWEEP_ID> \
  --num-runs 10 \
  --wandb-api-key-file wandb/wandb-api-key-file \
  -- \
  --num-envs 8192 \
  --seed 42 \
  > agent_custom.log 2>&1 &
```

## 5. Flag reference

| Flag | Required | Description |
|------|----------|-------------|
| `-s, --sweep-id` | ✅ | Target W&B sweep (e.g. `entity/project/abc123`). |
| `-n, --num-runs` | ❌ | Number of runs this agent should perform (default `1`). |
| `--headless` | ❌ | Force simulator headless mode. |
| `--rl-device` | ❌ | RL training device (`cuda:0`, `cpu`, etc.). |
| `--sim-device` | ❌ | PhysX/Isaac sim device. |
| `--physics-engine` | ❌ | Defaults to `physicsX`. |
| `--wandb-api-key` | ❌ | Provide API key inline (non-interactive). |
| `--wandb-api-key-file` | ❌ | Path to file containing only the API key (recommended). |

Anything after `--` is forwarded verbatim to `wandb/run_agent.py`, enabling
config-specific overrides such as `--env-class`, `--train-class`, `--resume`,
`--load-run`, or PPO limits (`--max-iterations`, `--seed`, etc.).
