# Docker Setup for rl_training_new

This guide explains how to build the container and use the virtual environment inside it for both server and local modes.

## Prerequisites (Host)

- Docker installed and running.
- NVIDIA drivers installed (for GPU training).
- NVIDIA Container Toolkit installed if you want GPU passthrough.
- Access to NVIDIA NGC (to pull the Isaac Sim base image).

### NGC login (required once)

The Docker image is based on the official Isaac Sim container from NVIDIA NGC.

1) Create an NGC account and an API key.
2) Login to NGC:

```bash
docker login nvcr.io
```

Use:
- Username: `$oauthtoken`
- Password: your NGC API key

## Build the Image

From the repo root (the build script will automatically clone Isaac Lab inside the image):

```bash
./docker/build.sh local
```

Or for a server/full dependency install:

```bash
./docker/build.sh server
```

Optional overrides (only if needed):

```bash
ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0 ./docker/build.sh server
ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git ISAACLAB_BRANCH=main ./docker/build.sh server
```

What gets installed inside the image:
- Isaac Sim base image is pulled from NGC and mounted at `/isaac-sim`.
- Isaac Lab is cloned during build into `/opt/IsaacLab` and installed into the venv.

## Launch the Container

Headless server mode:

```bash
./docker/launch.sh headless
```

Local interactive (GUI) mode:

```bash
./docker/launch.sh local
```

Notes:
- If GUI apps fail to open, run `xhost +local:root` on the host before launching and `xhost -local:root` afterwards.
- Isaac Sim requires EULA acceptance; `launch.sh` sets `ACCEPT_EULA=Y` automatically.
- `launch.sh` opens a shell inside the container (it overrides the Isaac Sim image entrypoint).
- `launch.sh` sources `/isaac-sim/setup_python_env.sh`, then appends Isaac Lab and repo paths to `PYTHONPATH`, and activates `/venv`.
- `launch.sh` sets `OMNI_EXTENSIONS_PATH`, `CARB_APP_PATH`, and `OMNI_APP_PATH` for Isaac Sim extensions (including `/isaac-sim/exts/isaacsim`).
- `launch.sh` sets `EXP_PATH=/isaac-sim/apps`, which Isaac Lab uses to locate experience files.
- `launch.sh` wraps `python`/`pip` to use `/isaac-sim/python.sh` (required for `SimulationApp`) and adds `/venv` site-packages to `PYTHONPATH`.
- `launch.sh` also adds `/workspace/rl_training_new/rsl_rl` and `/workspace/rl_training_new/legged_gym` to `PYTHONPATH` (legacy RL + helpers).
- On container start, `launch.sh` auto-installs the in-tree `rsl_rl` and `legged_gym` into the Isaac Sim Python environment if missing.
- `launch.sh` also ensures the `transformations` Python package is installed (required by legacy `legged_gym.utils.math`).
- `launch.sh` also ensures `gym==0.26.2` is installed (required by legacy `rsl_rl`).
- `launch.sh` also ensures `tensordict` is installed (required by `isaaclab_rl`).

## Virtual Environment Inside the Container

The image pre-creates a venv at `/venv` and sets it in `PATH`. You can still activate it explicitly:

```bash
source /venv/bin/activate
python -V
which python
```

Important:
- `python` in the shell is wrapped to `/isaac-sim/python.sh` so Isaac Sim extensions load correctly.
- `/venv/lib/python3.11/site-packages` is added to `PYTHONPATH`, so packages installed into `/venv` remain visible.
- If you need the raw venv interpreter, use `/venv/bin/python` or `/venv/bin/pip`.

Optional: install the extension in editable mode (PYTHONPATH is already set by `launch.sh`):

```bash
cd /workspace/rl_training_new
python -m pip install -e source/rl_training
```

Verify Isaac Lab import:

```bash
python -c "import isaaclab; print(isaaclab.__file__)"
```

Verify RSL-RL import:

```bash
python -c "from rsl_rl.runners import OnPolicyRunner; print(OnPolicyRunner)"
```

Verify legged_gym import:

```bash
python -c "import legged_gym; print(legged_gym.__file__)"
```

Verify Isaac Sim API is exposed:

```bash
python -c "from isaacsim import SimulationApp; print(SimulationApp)"
```

## Quick Train/Play Sanity Check

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-Lite3-v0 --headless
```

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Deeprobotics-Lite3-v0 --num_envs=10
```

## Server vs Local Summary

- `local`: minimal Python deps for quick iteration.
- `server`: installs `docker/requirements_server.txt` for full training deps.
