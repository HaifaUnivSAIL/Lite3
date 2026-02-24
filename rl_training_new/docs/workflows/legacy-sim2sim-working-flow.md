# Legacy Sim-to-Sim Working Flow (Reproducible)

This runbook captures the exact flow that produced a stable, working legacy deploy behavior (training sim -> ONNX export -> deploy sim-to-sim over UDP) without manual timing overrides.

## Scope

- Training stack: `rl_training_new`
- Deploy stack: `Lite3_rl_deploy`
- Deploy backend mode: legacy UDP bridge (`USE_PYBULLET=ON`) with MuJoCo Python simulator as UDP source

## Preconditions

1. Build deploy in legacy simulation mode:

```bash
cd /home/sail/Lite3/Lite3_rl_deploy
mkdir -p build && cd build
cmake .. -DBUILD_PLATFORM=x86 -DBUILD_SIM=ON -DUSE_PYBULLET=ON -DUSE_MJCPP=OFF -DSEND_REMOTE=OFF
make -j
```

2. Source runtime defaults:

```bash
source /home/sail/Lite3/Lite3_rl_deploy/environment_variables.sh
```

## Step 1: Generate aligned training-play dumps

Use deploy-aligned reset semantics for apples-to-apples comparison.

```bash
cd /workspace/rl_training_new/logs/rsl_rl/two_leg_stand_safe_slow_low_power/<run_id>

export LITE3_PLAY_FORCE_DEPLOY_RESET=1
export LITE3_PLAY_NUM_ENVS=1
export LITE3_DEBUG_PLAY_DUMPS=5
export LITE3_DEBUG_PLAY_EVERY=1
export LITE3_DEBUG_PLAY_DIR=/workspace/rl_training_new/lite3_debug/train/<run_id>_aligned

./run_play.sh --checkpoint model_10000.pt
```

## Step 2: Export ONNX from the same checkpoint

```bash
cd /workspace/rl_training_new/logs/rsl_rl/two_leg_stand_safe_slow_low_power/<run_id>
./run_export.sh --checkpoint model_10000.pt
```

Expected output:

```text
[OK] Exported ONNX to: /workspace/Lite3_rl_deploy/policy/ppo/policy.onnx
```

## Step 3: Run deploy sim-to-sim (legacy UDP path)

Terminal A (UDP simulator):

```bash
cd /home/sail/Lite3/Lite3_rl_deploy/interface/robot/simulation
python3 mujoco_simulation.py
```

Terminal B (deploy binary):

```bash
cd /home/sail/Lite3/Lite3_rl_deploy/build
source /home/sail/Lite3/Lite3_rl_deploy/environment_variables.sh

export LITE3_DEBUG_DUMPS=5
export LITE3_DEBUG_DUMP_DIR=/home/sail/Lite3/rl_training_new/lite3_debug/deploy

./rl_deploy
```

Expected ONNX timing line (no manual dt/decimation env vars needed):

```text
[ONNX] timing parity check: training(sim_dt=0.005, decimation=4, control_dt=0.02s) vs deploy(sim_dt=0.001, decimation=20, control_dt=0.02s)
[ONNX] sim_dt=0.001, decimation=20, control_dt=0.02s, sim_dt_source=default_legacy_udp
```

## Step 4: Compare deploy dumps vs aligned training dumps

```bash
cd /home/sail/Lite3
python3 rl_training_new/scripts/tools/compare_deploy_train_dumps.py \
  --deploy-dir rl_training_new/lite3_debug/deploy \
  --train-dir rl_training_new/lite3_debug/train/<run_id>_aligned \
  --steps 5 \
  --out rl_training_new/lite3_debug/compare_report_<run_id>_aligned_vs_deploy.json
```

## Validation notes

- Do not compare deploy dumps against non-aligned play dumps (`LITE3_PLAY_FORCE_DEPLOY_RESET=0`), because reset/history semantics differ by design.
- Legacy deploy timing now defaults to 20 ms policy period in this mode:
  - interface loop: 1 ms
  - policy decimation: 20
  - effective control dt: 0.02 s (matches training control period)
- Deploy now enforces startup timing parity assert:
  - expected training defaults: `sim.dt=0.005`, `decimation=4`
  - override only for intentional training timing changes:
    - `LITE3_TRAINING_SIM_DT`
    - `LITE3_TRAINING_DECIMATION`
