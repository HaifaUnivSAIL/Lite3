# Training Stack Deployment Checklist

This document is the release-ready runbook for training/inference parity with correct history semantics.

## 1) Required Behavior

- Default behavior: observation history is reset on episode done/reset.
- Debug-only override: `LITE3_UNREALISTIC_HISTORY_FEED=1` (legacy leak emulation).
- Do not use legacy override for production training or evaluation.

## 2) Preflight (before server run)

From workspace root:

```bash
cd /workspace
```

Run history semantics test:

```bash
pytest -q rl_training_new/tests/test_history_reset_semantics.py
```

Expected: `4 passed`.

## 3) Train (production/default history behavior)

```bash
cd /workspace/rl_training_new
unset LITE3_UNREALISTIC_HISTORY_FEED

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStand-Deeprobotics-Lite3-v0 \
  --headless \
  --run_name deploy_ready_v1
```

## 4) Play and dump training debug

```bash
cd /workspace/rl_training_new/logs/rsl_rl/two_leg_stand/<run_id>
unset LITE3_UNREALISTIC_HISTORY_FEED
./run_play.sh --checkpoint model_6000.pt
```

Expected logs include:

- `History mode: default_reset_on_done`

Training dumps:

- `/workspace/rl_training_new/lite3_debug/train/<run_id>/debug_play_step*.npz`

## 5) Export ONNX from training checkpoint

```bash
cd /workspace/Lite3_rl_deploy/policy
python pt2onnx.py \
  --ckpt /workspace/rl_training_new/logs/rsl_rl/two_leg_stand/<run_id>/model_6000.pt \
  --out ppo/policy.onnx \
  --num-obs 117 \
  --history-len 40
```

## 6) Run deploy and dump debug

```bash
cd /workspace/Lite3_rl_deploy/build

unset LITE3_HISTORY_SEED_FILE
export LITE3_FORCE_RL_START=1
export LITE3_MUJOCO_DT=0.005
export LITE3_POLICY_DECIMATION=4
export LITE3_POLICY_ASYNC=0
export LITE3_DEBUG_DUMPS=5
export LITE3_DEBUG_DUMP_DIR=/workspace/rl_training_new/lite3_debug/deploy

unset LITE3_FIXED_CMD
unset LITE3_DEFAULT_CMD
export LITE3_FIXED_CMD=disable

./rl_deploy
```

## 7) Compare deploy vs training dumps

```bash
python /workspace/rl_training_new/scripts/tools/compare_deploy_train_dumps.py \
  --deploy-dir /workspace/rl_training_new/lite3_debug/deploy \
  --train-dir /workspace/rl_training_new/lite3_debug/train/<run_id> \
  --steps 5 \
  --out /workspace/rl_training_new/lite3_debug/compare_report_<run_id>.json
```

## 8) Release Gates

Accept run for deployment only if:

- history test passes (`4 passed`);
- training/play ran with `LITE3_UNREALISTIC_HISTORY_FEED` unset;
- deploy ran with `LITE3_HISTORY_SEED_FILE` unset;
- behavior in simulation is stable and expected;
- compare report is archived with the run ID.

## 9) Debug-Only Legacy Experiment

Use only for diagnosis:

```bash
export LITE3_UNREALISTIC_HISTORY_FEED=1
```

This enables legacy history carry-over and is not valid for production metrics.

## 10) Push Preparation

Before pushing:

1. Ensure code changes are limited to source/tests/docs (exclude generated debug dumps).
2. Re-run:
   - `pytest -q rl_training_new/tests/test_history_reset_semantics.py`
3. Capture and archive:
   - training run ID,
   - checkpoint used,
   - compare report path,
   - deploy behavior note.
