# Experimentation Workflow (Training ⇄ Deploy Parity)

This is the exact, **container‑side** workflow used in this session to train, export, run deploy, dump debug data, and compare training vs deploy.

## 0) Conventions / Paths Used

- Workspace root (inside container): `/workspace`
- Training repo: `/workspace/rl_training_new`
- Deploy repo: `/workspace/Lite3_rl_deploy`
- Debug dump root (shared for both): `/workspace/rl_training_new/lite3_debug`

> All commands below are copy‑pasteable in the container.

## 1) Train (Two‑Leg Stand)

```bash
cd /workspace/rl_training_new
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStand-Deeprobotics-Lite3-v0 \
  --headless \
  --run_name parity_play_v1
```

This produces a run under:
```
/workspace/rl_training_new/logs/rsl_rl/two_leg_stand/<timestamp>_parity_play_v1
```

## 2) Play (Training stack, dump debug)

From the run directory:

```bash
cd /workspace/rl_training_new/logs/rsl_rl/two_leg_stand/<timestamp>_parity_play_v1
./run_play.sh --checkpoint model_1000.pt
```

This produces training dumps under:
```
/workspace/rl_training_new/lite3_debug/train/<timestamp>_parity_play_v1/
```

## 3) Export ONNX from the checkpoint (deploy‑compatible)

```bash
cd /workspace/Lite3_rl_deploy/policy
python pt2onnx.py \
  --ckpt /workspace/rl_training_new/logs/rsl_rl/two_leg_stand/<timestamp>_parity_play_v1/model_1000.pt \
  --out ppo/policy.onnx \
  --num-obs 117 \
  --history-len 40
```

This overwrites:
```
/workspace/Lite3_rl_deploy/policy/ppo/policy.onnx
```

## 4) Run deploy (MuJoCo) and dump debug

```bash
cd /workspace/Lite3_rl_deploy/build

unset LITE3_FIXED_CMD
unset LITE3_DEFAULT_CMD
# or explicitly:
export LITE3_FIXED_CMD=disable

export LITE3_DEBUG_DUMPS=5
export LITE3_DEBUG_DUMP_DIR=/workspace/rl_training_new/lite3_debug/deploy

./rl_deploy
```

This produces deploy dumps under:
```
/workspace/rl_training_new/lite3_debug/deploy/
```

## 5) Compare deploy vs training dumps

```bash
python /workspace/rl_training_new/scripts/tools/compare_deploy_train_dumps.py \
  --deploy-dir /workspace/rl_training_new/lite3_debug/deploy \
  --train-dir  /workspace/rl_training_new/lite3_debug/train/<timestamp>_parity_play_v1 \
  --steps 5 \
  --out /workspace/rl_training_new/lite3_debug/compare_report.json
```

If it succeeds, it prints step‑level max/mean diffs and writes:
```
/workspace/rl_training_new/lite3_debug/compare_report.json
```

## 6) (Optional) Parity sanity check on the ONNX

```bash
python /workspace/rl_training_new/scripts/tools/validate_deploy_parity.py \
  --policy /workspace/Lite3_rl_deploy/policy/ppo/policy.onnx
```

---

### Notes
- `run_play.sh` is expected to dump debug by default into
  `/workspace/rl_training_new/lite3_debug/train/<run_id>`.
- Deploy dumps are controlled via `LITE3_DEBUG_DUMPS` and `LITE3_DEBUG_DUMP_DIR`.

