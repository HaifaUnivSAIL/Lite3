# History-Seed Parity Experiment (Deploy vs Training)

## Goal

Identify whether the remaining deploy behavior gap is caused by:

- ONNX/PT inference mismatch, or
- observation-history distribution mismatch at reset.

## Key finding

When deploy is seeded with the same **unrealistic (leaky)** history observed in training, the robot stands correctly on two legs.

This strongly indicates the primary gap is **history/reset distribution**, not network inference parity.

## Background

Two behaviors were observed:

- Realistic history reset (clear on done): robot struggles more at episode start.
- Legacy/leaky history across done: robot appears to recover/stand more easily after resets.

This experiment intentionally replays the leaky history into deploy to test causality.

## Reproduction guide

### 1) Generate training dumps in legacy leak mode

From run directory:

```bash
cd /workspace/rl_training_new/logs/rsl_rl/two_leg_stand/2026-02-12_13-02-05_parity_latest

LITE3_UNREALISTIC_HISTORY_FEED=1 \
LITE3_PLAY_FORCE_DEPLOY_RESET=0 \
LITE3_DEBUG_PLAY_AFTER_FIRST_DONE=1 \
LITE3_DEBUG_PLAY_DUMPS=5 \
LITE3_DEBUG_PLAY_EVERY=1 \
LITE3_DEBUG_PLAY_DIR=/workspace/rl_training_new/lite3_debug/train_unrealistic/2026-02-12_13-02-05_parity_latest \
./run_play.sh --checkpoint model_6000.pt
```

Notes:

- `LITE3_UNREALISTIC_HISTORY_FEED=1` enables the legacy leak mode only for this experiment.
- `LITE3_DEBUG_PLAY_AFTER_FIRST_DONE=1` starts dump capture after the first reset event.

### 2) Export seed file from training dump

```bash
FIRST_DUMP=$(ls /workspace/rl_training_new/lite3_debug/train_unrealistic/2026-02-12_13-02-05_parity_latest/debug_play_step*.npz | sort -V | head -n 1)

python3 /workspace/rl_training_new/scripts/tools/export_history_seed_from_train_dump.py \
  --train-dump "$FIRST_DUMP" \
  --out /workspace/history_seed_unrealistic.txt \
  --include-obs-history
```

Expected output:

- `pos_hist=36`
- `vel_hist=24`
- `action_hist=24`
- `obs_history=4680`

### 3) Run deploy with history replay seed

```bash
cd /workspace/Lite3_rl_deploy/build

export LITE3_HISTORY_SEED_FILE=/workspace/history_seed_unrealistic.txt
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

### 4) Verify seed replay was active

Check `debug_cpp_step0.txt`:

- `history_seed_file_loaded 1`
- `history_seed_file_used 1`
- `history_seed_file_path /workspace/history_seed_unrealistic.txt`
- `action_history_mode replay_file_direct`

## Conclusion

The deploy robot matching training behavior under seeded leaky history is a direct causal signal:

- PT vs ONNX parity is not the limiting factor for this issue.
- The critical mismatch is in reset-time history/state distribution.

## Implications

- Legacy training behavior was partly supported by unrealistic history carry-over.
- Deploy with clean reset is exposing true cold-start robustness limits of the policy.

## Recommended realistic training plan

### A) Lock semantics

- Keep `LITE3_UNREALISTIC_HISTORY_FEED` unset (or `0`) in all real training/eval runs.
- Do not use history seed replay outside diagnostics.

### B) Train for cold-start robustness

- Include explicit early-episode stabilization emphasis (first 1-2 seconds).
- Keep reset-time history zero/clean, and randomize reset states within realistic bounds.
- Optionally add short warmup controller stage before RL if deploy uses one.

### C) Evaluation protocol (must-pass)

- Report separately:
  - first-episode success rate (cold start),
  - post-reset success rate,
  - time-to-stand,
  - fall rate.
- Keep current deploy-vs-train dump compare as regression guard.

### D) Migration path

- Short term: fine-tune current checkpoint under corrected history semantics.
- Medium term: retrain from scratch with corrected semantics and cold-start metrics as acceptance gates.
