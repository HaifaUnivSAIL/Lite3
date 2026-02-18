# Deploy/Training Parity Lock Plan

## Goal
Lock observation/action parity between:
- Training stack: `rl_training_new`
- Deploy stack: `Lite3_rl_deploy` (MuJoCo + ONNX runner)

So the next experiment cycle is deterministic, debuggable, and converges to matching behavior (no violent early divergence/flip caused by stack mismatch).

## Current confirmed symptoms
- Step-0 compare mismatch is already non-trivial (`obs_flat max_abs ~0.9`).
- Divergence accelerates from step-1 onward (`body_omega`, then `base_rpy` branch-level jumps near `~6.219`).
- `action_raw` amplifies quickly by step-4.
- Training `run_play.sh` can show stable two-leg stand while deploy can still flip, meaning parity break exists before policy quality is the bottleneck.

## Scope
- In scope: observation construction, history feed, angle/frame conventions, actuator/control-envelope parity, debug/diff tooling.
- Out of scope: reward redesign and curriculum redesign for this cycle.

## Phase 0: Freeze reproducible baseline (before code changes)
### Implement
- Store one canonical pair of dumps:
  - Train: `~/Lite3/rl_training_new/lite3_debug/train/<run_id>/`
  - Deploy: `~/Lite3/rl_training_new/lite3_debug/deploy/`
- Keep one compare artifact:
  - `~/Lite3/rl_training_new/lite3_debug/compare_report_<run_id>.json`

### Exit criteria
- Same commands reproduce the same compare pattern (within expected noise) twice in a row.

## Phase 1: Make step-0 contract explicit and identical
### Implement
- Add an explicit observation contract manifest (ordered term names and sizes) and use it in both stacks.
- Ensure deploy dump contains the exact term slices matching training dump semantics:
  - `cmd`, `base_rpy`, `body_omega`, `joint_pos`, `joint_vel`,
  - `joint_pos_history`, `joint_vel_history`, `action_history`.
- Confirm flat concatenation order is exactly:
  - current observation (117), then 40 history frames (117 each).

### Files
- `rl_training_new/scripts/reinforcement_learning/rsl_rl/play.py`
- `rl_training_new/scripts/tools/compare_deploy_train_dumps.py`
- `Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp`

### Exit criteria
- Compare report includes per-term diffs for steps 0-4 (not only flat summary).
- Step-0 top mismatches are clearly localized to named terms.

## Phase 2: History bootstrap parity (most likely root cause)
### Implement
- Rework deploy history seeding to match training reset semantics for:
  - `joint_pos_history`
  - `joint_vel_history`
  - `action_history` (including reset fallback behavior equivalent to training)
- Remove heuristic seeds that cannot be mapped to training state.
- Add explicit mode logging on policy enter:
  - `history_seed_mode`, source values, and first composed frame.

### Files
- `Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp`
- (if needed) `Lite3_rl_deploy/state_machine/rl_control_state_onnx.hpp`

### Exit criteria
- Step-0 `obs_flat` max_abs drops significantly and history terms no longer dominate mismatch.
- Step-0 `action_raw` mismatch decreases materially.

## Phase 3: Angle/frame convention hardening
### Implement
- Normalize and log all orientation paths:
  - quaternion source
  - rotation matrix
  - `base_rpy` with explicit wrapping policy
  - projected gravity (if used for cross-check)
- Ensure `body_omega` is used consistently (avoid accidental double rotate).
- Add guard against branch jump artifacts in debug report (wrap-aware diff for angles).

### Files
- `Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp`
- `rl_training_new/scripts/tools/compare_deploy_train_dumps.py`

### Exit criteria
- `base_rpy` no longer appears as dominant mismatch from step-2 with ~2pi-scale jumps.

## Phase 4: Control envelope parity (simulation amplification control)
### Implement
- Enforce training-consistent effort envelope in deploy before writing control:
  - Hip vs knee limits consistent with training assumptions.
- Re-check decimation and control dt alignment for deploy loop.
- Dump:
  - desired joint targets
  - PD raw torque
  - clamped torque
  - applied control

### Files
- `Lite3_rl_deploy/interface/robot/simulation/mujoco_interface.hpp`
- `Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp`

### Exit criteria
- Deploy no longer “instant-amplifies” into violent motion when step-0/1 mismatches are modest.

## Phase 5: Experiment harness robustness
### Implement
- Make compare tool robust and explicit:
  - validate directories and file counts
  - show matched steps and missing-step diagnosis
  - emit top-N per-term mismatch summary per step
- Keep default dump paths stable and writable:
  - `~/Lite3/rl_training_new/lite3_debug/train/<run_id>`
  - `~/Lite3/rl_training_new/lite3_debug/deploy`

### Files
- `rl_training_new/scripts/tools/compare_deploy_train_dumps.py`
- `rl_training_new/scripts/reinforcement_learning/rsl_rl/train.py` (if run-script env defaults need regeneration)

### Exit criteria
- One command reliably produces actionable compare output without manual environment setup.

## Final acceptance criteria for parity lock
- Compare (steps 0-4) shows:
  - no catastrophic angle branch mismatches
  - no history-channel domination at step-0
  - action mismatch is bounded and non-escalating
- Deploy behavior qualitatively matches train play startup trajectory (no immediate flip/back-fall).
- Workflow is repeatable from clean rebuild without ad-hoc debugging steps.

## Commands for next cycle (execution order)
1. Export ONNX from checkpoint:
   - `cd /workspace/Lite3_rl_deploy/policy`
   - `python pt2onnx.py --ckpt /path/to/model_XXXX.pt --out ppo/policy.onnx --num-obs 117 --history-len 40`
2. Run training play dump:
   - `cd /workspace/rl_training_new/logs/rsl_rl/two_leg_stand/<run_id>`
   - `./run_play.sh --checkpoint model_XXXX.pt`
3. Rebuild and run deploy (MuJoCo), trigger RL mode, collect deploy dumps.
4. Compare:
   - `python /workspace/rl_training_new/scripts/tools/compare_deploy_train_dumps.py --deploy-dir /workspace/rl_training_new/lite3_debug/deploy --train-dir /workspace/rl_training_new/lite3_debug/train/<run_id> --steps 5 --out /workspace/rl_training_new/lite3_debug/compare_report_<run_id>.json`

## Implementation queue
1. Phase 1 + Phase 2 together (contract + history bootstrap).
2. Phase 3 (angle/frame hardening).
3. Phase 4 (control envelope parity).
4. Phase 5 (harness polish and guardrails).
