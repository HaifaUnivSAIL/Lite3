# Two-Leg Stand: Safe/Slow/Low-Power Domain Randomization Plan

This runbook defines the implementation and validation path for augmenting:

- `TwoLegStandSafeSlowLowPower-Deeprobotics-Lite3-v0`

into:

- `TwoLegStandSafeSlowLowPowerDomainRand-Deeprobotics-Lite3-v0`

with robot-domain randomization enabled by default, and environment-domain randomization enabled only via explicit flag.

## 1) Scope and Non-Negotiables

Goal:
- Train a policy that still reaches and holds two-leg stand safely, while becoming robust to robot parameter mismatch and optional environment disturbances.

Non-negotiable constraints:
1. No cross-episode history leakage.
2. Observation conventions must remain deploy-correct (quaternion order and derived angles).
3. New task must be isolated (no overwrite of existing task IDs/behavior).
4. Safety curriculum principle remains: exploration first, stricter behavior later.

## 2) Current Scaffold (Already in Code)

Task and runner wiring:
- New env cfg: `Lite3TwoLegStandSafeSlowLowPowerDomainRandEnvCfg`
- New runner cfg: `Lite3TwoLegStandSafeSlowLowPowerDomainRandPPORunnerCfg`
- Registered task ID: `TwoLegStandSafeSlowLowPowerDomainRand-Deeprobotics-Lite3-v0`
- Legacy alias: `lite3_two_leg_stand_safe_slow_low_power_domain_rand`

Domain-randomization split:
- Robot DR default ON:
  - rigid body mass
  - COM
  - actuator gains (reset mode)
  - motor strength (reset mode)
- Environment DR default OFF:
  - friction/restitution
  - gravity
  - pushes
- Environment DR ON only when:
  - `LITE3_ENABLE_ENV_DOMAIN_RANDOMIZATION=1`

History/convention guards already enforced globally:
- `LITE3_UNREALISTIC_HISTORY_FEED=1` raises runtime error in wrappers and train entrypoint.
- Quaternion convention in observations is explicitly `wxyz`.

## 3) Runtime Modes

Mode A (default, robot DR only):
```bash
unset LITE3_ENABLE_ENV_DOMAIN_RANDOMIZATION
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStandSafeSlowLowPowerDomainRand-Deeprobotics-Lite3-v0 \
  --headless \
  --run_name safe_slow_domain_rand_robot_only_v1
```

Mode B (robot + environment DR):
```bash
LITE3_ENABLE_ENV_DOMAIN_RANDOMIZATION=1 \
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStandSafeSlowLowPowerDomainRand-Deeprobotics-Lite3-v0 \
  --headless \
  --run_name safe_slow_domain_rand_robot_env_v1
```

## 4) Sequential Plan

### Stage 1: Wiring and Safety Guards (Complete)

Deliverables:
1. New task and runner registration.
2. Robot DR default, env DR optional by flag.
3. Gravity randomization event hook.
4. Guardrails against unrealistic history feed.

Acceptance:
1. Task resolves and starts without flag.
2. Flag toggles environment DR terms.
3. `LITE3_UNREALISTIC_HISTORY_FEED=1` fails fast.

### Stage 2: Curriculum-Coupled DR Intensity (Next)

Problem:
- DR ranges are currently static across curriculum phases.

Plan:
1. Add DR intensity schedule keyed to existing curriculum phase index.
2. Keep phase-0 exploration less harsh (narrower DR).
3. Increase mismatch/disturbance ranges in later phases.

Implementation approach:
1. Extend curriculum callback to optionally update event-term params per phase.
2. Keep default behavior backward compatible for all existing tasks.
3. Apply schedule only in DomainRand task class.

Acceptance:
1. Logs show phase transitions and DR-range updates.
2. Early-phase reward/value remains stable (no immediate collapse).
3. Late phases expose stronger mismatch while preserving learning progress.

### Stage 3: Observation Realism Contract (Next)

Objective:
- Ensure policy only receives signals available or estimable at deployment.

Plan:
1. Audit each policy observation term for physical observability.
2. Keep hidden sim-only internals out of policy obs.
3. If privileged terms are needed, confine them to critic only.

Acceptance:
1. Policy observation schema documented.
2. No privileged-only signals in actor input path.
3. Sim-to-sim deploy parity check passes on exported policy.

### Stage 4: Robustness Evaluation Matrix (Next)

Experiments:
1. Baseline: safe_slow_realistic_v1 (no DR).
2. DomainRand robot-only.
3. DomainRand robot+env.

Metrics:
1. Stand success rate.
2. Hold duration.
3. Falls after stand.
4. Power / torque-limit exceedance.
5. Action-rate / acceleration smoothness.
6. Deploy parity deltas.

Acceptance:
1. DomainRand robot-only improves transfer robustness with minimal success drop.
2. DomainRand robot+env improves disturbance tolerance without destabilizing core task.

## 5) Hard-Fail Validation Gates

These checks are mandatory for this task family:

1. History leak gate:
   - `LITE3_UNREALISTIC_HISTORY_FEED` must remain unset.
   - Any attempt to enable it must crash fast.
2. Angular convention gate:
   - Quaternion unpacking must remain `w,x,y,z`.
   - RPY derivation tests must pass.
3. Episode-reset gate:
   - History buffers cleared on done/reset envs.
4. Randomization scope gate:
   - Environment DR stays OFF unless flag explicitly ON.

## 6) Tests and Status

Current scaffold tests:
- `test_safe_slow_low_power_domain_rand_task.py`
- `test_two_leg_stand_function_sanity.py`
- `test_history_reset_semantics.py` (requires `torch` in test environment)

Current status:
1. Scaffold tests pass locally.
2. History-reset semantics test is skipped when `torch` is missing in pytest environment; run in full training env before release tagging.

## 7) Release Checklist for DomainRand Task

1. Run Stage-2 implementation and verify no phase-1 collapse.
2. Run robot-only DR training from scratch and export policy.
3. Run sim-to-sim parity and robustness checks.
4. Run robot+env DR training variant.
5. Compare metrics to baseline and lock tuned ranges.
