# Two-Leg Stand: Safe, Slow, Low-Power Curriculum Plan

This document defines the implementation plan to augment the existing `TwoLegStand-Deeprobotics-Lite3-v0` training scheme with late-phase safety, low-dynamics transition behavior, and low-power standing persistence.

## 1) Goal

Train a policy that:

1. Reaches two-leg stand reliably.
2. Holds two-leg stand for long durations with stability.
3. Uses lower torque/velocity/power in late curriculum phases.
4. Is strongly discouraged from falling after stand is achieved.

Non-negotiable scope:
- Implement as a **new task** (new env cfg + new registration + new runner cfg).
- Do **not** modify behavior of existing task IDs.

## 2) Current Baseline (What We Keep)

- Task and runner stay the same:
  - `TwoLegStand-Deeprobotics-Lite3-v0`
  - `Lite3TwoLegStandEnvCfg`
  - `get_two_leg_stand_phases()` progression (`0`, `500`, `1000`, `2500`).
- Early exploration behavior remains permissive (minimal safety penalties in early phase).
- Existing core stand-shaping rewards remain active:
  - `front_legs_up_warmup`
  - `torso_upright_warmup`
  - `base_height_bonus`
  - `hind_leg_extension_geom`
  - `stand_still_roll_only`

Task isolation strategy:
- Add a new task ID (example): `TwoLegStandSafeSlowLowPower-Deeprobotics-Lite3-v0`.
- Add a new env config class derived from an existing base class (recommended parent: `Lite3TwoLegStandEnvCfg` or `Lite3TwoLegStandSafeEnvCfg`).
- Add a dedicated runner config class for this task.
- Register task in `config/lite3/__init__.py` without changing current registrations.

## 3) New Reward Terms to Add

All new reward functions go into:
- `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/mdp/rewards.py`

All new reward configs go into:
- `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/two_leg_stand_env_cfg.py`

All exports go into:
- `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/mdp/__init__.py`

### 3.1 `two_leg_state_hold_bonus` (positive, cumulative)

Purpose:
- Reward persistent stable standing after state is reached.

State detection:
- Reuse `two_leg_stand_metric`.
- Use hysteresis:
  - enter state when `metric >= enter_thr`
  - exit state when `metric < exit_thr`
  - `enter_thr > exit_thr` to avoid flicker.

Per-env memory:
- `in_two_leg_state` (bool)
- `hold_steps` (int)
- `ever_reached_two_leg_state` (bool)

Reward shape:
- `r_hold = in_state * (1 - exp(-hold_steps / tau_hold))`
- Optional cap: `r_hold = min(r_hold, hold_cap)`.

### 3.2 `transition_dynamics_penalty` (negative)

Purpose:
- Penalize aggressive or violent transition into stand.

Definition:
- `p_dyn = w_lin_z * lin_vel_z + w_ang_xy * ang_vel_xy + w_acc * dof_acc + w_rate * action_rate`

Gating:
- Apply strongly before first stable stand.
- After stable hold exceeds `hold_grace_steps`, reduce weight (or set to zero).

### 3.3 `effort_bundle_penalty` (negative, proportional)

Purpose:
- Penalize high effort / power / aggressive actuation.

Definition:
- `p_eff = w_tau_lim * torque_limits + w_vel_lim * dof_vel_limits + w_power * power + w_act_mag * action_magnitude`

Notes:
- Uses already-implemented primitive reward terms.
- Keeps tuning manageable by adjusting one bundle term instead of many independent terms.

### 3.4 `fall_after_stand_penalty` (negative event-like)

Purpose:
- Strongly penalize losing stability once two-leg stand was achieved.

Definition:
- On termination:
  - If `ever_reached_two_leg_state == False`: `0`
  - Else: `p_fall = base_fall_penalty * (1 + k_hold * clamp(hold_steps_before_fall / hold_ref, 0, 1))`

Effect:
- Falling after sustained stable hold becomes more costly than early exploratory failures.

## 4) Reward-Clipping Constraint (Critical)

Current config has:
- `only_positive_rewards = True` in `TwoLegStandEnvCfg`.
- Wrapper clips total reward at zero (except preserved termination handling).

Impact:
- New negative penalties can be weakened by clipping.

Planned handling:
1. Keep baseline behavior for early curriculum.
2. Add a controlled config switch in this task for this feature branch:
   - Set `only_positive_rewards=False` once low-power penalties are introduced (recommended for faithful penalties).
3. If we decide to keep clipping, penalties must be explicitly preserved in wrapper logic similar to termination preservation.

Decision checkpoint:
- Confirm which option to ship before implementation starts.

## 5) Curriculum Strategy (Updated Phases)

Keep existing phase boundaries, change active terms by phase.

### Phase 0 (`iter >= 0`) Exploration

Intent:
- Allow broad exploration and stand discovery.

Weights:
- High: `front_legs_up_warmup`, `torso_upright_warmup`, `base_height_bonus`
- Low/Off: `transition_dynamics_penalty`, `effort_bundle_penalty`, `fall_after_stand_penalty`
- On (small): `two_leg_state_hold_bonus` (to seed persistence behavior early)

### Phase 1 (`iter >= 500`) Stabilize Reach

Intent:
- Improve repeatable stand entry and short hold.

Weights:
- Increase `two_leg_state_hold_bonus`
- Turn on mild `transition_dynamics_penalty`
- Turn on mild `effort_bundle_penalty`
- Keep fall penalty low

### Phase 2 (`iter >= 1000`) Transition to Quality

Intent:
- Favor smoother, less aggressive stand-up and longer stable hold.

Weights:
- Stronger `two_leg_state_hold_bonus`
- Medium `transition_dynamics_penalty`
- Medium/high `effort_bundle_penalty`
- Medium `fall_after_stand_penalty`

### Phase 3 (`iter >= 2500`) Safe/Low-Power Polish

Intent:
- Deployment-style stable, calm, and efficient two-leg standing.

Weights:
- Highest `two_leg_state_hold_bonus`
- Highest `effort_bundle_penalty`
- Highest `fall_after_stand_penalty`
- Keep `transition_dynamics_penalty` medium/high (primarily for re-entry attempts)
- Increase `termination` magnitude to reinforce anti-fall behavior.

## 6) Implementation Sequence (Strict Order)

1. Add new isolated task plumbing:
   - New env config class in `config/lite3/base_env_cfg.py`.
   - New PPO runner cfg class in `config/lite3/agents/rsl_rl_ppo_cfg.py`.
   - New registration in `config/lite3/__init__.py`.
2. Add reward state memory helpers in `rewards.py`:
   - Per-env state buffers (`in_state`, `hold_steps`, `ever_reached`).
   - Reset-safe updates using `episode_length_buf == 0`.
3. Implement `two_leg_state_hold_bonus`.
4. Implement `transition_dynamics_penalty`.
5. Implement `effort_bundle_penalty`.
6. Implement `fall_after_stand_penalty`.
7. Register new `RewTerm`s in `TwoLegStandRewardsCfg`.
8. Export new symbols in `mdp/__init__.py`.
9. Add phase weights for new terms in a **new phase factory** (do not edit existing baseline phase factory used by current task).
10. Decide and apply clipping strategy (`only_positive_rewards` handling) in the new task only.
11. Add/update tests.
12. Run focused training smoke test and inspect reward-term logs.
13. Run longer training and compare with baseline on success/effort/fall metrics.

## 7) Test Plan

## 7.1 Unit Tests

Add new test module:
- `rl_training_new/tests/test_two_leg_state_rewards.py`

Test cases:
1. Hysteresis transitions:
   - Enter at `>= enter_thr`, exit at `< exit_thr`.
2. Hold accumulation:
   - Hold bonus increases monotonically and saturates.
3. Reset behavior:
   - Buffers clear on reset envs only.
4. Fall penalty trigger:
   - No penalty before ever reaching state.
   - Penalty applied after stand was achieved.
5. Dynamics and effort penalties:
   - Increase with larger velocity/acc/torque/power/action magnitude.

## 7.2 Integration Checks

1. Reward manager receives all new terms and curriculum modifies their weights by phase.
2. Logs show phase transitions and expected term activation.
3. No NaN/Inf in rewards or losses.

## 7.3 Training Evaluation Criteria

Track across checkpoints:
- `two_leg_stand_metric` mean and success rate.
- Mean stand hold duration (new derived metric).
- Mean episode fall count after first stand.
- Mean torque limit exceedance.
- Mean power.
- Mean action rate / dof acceleration.

Acceptance trend:
- Equal or better stand success than baseline.
- Lower late-phase effort metrics.
- Lower fall-after-stand frequency.

## 8) Initial Hyperparameter Table (Starting Point)

These are initial values for first tuning pass (not final).

- `two_leg_state_hold_bonus`:
  - `enter_thr=0.80`
  - `exit_thr=0.70`
  - `tau_hold=60` steps
  - phase weights: `[0.5, 1.5, 3.0, 5.0]`
- `transition_dynamics_penalty`:
  - internal coeffs: `w_lin_z=1.0, w_ang_xy=1.0, w_acc=0.1, w_rate=0.2`
  - phase weights: `[0.0, -0.3, -0.8, -1.0]`
- `effort_bundle_penalty`:
  - internal coeffs: `w_tau_lim=1.0, w_vel_lim=1.0, w_power=0.01, w_act_mag=0.05`
  - phase weights: `[0.0, -0.2, -0.7, -1.2]`
- `fall_after_stand_penalty`:
  - `base_fall_penalty=1.0`
  - `k_hold=1.0`
  - `hold_ref=120` steps
  - phase weights: `[0.0, -0.5, -1.0, -2.0]`

## 9) Risk Register

1. Reward clipping suppresses penalties.
   - Mitigation: disable clipping for this feature branch or preserve specific penalties.
2. Too-strong late penalties can collapse exploration recovery.
   - Mitigation: delayed ramp and per-phase warm starts.
3. State detector instability (metric flicker).
   - Mitigation: hysteresis + minimum hold time.
4. Overfitting to slow behavior (failing to stand quickly enough).
   - Mitigation: keep core stand-reaching rewards dominant through phase 1.

## 10) Deliverables

1. Code changes for new rewards and curriculum weights.
2. New tests for stateful reward logic.
3. One training report comparing baseline vs augmented scheme:
   - stand success
   - hold duration
   - fall-after-stand frequency
   - power/effort metrics
4. Final tuned reward table committed in curriculum file.
