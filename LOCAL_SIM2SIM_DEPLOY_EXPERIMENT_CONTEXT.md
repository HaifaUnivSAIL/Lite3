# Local Sim2sim Deploy Experiment Context

This document is a handoff for future sessions working on Lite3 policy deployment experiments.
It captures the stable mental model for moving from `rl_training_new` training runs to local
`Lite3_rl_deploy` MuJoCo sim2sim validation.

The goal is to make a trained policy pass local deploy simulation before touching the robot.
The robot stack may differ from this repo's local deploy stack, so the local stack is treated
as the controlled acceptance environment.

## Core Principle

The local acceptance test is not "can the ONNX run once".

Success means:

1. Train a policy in `rl_training_new`.
2. Export the selected checkpoint to ONNX.
3. Deliver that ONNX to the policy path used by `Lite3_rl_deploy`.
4. Run the full local deploy state machine with MuJoCo:
   - start simulator first,
   - start `./rl_deploy`,
   - press `z` for stand-up,
   - wait about 5 seconds,
   - press `c` for RL policy,
   - judge behavior from logs, debug dumps, and simulator behavior.
5. Only if local MuJoCo sim2sim behaves like training do we consider the policy a candidate for robot-side deployment.

`LITE3_FORCE_RL_START=1` is a diagnostic shortcut only. It is not the acceptance path because it bypasses
the deploy stand-up controller and can start RL from a state that the real deploy flow would not produce.

## Stack Roles

### `rl_training_new`

This is the training stack. It defines:

- task/config variants,
- reward/curriculum/domain-randomization choices,
- observation contract,
- control timing,
- history semantics,
- checkpoint and helper-script generation.

For the current two-leg stand family, important invariants are:

- policy observation frame: 117 dimensions,
- observation history length: 40 frames,
- ONNX flat input: `117 * (1 + 40) = 4797`,
- action dimension: 12,
- training control timing: `sim.dt=0.005`, `decimation=4`, policy period `0.02s`, policy rate 50 Hz,
- history reset-on-done is the production-safe behavior,
- `LITE3_UNREALISTIC_HISTORY_FEED=1` is not valid for real training/eval and should remain unset.

The current task family being worked on is:

```bash
TwoLegStandSafeSlowLowPowerDomainRandV2Incremental-Deeprobotics-Lite3-v0
```

But this document is intended to remain valid for future training tasks too. If a future task changes
observation size, history length, action scale, control timing, or joint ordering, those changes must be
propagated into export and deploy parity checks before running sim2sim.

### `Lite3_rl_deploy`

This is the local deploy/runtime stack. It owns:

- the C++ `rl_deploy` binary,
- state machine execution,
- keyboard/gamepad user command handling,
- ONNXRuntime policy execution,
- robot/simulator interface,
- debug dumps for deploy-side observations/actions/torque diagnostics.

The local x86 sim2sim architecture is two-process:

1. Python MuJoCo simulator process:
   - run from `Lite3_rl_deploy/interface/robot/simulation`,
   - receives joint commands on UDP `0.0.0.0:20001`,
   - sends robot state to deploy on UDP `127.0.0.1:30010`.
2. C++ deploy process:
   - run from `Lite3_rl_deploy/build` as `./rl_deploy`,
   - receives robot state on UDP `:30010`,
   - sends joint commands to the simulator on UDP `127.0.0.1:20001`,
   - runs the full state machine and ONNX policy.

In the current local build, `rl_deploy` is x86-64 and uses the legacy UDP sim path:

- `BUILD_PLATFORM=x86`,
- `BUILD_SIM=ON`,
- `USE_PYBULLET=ON`,
- `USE_MJCPP=OFF`,
- `SEND_REMOTE=OFF`.

Despite the `USE_PYBULLET` name, the same UDP interface is used by the Python MuJoCo simulator.

## State Machine Flow

The deploy experiment must use the full state machine:

```text
Idle -> StandUp -> RLControl -> JointDamping -> Idle
```

Keyboard mapping in simulation:

- `z`: request `StandingUp` from idle,
- `c`: request `RLControlMode` from stand-up or hind-stand,
- `r`: request `JointDamping` from RL.

The recommended human/operator rhythm is:

1. Start MuJoCo simulator.
2. Start `./rl_deploy`.
3. Wait until deploy is alive and reading simulator state.
4. Press `z`.
5. Wait about 5 seconds.
6. Press `c`.
7. Observe behavior.
8. If needed, press `r` to enter joint damping.
9. Reset MuJoCo to the initial state, or restart simulator for a clean deterministic trial.

The 5-second wait is meaningful. `StandUpState` is time-gated: it first moves toward a pre-height pose,
then to the stand height. The code does not transition to RL until roughly `2 * stand_duration` has elapsed.
The configured `stand_duration` is currently 2 seconds, so 5 seconds gives the stand-up phase time to finish.

## Policy Delivery

The policy that runs after pressing `c` is the ONNX policy loaded by `Lite3_rl_deploy`.

Default policy path:

```bash
Lite3_rl_deploy/policy/ppo/policy.onnx
```

Temporary override path:

```bash
export LITE3_POLICY_ONNX=/absolute/path/to/policy.onnx
```

For final local acceptance of a newly trained policy, copy/export the selected ONNX into the deploy policy path
or intentionally run with `LITE3_POLICY_ONNX` while clearly recording the model path and hash in logs.

Generated training run directories normally include helper scripts:

- `run_play.sh`,
- `run_resume.sh`,
- `run_export.sh`.

The export helper should be preferred because it uses the deploy exporter with the expected dimensions.
For the 117-D plus 40-history two-leg stand policy, export is equivalent to:

```bash
cd Lite3_rl_deploy/policy
python pt2onnx.py \
  --ckpt /path/to/model_N.pt \
  --out ppo/policy.onnx \
  --num-obs 117 \
  --history-len 40
```

If preserving the existing deploy policy is important, export to `/tmp/...onnx` first, validate, then copy into
`Lite3_rl_deploy/policy/ppo/policy.onnx` only when ready.

## Training-To-Deploy Gaps To Check

Before judging a failure as a bad policy, verify the known parity gaps.

### ONNX Shape And Export

Expected for current two-leg stand:

- input name: `obs`,
- input shape: `[1, 4797]`,
- output name: `action`,
- output shape: `[1, 12]`.

Run the static parity checker after export:

```bash
python rl_training_new/scripts/tools/validate_deploy_parity.py \
  --policy /path/to/policy.onnx
```

### Timing

Current expected timing:

- training: `sim.dt=0.005`, `decimation=4`, control period `0.02s`,
- deploy legacy UDP: interface dt `0.001`, deploy decimation `20`, control period `0.02s`.

The ONNX runner prints and enforces this timing parity at startup. If a future training task changes policy rate,
update the deploy timing expectations deliberately. Do not ignore a timing mismatch.

### Observation Contract

Current deploy observation frame:

```text
cmd(3)
base_rpy(3)
body_omega(3)
joint_pos(12)
joint_vel * 0.1(12)
joint_pos_history(36)
joint_vel_history(24)
action_history(24)
```

Then the deploy runner flattens:

```text
[current_obs, history_frame0, ..., history_frame39]
```

The training code and deploy code must agree on:

- field order,
- scaling,
- clipping,
- history frame order,
- reset behavior.

### History

Production-safe behavior is reset-on-done/reset. Do not rely on cross-episode history leakage.

Deploy clears history on ONNX state entry. It also has explicit history seeding modes and optional replay-file
support, but those are diagnostic tools, not the acceptance path.

Action history is especially easy to get subtly wrong. Current deploy parity mode uses delayed action history
to mirror the training wrapper behavior.

### Joint Order

Training policy joint order is grouped by joint type:

```text
FL_HipX, FR_HipX, HL_HipX, HR_HipX,
FL_HipY, FR_HipY, HL_HipY, HR_HipY,
FL_Knee, FR_Knee, HL_Knee, HR_Knee
```

Robot/deploy actuator order is leg-grouped:

```text
FL_HipX, FL_HipY, FL_Knee,
FR_HipX, FR_HipY, FR_Knee,
HL_HipX, HL_HipY, HL_Knee,
HR_HipX, HR_HipY, HR_Knee
```

Deploy maps robot state into policy order before inference and maps policy targets back into robot order.
If a future task changes asset joint ordering or action ordering, this must be revalidated.

### Reset / RL Entry State

Do not compare forced-RL dumps to training and conclude policy failure unless the reset states are intentionally
matched. Forced RL can start after idle has sent zero commands long enough for MuJoCo to drift or fall.

The meaningful local deploy start state for acceptance is the state produced by:

```text
MuJoCo initial state -> Idle -> z -> StandUp -> c -> RL
```

Training/eval should be configured to include or mirror the intended deploy reset/stand-up distribution when
the task requires cold-start robustness.

## Experiment Runbook

### 1. Training Command

For the current V2 incremental two-leg stand task, run from `rl_training_new`:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStandSafeSlowLowPowerDomainRandV2Incremental-Deeprobotics-Lite3-v0 \
  --run_name=testing \
  --headless
```

For full diagnostic training logs:

```bash
LITE3_TRAIN_TEE_LOG=1 \
LITE3_DEBUG_TRAIN_DUMPS=9000 \
LITE3_DEBUG_DUMP_EVERY=1 \
LITE3_TRAIN_EVAL_EVERY=200 \
LITE3_TRAIN_EVAL_STEPS=200 \
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStandSafeSlowLowPowerDomainRandV2Incremental-Deeprobotics-Lite3-v0 \
  --run_name=testing_diag \
  --headless
```

For future tasks, replace only the task ID and run name after reading the relevant docs/configs.

### 2. Export Policy

From the completed run directory:

```bash
./run_export.sh --checkpoint model_N.pt --export_path /tmp/lite3_candidate.onnx
```

Then validate:

```bash
python rl_training_new/scripts/tools/validate_deploy_parity.py \
  --policy /tmp/lite3_candidate.onnx
```

If validation passes and this is the policy under test, deliver it to deploy:

```bash
cp /tmp/lite3_candidate.onnx Lite3_rl_deploy/policy/ppo/policy.onnx
```

Alternatively, keep the file in `/tmp` and run deploy with:

```bash
export LITE3_POLICY_ONNX=/tmp/lite3_candidate.onnx
```

### 3. Start MuJoCo Simulator

From one terminal:

```bash
cd Lite3_rl_deploy/interface/robot/simulation
python3 mujoco_simulation.py
```

For headless/debug execution from automation, it is possible to import the module and set `USE_VIEWER=False`
before constructing `MuJoCoSimulation`, but the normal operator flow uses the Python file directly.

### 4. Start Deploy

From another terminal:

```bash
cd Lite3_rl_deploy/build
source ../environment_variables.sh
./rl_deploy
```

If testing a non-default ONNX without copying:

```bash
cd Lite3_rl_deploy/build
source ../environment_variables.sh
export LITE3_POLICY_ONNX=/absolute/path/to/policy.onnx
./rl_deploy
```

Useful behavior-neutral debug settings:

```bash
export LITE3_DEBUG_DUMPS=5
export LITE3_DEBUG_DUMP_DIR=/tmp/lite3_debug/deploy_trial
```

Avoid behavior-changing overrides for acceptance unless explicitly testing them:

```bash
unset LITE3_FORCE_RL_START
unset LITE3_FIXED_CMD
unset LITE3_DEFAULT_CMD
unset LITE3_HISTORY_SEED_FILE
unset LITE3_DISABLE_POSTURE_CHECK
```

### 5. Execute Trial

In the `rl_deploy` terminal:

1. Press `z`.
2. Wait about 5 seconds.
3. Press `c`.
4. Watch MuJoCo behavior and deploy/MuJoCo logs.
5. Press `r` if the policy is unsafe or the trial is complete.
6. Reset MuJoCo or restart simulator before the next clean trial.

## What To Read After A Trial

### Deploy stdout

Look for:

- active behavior overrides,
- ONNX model path and hash,
- ONNX input/output shape,
- timing parity line,
- state transitions:
  - `idle_state ------------> standup_state`,
  - `standup_state ------------> rl_control`,
  - `rl_control ------------> joint_damping`,
- initial RL observation snapshot,
- posture unsafe messages,
- policy execution mode.

### MuJoCo stdout/viewer

Look for:

- stable stand-up,
- RL entry from a reasonable pose,
- RPY growth,
- joint positions/velocities near limits,
- torque spikes,
- target positions that exceed plausible limits,
- whether the behavior visually matches training.

### Deploy Debug Dumps

Files:

```text
debug_cpp_step0.txt
debug_cpp_step1.txt
...
```

Key fields:

- `base_rpy`,
- `body_omega`,
- `joint_pos_policy`,
- `joint_vel_policy`,
- `joint_pos_history`,
- `joint_vel_history`,
- `action_history`,
- `obs_flat`,
- `action_raw`,
- `action_offset`,
- `target_joint_pos`,
- `target_joint_pos_clipped`,
- `pd_tau_raw`,
- `pd_tau_clipped`,
- `history_seed_file_loaded`,
- `action_history_mode`,
- `base_rpy_source`.

### Training Debug Dumps

Training play dumps can be generated with the run helper and compared against deploy dumps when the start
state is intentionally aligned:

```bash
python rl_training_new/scripts/tools/compare_deploy_train_dumps.py \
  --deploy-dir /path/to/deploy_dumps \
  --train-dir /path/to/train_dumps \
  --steps 5 \
  --out /tmp/lite3_debug/compare_report.json
```

Use this to classify observation/history/export mismatches. Do not use it as the only success criterion; the
final acceptance is the full local deploy state-machine behavior in MuJoCo.

## Success And Failure Classification

### Success

A local deploy trial is successful when:

- stand-up completes,
- pressing `c` enters RL cleanly,
- posture guard does not immediately trip,
- the robot maintains the intended behavior in MuJoCo,
- action targets and torques are not violent or saturating,
- behavior resembles the training-side policy,
- debug dumps do not show obvious observation/history/timing/order mismatch.

### Failure Types

Classify failures before changing training:

- **Startup/state-machine failure**: stand-up did not complete or RL entered from an unintended state.
- **Policy behavior failure**: deploy inputs look sane but action/behavior is unsafe or unstable.
- **Observation mismatch**: base RPY, body omega, joint ordering, scaling, or history differ from training.
- **Export mismatch**: ONNX shape/path/hash wrong, or static checker fails.
- **Timing mismatch**: policy control period differs from training.
- **Simulator/interface mismatch**: UDP packets, MuJoCo state, reset state, or torque application differ from expectation.
- **Safety guard failure**: policy exceeds roll/pitch guard or enters joint damping immediately.

## Current Known Lessons

- A forced-RL diagnostic run with the V2 `model_8500.pt` exported to `/tmp` passed static ONNX parity but did
  not constitute behavioral success because RL started from a mismatched/unsafe MuJoCo state.
- That run showed the need to use the real `z -> wait -> c` state-machine flow for acceptance.
- Comparing dumps is useful, but only after the training and deploy start distributions are intentionally aligned.
- The local `Lite3_rl_deploy` stack and the robot stack are not assumed identical. We do not touch the robot stack
  until local MuJoCo deploy behavior is successful.

## Session Checklist

At the start of a future session:

1. Read this file.
2. Read the active task workflow under `rl_training_new/docs/workflows/`.
3. Read the task config/agent config for the chosen task.
4. Confirm training observation/action/timing/history dimensions.
5. Confirm local deploy build mode and ONNX runner expectations.
6. Provide the user the training command.
7. After training completes, export and validate ONNX.
8. Deliver policy to deploy path or set `LITE3_POLICY_ONNX`.
9. Run MuJoCo first, then `./rl_deploy`.
10. Execute `z`, wait about 5 seconds, then `c`.
11. Read logs/dumps and conclude success/failure with a specific failure class.
