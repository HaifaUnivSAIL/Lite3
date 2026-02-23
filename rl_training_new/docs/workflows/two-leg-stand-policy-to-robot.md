# Two-Leg Stand Policy: Training to Robot Deployment

This document captures the exact workflow and stack details used to reach a stable two-leg-stand policy in deploy simulation, and what is currently documented for real robot execution.

## 1) Training Side (Policy Provenance)

### 1.1 Exact task/config used

- Task ID: `TwoLegStand-Deeprobotics-Lite3-v0` (`rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/config/lite3/__init__.py:27`).
- Env config: `Lite3TwoLegStandEnvCfg` (`rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/config/lite3/__init__.py:31`).
- PPO runner config: `Lite3TwoLegStandPPORunnerCfg` (`rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/config/lite3/__init__.py:32`).
- Control/sim timing: `decimation=4`, `sim.dt=0.005` (`rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/two_leg_stand_env_cfg.py:963` and `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/two_leg_stand_env_cfg.py:967`).
- Observation history length: `40` (`rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/two_leg_stand_env_cfg.py:944`).
- Command ranges in this task are standing-only zeros (no walking commands): `lin_vel_x/y=0`, `ang_vel_z=0` (`rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/two_leg_stand_env_cfg.py:102`).

### 1.2 Reward strategy and curriculum that produced the policy

The successful baseline was the default `get_two_leg_stand_phases()` curriculum (`rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/mdp/curriculums.py:171`), with the following phase strategy:

1. `phase_0_legs_up_warmup` (`trigger_thresh=0`):
   - `front_legs_up_warmup`, `torso_upright_warmup`, `base_height_bonus`
   - No front tap penalty in this first phase (`front_tap_penalty=0.0`)
2. `phase_0_basic` (`trigger_thresh=500`):
   - Same core upright/legs-up rewards, tap penalty enabled (`-1.0`)
3. `phase_1_posture_alignment` (`trigger_thresh=1000`):
   - Adds `stand_still_roll_only`, `hind_legs_calmness`
   - Near-goal reset probability increases to `0.45`
4. `phase_2_fine_standing_roll_supression` (`trigger_thresh=2500`):
   - Strong roll suppression (`stand_still_roll_only=10.0`)
   - Includes `hind_leg_extension_geom=8.0`
   - Near-goal reset probability `0.7`

Reference: `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/mdp/curriculums.py:175`, `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/mdp/curriculums.py:187`, `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/mdp/curriculums.py:200`, `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/mdp/curriculums.py:214`.

Base reward definitions (upright/legs-up/posture/height) are in:
`rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/config/lite3/base_env_cfg.py:76`.

### 1.3 Critical history semantics (parity fix)

Production behavior now resets observation history on episode done/reset by default.

- Default mode: reset-on-done.
- Debug-only legacy mode (unrealistic leak across episodes): `LITE3_UNREALISTIC_HISTORY_FEED=1`.

References:
- `rl_training_new/source/rl_training/rl_training/utils/env_wrappers.py:23`
- `rl_training_new/source/rl_training/rl_training/utils/env_wrappers.py:65`
- `rl_training_new/source/rl_training/rl_training/utils/env_wrappers.py:207`
- `rl_training_new/scripts/reinforcement_learning/rsl_rl/train.py:120`
- `rl_training_new/scripts/reinforcement_learning/rsl_rl/train.py:306`

### 1.4 Commands used in practice

Train:

```bash
cd /workspace/rl_training_new
unset LITE3_UNREALISTIC_HISTORY_FEED
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=TwoLegStand-Deeprobotics-Lite3-v0 \
  --headless \
  --run_name <run_name>
```

Resume (from run directory):

```bash
./run_resume.sh --checkpoint model_3000.pt
```

ONNX export (from run directory, generated automatically by `train.py`):

```bash
./run_export.sh --checkpoint model_6000.pt
# Optional:
./run_export.sh --checkpoint model_6000.pt --export_path /abs/path/policy.onnx
```

`run_export.sh` generation and defaults are defined in:
`rl_training_new/scripts/reinforcement_learning/rsl_rl/train.py:564` and `rl_training_new/scripts/reinforcement_learning/rsl_rl/train.py:666`.

## 2) Future Tasks Built from This Baseline

1. External push robustness:
   - Start from `TwoLegStandRobust-Deeprobotics-Lite3-v0`.
   - This config already enables stronger randomization and interval pushes (`randomize_push_robot`) in `Lite3TwoLegStandRobustEnvCfg`.
   - Reference: `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/config/lite3/base_env_cfg.py:314`.
2. Stricter actuator safety/torque behavior:
   - Start from `Lite3TwoLegStandSafeEnvCfg`, and tighten `torque_limits`, `dof_vel_limits`, `power`, and safety gates.
   - Reference: `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/config/lite3/base_env_cfg.py:228`.
3. Two-leg standing with commanded motion (toward bipedal walk):
   - Expand command ranges from the current all-zero command config (`lin_vel_x/y`, `ang_vel_z`) and introduce staged command curriculum.
   - Reference: `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/two_leg_stand_env_cfg.py:90`.

## 3) Deploy Stack (Technical Overview)

### 3.1 Runtime architecture

`rl_deploy` starts a state machine:
- `Idle -> StandUp -> RL -> JointDamping` (`Lite3_rl_deploy/README_EN.md:140`).
- Default simulation user input is keyboard; non-sim defaults to retroid gamepad (`Lite3_rl_deploy/state_machine/state_machine.hpp:118`).

The binary prints active behavior overrides at startup; if none are set it runs default control flow:
`Lite3_rl_deploy/main.cpp:18`.

### 3.2 Sim-to-sim (MuJoCo / PyBullet) message flow

Policy/controller side (`rl_deploy`) uses `PybulletInterface` UDP:
- Receives robot state on UDP `:30010` as:
  `double timestamp + 45 floats` (`rpy, acc, omega, q, dq, tau`) (`Lite3_rl_deploy/interface/robot/simulation/pybullet_interface.hpp:124`).
- Sends joint command on UDP `127.0.0.1:20001` as:
  `12f kp | 12f pos | 12f kd | 12f vel | 12f tau` (`Lite3_rl_deploy/interface/robot/simulation/pybullet_interface.hpp:223`).

Simulator side (both MuJoCo and PyBullet scripts) matches the same packet contract:
- MuJoCo defaults:
  - receives control on `0.0.0.0:20001`,
  - sends state to `127.0.0.1:30010`
  (`Lite3_rl_deploy/interface/robot/simulation/mujoco_simulation.py:17`, `Lite3_rl_deploy/interface/robot/simulation/mujoco_simulation.py:19`, `Lite3_rl_deploy/interface/robot/simulation/mujoco_simulation.py:228`, `Lite3_rl_deploy/interface/robot/simulation/mujoco_simulation.py:290`).
- PyBullet script uses equivalent packing/unpacking:
  (`Lite3_rl_deploy/interface/robot/simulation/pybullet_simulation.py:173`, `Lite3_rl_deploy/interface/robot/simulation/pybullet_simulation.py:188`).

### 3.3 ONNX policy execution details

- Model path default: `Lite3_rl_deploy/policy/ppo/policy.onnx`.
- Override: `LITE3_POLICY_ONNX`.
- Input/output tensor names: `obs` -> `action`.
- Observation contract in deploy runner: `117` current frame + `40` history frames.
- Action scale/parity: raw action clipped and scaled by `0.25`.

Reference:
`Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp:243`, `Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp:253`, `Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp:258`, `Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp:666`.

### 3.4 Rate/decimation alignment

Deploy defaults to training-equivalent control period:
- Training reference: `sim.dt=0.005`, `decimation=4`, so control dt `0.02s`.
- Deploy computes decimation from sim dt and allows override via `LITE3_POLICY_DECIMATION` / `LITE3_POLICY_CONTROL_DT`.

Reference:
- Training: `rl_training_new/source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/two_leg_stand_env_cfg.py:963`.
- Deploy: `Lite3_rl_deploy/state_machine/rl_control_state_onnx.hpp:185`.

## 4) Real Robot Path and Host-PC UDP Status

### 4.1 What is clearly documented in this repo

`Lite3_rl_deploy/README_EN.md` documents:
- Sim2sim build/run on x86 (`-DBUILD_SIM=ON`) (`Lite3_rl_deploy/README_EN.md:17`).
- Sim2real build/run on robot arm host (`-DBUILD_PLATFORM=arm -DBUILD_SIM=OFF`) (`Lite3_rl_deploy/README_EN.md:76`).

Hardware interface defaults in code:
- Robot target IP/ports are initialized as `192.168.2.1:43893` and local receive `43897`
  (`Lite3_rl_deploy/interface/robot/hardware/lite3_hardware_interface.hpp:25`).

Motion host networking workflow is documented in MotionSDK docs:
- edit `~/jy_exe/conf/network.toml`,
- set destination `ip` to host IP,
- ports `target_port=43897`, `local_port=43893`,
- restart motion services.
Reference: `Lite3_rl_deploy/third_party/Lite3_MotionSDK/README.md:117`.

### 4.2 Quick guide: host PC -> robot over UDP (documented path)

The documented host-PC UDP path in this repo is via the MotionSDK sample (`Lite_motion`), not the `rl_deploy` ONNX binary:

1. Connect host PC to robot network (WiFi or Ethernet), SSH to motion host, edit `~/jy_exe/conf/network.toml` and set:
   - `ip=<your host static IP>`
   - `target_port=43897`
   - `local_port=43893`
2. Restart robot motion services on the motion host.
3. On host PC, compile MotionSDK sample for x86:
   - `cmake .. -DBUILD_PLATFORM=x86`
   - `make -j`
4. Run `./Lite_motion` on host PC and verify:
   - robot state is received from robot,
   - control commands are accepted (robot can execute the demo behavior).

References:
- `Lite3_rl_deploy/third_party/Lite3_MotionSDK/README.md:117`
- `Lite3_rl_deploy/third_party/Lite3_MotionSDK/README.md:195`
- `Lite3_rl_deploy/third_party/Lite3_MotionSDK/README.md:215`
- `Lite3_rl_deploy/third_party/Lite3_MotionSDK/README.md:233`

### 4.3 Missing documentation (explicit note)

There is no dedicated, end-to-end document in this repo for running **this `rl_deploy` ONNX policy binary on an x86 host PC** and controlling the real robot via UDP as a first-class supported path.

The available host-side UDP procedure is documented for the generic MotionSDK sample (`Lite_motion`), not for `rl_deploy` ONNX policy serving.

Also, current CMake non-sim linking selects arm SDK when `BUILD_SIM=OFF`, while x86 SDK linking is in the `BUILD_SIM` branch:
- `Lite3_rl_deploy/CMakeLists.txt:157`
- `Lite3_rl_deploy/CMakeLists.txt:165`

So for host-PC real-robot ONNX deploy, a dedicated build/runtime guide and validation path is still needed.

## 5) Practical Pre-Flight Before Physical Test

1. Ensure history leak mode is off:
   - `unset LITE3_UNREALISTIC_HISTORY_FEED`.
2. Ensure deploy is in normal state-machine flow:
   - do not set `LITE3_FORCE_RL_START`, `LITE3_FIXED_CMD`, `LITE3_DEFAULT_CMD` unless intentionally testing overrides.
3. Use run-generated helper scripts:
   - `run_resume.sh`, `run_play.sh`, `run_export.sh`.
4. Validate ONNX parity in sim2sim before robot.
5. Keep posture safety guard enabled initially (do not set `LITE3_DISABLE_POSTURE_CHECK=1`) and tighten limits only after stable behavior.
