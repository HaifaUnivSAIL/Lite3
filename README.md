# Lite3 Project

The `lite3_project` repository integrates all the software components needed to train reinforcement learning (RL) policies in simulation and deploy them on the Lite3 quadruped robot. This includes training environments, deployment SDKs, ROS 1 communication packages, and the official motion control SDK from DeepRobotics.

---

## 🗂 Repository Structure

```
lite3_project/
├── Lite3_rl_training/     # RL training pipeline using Isaac Gym and RSL-RL - YOU NEED TO CLONE THAT - LOOK BELOW
├── Lite3_rl_deploy/       # C++ SDK for policy deployment on Lite3
├── lite3_ws/              # ROS 1 workspace (for legacy/noetic build)
├── Lite3_MotionSDK-main/  # Official Lite3 Motion SDK from DeepRobotics
```

---

## 🧰 Prerequisites

- Ubuntu 20.04 or 24.04
- NVIDIA GPU with CUDA installed
- Python 3.8–3.10 (recommended: 3.10)
- CMake ≥ 3.16
- Docker (for containerized builds)
- ROS 1 Noetic

---

## 🚀 Installation Guide

Each component of the project can be installed independently. Below are modular installation blocks for each.

---

### 📦 Clone the Repository

```bash
git clone --recurse-submodules https://github.com/yourusername/lite3_project.git
cd lite3_project
```

---

## 🧪 RL Training Environment (`Lite3_rl_training`)

This module allows training policies in simulation using Isaac Gym and the RSL-RL algorithm.

### 🔧 Requirements

- Isaac Gym Preview 4 (NVIDIA)
- PyTorch with CUDA
- Python ≥3.8

### 🔧 Setup

```bash
cd Lite3_rl_training

# Ensure Isaac Gym is placed in: ./isaacgym/
# This should include folders like 'gymapi', 'gymtorch', etc.

# Install legged_gym
cd legged_gym
pip install -e .

# Install RSL-RL
cd ../rsl_rl
pip install -e .
```

> ⚠️ Make sure your CUDA version and GPU driver are compatible with Isaac Gym Preview 4.

---

## 🤖 Deployment SDK (`Lite3_rl_deploy`)

C++ SDK to deploy trained RL policies on Lite3 platforms (x86 or Jetson NX).

### 🔧 Setup

```bash
git clone --recurse-submodules https://github.com/DeepRoboticsLab/Lite3_rl_deploy.git
# Set working directory (replace if cloned elsewhere)
cd ~/lite3_project

# Download and extract LibTorch (CUDA 12.1 build, compatible with CUDA 12.0)
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip -O libtorch_cuda121.zip
unzip libtorch_cuda121.zip -d libtorch

# Go to the rl_deploy package
cd Lite3_rl_deploy

# Create a build directory
mkdir -p build && cd build

# Run CMake with the correct Torch path and build options
cmake .. \
  -DBUILD_PLATFORM=x86 \
  -DBUILD_SIM=off \
  -DSEND_REMOTE=OFF \
  -DTorch_DIR=$(realpath ../../libtorch/share/cmake/Torch) \
  -DCMAKE_PREFIX_PATH=$(realpath ../../libtorch)

# Build the deployment binaries
make -j4
```

---

## 🧭 Motion SDK (`Lite3_MotionSDK-main`)

This SDK allows direct control over the Lite3 robot motors and sensor feedback, provided by DeepRobotics.

### 🔧 Setup

```bash
cd Lite3_MotionSDK-main
mkdir build && cd build

cmake ..
make -j$(nproc)
```

> 📌 This SDK may require additional dependencies or driver access for real-hardware communication. Refer to the SDK documentation inside the folder.

---

## 🤝 ROS 1 Integration (`Lite3_ROS` & `lite3_ws`)

Use these packages to bridge C++ deployment SDK with ROS 1 for telemetry, remote control, or diagnostics.

### 🐳 Docker Setup (Recommended)

```bash
cd lite3_ws/docker/ros1_noetic
./launch.sh
```

This mounts the `lite3_ws` workspace and starts the ROS 1 environment in a container.

### 🛠 Native Build (Alternative)

```bash
cd lite3_ws
catkin_make
source devel/setup.bash
```

---

## 🐳 Docker Overview

The repository includes Dockerfiles for:

- `Lite3_rl_training/docker`: for simulation training
- `lite3_ws/docker/ros1_noetic`: for ROS 1 development

Each Docker directory includes:

- `Dockerfile`
- `launch.sh`: a convenience script to mount source and build volumes

---

## 📈 Workflow Summary

1. Train policy in `Lite3_rl_training` using Isaac Gym.
2. Export trained model weights.
3. Compile `Lite3_rl_deploy` with policy weights.
4. (Optionally) Use `Lite3_ROS` and `Lite3_MotionSDK` to interface and control the physical robot via ROS 1.

---

## 📝 Version Control Notes

Some components (like `isaacgym` and `Lite3_MotionSDK-main`) may be excluded from the repository due to licensing or binary size. Use `.gitignore` accordingly and document manual steps for setup.

If needed, mark large or static folders as unchanged:

```bash
git update-index --assume-unchanged Lite3_rl_training/isaacgym/*
```

---

## ✅ TODO

- [ ] Add export tools for RL policy
- [ ] Provide launch scripts for evaluation on robot
- [ ] Include trained policy examples

---

## 🪪 License

This project includes third-party components under their respective licenses:

- Isaac Gym (NVIDIA EULA)
- RSL-RL (MIT)
- DeepRobotics SDK (proprietary)

Please verify license compliance before distribution or commercial use.

---

## 📬 Contact

For technical questions or collaboration inquiries, please contact the project maintainer or open an issue.
