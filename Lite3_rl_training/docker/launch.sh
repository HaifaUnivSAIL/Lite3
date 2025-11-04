#!/bin/bash
set -e

# -----------------------------
# Container and image settings
# -----------------------------
PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
IMAGE_NAME=lite3_rl_env
CONTAINER_NAME=lite3_rl_train_server

echo "======================================"
echo "Launching container: $CONTAINER_NAME"
echo "Project root:        $PROJECT_ROOT"
echo "Image:               $IMAGE_NAME"
echo "Mode:                Headless server (no GUI)"
echo "======================================"

docker run --gpus all \
  --runtime=nvidia \
  -it --rm \
  --name ${CONTAINER_NAME} \
  \
  -v "${PROJECT_ROOT}:/workspace/Lite3_rl_training" \
  -w /workspace/Lite3_rl_training \
  \
  -e ISAAC_GYM_ROOT_DIR=/workspace/Lite3_rl_training/isaacgym \
  -e LD_LIBRARY_PATH=/workspace/Lite3_rl_training/isaacgym/lib \
  -e PYTHONPATH="/workspace/Lite3_rl_training:/workspace/Lite3_rl_training/isaacgym/python" \
  \
  -e PYOPENGL_PLATFORM=egl \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  \
  --shm-size=2g \
  ${IMAGE_NAME} \
  bash
