#!/bin/bash
set -e

IMAGE_NAME=rl_training_new_env
BUILD_ENV=${1:-local}   # default = local, optional = server
ISAAC_SIM_IMAGE=${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}
ISAACLAB_REPO=${ISAACLAB_REPO:-https://github.com/isaac-sim/IsaacLab.git}
ISAACLAB_BRANCH=${ISAACLAB_BRANCH:-main}

echo "======================================"
echo " Building Docker image: $IMAGE_NAME"
echo " Build environment:    $BUILD_ENV"
echo "======================================"

# Move to the script's directory (docker/) to ensure relative paths work
cd "$(dirname "$0")"

# Build from project root context (one directory up)
docker build \
  --build-arg BUILD_ENV=${BUILD_ENV} \
  --build-arg ISAAC_SIM_IMAGE=${ISAAC_SIM_IMAGE} \
  --build-arg ISAACLAB_REPO=${ISAACLAB_REPO} \
  --build-arg ISAACLAB_BRANCH=${ISAACLAB_BRANCH} \
  -t ${IMAGE_NAME} \
  -f Dockerfile \
  ..

echo "✅ Docker build complete for environment: ${BUILD_ENV}"
