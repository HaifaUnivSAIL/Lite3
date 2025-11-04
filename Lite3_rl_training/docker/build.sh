#!/bin/bash
set -e

IMAGE_NAME=lite3_rl_env
BUILD_ENV=${1:-local}   # default = local, optional = server

echo "======================================"
echo " Building Docker image: $IMAGE_NAME"
echo " Build environment:    $BUILD_ENV"
echo "======================================"

# Move to the script's directory (docker/) to ensure relative paths work
cd "$(dirname "$0")"

# Build from project root context (one directory up)
docker build \
  --build-arg BUILD_ENV=${BUILD_ENV} \
  -t ${IMAGE_NAME} \
  -f Dockerfile \
  ..

echo "✅ Docker build complete for environment: ${BUILD_ENV}"

