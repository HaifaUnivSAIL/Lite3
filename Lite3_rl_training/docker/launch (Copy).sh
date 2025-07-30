#!/bin/bash

HOST_DIR=~/lite3_project/Lite3_rl_training

docker run --gpus all \
  -it --rm \
  -v ${HOST_DIR}:/workspace/Lite3_rl_training \
  -w /workspace/Lite3_rl_training \
  -e ISAAC_GYM_ROOT_DIR=/workspace/Lite3_rl_training/isaacgym \
  -e LD_LIBRARY_PATH=/workspace/Lite3_rl_training/isaacgym/lib \
  -e PYTHONPATH="/workspace/Lite3_rl_training:/workspace/Lite3_rl_training/isaacgym/python" \
  -e TORCH_CUDA_ARCH_LIST="8.9" \
  lite3_rl_env \
  bash -c " \
    pip install -e isaacgym/python && \
    pip install -e rsl_rl && \
    pip install -e legged_gym && \
    bash"

