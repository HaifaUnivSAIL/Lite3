#!/bin/bash

HOST_DIR=~/lite3_project/Lite3_rl_training

# Allow container to open GUI on host display
xhost +local:root > /dev/null

docker run --gpus all \
  --runtime=nvidia \
  -it --rm \
  -v ${HOST_DIR}:/workspace/Lite3_rl_training \
  -w /workspace/Lite3_rl_training \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e ISAAC_GYM_ROOT_DIR=/workspace/Lite3_rl_training/isaacgym \
  -e LD_LIBRARY_PATH=/workspace/Lite3_rl_training/isaacgym/lib \
  -e PYTHONPATH="/workspace/Lite3_rl_training:/workspace/Lite3_rl_training/isaacgym/python" \
  -e TORCH_CUDA_ARCH_LIST="8.9" \
  --name lite3_rl_gui \
  lite3_rl_env \
  bash

