#!/bin/bash

HOST_DIR=~/Lite3/Lite3_rl_training

xhost +local:root > /dev/null

docker run --gpus all \
  --runtime=nvidia \
  -it --rm \
  --name lite3_rl_gui \
  -v ${HOST_DIR}:/workspace/Lite3_rl_training \
  -w /workspace/Lite3_rl_training \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e ISAAC_GYM_ROOT_DIR=/workspace/Lite3_rl_training/isaacgym \
  -e LD_LIBRARY_PATH=/workspace/Lite3_rl_training/isaacgym/lib \
  -e PYTHONPATH="/workspace/Lite3_rl_training:/workspace/Lite3_rl_training/isaacgym/python" \
  -e TORCH_CUDA_ARCH_LIST="8.9" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /usr/share/vulkan:/usr/share/vulkan:ro \
  -v /usr/lib/x86_64-linux-gnu/libvulkan.so.1:/usr/lib/x86_64-linux-gnu/libvulkan.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libGL.so.1:/usr/lib/x86_64-linux-gnu/libGL.so.1:ro \
  lite3_rl_env \
  bash

