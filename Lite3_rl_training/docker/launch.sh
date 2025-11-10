#!/bin/bash
set -e

# -----------------------------
# Container and image settings
# -----------------------------
PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
IMAGE_NAME=lite3_rl_env
CONTAINER_NAME=lite3_rl_train_server

MODE=${1:-headless}

append_display_mounts() {
  # X11 path: share DISPLAY + authority socket
  if [[ -n "${DISPLAY}" ]]; then
    DOCKER_ARGS+=(
      -e DISPLAY="${DISPLAY}"
      -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    )
    local xa="${XAUTHORITY:-${HOME}/.Xauthority}"
    if [[ -f "${xa}" ]]; then
      DOCKER_ARGS+=(
        -e XAUTHORITY="${xa}"
        -v "${xa}:${xa}:ro"
      )
    fi
  fi

  # Wayland path: forward runtime dir sockets if available
  if [[ -n "${WAYLAND_DISPLAY}" ]]; then
    local runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    if [[ -S "${runtime_dir}/${WAYLAND_DISPLAY}" ]]; then
      DOCKER_ARGS+=(
        -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY}"
        -e XDG_RUNTIME_DIR=/tmp/xdg-runtime
        -v "${runtime_dir}:/tmp/xdg-runtime:rw"
      )
    fi
  fi
}

case "$MODE" in
  headless)
    echo "Mode:                Headless server (no GUI)"
    DOCKER_ARGS=(
      --gpus all
      --runtime=nvidia
      -e PYOPENGL_PLATFORM=egl
      -e NVIDIA_VISIBLE_DEVICES=all
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
    )
    ;;
  local)
    echo "Mode:                Local interactive (with GUI)"
    DOCKER_ARGS=(
      --gpus all
      --runtime=nvidia
      -e NVIDIA_VISIBLE_DEVICES=all
      -e NVIDIA_DRIVER_CAPABILITIES=all
    )
    append_display_mounts
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac

echo "======================================"
echo "Launching container: $CONTAINER_NAME"
echo "Project root:        $PROJECT_ROOT"
echo "Image:               $IMAGE_NAME"
echo "Mode:                ${MODE^}"
echo "======================================"

docker run "${DOCKER_ARGS[@]}" \
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
  --shm-size=2g \
  ${IMAGE_NAME} \
  bash
