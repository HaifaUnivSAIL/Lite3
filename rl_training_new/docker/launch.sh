#!/bin/bash
set -e

# -----------------------------
# Container and image settings
# -----------------------------
PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
IMAGE_NAME=rl_training_new_env
CONTAINER_NAME=rl_training_new_train_server

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

### -------------------------------------------------------------------------
### PATCH START: Auto-fix NVIDIA runtime only on systems where it's missing
### -------------------------------------------------------------------------

# 1. Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "unknown")

# 2. Check whether Docker recognizes the "nvidia" runtime
if docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    RUNTIME_AVAILABLE=true
else
    RUNTIME_AVAILABLE=false
fi

# 3. If OS is 24.04+ OR runtime is missing -> remove --runtime=nvidia
if [[ "$UBUNTU_VERSION" == "24.04" || "$UBUNTU_VERSION" > "24.04" || "$RUNTIME_AVAILABLE" = false ]]; then
    echo "[Patch] NVIDIA runtime not available on this system -> switching to --gpus all only"
    # Remove the --runtime=nvidia flag from DOCKER_ARGS
    FILTERED_ARGS=()
    for arg in "${DOCKER_ARGS[@]}"; do
        if [[ "$arg" != "--runtime=nvidia" ]]; then
            FILTERED_ARGS+=("$arg")
        fi
    done
    DOCKER_ARGS=("${FILTERED_ARGS[@]}")
fi

### -------------------------------------------------------------------------
### PATCH END
### -------------------------------------------------------------------------

# Isaac Lab and Isaac Sim paths inside the image
ISAACLAB_ROOT=/opt/IsaacLab
ISAACSIM_PATH=/isaac-sim

echo "======================================"
echo "Launching container: $CONTAINER_NAME"
echo "Project root:        $PROJECT_ROOT"
echo "Image:               $IMAGE_NAME"
echo "Mode:                ${MODE^}"
echo "======================================"

DOCKER_TTY_FLAGS=(-it)
if [[ -n "${NO_TTY}" ]]; then
  DOCKER_TTY_FLAGS=()
fi

docker run "${DOCKER_ARGS[@]}" \
  "${DOCKER_TTY_FLAGS[@]}" --rm \
  --name ${CONTAINER_NAME} \
  --entrypoint /bin/bash \
  \
  -v "${PROJECT_ROOT}:/workspace/rl_training_new" \
  -w /workspace/rl_training_new \
  \
  -e ACCEPT_EULA=Y \
  -e ISAACSIM_PATH="${ISAACSIM_PATH}" \
  -e ISAACLAB_ROOT="${ISAACLAB_ROOT}" \
  \
  --shm-size=2g \
  ${IMAGE_NAME} \
  -lc "if [ -f \"${ISAACSIM_PATH}/setup_python_env.sh\" ]; then source \"${ISAACSIM_PATH}/setup_python_env.sh\"; fi; \
        export EXP_PATH=\"${ISAACSIM_PATH}/apps\"; \
        export CARB_APP_PATH=\"${ISAACSIM_PATH}/kit\"; \
        export OMNI_APP_PATH=\"${ISAACSIM_PATH}/kit\"; \
        export OMNI_KIT_PATH=\"${ISAACSIM_PATH}/kit\"; \
        export OMNI_EXTENSIONS_PATH=\"${ISAACSIM_PATH}/exts:${ISAACSIM_PATH}/exts/isaacsim:${ISAACSIM_PATH}/extsDeprecated:${ISAACSIM_PATH}/extscache:${ISAACSIM_PATH}/kit/exts:\${OMNI_EXTENSIONS_PATH}\"; \
        export OMNI_USER_EXTENSIONS_PATH=\"\${OMNI_USER_EXTENSIONS_PATH:-}\"; \
        export PYTHONPATH=\"/workspace/rl_training_new/source/rl_training:/workspace/rl_training_new/rsl_rl:/workspace/rl_training_new/legged_gym:${ISAACLAB_ROOT}/source/isaaclab:${ISAACLAB_ROOT}/source/isaaclab_assets:${ISAACLAB_ROOT}/source/isaaclab_mimic:${ISAACLAB_ROOT}/source/isaaclab_rl:${ISAACLAB_ROOT}/source/isaaclab_tasks:\${ISAACSIM_PATH}/python_packages:\${ISAACSIM_PATH}/kit/python:\${PYTHONPATH}\"; \
        if [ -d \"/venv/lib/python3.11/site-packages\" ]; then export PYTHONPATH=\"/venv/lib/python3.11/site-packages:\${PYTHONPATH}\"; fi; \
        if [ -f \"/venv/bin/activate\" ]; then source /venv/bin/activate; fi; \
        if [ -d \"/workspace/rl_training_new/rsl_rl\" ]; then \
          if ! ${ISAACSIM_PATH}/python.sh -c \"import rsl_rl\" >/dev/null 2>&1; then \
            ${ISAACSIM_PATH}/python.sh -m pip install -e /workspace/rl_training_new/rsl_rl; \
          fi; \
        fi; \
        if [ -d \"/workspace/rl_training_new/legged_gym\" ]; then \
          if ! ${ISAACSIM_PATH}/python.sh -c \"import legged_gym\" >/dev/null 2>&1; then \
            ${ISAACSIM_PATH}/python.sh -m pip install -e /workspace/rl_training_new/legged_gym; \
          fi; \
        fi; \
        if ! ${ISAACSIM_PATH}/python.sh -c \"import transformations\" >/dev/null 2>&1; then \
          ${ISAACSIM_PATH}/python.sh -m pip install transformations; \
        fi; \
        if ! ${ISAACSIM_PATH}/python.sh -c \"import gym\" >/dev/null 2>&1; then \
          ${ISAACSIM_PATH}/python.sh -m pip install gym==0.26.2; \
        fi; \
        if ! ${ISAACSIM_PATH}/python.sh -c \"import tensordict\" >/dev/null 2>&1; then \
          ${ISAACSIM_PATH}/python.sh -m pip install tensordict; \
        fi; \
        if ! grep -q \"isaac-sim/python.sh\" /root/.bashrc 2>/dev/null; then \
          printf '\n# Isaac Sim python wrapper\npython(){ /isaac-sim/python.sh \"\$@\"; }\npython3(){ /isaac-sim/python.sh \"\$@\"; }\npip(){ /isaac-sim/python.sh -m pip \"\$@\"; }\n' >> /root/.bashrc; \
        fi; \
        echo \"[launch.sh] PYTHONPATH=\${PYTHONPATH}\"; \
        echo \"[launch.sh] ISAACSIM_PATH=\${ISAACSIM_PATH}\"; \
        echo \"[launch.sh] OMNI_EXTENSIONS_PATH=\${OMNI_EXTENSIONS_PATH}\"; \
        echo \"[launch.sh] EXP_PATH=\${EXP_PATH}\"; \
        exec bash"
