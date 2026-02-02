# Dockerfile
ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${ISAAC_SIM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV ISAACSIM_PATH=/isaac-sim
ENV PYTHONPATH=/isaac-sim/python_packages

# Isaac Sim base image uses a non-root user; switch to root for apt installs.
USER root

# --- System deps ---
RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip python3-dev \
    build-essential cmake git \
    libgl1 libglib2.0-0 libegl1 \
    libusb-1.0-0-dev libturbojpeg \
    libopencv-dev python3-opencv \
    libboost-all-dev libeigen3-dev \
    libxext6 libxrender1 libxi6 libxrandr2 libxinerama1 libsm6 \
    vulkan-tools ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --- Create venv baked into the image ---
RUN /isaac-sim/python.sh -m venv /venv
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"

# Keep pip toolchain modern
RUN pip install --upgrade pip setuptools wheel

# Optional build target
ARG BUILD_ENV=local

# Isaac Lab source
ARG ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git
ARG ISAACLAB_BRANCH=main
RUN git clone --depth 1 --branch ${ISAACLAB_BRANCH} --recurse-submodules ${ISAACLAB_REPO} /opt/IsaacLab
ENV ISAACLAB_ROOT=/opt/IsaacLab

# Install Isaac Lab packages if present
RUN for pkg in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do \
      if [ -d "/opt/IsaacLab/source/${pkg}" ]; then \
        pip install -e "/opt/IsaacLab/source/${pkg}"; \
      fi; \
    done

COPY docker/requirements_server.txt /tmp/requirements_server.txt

# --- Dependency install logic ---
RUN if [ "$BUILD_ENV" = "server" ]; then \
      echo "[Docker] Server build: installing from requirements_server.txt"; \
      pip install -r /tmp/requirements_server.txt; \
      pip install onnx onnxruntime; \
    else \
      echo "[Docker] Local build: installing minimal base"; \
      pip install numpy==1.26.4 torch torchvision gymnasium opencv-python transformations gym==0.26.2 tensordict onnx onnxruntime; \
    fi

# --- Isaac Sim python wrapper for interactive shells ---
RUN if ! grep -q "isaac-sim/python.sh" /root/.bashrc 2>/dev/null; then \
      printf '\n# Isaac Sim python wrapper\npython(){ /isaac-sim/python.sh "$@"; }\npython3(){ /isaac-sim/python.sh "$@"; }\npip(){ /isaac-sim/python.sh -m pip "$@"; }\n' >> /root/.bashrc; \
    fi

# --- Workdir as before ---
WORKDIR /workspace/rl_training_new

# Default shell
CMD ["/bin/bash"]
