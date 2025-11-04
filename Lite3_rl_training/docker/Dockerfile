# Dockerfile
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

# --- System deps ---
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-venv python3-pip python3-dev \
    build-essential cmake git \
    libgl1-mesa-glx libglib2.0-0 libegl1 \
    libusb-1.0-0-dev libturbojpeg \
    libopencv-dev python3-opencv \
    libboost-all-dev libeigen3-dev \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# --- Create venv baked into the image ---
RUN python3.8 -m venv /venv
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"

# Keep pip toolchain modern
RUN pip install --upgrade pip setuptools wheel

# We’ll allow server builds to install *all* deps from requirements_server.txt
# while local builds keep your current behavior.
ARG BUILD_ENV=local

# Copy only the requirements file into the image build context (if present)
# (This line is safe even if the file doesn't exist; if it does not, Docker will fail the build.
# If you sometimes build without the file, comment this COPY and keep the manual pip line below.)
COPY requirements_server.txt /workspace/requirements_server.txt

# --- Dependency install logic ---
# If BUILD_ENV=server -> install from requirements_server.txt
# Else (default local) -> keep your current minimal set
RUN if [ "$BUILD_ENV" = "server" ]; then \
      echo "[Docker] Server build: installing from requirements_server.txt"; \
      pip install -r /workspace/requirements_server.txt; \
    else \
      echo "[Docker] Local build: installing minimal base (unchanged behavior)"; \
      pip install torch torchvision numpy gym opencv-python pyrealsense2; \
    fi

# --- Workdir as before ---
WORKDIR /workspace/Lite3_rl_training

# Default shell
CMD ["/bin/bash"]

