# Dockerfile
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.8 python3.8-venv python3-pip \
    build-essential cmake git \
    libgl1-mesa-glx libglib2.0-0 libegl1 \
    libusb-1.0-0-dev libturbojpeg \
    libopencv-dev python3-opencv \
    libboost-all-dev libeigen3-dev \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --upgrade pip

WORKDIR /workspace/Lite3_rl_training
RUN pip install torch torchvision numpy gym opencv-python pyrealsense2

