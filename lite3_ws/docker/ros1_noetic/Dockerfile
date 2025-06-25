FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive

# Install ROS build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    python3-rosdep \
    ros-noetic-catkin \
    ros-noetic-tf \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rm -f /etc/ros/rosdep/sources.list.d/20-default.list && \
    rosdep init && \
    rosdep update

# Source ROS setup on startup
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

