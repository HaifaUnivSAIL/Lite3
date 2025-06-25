#!/bin/bash

# Absolute path to the lite3_ws folder on the host
HOST_WS=~/lite3_project/lite3_ws

# Docker image name
IMAGE=ros1-lite3

# Docker container name
CONTAINER_NAME=ros1-lite3-container

# Remove container if it already exists
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Removing existing container '${CONTAINER_NAME}'..."
    docker rm -f ${CONTAINER_NAME}
fi

# Run the container and start in /lite3_ws
docker run -it --rm \
    --name ${CONTAINER_NAME} \
    -v ${HOST_WS}:/lite3_ws \
    -w /lite3_ws \
    ${IMAGE}

