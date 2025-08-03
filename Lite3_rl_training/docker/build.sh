#!/bin/bash

IMAGE_NAME=lite3_rl_env

echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME .
