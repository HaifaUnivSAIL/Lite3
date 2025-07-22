#!/bin/bash

# Exit on error
set -e
echo "This script will follow the instructions in https://github.com/DeepRoboticsLab/Lite3_rl_training# and automatically install: Lite3_rl_training"

git clone git@github.com:DeepRoboticsLab/Lite3_rl_training.git

# Clone legged_gym
#git clone https://github.com/leggedrobotics/legged_gym.git

# Clone rsl_rl
#git clone https://github.com/leggedrobotics/rsl_rl.git


# Source conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "❌ Could not find conda.sh. Make sure Conda is installed and in your home directory."
    exit 1
fi

ENV_NAME=lite3_env
PYTHON_VERSION=3.8

# Step 1: Create and activate conda environment
if ! conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME

    # Step 2: Upgrade pip
    pip install --upgrade pip

    # Step 3: Install PyTorch with CUDA 11.3
    echo "Installing PyTorch with CUDA 11.3..."
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    
    # Step 4: Install Python dependencies
    echo "Installing Python dependencies..."
    pip install transformations matplotlib gym tensorboard numpy==1.23.5
    
else
    echo "✅ Conda environment '$ENV_NAME' already exists. Skipping creation."
fi

conda activate $ENV_NAME

echo "Go to https://developer.nvidia.com/isaac-gym/download download the IsaacGym_Preview_4_Package.tar.gz file and extract the isaacgym folder into Lite3 folder"
read -p "Press Enter once you've downloaded and extracted Isaac Gym..."

cd /home/jmuller/SAIL-Projects/Lite3/isaacgym/python
pip install -e .


# Step 6: Install legged_gym and rsl_rl
echo "Installing rsl_rl..."
cd /home/jmuller/SAIL-Projects/Lite3/Lite3_rl_training/rsl_rl/
pip install -e .

echo "Installing legged_gym..."
cd /home/jmuller/SAIL-Projects/Lite3/Lite3_rl_training/legged_gym/
pip install -e .

cd ..

echo "✅ All done!"
echo "👉 To activate the environment later, run: conda activate $ENV_NAME"

