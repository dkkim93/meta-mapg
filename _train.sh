#!/bin/bash

# Load python virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Set GPU device ID
export CUDA_VISIBLE_DEVICES=-1

# For MuJoCo
# Please note that below MuJoCo and GLEW path may differ
# depends on a computer setting
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Begin experiment
for SEED in {1..10}
do
    python3 main.py \
    --seed $SEED \
    --config "ipd.yaml" \
    --opponent-shaping \
    --prefix ""

    python3 main.py \
    --seed $SEED \
    --config "ipd.yaml" \
    --opponent-shaping \
    --test-mode \
    --prefix "test"
done
