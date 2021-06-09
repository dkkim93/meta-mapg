#!/bin/bash

# Load python virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Set GPU device ID
export CUDA_VISIBLE_DEVICES=-1

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
