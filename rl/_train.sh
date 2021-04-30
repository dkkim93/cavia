#!/bin/bash

# Load python virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# For MuJoCo
# NOTE Below MuJoCo and GLEW path may differ depends on a computer setting
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$HOME/.mujoco/mjpro150/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Set hyperparameter
FAST_BATCH_SIZE=20
EP_HORIZON=200
META_BATCH_SIZE=40
FAST_LR=0.1
N_INNER=3

# Begin experiment
for SEED in {1..1}
do
    # python3 main.py \
    # --env-name "HalfCheetahVel-v1" \
    # --seed $SEED \
    # --ep-horizon $EP_HORIZON \
    # --fast-batch-size $FAST_BATCH_SIZE \
    # --n-inner $N_INNER \
    # --meta-batch-size $META_BATCH_SIZE \
    # --fast-lr $FAST_LR

    # python3 main.py \
    # --env-name "AntVel-v1" \
    # --seed $SEED \
    # --ep-horizon $EP_HORIZON \
    # --fast-batch-size $FAST_BATCH_SIZE \
    # --n-inner $N_INNER \
    # --meta-batch-size $META_BATCH_SIZE \
    # --fast-lr $FAST_LR

    python3 main.py \
    --env-name "HalfCheetahVel-v1" "AntVel-v1" \
    --seed $SEED \
    --ep-horizon $EP_HORIZON \
    --fast-batch-size $FAST_BATCH_SIZE \
    --n-inner $N_INNER \
    --meta-batch-size $META_BATCH_SIZE \
    --fast-lr $FAST_LR
done
