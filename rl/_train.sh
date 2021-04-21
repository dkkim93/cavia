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
FAST_BATCH_SIZE=15
EP_HORIZON=50
FAST_LR=1
N_INNER=1

# Begin experiment
for SEED in {1..1}
do
    python3 main.py \
    --env-name "2DNavigationVel-v0" \
    --seed $SEED \
    --ep-horizon $EP_HORIZON \
    --fast-batch-size $FAST_BATCH_SIZE \
    --n-inner $N_INNER \
    --fast-lr $FAST_LR

    # python3 main.py \
    # --env-name "2DNavigationAcc-v0" \
    # --seed $SEED \
    # --ep-horizon $EP_HORIZON \
    # --fast-batch-size $FAST_BATCH_SIZE \
    # --fast-lr $FAST_LR

    # python3 main.py \
    # --env-name "2DNavigationVel-v0" "2DNavigationAcc-v0" \
    # --seed $SEED \
    # --ep-horizon $EP_HORIZON \
    # --fast-batch-size $FAST_BATCH_SIZE \
    # --fast-lr $FAST_LR
done
