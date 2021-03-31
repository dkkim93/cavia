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
FAST_BATCH_SIZE=5
EP_HORIZON=50

# Begin experiment
# python3 main.py \
# --env-name "2DNavigationVel-v0" \
# --ep-horizon $EP_HORIZON \
# --fast-batch-size $FAST_BATCH_SIZE

# python3 main.py \
# --env-name "2DNavigationAcc-v0" \
# --ep-horizon $EP_HORIZON \
# --fast-batch-size $FAST_BATCH_SIZE

python3 main.py \
--env-name "2DNavigationVel-v0" "2DNavigationAcc-v0" \
--ep-horizon $EP_HORIZON \
--fast-batch-size $FAST_BATCH_SIZE
