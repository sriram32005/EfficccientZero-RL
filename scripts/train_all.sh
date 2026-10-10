#!/usr/bin/env bash
# Train both agents. REINFORCE is the long pole -- start it first.
set -euo pipefail
CONFIG=${1:-configs/pong.yaml}
PY=${PY:-.venv/bin/python}

mkdir -p logs
# echo "[*] starting DQN (this is the long one)"
# CUDA_VISIBLE_DEVICES=${DQN_GPU:-0} nohup $PY -m src.dqn.train --config "$CONFIG" > logs/dqn.log 2>&1 &
# DQN_PID=$!

echo "[*] starting REINFORCE (this is the long one)"
CUDA_VISIBLE_DEVICES=${REINFORCE_GPU:-0} nohup $PY -m src.reinforce.train --config "$CONFIG" > logs/reinforce.log 2>&1 &
REINFORCE_PID=$!

echo "[*] starting EfficientZero"
CUDA_VISIBLE_DEVICES=${EZ_GPU:-0} nohup $PY -m src.efficientzero.train --config "$CONFIG" > logs/ez.log 2>&1 &
EZ_PID=$!

echo "    REINFORCE pid=$REINFORCE_PID  (tail -f logs/reinforce.log)"
echo "    EZ        pid=$EZ_PID          (tail -f logs/ez.log)"
wait $EZ_PID; echo "[*] EfficientZero finished"
wait $REINFORCE_PID; echo "[*] REINFORCE finished"
echo "[*] now run:  make demo CONFIG=$CONFIG"
