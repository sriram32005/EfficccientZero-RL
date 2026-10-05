#!/usr/bin/env bash
# Train both agents. DQN is the long pole -- start it first.
set -euo pipefail
CONFIG=${1:-configs/pong.yaml}
PY=${PY:-.venv/bin/python}

mkdir -p logs
echo "[*] starting DQN (this is the long one)"
CUDA_VISIBLE_DEVICES=${DQN_GPU:-0} nohup $PY -m src.dqn.train --config "$CONFIG" > logs/dqn.log 2>&1 &
DQN_PID=$!

echo "[*] starting EfficientZero"
CUDA_VISIBLE_DEVICES=${EZ_GPU:-0} nohup $PY -m src.efficientzero.train --config "$CONFIG" > logs/ez.log 2>&1 &
EZ_PID=$!

echo "    DQN pid=$DQN_PID  (tail -f logs/dqn.log)"
echo "    EZ  pid=$EZ_PID   (tail -f logs/ez.log)"
wait $EZ_PID; echo "[*] EfficientZero finished"
wait $DQN_PID; echo "[*] DQN finished"
echo "[*] now run:  make demo CONFIG=$CONFIG"
