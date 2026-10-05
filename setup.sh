#!/usr/bin/env bash
# Create the virtual environment and install everything.
#   bash setup.sh          -> CPU / default wheels
#   CUDA=cu121 bash setup.sh -> CUDA 12.1 torch wheels (strongly recommended)
set -euo pipefail

PY=${PY:-python3}
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel

if [[ -n "${CUDA:-}" ]]; then
  pip install torch --index-url "https://download.pytorch.org/whl/${CUDA}"
fi
pip install -r requirements.txt

# ale-py >= 0.10 ships the Atari ROMs; this just proves the env actually builds.
python - <<'PYEOF'
import gymnasium as gym, ale_py
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", frameskip=1)
obs, _ = env.reset(seed=0)
print("Atari OK:", obs.shape, env.action_space, env.unwrapped.get_action_meanings())
env.close()
PYEOF

echo
echo "Done. Activate with:  source .venv/bin/activate"
