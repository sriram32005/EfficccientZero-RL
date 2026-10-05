#!/usr/bin/env bash
# Build every demo asset and print where they landed.
set -euo pipefail
CONFIG=${1:-configs/pong.yaml}
PY=${PY:-.venv/bin/python}
$PY -m demo.build_demo --config "$CONFIG"
echo
echo "Open these during the talk:"
echo "  demo_out/demo_reel.mp4          <- the whole story, one file"
echo "  demo_out/pairs/*.mp4            <- individual side-by-side clips"
echo "  demo_out/showcase_efficientzero.mp4"
echo "  demo_out/learning_curves.png demo_out/metrics_table.png"
