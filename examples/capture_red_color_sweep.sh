#!/usr/bin/env bash
# capture_red_color_sweep.sh — capture a "red" vector from the model and emit a
# dose sweep (one .npy per norm) in one shot.
#
# Edit these three, then run:  ./examples/capture_red_color_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

GGUF="${GGUF:-model/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}"
PY="${PY:-venv/bin/python}"          # fall back to python3 if venv missing
OUT="${OUT:-red.vec.npy}"            # stem; sweep writes red.vec.n<N>.npy

command -v "$PY" >/dev/null || PY=python3

# 1. capture the "red" direction from a few rich contexts (reads the table once),
#    then 2. write one .npy per norm: 0.25 0.5 1 2 4
"$PY" examples/capture_vector.py \
    --gguf "$GGUF" \
    --text "My favorite color is red" \
    --text "The color of a ripe strawberry is red" \
    --text "Red is the color of blood and roses" \
    --text "I love the color red" \
    --at all \
    --sweep-norms 0.25 0.5 1 2 4 \
    --out "$OUT"

echo
echo "done. Files next to $OUT:"
ls -1 "${OUT%.npy}".n*.npy
echo
echo "Use one in a knowledge entry (op: add honors scale; set = full replace):"
echo '  { "trigger": "My favorite color is", "op": "set", "vector": "red.vec.n1.npy", "at": "all" }'
