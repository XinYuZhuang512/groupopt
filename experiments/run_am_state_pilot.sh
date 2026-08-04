#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" experiments/train_am_experiment.py \
  --base-mode adaptive_state \
  --output-dir artifacts/pilot/tsp50_adaptive_state_seed1234 \
  --graph-size 50 \
  --steps 2000 \
  --batch-size 512 \
  --validation-size 1024 \
  --embedding-dim 128 \
  --heads 8 \
  --encoder-layers 3 \
  --feed-forward-dim 512 \
  --normalization batch \
  --learning-rate 1e-4 \
  --eval-every 100 \
  --checkpoint-every 250 \
  --seed 1234 \
  --device cuda
