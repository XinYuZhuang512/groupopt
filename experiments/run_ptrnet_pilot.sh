#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODES=(fixed adaptive adaptive_state)

cd "${PROJECT_DIR}"

for base_mode in "${MODES[@]}"; do
  "${PYTHON_BIN}" experiments/train_ptrnet_experiment.py \
    --base-mode "${base_mode}" \
    --output-dir "artifacts/pilot/ptrnet_tsp50_${base_mode}_seed1234" \
    --graph-size 50 \
    --steps 2000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    --encoder-layers 1 \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed 1234 \
    --device cuda
done
