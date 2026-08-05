#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODES=(adaptive_state_mean adaptive_state_start)

cd "${PROJECT_DIR}"

run_one() {
  local family="$1"
  local base_mode="$2"
  local train_script="experiments/train_am_experiment.py"
  local encoder_layers=3
  local output_dir="artifacts/ablation/pilot/am_tsp50_${base_mode}_seed1234"

  if [[ "${family}" == "ptrnet" ]]; then
    train_script="experiments/train_ptrnet_experiment.py"
    encoder_layers=1
    output_dir="artifacts/ablation/pilot/ptrnet_tsp50_${base_mode}_seed1234"
  fi

  local checkpoint="${output_dir}/checkpoints/latest.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    return 1
  fi

  "${PYTHON_BIN}" "${train_script}" \
    --base-mode "${base_mode}" \
    --output-dir "${output_dir}" \
    --resume "${checkpoint}" \
    --graph-size 50 \
    --steps 10000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    --heads 8 \
    --encoder-layers "${encoder_layers}" \
    --feed-forward-dim 512 \
    --normalization batch \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed 1234 \
    --device cuda
}

for family in am ptrnet; do
  for mode in "${MODES[@]}"; do
    run_one "${family}" "${mode}"
  done
done
