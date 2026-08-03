#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS=(1234 2345 3456 4567 5678)

cd "${PROJECT_DIR}"

run_one() {
  local base_mode="$1"
  local seed="$2"
  local output_dir="artifacts/formal/tsp50_${base_mode}_seed${seed}"
  local resume_args=()
  local latest_checkpoint="${output_dir}/checkpoints/latest.pt"
  local interrupted_checkpoint="${output_dir}/checkpoints/interrupted.pt"

  if [[ -f "${interrupted_checkpoint}" ]] && \
     { [[ ! -f "${latest_checkpoint}" ]] || [[ "${interrupted_checkpoint}" -nt "${latest_checkpoint}" ]]; }; then
    resume_args=(--resume "${interrupted_checkpoint}")
  elif [[ -f "${latest_checkpoint}" ]]; then
    resume_args=(--resume "${latest_checkpoint}")
  fi

  "${PYTHON_BIN}" experiments/train_am_experiment.py \
    --base-mode "${base_mode}" \
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps 10000 \
    --batch-size 512 \
    --validation-size 2048 \
    --embedding-dim 128 \
    --heads 8 \
    --encoder-layers 3 \
    --feed-forward-dim 512 \
    --normalization batch \
    --learning-rate 1e-4 \
    --eval-every 250 \
    --checkpoint-every 1000 \
    --seed "${seed}" \
    --device cuda
}

for seed in "${SEEDS[@]}"; do
  run_one fixed "${seed}"
  run_one adaptive "${seed}"
done
