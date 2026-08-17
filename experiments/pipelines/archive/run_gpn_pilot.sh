#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODES=(fixed adaptive adaptive_state)

cd "${PROJECT_DIR}"

run_one() {
  local base_mode="$1"
  local output_dir="artifacts/pilot/gpn_tsp50_${base_mode}_seed1234"
  local latest_checkpoint="${output_dir}/checkpoints/latest.pt"
  local interrupted_checkpoint="${output_dir}/checkpoints/interrupted.pt"
  local resume_args=()

  if [[ -f "${interrupted_checkpoint}" ]] && \
     { [[ ! -f "${latest_checkpoint}" ]] || [[ "${interrupted_checkpoint}" -nt "${latest_checkpoint}" ]]; }; then
    resume_args=(--resume "${interrupted_checkpoint}")
  elif [[ -f "${latest_checkpoint}" ]]; then
    resume_args=(--resume "${latest_checkpoint}")
  fi

  "${PYTHON_BIN}" experiments/train/train_gpn_experiment.py \
    --base-mode "${base_mode}" \
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps 2000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    --encoder-layers 3 \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed 1234 \
    --device cuda
}

for mode in "${MODES[@]}"; do
  run_one "${mode}"
done
