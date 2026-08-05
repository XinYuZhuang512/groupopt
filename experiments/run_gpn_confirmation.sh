#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS=(2345 3456 4567)

if [[ "$#" -eq 0 ]]; then
  MODES=(fixed adaptive_state)
else
  MODES=("$@")
fi

cd "${PROJECT_DIR}"

run_one() {
  local base_mode="$1"
  local seed="$2"
  local output_dir="artifacts/formal/gpn_tsp50_${base_mode}_seed${seed}"
  local resume_args=()
  local latest_checkpoint="${output_dir}/checkpoints/latest.pt"
  local interrupted_checkpoint="${output_dir}/checkpoints/interrupted.pt"

  if [[ -f "${interrupted_checkpoint}" ]] && \
     { [[ ! -f "${latest_checkpoint}" ]] || [[ "${interrupted_checkpoint}" -nt "${latest_checkpoint}" ]]; }; then
    resume_args=(--resume "${interrupted_checkpoint}")
  elif [[ -f "${latest_checkpoint}" ]]; then
    resume_args=(--resume "${latest_checkpoint}")
  fi

  "${PYTHON_BIN}" experiments/train_gpn_experiment.py \
    --base-mode "${base_mode}" \
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps 10000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    --encoder-layers 3 \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed "${seed}" \
    --device cuda
}

for base_mode in "${MODES[@]}"; do
  case "${base_mode}" in
    fixed|adaptive_state) ;;
    *)
      echo "unsupported mode: ${base_mode}" >&2
      exit 2
      ;;
  esac

  for seed in "${SEEDS[@]}"; do
    run_one "${base_mode}" "${seed}"
  done
done
