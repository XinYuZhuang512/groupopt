#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_DIR}"

run_one() {
  local family="$1"
  local trainer
  local output_dir
  local model_args=()

  case "${family}" in
    am)
      trainer="experiments/train/train_am_experiment.py"
      output_dir="artifacts/pilot/tsp50_gated_adaptive_state_seed1234"
      model_args=(--heads 8 --encoder-layers 3 --feed-forward-dim 512 --normalization batch)
      ;;
    ptrnet)
      trainer="experiments/train/train_ptrnet_experiment.py"
      output_dir="artifacts/pilot/ptrnet_tsp50_gated_adaptive_state_seed1234"
      model_args=(--encoder-layers 1)
      ;;
    gpn)
      trainer="experiments/train/train_gpn_experiment.py"
      output_dir="artifacts/pilot/gpn_tsp50_gated_adaptive_state_seed1234"
      model_args=(--encoder-layers 3)
      ;;
    *)
      echo "unsupported family: ${family}" >&2
      return 2
      ;;
  esac

  local resume_args=()
  local latest_checkpoint="${output_dir}/checkpoints/latest.pt"
  local interrupted_checkpoint="${output_dir}/checkpoints/interrupted.pt"
  if [[ -f "${interrupted_checkpoint}" ]] && \
     { [[ ! -f "${latest_checkpoint}" ]] || [[ "${interrupted_checkpoint}" -nt "${latest_checkpoint}" ]]; }; then
    resume_args=(--resume "${interrupted_checkpoint}")
  elif [[ -f "${latest_checkpoint}" ]]; then
    resume_args=(--resume "${latest_checkpoint}")
  fi

  "${PYTHON_BIN}" "${trainer}" \
    --base-mode gated_adaptive_state \
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps 2000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    "${model_args[@]}" \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed 1234 \
    --device cuda
}

if [[ "$#" -eq 0 ]]; then
  FAMILIES=(am ptrnet gpn)
else
  FAMILIES=("$@")
fi

for family in "${FAMILIES[@]}"; do
  run_one "${family}"
done
