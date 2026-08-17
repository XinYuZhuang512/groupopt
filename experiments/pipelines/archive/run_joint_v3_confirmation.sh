#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS=(2345 3456 4567)
FAMILIES=(am ptrnet gpn)
MODES=(joint_fixed joint_free)

cd "${PROJECT_DIR}"

run_one() {
  local family="$1"
  local base_mode="$2"
  local seed="$3"
  local trainer
  local prefix
  local model_args=()

  case "${family}" in
    am)
      trainer="experiments/train/train_am_experiment.py"
      prefix="tsp50"
      model_args=(--heads 8 --encoder-layers 3 --feed-forward-dim 512 --normalization batch)
      ;;
    ptrnet)
      trainer="experiments/train/train_ptrnet_experiment.py"
      prefix="ptrnet_tsp50"
      model_args=(--encoder-layers 1)
      ;;
    gpn)
      trainer="experiments/train/train_gpn_experiment.py"
      prefix="gpn_tsp50"
      model_args=(--encoder-layers 3)
      ;;
    *)
      echo "unsupported family: ${family}" >&2
      return 2
      ;;
  esac

  local output_dir="artifacts/confirmation/${prefix}_${base_mode}_seed${seed}"
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
    --base-mode "${base_mode}" \
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
    --seed "${seed}" \
    --device cuda
}

for seed in "${SEEDS[@]}"; do
  for family in "${FAMILIES[@]}"; do
    for base_mode in "${MODES[@]}"; do
      run_one "${family}" "${base_mode}" "${seed}"
    done
  done
done
