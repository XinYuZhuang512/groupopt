#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FAMILIES=(am ptrnet gpn)
SEEDS=(2345 3456 4567)

cd "${PROJECT_DIR}"

run_one() {
  local family="$1"
  local seed="$2"
  local trainer
  local prefix
  local model_args=()
  case "${family}" in
    am)
      trainer="experiments/train_am_experiment.py"
      prefix="tsp50"
      model_args=(--heads 8 --encoder-layers 3 --feed-forward-dim 512 --normalization batch)
      ;;
    ptrnet)
      trainer="experiments/train_ptrnet_experiment.py"
      prefix="ptrnet_tsp50"
      model_args=(--encoder-layers 1)
      ;;
    gpn)
      trainer="experiments/train_gpn_experiment.py"
      prefix="gpn_tsp50"
      model_args=(--encoder-layers 3)
      ;;
    *)
      echo "unsupported family: ${family}" >&2
      return 2
      ;;
  esac

  local output_dir="artifacts/confirmation/${prefix}_joint_free_seed${seed}"
  local latest_checkpoint="${output_dir}/checkpoints/latest.pt"
  local interrupted_checkpoint="${output_dir}/checkpoints/interrupted.pt"
  local checkpoint="${latest_checkpoint}"
  if [[ -f "${interrupted_checkpoint}" ]] && \
     { [[ ! -f "${latest_checkpoint}" ]] || [[ "${interrupted_checkpoint}" -nt "${latest_checkpoint}" ]]; }; then
    checkpoint="${interrupted_checkpoint}"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing joint_free checkpoint for ${family} seed ${seed}" >&2
    return 1
  fi

  "${PYTHON_BIN}" "${trainer}" \
    --base-mode joint_free \
    --output-dir "${output_dir}" \
    --resume "${checkpoint}" \
    --graph-size 50 \
    --steps 10000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    "${model_args[@]}" \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed "${seed}" \
    --device cuda \
    > "${output_dir}/free_vs_original_extension.log" 2>&1
}

pending_pids=()
pending_labels=()
failed=0

wait_pending() {
  local index
  for index in "${!pending_pids[@]}"; do
    if ! wait "${pending_pids[${index}]}"; then
      echo "failed: ${pending_labels[${index}]}" >&2
      failed=1
    fi
  done
  pending_pids=()
  pending_labels=()
}

for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${family}" "${seed}" &
    pending_pids+=("$!")
    pending_labels+=("${family}-seed${seed}")
    if [[ "${#pending_pids[@]}" -eq 2 ]]; then
      wait_pending
    fi
  done
done
if [[ "${#pending_pids[@]}" -gt 0 ]]; then
  wait_pending
fi
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
