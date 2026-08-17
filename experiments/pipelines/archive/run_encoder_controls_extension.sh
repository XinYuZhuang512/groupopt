#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS=(transformer_ln gat)
MODES=(joint_fixed joint_free)

cd "${PROJECT_DIR}"

run_one() {
  local model="$1"
  local mode="$2"
  local trainer
  case "${model}" in
    transformer_ln)
      trainer="experiments/train/train_transformer_ln_experiment.py"
      ;;
    gat)
      trainer="experiments/train/train_gat_experiment.py"
      ;;
    *)
      echo "unsupported encoder control: ${model}" >&2
      return 2
      ;;
  esac

  local output_dir="artifacts/encoder_controls/pilot/${model}_tsp50_${mode}_seed1234"
  local latest_checkpoint="${output_dir}/checkpoints/latest.pt"
  local interrupted_checkpoint="${output_dir}/checkpoints/interrupted.pt"
  local checkpoint="${latest_checkpoint}"
  if [[ -f "${interrupted_checkpoint}" ]] && \
     { [[ ! -f "${latest_checkpoint}" ]] || [[ "${interrupted_checkpoint}" -nt "${latest_checkpoint}" ]]; }; then
    checkpoint="${interrupted_checkpoint}"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint for ${model} ${mode}: ${checkpoint}" >&2
    return 1
  fi

  "${PYTHON_BIN}" "${trainer}" \
    --base-mode "${mode}" \
    --output-dir "${output_dir}" \
    --resume "${checkpoint}" \
    --graph-size 50 \
    --steps 10000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    --heads 8 \
    --encoder-layers 3 \
    --feed-forward-dim 512 \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed 1234 \
    --device cuda \
    > "${output_dir}/extension.log" 2>&1
}

for model in "${MODELS[@]}"; do
  pids=()
  for mode in "${MODES[@]}"; do
    run_one "${model}" "${mode}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "one or more ${model} extension runs failed" >&2
    exit 1
  fi
done
