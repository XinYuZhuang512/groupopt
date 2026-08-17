#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODES=(joint_fixed joint_free)

cd "${PROJECT_DIR}"

run_one() {
  local mode="$1"
  local output_dir="artifacts/encoder_controls/pilot/pointerformer_tsp50_${mode}_seed1234"
  local resume_args=()
  mkdir -p "${output_dir}"
  if [[ -f "${output_dir}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${output_dir}/checkpoints/latest.pt")
  fi
  "${PYTHON_BIN}" experiments/train/train_pointerformer_experiment.py \
    --base-mode "${mode}" \
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps 2000 \
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
    > "${output_dir}/train.log" 2>&1
}

pids=()
for mode in "${MODES[@]}"; do
  run_one "${mode}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more Pointerformer runs failed" >&2
  exit 1
fi
