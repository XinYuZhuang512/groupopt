#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_SEED="${TEST_SEED:-20260805}"
TEST_SIZE="${TEST_SIZE:-10000}"
MODES=(adaptive_state_mean adaptive_state_start)

cd "${PROJECT_DIR}"

evaluate_one() {
  local family="$1"
  local mode="$2"
  local run_name="${family}_tsp50_${mode}_seed1234"
  local run_dir="artifacts/ablation/pilot/${run_name}"
  local output_dir="artifacts/independent_test/iid_tsp50_seed${TEST_SEED}/${run_name}"

  if [[ -f "${output_dir}/costs.pt" && -f "${output_dir}/summary.json" ]]; then
    echo "skip completed ${run_name}"
    return
  fi
  if [[ ! -f "${run_dir}/checkpoints/best.pt" ]]; then
    echo "missing best checkpoint for ${run_name}" >&2
    return 1
  fi

  "${PYTHON_BIN}" experiments/evaluate_checkpoint.py \
    --checkpoint "${run_dir}/checkpoints/best.pt" \
    --config "${run_dir}/config.json" \
    --output-dir "${output_dir}" \
    --test-size "${TEST_SIZE}" \
    --test-seed "${TEST_SEED}" \
    --graph-size 50 \
    --batch-size 512 \
    --device cuda
}

for family in am ptrnet; do
  for mode in "${MODES[@]}"; do
    evaluate_one "${family}" "${mode}"
  done
done
