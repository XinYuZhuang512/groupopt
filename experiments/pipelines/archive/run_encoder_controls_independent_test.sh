#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_SEED="${TEST_SEED:-20260805}"
TEST_SIZE="${TEST_SIZE:-10000}"
MODELS=(transformer_ln gat gru)
MODES=(joint_fixed joint_free)

cd "${PROJECT_DIR}"

for model in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    run_name="${model}_tsp50_${mode}_seed1234"
    run_dir="artifacts/encoder_controls/pilot/${run_name}"
    output_dir="artifacts/encoder_controls/independent_test/iid_tsp50_seed${TEST_SEED}/${run_name}"
    if [[ -f "${output_dir}/costs.pt" && -f "${output_dir}/summary.json" ]]; then
      echo "skip completed ${run_name}"
      continue
    fi
    "${PYTHON_BIN}" experiments/evaluation/evaluate_checkpoint.py \
      --checkpoint "${run_dir}/checkpoints/best.pt" \
      --config "${run_dir}/config.json" \
      --output-dir "${output_dir}" \
      --test-size "${TEST_SIZE}" \
      --test-seed "${TEST_SEED}" \
      --graph-size 50 \
      --batch-size 512 \
      --device cuda
  done
done
