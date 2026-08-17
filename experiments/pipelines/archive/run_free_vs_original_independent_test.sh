#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_SEED="${TEST_SEED:-20260805}"
TEST_SIZE="${TEST_SIZE:-10000}"
FAMILIES=(tsp50 ptrnet_tsp50 gpn_tsp50)
SEEDS=(2345 3456 4567)

cd "${PROJECT_DIR}"

for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_name="${family}_joint_free_seed${seed}"
    run_dir="artifacts/confirmation/${run_name}"
    output_dir="artifacts/independent_test/iid_tsp50_seed${TEST_SEED}/${family}_joint_free_10k_seed${seed}"
    if [[ -f "${output_dir}/costs.pt" && -f "${output_dir}/summary.json" ]]; then
      echo "skip completed ${run_name}"
      continue
    fi
    if [[ ! -f "${run_dir}/checkpoints/best.pt" ]]; then
      echo "missing best checkpoint for ${run_name}" >&2
      exit 1
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
