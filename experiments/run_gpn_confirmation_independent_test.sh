#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_SEED="${TEST_SEED:-20260805}"
TEST_SIZE="${TEST_SIZE:-10000}"
SEEDS=(2345 3456 4567)

if [[ "$#" -eq 0 ]]; then
  MODES=(fixed adaptive_state)
else
  MODES=("$@")
fi

cd "${PROJECT_DIR}"

evaluate_one() {
  local base_mode="$1"
  local seed="$2"
  local run_name="gpn_tsp50_${base_mode}_seed${seed}"
  local run_dir="artifacts/formal/${run_name}"
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

for base_mode in "${MODES[@]}"; do
  case "${base_mode}" in
    fixed|adaptive_state) ;;
    *)
      echo "unsupported mode: ${base_mode}" >&2
      exit 2
      ;;
  esac

  for seed in "${SEEDS[@]}"; do
    evaluate_one "${base_mode}" "${seed}"
  done
done
