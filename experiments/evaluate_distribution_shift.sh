#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_SEED="${TEST_SEED:-20260813}"
TEST_SIZE="${TEST_SIZE:-10000}"
STEPS="${STEPS:-2000}"
TRAIN_SEEDS=(${TRAIN_SEEDS:-1234 4321})
FAMILIES=(${FAMILIES:-am ptrnet gpn})
MODES=(native_conditional_fixed native_conditional_free)
DISTRIBUTIONS=(
  uniform
  clustered
  clustered_strong
  corner_biased
  narrow_strip
  clustered_outliers
)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

for distribution in "${DISTRIBUTIONS[@]}"; do
  for family in "${FAMILIES[@]}"; do
    case "${family}" in
      am) batch_size="${AM_BATCH_SIZE:-512}" ;;
      ptrnet|gpn) batch_size="${POINTER_BATCH_SIZE:-128}" ;;
      *) echo "unsupported family: ${family}" >&2; exit 2 ;;
    esac
    for seed in "${TRAIN_SEEDS[@]}"; do
      for mode in "${MODES[@]}"; do
        run_dir="artifacts/native_conditional/${family}_tsp50_${mode}_seed${seed}"
        checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
        output_dir="artifacts/distribution_shift_eval/tsp50_seed${TEST_SEED}/${distribution}/${family}_${mode}_final${STEPS}_seed${seed}"
        if [[ -f "${output_dir}/summary.json" && -f "${output_dir}/costs.pt" ]]; then
          echo "skip ${distribution} ${family} ${mode} seed=${seed}"
          continue
        fi
        "${PYTHON_BIN}" experiments/evaluate.py \
          --checkpoint "${checkpoint}" \
          --config "${run_dir}/config.json" \
          --output-dir "${output_dir}" \
          --test-size "${TEST_SIZE}" \
          --test-seed "${TEST_SEED}" \
          --distribution "${distribution}" \
          --graph-size 50 \
          --batch-size "${batch_size}" \
          --device cuda
      done
    done
  done
done
