#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-300}"
SEED="${SEED:-1234}"
TEST_SEED="${TEST_SEED:-20260830}"
TEST_SIZE="${TEST_SIZE:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
POMO_SIZE="${POMO_SIZE:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/pomo_interface_pilot}"
FULL_MODE="native_conditional_free"
MODES=(
  native_forest_fixed
  native_free_no_head_summary
  native_free_no_path_state
  native_free_no_last_head
)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/summary/ablations"

for mode in "${MODES[@]}"; do
  run_dir="${OUTPUT_ROOT}/train/pomo_tsp50_${mode}_seed${SEED}"
  checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
  if [[ ! -f "${checkpoint}" ]]; then
    resume_args=()
    if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
      resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
    fi
    "${PYTHON_BIN}" experiments/train_pomo.py \
      --base-mode "${mode}" \
      --training-scheme pomo \
      --output-dir "${run_dir}" \
      "${resume_args[@]}" \
      --graph-size 50 \
      --pomo-size "${POMO_SIZE}" \
      --steps "${STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --validation-size 64 \
      --embedding-dim 128 \
      --heads 8 \
      --qkv-dim 16 \
      --encoder-layers 6 \
      --feed-forward-dim 512 \
      --learning-rate 1e-4 \
      --eval-every 100 \
      --checkpoint-every 100 \
      --seed "${SEED}" \
      --device cuda \
      >"${OUTPUT_ROOT}/logs/${mode}_seed${SEED}.log" 2>&1
  fi

  output_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/pomo_tsp50_${mode}_final${STEPS}_seed${SEED}"
  if [[ ! -f "${output_dir}/summary.json" || ! -f "${output_dir}/costs.pt" ]]; then
    "${PYTHON_BIN}" experiments/evaluate.py \
      --checkpoint "${checkpoint}" \
      --config "${run_dir}/config.json" \
      --output-dir "${output_dir}" \
      --test-size "${TEST_SIZE}" \
      --test-seed "${TEST_SEED}" \
      --graph-size 50 \
      --batch-size 8 \
      --device cuda
  fi

  "${PYTHON_BIN}" experiments/summarize_model_comparison.py \
    --root "${OUTPUT_ROOT}/eval" \
    --output-dir "${OUTPUT_ROOT}/summary/ablations/${mode}_vs_full" \
    --families pomo \
    --seeds "${SEED}" \
    --steps "${STEPS}" \
    --test-seed "${TEST_SEED}" \
    --original-mode "${mode}" \
    --ours-mode "${FULL_MODE}"
done
