#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-300}"
SEEDS_TEXT="${SEEDS:-1234}"
TEST_SEED="${TEST_SEED:-20260830}"
TEST_SIZE="${TEST_SIZE:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
POMO_SIZE="${POMO_SIZE:-8}"
VALIDATION_SIZE="${VALIDATION_SIZE:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/pomo_pilot}"
read -r -a SEED_LIST <<< "${SEEDS_TEXT}"
MODES=(native_original native_conditional_free)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/logs"

train_one() {
  local seed="$1"
  local mode="$2"
  local run_dir="${OUTPUT_ROOT}/train/pomo_tsp50_${mode}_seed${seed}"
  local checkpoint
  checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
  if [[ -f "${checkpoint}" ]]; then
    echo "skip completed POMO ${mode} seed=${seed}"
    return
  fi
  local resume_args=()
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
    --validation-size "${VALIDATION_SIZE}" \
    --embedding-dim 128 \
    --heads 8 \
    --qkv-dim 16 \
    --encoder-layers 6 \
    --feed-forward-dim 512 \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 100 \
    --seed "${seed}" \
    --device cuda
}

for seed in "${SEED_LIST[@]}"; do
  pids=()
  for mode in "${MODES[@]}"; do
    train_one "${seed}" "${mode}" \
      >"${OUTPUT_ROOT}/logs/${mode}_seed${seed}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

for seed in "${SEED_LIST[@]}"; do
  for mode in "${MODES[@]}"; do
    run_dir="${OUTPUT_ROOT}/train/pomo_tsp50_${mode}_seed${seed}"
    checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
    output_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/pomo_tsp50_${mode}_final${STEPS}_seed${seed}"
    if [[ -f "${output_dir}/summary.json" && -f "${output_dir}/costs.pt" ]]; then
      continue
    fi
    "${PYTHON_BIN}" experiments/evaluate.py \
      --checkpoint "${checkpoint}" \
      --config "${run_dir}/config.json" \
      --output-dir "${output_dir}" \
      --test-size "${TEST_SIZE}" \
      --test-seed "${TEST_SEED}" \
      --graph-size 50 \
      --batch-size 8 \
      --device cuda
  done
done

"${PYTHON_BIN}" experiments/summarize_model_comparison.py \
  --root "${OUTPUT_ROOT}/eval" \
  --output-dir "${OUTPUT_ROOT}/summary" \
  --families pomo \
  --seeds "${SEED_LIST[@]}" \
  --steps "${STEPS}" \
  --test-seed "${TEST_SEED}"
