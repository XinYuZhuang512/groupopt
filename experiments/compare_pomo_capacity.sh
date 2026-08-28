#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-1000}"
TEST_SEED="${TEST_SEED:-20260830}"
TEST_SIZE="${TEST_SIZE:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
POMO_SIZE="${POMO_SIZE:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/pomo_interface_pilot}"
read -r -a SEED_LIST <<< "${SEEDS:-1234 4321}"
MODE="native_capacity_single_chain"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/logs"

for seed in "${SEED_LIST[@]}"; do
  run_dir="${OUTPUT_ROOT}/train/pomo_tsp50_${MODE}_seed${seed}"
  checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
  if [[ ! -f "${checkpoint}" ]]; then
    resume_args=()
    if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
      resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
    fi
    "${PYTHON_BIN}" experiments/train_pomo.py \
      --base-mode "${MODE}" \
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
      --seed "${seed}" \
      --device cuda \
      >"${OUTPUT_ROOT}/logs/${MODE}_seed${seed}.log" 2>&1
  fi

  output_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/pomo_tsp50_${MODE}_final${STEPS}_seed${seed}"
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
done

"${PYTHON_BIN}" experiments/summarize_model_comparison.py \
  --root "${OUTPUT_ROOT}/eval" \
  --output-dir "${OUTPUT_ROOT}/summary/capacity_vs_free" \
  --families pomo \
  --seeds "${SEED_LIST[@]}" \
  --steps "${STEPS}" \
  --test-seed "${TEST_SEED}" \
  --original-mode "${MODE}" \
  --ours-mode native_conditional_free
