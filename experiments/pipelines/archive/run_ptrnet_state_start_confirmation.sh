#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_SEED="${TEST_SEED:-20260805}"
TEST_SIZE="${TEST_SIZE:-10000}"
SEEDS=(2345 3456 4567)

cd "${PROJECT_DIR}"

train_and_evaluate() {
  local seed="$1"
  local run_name="ptrnet_tsp50_adaptive_state_start_seed${seed}"
  local output_dir="artifacts/ablation/confirm/${run_name}"
  local test_dir="artifacts/independent_test/iid_tsp50_seed${TEST_SEED}/${run_name}"
  local latest_checkpoint="${output_dir}/checkpoints/latest.pt"
  local interrupted_checkpoint="${output_dir}/checkpoints/interrupted.pt"
  local resume_args=()

  if [[ -f "${interrupted_checkpoint}" ]] && \
     { [[ ! -f "${latest_checkpoint}" ]] || [[ "${interrupted_checkpoint}" -nt "${latest_checkpoint}" ]]; }; then
    resume_args=(--resume "${interrupted_checkpoint}")
  elif [[ -f "${latest_checkpoint}" ]]; then
    resume_args=(--resume "${latest_checkpoint}")
  fi

  "${PYTHON_BIN}" experiments/train/train_ptrnet_experiment.py \
    --base-mode adaptive_state_start \
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps 10000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    --heads 8 \
    --encoder-layers 1 \
    --feed-forward-dim 512 \
    --normalization batch \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed "${seed}" \
    --device cuda

  if [[ -f "${test_dir}/costs.pt" && -f "${test_dir}/summary.json" ]]; then
    echo "skip completed independent test ${run_name}"
    return
  fi

  "${PYTHON_BIN}" experiments/evaluation/evaluate_checkpoint.py \
    --checkpoint "${output_dir}/checkpoints/best.pt" \
    --config "${output_dir}/config.json" \
    --output-dir "${test_dir}" \
    --test-size "${TEST_SIZE}" \
    --test-seed "${TEST_SEED}" \
    --graph-size 50 \
    --batch-size 512 \
    --device cuda
}

for seed in "${SEEDS[@]}"; do
  train_and_evaluate "${seed}"
done
