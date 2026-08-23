#!/usr/bin/env bash

# 官方 AM 原生性通过后使用的短程 IID 筛选实验。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_ROOT="${GROUPOPT_OFFICIAL_AM_ROOT:?必须设置 GROUPOPT_OFFICIAL_AM_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-300}"
SEED="${SEED:-1234}"
TEST_SEED="${TEST_SEED:-20260823}"
TEST_SIZE="${TEST_SIZE:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VALIDATION_SIZE="${VALIDATION_SIZE:-256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/iclr_official_am_pilot}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

for mode in official_original native_conditional_free; do
  run_dir="${OUTPUT_ROOT}/train/${mode}_seed${SEED}"
  final_checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
  resume_args=()
  if [[ ! -f "${final_checkpoint}" ]]; then
    if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
      resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
    fi
    "${PYTHON_BIN}" experiments/train_official_am.py \
      --base-mode "${mode}" \
      --output-dir "${run_dir}" \
      "${resume_args[@]}" \
      --graph-size 50 \
      --steps "${STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --validation-size "${VALIDATION_SIZE}" \
      --embedding-dim 128 \
      --heads 8 \
      --encoder-layers 3 \
      --normalization batch \
      --learning-rate 1e-4 \
      --eval-every 50 \
      --checkpoint-every 100 \
      --seed "${SEED}" \
      --device cuda
  fi

  evaluation_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/${mode}_seed${SEED}"
  if [[ ! -f "${evaluation_dir}/summary.json" ]]; then
    "${PYTHON_BIN}" experiments/evaluate.py \
      --checkpoint "${final_checkpoint}" \
      --config "${run_dir}/config.json" \
      --output-dir "${evaluation_dir}" \
      --test-size "${TEST_SIZE}" \
      --test-seed "${TEST_SEED}" \
      --graph-size 50 \
      --batch-size "${BATCH_SIZE}" \
      --device cuda
  fi
done

