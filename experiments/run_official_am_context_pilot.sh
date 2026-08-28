#!/usr/bin/env bash

# 快速定位官方 AM 的 first-node context 在 GroupOpt 中是否发生语义错配。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OFFICIAL_ROOT="${GROUPOPT_OFFICIAL_AM_ROOT:?必须设置 GROUPOPT_OFFICIAL_AM_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/iclr_official_am_pilot}"
STEPS="${STEPS:-600}"
EVAL_STEPS="${EVAL_STEPS:-300 600}"
SEED="${SEED:-1234}"
TEST_SEED="${TEST_SEED:-20260823}"
TEST_SIZE="${TEST_SIZE:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VALIDATION_SIZE="${VALIDATION_SIZE:-256}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" experiments/validate_official_am.py \
  --official-am-root "${OFFICIAL_ROOT}" \
  --output "${OUTPUT_ROOT}/summary/context_adapter_validation.json" \
  --device cuda

for mode in official_groupopt_global_anchor official_groupopt_graph_tail; do
  run_dir="${OUTPUT_ROOT}/train/${mode}_seed${SEED}"
  final_checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
  resume_args=()
  if [[ ! -f "${final_checkpoint}" ]]; then
    if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
      resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
    fi
    mkdir -p "${run_dir}"
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
      --device cuda \
      >"${run_dir}/train.log" 2>&1
  fi

  for step in ${EVAL_STEPS}; do
    checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${step}")"
    evaluation_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/${mode}_step${step}_seed${SEED}"
    if [[ ! -f "${evaluation_dir}/summary.json" || ! -f "${evaluation_dir}/costs.pt" ]]; then
      "${PYTHON_BIN}" experiments/evaluate.py \
        --checkpoint "${checkpoint}" \
        --config "${run_dir}/config.json" \
        --output-dir "${evaluation_dir}" \
        --test-size "${TEST_SIZE}" \
        --test-seed "${TEST_SEED}" \
        --graph-size 50 \
        --batch-size "${BATCH_SIZE}" \
        --device cuda
    fi
  done
done

for mode in official_original native_conditional_free; do
  for step in ${EVAL_STEPS}; do
    evaluation_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/${mode}_step${step}_seed${SEED}"
    if [[ "${step}" -eq 300 && ! -d "${evaluation_dir}" ]]; then
      evaluation_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/${mode}_seed${SEED}"
    fi
    if [[ ! -f "${evaluation_dir}/summary.json" || ! -f "${evaluation_dir}/costs.pt" ]]; then
      echo "缺少既有 context 对照结果: ${evaluation_dir}" >&2
      exit 8
    fi
  done
done

"${PYTHON_BIN}" experiments/summarize_official_am_bridge.py \
  --study contexts \
  --output-root "${OUTPUT_ROOT}" \
  --test-seed "${TEST_SEED}" \
  --train-seed "${SEED}" \
  --steps ${EVAL_STEPS}
