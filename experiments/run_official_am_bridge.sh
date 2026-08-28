#!/usr/bin/env bash

# 复用既有 Original/GroupOpt 结果，并并行补齐两个固定 tail 桥接对照。
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OFFICIAL_ROOT="${GROUPOPT_OFFICIAL_AM_ROOT:?必须设置 GROUPOPT_OFFICIAL_AM_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/iclr_official_am_pilot}"
STEPS="${STEPS:-2400}"
EVAL_STEPS="${EVAL_STEPS:-600 1200 2400}"
SEED="${SEED:-1234}"
TEST_SEED="${TEST_SEED:-20260823}"
TEST_SIZE="${TEST_SIZE:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VALIDATION_SIZE="${VALIDATION_SIZE:-256}"

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" experiments/validate_official_am.py \
  --official-am-root "${OFFICIAL_ROOT}" \
  --output "${OUTPUT_ROOT}/summary/adapter_validation.json" \
  --device cuda

train_mode() {
  local mode="$1"
  local run_dir="${OUTPUT_ROOT}/train/${mode}_seed${SEED}"
  local final_checkpoint
  final_checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
  if [[ -f "${final_checkpoint}" ]]; then
    return 0
  fi
  local resume_args=()
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
}

pids=()
for mode in official_conditional_fixed official_forest_fixed; do
  train_mode "${mode}" &
  pids+=("$!")
done

training_status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    training_status=1
  fi
done
if [[ "${training_status}" -ne 0 ]]; then
  echo "至少一个 fixed 训练失败，保留日志与现场。" >&2
  exit 4
fi

evaluate_mode_step() {
  local mode="$1"
  local step="$2"
  local run_dir="${OUTPUT_ROOT}/train/${mode}_seed${SEED}"
  local checkpoint
  checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${step}")"
  local evaluation_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/${mode}_step${step}_seed${SEED}"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "缺少待评估 checkpoint: ${checkpoint}" >&2
    return 5
  fi
  if [[ -f "${evaluation_dir}/summary.json" && -f "${evaluation_dir}/costs.pt" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" experiments/evaluate.py \
    --checkpoint "${checkpoint}" \
    --config "${run_dir}/config.json" \
    --output-dir "${evaluation_dir}" \
    --test-size "${TEST_SIZE}" \
    --test-seed "${TEST_SEED}" \
    --graph-size 50 \
    --batch-size "${BATCH_SIZE}" \
    --device cuda
}

for step in ${EVAL_STEPS}; do
  pids=()
  for mode in official_conditional_fixed official_forest_fixed; do
    evaluate_mode_step "${mode}" "${step}" &
    pids+=("$!")
  done
  evaluation_status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      evaluation_status=1
    fi
  done
  if [[ "${evaluation_status}" -ne 0 ]]; then
    echo "至少一个 fixed 独立评估失败。" >&2
    exit 6
  fi
done

# Original 与 GroupOpt-Free 直接复用相同配置下已经完成的独立测试。
for mode in official_original native_conditional_free; do
  for step in ${EVAL_STEPS}; do
    evaluation_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/${mode}_step${step}_seed${SEED}"
    if [[ ! -f "${evaluation_dir}/summary.json" || ! -f "${evaluation_dir}/costs.pt" ]]; then
      echo "缺少既有对照结果: ${evaluation_dir}" >&2
      exit 7
    fi
  done
done

"${PYTHON_BIN}" experiments/summarize_official_am_bridge.py \
  --output-root "${OUTPUT_ROOT}" \
  --test-seed "${TEST_SEED}" \
  --train-seed "${SEED}" \
  --steps ${EVAL_STEPS}
