#!/usr/bin/env bash

# 复用两个已验证种子，只补第三种子，并统一在同一独立测试集上重评论文主表。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/paper_main_v1}"
LEGACY_ROOT="${LEGACY_ROOT:?必须设置前两种子 AM/PtrNet/GPN 产物目录}"
POMO_ROOT="${POMO_ROOT:?必须设置前两种子 POMO 产物目录}"
NEW_SEED="${NEW_SEED:-2468}"
TEST_SEED="${TEST_SEED:-20260902}"
TEST_SIZE="${TEST_SIZE:-10000}"
SEEDS=(1234 4321 "${NEW_SEED}")
FAMILIES=(am ptrnet gpn pomo)
MODES=(native_original native_conditional_free)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/summary"

"${PYTHON_BIN}" experiments/validate_native_original.py \
  --output "${OUTPUT_ROOT}/summary/native_original_validation.json" \
  --device cuda

configure_family() {
  case "$1" in
    am)
      TRAINER="experiments/train.py"
      STEPS=2000
      BATCH_SIZE=512
      VALIDATION_SIZE=1024
      CHECKPOINT_EVERY=250
      MODEL_ARGS=(--heads 8 --encoder-layers 3 --feed-forward-dim 512 --normalization batch)
      ;;
    ptrnet)
      TRAINER="experiments/train_ptrnet.py"
      STEPS=2000
      BATCH_SIZE=128
      VALIDATION_SIZE=1024
      CHECKPOINT_EVERY=250
      MODEL_ARGS=(--encoder-layers 1)
      ;;
    gpn)
      TRAINER="experiments/train_gpn.py"
      STEPS=2000
      BATCH_SIZE=128
      VALIDATION_SIZE=1024
      CHECKPOINT_EVERY=250
      MODEL_ARGS=(--encoder-layers 3)
      ;;
    pomo)
      TRAINER="experiments/train_pomo.py"
      STEPS=1000
      BATCH_SIZE=64
      VALIDATION_SIZE=64
      CHECKPOINT_EVERY=100
      MODEL_ARGS=(--training-scheme pomo --pomo-size 8 --qkv-dim 16 --heads 8 --encoder-layers 6 --feed-forward-dim 512)
      ;;
    *)
      echo "不支持的宿主：$1" >&2
      return 2
      ;;
  esac
}

train_one() {
  local family="$1"
  local mode="$2"
  configure_family "${family}"
  local run_dir="${OUTPUT_ROOT}/train/${family}_tsp50_${mode}_seed${NEW_SEED}"
  local checkpoint
  checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
  if [[ -f "${checkpoint}" ]]; then
    return 0
  fi
  local resume_args=()
  if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
  fi
  "${PYTHON_BIN}" "${TRAINER}" \
    --base-mode "${mode}" \
    --output-dir "${run_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --validation-size "${VALIDATION_SIZE}" \
    --embedding-dim 128 \
    "${MODEL_ARGS[@]}" \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --seed "${NEW_SEED}" \
    --device cuda
}

# 同一宿主的 Original 显存很小，可与 Full 并行；宿主之间串行以避免 OOM。
for family in "${FAMILIES[@]}"; do
  pids=()
  for mode in "${MODES[@]}"; do
    train_one "${family}" "${mode}" \
      >"${OUTPUT_ROOT}/logs/train_${family}_${mode}_seed${NEW_SEED}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "${family} 第三种子训练失败，保留日志和 checkpoint。" >&2
    exit 10
  fi
done

resolve_run() {
  local family="$1"
  local mode="$2"
  local seed="$3"
  configure_family "${family}"
  EVAL_BATCH=64
  if [[ "${family}" == "pomo" ]]; then
    EVAL_BATCH=8
  fi
  if [[ "${seed}" == "${NEW_SEED}" ]]; then
    RUN_DIR="${OUTPUT_ROOT}/train/${family}_tsp50_${mode}_seed${seed}"
    return
  fi
  if [[ "${family}" == "pomo" ]]; then
    RUN_DIR="${POMO_ROOT}/pomo_tsp50_${mode}_seed${seed}"
    return
  fi
  if [[ "${mode}" == "native_original" ]]; then
    RUN_DIR="${LEGACY_ROOT}/${family}_tsp50_native_conditional_fixed_seed${seed}"
  else
    RUN_DIR="${LEGACY_ROOT}/${family}_tsp50_${mode}_seed${seed}"
  fi
}

for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
      resolve_run "${family}" "${mode}" "${seed}"
      checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${RUN_DIR}" "${STEPS}")"
      evaluation_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/${family}/${mode}/seed${seed}"
      if [[ ! -f "${checkpoint}" || ! -f "${RUN_DIR}/config.json" ]]; then
        echo "缺少 checkpoint 或配置：${RUN_DIR}" >&2
        exit 20
      fi
      if [[ -f "${evaluation_dir}/summary.json" && -f "${evaluation_dir}/costs.pt" ]]; then
        continue
      fi
      "${PYTHON_BIN}" experiments/evaluate.py \
        --checkpoint "${checkpoint}" \
        --config "${RUN_DIR}/config.json" \
        --base-mode-override "${mode}" \
        --output-dir "${evaluation_dir}" \
        --test-size "${TEST_SIZE}" \
        --test-seed "${TEST_SEED}" \
        --graph-size 50 \
        --batch-size "${EVAL_BATCH}" \
        --device cuda \
        >"${OUTPUT_ROOT}/logs/eval_${family}_${mode}_seed${seed}.log" 2>&1
    done
  done
done

"${PYTHON_BIN}" experiments/paper/summarize_main_tsp50.py \
  --root "${OUTPUT_ROOT}/eval" \
  --output-dir "${OUTPUT_ROOT}/summary" \
  --families "${FAMILIES[@]}" \
  --seeds "${SEEDS[@]}" \
  --test-seed "${TEST_SEED}"
