#!/usr/bin/env bash

# 在同一官方 AM 宿主、训练预算和独立测试集上完成 GroupOpt 信息消融。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OFFICIAL_ROOT="${GROUPOPT_OFFICIAL_AM_ROOT:?必须设置 GROUPOPT_OFFICIAL_AM_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/official_amstyle_ablation}"
STEPS="${STEPS:-600}"
SEEDS="${SEEDS:-1234 4321}"
TEST_SEED="${TEST_SEED:-20260901}"
TEST_SIZE="${TEST_SIZE:-5000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
VALIDATION_SIZE="${VALIDATION_SIZE:-256}"
FULL_MODE="official_amstyle_free"
MODES=(
  official_original
  official_amstyle_free
  official_amstyle_forest_fixed
  official_amstyle_free_no_head_summary
  official_amstyle_free_no_path_state
  official_amstyle_free_no_last_head
)
ABLATIONS=(
  official_original
  official_amstyle_forest_fixed
  official_amstyle_free_no_head_summary
  official_amstyle_free_no_path_state
  official_amstyle_free_no_last_head
)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/summary"

"${PYTHON_BIN}" tmp/validate_official_am_ablations.py \
  --official-am-root "${OFFICIAL_ROOT}" \
  --output "${OUTPUT_ROOT}/summary/implementation_validation.json" \
  --device cuda

for seed in ${SEEDS}; do
  for mode in "${MODES[@]}"; do
    run_dir="${OUTPUT_ROOT}/train/official_am_tsp50_${mode}_seed${seed}"
    checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
    if [[ ! -f "${checkpoint}" ]]; then
      resume_args=()
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
        --eval-every 100 \
        --checkpoint-every 100 \
        --seed "${seed}" \
        --device cuda \
        >"${OUTPUT_ROOT}/logs/${mode}_seed${seed}.log" 2>&1
    fi

    evaluation_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/official_am_tsp50_${mode}_final${STEPS}_seed${seed}"
    if [[ ! -f "${evaluation_dir}/summary.json" || ! -f "${evaluation_dir}/costs.pt" ]]; then
      "${PYTHON_BIN}" experiments/evaluate.py \
        --checkpoint "${checkpoint}" \
        --config "${run_dir}/config.json" \
        --output-dir "${evaluation_dir}" \
        --test-size "${TEST_SIZE}" \
        --test-seed "${TEST_SEED}" \
        --graph-size 50 \
        --batch-size 64 \
        --device cuda
    fi
  done
done

for ablation in "${ABLATIONS[@]}"; do
  "${PYTHON_BIN}" experiments/summarize_model_comparison.py \
    --root "${OUTPUT_ROOT}/eval" \
    --output-dir "${OUTPUT_ROOT}/summary/${ablation}_vs_full" \
    --families official_am \
    --seeds ${SEEDS} \
    --steps "${STEPS}" \
    --test-seed "${TEST_SEED}" \
    --original-mode "${ablation}" \
    --ours-mode "${FULL_MODE}"
done
