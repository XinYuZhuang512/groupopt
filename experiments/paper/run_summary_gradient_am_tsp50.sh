#!/usr/bin/env bash

# AM TSP50 head-summary 梯度路由消融；Detached 与 No Summary 复用正式结果。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/paper_summary_gradient_am_tsp50_v1}"
MAIN_ROOT="${MAIN_ROOT:-${PROJECT_ROOT}/artifacts/paper_final_tsp50_v1}"
ABLATION_ROOT="${ABLATION_ROOT:-${PROJECT_ROOT}/artifacts/paper_ablation_am_tsp50_v1}"
TEST_SEED="${TEST_SEED:-20260904}"
TEST_SIZE="${TEST_SIZE:-10000}"
SEEDS=(1234 4321 2468)
MODE="native_free_end_to_end_summary"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/summary"
cp "${SCRIPT_DIR}/protocol_summary_gradient_am_tsp50_v1.json" \
  "${OUTPUT_ROOT}/protocol_snapshot.json"

"${PYTHON_BIN}" experiments/paper/validate_summary_gradient_am.py \
  --output "${OUTPUT_ROOT}/summary/gradient_route_validation.json" \
  --device cuda \
  >"${OUTPUT_ROOT}/logs/validate_gradient_route.log" 2>&1

train_one() {
  local seed="$1"
  local run_dir="${OUTPUT_ROOT}/train/am_tsp50_${MODE}_seed${seed}"
  local checkpoint="${run_dir}/checkpoints/step-010000.pt"
  [[ ! -f "${checkpoint}" ]] || return 0
  local resume_args=()
  if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
  elif [[ -f "${run_dir}/checkpoints/best.pt" ]]; then
    # 启动阶段 OOM 时可能只有包含完整 RNG 状态的 step-0 checkpoint。
    resume_args=(--resume "${run_dir}/checkpoints/best.pt")
  fi
  "${PYTHON_BIN}" experiments/train.py \
    --problem tsp \
    --base-mode "${MODE}" \
    --output-dir "${run_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps 10000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    --heads 8 \
    --encoder-layers 3 \
    --feed-forward-dim 512 \
    --normalization batch \
    --learning-rate 1e-4 \
    --max-grad-norm 1.0 \
    --eval-every 100 \
    --checkpoint-every 1000 \
    --seed "${seed}" \
    --device cuda
}

# End-to-End Summary 保留整个 O(n^2) proposal 的反向图，单进程峰值约
# 15--16GB；32GB GPU 上串行运行以保持 batch size 和科学配置不变。
for seed in "${SEEDS[@]}"; do
  if ! train_one "${seed}" \
    >"${OUTPUT_ROOT}/logs/train_${MODE}_seed${seed}.log" 2>&1; then
    echo "End-to-End Summary 训练失败；保留日志和 checkpoint。" >&2
    exit 10
  fi
done

for seed in "${SEEDS[@]}"; do
  run_dir="${OUTPUT_ROOT}/train/am_tsp50_${MODE}_seed${seed}"
  checkpoint="${run_dir}/checkpoints/step-010000.pt"
  eval_dir="${OUTPUT_ROOT}/eval/iid_tsp50_seed${TEST_SEED}/am/${MODE}/seed${seed}"
  if [[ ! -f "${eval_dir}/summary.json" || ! -f "${eval_dir}/costs.pt" ]]; then
    "${PYTHON_BIN}" experiments/evaluate.py \
      --checkpoint "${checkpoint}" \
      --config "${run_dir}/config.json" \
      --base-mode-override "${MODE}" \
      --output-dir "${eval_dir}" \
      --test-size "${TEST_SIZE}" \
      --test-seed "${TEST_SEED}" \
      --graph-size 50 \
      --batch-size 64 \
      --device cuda \
      >"${OUTPUT_ROOT}/logs/eval_${MODE}_seed${seed}.log" 2>&1
  fi
done

"${PYTHON_BIN}" experiments/paper/summarize_summary_gradient_am.py \
  --new-root "${OUTPUT_ROOT}" \
  --main-root "${MAIN_ROOT}" \
  --ablation-root "${ABLATION_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/summary" \
  --test-seed "${TEST_SEED}" \
  >"${OUTPUT_ROOT}/logs/summarize.log" 2>&1

checkpoints=$(find "${OUTPUT_ROOT}/train" -path '*/checkpoints/step-010000.pt' | wc -l)
summaries=$(find "${OUTPUT_ROOT}/eval" -name summary.json | wc -l)
costs=$(find "${OUTPUT_ROOT}/eval" -name costs.pt | wc -l)
if [[ "${checkpoints}" -ne 3 || "${summaries}" -ne 3 || "${costs}" -ne 3 ]]; then
  echo "梯度路由消融产物不完整: checkpoints=${checkpoints}, summaries=${summaries}, costs=${costs}" >&2
  exit 20
fi
test -f "${OUTPUT_ROOT}/summary/gradient_route_validation.json"
test -f "${OUTPUT_ROOT}/summary/summary_gradient_ablation.json"
test -f "${OUTPUT_ROOT}/summary/mode_results.csv"
test -f "${OUTPUT_ROOT}/summary/paired_comparisons.csv"
