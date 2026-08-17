#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_SEED="${TEST_SEED:-20260809}"
TEST_SIZE="${TEST_SIZE:-10000}"
STEPS="${STEPS:-10000}"
SEEDS=(${SEEDS:-2025 3407})
MODELS=(geometric moe_transformer)
MODES=(joint_fixed joint_free)

cd "${PROJECT_DIR}"

trainer_for() {
  case "$1" in
    geometric) TRAINER="experiments/train/train_geometric_experiment.py" ;;
    moe_transformer) TRAINER="experiments/train/train_moe_transformer_experiment.py" ;;
    *) echo "unsupported model: $1" >&2; return 2 ;;
  esac
}

train_one() {
  local model="$1"
  local mode="$2"
  local seed="$3"
  trainer_for "${model}"
  local run_dir="artifacts/model_range/modern/${model}_tsp50_${mode}_seed${seed}"
  local final_checkpoint
  final_checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "skip completed training ${model} ${mode} seed=${seed}"
    return
  fi
  local resume_args=()
  if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
  fi
  mkdir -p "${run_dir}"
  "${PYTHON_BIN}" "${TRAINER}" \
    --base-mode "${mode}" \
    --output-dir "${run_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps "${STEPS}" \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    --heads 8 \
    --encoder-layers 3 \
    --feed-forward-dim 512 \
    --learning-rate 1e-4 \
    --eval-every 250 \
    --checkpoint-every 500 \
    --seed "${seed}" \
    --device cuda \
    > "${run_dir}/confirmation.log" 2>&1
}

# Pair fixed/free on the same seed and run the pair concurrently. Models and
# seeds remain sequential to avoid oversubscribing one GPU with four jobs.
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    pids=()
    for mode in "${MODES[@]}"; do
      train_one "${model}" "${mode}" "${seed}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  done
done
for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
      run_dir="artifacts/model_range/modern/${model}_tsp50_${mode}_seed${seed}"
      checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
      output_dir="artifacts/model_range/modern_confirm/iid_tsp50_seed${TEST_SEED}/${model}_tsp50_${mode}_final${STEPS}_seed${seed}"
      if [[ -f "${output_dir}/summary.json" && -f "${output_dir}/costs.pt" ]]; then
        echo "skip completed evaluation ${model} ${mode} seed=${seed}"
        continue
      fi
      "${PYTHON_BIN}" experiments/evaluation/evaluate_checkpoint.py \
        --checkpoint "${checkpoint}" \
        --config "${run_dir}/config.json" \
        --output-dir "${output_dir}" \
        --test-size "${TEST_SIZE}" \
        --test-seed "${TEST_SEED}" \
        --graph-size 50 \
        --batch-size 512 \
        --device cuda
    done
  done
done
