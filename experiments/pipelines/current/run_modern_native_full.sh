#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/groupopt-data}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-2000}"
TEST_SIZE="${TEST_SIZE:-10000}"
TEST_SEED="${TEST_SEED:-20260813}"
TRAIN_SEEDS=(1234 4321)
MODELS=(reversible_transformer geometric_transformer sparse_moe_transformer)
MODES=(native_conditional_fixed native_conditional_free)
DISTRIBUTIONS=(
  uniform
  clustered
  clustered_strong
  corner_biased
  narrow_strip
  clustered_outliers
)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

while [[ ! -f "${DATA_DIR}/distribution_shift_eval.exit" ]]; do
  sleep 60
done
if [[ "$(tr -d '[:space:]' < "${DATA_DIR}/distribution_shift_eval.exit")" != "0" ]]; then
  echo "prerequisite distribution_shift_eval failed" >&2
  exit 3
fi

configure_model() {
  case "$1" in
    reversible_transformer)
      TRAINER="experiments/train/train_reversible_transformer_experiment.py"
      TRAIN_BATCH="${REVERSIBLE_BATCH_SIZE:-256}"
      EVAL_BATCH="${REVERSIBLE_EVAL_BATCH_SIZE:-256}"
      ;;
    geometric_transformer)
      TRAINER="experiments/train/train_geometric_transformer_experiment.py"
      TRAIN_BATCH="${GEOMETRIC_BATCH_SIZE:-64}"
      EVAL_BATCH="${GEOMETRIC_EVAL_BATCH_SIZE:-64}"
      ;;
    sparse_moe_transformer)
      TRAINER="experiments/train/train_sparse_moe_transformer_experiment.py"
      TRAIN_BATCH="${MOE_BATCH_SIZE:-64}"
      EVAL_BATCH="${MOE_EVAL_BATCH_SIZE:-64}"
      ;;
    *) echo "unsupported model: $1" >&2; return 2 ;;
  esac
}

for model in "${MODELS[@]}"; do
  configure_model "${model}"
  for seed in "${TRAIN_SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
      run_dir="artifacts/modern_native/${model}_tsp50_${mode}_seed${seed}"
      final_checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
      if [[ -f "${final_checkpoint}" ]]; then
        echo "skip completed training ${model} ${mode} seed=${seed}"
        continue
      fi
      resume_args=()
      if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
        resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
      fi
      "${PYTHON_BIN}" "${TRAINER}" \
        --base-mode "${mode}" \
        --output-dir "${run_dir}" \
        "${resume_args[@]}" \
        --graph-size 50 \
        --steps "${STEPS}" \
        --batch-size "${TRAIN_BATCH}" \
        --validation-size 256 \
        --embedding-dim 128 \
        --heads 8 \
        --encoder-layers 3 \
        --feed-forward-dim 512 \
        --learning-rate 1e-4 \
        --eval-every 100 \
        --checkpoint-every 250 \
        --seed "${seed}" \
        --device cuda
    done
  done
done

eval_root="artifacts/distribution_shift_eval/tsp50_seed${TEST_SEED}"
for distribution in "${DISTRIBUTIONS[@]}"; do
  for model in "${MODELS[@]}"; do
    configure_model "${model}"
    for seed in "${TRAIN_SEEDS[@]}"; do
      for mode in "${MODES[@]}"; do
        run_dir="artifacts/modern_native/${model}_tsp50_${mode}_seed${seed}"
        checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
        output_dir="${eval_root}/${distribution}/${model}_${mode}_final${STEPS}_seed${seed}"
        if [[ -f "${output_dir}/summary.json" && -f "${output_dir}/costs.pt" ]]; then
          echo "skip completed evaluation ${distribution} ${model} ${mode} seed=${seed}"
          continue
        fi
        "${PYTHON_BIN}" experiments/evaluation/evaluate_checkpoint.py \
          --checkpoint "${checkpoint}" \
          --config "${run_dir}/config.json" \
          --output-dir "${output_dir}" \
          --test-size "${TEST_SIZE}" \
          --test-seed "${TEST_SEED}" \
          --distribution "${distribution}" \
          --graph-size 50 \
          --batch-size "${EVAL_BATCH}" \
          --device cuda
      done
    done
  done
done

"${PYTHON_BIN}" experiments/analysis/summarize_and_plot_final.py \
  --eval-root "${eval_root}" \
  --output-dir artifacts/final_report \
  --steps "${STEPS}" \
  --train-seeds "${TRAIN_SEEDS[@]}" \
  --distribution-seed "${TEST_SEED}"
