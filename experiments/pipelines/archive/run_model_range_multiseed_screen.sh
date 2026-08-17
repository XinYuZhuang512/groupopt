#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS=(transformer_ln gat pointerformer gru)
MODES=(joint_fixed joint_free)
SEEDS=(2345 3456 4567)

cd "${PROJECT_DIR}"

model_settings() {
  local model="$1"
  case "${model}" in
    transformer_ln)
      TRAINER="experiments/train/train_transformer_ln_experiment.py"
      LAYERS=3
      ;;
    gat)
      TRAINER="experiments/train/train_gat_experiment.py"
      LAYERS=3
      ;;
    pointerformer)
      TRAINER="experiments/train/train_pointerformer_experiment.py"
      LAYERS=3
      ;;
    gru)
      TRAINER="experiments/train/train_gru_experiment.py"
      LAYERS=1
      ;;
    *)
      echo "unsupported model: ${model}" >&2
      return 2
      ;;
  esac
}

run_one() {
  local model="$1"
  local mode="$2"
  local seed="$3"
  model_settings "${model}"
  local output_dir="artifacts/model_range/screen/${model}_tsp50_${mode}_seed${seed}"
  if [[ -f "${output_dir}/checkpoints/step-002000.pt" ]]; then
    echo "skip completed ${model} ${mode} seed ${seed}"
    return
  fi
  local resume_args=()
  if [[ -f "${output_dir}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${output_dir}/checkpoints/latest.pt")
  fi
  mkdir -p "${output_dir}"
  "${PYTHON_BIN}" "${TRAINER}" \
    --base-mode "${mode}" \
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps 2000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    --heads 8 \
    --encoder-layers "${LAYERS}" \
    --feed-forward-dim 512 \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed "${seed}" \
    --device cuda \
    > "${output_dir}/train.log" 2>&1
}

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    pids=()
    for mode in "${MODES[@]}"; do
      run_one "${model}" "${mode}" "${seed}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  done
done
