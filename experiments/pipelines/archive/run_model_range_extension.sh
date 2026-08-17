#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS=(transformer_ln gat pointerformer gru)
MODES=(joint_fixed joint_free)
SEEDS=(1234 2345 3456 4567)

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

run_directory() {
  local model="$1"
  local mode="$2"
  local seed="$3"
  if [[ "${seed}" == "1234" ]]; then
    RUN_DIR="artifacts/encoder_controls/pilot/${model}_tsp50_${mode}_seed${seed}"
  else
    RUN_DIR="artifacts/model_range/screen/${model}_tsp50_${mode}_seed${seed}"
  fi
}

resume_checkpoint() {
  local run_dir="$1"
  local latest="${run_dir}/checkpoints/latest.pt"
  local interrupted="${run_dir}/checkpoints/interrupted.pt"
  if [[ -f "${interrupted}" ]] && \
     { [[ ! -f "${latest}" ]] || [[ "${interrupted}" -nt "${latest}" ]]; }; then
    printf '%s\n' "${interrupted}"
  else
    printf '%s\n' "${latest}"
  fi
}

extend_one() {
  local model="$1"
  local mode="$2"
  local seed="$3"
  model_settings "${model}"
  run_directory "${model}" "${mode}" "${seed}"
  if [[ -f "${RUN_DIR}/checkpoints/step-010000.pt" ]]; then
    echo "skip completed ${model} ${mode} seed ${seed}"
    return
  fi
  local checkpoint
  checkpoint="$(resume_checkpoint "${RUN_DIR}")"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing resume checkpoint: ${checkpoint}" >&2
    return 1
  fi
  "${PYTHON_BIN}" "${TRAINER}" \
    --base-mode "${mode}" \
    --output-dir "${RUN_DIR}" \
    --resume "${checkpoint}" \
    --graph-size 50 \
    --steps 10000 \
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
    > "${RUN_DIR}/model_range_extension.log" 2>&1
}

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    pids=()
    for mode in "${MODES[@]}"; do
      extend_one "${model}" "${mode}" "${seed}" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  done
done
