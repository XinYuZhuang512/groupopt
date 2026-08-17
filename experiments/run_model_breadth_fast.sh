#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_SEED="${TEST_SEED:-20260805}"
TEST_SIZE="${TEST_SIZE:-10000}"
MODELS=(transformer_ln gat pointerformer gru)
MODES=(joint_fixed joint_free)

cd "${PROJECT_DIR}"

trainer_for() {
  local model="$1"
  case "${model}" in
    transformer_ln) TRAINER="experiments/train_transformer_ln_experiment.py"; LAYERS=3 ;;
    gat) TRAINER="experiments/train_gat_experiment.py"; LAYERS=3 ;;
    pointerformer) TRAINER="experiments/train_pointerformer_experiment.py"; LAYERS=3 ;;
    gru) TRAINER="experiments/train_gru_experiment.py"; LAYERS=1 ;;
    *) echo "unsupported model: ${model}" >&2; return 2 ;;
  esac
}

extend_gru() {
  local mode="$1"
  trainer_for gru
  local run_dir="artifacts/encoder_controls/pilot/gru_tsp50_${mode}_seed1234"
  local final_checkpoint="${run_dir}/checkpoints/step-010000.pt"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "skip completed gru ${mode}"
    return
  fi
  local checkpoint="${run_dir}/checkpoints/latest.pt"
  local interrupted="${run_dir}/checkpoints/interrupted.pt"
  if [[ -f "${interrupted}" && "${interrupted}" -nt "${checkpoint}" ]]; then
    checkpoint="${interrupted}"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing GRU checkpoint: ${checkpoint}" >&2
    return 1
  fi
  "${PYTHON_BIN}" "${TRAINER}" \
    --base-mode "${mode}" \
    --output-dir "${run_dir}" \
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
    --seed 1234 \
    --device cuda \
    > "${run_dir}/breadth_extension.log" 2>&1
}

pids=()
for mode in "${MODES[@]}"; do
  extend_gru "${mode}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

for model in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    run_dir="artifacts/encoder_controls/pilot/${model}_tsp50_${mode}_seed1234"
    checkpoint="${run_dir}/checkpoints/step-010000.pt"
    output_dir="artifacts/model_range/breadth_dev/iid_tsp50_seed${TEST_SEED}/${model}_tsp50_${mode}_final10k_seed1234"
    if [[ -f "${output_dir}/summary.json" && -f "${output_dir}/costs.pt" ]]; then
      echo "skip completed evaluation ${model} ${mode}"
      continue
    fi
    if [[ ! -f "${checkpoint}" ]]; then
      echo "missing final checkpoint: ${checkpoint}" >&2
      exit 1
    fi
    "${PYTHON_BIN}" experiments/evaluate_checkpoint.py \
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

"${PYTHON_BIN}" experiments/summarize_model_breadth.py \
  --root "artifacts/model_range/breadth_dev/iid_tsp50_seed${TEST_SEED}" \
  --output "artifacts/model_range/breadth_dev/iid_tsp50_seed${TEST_SEED}/summary.json"
