#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SEED="${SEED:-1234}"
MODES=(native_fixed native_free)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

for mode in "${MODES[@]}"; do
  output_dir="artifacts/native_plugin/am_tsp50_${mode}_seed${SEED}"
  final_checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${output_dir}" "${STEPS}")"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "skip completed AM ${mode} seed=${SEED}"
    continue
  fi

  resume_args=()
  if [[ -f "${output_dir}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${output_dir}/checkpoints/latest.pt")
  fi

  "${PYTHON_BIN}" experiments/train_am_experiment.py \
    --base-mode "${mode}" \
    --output-dir "${output_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --validation-size 1024 \
    --embedding-dim 128 \
    --heads 8 \
    --encoder-layers 3 \
    --feed-forward-dim 512 \
    --normalization batch \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed "${SEED}" \
    --device cuda
done
