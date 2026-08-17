#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEED="${SEED:-1234}"
TEST_SEED="${TEST_SEED:-20260811}"
TEST_SIZE="${TEST_SIZE:-10000}"
FAMILIES=(ptrnet gpn)
MODES=(native_fixed native_free)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

trainer_for() {
  case "$1" in
    ptrnet) TRAINER="experiments/train_ptrnet_experiment.py"; LAYERS=1 ;;
    gpn) TRAINER="experiments/train_gpn_experiment.py"; LAYERS=3 ;;
    *) echo "unsupported family: $1" >&2; return 2 ;;
  esac
}

for family in "${FAMILIES[@]}"; do
  trainer_for "${family}"
  for mode in "${MODES[@]}"; do
    output_dir="artifacts/native_plugin/${family}_tsp50_${mode}_seed${SEED}"
    final_checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${output_dir}" "${STEPS}")"
    if [[ -f "${final_checkpoint}" ]]; then
      echo "skip completed ${family} ${mode} seed=${SEED}"
      continue
    fi
    resume_args=()
    if [[ -f "${output_dir}/checkpoints/latest.pt" ]]; then
      resume_args=(--resume "${output_dir}/checkpoints/latest.pt")
    fi
    "${PYTHON_BIN}" "${TRAINER}" \
      --base-mode "${mode}" \
      --output-dir "${output_dir}" \
      "${resume_args[@]}" \
      --graph-size 50 \
      --steps "${STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --validation-size 1024 \
      --embedding-dim 128 \
      --encoder-layers "${LAYERS}" \
      --learning-rate 1e-4 \
      --eval-every 100 \
      --checkpoint-every 250 \
      --seed "${SEED}" \
      --device cuda
  done
done

for family in "${FAMILIES[@]}"; do
  for mode in "${MODES[@]}"; do
    run_dir="artifacts/native_plugin/${family}_tsp50_${mode}_seed${SEED}"
    checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
    output_dir="artifacts/native_plugin_eval/iid_tsp50_seed${TEST_SEED}/${family}_tsp50_${mode}_final${STEPS}_seed${SEED}"
    if [[ -f "${output_dir}/summary.json" && -f "${output_dir}/costs.pt" ]]; then
      echo "skip completed evaluation ${family} ${mode}"
      continue
    fi
    "${PYTHON_BIN}" experiments/evaluate_checkpoint.py \
      --checkpoint "${checkpoint}" \
      --config "${run_dir}/config.json" \
      --output-dir "${output_dir}" \
      --test-size "${TEST_SIZE}" \
      --test-seed "${TEST_SEED}" \
      --graph-size 50 \
      --batch-size "${BATCH_SIZE}" \
      --device cuda
  done
done
