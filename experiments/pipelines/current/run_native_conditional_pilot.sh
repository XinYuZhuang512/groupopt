#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-1234}"
TEST_SEED="${TEST_SEED:-20260812}"
TEST_SIZE="${TEST_SIZE:-10000}"
MODES=(native_conditional_fixed native_conditional_free)
FAMILIES=(am ptrnet gpn)

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

configure_family() {
  case "$1" in
    am)
      TRAINER="experiments/train/train_am_experiment.py"
      BATCH_SIZE="${AM_BATCH_SIZE:-512}"
      MODEL_ARGS=(--heads 8 --encoder-layers 3 --feed-forward-dim 512 --normalization batch)
      ;;
    ptrnet)
      TRAINER="experiments/train/train_ptrnet_experiment.py"
      BATCH_SIZE="${POINTER_BATCH_SIZE:-128}"
      MODEL_ARGS=(--encoder-layers 1)
      ;;
    gpn)
      TRAINER="experiments/train/train_gpn_experiment.py"
      BATCH_SIZE="${POINTER_BATCH_SIZE:-128}"
      MODEL_ARGS=(--encoder-layers 3)
      ;;
    *) echo "unsupported family: $1" >&2; return 2 ;;
  esac
}

for family in "${FAMILIES[@]}"; do
  configure_family "${family}"
  for mode in "${MODES[@]}"; do
    output_dir="artifacts/native_conditional/${family}_tsp50_${mode}_seed${SEED}"
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
      "${MODEL_ARGS[@]}" \
      --learning-rate 1e-4 \
      --eval-every 100 \
      --checkpoint-every 250 \
      --seed "${SEED}" \
      --device cuda
  done
done

for family in "${FAMILIES[@]}"; do
  configure_family "${family}"
  for mode in "${MODES[@]}"; do
    run_dir="artifacts/native_conditional/${family}_tsp50_${mode}_seed${SEED}"
    checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
    output_dir="artifacts/native_conditional_eval/iid_tsp50_seed${TEST_SEED}/${family}_tsp50_${mode}_final${STEPS}_seed${SEED}"
    if [[ -f "${output_dir}/summary.json" && -f "${output_dir}/costs.pt" ]]; then
      echo "skip completed evaluation ${family} ${mode}"
      continue
    fi
    "${PYTHON_BIN}" experiments/evaluation/evaluate_checkpoint.py \
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
