#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-2000}"
TEST_SIZE="${TEST_SIZE:-5000}"
TEST_SEED="${TEST_SEED:-20260816}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-512}"
SYMMETRY_FACTOR="${SYMMETRY_FACTOR:-4}"
SYMMETRY_ALPHA="${SYMMETRY_ALPHA:-0.1}"
RUN_ROOT="${RUN_ROOT:-artifacts/symnco_factorial_pilot}"
SEEDS=(${SEEDS:-1234 4321})
VARIANTS=(original ours)
SCHEMES=(reinforce symnco_am)
DISTRIBUTIONS=(uniform clustered narrow_strip clustered_outliers)

if (( EFFECTIVE_BATCH_SIZE % SYMMETRY_FACTOR != 0 )); then
  echo "EFFECTIVE_BATCH_SIZE must be divisible by SYMMETRY_FACTOR" >&2
  exit 2
fi
SYM_BASE_BATCH_SIZE=$((EFFECTIVE_BATCH_SIZE / SYMMETRY_FACTOR))

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

mode_for_variant() {
  case "$1" in
    original) printf '%s\n' native_conditional_fixed ;;
    ours) printf '%s\n' native_conditional_free ;;
    *) echo "unknown variant: $1" >&2; return 2 ;;
  esac
}

train_cell() {
  local seed="$1"
  local scheme="$2"
  local variant="$3"
  local batch_size="$4"
  local mode run_dir checkpoint
  mode="$(mode_for_variant "${variant}")"
  run_dir="${RUN_ROOT}/train/am_tsp50_${variant}_${scheme}_seed${seed}"
  checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
  if [[ -f "${checkpoint}" ]]; then
    echo "skip completed training variant=${variant} scheme=${scheme} seed=${seed}"
    return
  fi
  mkdir -p "${run_dir}"
  local resume_args=()
  if [[ -f "${run_dir}/checkpoints/latest.pt" ]]; then
    resume_args=(--resume "${run_dir}/checkpoints/latest.pt")
  fi
  echo "train variant=${variant} scheme=${scheme} seed=${seed} base_batch=${batch_size} effective_batch=${EFFECTIVE_BATCH_SIZE}"
  "${PYTHON_BIN}" experiments/train.py \
    --base-mode "${mode}" \
    --training-scheme "${scheme}" \
    --symmetry-factor "${SYMMETRY_FACTOR}" \
    --symmetry-alpha "${SYMMETRY_ALPHA}" \
    --output-dir "${run_dir}" \
    "${resume_args[@]}" \
    --graph-size 50 \
    --steps "${STEPS}" \
    --batch-size "${batch_size}" \
    --validation-size 1024 \
    --embedding-dim 128 \
    --heads 8 \
    --encoder-layers 3 \
    --feed-forward-dim 512 \
    --normalization batch \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed "${seed}" \
    --device cuda > "${run_dir}/train.log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  for scheme in "${SCHEMES[@]}"; do
    if [[ "${scheme}" == "symnco_am" ]]; then
      batch_size="${SYM_BASE_BATCH_SIZE}"
    else
      batch_size="${EFFECTIVE_BATCH_SIZE}"
    fi
    pids=()
    for variant in "${VARIANTS[@]}"; do
      train_cell "${seed}" "${scheme}" "${variant}" "${batch_size}" &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    if (( failed != 0 )); then
      echo "parallel training failed scheme=${scheme} seed=${seed}" >&2
      exit 1
    fi
    for variant in "${VARIANTS[@]}"; do
      tail -n 2 "${RUN_ROOT}/train/am_tsp50_${variant}_${scheme}_seed${seed}/train.log"
    done
  done
done

evaluate_cell() {
  local distribution="$1"
  local seed="$2"
  local scheme="$3"
  local variant="$4"
  local run_dir checkpoint eval_dir
  run_dir="${RUN_ROOT}/train/am_tsp50_${variant}_${scheme}_seed${seed}"
  checkpoint="$(printf '%s/checkpoints/step-%06d.pt' "${run_dir}" "${STEPS}")"
  eval_dir="${RUN_ROOT}/eval/${distribution}/am_tsp50_${variant}_${scheme}_seed${seed}"
  if [[ -f "${eval_dir}/summary.json" && -f "${eval_dir}/costs.pt" ]]; then
    echo "skip completed evaluation distribution=${distribution} variant=${variant} scheme=${scheme} seed=${seed}"
    return
  fi
  mkdir -p "${eval_dir}"
  echo "evaluate distribution=${distribution} variant=${variant} scheme=${scheme} seed=${seed}"
  "${PYTHON_BIN}" experiments/evaluate.py \
    --checkpoint "${checkpoint}" \
    --config "${run_dir}/config.json" \
    --output-dir "${eval_dir}" \
    --test-size "${TEST_SIZE}" \
    --test-seed "${TEST_SEED}" \
    --distribution "${distribution}" \
    --graph-size 50 \
    --batch-size 512 \
    --device cuda > "${eval_dir}/evaluate.log" 2>&1
}

for distribution in "${DISTRIBUTIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for scheme in "${SCHEMES[@]}"; do
      pids=()
      for variant in "${VARIANTS[@]}"; do
        evaluate_cell "${distribution}" "${seed}" "${scheme}" "${variant}" &
        pids+=("$!")
      done
      failed=0
      for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
          failed=1
        fi
      done
      if (( failed != 0 )); then
        echo "parallel evaluation failed distribution=${distribution} scheme=${scheme} seed=${seed}" >&2
        exit 1
      fi
      for variant in "${VARIANTS[@]}"; do
        tail -n 1 "${RUN_ROOT}/eval/${distribution}/am_tsp50_${variant}_${scheme}_seed${seed}/evaluate.log"
      done
    done
  done
done

"${PYTHON_BIN}" experiments/summarize_symnco.py \
  --root "${RUN_ROOT}/eval" \
  --output-dir "${RUN_ROOT}/summary"

for seed in "${SEEDS[@]}"; do
  for scheme in "${SCHEMES[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      test -f "$(printf '%s/train/am_tsp50_%s_%s_seed%s/checkpoints/step-%06d.pt' "${RUN_ROOT}" "${variant}" "${scheme}" "${seed}" "${STEPS}")"
      for distribution in "${DISTRIBUTIONS[@]}"; do
        eval_dir="${RUN_ROOT}/eval/${distribution}/am_tsp50_${variant}_${scheme}_seed${seed}"
        test -f "${eval_dir}/summary.json"
        test -f "${eval_dir}/costs.pt"
      done
    done
  done
done
test -f "${RUN_ROOT}/summary/factorial_summary.json"
echo "SYM-NCO factorial pilot complete: ${RUN_ROOT}"
