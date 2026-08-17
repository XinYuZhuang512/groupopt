#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_SEED="${TEST_SEED:-20260809}"
TEST_SIZE="${TEST_SIZE:-20000}"
MAX_PARALLEL_TRAINING="${MAX_PARALLEL_TRAINING:-2}"
FAMILIES=(am ptrnet gpn)
SEEDS=(1234 2345 3456 4567)

cd "${PROJECT_DIR}"

family_settings() {
  local family="$1"
  case "${family}" in
    am)
      TRAINER="experiments/train_am_experiment.py"
      PREFIX="tsp50"
      MODEL_ARGS=(--heads 8 --encoder-layers 3 --feed-forward-dim 512 --normalization batch)
      ;;
    ptrnet)
      TRAINER="experiments/train_ptrnet_experiment.py"
      PREFIX="ptrnet_tsp50"
      MODEL_ARGS=(--encoder-layers 1)
      ;;
    gpn)
      TRAINER="experiments/train_gpn_experiment.py"
      PREFIX="gpn_tsp50"
      MODEL_ARGS=(--encoder-layers 3)
      ;;
    *)
      echo "unsupported family: ${family}" >&2
      return 2
      ;;
  esac
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

extend_joint_fixed() {
  local family="$1"
  local seed="$2"
  family_settings "${family}"
  local run_dir="artifacts/confirmation/${PREFIX}_joint_fixed_seed${seed}"
  local final_checkpoint="${run_dir}/checkpoints/step-010000.pt"
  if [[ -f "${final_checkpoint}" ]]; then
    echo "skip completed joint_fixed ${family} seed ${seed}"
    return
  fi
  local checkpoint
  checkpoint="$(resume_checkpoint "${run_dir}")"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing joint_fixed resume checkpoint: ${checkpoint}" >&2
    return 1
  fi

  "${PYTHON_BIN}" "${TRAINER}" \
    --base-mode joint_fixed \
    --output-dir "${run_dir}" \
    --resume "${checkpoint}" \
    --graph-size 50 \
    --steps 10000 \
    --batch-size 512 \
    --validation-size 1024 \
    --embedding-dim 128 \
    "${MODEL_ARGS[@]}" \
    --learning-rate 1e-4 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --seed "${seed}" \
    --device cuda \
    > "${run_dir}/credibility_joint_fixed_extension.log" 2>&1
}

run_dir_for() {
  local family="$1"
  local arm="$2"
  local seed="$3"
  family_settings "${family}"
  case "${arm}" in
    original)
      if [[ "${family}" == "gpn" && "${seed}" == "1234" ]]; then
        RUN_DIR="artifacts/pilot/${PREFIX}_fixed_seed${seed}"
      else
        RUN_DIR="artifacts/formal/${PREFIX}_fixed_seed${seed}"
      fi
      MODE_LABEL="original_fixed"
      ;;
    joint_fixed|joint_free)
      if [[ "${seed}" == "1234" ]]; then
        RUN_DIR="artifacts/pilot/${PREFIX}_${arm}_seed${seed}"
      else
        RUN_DIR="artifacts/confirmation/${PREFIX}_${arm}_seed${seed}"
      fi
      MODE_LABEL="${arm}"
      ;;
    *)
      echo "unsupported arm: ${arm}" >&2
      return 2
      ;;
  esac
}

evaluate_final_checkpoint() {
  local family="$1"
  local arm="$2"
  local seed="$3"
  run_dir_for "${family}" "${arm}" "${seed}"
  local checkpoint="${RUN_DIR}/checkpoints/step-010000.pt"
  local config="${RUN_DIR}/config.json"
  local output_dir="artifacts/credibility_audit/iid_tsp50_seed${TEST_SEED}/${PREFIX}_${MODE_LABEL}_final10k_seed${seed}"
  if [[ -f "${output_dir}/costs.pt" && -f "${output_dir}/summary.json" ]]; then
    echo "skip completed evaluation ${family} ${arm} seed ${seed}"
    return
  fi
  if [[ ! -f "${checkpoint}" || ! -f "${config}" ]]; then
    echo "missing final checkpoint or config for ${family} ${arm} seed ${seed}" >&2
    return 1
  fi
  "${PYTHON_BIN}" experiments/evaluate_checkpoint.py \
    --checkpoint "${checkpoint}" \
    --config "${config}" \
    --output-dir "${output_dir}" \
    --test-size "${TEST_SIZE}" \
    --test-seed "${TEST_SEED}" \
    --graph-size 50 \
    --batch-size 512 \
    --device cuda
}

# Finish every parameter-matched control before revealing any new-test result.
pending_pids=()
for family in "${FAMILIES[@]}"; do
  for seed in 2345 3456 4567; do
    extend_joint_fixed "${family}" "${seed}" &
    pending_pids+=("$!")
    if [[ "${#pending_pids[@]}" -eq "${MAX_PARALLEL_TRAINING}" ]]; then
      for pid in "${pending_pids[@]}"; do
        wait "${pid}"
      done
      pending_pids=()
    fi
  done
done
for pid in "${pending_pids[@]}"; do
  wait "${pid}"
done

# Evaluate exact step-10k checkpoints under one locked, previously unseen protocol.
for family in "${FAMILIES[@]}"; do
  for arm in original joint_fixed joint_free; do
    for seed in "${SEEDS[@]}"; do
      evaluate_final_checkpoint "${family}" "${arm}" "${seed}"
    done
  done
done

"${PYTHON_BIN}" experiments/summarize_credibility_audit.py \
  --root "artifacts/credibility_audit/iid_tsp50_seed${TEST_SEED}" \
  --output "artifacts/credibility_audit/iid_tsp50_seed${TEST_SEED}/summary.json"
