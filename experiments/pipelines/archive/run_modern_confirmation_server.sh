#!/usr/bin/env bash

set +e

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/groupopt}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/groupopt-data}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

mkdir -p "${DATA_DIR}"
rm -f "${DATA_DIR}/modern_confirmation.exit"

cd "${PROJECT_DIR}" || exit 2
PYTHON_BIN="${PYTHON_BIN}" bash experiments/pipelines/archive/run_modern_confirmation.sh \
  > "${DATA_DIR}/modern_confirmation.log" 2>&1
status=$?

printf '%s\n' "${status}" > "${DATA_DIR}/modern_confirmation.exit.tmp"
mv "${DATA_DIR}/modern_confirmation.exit.tmp" "${DATA_DIR}/modern_confirmation.exit"
exit "${status}"
