#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR="/root/groupopt"
DATA_DIR="/root/autodl-tmp/groupopt-data"
PYTHON_BIN="/root/autodl-tmp/groupopt-venv/bin/python"

cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN}" bash experiments/pipelines/archive/run_model_range_extension.sh \
  > "${DATA_DIR}/model_range_extension.log" 2>&1
status=$?
printf '%s\n' "${status}" > "${DATA_DIR}/model_range_extension.exit.tmp"
mv "${DATA_DIR}/model_range_extension.exit.tmp" "${DATA_DIR}/model_range_extension.exit"
exit "${status}"
