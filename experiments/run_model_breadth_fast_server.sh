#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR="/root/groupopt"
DATA_DIR="/root/autodl-tmp/groupopt-data"
PYTHON_BIN="/root/autodl-tmp/groupopt-venv/bin/python"

cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN}" bash experiments/run_model_breadth_fast.sh \
  > "${DATA_DIR}/model_breadth_fast.log" 2>&1
status=$?
printf '%s\n' "${status}" > "${DATA_DIR}/model_breadth_fast.exit.tmp"
mv "${DATA_DIR}/model_breadth_fast.exit.tmp" "${DATA_DIR}/model_breadth_fast.exit"
exit "${status}"
