#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR="/root/groupopt"
DATA_DIR="/root/autodl-tmp/groupopt-data"
PYTHON_BIN="/root/autodl-tmp/groupopt-venv/bin/python"

cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN}" bash experiments/run_modern_breadth.sh \
  > "${DATA_DIR}/modern_breadth.log" 2>&1
status=$?
printf '%s\n' "${status}" > "${DATA_DIR}/modern_breadth.exit.tmp"
mv "${DATA_DIR}/modern_breadth.exit.tmp" "${DATA_DIR}/modern_breadth.exit"
exit "${status}"
