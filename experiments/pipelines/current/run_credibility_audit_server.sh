#!/usr/bin/env bash

set -uo pipefail

PROJECT_DIR="/root/groupopt"
DATA_DIR="/root/autodl-tmp/groupopt-data"
PYTHON_BIN="/root/autodl-tmp/groupopt-venv/bin/python"

cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN}" bash experiments/pipelines/current/run_credibility_audit.sh \
  > "${DATA_DIR}/credibility_audit.log" 2>&1
status=$?
printf '%s\n' "${status}" > "${DATA_DIR}/credibility_audit.exit.tmp"
mv "${DATA_DIR}/credibility_audit.exit.tmp" "${DATA_DIR}/credibility_audit.exit"
exit "${status}"
