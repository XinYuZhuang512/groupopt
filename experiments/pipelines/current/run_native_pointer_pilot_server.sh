#!/usr/bin/env bash

set +e

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/groupopt}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/groupopt-data}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

mkdir -p "${DATA_DIR}"
rm -f "${DATA_DIR}/native_pointer_pilot.exit"
cd "${PROJECT_DIR}" || exit 2

PYTHON_BIN="${PYTHON_BIN}" bash experiments/pipelines/current/run_native_pointer_pilot.sh \
  > "${DATA_DIR}/native_pointer_pilot.log" 2>&1
status=$?

printf '%s\n' "${status}" > "${DATA_DIR}/native_pointer_pilot.exit.tmp"
mv "${DATA_DIR}/native_pointer_pilot.exit.tmp" "${DATA_DIR}/native_pointer_pilot.exit"
exit "${status}"
