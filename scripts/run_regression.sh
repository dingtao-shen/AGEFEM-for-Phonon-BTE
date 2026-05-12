#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build}"
CONFIG="${CONFIG:-${PROJECT_DIR}/config/control.example.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/regression}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${OUTPUT_DIR}/candidate}"
BASELINE="${BASELINE:-}"
BASELINE_RESIDUAL="${BASELINE_RESIDUAL:-}"
MAX_STEPS="${MAX_STEPS:-20}"
OUTPUT_SAMPLES="${OUTPUT_SAMPLES:-109}"
COMPARE_COLUMNS="${COMPARE_COLUMNS:-T,qx,qy,Nxx,Nxy,Nyy}"
COMPARE_ATOL="${COMPARE_ATOL:-1.0e-7}"
COMPARE_RTOL="${COMPARE_RTOL:-1.0e-6}"
RESIDUAL_ATOL="${RESIDUAL_ATOL:-2.0e-8}"
RESIDUAL_RTOL="${RESIDUAL_RTOL:-1.0e-8}"
RESIDUAL_COLUMNS="${RESIDUAL_COLUMNS:-residual,mass}"

mkdir -p "${OUTPUT_DIR}"

cmake --build "${BUILD_DIR}" -j

pushd "${REPO_DIR}" >/dev/null
"${BUILD_DIR}/callaway_mfem" \
  --config "${CONFIG}" \
  --solve \
  --max-steps "${MAX_STEPS}" \
  --write-output \
  --output-prefix "${OUTPUT_PREFIX}" \
  --output-samples "${OUTPUT_SAMPLES}"
popd >/dev/null

echo "Candidate field: ${OUTPUT_PREFIX}_field.dat"
echo "Candidate residual: ${OUTPUT_PREFIX}_residual.csv"

if [[ -n "${BASELINE}" ]]; then
  python3 "${SCRIPT_DIR}/compare_tecplot.py" \
    "${BASELINE}" \
    "${OUTPUT_PREFIX}_field.dat" \
    --columns "${COMPARE_COLUMNS}" \
    --atol "${COMPARE_ATOL}" \
    --rtol "${COMPARE_RTOL}"
else
  echo "BASELINE is not set; skipping Tecplot field comparison."
fi

if [[ -n "${BASELINE_RESIDUAL}" ]]; then
  python3 "${SCRIPT_DIR}/compare_residual.py" \
    "${BASELINE_RESIDUAL}" \
    "${OUTPUT_PREFIX}_residual.csv" \
    --columns "${RESIDUAL_COLUMNS}" \
    --atol "${RESIDUAL_ATOL}" \
    --rtol "${RESIDUAL_RTOL}"
else
  echo "BASELINE_RESIDUAL is not set; skipping residual-history comparison."
fi
