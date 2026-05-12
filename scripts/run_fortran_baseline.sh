#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

FORTRAN_EXE="${FORTRAN_EXE:-${REPO_DIR}/DGACC}"
CONTROL="${CONTROL:-${REPO_DIR}/control.in}"
MESH="${MESH:-${REPO_DIR}/A1_Nx11_Ny11.msh}"
WORK_DIR="${WORK_DIR:-${PROJECT_DIR}/output/fortran_baseline}"
FORTRAN_LIBRARY_PATH="${FORTRAN_LIBRARY_PATH:-}"
MAX_STEPS="${MAX_STEPS:-20}"
SCHEME="${SCHEME:-cis}"
QUIET="${QUIET:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_NUM_THREADS

case "${SCHEME}" in
  cis|CIS)
    ACCFLAG=0
    ;;
  gsis|GSIS)
    ACCFLAG=1
    ;;
  *)
    echo "SCHEME must be 'cis' or 'gsis'." >&2
    exit 2
    ;;
esac

if [[ ! -f "${FORTRAN_EXE}" ]]; then
  echo "Fortran executable is missing: ${FORTRAN_EXE}" >&2
  exit 2
fi

if [[ -z "${FORTRAN_LIBRARY_PATH}" ]]; then
  for candidate in \
    "${CONDA_PREFIX:-}/lib" \
    "${HOME}/miniconda3/envs/dev/lib" \
    "${HOME}/miniconda3/lib"; do
    if [[ -f "${candidate}/libmkl_core.so" && -f "${candidate}/libiomp5.so" ]]; then
      FORTRAN_LIBRARY_PATH="${candidate}"
      break
    fi
  done
fi

if [[ -n "${FORTRAN_LIBRARY_PATH}" ]]; then
  primary_library_path="${FORTRAN_LIBRARY_PATH%%:*}"
  compat_dir="${PROJECT_DIR}/output/mkl_compat"
  if [[ -f "${primary_library_path}/libmkl_core.so.2" &&
        ! -f "${primary_library_path}/libmkl_core.so.1" ]]; then
    mkdir -p "${compat_dir}"
    for lib in libmkl_intel_lp64 libmkl_intel_thread libmkl_core; do
      if [[ -f "${primary_library_path}/${lib}.so.2" ]]; then
        ln -sf "${primary_library_path}/${lib}.so.2" "${compat_dir}/${lib}.so.1"
      fi
    done
    FORTRAN_LIBRARY_PATH="${compat_dir}:${FORTRAN_LIBRARY_PATH}"
  fi
fi

RUNTIME_LIBRARY_PATH="${FORTRAN_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${RUNTIME_LIBRARY_PATH}" ]]; then
  missing_report="$(env LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}" ldd "${FORTRAN_EXE}" | grep "not found" || true)"
else
  missing_report="$(ldd "${FORTRAN_EXE}" | grep "not found" || true)"
fi

if [[ -n "${missing_report}" ]]; then
  echo "Fortran executable has missing runtime libraries:" >&2
  echo "${missing_report}" >&2
  echo "Set FORTRAN_LIBRARY_PATH or LD_LIBRARY_PATH to a compatible Intel/MKL runtime and rerun." >&2
  exit 3
fi

mkdir -p "${WORK_DIR}"
find "${WORK_DIR}" -maxdepth 1 -type f \( \
  -name "2D_*.dat" -o \
  -name "Conduction_A.dat" -o \
  -name "RunTime.txt" -o \
  -name "VDF*.out" -o \
  -name "fortran_stdout.log" -o \
  -name "fortran_residual.csv" -o \
  -name "DGACC" -o \
  -name "control.in" -o \
  -name "$(basename "${MESH}")" \
\) -delete

cp "${FORTRAN_EXE}" "${WORK_DIR}/DGACC"
chmod +x "${WORK_DIR}/DGACC"
cp "${MESH}" "${WORK_DIR}/$(basename "${MESH}")"

python3 - "${CONTROL}" "${WORK_DIR}/control.in" "${ACCFLAG}" "${MAX_STEPS}" "$(basename "${MESH}")" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
accflag = sys.argv[3]
max_steps = sys.argv[4]
mesh_name = sys.argv[5]

text = source.read_text(encoding="utf-8")
text = re.sub(r"(TMAX\s*=\s*)[^,\n]+", rf"\g<1>{max_steps}", text, flags=re.IGNORECASE)
text = re.sub(r"(ACCFLAG\s*=\s*)[^,\n]+", rf"\g<1>{accflag}", text, flags=re.IGNORECASE)
text = re.sub(
    r"(FNAME_MSH\s*=\s*)'[^']*'",
    rf"\g<1>'./{mesh_name}'",
    text,
    flags=re.IGNORECASE,
)
target.write_text(text, encoding="utf-8")
PY

pushd "${WORK_DIR}" >/dev/null
if [[ "${QUIET}" == "1" ]]; then
  if [[ -n "${RUNTIME_LIBRARY_PATH}" ]]; then
    env LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}" ./DGACC > fortran_stdout.log
  else
    ./DGACC > fortran_stdout.log
  fi
else
  if [[ -n "${RUNTIME_LIBRARY_PATH}" ]]; then
    env LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}" ./DGACC | tee fortran_stdout.log
  else
    ./DGACC | tee fortran_stdout.log
  fi
fi
popd >/dev/null

python3 "${SCRIPT_DIR}/parse_fortran_residual.py" \
  "${WORK_DIR}/fortran_stdout.log" \
  -o "${WORK_DIR}/fortran_residual.csv"

echo "Fortran baseline directory: ${WORK_DIR}"
echo "Fortran residual: ${WORK_DIR}/fortran_residual.csv"
tail -n 8 "${WORK_DIR}/fortran_stdout.log"
find "${WORK_DIR}" -maxdepth 1 -type f \( -name "2D_*.dat" -o -name "Conduction_A.dat" \) -print
