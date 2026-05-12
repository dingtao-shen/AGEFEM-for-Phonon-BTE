# Regression Status

This document records the current Fortran-to-C++ regression checks for the
MFEM refactor. The Fortran executable used here is the existing `DGACC` binary
in the repository root. It is run from an isolated work directory through
`scripts/run_fortran_baseline.sh`.

## Runtime Notes

- The local system does not provide `ifort`.
- The existing `DGACC` binary requires older MKL runtime names:
  `libmkl_intel_lp64.so.1`, `libmkl_intel_thread.so.1`, and `libmkl_core.so.1`.
- Compatible MKL and Intel OpenMP libraries are available in the Conda `dev`
  environment. The baseline script creates isolated `.so.1` symlinks under
  `mfem_cpp_refactor/output/mkl_compat` when only `.so.2` names are present.

## Baseline Commands

Generate five-step Fortran baselines:

```bash
SCHEME=cis MAX_STEPS=5 \
WORK_DIR=mfem_cpp_refactor/output/fortran_baseline_cis_5 \
mfem_cpp_refactor/scripts/run_fortran_baseline.sh

SCHEME=gsis MAX_STEPS=5 \
WORK_DIR=mfem_cpp_refactor/output/fortran_baseline_gsis_5 \
mfem_cpp_refactor/scripts/run_fortran_baseline.sh
```

Compare C++ against those baselines:

```bash
BASELINE='mfem_cpp_refactor/output/fortran_baseline_cis_5/2D_P3_T  200_tR 1.00E-03_tN 1.00E+05_ CIS.dat' \
BASELINE_RESIDUAL=mfem_cpp_refactor/output/fortran_baseline_cis_5/fortran_residual.csv \
CONFIG=mfem_cpp_refactor/config/control.example.yaml \
MAX_STEPS=5 \
OUTPUT_SAMPLES=109 \
OUTPUT_DIR=mfem_cpp_refactor/output/regression_cis_vs_fortran_5 \
mfem_cpp_refactor/scripts/run_regression.sh

BASELINE='mfem_cpp_refactor/output/fortran_baseline_gsis_5/2D_P3_T  200_tR 1.00E-03_tN 1.00E+05_GSIS.dat' \
BASELINE_RESIDUAL=mfem_cpp_refactor/output/fortran_baseline_gsis_5/fortran_residual.csv \
CONFIG=mfem_cpp_refactor/config/control.gsis.example.yaml \
MAX_STEPS=5 \
OUTPUT_SAMPLES=109 \
OUTPUT_DIR=mfem_cpp_refactor/output/regression_gsis_vs_fortran_5 \
mfem_cpp_refactor/scripts/run_regression.sh
```

## Current Results

All checks below use the full default mesh and discretization:

- Mesh: `A1_Nx11_Ny11.msh`
- DG degree: 3
- Spatial elements: 200 triangles
- Angular directions: 20 polar by 40 azimuthal directions
- Output grid: `109 x 109`
- Flow: `tau_R = 1.0e-3`, `tau_N = 1.0e5`

### CIS, 5 Steps

- Final residual: `0.138989413114`
- Final mass: `9.16628641527e-4`
- Field comparison: passed for `T`, `qx`, `qy`, `Nxx`, `Nxy`, and `Nyy`
- History comparison: passed for `residual` and `mass`
  - Maximum residual difference: `4.47e-15`
  - Maximum mass difference: `1.19e-16`

### GSIS, 5 Steps

- Final residual: `0.0869807973571`
- Final mass: `0.207795897234`
- Trace GMRES on the final step: 529 iterations, converged
- Field comparison: passed for `T`, `qx`, `qy`, `Nxx`, `Nxy`, and `Nyy`
- History comparison: passed for `residual` and `mass`
  - Maximum residual difference: `2.28e-9`
  - Maximum mass difference: `1.25e-8`

The regression tolerances are set for the six-digit scientific notation used by
the Fortran Tecplot writer: `COMPARE_ATOL=1.0e-7` and `COMPARE_RTOL=1.0e-6`.
History comparison defaults to `RESIDUAL_COLUMNS=residual,mass`,
`RESIDUAL_ATOL=2.0e-8`, and `RESIDUAL_RTOL=1.0e-8`. The mass tolerance accounts
for the current C++ GSIS trace solve using iterative GMRES while Fortran uses a
PARDISO direct solve.

## Important Parity Fix

The GSIS trace unknowns live on global faces. Therefore element-face basis
projections must use the global face orientation, not the local element edge
orientation. This is required for parity with `Synthetic_Acceleration.f90` and
is now handled in `IntegrationCache::ElementFaceBasisMass`.
