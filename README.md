# Callaway MFEM Refactor

This directory contains the C++/MFEM refactor of the existing Fortran
2D-2V linearized Callaway DG solver.

The first implementation milestone provides:

- YAML configuration loading.
- Boundary-condition data models with extension points for future wall types.
- Angular quadrature matching the current Fortran velocity mesh.
- MFEM mesh loading and mesh-summary validation.
- Fortran-equivalent nodal basis generation for orders 1 through 4.
- Volume and face integration caches needed by the DG kinetic sweep.
- Direction-wise CIS kinetic sweep with thermalizing-wall inflow.
- CIS iteration driver with Fortran-style temperature residual history.
- Cached per-angle/per-element LU factors for the CIS local DG matrices.
- Tecplot-style field output, residual CSV output, and the square Fourier
  reference solution used by the original Fortran post-processing.
- MFEM `ParaViewDataCollection` output for DG0 cell-average temperature and
  heat-flux fields.
- GSIS groundwork matching `Synthetic_Acceleration.f90`: local seven-component
  macroscopic block matrices, cached local LU factors, high-order DVM source
  moments, trace coupling tensors, MFEM sparse trace-system assembly, GMRES
  trace solve entry point, local macroscopic reconstruction, VDF correction,
  and a full GSIS iteration driver wired into `callaway_mfem`.
- Smoke/unit tests for configuration, quadrature, mesh input, basis properties,
  integration invariants, equilibrium sweep preservation, CIS iteration, field
  output, synthetic acceleration kernels, and a lightweight GSIS executable run.

Build the serial version with:

```bash
cmake -S mfem_cpp_refactor -B mfem_cpp_refactor/build \
  -DMFEM_DIR=/usr/local/lib/cmake/mfem
cmake --build mfem_cpp_refactor/build -j
ctest --test-dir mfem_cpp_refactor/build --output-on-failure
```

Run the current smoke executable with:

```bash
mfem_cpp_refactor/build/callaway_mfem \
  --config mfem_cpp_refactor/config/control.example.yaml
```

Run a short CIS solve with:

```bash
mfem_cpp_refactor/build/callaway_mfem \
  --config mfem_cpp_refactor/config/control.example.yaml \
  --solve --max-steps 20
```

The output prefix can be overridden from the command line:

```bash
mfem_cpp_refactor/build/callaway_mfem \
  --config mfem_cpp_refactor/config/control.example.yaml \
  --solve --max-steps 20 \
  --write-output --output-prefix mfem_cpp_refactor/output/cis_candidate
```

Run a lightweight GSIS smoke solve with:

```bash
mfem_cpp_refactor/build/callaway_mfem \
  --config mfem_cpp_refactor/config/control.gsis_smoke.yaml \
  --solve
```

Run the full-resolution one-step GSIS configuration with:

```bash
mfem_cpp_refactor/build/callaway_mfem \
  --config mfem_cpp_refactor/config/control.gsis.example.yaml \
  --solve --max-steps 1
```

GSIS trace-solver controls are available under the `gsis` YAML section:
`trace_relative_tolerance`, `trace_absolute_tolerance`,
`trace_max_iterations`, `trace_print_level`, and `trace_preconditioner`.
Supported preconditioners are `none` and `jacobi`.

Add `--write-output --output-samples 109` to write
`output_field.dat`, `output_residual.csv`, `output_reference.dat`, and the
`output_paraview` collection from the configured `files.output_prefix`.
The same sample count can also be set with `files.output_samples` in the YAML
configuration. CIS residual output keeps the Fortran-style `step,residual`
columns; GSIS residual output also includes trace GMRES iterations, convergence,
and initial/final trace residual norms.

Non-thermalizing walls and periodic boundaries are interface-defined but not
implemented in this milestone.

For field regression against a Fortran Tecplot baseline, use:

```bash
BASELINE=/path/to/fortran_field.dat \
CONFIG=mfem_cpp_refactor/config/control.example.yaml \
MAX_STEPS=20 \
OUTPUT_SAMPLES=109 \
mfem_cpp_refactor/scripts/run_regression.sh
```

The underlying comparison tool is `scripts/compare_tecplot.py`; it reports
per-column max, L2, RMS, and relative L2 differences for matching structured
Tecplot files. The regression script defaults to `COMPARE_ATOL=1.0e-7` and
`COMPARE_RTOL=1.0e-6`, matching the six-digit scientific output precision used
by the Fortran Tecplot writer. It also compares residual-history columns
`residual,mass` by default with `RESIDUAL_ATOL=2.0e-8` and
`RESIDUAL_RTOL=1.0e-8`. Override these variables for stricter internal C++
comparisons.

The legacy Fortran namelist can be converted to YAML with:

```bash
mfem_cpp_refactor/scripts/convert_control_in.py control.in \
  -o mfem_cpp_refactor/output/control.from_fortran.yaml
```

The converter resolves the mesh path relative to the original `control.in`
location and fills the C++ GSIS trace-solver defaults.

If a runnable Fortran executable and Intel/MKL runtime are available, generate a
baseline without modifying the original Fortran directory with:

```bash
SCHEME=cis MAX_STEPS=20 mfem_cpp_refactor/scripts/run_fortran_baseline.sh
SCHEME=gsis MAX_STEPS=20 mfem_cpp_refactor/scripts/run_fortran_baseline.sh
```

The script creates a separate work directory under
`mfem_cpp_refactor/output/fortran_baseline`, rewrites only the copied
`control.in`, and checks missing runtime libraries before launching `DGACC`.
If it finds a Conda MKL runtime with `.so.2` library names, it creates isolated
compatibility symlinks under `mfem_cpp_refactor/output/mkl_compat` for the older
`.so.1` names required by the existing `DGACC` binary.

The current five-step CIS and GSIS parity results are documented in
`docs/regression_status.md`.
