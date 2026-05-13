# Callaway MFEM Refactor

This project is a C++17/MFEM refactor of the 2D-2V linearized Callaway
phonon Boltzmann DG solver.

The current implementation provides:

- YAML configuration loading.
- Boundary-condition data models with extension points for future wall types.
- Angular quadrature matching the reference velocity mesh.
- MFEM mesh loading and mesh-summary validation.
- Fortran-equivalent nodal basis generation for orders 1 through 4.
- Volume and face integration caches needed by the DG kinetic sweep.
- Direction-wise CIS kinetic sweep with thermalizing-wall inflow.
- CIS iteration with temperature residual history.
- Cached per-angle/per-element LU factors for the CIS local DG matrices.
- Tecplot-style field output, residual CSV output, Fourier reference output,
  and MFEM `ParaViewDataCollection` output.
- GSIS synthetic acceleration with local seven-component macroscopic blocks,
  trace-system assembly, GMRES or Eigen/SparseLU trace solve, local
  macroscopic reconstruction, VDF correction, and a GSIS iteration driver.
- Unit/smoke tests for configuration, quadrature, mesh input, basis properties,
  integration invariants, equilibrium sweep preservation, CIS iteration, field
  output, synthetic acceleration kernels, and a lightweight GSIS run.

The default configurations use the reference square mesh at
`reference/FortranCodes/A1_Nx11_Ny11.msh`.

## Build And Test

```bash
cmake -S . -B build -DMFEM_DIR=/usr/local/lib/cmake/mfem
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Enable OpenMP with:

```bash
cmake -S . -B build-openmp \
  -DMFEM_DIR=/usr/local/lib/cmake/mfem \
  -DCALLAWAY_USE_OPENMP=ON
cmake --build build-openmp -j
```

## Run

Smoke run without iterations:

```bash
build/callaway_mfem --config config/control.example.yaml
```

Short CIS solve:

```bash
build/callaway_mfem \
  --config config/control.example.yaml \
  --solve --max-steps 20
```

Lightweight GSIS smoke solve:

```bash
build/callaway_mfem \
  --config config/control.gsis_smoke.yaml \
  --solve
```

Full-resolution one-step GSIS solve:

```bash
build/callaway_mfem \
  --config config/control.gsis.example.yaml \
  --solve --max-steps 1
```

Add `--write-output --output-prefix output/candidate --output-samples 109`
to write:

- `output/candidate_field.dat`
- `output/candidate_residual.csv`
- `output/candidate_reference.dat`
- `output/candidate_paraview`

GSIS trace-solver controls are available under the `gsis` YAML section:
`trace_relative_tolerance`, `trace_absolute_tolerance`,
`trace_max_iterations`, `trace_print_level`, and `trace_preconditioner`.
Supported trace solve modes are `none`, `jacobi`, and `direct`; `direct`
uses Eigen/SparseLU when Eigen3 is available at build time.
`boundary_heat_flux_from_vdf` enables the paper's special GSIS moment-boundary
treatment for the `KnR=10, KnN=0.01` square-domain case.

Non-thermalizing walls and periodic boundaries are interface-defined but not
implemented in this milestone.

## Paper Validation

Section 4.3 of Liu et al., JCP 467 (2022) 111436 can be reproduced with:

```bash
cmake -S . -B build-openmp \
  -DMFEM_DIR=/usr/local/lib/cmake/mfem \
  -DCALLAWAY_USE_OPENMP=ON
cmake --build build-openmp -j
OMP_NUM_THREADS=24 python3 tools/validate_square_paper.py \
  --executable build-openmp/callaway_mfem \
  --resume
```

The validation script requires Python with `numpy` and `matplotlib`. It writes
configs, residual CSV files, Tecplot fields, ParaView collections, Fourier
references, temperature contour figures, and summary tables under
`validation/`.
