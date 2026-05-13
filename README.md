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

## Code Organization

The C++ implementation is split into a reusable `callaway_core` library and the
`callaway_mfem` executable. Headers live in `include/callaway/`; most source
files in `src/` implement the same-named header.

### Core Library Modules

| File(s) | Main symbols | Role |
|---|---|---|
| `include/callaway/angular_quadrature.hpp`, `src/angular_quadrature.cpp` | `Direction`, `AngularQuadrature`, `AngularQuadrature::GaussLegendre` | Builds the discrete velocity-space quadrature. It stores polar/azimuthal angles, weights, and Cartesian velocity components, and provides moment checks such as `SumWeights()` and `MomentCxCx()`. |
| `include/callaway/boundary.hpp` | `BoundaryType`, `BoundaryCondition`, `BoundaryTypeFromString` | Defines the boundary-condition data model used by config loading, mesh validation, kinetic inflow, and GSIS trace projection. Type conversion is implemented with the config parser. |
| `include/callaway/config.hpp`, `src/config.cpp` | `Config`, `IterationSettings`, `SyntheticAccelerationSettings`, `VelocityMeshSettings`, `DgSettings`, `FlowSettings`, `LoadConfig` | Loads the project's small YAML-like control files, validates numerical settings, computes derived flow data such as `tau_combined()`, and parses GSIS trace solver controls. |
| `include/callaway/dense_solver.hpp`, `src/dense_solver.cpp` | `FactorDenseMatrixInPlace`, `SolveDenseFactoredSystem`, `SolveDenseLinearSystem` | Provides compact dense LU utilities for element-local DG and GSIS macroscopic systems. These routines avoid a hard dependency on LAPACK for the small local matrices. |
| `include/callaway/distribution.hpp`, `src/distribution.cpp` | `Distribution`, `MomentFields` | Owns the kinetic unknown `e(angle, element, dof)` and the derived temperature/heat-flux moment fields at both cell-average and DG-dof level. |
| `include/callaway/integration_cache.hpp`, `src/integration_cache.cpp` | `ElementGeometry`, `IntegrationCache`, `ReferenceMonomialIntegral` | Precomputes geometry, mass matrices, gradient matrices, face mass matrices, face-basis projections, neighbor-face couplings, normals, areas, and element heights needed by the sweep and GSIS equations. |
| `include/callaway/mesh_adapter.hpp`, `src/mesh_adapter.cpp` | `MeshAdapter`, `MeshSummary`, `FaceData` | Wraps MFEM mesh loading, boundary attribute checks, face extraction, element-to-face connectivity, neighbor lookup, and simple mesh summaries. |
| `include/callaway/nodal_basis.hpp`, `src/nodal_basis.cpp` | `NodalBasis` | Builds the Fortran-compatible nodal polynomial basis on triangles and faces for DG orders 1 through 4, including coefficient tables and basis evaluation. |
| `include/callaway/moment_calculator.hpp`, `src/moment_calculator.cpp` | `MomentCalculator::Compute` | Integrates the distribution function over angular quadrature to produce temperature and heat-flux moments used by CIS, GSIS, output, and residual checks. |
| `include/callaway/sweep_ordering.hpp`, `src/sweep_ordering.cpp` | `SweepOrdering` | Computes direction-dependent element traversal order for upwind kinetic sweeps so neighboring inflow values are available when solving each element. |
| `include/callaway/kinetic_sweep_solver.hpp`, `src/kinetic_sweep_solver.cpp` | `KineticSweepSolver`, `Sweep` | Assembles and solves the local steady kinetic DG equation for each angular direction and element. It handles thermalizing-wall inflow, neighbor inflow, optional cached local LU factors, and OpenMP over angles when enabled. |
| `include/callaway/iteration_driver.hpp`, `src/iteration_driver.cpp` | `IterationResult`, `CisIterationDriver`, `GsisIterationDriver`, `TraceSolverSettings` | Runs nonlinear fixed-point iterations. CIS repeatedly sweeps and recomputes moments; GSIS additionally builds high-order sources, solves the synthetic trace problem, reconstructs macro fields, corrects the VDF, and records trace-solver diagnostics. |
| `include/callaway/synthetic_acceleration_solver.hpp`, `src/synthetic_acceleration_solver.cpp` | `MacroComponent`, `MacroState`, `TraceSystem`, `TraceSolveResult`, `SyntheticAccelerationSolver` | Implements GSIS macroscopic synthetic acceleration. It builds local seven-component macro systems, trace-response/projection blocks, a cached sparse trace matrix, GMRES or Eigen/SparseLU trace solves, local reconstruction, limiter-based VDF correction, and the special boundary heat-flux update used by the hydrodynamic paper case. |
| `include/callaway/output_manager.hpp`, `src/output_manager.cpp` | `FieldSample`, `OutputManager` | Samples fields on the Fortran-style square output grid, writes Tecplot conduction data, writes the Fourier reference solution, exports cell-average ParaView data through MFEM, and writes residual history CSV files. |

### Executable And Tooling

| File | Role |
|---|---|
| `src/main.cpp` | Command-line executable entry point. It loads config, applies CLI overrides, constructs quadrature/mesh/basis/integration/sweep objects, runs CIS or GSIS when `--solve` is provided, and writes optional field, residual, Fourier reference, and ParaView outputs. |
| `tools/validate_square_paper.py` | Reproduction driver for the square-domain tests in Section 4.3 of Liu et al. It generates validation configs, runs all GSIS/CIS cases, parses solver stdout, writes summaries, compares GSIS and CIS fields, computes Fourier-limit metrics, and generates temperature-contour figures under `validation/`. |
| `CMakeLists.txt` | Defines the `callaway_core` static library, `callaway_mfem` executable, optional OpenMP/Eigen integration, and all CTest targets. |

### Configuration Files

| File | Role |
|---|---|
| `config/control.example.yaml` | Default CIS-style square-domain setup using the reference mesh. |
| `config/control.gsis.example.yaml` | Full-resolution GSIS example for the same square-domain problem. |
| `config/control.gsis_smoke.yaml` | Lightweight one-step GSIS smoke configuration used by the executable-level CTest. |
| `validation/configs/*.yaml` | Generated, retained paper-validation inputs for each Section 4.3 case and scheme. |

### Tests

| File | Coverage |
|---|---|
| `tests/unit/test_config_quadrature.cpp` | Config parsing, quadrature moments, mesh loading, and boundary attribute validation. |
| `tests/unit/test_dense_solver.cpp` | Dense LU factorization and solve utilities. |
| `tests/unit/test_basis_integration.cpp` | Nodal-basis interpolation properties and integration-cache geometry/matrix invariants. |
| `tests/unit/test_moments_sweep.cpp` | Moment calculation and sweep-ordering consistency. |
| `tests/unit/test_kinetic_sweep.cpp` | Kinetic sweep equilibrium preservation and basic thermalizing-wall behavior. |
| `tests/unit/test_cis_iteration.cpp` | CIS iteration convergence behavior on the reference mesh. |
| `tests/unit/test_output_manager.cpp` | Tecplot, Fourier reference, residual CSV, and ParaView output plumbing. |
| `tests/unit/test_synthetic_acceleration.cpp` | GSIS source, local macro solve, trace coupling, trace solve, reconstruction, and correction kernels. |

### Generated And Reference Data

| Path | Role |
|---|---|
| `reference/FortranCodes/` | Original Fortran implementation and reference mesh used as the mathematical and mesh reference for the C++ refactor. |
| `reference/doc/` | Papers used to define the numerical reproduction targets. |
| `validation/` | Retained paper-reproduction outputs: generated configs, residual CSV files, Tecplot fields, ParaView collections, Fourier references, summary tables, and contour figures. |

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
